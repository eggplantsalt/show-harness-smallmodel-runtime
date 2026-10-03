"""SAM3-backed object-relative geometry for one bounded Runtime V3 alignment."""

from __future__ import annotations

import base64
import io
import json
import math
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from core.capabilities.camera_geometry import (
    CameraCalibration,
    make_mujoco_calibrations,
    project_point,
)
from .canonical_image import CanonicalImageAdapter
from .depth import DepthProvider, metric_entity_reference_from_estimate
from .grounding_query import GroundingQueryNormalizer
from .grounding import EntityGroundingEvidence, SemanticGroundingBinder, SemanticGroundingResult
from .metric_entity import MetricEntityReference, freeze_metric_reference
from .observer import RobotObservation
from .options import BoundedMicroMotionSpec, PrimitiveCommand, RuntimeOption
from .readiness import EntityObservationReady, SceneMotionReady
from .scene_settling import SceneReadyEvidence
from .state import BeliefState, ObjectRelativeState, RuntimeEntityState
from .task_spec import EntitySpec


DIRECTION_ORDER = ("FWD", "BACK", "LEFT", "RIGHT", "UP", "DOWN")
REQUESTED_ALIGNMENT_M = 0.003
CONTROL_TICK_STEP_M = 0.005
MAX_ALIGNMENT_TICKS = 5
SAM3_CONFIDENCE_THRESHOLD = 0.05
TARGET_MASK_IOU_MIN = 0.25
TARGET_CENTROID_DISPLACEMENT_DIAGONAL_MAX = 0.50
TARGET_CENTROID_DISPLACEMENT_FLOOR_PX = 12.0


@dataclass(frozen=True)
class TargetCandidate:
    candidate_id: str
    rank: int | None
    backend_index: int | None
    mask: np.ndarray | None
    centroid_px: tuple[float, float] | None
    bbox_xyxy: tuple[int, int, int, int] | None
    area_px: int | None
    score: float | None


@dataclass(frozen=True)
class TargetIdentityAnchor:
    """One-trial pixel evidence used only to verify the same target instance."""

    target_phrase: str
    initial_mask: np.ndarray
    centroid_px: tuple[float, float]
    bbox_xyxy: tuple[int, int, int, int]
    mask_area: int
    candidate_id: str
    frame_id: int | None


@dataclass(frozen=True)
class TargetReferenceAnchor:
    """Fixed visual pixel reference for one camera-stable pre-contact stage."""

    target_identity: str
    camera: str
    reference_point_px: tuple[float, float]
    reference_source: str
    source_frame_id: int | None
    source_bbox_px: tuple[int, int, int, int] | None
    source_mask_area: int | None
    camera_signature: tuple[Any, ...]
    valid: bool = True
    invalidation_reason: str | None = None

    def invalidate(self, reason: str) -> "TargetReferenceAnchor":
        return replace(self, valid=False, invalidation_reason=str(reason))


def camera_calibration_signature(
    calibration: CameraCalibration, *, image_orientation: str = "identity",
) -> tuple[Any, ...]:
    """Return a stable, hashable signature for the camera/image projection."""
    position = np.asarray(calibration.position_world, dtype=float).reshape(3)
    rotation = np.asarray(calibration.camera_to_world, dtype=float).reshape(3, 3)
    return (
        str(calibration.name), int(calibration.width), int(calibration.height),
        round(float(calibration.fovy_deg), 9),
        *(round(float(value), 9) for value in position),
        *(round(float(value), 9) for value in rotation.reshape(-1)),
        int(calibration.rotation_degrees), str(calibration.flip), str(image_orientation),
    )


def make_target_reference_anchor(
    identity_anchor: TargetIdentityAnchor,
    *,
    camera: str,
    camera_signature: Sequence[Any],
    reference_source: str = "initial_associated_sam_mask_centroid",
) -> TargetReferenceAnchor:
    """Freeze a caller-selected associated centroid as the control reference."""
    return TargetReferenceAnchor(
        target_identity=identity_anchor.target_phrase,
        camera=str(camera),
        reference_point_px=tuple(float(value) for value in identity_anchor.centroid_px),
        reference_source=str(reference_source),
        source_frame_id=identity_anchor.frame_id,
        source_bbox_px=identity_anchor.bbox_xyxy,
        source_mask_area=identity_anchor.mask_area,
        camera_signature=tuple(camera_signature),
    )


def invalidate_target_reference_anchor(
    anchor: TargetReferenceAnchor | None, reason: str
) -> TargetReferenceAnchor | None:
    return anchor.invalidate(reason) if anchor is not None and anchor.valid else anchor


def frozen_reference_error(
    reference_point_px: Sequence[float] | None,
    eef_projection_px: Sequence[float] | None,
    *,
    reference_valid: bool = True,
) -> float | None:
    """Euclidean pixel distance to a fixed visual target reference."""
    if not reference_valid or reference_point_px is None or eef_projection_px is None:
        return None
    try:
        reference = np.asarray(reference_point_px, dtype=float).reshape(2)
        eef = np.asarray(eef_projection_px, dtype=float).reshape(2)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(eef)):
        return None
    return float(np.linalg.norm(reference - eef))


def compare_alignment_improvements(
    *, predicted_improvement_px: float | None, actual_improvement_px: float | None,
) -> dict[str, float | None]:
    """Compare projected model progress with observed fixed-reference progress."""
    if predicted_improvement_px is None or actual_improvement_px is None:
        return {"predicted_improvement_px": predicted_improvement_px,
                "actual_improvement_px": actual_improvement_px,
                "prediction_residual_actual_minus_predicted_px": None}
    predicted, actual = float(predicted_improvement_px), float(actual_improvement_px)
    if not math.isfinite(predicted) or not math.isfinite(actual):
        return {"predicted_improvement_px": predicted,
                "actual_improvement_px": actual,
                "prediction_residual_actual_minus_predicted_px": None}
    return {"predicted_improvement_px": predicted,
            "actual_improvement_px": actual,
            "prediction_residual_actual_minus_predicted_px": actual - predicted}


def reference_invalidation_reason(
    *,
    previous_camera_signature: Sequence[Any] | None,
    current_camera_signature: Sequence[Any] | None,
    possible_contact: bool = False,
    target_motion_evidence: bool = False,
    grasp_event: bool = False,
    release_event: bool = False,
    explicit_reground: bool = False,
) -> str | None:
    if explicit_reground:
        return "explicit_re_ground"
    if (previous_camera_signature is not None and current_camera_signature is not None
            and tuple(previous_camera_signature) != tuple(current_camera_signature)):
        return "camera_changed"
    if possible_contact:
        return "possible_contact"
    if target_motion_evidence:
        return "target_motion_evidence"
    if grasp_event:
        return "grasp_event"
    if release_event:
        return "release_event"
    return None


def _runtime_invalidation_signals(evidence: Mapping[str, Any]) -> dict[str, bool]:
    """Read only explicit contact/gripper event signals already present in evidence."""
    def active(value: Any, states: set[str] | None = None) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().upper() in (states or {"TRUE", "YES", "ACTIVE"})
        return False

    contact_state = evidence.get("contact_state")
    gripper_event = evidence.get("gripper_event")
    gripper_event_name = (gripper_event.strip().upper()
                          if isinstance(gripper_event, str) else "")
    return {
        "possible_contact": active(evidence.get("possible_contact"),
                                   {"TRUE", "YES", "CONTACT", "POSSIBLE_CONTACT"})
        or active(evidence.get("contact_detected"), {"TRUE", "YES", "CONTACT"})
        or active(contact_state, {"TRUE", "YES", "CONTACT", "POSSIBLE_CONTACT"})
        or (isinstance(contact_state, str)
            and contact_state.strip().upper() in {"CONTACT", "POSSIBLE_CONTACT"}),
        "target_motion_evidence": active(evidence.get("target_motion_evidence"),
                                         {"TRUE", "YES", "MOVED", "MOTION"}),
        "grasp_event": active(evidence.get("grasp_event"), {"TRUE", "YES", "GRASP", "GRASPED"})
        or gripper_event_name in {"GRASP", "GRASPED"},
        "release_event": active(evidence.get("release_event"), {"TRUE", "YES", "RELEASE", "RELEASED"})
        or gripper_event_name in {"RELEASE", "RELEASED"},
    }


@dataclass(frozen=True)
class TargetAssociation:
    status: str
    candidate: TargetCandidate | None
    candidate_metrics: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class TargetSegmentation:
    visible: bool
    mask: np.ndarray | None
    centroid_px: tuple[float, float] | None
    bbox_xyxy: tuple[int, int, int, int] | None
    area_px: int | None
    quality_score: float | None
    response: Mapping[str, Any]
    candidates: tuple[TargetCandidate, ...] = ()
    selected_candidate_id: str | None = None
    identity_status: str = "UNANCHORED"
    association_metrics: Mapping[str, Any] | None = None


def decode_sam3_mask(value: Any, expected_shape: tuple[int, int]) -> np.ndarray | None:
    """Decode a SAM3 mask and reject resolution mismatches without resizing."""
    if not isinstance(value, Mapping):
        return None
    encoded = value.get("base64")
    if not isinstance(encoded, str) or not encoded:
        return None
    try:
        with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    except (ValueError, OSError, TypeError):
        return None
    if mask.shape != tuple(int(value) for value in expected_shape):
        return None
    return mask


def segmentation_from_response(
    response: Mapping[str, Any], image_shape: tuple[int, int]
) -> TargetSegmentation:
    details = response.get("details")
    detections = details.get("detections") if isinstance(details, Mapping) else None
    if not bool(response.get("success")) or not isinstance(detections, list):
        return TargetSegmentation(False, None, None, None, None, None, response)
    candidates = tuple(
        _candidate_from_detection(item, index=index, image_shape=image_shape)
        for index, item in enumerate(detections)
        if isinstance(item, Mapping)
    )
    candidate = next((item for item in candidates if item.mask is not None and item.area_px), None)
    if candidate is None:
        return TargetSegmentation(False, None, None, None, None, None, response,
                                  candidates=candidates)
    return TargetSegmentation(
        visible=True,
        mask=candidate.mask,
        centroid_px=candidate.centroid_px,
        bbox_xyxy=candidate.bbox_xyxy,
        area_px=candidate.area_px,
        quality_score=candidate.score,
        response=response,
        candidates=candidates,
        selected_candidate_id=candidate.candidate_id,
    )


def _segmentation_from_grounding_result(
    result: SemanticGroundingResult, image_shape: tuple[int, int]
) -> TargetSegmentation:
    candidates: list[TargetCandidate] = []
    for rank, item in enumerate(result.candidates):
        mask = np.asarray(item.mask, dtype=bool)
        if mask.shape != image_shape or not bool(mask.any()):
            continue
        ys, xs = np.nonzero(mask)
        candidates.append(TargetCandidate(
            candidate_id=item.candidate_id,
            rank=rank,
            backend_index=None,
            mask=mask,
            centroid_px=(float(xs.mean()), float(ys.mean())),
            bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)),
            area_px=int(mask.sum()),
            score=item.proposal_score,
        ))
    selected = next((item for item in candidates
                     if item.candidate_id == result.evidence.candidate_id), None)
    if selected is None:
        return TargetSegmentation(
            False, None, None, None, None, None, result.baseline_response,
            candidates=tuple(candidates),
        )
    return TargetSegmentation(
        True, selected.mask, selected.centroid_px, selected.bbox_xyxy,
        selected.area_px, selected.score, result.baseline_response,
        candidates=tuple(candidates), selected_candidate_id=selected.candidate_id,
    )


def _candidate_from_detection(
    detection: Mapping[str, Any], *, index: int, image_shape: tuple[int, int]
) -> TargetCandidate:
    mask = decode_sam3_mask(detection.get("mask"), image_shape)
    centroid = None
    bbox = None
    area = None
    if mask is not None and bool(mask.any()):
        ys, xs = np.nonzero(mask)
        centroid = (float(xs.mean()), float(ys.mean()))
        bbox = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
        area = int(mask.sum())
    else:
        raw_bbox = detection.get("bbox_xyxy")
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
            try:
                bbox = tuple(int(round(float(value))) for value in raw_bbox)
            except (TypeError, ValueError):
                bbox = None
        try:
            area = int(detection["area_px"]) if detection.get("area_px") is not None else None
        except (TypeError, ValueError):
            area = None
    try:
        score = float(detection["score"]) if detection.get("score") is not None else None
    except (TypeError, ValueError):
        score = None
    if score is not None and not math.isfinite(score):
        score = None
    try:
        rank = int(detection["rank"]) if detection.get("rank") is not None else index
    except (TypeError, ValueError):
        rank = index
    try:
        backend_index = (int(detection["backend_index"])
                         if detection.get("backend_index") is not None else None)
    except (TypeError, ValueError):
        backend_index = None
    candidate_id = str(backend_index if backend_index is not None else rank)
    return TargetCandidate(candidate_id, rank, backend_index, mask, centroid, bbox, area, score)


def make_target_identity_anchor(
    segmentation: TargetSegmentation, *, target_phrase: str, frame_id: int | None
) -> TargetIdentityAnchor | None:
    candidate = next(
        (item for item in segmentation.candidates
         if item.candidate_id == segmentation.selected_candidate_id), None
    )
    if (candidate is None or candidate.mask is None or candidate.centroid_px is None
            or candidate.bbox_xyxy is None or not candidate.area_px):
        return None
    return TargetIdentityAnchor(
        target_phrase=str(target_phrase),
        initial_mask=candidate.mask.copy(),
        centroid_px=candidate.centroid_px,
        bbox_xyxy=candidate.bbox_xyxy,
        mask_area=int(candidate.area_px),
        candidate_id=candidate.candidate_id,
        frame_id=frame_id,
    )


def associate_target_candidate(
    anchor: TargetIdentityAnchor,
    candidates: Sequence[TargetCandidate],
) -> TargetAssociation:
    """Match by overlap and local centroid continuity; never force a weak match."""
    ax0, ay0, ax1, ay1 = anchor.bbox_xyxy
    diagonal = math.hypot(ax1 - ax0, ay1 - ay0)
    max_displacement = max(
        TARGET_CENTROID_DISPLACEMENT_FLOOR_PX,
        TARGET_CENTROID_DISPLACEMENT_DIAGONAL_MAX * diagonal,
    )
    metrics: list[dict[str, Any]] = []
    eligible: list[tuple[TargetCandidate, dict[str, Any]]] = []
    for candidate in candidates:
        if (candidate.mask is None or candidate.mask.shape != anchor.initial_mask.shape
                or candidate.centroid_px is None or candidate.bbox_xyxy is None
                or not candidate.area_px):
            metrics.append({"candidate_id": candidate.candidate_id, "eligible": False,
                            "reason": "candidate_mask_or_geometry_unavailable"})
            continue
        intersection = int(np.logical_and(anchor.initial_mask, candidate.mask).sum())
        union = int(np.logical_or(anchor.initial_mask, candidate.mask).sum())
        mask_iou = float(intersection / union) if union else 0.0
        bx0, by0, bx1, by1 = candidate.bbox_xyxy
        ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
        bbox_intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        anchor_box_area = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        candidate_box_area = max(0, bx1 - bx0) * max(0, by1 - by0)
        bbox_union = anchor_box_area + candidate_box_area - bbox_intersection
        bbox_iou = float(bbox_intersection / bbox_union) if bbox_union else 0.0
        displacement = float(np.linalg.norm(
            np.asarray(candidate.centroid_px, dtype=float) - np.asarray(anchor.centroid_px, dtype=float)
        ))
        area_ratio = float(candidate.area_px / anchor.mask_area)
        is_eligible = mask_iou >= TARGET_MASK_IOU_MIN and displacement <= max_displacement
        item = {
            "candidate_id": candidate.candidate_id,
            "rank": candidate.rank,
            "backend_index": candidate.backend_index,
            "sam_score": candidate.score,
            "mask_iou": mask_iou,
            "bbox_iou": bbox_iou,
            "centroid_displacement_px": displacement,
            "centroid_displacement_limit_px": max_displacement,
            "area_ratio": area_ratio,
            "eligible": bool(is_eligible),
            "reason": "continuity_gates_passed" if is_eligible else "continuity_gate_failed",
        }
        metrics.append(item)
        if is_eligible:
            eligible.append((candidate, item))
    if not eligible:
        return TargetAssociation("TARGET_IDENTITY_LOST", None, tuple(metrics))
    candidate, _ = max(
        eligible,
        key=lambda pair: (float(pair[1]["mask_iou"]),
                          -float(pair[1]["centroid_displacement_px"])),
    )
    return TargetAssociation("SAME_TARGET", candidate, tuple(metrics))


def alignment_verification_metrics(
    error_before_px: float | None,
    error_after_px: float | None,
    *,
    identity_status: str,
) -> dict[str, Any]:
    if identity_status != "SAME_TARGET" or error_before_px is None or error_after_px is None:
        return {"verification_status": "TARGET_IDENTITY_LOST", "error_after_px": None,
                "actual_improvement_px": None, "alignment_improved": None}
    before, after = float(error_before_px), float(error_after_px)
    return {"verification_status": "SAME_TARGET", "error_after_px": after,
            "actual_improvement_px": before - after, "alignment_improved": bool(after < before)}


def resolve_object_relative_geometry(
    *,
    target_centroid_px: Sequence[float] | None = None,
    target_reference_px: Sequence[float] | None = None,
    eef_position_xyz_m: Sequence[float] | None,
    calibration: CameraCalibration | None,
    move_vectors: Mapping[str, Sequence[float]],
    workspace_z_bounds_m: Sequence[float] | None,
    requested_displacement_m: float = REQUESTED_ALIGNMENT_M,
    candidate_scales_m: Sequence[float] | None = None,
    scale_contracts: Mapping[float, Mapping[str, Any]] | None = None,
    canonical_image_adapter: CanonicalImageAdapter | None = None,
) -> dict[str, Any]:
    """Project six physical hypotheses and rank them by predicted image error."""
    invalid = {
        "camera_projection_valid": False,
        "object_relative_alignment_valid": False,
        "candidate_directions": [],
        "chosen_candidate": None,
        "candidate_lattice": [],
        "chosen_lattice_candidate": None,
        "reason": None,
    }
    if calibration is None:
        invalid["reason"] = "camera_calibration_invalid"
        return invalid
    try:
        selected_reference = (target_reference_px if target_reference_px is not None
                              else target_centroid_px)
        target = np.asarray(selected_reference, dtype=float).reshape(2)
        position = np.asarray(eef_position_xyz_m, dtype=float).reshape(3)
        bounds = np.asarray(workspace_z_bounds_m, dtype=float).reshape(2)
    except (TypeError, ValueError):
        invalid["reason"] = "target_or_eef_geometry_missing"
        return invalid
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(position)):
        invalid["reason"] = "target_or_eef_geometry_nonfinite"
        return invalid
    if not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
        invalid["reason"] = "workspace_bounds_invalid"
        return invalid
    if not math.isfinite(float(requested_displacement_m)) or requested_displacement_m <= 0:
        invalid["reason"] = "requested_displacement_invalid"
        return invalid
    current = project_point(calibration, position)
    if current is None or not current["in_frame"]:
        invalid["reason"] = "eef_projection_invalid"
        return invalid
    def project_to_observation(point_xy: Sequence[float]) -> tuple[float, float]:
        if canonical_image_adapter is None:
            # Preserve the camera_geometry row-down convention for standalone callers.
            return float(point_xy[0]), float(point_xy[1])
        return canonical_image_adapter.transform_projected_point(
            point_xy, width=calibration.width, height=calibration.height)

    current_pixel = np.asarray(project_to_observation(current["pixel_xy"]), dtype=float)
    before_error = float(np.linalg.norm(target - current_pixel))
    candidates: list[dict[str, Any]] = []
    for direction in DIRECTION_ORDER:
        token = f"MV_{direction}"
        try:
            unit = np.asarray(move_vectors[token], dtype=float).reshape(3)
        except (KeyError, TypeError, ValueError):
            candidates.append({"direction": direction, "valid": False,
                               "reason": "direction_mapping_missing"})
            continue
        norm = float(np.linalg.norm(unit))
        if not np.all(np.isfinite(unit)) or not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            candidates.append({"direction": direction, "valid": False,
                               "reason": "direction_mapping_invalid"})
            continue
        hypothetical = position + unit * float(requested_displacement_m)
        if hypothetical[2] < bounds[0] or hypothetical[2] > bounds[1]:
            candidates.append({"direction": direction, "direction_unit": unit.tolist(),
                               "valid": False, "reason": "workspace_boundary"})
            continue
        projected = project_point(calibration, hypothetical)
        if projected is None or not projected["in_frame"]:
            candidates.append({"direction": direction, "direction_unit": unit.tolist(),
                               "hypothetical_eef_xyz_m": hypothetical.tolist(), "valid": False,
                               "reason": "hypothetical_projection_invalid"})
            continue
        projected_canonical = project_to_observation(projected["pixel_xy"])
        after_error = float(np.linalg.norm(target - np.asarray(projected_canonical, dtype=float)))
        candidates.append({
            "direction": direction,
            "physical_vector_xyz": unit.tolist(),
            "direction_unit": unit.tolist(),
            "hypothetical_eef_xyz_m": hypothetical.tolist(),
            "hypothetical_projection_px": list(projected_canonical),
            "hypothetical_projection_raw_px": projected["pixel_xy"],
            "predicted_error_after_px": after_error,
            "predicted_improvement_px": before_error - after_error,
            "valid": True,
        })
    valid = [item for item in candidates if item.get("valid")]
    if not valid:
        invalid.update({"camera_projection_valid": True, "candidate_directions": candidates,
                        "reason": "no_valid_physical_candidate"})
        return invalid
    best = max(valid, key=lambda item: item["predicted_improvement_px"])
    result = {
        "camera_projection_valid": True,
        "object_relative_alignment_valid": bool(best["predicted_improvement_px"] > 0.0),
        "eef_projection_px": current_pixel.tolist(),
        "eef_projection_raw_px": current["pixel_xy"],
        "pixel_error_before_px": before_error,
        "candidate_directions": candidates,
        "chosen_candidate": best if best["predicted_improvement_px"] > 0.0 else None,
        "reason": ("deterministic_object_relative_geometry"
                   if best["predicted_improvement_px"] > 0.0
                   else "no_candidate_predicted_to_improve_alignment"),
    }
    if candidate_scales_m is not None:
        contracts = {float(scale): contract for scale, contract in (scale_contracts or {}).items()}
        lattice: list[dict[str, Any]] = []
        for raw_scale in candidate_scales_m:
            try:
                scale = float(raw_scale)
            except (TypeError, ValueError):
                continue
            contract = contracts.get(scale)
            contract_valid = bool(contract and contract.get("verified"))
            try:
                max_ticks = int(contract["max_ticks"]) if contract else None
            except (KeyError, TypeError, ValueError):
                max_ticks = None
            if max_ticks is None or not 1 <= max_ticks <= 10:
                contract_valid = False
            for direction in DIRECTION_ORDER:
                token = f"MV_{direction}"
                row: dict[str, Any] = {
                    "direction": direction,
                    "displacement_m": scale,
                    "displacement_mm": scale * 1000.0,
                    "scale_contract_valid": contract_valid,
                    "max_ticks": max_ticks,
                    "workspace_valid": False,
                    "valid": False,
                    "reason": None,
                }
                if not math.isfinite(scale) or not 0 < scale <= 0.009:
                    row["reason"] = "scale_outside_calibrated_range"
                    lattice.append(row)
                    continue
                try:
                    unit = np.asarray(move_vectors[token], dtype=float).reshape(3)
                except (KeyError, TypeError, ValueError):
                    row["reason"] = "direction_mapping_missing"
                    lattice.append(row)
                    continue
                norm = float(np.linalg.norm(unit))
                if (not np.all(np.isfinite(unit))
                        or not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6)):
                    row["reason"] = "direction_mapping_invalid"
                    lattice.append(row)
                    continue
                hypothetical = position + unit * scale
                row.update({"direction_unit": unit.tolist(),
                            "hypothetical_eef_xyz_m": hypothetical.tolist()})
                if hypothetical[2] < bounds[0] or hypothetical[2] > bounds[1]:
                    row["reason"] = "workspace_boundary"
                    lattice.append(row)
                    continue
                row["workspace_valid"] = True
                projected = project_point(calibration, hypothetical)
                if projected is None or not projected["in_frame"]:
                    row["reason"] = "hypothetical_projection_invalid"
                    lattice.append(row)
                    continue
                projected_canonical = project_to_observation(projected["pixel_xy"])
                after_error = float(np.linalg.norm(target - np.asarray(projected_canonical)))
                improvement = before_error - after_error
                final_valid = contract_valid and improvement > 0.0
                row.update({
                    "predicted_projection_px": list(projected_canonical),
                    "predicted_error_px": after_error,
                    "predicted_improvement_px": improvement,
                    "valid": final_valid,
                    "reason": ("scale_contract_not_verified" if not contract_valid else
                               None if improvement > 0.0 else "no_predicted_improvement"),
                })
                lattice.append(row)
        eligible = [item for item in lattice if item.get("valid")]
        result["candidate_lattice"] = lattice
        result["chosen_lattice_candidate"] = (
            max(eligible, key=lambda item: item["predicted_improvement_px"])
            if eligible else None
        )
        result["multiscale_alignment_valid"] = bool(eligible)
    return result


def make_alignment_option(state: BeliefState) -> RuntimeOption | None:
    """Turn a pre-resolved Runtime geometry result into one sealed semantic option."""
    relative = state.object_relative_state
    geometry = state.relevant_geometry
    choice = geometry.get("chosen_candidate")
    if (relative is None or not relative.target_visible or relative.eef_projection_px is None
            or relative.target_centroid_px is None or relative.target_reference_point_px is None
            or not relative.target_reference_valid or not isinstance(choice, Mapping)
            or relative.target_identity_status not in {"ANCHORED", "SAME_TARGET"}
            or not bool(geometry.get("camera_projection_valid"))
            or not bool(geometry.get("workspace_valid"))
            or not bool(geometry.get("object_relative_alignment_valid"))):
        return None
    try:
        direction = str(choice["direction"])
        unit = tuple(float(value) for value in choice["direction_unit"])
        before = float(geometry["pixel_error_before_px"])
        predicted_after = float(choice["predicted_error_after_px"])
        predicted_improvement = float(choice["predicted_improvement_px"])
    except (KeyError, TypeError, ValueError):
        return None
    if len(unit) != 3 or not all(math.isfinite(value) for value in (before, predicted_after,
                                                                     predicted_improvement)):
        return None
    spec = BoundedMicroMotionSpec(
        direction=direction,
        direction_unit=unit,
        requested_displacement_m=REQUESTED_ALIGNMENT_M,
        max_ticks=MAX_ALIGNMENT_TICKS,
        control_tick_step_m=CONTROL_TICK_STEP_M,
    )
    return RuntimeOption(
        option_id="ALIGN_TO_TARGET_SMALL",
        option_type="object_relative_verified_alignment",
        description="Reduce the projected EEF-to-target image error with one bounded micro-motion.",
        preconditions={
            "observation_fresh": True,
            "object_relative_state.target_visible": True,
            "relevant_geometry.camera_projection_valid": True,
            "relevant_geometry.workspace_valid": True,
            "relevant_geometry.object_relative_alignment_valid": True,
        },
        expected_effect={
            "entity_key": (state.runtime_entity_state.entity_key
                           if state.runtime_entity_state is not None else "target"),
            "target_reference_px": list(relative.target_reference_point_px),
            "image_error_before_px": before,
            "predicted_image_error_after_px": predicted_after,
            "predicted_improvement_px": predicted_improvement,
            "verification": "reproject_eef_to_frozen_target_reference",
        },
        primitive=PrimitiveCommand(
            kind="micro_motion",
            max_steps=1,
            max_duration_s=5.0,
            micro_motion_spec=spec,
        ),
        confidence=1.0,
        evidence=state.evidence_refs,
        evidence_frame_id=state.frame_id,
    )


class ObjectRelativeAlignmentOptionGenerator:
    """Expose one semantic alignment option only after Runtime resolves geometry."""

    def generate(self, state: BeliefState) -> list[RuntimeOption]:
        option = make_alignment_option(state)
        return [option] if option is not None else []


def make_multiscale_alignment_option(state: BeliefState) -> RuntimeOption | None:
    """Seal the geometry-selected, Stage-A-verified scale into one semantic option."""
    relative = state.object_relative_state
    geometry = state.relevant_geometry
    choice = geometry.get("chosen_lattice_candidate")
    if (relative is None or not relative.target_visible or relative.eef_projection_px is None
            or relative.target_reference_point_px is None or not relative.target_reference_valid
            or relative.target_identity_status not in {"ANCHORED", "SAME_TARGET"}
            or not bool(geometry.get("camera_projection_valid"))
            or not bool(geometry.get("workspace_valid"))
            or not bool(geometry.get("multiscale_alignment_valid"))
            or not isinstance(choice, Mapping)
            or not bool(choice.get("valid"))
            or not bool(choice.get("scale_contract_valid"))
            or not bool(choice.get("workspace_valid"))):
        return None
    try:
        direction = str(choice["direction"])
        unit = tuple(float(value) for value in choice["direction_unit"])
        displacement = float(choice["displacement_m"])
        max_ticks = int(choice["max_ticks"])
        before = float(geometry["pixel_error_before_px"])
        predicted_after = float(choice["predicted_error_px"])
        predicted_improvement = float(choice["predicted_improvement_px"])
    except (KeyError, TypeError, ValueError):
        return None
    if len(unit) != 3 or not all(math.isfinite(value) for value in
                                 (displacement, before, predicted_after, predicted_improvement)):
        return None
    spec = BoundedMicroMotionSpec(
        direction=direction, direction_unit=unit,
        requested_displacement_m=displacement, max_ticks=max_ticks,
        control_tick_step_m=CONTROL_TICK_STEP_M,
    )
    return RuntimeOption(
        option_id="ALIGN_TO_TARGET_BOUNDED",
        option_type="object_relative_verified_alignment",
        description="Reduce frozen target-reference error using a verified bounded realization.",
        preconditions={
            "observation_fresh": True,
            "object_relative_state.target_visible": True,
            "relevant_geometry.camera_projection_valid": True,
            "relevant_geometry.workspace_valid": True,
            "relevant_geometry.multiscale_alignment_valid": True,
        },
        expected_effect={
            "entity_key": (state.runtime_entity_state.entity_key
                           if state.runtime_entity_state is not None else "target"),
            "target_reference_px": list(relative.target_reference_point_px),
            "image_error_before_px": before,
            "predicted_image_error_after_px": predicted_after,
            "predicted_improvement_px": predicted_improvement,
            "physical_direction": direction,
            "physical_displacement_m": displacement,
            "physical_scale_contract_verified": True,
            "verification": "reproject_eef_to_frozen_target_reference",
        },
        primitive=PrimitiveCommand(
            kind="micro_motion", max_steps=1, max_duration_s=5.0,
            micro_motion_spec=spec,
        ),
        confidence=1.0, evidence=state.evidence_refs, evidence_frame_id=state.frame_id,
    )


class MultiScaleAlignmentOptionGenerator:
    """Expose one semantic ALIGN option for the lowest-error valid lattice point."""

    def generate(self, state: BeliefState) -> list[RuntimeOption]:
        option = make_multiscale_alignment_option(state)
        return [option] if option is not None else []


class ObjectRelativePerceptionObserver:
    """Add SAM3 target evidence and camera-calibrated hypotheses to an observer."""

    def __init__(
        self,
        base_observer: Any,
        sam3: Any,
        *,
        target_phrase: str | None = None,
        entity_spec: EntitySpec | None = None,
        move_vectors: Mapping[str, Sequence[float]],
        confidence_threshold: float = SAM3_CONFIDENCE_THRESHOLD,
        canonical_image_adapter: CanonicalImageAdapter | None = None,
        scene_ready_required: bool = False,
        alignment_scales_m: Sequence[float] | None = None,
        scale_contracts: Mapping[float, Mapping[str, Any]] | None = None,
        metric_depth_provider: DepthProvider | None = None,
        semantic_grounding_binder: SemanticGroundingBinder | None = None,
    ) -> None:
        self.base_observer = base_observer
        self.sam3 = sam3
        if entity_spec is None:
            if target_phrase is None:
                raise ValueError("entity_spec is required")
            entity_spec = EntitySpec("target", str(target_phrase), "MANIPULAND")
        elif target_phrase is not None and str(target_phrase) != entity_spec.semantic_phrase:
            raise ValueError("target_phrase must agree with entity_spec.semantic_phrase")
        self.entity_spec = entity_spec
        self.target_phrase = entity_spec.semantic_phrase
        self.confidence_threshold = float(confidence_threshold)
        self.move_vectors = {str(key): tuple(float(v) for v in value)
                             for key, value in move_vectors.items()}
        self.canonical_image_adapter = canonical_image_adapter or CanonicalImageAdapter()
        self.scene_ready_required = bool(scene_ready_required)
        self.alignment_scales_m = (tuple(float(v) for v in alignment_scales_m)
                                   if alignment_scales_m is not None else None)
        self.scale_contracts = dict(scale_contracts or {})
        self.metric_depth_provider = metric_depth_provider
        self.semantic_grounding_binder = semantic_grounding_binder
        self.grounding_evidence: EntityGroundingEvidence | None = None
        self.semantic_grounding_result: SemanticGroundingResult | None = None
        self._grounding_attempted = False
        self.metric_reference_anchor: MetricEntityReference | None = None
        # SceneReadyEvidence remains for historical/offline comparisons only. The
        # formal gate is the conjunction of target-independent RGB motion and
        # same-entity visual evidence below.
        self.scene_ready_evidence = SceneReadyEvidence() if self.scene_ready_required else None
        self.scene_motion_evidence = SceneMotionReady() if self.scene_ready_required else None
        self.entity_observation_evidence = (
            EntityObservationReady() if self.scene_ready_required else None
        )
        self.grounding_query_normalizer = GroundingQueryNormalizer()
        self.grounding_query = self.grounding_query_normalizer.normalize(self.target_phrase)
        self._readiness_active = bool(self.scene_ready_required)
        self.identity_anchor: TargetIdentityAnchor | None = None
        self.reference_anchor: TargetReferenceAnchor | None = None
        self._camera_signature: tuple[Any, ...] | None = None
        self._reference_reground_requested = False
        self._last_reference_invalidation_reason: str | None = None
        self.last_segmentation: TargetSegmentation | None = None
        self.last_calibration: CameraCalibration | None = None
        self.last_resolution: dict[str, Any] = {}
        self.perception_history: list[dict[str, Any]] = []

    @property
    def scene_ready(self) -> bool:
        if not self.scene_ready_required:
            return True
        return bool(
            self.scene_motion_ready and self.entity_observation_ready
            and self.grounding_success and self.identity_valid
        )

    @property
    def scene_motion_ready(self) -> bool:
        return (self.scene_motion_evidence.ready
                if self.scene_motion_evidence is not None else True)

    @property
    def entity_observation_ready(self) -> bool:
        return (self.entity_observation_evidence.ready
                if self.entity_observation_evidence is not None else True)

    @property
    def grounding_success(self) -> bool:
        segmentation = self.last_segmentation
        return bool(segmentation and segmentation.visible)

    @property
    def identity_valid(self) -> bool:
        segmentation = self.last_segmentation
        return bool(segmentation and segmentation.identity_status in {"ANCHORED", "SAME_TARGET"})

    @property
    def reference_valid(self) -> bool:
        return bool(self.reference_anchor and self.reference_anchor.valid)

    def begin_readiness(self) -> None:
        """Start fresh readiness evidence after RobotReady has completed."""
        if not self.scene_ready_required:
            return
        self._readiness_active = True
        self.scene_motion_evidence.reset()
        self.entity_observation_evidence.reset()
        if self.scene_ready_evidence is not None:
            self.scene_ready_evidence.reset()

    def invalidate_target_reference(self, reason: str) -> None:
        """Invalidate the active reference; it cannot be re-established implicitly."""
        self._last_reference_invalidation_reason = str(reason)
        self.reference_anchor = invalidate_target_reference_anchor(self.reference_anchor, reason)
        if self.metric_reference_anchor is not None and self.metric_reference_anchor.valid:
            self.metric_reference_anchor = self.metric_reference_anchor.invalidate(reason)

    def request_target_reference_reground(self) -> None:
        """Explicitly request a new identity association and visual reference."""
        self.invalidate_target_reference("explicit_re_ground")
        self._reference_reground_requested = True

    def observe_for_execution_tick(self, environment: Any) -> RobotObservation:
        """Collect EEF-only tick evidence; SAM3 runs once after the bounded motion."""
        return self.base_observer.observe(environment)

    def observe(self, environment: Any) -> RobotObservation:
        return self.observe_from_base_observation(
            environment, self.base_observer.observe(environment)
        )

    def observe_from_base_observation(
        self, environment: Any, base: RobotObservation,
    ) -> RobotObservation:
        """Add visual target evidence to one already acquired RGB observation."""
        raw_image = np.asarray(base.images.get("agentview"))
        if raw_image.ndim != 3 or raw_image.shape[2] != 3:
            raise ValueError("agentview source must be an HxWx3 RGB array")
        image = self.canonical_image_adapter.transform_image(raw_image)
        height, width = image.shape[:2]
        if self._reference_reground_requested:
            self.identity_anchor = None
            self.reference_anchor = None
            self.metric_reference_anchor = None
            self.grounding_evidence = None
            self.semantic_grounding_result = None
            self._grounding_attempted = False
            if self.scene_ready_evidence is not None:
                self.scene_ready_evidence.reset()
            if self.entity_observation_evidence is not None:
                self.entity_observation_evidence.reset()
            if self.scene_motion_evidence is not None:
                self.scene_motion_evidence.reset()
            self._readiness_active = True
            self._reference_reground_requested = False
        if self._readiness_active and self.scene_motion_evidence is not None:
            self.scene_motion_evidence.update(image)
        if (self.semantic_grounding_binder is not None
                and not self._grounding_attempted and self.scene_motion_ready):
            self._grounding_attempted = True
            self.semantic_grounding_result = self.semantic_grounding_binder.ground(
                image,
                entity_key=self.entity_spec.key,
                semantic_phrase=self.entity_spec.semantic_phrase,
                semantic_query=self.grounding_query,
            )
            self.grounding_evidence = self.semantic_grounding_result.evidence
            response = self.semantic_grounding_result.baseline_response
            candidate_segmentation = _segmentation_from_grounding_result(
                self.semantic_grounding_result, (height, width)
            )
        elif (self.semantic_grounding_binder is not None
              and not self._grounding_attempted):
            response = {"success": False, "deferred_until_scene_motion_ready": True}
            candidate_segmentation = TargetSegmentation(
                False, None, None, None, None, None, response,
            )
        elif self.semantic_grounding_binder is not None and self.identity_anchor is None:
            response = (self.semantic_grounding_result.baseline_response
                        if self.semantic_grounding_result is not None else {"success": False})
            candidate_segmentation = TargetSegmentation(
                False, None, None, None, None, None, response,
                candidates=(self.last_segmentation.candidates
                            if self.last_segmentation is not None else ()),
            )
        elif self.semantic_grounding_binder is not None:
            center_x, center_y = self.identity_anchor.centroid_px
            response = self.sam3.segment_points(
                image, [{"x": float(center_x), "y": float(center_y), "label": 1}],
            )
            candidate_segmentation = segmentation_from_response(response, (height, width))
        else:
            response = self.sam3.segment(
                image, self.grounding_query, confidence_threshold=self.confidence_threshold
            )
            candidate_segmentation = segmentation_from_response(response, (height, width))
        details = response.get("details") if isinstance(response, Mapping) else None
        metadata = details.get("metadata") if isinstance(details, Mapping) else None
        reported_size = metadata.get("image_size") if isinstance(metadata, Mapping) else None
        if reported_size is not None and reported_size != [width, height]:
            candidate_segmentation = TargetSegmentation(
                False, None, None, None, None, None,
                {**dict(response), "resolution_mismatch": {
                    "expected_width_height": [width, height],
                    "reported_width_height": reported_size,
                }},
                candidates=candidate_segmentation.candidates,
            )
        if self.identity_anchor is None:
            self.identity_anchor = make_target_identity_anchor(
                candidate_segmentation,
                target_phrase=self.target_phrase,
                frame_id=base.frame_id,
            )
            if self.identity_anchor is not None:
                segmentation = TargetSegmentation(
                    candidate_segmentation.visible,
                    candidate_segmentation.mask,
                    candidate_segmentation.centroid_px,
                    candidate_segmentation.bbox_xyxy,
                    candidate_segmentation.area_px,
                    candidate_segmentation.quality_score,
                    candidate_segmentation.response,
                    candidates=candidate_segmentation.candidates,
                    selected_candidate_id=candidate_segmentation.selected_candidate_id,
                    identity_status="ANCHORED",
                )
            else:
                segmentation = TargetSegmentation(
                    False, None, None, None, None, None, candidate_segmentation.response,
                    candidates=candidate_segmentation.candidates,
                    identity_status="ANCHOR_UNAVAILABLE",
                )
        else:
            association = associate_target_candidate(
                self.identity_anchor, candidate_segmentation.candidates,
            )
            associated = association.candidate
            segmentation = TargetSegmentation(
                visible=associated is not None,
                mask=associated.mask if associated is not None else None,
                centroid_px=associated.centroid_px if associated is not None else None,
                bbox_xyxy=associated.bbox_xyxy if associated is not None else None,
                area_px=associated.area_px if associated is not None else None,
                quality_score=associated.score if associated is not None else None,
                response=candidate_segmentation.response,
                candidates=candidate_segmentation.candidates,
                selected_candidate_id=associated.candidate_id if associated is not None else None,
                identity_status=association.status,
                association_metrics={
                    "selected_candidate_id": associated.candidate_id if associated is not None else None,
                    "candidates": list(association.candidate_metrics),
                },
            )
        self.last_segmentation = segmentation
        raw = getattr(self.base_observer, "last_raw", None)
        position = getattr(raw, "eef_position_xyz", None)
        env = getattr(environment, "env", environment)
        calibration = None
        try:
            sim = getattr(env, "sim", None)
            if sim is None:
                sim = getattr(getattr(env, "env", None), "sim", None)
            calib = make_mujoco_calibrations(
                env if hasattr(env, "sim") else getattr(env, "env", env),
                {"agentview": "agentview"},
                image_shapes={"agentview": (height, width)},
                # MuJoCo projection first yields raw OpenGL-frame pixels; the shared
                # CanonicalImageAdapter below applies the same vertical row flip as RGB.
                rotations={"agentview": 0},
                flips={"agentview": "none"},
            )
            calibration = calib["agentview"]
        except (AttributeError, KeyError, TypeError, ValueError):
            calibration = None
        self.last_calibration = calibration
        metric_candidate: MetricEntityReference | None = None
        metric_depth: np.ndarray | None = None
        if self.metric_depth_provider is not None:
            try:
                estimate = self.metric_depth_provider.estimate(image)
                metric_depth = estimate.depth_m
                metric_candidate = metric_entity_reference_from_estimate(
                    entity_key=self.entity_spec.key,
                    camera="agentview",
                    source_frame_id=base.frame_id,
                    target_mask=(segmentation.mask if segmentation.visible else None),
                    estimate=estimate,
                    calibration=calibration,
                )
            except (AttributeError, TypeError, ValueError, RuntimeError, OverflowError):
                metric_candidate = None
        current_camera_signature = (camera_calibration_signature(
            calibration, image_orientation=self.canonical_image_adapter.orientation,
        )
                                   if calibration is not None else None)
        signals = _runtime_invalidation_signals(base.evidence)
        invalidation_reason = reference_invalidation_reason(
            previous_camera_signature=self._camera_signature,
            current_camera_signature=current_camera_signature,
            **signals,
        )
        if invalidation_reason is not None:
            self.invalidate_target_reference(invalidation_reason)
            if self.scene_ready_evidence is not None:
                self.scene_ready_evidence.reset()
        self._camera_signature = current_camera_signature

        scene_was_ready = self.scene_ready
        if (self._readiness_active and self.entity_observation_evidence is not None
                and self.scene_motion_evidence is not None and not scene_was_ready):
            self.entity_observation_evidence.update(
                entity_key=self.entity_spec.key,
                grounding_query=self.grounding_query,
                identity_status=segmentation.identity_status,
                candidate_id=segmentation.selected_candidate_id,
                mask=segmentation.mask if segmentation.visible else None,
                centroid_px=segmentation.centroid_px,
                bbox_xyxy=segmentation.bbox_xyxy,
                mask_area_px=segmentation.area_px,
                frame_id=base.frame_id,
            )
            if self.entity_observation_evidence.ready:
                representative = self.entity_observation_evidence.representative_sample()
                if representative is not None and self.identity_anchor is not None:
                    # Promote a same-identity medoid mask and median centroid. The
                    # original semantic phrase remains attached to the entity.
                    self.identity_anchor = replace(
                        self.identity_anchor,
                        initial_mask=representative["mask"].copy(),
                        centroid_px=tuple(float(value) for value in
                                          representative["reference_centroid_px"]),
                        bbox_xyxy=tuple(int(value) for value in representative["bbox_xyxy"]),
                        mask_area=int(representative["mask_area_px"]),
                        candidate_id=str(representative["candidate_id"]),
                        frame_id=representative["frame_id"],
                    )
        if self.scene_ready_evidence is not None and not scene_was_ready:
            # Compatibility evidence is retained for comparisons only and is not
            # consulted by scene_ready or any Runtime decision.
            self.scene_ready_evidence.update(
                target_identity_status=segmentation.identity_status,
                centroid_px=segmentation.centroid_px,
                bbox_xyxy=segmentation.bbox_xyxy,
                mask_area_px=segmentation.area_px,
            )

        # The initial identity association establishes the reference exactly once.
        # A later SAM centroid is never allowed to replace it. Re-grounding has an
        # explicit API so the caller must opt into a new identity/reference epoch.
        if (self.reference_anchor is None and self.identity_anchor is not None
                and segmentation.visible and current_camera_signature is not None):
            if self.scene_ready:
                self.reference_anchor = make_target_reference_anchor(
                    self.identity_anchor,
                    camera=calibration.name,
                    camera_signature=current_camera_signature,
                    reference_source=(
                        "same_entity_mask_medoid_median_centroid"
                        if self.scene_ready_required else "initial_associated_sam_mask_centroid"
                    ),
                )
                self._reference_reground_requested = False
        if (self.metric_depth_provider is not None and self.scene_ready
                and metric_candidate is not None):
            self.metric_reference_anchor = freeze_metric_reference(
                self.metric_reference_anchor, metric_candidate
            )
        reference_valid = bool(self.reference_anchor and self.reference_anchor.valid)
        reference_point = (self.reference_anchor.reference_point_px if reference_valid else None)
        workspace = base.evidence.get("relevant_geometry", {}).get("workspace_z_bounds_m")
        geometry = resolve_object_relative_geometry(
            target_reference_px=reference_point,
            eef_position_xyz_m=position,
            calibration=calibration,
            move_vectors=self.move_vectors,
            workspace_z_bounds_m=workspace,
            candidate_scales_m=self.alignment_scales_m,
            scale_contracts=self.scale_contracts,
            canonical_image_adapter=self.canonical_image_adapter,
        )
        self.last_resolution = geometry
        eef_projection = geometry.get("eef_projection_px")
        image_error = None
        error_norm = None
        if reference_valid and eef_projection is not None and reference_point is not None:
            delta = np.asarray(reference_point) - np.asarray(eef_projection)
            image_error = (float(delta[0]), float(delta[1]))
            error_norm = float(np.linalg.norm(delta))
        dynamic_sam_error = None
        if segmentation.visible and eef_projection is not None and segmentation.centroid_px is not None:
            dynamic_sam_error = float(np.linalg.norm(
                np.asarray(segmentation.centroid_px, dtype=float)
                - np.asarray(eef_projection, dtype=float)
            ))
        relative = ObjectRelativeState(
            target_phrase=self.target_phrase,
            target_visible=segmentation.visible,
            target_quality_score=segmentation.quality_score,
            target_centroid_px=segmentation.centroid_px,
            target_bbox_xyxy=(tuple(float(v) for v in segmentation.bbox_xyxy)
                              if segmentation.bbox_xyxy is not None else None),
            target_mask_area_px=segmentation.area_px,
            eef_projection_px=(tuple(float(v) for v in eef_projection)
                               if eef_projection is not None else None),
            image_error_px=image_error,
            image_error_norm_px=error_norm,
            camera="agentview",
            evidence_timestamp=time.monotonic(),
            source_width=width,
            source_height=height,
            target_identity_status=segmentation.identity_status,
            target_candidate_id=segmentation.selected_candidate_id,
            target_reference_point_px=(tuple(float(value) for value in reference_point)
                                       if reference_point is not None else None),
            target_reference_valid=reference_valid,
            target_reference_camera=(self.reference_anchor.camera if self.reference_anchor else None),
            target_reference_invalidation_reason=(
                self.reference_anchor.invalidation_reason if self.reference_anchor else None
            ),
            metric_entity_reference=self.metric_reference_anchor,
            scene_ready=self.scene_ready,
            scene_ready_gate_enabled=self.scene_ready_required,
            scene_motion_ready=self.scene_motion_ready,
            entity_observation_ready=self.entity_observation_ready,
        )
        runtime_entity_state = RuntimeEntityState(
            entity_key=self.entity_spec.key,
            semantic_phrase=self.entity_spec.semantic_phrase,
            role=self.entity_spec.role,
            identity_anchor=self.identity_anchor,
            reference_anchor=self.reference_anchor,
            visible=segmentation.visible,
            valid=(segmentation.visible
                   and segmentation.identity_status in {"ANCHORED", "SAME_TARGET"}
                   and reference_valid
                   and (self.semantic_grounding_binder is None
                        or bool(self.grounding_evidence and self.grounding_evidence.valid))),
            grounding_evidence=self.grounding_evidence,
        )
        self.perception_history.append({
            "image": image.copy(),
            "raw_image": np.ascontiguousarray(raw_image).copy(),
            "segmentation": segmentation,
            "candidates": segmentation.candidates,
            "identity_anchor": self.identity_anchor,
            "reference_anchor": self.reference_anchor,
            "calibration": calibration,
            "camera_signature": current_camera_signature,
            "reference_invalidation_signals": signals,
            "reference_epoch_reset_reason": self._last_reference_invalidation_reason,
            "scene_ready": self.scene_ready,
            "scene_ready_gate_enabled": self.scene_ready_required,
            "scene_motion_ready": self.scene_motion_ready,
            "entity_observation_ready": self.entity_observation_ready,
            "scene_motion_score": (
                self.scene_motion_evidence.last_normalized_rgb_difference
                if self.scene_motion_evidence is not None else None
            ),
            "scene_ready_evidence": (self.scene_ready_evidence.to_record()
                                     if self.scene_ready_evidence is not None else None),
            "grounding_query": self.grounding_query,
            "grounding_evidence": self.grounding_evidence,
            "semantic_grounding_result": self.semantic_grounding_result,
            "scene_motion_evidence": (self.scene_motion_evidence.to_record()
                                      if self.scene_motion_evidence is not None else None),
            "entity_observation_evidence": (self.entity_observation_evidence.to_record()
                                             if self.entity_observation_evidence is not None else None),
            "resolution": dict(geometry),
            "dynamic_sam_error_px": dynamic_sam_error,
            "object_relative_state": relative,
            "runtime_entity_state": runtime_entity_state,
            "frame_id": base.frame_id,
            "metric_entity_reference_candidate": metric_candidate,
            "metric_entity_reference_anchor": self.metric_reference_anchor,
            "metric_depth_estimate_m": metric_depth,
        })
        evidence = dict(base.evidence)
        evidence["target_identity"] = self.target_phrase
        evidence["object_relative_state"] = relative
        evidence["runtime_entity_state"] = runtime_entity_state
        merged_geometry = dict(evidence.get("relevant_geometry", {}) or {})
        merged_geometry.update({
            "camera_projection_valid": bool(geometry.get("camera_projection_valid")),
            "object_relative_alignment_valid": bool(geometry.get("object_relative_alignment_valid"))
                                                    and segmentation.visible,
            "object_relative_decision_owner": "runtime",
            "object_relative_decision_reason": geometry.get("reason"),
            "pixel_error_before_px": error_norm,
            "target_reference_point_px": reference_point,
            "target_reference_valid": reference_valid,
            "target_reference_invalidation_reason": (
                self.reference_anchor.invalidation_reason if self.reference_anchor else None
            ),
            "dynamic_sam_error_px": dynamic_sam_error,
            "candidate_directions": geometry.get("candidate_directions", []),
            "candidate_lattice": geometry.get("candidate_lattice", []),
            "chosen_lattice_candidate": geometry.get("chosen_lattice_candidate"),
            "multiscale_alignment_valid": bool(geometry.get("multiscale_alignment_valid")),
            "chosen_candidate": (geometry.get("chosen_candidate") if segmentation.visible else None),
            "sam3_error": response.get("error") if not segmentation.visible else None,
            "sam3_confidence_threshold": self.confidence_threshold,
            "sam3_candidate_count": len(segmentation.candidates),
            "target_identity_status": segmentation.identity_status,
            "target_identity_association": segmentation.association_metrics,
            "scene_ready": self.scene_ready,
            "scene_ready_gate_enabled": self.scene_ready_required,
            "scene_ready_evidence": (self.scene_ready_evidence.to_record()
                                     if self.scene_ready_evidence is not None else None),
            "scene_motion_ready": self.scene_motion_ready,
            "entity_observation_ready": self.entity_observation_ready,
            "scene_motion_evidence": (self.scene_motion_evidence.to_record()
                                      if self.scene_motion_evidence is not None else None),
            "entity_observation_evidence": (self.entity_observation_evidence.to_record()
                                             if self.entity_observation_evidence is not None else None),
        })
        if not segmentation.visible:
            merged_geometry["object_relative_alignment_valid"] = False
        evidence["relevant_geometry"] = merged_geometry
        return RobotObservation(
            observation_id=base.observation_id,
            frame_id=base.frame_id,
            images={**dict(base.images), "agentview": image},
            proprioception=base.proprioception,
            evidence=evidence,
            evidence_refs=base.evidence_refs,
            fresh=base.fresh,
            done=base.done,
        )

    def save_visual_artifacts(
        self,
        directory: str,
        *,
        image: np.ndarray,
        prefix: str,
        segmentation: TargetSegmentation | None = None,
        resolution: Mapping[str, Any] | None = None,
        selected_direction: str | None = None,
    ) -> dict[str, str]:
        """Write exact source RGB, decoded mask, and diagnostic overlay for review."""
        from pathlib import Path

        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        source = np.ascontiguousarray(image, dtype=np.uint8)
        height, width = source.shape[:2]
        rgb_path = output / f"{prefix}_rgb.png"
        Image.fromarray(source, mode="RGB").save(rgb_path)
        segmentation = segmentation if segmentation is not None else self.last_segmentation
        result = {"rgb": str(rgb_path)}
        overlay = Image.fromarray(source.copy(), mode="RGB").convert("RGBA")
        draw = ImageDraw.Draw(overlay)
        palette = ((30, 180, 255, 48), (40, 220, 80, 48), (255, 180, 20, 48),
                   (200, 60, 255, 48), (20, 220, 200, 48), (255, 120, 40, 48))
        candidate_records = []
        if segmentation is not None:
            for index, candidate in enumerate(segmentation.candidates):
                candidate_record = {
                    "candidate_id": candidate.candidate_id,
                    "rank": candidate.rank,
                    "backend_index": candidate.backend_index,
                    "score": candidate.score,
                    "area_px": candidate.area_px,
                    "centroid_px": candidate.centroid_px,
                    "bbox_xyxy": candidate.bbox_xyxy,
                    "selected": candidate.candidate_id == segmentation.selected_candidate_id,
                }
                if candidate.mask is not None:
                    candidate_path = output / f"{prefix}_candidate_{index:02d}_mask.png"
                    Image.fromarray(candidate.mask.astype(np.uint8) * 255, mode="L").save(candidate_path)
                    candidate_record["mask"] = str(candidate_path)
                    if candidate.candidate_id != segmentation.selected_candidate_id:
                        tint = np.zeros((*candidate.mask.shape, 4), dtype=np.uint8)
                        tint[candidate.mask] = palette[index % len(palette)]
                        overlay = Image.alpha_composite(overlay, Image.fromarray(tint, mode="RGBA"))
                candidate_records.append(candidate_record)
        draw = ImageDraw.Draw(overlay)
        if segmentation is not None:
            for candidate in segmentation.candidates:
                if candidate.bbox_xyxy is None:
                    continue
                x0, y0, x1, y1 = candidate.bbox_xyxy
                color = (255, 255, 0, 255) if candidate.candidate_id == segmentation.selected_candidate_id \
                    else (255, 180, 40, 255)
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=2)
            draw.text((x0, max(0, y0 - 12)),
                          f"{candidate.candidate_id} score={candidate.score}", fill=color)
        if segmentation is not None and segmentation.mask is not None:
            mask_path = output / f"{prefix}_mask.png"
            Image.fromarray(segmentation.mask.astype(np.uint8) * 255, mode="L").save(mask_path)
            tint = np.zeros((*segmentation.mask.shape, 4), dtype=np.uint8)
            tint[segmentation.mask] = (255, 24, 24, 96)
            overlay = Image.alpha_composite(overlay, Image.fromarray(tint, mode="RGBA"))
            draw = ImageDraw.Draw(overlay)
            if segmentation.centroid_px is not None:
                x, y = segmentation.centroid_px
                centroid_color = (255, 45, 45, 255) if prefix == "after" else (255, 255, 0, 255)
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=centroid_color, width=2)
                if prefix == "after":
                    draw.text((x + 7, y + 6), "SAM centroid: DIAGNOSTIC ONLY",
                              fill=centroid_color)
            result["mask"] = str(mask_path)
        reference_anchor = self.reference_anchor
        if reference_anchor is not None and source.shape[:2] == (height, width):
            x, y = reference_anchor.reference_point_px
            color = (255, 215, 0, 255) if reference_anchor.valid else (255, 80, 80, 255)
            draw = ImageDraw.Draw(overlay)
            draw.polygon(((x, y - 7), (x + 7, y), (x, y + 7), (x - 7, y)),
                         outline=color, fill=(0, 0, 0, 0))
            label = "FIXED TARGET REFERENCE" if reference_anchor.valid else "REFERENCE INVALID"
            draw.text((x + 8, y - 12), label, fill=color)
        current_resolution = resolution if resolution is not None else self.last_resolution
        eef = current_resolution.get("eef_projection_px")
        if eef is not None:
            x, y = (float(eef[0]), float(eef[1]))
            draw = ImageDraw.Draw(overlay)
            draw.line((x - 7, y, x + 7, y), fill=(0, 255, 255, 255), width=2)
            draw.line((x, y - 7, x, y + 7), fill=(0, 255, 255, 255), width=2)
            draw.text((x + 8, y + 4), "EEF", fill=(0, 255, 255, 255))
        candidate_directions = current_resolution.get("candidate_directions", [])
        selected_projection = None
        for item in candidate_directions:
            pixel = item.get("hypothetical_projection_px")
            if not item.get("valid") or pixel is None:
                continue
            x, y = (float(pixel[0]), float(pixel[1]))
            is_selected = item.get("direction") == selected_direction
            color = (255, 0, 255, 255) if is_selected else (255, 255, 255, 230)
            r = 6 if is_selected else 3
            draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)
            draw.text((x + r + 1, y), str(item.get("direction", "?")), fill=color)
            if is_selected:
                selected_projection = item.get("hypothetical_projection_px")
        if selected_projection is not None and eef is not None:
            draw.line((float(eef[0]), float(eef[1]), float(selected_projection[0]),
                       float(selected_projection[1])), fill=(255, 0, 255, 190), width=2)
        anchor = self.identity_anchor
        if anchor is not None and anchor.initial_mask.shape == source.shape[:2]:
            anchor_path = output / "target_anchor_mask.png"
            Image.fromarray(anchor.initial_mask.astype(np.uint8) * 255, mode="L").save(anchor_path)
            result["target_anchor_mask"] = str(anchor_path)
            anchor_tint = np.zeros((*anchor.initial_mask.shape, 4), dtype=np.uint8)
            anchor_tint[anchor.initial_mask] = (0, 210, 255, 70)
            anchor_overlay = Image.alpha_composite(Image.fromarray(source.copy(), mode="RGB").convert("RGBA"),
                                                   Image.fromarray(anchor_tint, mode="RGBA"))
            ImageDraw.Draw(anchor_overlay).text((8, 8),
                f"TARGET IDENTITY ANCHOR frame={anchor.frame_id} candidate={anchor.candidate_id}",
                fill=(0, 120, 255, 255))
            anchor_overlay_path = output / f"{prefix}_anchor_overlay.png"
            anchor_overlay.convert("RGB").save(anchor_overlay_path)
            result["anchor_overlay"] = str(anchor_overlay_path)
        if segmentation is not None and segmentation.identity_status == "TARGET_IDENTITY_LOST":
            draw = ImageDraw.Draw(overlay)
            draw.rectangle((0, 0, source.shape[1] - 1, 38), fill=(140, 0, 0, 230))
            draw.text((10, 12), "TARGET IDENTITY LOST", fill=(255, 255, 255, 255))
        overlay_path = output / f"{prefix}_overlay.png"
        overlay.convert("RGB").save(overlay_path)
        result["overlay"] = str(overlay_path)
        candidate_json_path = output / f"{prefix}_candidates.json"
        candidate_json_path.write_text(
            json.dumps({
                "identity_status": segmentation.identity_status if segmentation else None,
                "selected_candidate_id": segmentation.selected_candidate_id if segmentation else None,
                "association_metrics": segmentation.association_metrics if segmentation else None,
                "candidates": candidate_records,
            }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        result["candidates"] = str(candidate_json_path)
        return result

"""SAM3-backed object-relative geometry for one bounded Runtime V3 alignment."""

from __future__ import annotations

import base64
import io
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from core.capabilities.camera_geometry import (
    CameraCalibration,
    make_mujoco_calibrations,
    project_point,
)
from .observer import RobotObservation
from .options import BoundedMicroMotionSpec, PrimitiveCommand, RuntimeOption
from .state import BeliefState, ObjectRelativeState


DIRECTION_ORDER = ("FWD", "BACK", "LEFT", "RIGHT", "UP", "DOWN")
REQUESTED_ALIGNMENT_M = 0.003
CONTROL_TICK_STEP_M = 0.005
MAX_ALIGNMENT_TICKS = 5
SAM3_CONFIDENCE_THRESHOLD = 0.05


@dataclass(frozen=True)
class TargetSegmentation:
    visible: bool
    mask: np.ndarray | None
    centroid_px: tuple[float, float] | None
    bbox_xyxy: tuple[int, int, int, int] | None
    area_px: int | None
    quality_score: float | None
    response: Mapping[str, Any]


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
    detection = next((item for item in detections if isinstance(item, Mapping)), None)
    if detection is None:
        return TargetSegmentation(False, None, None, None, None, None, response)
    mask = decode_sam3_mask(detection.get("mask"), image_shape)
    if mask is None or not bool(mask.any()):
        return TargetSegmentation(False, None, None, None, None, None, response)
    ys, xs = np.nonzero(mask)
    score_raw = detection.get("score")
    try:
        quality_score = float(score_raw) if score_raw is not None else None
    except (TypeError, ValueError):
        quality_score = None
    if quality_score is not None and not math.isfinite(quality_score):
        quality_score = None
    return TargetSegmentation(
        visible=True,
        mask=mask,
        centroid_px=(float(xs.mean()), float(ys.mean())),
        bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)),
        area_px=int(mask.sum()),
        quality_score=quality_score,
        response=response,
    )


def resolve_object_relative_geometry(
    *,
    target_centroid_px: Sequence[float] | None,
    eef_position_xyz_m: Sequence[float] | None,
    calibration: CameraCalibration | None,
    move_vectors: Mapping[str, Sequence[float]],
    workspace_z_bounds_m: Sequence[float] | None,
    requested_displacement_m: float = REQUESTED_ALIGNMENT_M,
) -> dict[str, Any]:
    """Project six physical hypotheses and rank them by predicted image error."""
    invalid = {
        "camera_projection_valid": False,
        "object_relative_alignment_valid": False,
        "candidate_directions": [],
        "chosen_candidate": None,
        "reason": None,
    }
    if calibration is None:
        invalid["reason"] = "camera_calibration_invalid"
        return invalid
    try:
        target = np.asarray(target_centroid_px, dtype=float).reshape(2)
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
    current_pixel = np.asarray(current["pixel_xy"], dtype=float)
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
        after_error = float(np.linalg.norm(target - np.asarray(projected["pixel_xy"], dtype=float)))
        candidates.append({
            "direction": direction,
            "physical_vector_xyz": unit.tolist(),
            "direction_unit": unit.tolist(),
            "hypothetical_eef_xyz_m": hypothetical.tolist(),
            "hypothetical_projection_px": projected["pixel_xy"],
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
        "eef_projection_px": current["pixel_xy"],
        "pixel_error_before_px": before_error,
        "candidate_directions": candidates,
        "chosen_candidate": best if best["predicted_improvement_px"] > 0.0 else None,
        "reason": ("deterministic_object_relative_geometry"
                   if best["predicted_improvement_px"] > 0.0
                   else "no_candidate_predicted_to_improve_alignment"),
    }
    return result


def make_alignment_option(state: BeliefState) -> RuntimeOption | None:
    """Turn a pre-resolved Runtime geometry result into one sealed semantic option."""
    relative = state.object_relative_state
    geometry = state.relevant_geometry
    choice = geometry.get("chosen_candidate")
    if (relative is None or not relative.target_visible or relative.eef_projection_px is None
            or relative.target_centroid_px is None or not isinstance(choice, Mapping)
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
            "target_phrase": relative.target_phrase,
            "image_error_before_px": before,
            "predicted_image_error_after_px": predicted_after,
            "predicted_improvement_px": predicted_improvement,
            "verification": "resegment_and_reproject_target_relative_error",
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


class ObjectRelativePerceptionObserver:
    """Add SAM3 target evidence and camera-calibrated hypotheses to an observer."""

    def __init__(
        self,
        base_observer: Any,
        sam3: Any,
        *,
        target_phrase: str,
        move_vectors: Mapping[str, Sequence[float]],
        confidence_threshold: float = SAM3_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.base_observer = base_observer
        self.sam3 = sam3
        self.target_phrase = str(target_phrase)
        self.confidence_threshold = float(confidence_threshold)
        self.move_vectors = {str(key): tuple(float(v) for v in value)
                             for key, value in move_vectors.items()}
        self.last_segmentation: TargetSegmentation | None = None
        self.last_calibration: CameraCalibration | None = None
        self.last_resolution: dict[str, Any] = {}
        self.perception_history: list[dict[str, Any]] = []

    def observe_for_execution_tick(self, environment: Any) -> RobotObservation:
        """Collect EEF-only tick evidence; SAM3 runs once after the bounded motion."""
        return self.base_observer.observe(environment)

    def observe(self, environment: Any) -> RobotObservation:
        base = self.base_observer.observe(environment)
        image = np.asarray(base.images.get("agentview"))
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("agentview source must be an HxWx3 RGB array")
        height, width = image.shape[:2]
        response = self.sam3.segment(
            image, self.target_phrase, confidence_threshold=self.confidence_threshold
        )
        segmentation = segmentation_from_response(response, (height, width))
        details = response.get("details") if isinstance(response, Mapping) else None
        metadata = details.get("metadata") if isinstance(details, Mapping) else None
        reported_size = metadata.get("image_size") if isinstance(metadata, Mapping) else None
        if reported_size is not None and reported_size != [width, height]:
            segmentation = TargetSegmentation(
                False, None, None, None, None, None,
                {**dict(response), "resolution_mismatch": {
                    "expected_width_height": [width, height],
                    "reported_width_height": reported_size,
                }},
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
                # Existing LIBERO policy frames use this orientation transform;
                # preserve raw pixels while projecting into their image axes.
                rotations={"agentview": 180},
                flips={"agentview": "none"},
            )
            calibration = calib["agentview"]
        except (AttributeError, KeyError, TypeError, ValueError):
            calibration = None
        self.last_calibration = calibration
        workspace = base.evidence.get("relevant_geometry", {}).get("workspace_z_bounds_m")
        geometry = resolve_object_relative_geometry(
            target_centroid_px=segmentation.centroid_px if segmentation.visible else None,
            eef_position_xyz_m=position,
            calibration=calibration,
            move_vectors=self.move_vectors,
            workspace_z_bounds_m=workspace,
        )
        self.last_resolution = geometry
        eef_projection = geometry.get("eef_projection_px")
        image_error = None
        error_norm = None
        if segmentation.visible and eef_projection is not None and segmentation.centroid_px is not None:
            delta = np.asarray(segmentation.centroid_px) - np.asarray(eef_projection)
            image_error = (float(delta[0]), float(delta[1]))
            error_norm = float(np.linalg.norm(delta))
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
        )
        self.perception_history.append({
            "image": image.copy(),
            "segmentation": segmentation,
            "calibration": calibration,
            "resolution": dict(geometry),
            "object_relative_state": relative,
        })
        evidence = dict(base.evidence)
        evidence["target_identity"] = self.target_phrase
        evidence["object_relative_state"] = relative
        merged_geometry = dict(evidence.get("relevant_geometry", {}) or {})
        merged_geometry.update({
            "camera_projection_valid": bool(geometry.get("camera_projection_valid")),
            "object_relative_alignment_valid": bool(geometry.get("object_relative_alignment_valid"))
                                                    and segmentation.visible,
            "object_relative_decision_owner": "runtime",
            "object_relative_decision_reason": geometry.get("reason"),
            "pixel_error_before_px": error_norm,
            "candidate_directions": geometry.get("candidate_directions", []),
            "chosen_candidate": (geometry.get("chosen_candidate") if segmentation.visible else None),
            "sam3_error": response.get("error") if not segmentation.visible else None,
            "sam3_confidence_threshold": self.confidence_threshold,
        })
        if not segmentation.visible:
            merged_geometry["object_relative_alignment_valid"] = False
        evidence["relevant_geometry"] = merged_geometry
        return RobotObservation(
            observation_id=base.observation_id,
            frame_id=base.frame_id,
            images=base.images,
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
    ) -> dict[str, str]:
        """Write exact source RGB, decoded mask, and diagnostic overlay for review."""
        from pathlib import Path

        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        source = np.ascontiguousarray(image, dtype=np.uint8)
        rgb_path = output / f"{prefix}_rgb.png"
        Image.fromarray(source, mode="RGB").save(rgb_path)
        mask_path = None
        overlay_path = None
        segmentation = segmentation if segmentation is not None else self.last_segmentation
        if segmentation is not None and segmentation.mask is not None:
            mask_path = output / f"{prefix}_mask.png"
            Image.fromarray(segmentation.mask.astype(np.uint8) * 255, mode="L").save(mask_path)
            overlay = Image.fromarray(source.copy(), mode="RGB").convert("RGBA")
            tint = np.zeros((*segmentation.mask.shape, 4), dtype=np.uint8)
            tint[segmentation.mask] = (255, 24, 24, 96)
            overlay = Image.alpha_composite(overlay, Image.fromarray(tint, mode="RGBA"))
            draw = ImageDraw.Draw(overlay)
            if segmentation.centroid_px is not None:
                x, y = segmentation.centroid_px
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(255, 255, 0, 255), width=2)
            eef = (resolution if resolution is not None else self.last_resolution).get("eef_projection_px")
            if eef is not None:
                x, y = (float(eef[0]), float(eef[1]))
                draw.line((x - 7, y, x + 7, y), fill=(0, 255, 255, 255), width=2)
                draw.line((x, y - 7, x, y + 7), fill=(0, 255, 255, 255), width=2)
            overlay_path = output / f"{prefix}_overlay.png"
            overlay.convert("RGB").save(overlay_path)
        result = {"rgb": str(rgb_path)}
        if mask_path is not None:
            result["mask"] = str(mask_path)
        if overlay_path is not None:
            result["overlay"] = str(overlay_path)
        return result

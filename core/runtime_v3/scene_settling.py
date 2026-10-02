"""Pure, diagnostic-only calculations for Runtime V3 scene settling trials.

This module intentionally has no simulator or Runtime state dependencies. Oracle
poses are accepted only by explicit reporting functions and are never copied
into RobotObservation or BeliefState objects.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np

from core.capabilities.camera_geometry import CameraCalibration, project_point
from .canonical_image import CanonicalImageAdapter


SCENE_READY_WINDOW_OBSERVATIONS = 3
SCENE_READY_MAX_CENTROID_SHIFT_PX = 0.02
SCENE_READY_MAX_BBOX_EDGE_SHIFT_PX = 0.0
SCENE_READY_MAX_MASK_AREA_CHANGE_PX = 1
SCENE_READY_THRESHOLD_SOURCE = (
    "M3.1c stable HOLD samples at ticks 10/15/20 across init states 0/1/2: "
    "maximum paired centroid variation 0.0162061 px, bbox-edge variation 0 px, "
    "mask-area variation 1 px; centroid limit rounded up to 0.02 px"
)


class SceneReadyEvidence:
    """Small visual-only gate requiring a window of stable associated masks."""

    def __init__(
        self,
        *,
        window_observations: int = SCENE_READY_WINDOW_OBSERVATIONS,
        max_centroid_shift_px: float = SCENE_READY_MAX_CENTROID_SHIFT_PX,
        max_bbox_edge_shift_px: float = SCENE_READY_MAX_BBOX_EDGE_SHIFT_PX,
        max_mask_area_change_px: int = SCENE_READY_MAX_MASK_AREA_CHANGE_PX,
    ) -> None:
        if int(window_observations) < 2:
            raise ValueError("SceneReady requires at least two consecutive observations")
        self.window_observations = int(window_observations)
        self.max_centroid_shift_px = float(max_centroid_shift_px)
        self.max_bbox_edge_shift_px = float(max_bbox_edge_shift_px)
        self.max_mask_area_change_px = int(max_mask_area_change_px)
        self._history: deque[dict[str, Any]] = deque(maxlen=self.window_observations)
        self.last_interval: dict[str, Any] | None = None
        self.ready = False

    def reset(self) -> None:
        self._history.clear()
        self.last_interval = None
        self.ready = False

    def update(
        self,
        *,
        target_identity_status: str,
        centroid_px: Sequence[float] | None,
        bbox_xyxy: Sequence[float] | None,
        mask_area_px: int | None,
    ) -> bool:
        """Consume associated SAM geometry only; no oracle argument exists."""
        sample = scene_readiness_sample(
            target_identity_status=target_identity_status,
            centroid_px=centroid_px,
            bbox_xyxy=bbox_xyxy,
            mask_area_px=mask_area_px,
        )
        valid_status = sample["target_identity_status"] in {"ANCHORED", "PROVISIONAL", "SAME_TARGET"}
        valid_geometry = (sample["centroid_px"] is not None and sample["bbox_xyxy"] is not None
                          and sample["mask_area_px"] is not None and sample["mask_area_px"] > 0)
        if not valid_status or not valid_geometry:
            self.reset()
            return False
        self.last_interval = None
        if self._history:
            previous = self._history[-1]
            centroid_shift = float(np.linalg.norm(
                np.asarray(sample["centroid_px"], dtype=float)
                - np.asarray(previous["centroid_px"], dtype=float)
            ))
            bbox_edge_shift = max(abs(left - right) for left, right in zip(
                sample["bbox_xyxy"], previous["bbox_xyxy"]
            ))
            area_change = abs(int(sample["mask_area_px"]) - int(previous["mask_area_px"]))
            self.last_interval = {
                "centroid_shift_px": centroid_shift,
                "bbox_max_edge_shift_px": float(bbox_edge_shift),
                "mask_area_change_px": int(area_change),
            }
            stable = (centroid_shift <= self.max_centroid_shift_px
                      and bbox_edge_shift <= self.max_bbox_edge_shift_px
                      and area_change <= self.max_mask_area_change_px)
            if not stable:
                self._history.clear()
        self._history.append(sample)
        self.ready = len(self._history) >= self.window_observations
        return self.ready

    def to_record(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "stable_observation_count": len(self._history),
            "required_observations": self.window_observations,
            "max_centroid_shift_px": self.max_centroid_shift_px,
            "max_bbox_edge_shift_px": self.max_bbox_edge_shift_px,
            "max_mask_area_change_px": self.max_mask_area_change_px,
            "threshold_source": SCENE_READY_THRESHOLD_SOURCE,
            "last_visual_interval": self.last_interval,
            "evidence_source": "associated_sam_visual_observation",
            "oracle_used": False,
        }


def displacement(before_xyz_m: Sequence[float], after_xyz_m: Sequence[float]) -> dict[str, Any]:
    """Return raw xyz displacement and its norm for two world-frame samples."""
    before = np.asarray(before_xyz_m, dtype=float).reshape(-1)
    after = np.asarray(after_xyz_m, dtype=float).reshape(-1)
    if before.shape != (3,) or after.shape != (3,):
        raise ValueError("world positions must each contain three values")
    if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
        raise ValueError("world positions must be finite")
    delta = after - before
    return {"delta_xyz_m": delta.tolist(), "norm_m": float(np.linalg.norm(delta))}


def action_excess_motion(
    action_delta_xyz_m: Sequence[float], no_action_delta_xyz_m: Sequence[float],
) -> dict[str, Any]:
    """Subtract a tick-matched HOLD vector from the action displacement vector."""
    action = np.asarray(action_delta_xyz_m, dtype=float).reshape(-1)
    control = np.asarray(no_action_delta_xyz_m, dtype=float).reshape(-1)
    if action.shape != (3,) or control.shape != (3,):
        raise ValueError("action and NO_ACTION displacement must each contain three values")
    if not np.all(np.isfinite(action)) or not np.all(np.isfinite(control)):
        raise ValueError("action and NO_ACTION displacement must be finite")
    excess = action - control
    return {
        "delta_xyz_m": excess.tolist(),
        "norm_m": float(np.linalg.norm(excess)),
        "action_norm_m": float(np.linalg.norm(action)),
        "no_action_norm_m": float(np.linalg.norm(control)),
        "norm_difference_m": float(np.linalg.norm(action) - np.linalg.norm(control)),
    }


def require_matched_duration(action_ticks: int, no_action_ticks: int) -> None:
    """Reject a purported matched pair with unequal executed simulation ticks."""
    if int(action_ticks) < 0 or int(no_action_ticks) < 0:
        raise ValueError("tick durations cannot be negative")
    if int(action_ticks) != int(no_action_ticks):
        raise ValueError("ACTION and NO_ACTION durations must match exactly")


def validate_no_action_commands(commands: Sequence[Mapping[str, Any]]) -> None:
    """Require every command in a NO_ACTION trace to be a translation-free HOLD."""
    for index, command in enumerate(commands):
        if command.get("kind") != "HOLD" or int(command.get("translation_command_count", -1)) != 0:
            raise ValueError(f"NO_ACTION command {index} is not a zero-translation HOLD")


def target_motion_curve(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalize a tick-ordered per-sample settling time series."""
    ordered = sorted(samples, key=lambda item: int(item["tick"]))
    seen: set[int] = set()
    curve: list[dict[str, Any]] = []
    for sample in ordered:
        tick = int(sample["tick"])
        if tick in seen:
            raise ValueError(f"duplicate settling tick {tick}")
        seen.add(tick)
        target = np.asarray(sample["target_world_position_m"], dtype=float).reshape(-1)
        eef = np.asarray(sample["eef_world_position_m"], dtype=float).reshape(-1)
        if target.shape != (3,) or eef.shape != (3,):
            raise ValueError("settling target and EEF positions must contain three values")
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(eef)):
            raise ValueError("settling positions must be finite")
        target_delta = None if not curve else target - np.asarray(curve[-1]["target_world_position_m"])
        eef_delta = None if not curve else eef - np.asarray(curve[-1]["eef_world_position_m"])
        curve.append({
            "tick": tick,
            "environment_step": int(sample["environment_step"]),
            "simulation_time_s": float(sample["simulation_time_s"]),
            "target_world_position_m": target.tolist(),
            "eef_target_distance_m": sample.get("eef_target_distance_m"),
            "target_delta_from_previous_m": target_delta.tolist() if target_delta is not None else None,
            "target_cumulative_delta_from_tick0_m": (
                (target - np.asarray(ordered[0]["target_world_position_m"], dtype=float)).tolist()
            ),
            "eef_world_position_m": eef.tolist(),
            "eef_delta_from_previous_m": eef_delta.tolist() if eef_delta is not None else None,
            "eef_cumulative_delta_from_tick0_m": (
                (eef - np.asarray(ordered[0]["eef_world_position_m"], dtype=float)).tolist()
            ),
            "sam_centroid_px": sample.get("sam_centroid_px"),
            "sam_bbox_xyxy": sample.get("sam_bbox_xyxy"),
            "sam_mask_area_px": sample.get("sam_mask_area_px"),
            "image_path": sample.get("image_path"),
            "overlay_path": sample.get("overlay_path"),
        })
    return curve


def project_world_motion_to_canonical_pixels(
    before_xyz_m: Sequence[float],
    after_xyz_m: Sequence[float],
    calibration: CameraCalibration,
    image_adapter: CanonicalImageAdapter,
) -> dict[str, Any]:
    """Project both world body origins through one fixed camera and image transform."""
    before = project_point(calibration, np.asarray(before_xyz_m, dtype=float))
    after = project_point(calibration, np.asarray(after_xyz_m, dtype=float))
    if before is None or after is None:
        return {"available": False, "reason": "target_body_origin_outside_camera_projection"}
    width, height = int(calibration.width), int(calibration.height)
    before_px = image_adapter.transform_projected_point(before["pixel_xy"], width=width, height=height)
    after_px = image_adapter.transform_projected_point(after["pixel_xy"], width=width, height=height)
    shift = np.asarray(after_px, dtype=float) - np.asarray(before_px, dtype=float)
    return {
        "available": True,
        "camera": calibration.name,
        "image_size": [width, height],
        "canonical_orientation": image_adapter.orientation,
        "oracle_px_before": [float(before_px[0]), float(before_px[1])],
        "oracle_px_after": [float(after_px[0]), float(after_px[1])],
        "oracle_pixel_shift": shift.tolist(),
        "projection_camera_signature": {
            "position_world": np.asarray(calibration.position_world, dtype=float).tolist(),
            "camera_to_world": np.asarray(calibration.camera_to_world, dtype=float).tolist(),
            "fovy_deg": float(calibration.fovy_deg),
        },
        "diagnostic_only": True,
    }


def compare_oracle_and_sam_pixel_motion(
    oracle_pixel_shift: Sequence[float] | None,
    sam_centroid_before_px: Sequence[float] | None,
    sam_centroid_after_px: Sequence[float] | None,
) -> dict[str, Any]:
    """Report SAM minus projected-oracle motion, retaining both raw vectors."""
    if oracle_pixel_shift is None or sam_centroid_before_px is None or sam_centroid_after_px is None:
        return {"available": False, "reason": "missing_oracle_projection_or_associated_centroid"}
    oracle = np.asarray(oracle_pixel_shift, dtype=float).reshape(-1)
    before = np.asarray(sam_centroid_before_px, dtype=float).reshape(-1)
    after = np.asarray(sam_centroid_after_px, dtype=float).reshape(-1)
    if oracle.shape != (2,) or before.shape != (2,) or after.shape != (2,):
        raise ValueError("pixel vectors must contain two values")
    if not np.all(np.isfinite(oracle)) or not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
        raise ValueError("pixel vectors must be finite")
    sam_shift = after - before
    residual = sam_shift - oracle
    return {
        "available": True,
        "oracle_pixel_shift": oracle.tolist(),
        "sam_centroid_before_px": before.tolist(),
        "sam_centroid_after_px": after.tolist(),
        "sam_centroid_shift": sam_shift.tolist(),
        "residual_sam_minus_oracle_px": residual.tolist(),
        "residual_norm_px": float(np.linalg.norm(residual)),
        "sam_shift_norm_px": float(np.linalg.norm(sam_shift)),
        "oracle_shift_norm_px": float(np.linalg.norm(oracle)),
    }


def scene_readiness_sample(
    *, target_identity_status: str, centroid_px: Sequence[float] | None,
    bbox_xyxy: Sequence[float] | None, mask_area_px: int | None,
) -> dict[str, Any]:
    """Build a visual-only SceneReady sample; intentionally accepts no oracle input."""
    centroid = None if centroid_px is None else [float(v) for v in centroid_px]
    bbox = None if bbox_xyxy is None else [float(v) for v in bbox_xyxy]
    if centroid is not None and len(centroid) != 2:
        raise ValueError("centroid_px must contain two values")
    if bbox is not None and len(bbox) != 4:
        raise ValueError("bbox_xyxy must contain four values")
    return {
        "target_identity_status": str(target_identity_status),
        "centroid_px": centroid,
        "bbox_xyxy": bbox,
        "mask_area_px": None if mask_area_px is None else int(mask_area_px),
        "evidence_source": "associated_sam_visual_observation",
    }

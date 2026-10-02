"""Experiment-only diagnostics for MuJoCo depth renders.

This module is deliberately outside ``core.runtime_v3``. Its outputs describe
simulator depth for research and must never enter Runtime state or decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np

from core.capabilities.camera_geometry import (
    CameraCalibration,
    opencv_camera_points_to_world,
    project_point,
)
from core.runtime_v3.canonical_image import CanonicalImageAdapter


class DiagnosticSimulatorDepthProvider:
    """Explicit upper-bound source; this provider is forbidden in Runtime core."""

    diagnostic_only = True
    source = "simulator_gt"

    def estimate_normalized_buffer(self, raw_observation: Any, camera: str) -> np.ndarray | None:
        key = ("agentview_depth" if camera == "agentview"
               else "robot0_eye_in_hand_depth" if camera == "wrist" else None)
        if key is None:
            raise ValueError("camera must be agentview or wrist")
        value = raw_observation.get(key) if isinstance(raw_observation, Mapping) else None
        return None if value is None else np.asarray(value, dtype=np.float32)


@dataclass(frozen=True)
class MetricEntityReference:
    """Diagnostic visible-surface point estimated from simulator depth."""

    entity_key: str
    camera: str
    frame: str
    reference_world_m: tuple[float, float, float] | None
    valid_depth_count: int
    mask_pixel_count: int
    valid_depth_ratio: float
    depth_median_m: float | None
    depth_spread_m: float | None
    source_frame_id: str
    valid: bool
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        key = str(self.entity_key).strip()
        camera = str(self.camera).strip()
        frame = str(self.frame).strip()
        if not key or not camera or not frame:
            raise ValueError("entity_key, camera, and frame must be non-empty")
        count = int(self.valid_depth_count)
        area = int(self.mask_pixel_count)
        ratio = float(self.valid_depth_ratio)
        if count < 0 or area < 0 or count > area:
            raise ValueError("depth sample counts must satisfy 0 <= valid <= mask pixels")
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError("valid_depth_ratio must be finite and in [0, 1]")
        point = self.reference_world_m
        if point is not None:
            point = tuple(float(value) for value in point)
            if len(point) != 3 or not all(math.isfinite(value) for value in point):
                raise ValueError("reference_world_m must contain three finite coordinates")
        if self.valid and (point is None or count == 0 or area == 0):
            raise ValueError("a valid reference requires a point and valid masked depth samples")
        if not self.valid and not self.invalid_reason:
            raise ValueError("invalid references require an invalid_reason")
        for name in ("depth_median_m", "depth_spread_m"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        object.__setattr__(self, "entity_key", key)
        object.__setattr__(self, "camera", camera)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "valid_depth_count", count)
        object.__setattr__(self, "mask_pixel_count", area)
        object.__setattr__(self, "valid_depth_ratio", ratio)
        object.__setattr__(self, "source_frame_id", str(self.source_frame_id))
        if point is not None:
            object.__setattr__(self, "reference_world_m", point)

    def invalidate(self, reason: str) -> "MetricEntityReference":
        return replace(self, valid=False, invalid_reason=str(reason))


def mujoco_depth_clip_planes_m(model: Any) -> tuple[float, float]:
    """Read the same extent-scaled near/far planes used by robosuite."""
    extent = float(model.stat.extent)
    near = float(model.vis.map.znear) * extent
    far = float(model.vis.map.zfar) * extent
    if not all(math.isfinite(value) for value in (near, far)) or near <= 0.0 or far <= near:
        raise ValueError("MuJoCo camera depth clip planes must be finite with 0 < near < far")
    return near, far


def normalized_depth_to_metric(
    depth_buffer: np.ndarray,
    *,
    near_m: float,
    far_m: float,
) -> np.ndarray:
    """Convert MuJoCo's normalized z-buffer with robosuite's standard formula.

    Invalid transport values (non-finite, zero, negative, or the far-plane clear
    value 1) become NaN so later mask filtering cannot mistake them for surfaces.
    """
    near, far = float(near_m), float(far_m)
    if not math.isfinite(near) or not math.isfinite(far) or near <= 0.0 or far <= near:
        raise ValueError("near_m and far_m must satisfy 0 < near_m < far_m")
    raw = np.asarray(depth_buffer, dtype=np.float64)
    valid = np.isfinite(raw) & (raw > 0.0) & (raw < 1.0)
    metric = np.full(raw.shape, np.nan, dtype=np.float32)
    denominator = 1.0 - raw[valid] * (1.0 - near / far)
    metric[valid] = (near / denominator).astype(np.float32)
    metric[~np.isfinite(metric) | (metric <= 0.0)] = np.nan
    return metric


def invalid_metric_entity_reference(
    *, entity_key: str, camera: str, source_frame_id: Any,
    invalid_reason: str, mask_pixel_count: int = 0,
) -> MetricEntityReference:
    return MetricEntityReference(
        entity_key=entity_key,
        camera=camera,
        frame="world",
        reference_world_m=None,
        valid_depth_count=0,
        mask_pixel_count=max(0, int(mask_pixel_count)),
        valid_depth_ratio=0.0,
        depth_median_m=None,
        depth_spread_m=None,
        source_frame_id=str(source_frame_id),
        valid=False,
        invalid_reason=str(invalid_reason),
    )


def metric_entity_reference_from_rgbd(
    *,
    entity_key: str,
    camera: str,
    source_frame_id: Any,
    target_mask_canonical: np.ndarray | None,
    depth_buffer_raw: np.ndarray | None,
    near_m: float,
    far_m: float,
    calibration: CameraCalibration | None,
    image_adapter: CanonicalImageAdapter,
) -> tuple[MetricEntityReference, np.ndarray | None]:
    """Diagnostic back-projection of a SAM mask using simulator depth.

    The returned depth raster is canonical metric depth for diagnostic overlays.
    No resizing or simulator entity state is consulted.
    """
    if target_mask_canonical is None:
        return (invalid_metric_entity_reference(
            entity_key=entity_key, camera=camera, source_frame_id=source_frame_id,
            invalid_reason="target_mask_missing"), None)
    mask = np.asarray(target_mask_canonical, dtype=bool)
    if mask.ndim != 2:
        return (invalid_metric_entity_reference(
            entity_key=entity_key, camera=camera, source_frame_id=source_frame_id,
            invalid_reason="target_mask_not_2d"), None)
    mask_count = int(mask.sum())
    if mask_count == 0:
        return (invalid_metric_entity_reference(
            entity_key=entity_key, camera=camera, source_frame_id=source_frame_id,
            invalid_reason="target_mask_empty"), None)
    if depth_buffer_raw is None:
        return (invalid_metric_entity_reference(
            entity_key=entity_key, camera=camera, source_frame_id=source_frame_id,
            invalid_reason="depth_missing", mask_pixel_count=mask_count), None)
    source_buffer = np.asarray(depth_buffer_raw)
    if source_buffer.ndim == 3 and source_buffer.shape[-1] == 1:
        source_buffer = source_buffer[..., 0]
    if source_buffer.ndim != 2:
        return (invalid_metric_entity_reference(
            entity_key=entity_key, camera=camera, source_frame_id=source_frame_id,
            invalid_reason="depth_not_single_channel_2d", mask_pixel_count=mask_count), None)
    try:
        canonical_buffer = image_adapter.transform_image(source_buffer)
    except (TypeError, ValueError):
        canonical_buffer = source_buffer
    if canonical_buffer.ndim != 2 or canonical_buffer.shape != mask.shape:
        return (invalid_metric_entity_reference(
            entity_key=entity_key, camera=camera, source_frame_id=source_frame_id,
            invalid_reason="mask_depth_resolution_mismatch", mask_pixel_count=mask_count), None)
    try:
        metric_depth = normalized_depth_to_metric(canonical_buffer, near_m=near_m, far_m=far_m)
    except (TypeError, ValueError, OverflowError):
        return (invalid_metric_entity_reference(
            entity_key=entity_key, camera=camera, source_frame_id=source_frame_id,
            invalid_reason="depth_conversion_failed", mask_pixel_count=mask_count), None)
    if calibration is None:
        return (invalid_metric_entity_reference(
            entity_key=entity_key, camera=camera, source_frame_id=source_frame_id,
            invalid_reason="camera_calibration_missing", mask_pixel_count=mask_count), metric_depth)
    if metric_depth.shape != (int(calibration.height), int(calibration.width)):
        return (invalid_metric_entity_reference(
            entity_key=entity_key, camera=camera, source_frame_id=source_frame_id,
            invalid_reason="depth_calibration_resolution_mismatch", mask_pixel_count=mask_count), metric_depth)
    valid = mask & np.isfinite(metric_depth) & (metric_depth > 0.0)
    ys, xs = np.nonzero(valid)
    count = int(xs.size)
    ratio = float(count / mask_count)
    if count == 0:
        return (MetricEntityReference(
            entity_key=entity_key, camera=camera, frame="world", reference_world_m=None,
            valid_depth_count=0, mask_pixel_count=mask_count, valid_depth_ratio=ratio,
            depth_median_m=None, depth_spread_m=None, source_frame_id=str(source_frame_id),
            valid=False, invalid_reason="no_valid_mask_depth_pixels",
        ), metric_depth)

    z = metric_depth[ys, xs].astype(np.float64)
    height, width = metric_depth.shape
    focal = (height / 2.0) / math.tan(math.radians(float(calibration.fovy_deg)) / 2.0)
    x_camera = (xs.astype(np.float64) - width / 2.0) * z / focal
    y_camera = (ys.astype(np.float64) - height / 2.0) * z / focal
    points_camera = np.column_stack((x_camera, y_camera, z))
    points_world = opencv_camera_points_to_world(
        points_camera,
        camera_to_world=calibration.camera_to_world,
        position_world=calibration.position_world,
    )
    reference = np.median(points_world, axis=0)
    depth_median = float(np.median(z))
    depth_mad = float(np.median(np.abs(z - depth_median)))
    spread = 1.4826 * depth_mad
    result = MetricEntityReference(
        entity_key=entity_key,
        camera=camera,
        frame="world",
        reference_world_m=tuple(float(value) for value in reference),
        valid_depth_count=count,
        mask_pixel_count=mask_count,
        valid_depth_ratio=ratio,
        depth_median_m=depth_median,
        depth_spread_m=spread,
        source_frame_id=str(source_frame_id),
        valid=True,
    )
    return result, metric_depth


def freeze_metric_reference(
    current: MetricEntityReference | None,
    candidate: MetricEntityReference,
) -> MetricEntityReference | None:
    """Keep the first valid research reference; this is not Runtime state."""
    if current is not None:
        return current
    return candidate if candidate.valid else None


def metric_proximity_distance_m(
    eef_position_world_m: Sequence[float] | None,
    reference: MetricEntityReference | None,
) -> float | None:
    """Research-only distance from EEF proprioception to a depth-derived point."""
    frame = getattr(reference, "coordinate_frame", getattr(reference, "frame", None))
    if reference is None or not reference.valid or frame != "world":
        return None
    if eef_position_world_m is None or reference.reference_world_m is None:
        return None
    try:
        eef = np.asarray(eef_position_world_m, dtype=np.float64).reshape(3)
        target = np.asarray(reference.reference_world_m, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(eef)) or not np.all(np.isfinite(target)):
        return None
    return float(np.linalg.norm(eef - target))


def classify_metric_reference_visibility(
    reference: MetricEntityReference | None,
    calibration: CameraCalibration | None,
    *,
    image_adapter: CanonicalImageAdapter,
    sam_detected: bool,
) -> dict[str, Any]:
    """Classify projected metric-reference coverage independently of target pose."""
    if reference is None or not reference.valid or reference.reference_world_m is None:
        return {"status": "UNRESOLVED", "projected_pixel_px": None,
                "in_front": None, "in_frame": None, "reason": "metric_reference_invalid"}
    if calibration is None:
        return {"status": "UNRESOLVED", "projected_pixel_px": None,
                "in_front": None, "in_frame": None, "reason": "camera_calibration_missing"}
    projected = project_point(calibration, np.asarray(reference.reference_world_m, dtype=float))
    if projected is None:
        return {"status": "OUTSIDE_FRUSTUM", "projected_pixel_px": None,
                "in_front": False, "in_frame": False, "reason": "point_not_in_front"}
    pixel = image_adapter.transform_projected_point(
        projected["pixel_xy"], width=calibration.width, height=calibration.height
    )
    in_frame = bool(0.0 <= pixel[0] < calibration.width and 0.0 <= pixel[1] < calibration.height)
    if not in_frame:
        status = "OUTSIDE_FRUSTUM"
    elif sam_detected:
        status = "VISIBLE_DETECTED"
    else:
        status = "INSIDE_FRUSTUM_SAM_MISS"
    return {"status": status, "projected_pixel_px": [float(pixel[0]), float(pixel[1])],
            "in_front": True, "in_frame": in_frame, "reason": None,
            "camera_point": projected["camera_point"]}

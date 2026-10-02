"""Deployable metric geometry for semantically grounded visual entities.

This formal Runtime representation accepts metric monocular estimates or a
future real RGB-D sensor provider. Simulator depth is rejected by its type and
source validation and is implemented separately under experiment scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class MetricEntityReference:
    entity_key: str
    camera: str
    coordinate_frame: str
    reference_world_m: tuple[float, float, float] | None
    valid_depth_count: int
    mask_pixel_count: int
    valid_depth_ratio: float
    depth_median_m: float | None
    depth_spread_m: float | None
    depth_source: str
    source_frame_id: str
    valid: bool
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        key, camera, frame, source = (
            str(self.entity_key).strip(), str(self.camera).strip(),
            str(self.coordinate_frame).strip(), str(self.depth_source).strip(),
        )
        if not key or not camera or not frame:
            raise ValueError("entity_key, camera, and coordinate_frame must be non-empty")
        if source not in {"monocular_metric", "rgbd_sensor"}:
            raise ValueError("formal metric references cannot use privileged simulator depth")
        count, area = int(self.valid_depth_count), int(self.mask_pixel_count)
        ratio = float(self.valid_depth_ratio)
        if count < 0 or area < 0 or count > area:
            raise ValueError("depth counts must satisfy 0 <= valid_depth_count <= mask_pixel_count")
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError("valid_depth_ratio must be finite and in [0, 1]")
        point = self.reference_world_m
        if point is not None:
            point = tuple(float(value) for value in point)
            if len(point) != 3 or not all(math.isfinite(value) for value in point):
                raise ValueError("reference_world_m must contain three finite coordinates")
        if self.valid and (point is None or count == 0 or area == 0):
            raise ValueError("valid reference requires a point and valid masked depth samples")
        if not self.valid and not self.invalid_reason:
            raise ValueError("invalid reference requires an invalid_reason")
        for name in ("depth_median_m", "depth_spread_m"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        object.__setattr__(self, "entity_key", key)
        object.__setattr__(self, "camera", camera)
        object.__setattr__(self, "coordinate_frame", frame)
        object.__setattr__(self, "depth_source", source)
        object.__setattr__(self, "source_frame_id", str(self.source_frame_id))
        object.__setattr__(self, "valid_depth_count", count)
        object.__setattr__(self, "mask_pixel_count", area)
        object.__setattr__(self, "valid_depth_ratio", ratio)
        if point is not None:
            object.__setattr__(self, "reference_world_m", point)

    def invalidate(self, reason: str) -> "MetricEntityReference":
        return replace(self, valid=False, invalid_reason=str(reason))


def freeze_metric_reference(current: MetricEntityReference | None,
                            candidate: MetricEntityReference) -> MetricEntityReference | None:
    """Retain the first valid reference within one explicitly managed epoch."""
    if current is not None:
        return current
    return candidate if candidate.valid else None


def metric_proximity_distance_m(eef_position_world_m: Sequence[float] | None,
                                reference: MetricEntityReference | None) -> float | None:
    """Measure EEF-to-visible-reference distance from proprioception and perception."""
    if reference is None or not reference.valid or reference.coordinate_frame != "world":
        return None
    if eef_position_world_m is None or reference.reference_world_m is None:
        return None
    try:
        eef = np.asarray(eef_position_world_m, dtype=np.float64).reshape(3)
        target = np.asarray(reference.reference_world_m, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(eef).all() or not np.isfinite(target).all():
        return None
    return float(np.linalg.norm(eef - target))

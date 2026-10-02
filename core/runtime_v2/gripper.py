"""Embodiment-relative gripper envelope checks."""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .types import GripperEnvelope


def envelope_from_config(config: dict[str, Any] | None, *, name: str = "configured") -> GripperEnvelope:
    cfg = config if isinstance(config, dict) else {}
    extents = cfg.get("swept_volume_extents_m", cfg.get("extents", (0.0, 0.0, 0.0)))
    try:
        extents = tuple(float(x) for x in extents[:3])
    except (TypeError, ValueError, IndexError):
        extents = (0.0, 0.0, 0.0)
    if len(extents) != 3:
        extents = (0.0, 0.0, 0.0)
    return GripperEnvelope(
        name=str(cfg.get("name", name)),
        finger_clearance_m=float(cfg.get("finger_clearance_m", cfg.get("open_joint_m", 0.0))),
        fingertip_depth_m=float(cfg.get("fingertip_depth_m", 0.0)),
        closing_axis=tuple(float(x) for x in cfg.get("closing_axis", (0.0, 1.0, 0.0))),
        approach_axis=tuple(float(x) for x in cfg.get("approach_axis", (0.0, 0.0, 1.0))),
        swept_volume_extents_m=extents,
    )


def classify_points_in_envelope(
    points_gripper: Sequence[Sequence[float]], envelope: GripperEnvelope, *, uncertainty_m: float = 0.0
) -> dict[str, Any]:
    try:
        points = np.asarray(points_gripper, dtype=float).reshape(-1, 3)
    except (TypeError, ValueError):
        return {"relation": "UNKNOWN", "inside_fraction": 0.0, "valid": False}
    if len(points) == 0 or not np.all(np.isfinite(points)):
        return {"relation": "UNKNOWN", "inside_fraction": 0.0, "valid": False}
    ex, ey, ez = (abs(float(x)) for x in envelope.swept_volume_extents_m)
    margin = max(0.0, float(uncertainty_m))
    inside = (
        (np.abs(points[:, 0]) <= ex + margin)
        & (np.abs(points[:, 1]) <= ey + margin)
        & (np.abs(points[:, 2]) <= ez + margin)
    )
    fraction = float(np.mean(inside))
    centroid = np.mean(points, axis=0)
    if fraction >= 0.5:
        relation = "INSIDE_ENVELOPE"
    elif abs(float(centroid[2])) > ez + margin:
        relation = "FRONT" if centroid[2] > 0 else "BACK"
    elif abs(float(centroid[1])) > ey + margin:
        relation = "LEFT" if centroid[1] > 0 else "RIGHT"
    else:
        relation = "UNKNOWN"
    return {
        "relation": relation,
        "inside_fraction": fraction,
        "centroid_gripper": [float(v) for v in centroid],
        "valid": True,
    }

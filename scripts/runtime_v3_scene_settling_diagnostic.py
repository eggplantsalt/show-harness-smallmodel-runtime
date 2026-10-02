"""Privileged simulator comparisons for the Runtime V3 SceneReady study.

Nothing in this module is imported by Runtime V3 core. These helpers consume
simulator target positions only to write diagnostic artifacts.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from core.capabilities.camera_geometry import CameraCalibration, project_point
from core.runtime_v3.canonical_image import CanonicalImageAdapter


def target_motion_curve(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
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
    before_xyz_m: Sequence[float], after_xyz_m: Sequence[float],
    calibration: CameraCalibration, image_adapter: CanonicalImageAdapter,
) -> dict[str, Any]:
    """Project simulator target body origins for an offline comparison only."""
    before = project_point(calibration, np.asarray(before_xyz_m, dtype=float))
    after = project_point(calibration, np.asarray(after_xyz_m, dtype=float))
    if before is None or after is None:
        return {"available": False, "reason": "target_body_origin_outside_camera_projection"}
    width, height = int(calibration.width), int(calibration.height)
    before_px = image_adapter.transform_projected_point(before["pixel_xy"], width=width, height=height)
    after_px = image_adapter.transform_projected_point(after["pixel_xy"], width=width, height=height)
    shift = np.asarray(after_px, dtype=float) - np.asarray(before_px, dtype=float)
    return {
        "available": True, "camera": calibration.name, "image_size": [width, height],
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
    """Compare a privileged projection with deployable SAM motion offline."""
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
        "available": True, "oracle_pixel_shift": oracle.tolist(),
        "sam_centroid_before_px": before.tolist(),
        "sam_centroid_after_px": after.tolist(), "sam_centroid_shift": sam_shift.tolist(),
        "residual_sam_minus_oracle_px": residual.tolist(),
        "residual_norm_px": float(np.linalg.norm(residual)),
        "sam_shift_norm_px": float(np.linalg.norm(sam_shift)),
        "oracle_shift_norm_px": float(np.linalg.norm(oracle)),
        "diagnostic_only": True,
    }

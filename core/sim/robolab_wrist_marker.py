"""Optional wrist reference derived solely from robot and camera calibration.

The default geometry describes RoboLab's Show-Harness short-finger Panda, not
the stock Panda or DROID gripper.  The marker is the projection of a fixed 3-D
grasp point.  It is useful near contact; parallax means a distant object with
the same world XY need not overlap it.  It is not an object detector or proof
of a successful grasp.  This module never accesses an environment or objects.

The pixel transform below follows core.record.images.prepare_view and the
OpenCV pixel-centre convention used by resize_with_pad.  Draw only after the
view is prepared so the small cross remains readable at the final resolution.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"wrist_marker.{name} must contain {size} finite numbers")
    return result


def _transform_pixel(
    pixel: tuple[float, float],
    raw_shape: Sequence[int],
    rotation_degrees: int,
    flip: str,
    crop_aspect: float | None,
    square_size: int | None,
) -> tuple[tuple[float, float], tuple[int, int], bool]:
    """Project a raw pixel through the exact rotate, flip, crop, pad sequence."""
    h, w = int(raw_shape[0]), int(raw_shape[1])
    if h <= 0 or w <= 0:
        raise ValueError("wrist_marker raw image dimensions must be positive")
    x, y = pixel
    visible = 0 <= x <= w - 1 and 0 <= y <= h - 1
    if int(rotation_degrees) % 90:
        raise ValueError("wrist_marker requires rotation in multiples of 90 degrees")
    for _ in range((int(rotation_degrees) % 360) // 90):
        x, y = y, w - 1 - x
        h, w = w, h

    mode = str(flip or "none").lower()
    if mode not in {"none", "vertical", "horizontal", "both"}:
        raise ValueError(f"Unknown wrist_marker image flip: {mode}")
    if mode in {"vertical", "both"}:
        y = h - 1 - y
    if mode in {"horizontal", "both"}:
        x = w - 1 - x

    if crop_aspect and float(crop_aspect) > 0:
        aspect = float(crop_aspect)
        if not math.isfinite(aspect):
            raise ValueError("wrist_marker crop aspect must be finite")
        if abs(w / h - aspect) >= 1e-6:
            if w / h > aspect:
                new_w = int(round(h * aspect))
                x -= (w - new_w) // 2
                w = new_w
            else:
                new_h = int(round(w / aspect))
                y -= (h - new_h) // 2
                h = new_h
    if h <= 0 or w <= 0:
        raise ValueError("wrist_marker crop produces an empty image")
    visible = visible and 0 <= x <= w - 1 and 0 <= y <= h - 1

    if square_size:
        size = int(square_size)
        if size <= 0:
            raise ValueError("wrist_marker square size must be positive")
        scale = min(size / w, size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w <= 0 or new_h <= 0:
            raise ValueError("wrist_marker letterbox produces an empty image")
        x = (x + 0.5) * new_w / w - 0.5 + (size - new_w) // 2
        y = (y + 0.5) * new_h / h - 0.5 + (size - new_h) // 2
        h = w = size
    return (x, y), (h, w), visible


def annotate_wrist_grasp_point(
    prepared_image: np.ndarray,
    *,
    marker_cfg: Mapping[str, Any] | None = None,
    raw_shape: Sequence[int],
    rotation_degrees: int = 0,
    flip: str = "none",
    crop_aspect: float | None = None,
    square_size: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a copied image with a reference cross and auditable calibration.

    Set marker_cfg.enabled to true to opt in.  Camera position and quaternion
    describe its ROS frame in panda_hand coordinates (+Z forward, +X right,
    +Y down); grasp_point_m is also expressed in panda_hand coordinates.
    A disabled marker returns the input unchanged.  Out-of-frame projections
    are recorded and not drawn, rather than being clamped to a misleading edge.
    The returned metadata deliberately says runtime_validated=False: this
    implementation has not been validated by running a new simulation.
    """
    cfg = dict(marker_cfg or {})
    enabled = cfg.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("wrist_marker.enabled must be a boolean")
    metadata: dict[str, Any] = {
        "enabled": enabled,
        "drawn": False,
        "source": "robot_camera_geometry",
        "runtime_validated": False,
        "no_object_state": True,
        "purpose": "fixed grasp-point reference near contact; not object detection",
    }
    if not enabled:
        return prepared_image, metadata

    image = np.asarray(prepared_image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("wrist_marker expects a prepared uint8 RGB image")
    if len(raw_shape) < 2 or int(raw_shape[0]) <= 0 or int(raw_shape[1]) <= 0:
        raise ValueError("wrist_marker requires the original positive image height/width")
    position = _vector(cfg.get("camera_position_m", (0.035, 0.0, 0.036)), 3, "camera_position_m")
    grasp = _vector(cfg.get("grasp_point_m", (0.0, 0.0, 0.130)), 3, "grasp_point_m")
    quat = _vector(cfg.get("camera_quaternion_wxyz", (1.0, 0.0, 0.0, 0.0)), 4, "camera_quaternion_wxyz")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        raise ValueError("wrist_marker camera quaternion must be nonzero")
    qw, qx, qy, qz = quat / norm
    rotation = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    camera_point = rotation.T @ (grasp - position)
    hfov = float(cfg.get("horizontal_fov_degrees", 90.0))
    vfov = float(cfg.get("vertical_fov_degrees", 90.0))
    if not (0 < hfov < 180 and 0 < vfov < 180):
        raise ValueError("wrist_marker fields of view must be between 0 and 180 degrees")
    metadata["calibration"] = {
        "embodiment": str(cfg.get("embodiment", "franka_short_finger")),
        "camera_position_m": position.tolist(),
        "camera_quaternion_wxyz": (quat / norm).tolist(),
        "grasp_point_m": grasp.tolist(),
        "horizontal_fov_degrees": hfov,
        "vertical_fov_degrees": vfov,
        "raw_shape": [int(raw_shape[0]), int(raw_shape[1])],
        "rotation_degrees": int(rotation_degrees),
        "flip": str(flip),
        "crop_aspect": crop_aspect,
        "square_size": square_size,
    }
    if camera_point[2] <= 1e-9:
        metadata["reason"] = "grasp point is behind the camera"
        return prepared_image, metadata

    height, width = int(raw_shape[0]), int(raw_shape[1])
    fx = width / (2 * math.tan(math.radians(hfov) / 2))
    fy = height / (2 * math.tan(math.radians(vfov) / 2))
    raw_pixel = (
        float(width / 2 + fx * camera_point[0] / camera_point[2]),
        float(height / 2 + fy * camera_point[1] / camera_point[2]),
    )
    pixel, expected_shape, visible = _transform_pixel(
        raw_pixel, raw_shape, rotation_degrees, flip, crop_aspect, square_size
    )
    if tuple(image.shape[:2]) != expected_shape:
        raise ValueError(
            f"wrist_marker transform predicts {expected_shape}, got {tuple(image.shape[:2])}; "
            "pass the same transforms used by prepare_view"
        )
    metadata.update(raw_pixel=list(raw_pixel), prepared_pixel=list(pixel))
    if not visible:
        metadata["reason"] = "grasp point is outside the visible image crop"
        return prepared_image, metadata

    radius = int(cfg.get("radius_px", 5))
    if not 1 <= radius <= 20:
        raise ValueError("wrist_marker.radius_px must be between 1 and 20")
    color = _vector(cfg.get("color", (0, 255, 255)), 3, "color")
    if np.any(color < 0) or np.any(color > 255):
        raise ValueError("wrist_marker.color values must be between 0 and 255")
    rgb = tuple(int(value) for value in color)
    cx, cy = (int(round(value)) for value in pixel)
    output = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(output)
    for endpoints in ((cx - radius, cy, cx + radius, cy), (cx, cy - radius, cx, cy + radius)):
        draw.line(endpoints, fill=(0, 0, 0), width=3)
        draw.line(endpoints, fill=rgb, width=1)
    metadata.update(drawn=True, radius_px=radius, color=list(rgb))
    return np.array(output), metadata

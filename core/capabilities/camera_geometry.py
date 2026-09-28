"""Camera-calibrated image geometry used as host evidence.

This module projects only robot proprioception through camera calibration.  It
never reads an object pose or a task success predicate, so a projected end
effector point remains a measurement of the robot state rather than an oracle
target location.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    name: str
    width: int
    height: int
    fovy_deg: float
    position_world: np.ndarray
    camera_to_world: np.ndarray
    rotation_degrees: int = 0
    flip: str = "none"


def _transform_pixel(
    x: float,
    y: float,
    *,
    width: int,
    height: int,
    rotation_degrees: int,
    flip: str,
) -> tuple[float, float, int, int]:
    """Apply the same discrete orientation transform as ``prepare_view``."""
    k = (int(rotation_degrees) % 360) // 90
    if k == 1:
        x, y = y, width - 1 - x
        width, height = height, width
    elif k == 2:
        x, y = width - 1 - x, height - 1 - y
    elif k == 3:
        x, y = height - 1 - y, x
        width, height = height, width

    mode = str(flip or "none").lower()
    if mode in {"vertical", "both"}:
        y = height - 1 - y
    if mode in {"horizontal", "both"}:
        x = width - 1 - x
    return float(x), float(y), int(width), int(height)


def project_point(
    calibration: CameraCalibration,
    point_world: np.ndarray,
) -> Optional[dict[str, Any]]:
    """Project one world-space robot point into the policy image."""
    point = np.asarray(point_world, dtype=float).reshape(3)
    position = np.asarray(calibration.position_world, dtype=float).reshape(3)
    rotation = np.asarray(calibration.camera_to_world, dtype=float).reshape(3, 3)
    # MuJoCo cam_xmat maps camera-local vectors into world coordinates.
    point_camera = rotation.T @ (point - position)
    if not np.all(np.isfinite(point_camera)) or float(point_camera[2]) >= -1e-6:
        return None

    height = int(calibration.height)
    width = int(calibration.width)
    focal = (height / 2.0) / np.tan(np.deg2rad(float(calibration.fovy_deg)) / 2.0)
    raw_x = width / 2.0 + focal * float(point_camera[0]) / -float(point_camera[2])
    # MuJoCo's camera y basis points upward; image row coordinates point down.
    raw_y = height / 2.0 + focal * float(point_camera[1]) / -float(point_camera[2])
    x, y, output_width, output_height = _transform_pixel(
        raw_x,
        raw_y,
        width=width,
        height=height,
        rotation_degrees=calibration.rotation_degrees,
        flip=calibration.flip,
    )
    return {
        "pixel_xy": [round(x, 2), round(y, 2)],
        "raw_pixel_xy": [round(raw_x, 2), round(raw_y, 2)],
        "camera_point": [round(float(v), 5) for v in point_camera],
        "image_size": [output_width, output_height],
        "in_frame": bool(0.0 <= x < output_width and 0.0 <= y < output_height),
        "source": "proprioception_camera_calibration",
    }


def backproject_pixel_to_plane(
    calibration: CameraCalibration,
    pixel_xy: tuple[float, float] | list[float],
    plane_z: float,
) -> Optional[np.ndarray]:
    """Back-project an image pixel onto a horizontal world-Z plane.

    This is camera geometry only.  It does not inspect an object pose or a task
    predicate; callers supply the visual pixel and the calibrated reference
    height they want to use (for example, an approximate payload or receptacle
    reference height supplied by the caller).
    """
    try:
        x, y = float(pixel_xy[0]), float(pixel_xy[1])
    except (TypeError, ValueError, IndexError):
        return None

    width = int(calibration.width)
    height = int(calibration.height)
    output_width = width if int(calibration.rotation_degrees) % 180 == 0 else height
    output_height = height if int(calibration.rotation_degrees) % 180 == 0 else width

    # Undo the optional post-rotation flip first.
    mode = str(calibration.flip or "none").lower()
    if mode in {"vertical", "both"}:
        y = output_height - 1 - y
    if mode in {"horizontal", "both"}:
        x = output_width - 1 - x

    k = (int(calibration.rotation_degrees) % 360) // 90
    if k == 1:
        raw_x, raw_y = width - 1 - y, x
    elif k == 2:
        raw_x, raw_y = width - 1 - x, height - 1 - y
    elif k == 3:
        raw_x, raw_y = y, height - 1 - x
    else:
        raw_x, raw_y = x, y

    focal = (height / 2.0) / np.tan(np.deg2rad(float(calibration.fovy_deg)) / 2.0)
    point_camera = np.array(
        [(raw_x - width / 2.0) / focal, (raw_y - height / 2.0) / focal, -1.0],
        dtype=float,
    )
    position = np.asarray(calibration.position_world, dtype=float).reshape(3)
    rotation = np.asarray(calibration.camera_to_world, dtype=float).reshape(3, 3)
    ray_world = rotation @ point_camera
    if not np.all(np.isfinite(ray_world)) or abs(float(ray_world[2])) < 1e-9:
        return None
    scale = (float(plane_z) - float(position[2])) / float(ray_world[2])
    if scale <= 0.0 or not np.isfinite(scale):
        return None
    point_world = position + scale * ray_world
    return point_world if np.all(np.isfinite(point_world)) else None


def estimate_vertical_line_height(
    calibration: CameraCalibration,
    pixel_xy: tuple[float, float] | list[float],
    world_xy: tuple[float, float] | list[float],
) -> Optional[dict[str, Any]]:
    """Estimate the height where a pixel ray meets a known world-X/Y vertical line.

    This is camera geometry only.  ``world_xy`` normally comes from a visual bbox
    back-projected to the table plane; no simulator object pose is consulted.  The
    returned residual is useful for rejecting a pixel that is not consistent with
    that vertical line (for example a clipped or mismatched detection).
    """
    try:
        x, y = float(pixel_xy[0]), float(pixel_xy[1])
        target_xy = np.asarray([float(world_xy[0]), float(world_xy[1])], dtype=float)
    except (TypeError, ValueError, IndexError):
        return None

    width = int(calibration.width)
    height = int(calibration.height)
    output_width = width if int(calibration.rotation_degrees) % 180 == 0 else height
    output_height = height if int(calibration.rotation_degrees) % 180 == 0 else width

    mode = str(calibration.flip or "none").lower()
    if mode in {"vertical", "both"}:
        y = output_height - 1 - y
    if mode in {"horizontal", "both"}:
        x = output_width - 1 - x

    k = (int(calibration.rotation_degrees) % 360) // 90
    if k == 1:
        raw_x, raw_y = width - 1 - y, x
    elif k == 2:
        raw_x, raw_y = width - 1 - x, height - 1 - y
    elif k == 3:
        raw_x, raw_y = y, height - 1 - x
    else:
        raw_x, raw_y = x, y

    focal = (height / 2.0) / np.tan(np.deg2rad(float(calibration.fovy_deg)) / 2.0)
    ray_camera = np.array(
        [(raw_x - width / 2.0) / focal, (raw_y - height / 2.0) / focal, -1.0],
        dtype=float,
    )
    position = np.asarray(calibration.position_world, dtype=float).reshape(3)
    rotation = np.asarray(calibration.camera_to_world, dtype=float).reshape(3, 3)
    ray_world = rotation @ ray_camera
    if not np.all(np.isfinite(ray_world)):
        return None

    direction_xy = ray_world[:2]
    denom = float(np.dot(direction_xy, direction_xy))
    if denom < 1e-12:
        return None
    scale = float(np.dot(target_xy - position[:2], direction_xy) / denom)
    if scale <= 0.0 or not np.isfinite(scale):
        return None
    closest_xy = position[:2] + scale * direction_xy
    residual = float(np.linalg.norm(closest_xy - target_xy))
    height_world = float(position[2] + scale * ray_world[2])
    if not np.isfinite(height_world):
        return None
    return {
        "height_m": height_world,
        "residual_m": residual,
        "ray_scale": scale,
        "world_xy": [float(target_xy[0]), float(target_xy[1])],
    }


def make_mujoco_calibrations(
    env: Any,
    camera_names: Mapping[str, str],
    *,
    image_shapes: Mapping[str, tuple[int, int]],
    rotations: Mapping[str, int] | None = None,
    flips: Mapping[str, str] | None = None,
) -> dict[str, CameraCalibration]:
    """Read camera calibration from MuJoCo, not scene object state."""
    model = env.sim.model
    data = env.sim.data
    rotations = rotations or {}
    flips = flips or {}
    result: dict[str, CameraCalibration] = {}
    for logical_name, sim_name in camera_names.items():
        height, width = image_shapes[logical_name]
        camera_id = int(model.camera_name2id(sim_name))
        result[logical_name] = CameraCalibration(
            name=logical_name,
            width=int(width),
            height=int(height),
            fovy_deg=float(model.cam_fovy[camera_id]),
            position_world=np.asarray(data.cam_xpos[camera_id], dtype=float).copy(),
            camera_to_world=np.asarray(data.cam_xmat[camera_id], dtype=float)
            .reshape(3, 3)
            .copy(),
            rotation_degrees=int(rotations.get(logical_name, 0)),
            flip=str(flips.get(logical_name, "none")),
        )
    return result


def build_robot_geometry_context(
    calibrations: Mapping[str, CameraCalibration],
    point_world: np.ndarray,
    target_bbox: Optional[tuple[float, float, float, float]] = None,
) -> dict[str, Any]:
    """Return per-camera robot projection and, when available, pixel error."""
    context: dict[str, Any] = {}
    for name, calibration in calibrations.items():
        projected = project_point(calibration, point_world)
        if projected is None:
            context[name] = {"valid": False, "source": "proprioception_camera_calibration"}
            continue
        projected["valid"] = True
        if target_bbox is not None:
            x0, y0, x1, y1 = target_bbox
            target_center = [(float(x0) + float(x1)) / 2.0, (float(y0) + float(y1)) / 2.0]
            eef = projected["pixel_xy"]
            error = [target_center[0] - eef[0], target_center[1] - eef[1]]
            projected["target_center_xy"] = [round(v, 2) for v in target_center]
            projected["target_minus_eef_px"] = [round(v, 2) for v in error]
            projected["target_screen_relation"] = {
                "horizontal": "right" if error[0] > 3 else "left" if error[0] < -3 else "aligned",
                "vertical": "down" if error[1] > 3 else "up" if error[1] < -3 else "aligned",
            }
        context[name] = projected
    return context

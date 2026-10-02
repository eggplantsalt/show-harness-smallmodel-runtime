from __future__ import annotations

import numpy as np
import pytest

from core.capabilities.camera_geometry import (
    CameraCalibration,
    camera_point_roundtrip_error,
    opencv_camera_points_to_world,
    project_point,
)


@pytest.mark.parametrize(
    "rotation,flip",
    [(0, "none"), (90, "none"), (180, "none"), (270, "none"),
     (0, "horizontal"), (90, "vertical"), (180, "both"), (270, "horizontal")],
)
def test_opencv_moge_points_roundtrip_through_policy_image_orientation(rotation, flip) -> None:
    # Square calibration keeps the focal scale invariant across quarter-turns;
    # this isolates the coordinate-basis and flip composition being tested.
    size = 128
    calibration = CameraCalibration(
        name="synthetic",
        width=size,
        height=size,
        fovy_deg=60.0,
        position_world=np.zeros(3),
        camera_to_world=np.eye(3),
        rotation_degrees=rotation,
        flip=flip,
    )
    policy_pixel = np.array([[42.0, 76.0]])
    focal = (size / 2.0) / np.tan(np.deg2rad(60.0) / 2.0)
    points_cv = np.array([[
        (policy_pixel[0, 0] - size / 2.0) / focal,
        (policy_pixel[0, 1] - size / 2.0) / focal,
        1.0,
    ]])
    world = opencv_camera_points_to_world(
        points_cv,
        camera_to_world=np.eye(3),
        position_world=np.zeros(3),
        rotation_degrees=rotation,
        flip=flip,
    )

    projected = project_point(calibration, world[0])
    assert projected is not None
    # Pixel-center definitions differ by at most one raster cell across
    # quarter-turns; no larger orientation-dependent offset is acceptable.
    assert np.allclose(projected["pixel_xy"], policy_pixel[0], atol=1.01)
    assert camera_point_roundtrip_error(calibration, world, policy_pixel)[0] < 1.5

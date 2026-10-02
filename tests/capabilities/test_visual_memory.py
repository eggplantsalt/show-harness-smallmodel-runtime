from __future__ import annotations

import numpy as np

from core.capabilities.camera_geometry import CameraCalibration, project_point
from core.capabilities.visual_memory import CpuVisualTracker, EpisodicVisualMemory


def _frame(offset: int = 0) -> np.ndarray:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    patch = np.arange(20 * 16 * 3, dtype=np.uint8).reshape(20, 16, 3)
    x0 = 30 + offset
    frame[25:45, x0 : x0 + 16] = patch
    return frame


def test_cpu_tracker_follows_small_motion() -> None:
    tracker = CpuVisualTracker(min_match=0.2, max_age=2)
    tracker.seed(_frame(0), (30, 25, 46, 45), 0.9)
    result = tracker.update(_frame(3))
    assert result is not None
    assert abs(result["bbox_xyxy"][0] - 33) <= 2
    assert result["source"] == "cpu_template_tracker"


def test_memory_reports_progress_and_invalidates_scene() -> None:
    memory = EpisodicVisualMemory(max_entries=4, ttl_frames=2)
    first = memory.append(
        frame_id=1,
        camera="agentview",
        image_hash_value="frame-1",
        stage="APPROACH",
        target="salad dressing",
        action="MV_FWD",
        bbox_xyxy=(10, 10, 20, 20),
        confidence=0.9,
        source="sam3",
    )
    second = memory.append(
        frame_id=2,
        camera="agentview",
        image_hash_value="frame-2",
        stage="APPROACH",
        target="salad dressing",
        action="MV_LEFT",
        bbox_xyxy=(14, 10, 24, 20),
        confidence=0.8,
        source="tracker",
    )
    assert first.progress_delta_px is None
    assert second.progress_delta_px == 4.0
    context = memory.context(target="salad dressing", frame_id=2)
    assert "salad dressing" in context
    # The machine field remains ``progress_delta_px``; prompts use a clearer
    # human-facing label so the model does not confuse motion with task success.
    assert "motion_since_previous=4.0px" in context
    memory.invalidate_scene("test_motion")
    assert memory.scene_epoch == 1
    assert "no current verified target evidence" in memory.context(
        target="salad dressing", frame_id=2
    )


def test_camera_geometry_projects_robot_only_and_applies_orientation() -> None:
    calibration = CameraCalibration(
        name="agentview",
        width=100,
        height=100,
        fovy_deg=90.0,
        position_world=np.zeros(3),
        camera_to_world=np.eye(3),
        rotation_degrees=180,
    )
    projected = project_point(calibration, np.asarray([0.1, -0.1, -1.0]))
    assert projected is not None
    assert projected["source"] == "proprioception_camera_calibration"
    assert projected["in_frame"]
    # The point lands below/right of center in the raw render image, then the
    # configured 180-degree policy-view rotation mirrors both raster axes.
    assert projected["raw_pixel_xy"][0] > 50.0
    assert projected["raw_pixel_xy"][1] > 50.0
    assert projected["pixel_xy"][0] < 50.0
    assert projected["pixel_xy"][1] < 50.0

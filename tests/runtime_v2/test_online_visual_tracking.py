from __future__ import annotations

import numpy as np

from core.sim.zeroshot_robolab_runner import _track_v22_target_views


def test_v22_tracker_keeps_primary_and_secondary_camera_streams_separate() -> None:
    agentview = np.zeros((8, 8, 3), dtype=np.uint8)
    wrist = np.ones((8, 8, 3), dtype=np.uint8)
    agent_mask = {"format": "row_span_rle", "shape": [8, 8], "rle": [[1, 1, 2]]}
    wrist_mask = {"format": "row_span_rle", "shape": [8, 8], "rle": [[5, 4, 6]]}
    calls = []

    def tracker(**kwargs):
        calls.append(kwargs)
        return {"health": "WARMING", "camera": kwargs["camera"]}

    results = _track_v22_target_views(
        tracker=tracker,
        stage="GRASP",
        capability_evidence={
            "camera": "wrist",
            "mask": wrist_mask,
            "secondary_view": {
                "camera": "agentview",
                "age_frames": 0,
                "mask": agent_mask,
            },
        },
        camera_images={"agentview": agentview, "wrist": wrist},
        frame_id=18,
        instance_id="instance-1",
        grasp_epoch=2,
    )

    assert list(results) == ["wrist", "agentview"]
    assert calls[0]["image"] is wrist
    assert calls[0]["mask"] == wrist_mask
    assert calls[0]["camera"] == "wrist"
    assert calls[1]["image"] is agentview
    assert calls[1]["mask"] == agent_mask
    assert calls[1]["camera"] == "agentview"
    assert all(call["frame_id"] == 18 for call in calls)
    assert all(call["instance_id"] == "instance-1" for call in calls)
    assert all(call["grasp_epoch"] == 2 for call in calls)


def test_v22_tracker_does_not_seed_from_stale_secondary_mask_or_other_stages() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    calls = []

    def tracker(**kwargs):
        calls.append(kwargs)
        return {"health": "WARMING"}

    _track_v22_target_views(
        tracker=tracker,
        stage="APPROACH",
        capability_evidence={
            "camera": "agentview",
            "mask": None,
            "secondary_view": {
                "camera": "wrist",
                "age_frames": 2,
                "mask": {"format": "stale-mask"},
            },
        },
        camera_images={"agentview": image, "wrist": image},
        frame_id=20,
        instance_id="instance-1",
        grasp_epoch=0,
    )
    assert [call["camera"] for call in calls] == ["agentview", "wrist"]
    assert calls[1]["mask"] is None

    calls.clear()
    _track_v22_target_views(
        tracker=tracker,
        stage="TRANSPORT",
        capability_evidence={"camera": "agentview", "mask": {}},
        camera_images={"agentview": image},
        frame_id=21,
        instance_id="instance-1",
        grasp_epoch=0,
    )
    assert calls == []

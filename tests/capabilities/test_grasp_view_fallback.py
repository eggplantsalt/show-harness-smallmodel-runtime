from __future__ import annotations

import numpy as np

from core.capabilities.visual_harness import VisualHarness


class _FakeSam3:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def segment(self, image, query, *, confidence_threshold):
        camera = "wrist" if int(image[0, 0, 0]) == 1 else "agentview"
        self.calls.append((camera, query))
        detections = [] if camera == "wrist" else [
            {
                "bbox_xyxy": [10, 20, 30, 40],
                "score": 0.8,
                "label": "bottle",
            }
        ]
        return {
            "success": True,
            "details": {
                "detections": detections,
                "metadata": {"fake_camera": camera},
            },
        }


def test_grasp_wrist_abstain_falls_back_to_agentview_and_persists() -> None:
    harness = VisualHarness(
        enabled=True,
        sam3_enabled=False,
        tracker_enabled=False,
        memory_enabled=False,
        geometry_enabled=False,
        confidence_threshold=0.4,
        grasp_agentview_fallback_enabled=True,
    )
    fake = _FakeSam3()
    harness.sam3 = fake
    agentview = np.zeros((64, 64, 3), dtype=np.uint8)
    wrist = np.ones((64, 64, 3), dtype=np.uint8)

    first = harness.update(
        agentview=agentview,
        wrist=wrist,
        stage="GRASP",
        target="green-capped bottle",
        affordance="neck",
        frame_id=1,
    )
    second = harness.update(
        agentview=agentview,
        wrist=wrist,
        stage="GRASP",
        target="green-capped bottle",
        affordance="neck",
        frame_id=2,
    )

    assert first["camera"] == "agentview"
    assert first["source"] == "sam3"
    assert first["bbox_xyxy"] == [10, 20, 30, 40]
    assert first["tool"]["fallback_reason"] == "wrist_no_detection"
    assert first["tool"]["primary_camera"] == "wrist"
    assert first["tool"]["fallback_camera"] == "agentview"
    assert second["camera"] == "agentview"
    assert fake.calls == [
        ("wrist", "green-capped bottle, neck"),
        ("agentview", "green-capped bottle, neck"),
    ]


def test_sam3_does_not_pick_an_ambiguous_first_bottle() -> None:
    harness = VisualHarness(
        enabled=True,
        sam3_enabled=False,
        tracker_enabled=False,
        memory_enabled=False,
        geometry_enabled=False,
        confidence_threshold=0.4,
        sam3_ambiguity_margin=0.05,
    )

    class _AmbiguousSam3:
        def segment(self, image, query, *, confidence_threshold):
            return {
                "success": True,
                "details": {
                    "detections": [
                        {"bbox_xyxy": [10, 10, 20, 20], "score": 0.68},
                        {"bbox_xyxy": [40, 40, 50, 50], "score": 0.67},
                    ]
                },
            }

    harness.sam3 = _AmbiguousSam3()
    bbox, score, source, meta = harness._sam3_ground(
        np.zeros((64, 64, 3), dtype=np.uint8), "bottle", None
    )

    assert bbox is None
    assert score == 0.68
    assert source == "sam3_abstain"
    assert meta["abstain_reason"] == "ambiguous_top_detections"


def test_transport_semantic_refresh_keeps_grasp_instance_identity() -> None:
    harness = VisualHarness(
        enabled=True,
        sam3_enabled=False,
        tracker_enabled=False,
        memory_enabled=False,
        geometry_enabled=False,
        confidence_threshold=0.4,
    )

    def fake_ground(image, target, affordance):
        return (
            (181, 95, 199, 134),
            0.92,
            "sam3",
            {
                "candidates": [
                    {"bbox_xyxy": [181, 95, 199, 134], "score": 0.92},
                    {"bbox_xyxy": [146, 97, 166, 136], "score": 0.75},
                ]
            },
        )

    harness._sam3_ground = fake_ground
    held = harness._update_held_object(
        agentview=np.zeros((256, 256, 3), dtype=np.uint8),
        stage="TRANSPORT",
        held_target="bottle",
        held_affordance="body",
        frame_id=1,
        handoff_evidence={
            "bbox_xyxy": [148, 97, 167, 136],
            "confidence": 0.9,
        },
    )
    assert held["bbox_xyxy"] == [146, 97, 166, 136]
    assert held["source"] == "sam3_instance_associated"
    assert held["tool"]["instance_association"]["distance_px"] < 3.0

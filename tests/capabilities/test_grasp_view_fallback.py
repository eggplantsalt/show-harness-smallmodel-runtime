from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

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
    # Query compression is an ordered, bounded ladder.  The first update must
    # exhaust Wrist variants before falling back to AgentView; the persisted
    # AgentView result may then be bridged by short occlusion memory while Wrist
    # is probed again.
    assert fake.calls[:4] == [
        ("wrist", "green-capped bottle, neck"),
        ("wrist", "neck bottle"),
        ("wrist", "bottle"),
        ("agentview", "green-capped bottle, neck"),
    ]
    assert fake.calls[4:] == [
        ("wrist", "green-capped bottle, neck"),
        ("wrist", "neck bottle"),
        ("wrist", "bottle"),
    ]


def test_selected_fallback_and_wrist_masks_reach_tracker_evidence() -> None:
    harness = VisualHarness(
        enabled=True,
        sam3_enabled=False,
        tracker_enabled=False,
        memory_enabled=False,
        geometry_enabled=False,
        confidence_threshold=0.4,
        reacquire_every=1,
        grasp_agentview_fallback_enabled=True,
    )
    selected_mask = {
        "format": "row_span_rle",
        "shape": [64, 64],
        "rle": [[24, 20, 40]],
        "area_px": 21,
        "bbox_xyxy": [20, 24, 41, 25],
    }
    wrist_calls = 0

    def fake_ground(image, target, affordance):
        nonlocal wrist_calls
        is_wrist = int(image[0, 0, 0]) == 1
        if is_wrist:
            wrist_calls += 1
            if wrist_calls == 1:
                return None, 0.0, "sam3_abstain", {"mask": None}
        return (
            (20, 20, 42, 48),
            0.8,
            "sam3",
            {"mask": selected_mask, "label": "target"},
        )

    harness._sam3_ground = fake_ground
    agentview = np.zeros((64, 64, 3), dtype=np.uint8)
    wrist = np.ones((64, 64, 3), dtype=np.uint8)

    fallback = harness.update(
        agentview=agentview,
        wrist=wrist,
        stage="GRASP",
        target="selected target",
        affordance="body",
        frame_id=1,
    )
    refreshed_wrist = harness.update(
        agentview=agentview,
        wrist=wrist,
        stage="GRASP",
        target="selected target",
        affordance="body",
        frame_id=2,
    )

    assert fallback["camera"] == "agentview"
    assert fallback["mask"] == selected_mask
    assert fallback["tool"]["mask"] == selected_mask
    assert refreshed_wrist["camera"] == "wrist"
    assert refreshed_wrist["mask"] == selected_mask
    assert refreshed_wrist["tool"]["mask"] == selected_mask


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


def test_ambiguous_secondary_masks_are_transient_and_camera_bound() -> None:
    def png_mask(box: tuple[int, int, int, int]) -> dict[str, str]:
        mask = np.zeros((64, 64), dtype=np.uint8)
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = 255
        stream = io.BytesIO()
        Image.fromarray(mask).save(stream, format="PNG")
        return {"base64": base64.b64encode(stream.getvalue()).decode("ascii")}

    class AmbiguousViews:
        def segment(self, image, query, *, confidence_threshold):
            is_wrist = int(image[0, 0, 0]) == 1
            detections = [] if is_wrist else [
                {
                    "bbox_xyxy": [10, 10, 20, 25],
                    "score": 0.68,
                    "label": "candidate-a",
                    "mask": png_mask((10, 10, 20, 25)),
                },
                {
                    "bbox_xyxy": [40, 35, 52, 55],
                    "score": 0.67,
                    "label": "candidate-b",
                    "mask": png_mask((40, 35, 52, 55)),
                },
            ]
            return {"success": True, "details": {"detections": detections}}

    harness = VisualHarness(
        enabled=True,
        sam3_enabled=False,
        tracker_enabled=False,
        memory_enabled=False,
        geometry_enabled=False,
        confidence_threshold=0.4,
        sam3_ambiguity_margin=0.05,
        grasp_agentview_fallback_enabled=True,
    )
    harness.sam3 = AmbiguousViews()
    agentview = np.zeros((64, 64, 3), dtype=np.uint8)
    wrist = np.ones((64, 64, 3), dtype=np.uint8)

    evidence = harness.update(
        agentview=agentview,
        wrist=wrist,
        stage="GRASP",
        target="target",
        frame_id=4,
    )

    rows = evidence["_transient_candidate_masks_by_camera"]["agentview"]
    assert len(rows) == 2
    assert rows[0]["bbox_xyxy"] == [10, 10, 20, 25]
    assert rows[0]["mask"]["shape"] == [64, 64]
    assert rows[0]["mask"]["area_px"] == 150
    assert evidence["secondary_view"]["camera"] == "agentview"
    assert "_candidate_masks" not in evidence["secondary_view"]["tool"]
    assert "_transient_candidate_masks_by_camera" not in harness.last_evidence


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

from __future__ import annotations

from core.capabilities.visual_harness import VisualHarness


def test_active_approach_completion_requires_fresh_aligned_evidence() -> None:
    harness = VisualHarness(
        enabled=True,
        sam3_enabled=False,
        tracker_enabled=False,
        memory_enabled=False,
        geometry_enabled=True,
        confidence_threshold=0.4,
        approach_completion_guard_enabled=True,
        approach_completion_guard_mode="active",
    )
    harness.last_frame_id = 10
    harness.last_grounding_frame = 10
    harness.last_evidence = {
        "visible": True,
        "source": "sam3",
        "confidence": 0.52,
        "geometry": {
            "target_minus_eef_px": [-2.0, 8.0],
            "alignment_ready": True,
        },
    }

    decision = harness.authorize_stage_completion("APPROACH")

    assert decision["eligible"] is True
    assert decision["applied"] is True
    assert decision["reason"] == "fresh_visible_aligned_evidence"


def test_approach_completion_guard_does_not_authorize_stale_or_other_stages() -> None:
    harness = VisualHarness(
        enabled=True,
        sam3_enabled=False,
        tracker_enabled=False,
        memory_enabled=False,
        geometry_enabled=True,
        approach_completion_guard_enabled=True,
        approach_completion_guard_mode="active",
        approach_completion_guard_freshness_frames=0,
    )
    harness.last_frame_id = 12
    harness.last_grounding_frame = 10
    harness.last_evidence = {
        "visible": True,
        "confidence": 0.9,
        "geometry": {
            "target_minus_eef_px": [0.0, 0.0],
            "alignment_ready": True,
        },
    }

    stale = harness.authorize_stage_completion("APPROACH")
    grasp = harness.authorize_stage_completion("GRASP")

    assert stale["eligible"] is False
    assert stale["applied"] is False
    assert grasp["reason"] == "stage_not_supported"


def test_agentview_grasp_guard_does_not_require_wrist_visibility() -> None:
    harness = VisualHarness(
        enabled=True,
        sam3_enabled=False,
        tracker_enabled=False,
        memory_enabled=False,
        geometry_enabled=True,
        confidence_threshold=0.4,
        grasp_agentview_guard_enabled=True,
        grasp_agentview_guard_mode="active",
        grasp_agentview_guard_alignment_px=12.0,
        grasp_agentview_guard_height_m=0.18,
    )
    harness.last_frame_id = 147
    harness.last_grounding_frame = 147
    harness.last_evidence = {
        "camera": "agentview",
        "visible": True,
        "source": "cpu_template_tracker",
        "confidence": 0.85,
        "geometry": {
            "target_minus_eef_px": [-3.0, 6.6],
            "eef_height_m": 0.1516,
        },
    }

    decision = harness.authorize_grasp("GRASP")

    assert decision["eligible"] is True
    assert decision["applied"] is True
    assert decision["wrist_required"] is False

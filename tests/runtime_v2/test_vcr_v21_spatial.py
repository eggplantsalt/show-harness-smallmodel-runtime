from __future__ import annotations

import numpy as np
import pytest

from core.runtime_v2 import (
    SpatialHealth,
    SpatialRelation,
    SpatialToolResult,
    ObservationHealth,
    VisualMemory,
    VisualMemoryEntry,
    VerifiedCapabilityRuntime,
    fuse_spatial_results,
)
from core.runtime_v2.providers import CoTrackerOnlineProvider, MogeDepthProvider
from core.runtime_v2.spatial import classify_relation, triangulate_metric_points_checked
from core.runtime_v2.types import EvidenceValue, EntityTrack, TruthValue
from core.runtime_v2.control import MOVE_UP, STOP


def test_spatial_fusion_rejects_conflicting_depth_sources() -> None:
    belief = fuse_spatial_results(
        [
            SpatialToolResult("moge3", "obj", 4, target_to_gripper_xyz=(0.05, 0.0, 0.0), covariance=(0.002, 0.002, 0.002), health=SpatialHealth.VALID),
            SpatialToolResult("parallax", "obj", 4, target_to_gripper_xyz=(-0.05, 0.0, 0.0), covariance=(0.002, 0.002, 0.002), health=SpatialHealth.VALID),
        ],
        frame_id=4,
        instance_id="obj",
    )
    assert belief.health == SpatialHealth.AMBIGUOUS
    assert belief.relations == (SpatialRelation.UNKNOWN,)


def test_v22_single_source_keeps_uncertainty_and_unknown_sources_do_not_vote() -> None:
    one = SpatialToolResult(
        "calibrated_depth", "obj", 4,
        target_to_gripper_xyz=(0.01, 0.0, 0.0),
        uncertainty_std_m=(0.004, 0.005, 0.006),
        health=SpatialHealth.VALID,
    )
    unknown = SpatialToolResult(
        "unqualified_depth", "obj", 4,
        target_to_gripper_xyz=(0.0, 0.0, 0.0),
        health=SpatialHealth.VALID,
    )
    belief = fuse_spatial_results(
        [one, unknown], frame_id=4, instance_id="obj", require_uncertainty=True
    )
    assert belief.health == SpatialHealth.VALID
    assert np.allclose(belief.uncertainty, (0.004, 0.005, 0.006), atol=1e-9)
    assert belief.agreeing_sources == ("calibrated_depth",)


def test_inside_gripper_envelope_requires_all_three_axes() -> None:
    labels = classify_relation((0.0, 0.08, 0.0), (0.002, 0.002, 0.002))
    assert SpatialRelation.INSIDE_ENVELOPE not in labels


def test_semantic_depth_choice_compiles_from_signed_belief() -> None:
    runtime = VerifiedCapabilityRuntime(
        semantic_pregrasp_enabled=True,
        require_spatial_ready_for_grasp=True,
        approach_min_height_m=0.10,
    )
    evidence = {
        "frame_id": 1,
        "camera": "wrist",
        "stage": "GRASP",
        "target": "object",
        "bbox_xyxy": [100, 100, 140, 150],
        "confidence": 0.9,
        "visible": True,
        "geometry": {"target_minus_eef_px": [0, 0], "pixel_xy": [128, 128]},
        "spatial_belief": {
            "health": "VALID",
            "relations": ["BACK"],
            "fused_relative_xyz": [-0.04, 0.0, 0.0],
            "uncertainty": [0.003, 0.003, 0.003],
        },
    }
    request = runtime.observe_frame(
        stage="GRASP", evidence=evidence, previous_action=None,
        agentview=np.zeros((256, 256, 3), dtype=np.uint8),
        wrist=np.zeros((256, 256, 3), dtype=np.uint8), eef_xyz=(0.0, 0.0, 0.12),
    )["critical_decision"]
    assert "CORRECT_DEPTH" in request["allowed_answers"]
    committed = runtime.apply_critical_decision("CORRECT_DEPTH")
    assert committed["next_action"] == "MV_BACK"
    assert committed["semantic_choice"] == "CORRECT_DEPTH"
    receipt = runtime.commit_executed_action(executed_action="MV_BACK", authorized_action="MV_BACK")
    assert receipt["committed"]


def test_v21_thinking_decision_is_citation_checked_and_written_to_episode_memory() -> None:
    runtime = VerifiedCapabilityRuntime(
        semantic_pregrasp_enabled=True,
        require_spatial_ready_for_grasp=True,
        pregrasp_reflection_mode="double",
    )
    evidence = {
        "frame_id": 5,
        "camera": "wrist",
        "stage": "GRASP",
        "target": "target",
        "bbox_xyxy": [100, 100, 140, 150],
        "confidence": 0.9,
        "visible": True,
        "geometry": {"target_minus_eef_px": [2, 1], "pixel_xy": [120, 125]},
        "spatial_belief": {
            "health": "VALID",
            "relations": ["BACK"],
            "fused_relative_xyz": [-0.04, 0.0, 0.0],
            "uncertainty": [0.003, 0.003, 0.003],
            "frame_id": 5,
            "instance_id": "target-instance",
        },
    }
    request = runtime.observe_frame(
        stage="GRASP",
        evidence=evidence,
        previous_action=None,
        agentview=np.zeros((64, 64, 3), dtype=np.uint8),
        wrist=np.zeros((64, 64, 3), dtype=np.uint8),
        image_refs={
            "agentview": "images/raw_agentview/0005.png",
            "wrist": "images/raw_wrist/0005.png",
        },
        eef_xyz=(0.0, 0.0, 0.12),
    )["critical_decision"]
    assert request["reflection_mode"] == "double"
    assert request["evidence_frame_ids"] == [5]

    decision = {
        "selected": "CORRECT_DEPTH",
        "state_hypothesis": "target is behind the gripper",
        "evidence_for": [{
            "frame_id": 5,
            "camera": "wrist",
            "observation": "the target is visibly behind the finger plane",
        }],
        "evidence_against": [],
        "missing_observation": "fresh view after the signed correction",
        "expected_effect": "target-to-gripper depth residual decreases",
        "failure_condition": "residual grows or remains unchanged",
        "summary": "fresh geometry and wrist view agree on depth direction",
        "frame_ids": [5],
        "validated": True,
    }
    committed = runtime.apply_critical_decision(
        "CORRECT_DEPTH", details={"agent_decision": decision}
    )

    assert committed["next_action"] == "MV_BACK"
    target_id = runtime.belief.target.instance_id
    remembered = runtime.visual_memory.placement_bundle(
        instance_id=target_id, grasp_epoch=0
    )[-1]["decision_summary"]
    assert remembered["relation"] == "target is behind the gripper"
    assert remembered["selected_option"] == "CORRECT_DEPTH"
    assert remembered["status"] == "UNVERIFIED_AGENT_HYPOTHESIS"


def test_v21_thinking_decision_with_stale_frame_cannot_authorize_action() -> None:
    runtime = VerifiedCapabilityRuntime(
        semantic_pregrasp_enabled=True,
        require_spatial_ready_for_grasp=True,
    )
    evidence = {
        "frame_id": 5, "camera": "wrist", "stage": "GRASP", "target": "target",
        "bbox_xyxy": [100, 100, 140, 150], "confidence": 0.9, "visible": True,
        "geometry": {"pixel_xy": [120, 125]},
        "spatial_belief": {
            "health": "VALID", "relations": ["BACK"],
            "fused_relative_xyz": [-0.04, 0.0, 0.0],
            "uncertainty": [0.003, 0.003, 0.003],
            "frame_id": 5, "instance_id": "target-instance",
        },
    }
    runtime.observe_frame(
        stage="GRASP", evidence=evidence, previous_action=None,
        agentview=np.zeros((64, 64, 3), dtype=np.uint8),
        wrist=np.zeros((64, 64, 3), dtype=np.uint8),
        image_refs={
            "agentview": "images/raw_agentview/0005.png",
            "wrist": "images/raw_wrist/0005.png",
        },
        eef_xyz=(0.0, 0.0, 0.12),
    )
    stale_decision = {
        "selected": "CORRECT_DEPTH",
        "state_hypothesis": "target is behind",
        "evidence_for": [{
            "frame_id": 4, "camera": "wrist", "observation": "old relation",
        }],
        "evidence_against": [], "missing_observation": "",
        "expected_effect": "", "failure_condition": "", "summary": "stale evidence",
        "frame_ids": [5], "validated": True,
    }

    committed = runtime.apply_critical_decision(
        "CORRECT_DEPTH", details={"agent_decision": stale_decision}
    )

    assert committed["answer"] == "UNKNOWN"
    assert committed["next_action"] == STOP
    target_id = runtime.belief.target.instance_id
    memory = runtime.visual_memory.placement_bundle(
        instance_id=target_id, grasp_epoch=0
    )[-1]
    assert memory["decision_summary"] == {}


def test_select_grasp_cannot_bypass_depth_envelope() -> None:
    runtime = VerifiedCapabilityRuntime(
        semantic_pregrasp_enabled=True,
        require_spatial_ready_for_grasp=True,
    )
    evidence = {
        "frame_id": 1,
        "camera": "wrist",
        "stage": "GRASP",
        "target": "object",
        "bbox_xyxy": [100, 100, 140, 150],
        "confidence": 0.9,
        "visible": True,
        "geometry": {"target_minus_eef_px": [0, 0], "pixel_xy": [128, 128]},
        "spatial_belief": {
            "health": "VALID",
            "relations": ["BACK"],
            "fused_relative_xyz": [-0.04, 0.0, 0.0],
            "uncertainty": [0.003, 0.003, 0.003],
        },
    }
    runtime.observe_frame(
        stage="GRASP", evidence=evidence, previous_action=None,
        agentview=np.zeros((256, 256, 3), dtype=np.uint8),
        wrist=np.zeros((256, 256, 3), dtype=np.uint8), eef_xyz=(0.0, 0.0, 0.12),
    )
    committed = runtime.apply_critical_decision("SELECT_GRASP")
    assert committed["next_action"] == "MV_BACK"
    assert "depth" in committed["reason"]


def test_visual_memory_is_bounded_per_grasp_epoch() -> None:
    memory = VisualMemory(max_entries_per_epoch=2)
    for frame in range(4):
        memory.append(VisualMemoryEntry(instance_id="obj", grasp_epoch=0, frame_id=frame))
    assert [x.frame_id for x in memory.recent(instance_id="obj", grasp_epoch=0, limit=8)] == [3, 2]
    assert memory.refs(instance_id="obj", grasp_epoch=0) == ()


def test_visual_memory_deduplicates_repeated_event_key_and_expires_decisions() -> None:
    memory = VisualMemory(max_entries_per_epoch=5)
    memory.append(VisualMemoryEntry("obj", 2, 10, agentview_ref="images/raw_agentview/0010.png"))
    memory.mark(instance_id="obj", grasp_epoch=2, frame_id=10, tag="CONTACT", event_key="2:contact")
    memory.append(VisualMemoryEntry("obj", 2, 11, agentview_ref="images/raw_agentview/0011.png"))
    memory.mark(instance_id="obj", grasp_epoch=2, frame_id=11, tag="CONTACT", event_key="2:contact")
    assert [item["frame_id"] for item in memory.placement_bundle(instance_id="obj", grasp_epoch=2)] == [10, 11]
    assert "CONTACT" not in memory.placement_bundle(instance_id="obj", grasp_epoch=2)[-1]["tags"]

    assert memory.record_decision(
        instance_id="obj", grasp_epoch=2, frame_id=11, route_epoch=3,
        relation="UNKNOWN", reasoning="support not visible", lifetime_frames=2,
    )
    current = memory.placement_bundle(instance_id="obj", grasp_epoch=2, route_epoch=4)
    assert current[-1]["decision_summary"] == {}


def test_cotracker_requires_exact_grounded_mask_and_resets_identity_epoch() -> None:
    provider = CoTrackerOnlineProvider(
        repo_dir="/missing/cotracker", checkpoint="/missing/weights.pth"
    )
    image = np.zeros((12, 14, 3), dtype=np.uint8)
    assert provider.infer(
        image=image, mask=None, camera="wrist", frame_id=1,
        instance_id="a", grasp_epoch=0,
    )["reason"] == "grounded_instance_mask_required"
    mask = {"format": "row_span_rle", "shape": [12, 14], "rle": [[3, 4, 7]]}
    decoded = provider._decode_mask(mask, (12, 14))
    assert decoded is not None and int(decoded.sum()) == 4
    assert provider._decode_mask(mask, (13, 14)) is None
    result = provider.infer(
        image=image, mask=mask, camera="wrist", frame_id=2,
        instance_id="a", grasp_epoch=0,
    )
    assert result["health"] == "SENSOR_FAULT"
    assert "checkpoint not found" in result["reason"]
    provider._reset_stream(("wrist", "a", 0))
    assert provider._key == ("wrist", "a", 0)
    provider._reset_stream(("agentview", "b", 1))
    assert provider._total_frames == 0 and not provider._initialized


def test_runtime_summarizes_valid_cotracker_motion_without_promoting_geometry() -> None:
    class Tracker:
        def infer(self, **kwargs):
            return {
                "health": "VALID",
                "camera": kwargs["camera"],
                "query_frame_id": 4,
                "frame_id": kwargs["frame_id"],
                "visible_count": 3,
                "query_points_xy": [[10, 10], [20, 20], [30, 30]],
                "current_points_xy": [[11, 12], [23, 24], [35, 36]],
                "inference_latency_s": 0.03,
            }

    runtime = VerifiedCapabilityRuntime(placement_v22_enabled=True)
    runtime.visual_point_tracker = Tracker()
    result = runtime.track_visual_points(
        image=np.zeros((40, 40, 3), dtype=np.uint8),
        mask=np.ones((40, 40), dtype=bool),
        camera="agentview",
        frame_id=12,
        instance_id="episode-target",
        grasp_epoch=2,
    )
    assert result["health"] == "VALID"
    assert result["motion_summary"]["median_displacement_px"] == [3.0, 4.0]
    assert result["motion_summary"]["median_track_motion_px"] == 5.0
    assert "target_points_world" not in result


def test_moge_model_version_is_explicit_and_v2_alias_resolves_to_v2() -> None:
    provider = MogeDepthProvider(model_version="v2_fallback")
    assert provider.model_version == "v2"
    assert provider.source == "moge2"
    try:
        MogeDepthProvider(model_version="v4")
    except ValueError as exc:
        assert "unsupported MoGe model version" in str(exc)
    else:
        raise AssertionError("unknown MoGe versions must fail closed")


def test_moge_direct_provider_returns_masked_points_with_matching_pixels() -> None:
    torch = pytest.importorskip("torch")

    class _Model:
        def infer(self, image, *, fov_x):
            height, width = image.shape[-2:]
            points = torch.zeros((height, width, 3), dtype=torch.float32)
            points[..., 2] = 1.0
            return {
                "points": points,
                "mask": torch.ones((height, width), dtype=torch.bool),
                "intrinsics": None,
            }

    provider = MogeDepthProvider(model_version="v2", device="cpu")
    provider.model = _Model()
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    mask = {
        "format": "row_span_rle",
        "shape": [8, 10],
        "rle": [[2, 2, 6], [3, 2, 6]],
    }

    result = provider.infer(
        image=image,
        mask=mask,
        frame_id=7,
        instance_id="same-instance",
        camera_calibration={"width": 10, "height": 8, "fovy_deg": 60.0},
    )

    assert result.health == SpatialHealth.VALID
    assert len(result.target_points_camera) == 10
    assert len(result.target_points_pixels) == len(result.target_points_camera)
    assert all(2 <= point[0] <= 6 and 2 <= point[1] <= 3 for point in result.target_points_pixels)
    assert result.diagnostics["instance_mask_applied"] is True
    assert result.diagnostics["sample_count"] == 10


def test_runtime_episode_reset_clears_visual_memory_and_online_track_stream() -> None:
    runtime = VerifiedCapabilityRuntime(placement_v22_enabled=True)
    runtime.visual_memory.append(VisualMemoryEntry("obj", 1, 3))
    tracker = CoTrackerOnlineProvider(repo_dir="/missing/cotracker", checkpoint="/missing/weights.pth")
    tracker._reset_stream(("wrist", "obj", 1))
    runtime.visual_point_tracker = tracker

    runtime.reset(episode_id="episode-next")

    assert runtime.visual_memory.episode_id == "episode-next"
    assert runtime.visual_memory.refs(instance_id="obj", grasp_epoch=1) == ()
    assert tracker._key is None and tracker._total_frames == 0


def test_v22_grasp_yes_is_candidate_and_width_cannot_commit_held_state() -> None:
    runtime = VerifiedCapabilityRuntime(placement_v22_enabled=True)
    runtime.belief.frame_id = 9
    runtime.belief.observation_health = ObservationHealth.VALID
    runtime.belief.target = EntityTrack(
        instance_id="episode-instance", semantic_label="target",
        bbox_xyxy=(40.0, 50.0, 80.0, 110.0), confidence=0.9,
        source="visual", last_confirmed_frame=9, camera="agentview",
    )
    runtime.belief.eef_xyz = EvidenceValue((0.0, 0.0, 0.1), TruthValue.TRUE, "robot", 1.0, 9)
    runtime.belief.alignment_residual = EvidenceValue((2.0, -1.0), TruthValue.TRUE, "visual", 0.9, 9)

    result = runtime.report_grasp_verdict(
        verdict="YES", frame_id=9, mechanically_empty=True,
        diagnostic_lift_clear=True,
        evidence_for=[{"frame_id": 9, "camera": "agentview", "observation": "target enclosed by the fingers"}],
    )

    assert result["verdict"] == "CANDIDATE"
    assert result["diagnostic_lift_authorized"] is True
    assert runtime.belief.held.truth == TruthValue.UNKNOWN
    assert runtime.belief.grasp_epoch == 0


def test_v22_grasp_candidate_without_visible_lift_clearance_fails_closed() -> None:
    runtime = VerifiedCapabilityRuntime(placement_v22_enabled=True)
    runtime.belief.frame_id = 3
    runtime.belief.observation_health = ObservationHealth.VALID
    runtime.belief.target = EntityTrack(
        instance_id="i", semantic_label="target", bbox_xyxy=(1.0, 2.0, 10.0, 12.0),
        confidence=0.9, source="visual", last_confirmed_frame=3, camera="wrist",
    )
    runtime.belief.eef_xyz = EvidenceValue((0.0, 0.0, 0.1), TruthValue.TRUE, "robot", 1.0, 3)
    runtime.belief.alignment_residual = EvidenceValue((1.0, 0.0), TruthValue.TRUE, "visual", 0.9, 3)
    result = runtime.report_grasp_verdict(
        verdict="YES", frame_id=3, mechanically_empty=False,
        diagnostic_lift_clear=False,
        evidence_for=[{"frame_id": 3, "camera": "wrist", "observation": "target visible after close"}],
    )
    assert result["verdict"] == "UNKNOWN"
    assert result["diagnostic_lift_authorized"] is False
    assert runtime.belief.held.truth == TruthValue.UNKNOWN


def test_v22_diagnostic_lift_requires_executed_motion_and_fresh_stability() -> None:
    runtime = VerifiedCapabilityRuntime(placement_v22_enabled=True)
    runtime.belief.target = EntityTrack(
        instance_id="i", semantic_label="target", bbox_xyxy=(10.0, 10.0, 30.0, 40.0),
        confidence=0.9, source="visual", last_confirmed_frame=4, camera="wrist",
    )
    runtime.belief.held = EvidenceValue(None, TruthValue.UNKNOWN, "candidate", 0.0, 4)
    runtime._grasp_candidate = {"frame_id": 4, "instance_id": "i", "camera": "wrist"}
    runtime._grasp_diagnostic_lift_authorized = True

    requested = runtime._advance_grasp_candidate(
        stage="GRASP", frame_id=5, health=ObservationHealth.VALID,
        evidence={"visible": True}, camera="wrist", eef_xyz=(0.0, 0.0, 0.1),
        previous_eef_xyz=(0.0, 0.0, 0.1), previous_residual=(0.0, 0.0),
        observed_residual=(0.0, 0.0), gripper_closed=True,
    )
    assert requested.action_token == MOVE_UP
    runtime.last_event["grasp_diagnostic_lift_offer"] = True
    runtime.commit_executed_action(executed_action=MOVE_UP, authorized_action=MOVE_UP)

    first_after_lift = runtime._advance_grasp_candidate(
        stage="GRASP", frame_id=5, health=ObservationHealth.VALID,
        evidence={"visible": True}, camera="wrist", eef_xyz=(0.0, 0.0, 0.01),
        previous_eef_xyz=(0.0, 0.0, 0.0), previous_residual=(0.0, 0.0),
        observed_residual=(0.5, 0.2), gripper_closed=True,
    )
    assert first_after_lift.action_token == STOP
    assert runtime.belief.held.truth == TruthValue.UNKNOWN
    runtime.commit_executed_action(executed_action=STOP, authorized_action=STOP)

    stable = runtime._advance_grasp_candidate(
        stage="GRASP", frame_id=6, health=ObservationHealth.VALID,
        evidence={"visible": True}, camera="wrist", eef_xyz=(0.0, 0.0, 0.0105),
        previous_eef_xyz=(0.0, 0.0, 0.01), previous_residual=(0.5, 0.2),
        observed_residual=(0.7, 0.1), gripper_closed=True,
    )
    assert stable.action_token == STOP
    assert runtime.belief.held.truth == TruthValue.TRUE
    assert runtime.belief.grasp_epoch == 1


def test_v22_failed_co_motion_does_not_commit_held() -> None:
    runtime = VerifiedCapabilityRuntime(placement_v22_enabled=True)
    runtime.belief.target = EntityTrack(
        instance_id="i", semantic_label="target", bbox_xyxy=(10.0, 10.0, 30.0, 40.0),
        confidence=0.9, source="visual", last_confirmed_frame=4, camera="wrist",
    )
    runtime._grasp_candidate = {"frame_id": 4, "instance_id": "i", "camera": "wrist"}
    runtime._grasp_diagnostic_lift_attempted = True
    runtime.last_event["action_receipt"] = {
        "executed_action": MOVE_UP, "authorized_action": MOVE_UP,
    }
    result = runtime._advance_grasp_candidate(
        stage="GRASP", frame_id=5, health=ObservationHealth.VALID,
        evidence={"visible": True}, camera="wrist", eef_xyz=(0.0, 0.0, 0.01),
        previous_eef_xyz=(0.0, 0.0, 0.0), previous_residual=(0.0, 0.0),
        observed_residual=(12.0, 0.0), gripper_closed=True,
    )
    assert result.action_token == STOP
    assert runtime.belief.held.truth == TruthValue.UNKNOWN


def test_moge_requires_grounded_mask_and_never_substitutes_bbox() -> None:
    provider = MogeDepthProvider(
        model_version="v2", service_python="/unused/isolated/python"
    )
    image = np.zeros((12, 14, 3), dtype=np.uint8)
    missing = provider.infer(
        image=image, mask=None, bbox_xyxy=(2, 2, 10, 10), frame_id=1,
        instance_id="a",
    )
    assert missing.health == SpatialHealth.UNKNOWN
    assert missing.diagnostics["reason"] == "grounded_instance_mask_required"
    assert provider._process is None
    mismatched = provider.infer(
        image=image,
        mask={"format": "row_span_rle", "shape": [11, 14], "rle": [[2, 2, 8]]},
        bbox_xyxy=(2, 2, 10, 10), frame_id=2, instance_id="a",
    )
    assert mismatched.health != SpatialHealth.VALID
    assert mismatched.target_points_camera == ()
    assert provider._process is None


def test_v21_moge_geometry_must_reproject_before_it_can_drive_grasp() -> None:
    class _BadMogeProvider:
        def infer(self, **kwargs):
            return SpatialToolResult(
                source="moge2",
                instance_id="target",
                frame_id=4,
                target_points_camera=(
                    (0.0, 0.0, 1.0), (0.02, 0.0, 1.0),
                    (0.0, 0.02, 1.0), (0.02, 0.02, 1.0),
                ),
                # Deliberately inconsistent with the metric points.
                target_points_pixels=((15.0, 15.0),) * 4,
                uncertainty_std_m=(0.005, 0.005, 0.005),
                health=SpatialHealth.VALID,
            )

    runtime = VerifiedCapabilityRuntime(placement_v22_enabled=False)
    runtime.spatial_bus.register("moge2", _BadMogeProvider())
    belief = runtime.infer_spatial(
        image=np.zeros((128, 128, 3), dtype=np.uint8),
        bbox_xyxy=(40.0, 40.0, 80.0, 80.0),
        frame_id=4,
        instance_id="target",
        instance_mask=np.ones((128, 128), dtype=bool),
        eef_xyz=(0.0, 0.0, 0.0),
        camera_calibration={
            "name": "test",
            "width": 128,
            "height": 128,
            "fovy_deg": 60.0,
            "position_world": (0.0, 0.0, 0.0),
            "camera_to_world": np.eye(3).tolist(),
            "rotation_degrees": 0,
            "flip": "none",
        },
    )

    assert belief["health"] == SpatialHealth.UNKNOWN.value
    assert belief["fused_relative_xyz"] is None


def test_v21_moge_provider_receives_calibration_and_valid_geometry_is_accepted() -> None:
    calibration = {
        "name": "test",
        "width": 128,
        "height": 128,
        "fovy_deg": 60.0,
        # Place the calibrated camera high enough that the positive-depth
        # OpenCV points map above the configured support plane.
        "position_world": (0.0, 0.0, 2.0),
        "camera_to_world": np.eye(3).tolist(),
        "rotation_degrees": 0,
        "flip": "none",
    }
    pixels = np.asarray(
        [[48.0, 48.0], [80.0, 48.0], [48.0, 80.0], [80.0, 80.0]]
    )
    focal = (128.0 / 2.0) / np.tan(np.deg2rad(60.0) / 2.0)
    camera_points = np.column_stack(
        ((pixels[:, 0] - 64.0) / focal, (pixels[:, 1] - 64.0) / focal, np.ones(4))
    )

    class _MogeProvider:
        received_calibration = None

        def infer(self, **kwargs):
            self.received_calibration = kwargs["camera_calibration"]
            return SpatialToolResult(
                source="moge2",
                instance_id=kwargs["instance_id"],
                frame_id=kwargs["frame_id"],
                target_points_camera=tuple(map(tuple, camera_points.tolist())),
                target_points_pixels=tuple(map(tuple, pixels.tolist())),
                uncertainty_std_m=(0.01, 0.01, 0.01),
                health=SpatialHealth.VALID,
            )

    provider = _MogeProvider()
    runtime = VerifiedCapabilityRuntime(placement_v22_enabled=False)
    runtime.spatial_bus.register("moge2", provider)
    belief = runtime.infer_spatial(
        image=np.zeros((128, 128, 3), dtype=np.uint8),
        bbox_xyxy=(40.0, 40.0, 80.0, 80.0),
        frame_id=5,
        instance_id="target",
        instance_mask=np.ones((128, 128), dtype=bool),
        eef_xyz=(0.0, 0.0, -0.9),
        camera_calibration=calibration,
    )

    assert provider.received_calibration == calibration
    assert belief["health"] == SpatialHealth.VALID.value
    assert belief["fused_relative_xyz"] is not None
    provider_diagnostics = belief["diagnostics"]["providers"][0]["diagnostics"]
    assert provider_diagnostics["roundtrip_projectable_count"] == 4
    assert provider_diagnostics["roundtrip_median_px"] < 1.5
    assert provider_diagnostics["support_plane_check"]["passed"] is True


def test_v21_moge_reprojection_does_not_certify_points_below_support_plane() -> None:
    pixels = np.asarray(
        [[48.0, 48.0], [80.0, 48.0], [48.0, 80.0], [80.0, 80.0]]
    )
    focal = (128.0 / 2.0) / np.tan(np.deg2rad(60.0) / 2.0)
    camera_points = np.column_stack(
        ((pixels[:, 0] - 64.0) / focal, (pixels[:, 1] - 64.0) / focal, np.ones(4))
    )

    class _BelowPlaneMoge:
        def infer(self, **kwargs):
            return SpatialToolResult(
                source="moge2",
                instance_id=kwargs["instance_id"],
                frame_id=kwargs["frame_id"],
                target_points_camera=tuple(map(tuple, camera_points.tolist())),
                target_points_pixels=tuple(map(tuple, pixels.tolist())),
                health=SpatialHealth.VALID,
            )

    runtime = VerifiedCapabilityRuntime(
        placement_v22_enabled=False, support_plane_z_m=0.015
    )
    runtime.spatial_bus.register("moge2", _BelowPlaneMoge())
    belief = runtime.infer_spatial(
        image=np.zeros((128, 128, 3), dtype=np.uint8),
        bbox_xyxy=(40.0, 40.0, 80.0, 80.0),
        frame_id=6,
        instance_id="target",
        instance_mask=np.ones((128, 128), dtype=bool),
        eef_xyz=(0.0, 0.0, -1.1),
        camera_calibration={
            "name": "test",
            "width": 128,
            "height": 128,
            "fovy_deg": 60.0,
            "position_world": (0.0, 0.0, 0.0),
            "camera_to_world": np.eye(3).tolist(),
            "rotation_degrees": 0,
            "flip": "none",
        },
    )

    assert belief["health"] == SpatialHealth.UNKNOWN.value
    assert belief["fused_relative_xyz"] is None
    provider_diagnostics = belief["diagnostics"]["rejected_providers"][0]["diagnostics"]
    assert provider_diagnostics["roundtrip_median_px"] < 1.5
    assert provider_diagnostics["support_plane_check"]["passed"] is False

def test_triangulation_rejects_negative_depth_and_bad_reprojection() -> None:
    projection_a = np.array(
        [[100.0, 0.0, 50.0, 0.0], [0.0, 100.0, 50.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    projection_b = np.array(
        [[100.0, 0.0, 50.0, -10.0], [0.0, 100.0, 50.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    points = np.array([[x, y, 2.0] for x, y in ((0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1))])

    def project(projection, xyz):
        homogeneous = np.column_stack([xyz, np.ones(len(xyz))])
        pixels = (projection @ homogeneous.T).T
        return pixels[:, :2] / pixels[:, 2:3]

    valid = triangulate_metric_points_checked(
        project(projection_a, points), project(projection_b, points),
        projection_a, projection_b, min_correspondences=4,
    )
    assert valid is not None
    assert valid["valid_correspondences"] == 4
    assert "surface_spread_m" in valid and "median_reprojection_error_px" in valid

    behind = points.copy()
    behind[:, 2] *= -1.0
    assert triangulate_metric_points_checked(
        project(projection_a, behind), project(projection_b, behind),
        projection_a, projection_b, min_correspondences=4,
    ) is None
    mismatched = project(projection_b, points) + np.array([8.0, -4.0])
    assert triangulate_metric_points_checked(
        project(projection_a, points), mismatched,
        projection_a, projection_b, min_correspondences=4,
    ) is None

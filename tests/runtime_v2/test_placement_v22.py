from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from core.capabilities.camera_geometry import (
    CameraCalibration,
    backproject_pixel_to_plane,
    project_point,
)
from core.runtime_v2 import (
    AnyPlaceShadowProvider,
    PlacementHysteresis,
    PlacementRelation,
    PlacementSpatialHarness,
    SemanticPlacementAction,
    VerifiedCapabilityRuntime,
)
from core.runtime_v2.control import DONE, MOVE_DOWN, MOVE_UP, STOP
from core.runtime_v2.types import EvidenceValue, FailureCode, OptionName, PlacementBelief, TruthValue
from plugins.visual_route import VisualRoutePlugin


def _calibration() -> CameraCalibration:
    return CameraCalibration(
        name="agentview",
        width=256,
        height=256,
        fovy_deg=45.0,
        position_world=np.array([0.896577, 0.000001, 0.65]),
        camera_to_world=np.array(
            [
                [-0.000002, -0.52877, 0.848765],
                [1.0, -0.000001, 0.000002],
                [-0.0, 0.848765, 0.52877],
            ]
        ),
        rotation_degrees=180,
        flip="none",
    )


def _geometry() -> dict:
    calibration = _calibration()
    return {
        "agentview": {
            "camera_calibration": {
                "width": calibration.width,
                "height": calibration.height,
                "fovy_deg": calibration.fovy_deg,
                "position_world": calibration.position_world.tolist(),
                "camera_to_world": calibration.camera_to_world.tolist(),
                "rotation_degrees": calibration.rotation_degrees,
                "flip": calibration.flip,
            }
        }
    }


def _mask(x0: int = 30, x1: int = 74, y0: int = 103, y1: int = 125) -> dict:
    return {
        "format": "row_span_rle",
        "shape": [256, 256],
        "rle": [[y, x0, x1] for y in range(y0, y1 + 1)],
        "area": (x1 - x0 + 1) * (y1 - y0 + 1),
        "bbox_xyxy": [x0, y0, x1, y1],
    }


def _route_evidence(relation: str, frame_id: int, candidates=None) -> dict:
    return {
        "placement_belief": {
            "relation": relation,
            "frame_id": frame_id,
            "grasp_epoch": 4,
            "opening_xy_world": [0.1, 0.2],
            "rim_plane_z_m": 0.07,
            "rim_clearance_m": 0.001,
            "eef_residual_world": [0.0, 0.0, 0.0],
            "uncertainty_m": [0.001, 0.001, 0.001],
            "containment_margin_m": 0.04,
            "fresh": True,
            "evidence_sources": ["sam3_mask", "moge2", "active_parallax"],
        },
        "progress": {"route_direction_candidates": candidates or []},
    }


def test_opening_target_is_backprojected_at_rim_plane_not_support_plane() -> None:
    plugin = VisualRoutePlugin(enabled=True, mode="active", client=None)
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    # Synthetic bbox projected under MuJoCo's +Y-up camera convention after
    # the policy-view rotation; keep this consistent with _geometry().
    held = {"bbox_xyxy": [148, 70, 167, 95], "confidence": 0.9}
    destination = {
        "bbox_xyxy": [17, 103, 77, 160],
        "opening_bbox_xyxy": [30, 103, 74, 125],
        "opening_mask": _mask(),
        "confidence": 0.9,
    }
    output = plugin.update(
        agentview=image,
        wrist=None,
        stage="TRANSPORT",
        subgoals=[],
        current_index=0,
        frame_id=0,
        eef_world=[0.06744, -0.09998, 0.11518],
        geometry=_geometry(),
        held_evidence=held,
        destination_evidence=destination,
        gripper_closed=True,
    )
    route = output["evidence"]["route"]
    assert route["valid"]
    assert route["estimated_rim_height_m"] > plugin.table_height_m
    support = backproject_pixel_to_plane(
        _calibration(), [52.0, 114.0], plugin.table_height_m
    )
    assert support is not None
    assert np.linalg.norm(np.asarray(route["opening_xy_world"]) - support[:2]) > 1e-3
    assert route["geometry_source"] == "calibrated_rim_plane_locked_grasp_offset"


def test_locked_grasp_offset_moves_eef_target_without_scene_constants() -> None:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    kwargs = dict(
        agentview=image,
        wrist=None,
        stage="TRANSPORT",
        subgoals=[],
        current_index=0,
        frame_id=0,
        eef_world=[0.06744, -0.09998, 0.11518],
        geometry=_geometry(),
        destination_evidence={
            "bbox_xyxy": [17, 103, 77, 160],
            "opening_bbox_xyxy": [30, 103, 74, 125],
            "opening_mask": _mask(),
        },
        gripper_closed=True,
    )
    centered = VisualRoutePlugin(enabled=True, mode="active", client=None).update(
        held_evidence={"bbox_xyxy": [148, 70, 167, 95]}, **kwargs
    )["evidence"]["route"]
    offset = [0.03, -0.02, 0.0]
    eccentric = VisualRoutePlugin(enabled=True, mode="active", client=None).update(
        held_evidence={"bbox_xyxy": [148, 70, 167, 95], "object_to_gripper_xyz": offset},
        **kwargs,
    )["evidence"]["route"]
    delta = np.asarray(eccentric["destination_xy_world"]) - np.asarray(centered["destination_xy_world"])
    np.testing.assert_allclose(delta, -np.asarray(offset[:2]), atol=2e-5)


def test_height_change_does_not_create_formal_world_xy_residual() -> None:
    calibration = CameraCalibration(
        name="test",
        width=256,
        height=256,
        fovy_deg=60.0,
        position_world=np.array([0.0, 0.0, 1.0]),
        camera_to_world=np.eye(3),
    )
    low = np.array([0.08, -0.06, 0.20])
    high = np.array([0.08, -0.06, 0.50])
    low_pixel = project_point(calibration, low)["pixel_xy"]
    high_pixel = project_point(calibration, high)["pixel_xy"]
    assert np.linalg.norm(np.asarray(low_pixel) - np.asarray(high_pixel)) > 1.0
    low_back = backproject_pixel_to_plane(calibration, low_pixel, low[2])
    high_back = backproject_pixel_to_plane(calibration, high_pixel, high[2])
    assert low_back is not None and high_back is not None
    np.testing.assert_allclose(low_back[:2], high_back[:2], atol=3e-3)


def test_spatial_harness_compiles_physical_relations_to_one_action() -> None:
    harness = PlacementSpatialHarness(confirm_frames=1)
    assert harness.compile_action(relation=PlacementRelation.ABOVE_ALIGNED).action_token == MOVE_DOWN
    assert harness.compile_action(relation=PlacementRelation.DESCENDING_CLEAR).action_token == MOVE_DOWN
    assert harness.compile_action(relation=PlacementRelation.RIM_CONTACT).action_token == MOVE_UP
    assert harness.compile_action(relation=PlacementRelation.SEATED_HELD).action_token == DONE
    assert harness.compile_action(relation=PlacementRelation.UNKNOWN).action_token == STOP
    correction = harness.compile_action(
        relation=PlacementRelation.ABOVE_UNALIGNED,
        route_candidates=[{"token": "MV_LEFT", "cosine": 0.9}],
    )
    assert correction.action_token == "MV_LEFT"
    assert harness.compile_action(
        relation=PlacementRelation.ABOVE_UNALIGNED,
        residual_world=[0.0, 0.0, 0.0],
    ).action_token == MOVE_DOWN
    assert harness.semantic_action_for_relation(
        PlacementRelation.ABOVE_ALIGNED
    ) == SemanticPlacementAction.DESCEND
    assert harness.semantic_action_for_relation(
        PlacementRelation.RIM_CONTACT
    ) == SemanticPlacementAction.RECOVER_CLEAR


def test_placement_hysteresis_rejects_one_frame_boundary_noise() -> None:
    latch = PlacementHysteresis(enter_margin_m=0.02, exit_margin_m=0.01, confirm_frames=2)
    assert latch.update(PlacementRelation.ABOVE_ALIGNED, margin_m=0.021) == PlacementRelation.UNKNOWN
    assert latch.update(PlacementRelation.ABOVE_ALIGNED, margin_m=0.021) == PlacementRelation.ABOVE_ALIGNED
    assert latch.update(PlacementRelation.ABOVE_UNALIGNED, margin_m=0.0105) == PlacementRelation.ABOVE_ALIGNED
    assert latch.update(PlacementRelation.ABOVE_UNALIGNED, margin_m=0.0105) == PlacementRelation.ABOVE_ALIGNED
    assert latch.update(PlacementRelation.ABOVE_UNALIGNED, margin_m=0.03) == PlacementRelation.ABOVE_ALIGNED
    assert latch.update(PlacementRelation.ABOVE_UNALIGNED, margin_m=0.03) == PlacementRelation.ABOVE_UNALIGNED


def test_runtime_v22_above_aligned_is_compiled_to_descent_after_confirmation() -> None:
    runtime = VerifiedCapabilityRuntime(placement_v22_enabled=True, placement_confirm_frames=2)
    first = runtime._control_from_route(_route_evidence(PlacementRelation.ABOVE_ALIGNED.value, 1))
    second = runtime._control_from_route(_route_evidence(PlacementRelation.ABOVE_ALIGNED.value, 2))
    assert first is not None and first.action_token == STOP
    assert second is not None and second.action_token == MOVE_DOWN


def test_runtime_v22_transport_never_falls_back_to_pixel_alignment_controller() -> None:
    runtime = VerifiedCapabilityRuntime(placement_v22_enabled=True, placement_confirm_frames=1)
    evidence = _route_evidence(
        PlacementRelation.ABOVE_ALIGNED.value,
        1,
    )
    evidence["held_object_alignment"] = {
        "known": True,
        "aligned": False,
        "destination_minus_held_center_px": [80.0, 80.0],
        "correction_candidates": {"horizontal": "MV_LEFT", "vertical": "MV_FWD"},
    }
    decision = runtime._choose_action(
        option=OptionName.ALIGN_OPENING,
        stage="TRANSPORT",
        evidence={"visual_route": evidence},
        previous_action=None,
        eef_xyz=(0.0, 0.0, 0.2),
    )
    assert decision.action_token == MOVE_DOWN


def test_runtime_v22_placement_transaction_uses_transfer_budget() -> None:
    runtime = VerifiedCapabilityRuntime(placement_v22_enabled=True)
    assert runtime._budget_for(OptionName.ALIGN_OPENING) == runtime.limits.transfer
    assert runtime._budget_for(OptionName.DESCEND_TO_SEAT) == runtime.limits.transfer


def test_v22_clearance_is_unaligned_metric_evidence_with_signed_lift() -> None:
    plugin = VisualRoutePlugin(
        enabled=True,
        mode="active",
        placement_v22_enabled=True,
        move_vectors={"MV_UP": [0.0, 0.0, 1.0]},
    )
    output = plugin.update(
        agentview=np.zeros((256, 256, 3), dtype=np.uint8),
        wrist=None,
        stage="TRANSPORT",
        subgoals=[],
        current_index=0,
        frame_id=9,
        eef_world=[0.06744, -0.09998, 0.11518],
        geometry=_geometry(),
        held_evidence={"bbox_xyxy": [148, 70, 167, 95], "confidence": 0.9},
        destination_evidence={
            "bbox_xyxy": [17, 103, 77, 160],
            "opening_bbox_xyxy": [30, 103, 74, 125],
            "opening_mask": _mask(),
            "confidence": 0.9,
        },
        gripper_closed=True,
    )
    evidence = output["evidence"]
    assert evidence["route"]["active_leg"] == "CLEARANCE"
    assert evidence["placement_belief"]["relation"] == PlacementRelation.ABOVE_UNALIGNED.value
    decision = VerifiedCapabilityRuntime(
        placement_v22_enabled=True, placement_confirm_frames=1
    )._control_from_route(evidence)
    assert decision is not None and decision.action_token == MOVE_UP


def test_runtime_v22_unaligned_and_contact_have_distinct_recovery_semantics() -> None:
    runtime = VerifiedCapabilityRuntime(placement_v22_enabled=True, placement_confirm_frames=1)
    correction = runtime._control_from_route(
        _route_evidence(
            PlacementRelation.ABOVE_UNALIGNED.value,
            1,
            [{"token": "MV_RIGHT", "cosine": 0.8}],
        )
    )
    contact = runtime._control_from_route(
        _route_evidence(PlacementRelation.RIM_CONTACT.value, 2)
    )
    unknown = runtime._control_from_route(
        _route_evidence(PlacementRelation.UNKNOWN.value, 3)
    )
    assert correction is not None and correction.action_token == "MV_RIGHT"
    assert contact is not None and contact.action_token == STOP
    assert unknown is not None and unknown.action_token == STOP


def test_runtime_v22_contact_verifier_negative_commits_one_clear_recovery() -> None:
    runtime = VerifiedCapabilityRuntime(placement_v22_enabled=True, placement_confirm_frames=1)
    runtime._pending_critical_kind = "VERIFY_SEATED"
    runtime._pending_contact_supported = True
    response = runtime.apply_critical_decision(
        PlacementRelation.RIM_CONTACT.value, details={"next_step": "CLEAR_RIM"}
    )
    assert response["accepted"] is True
    assert runtime.belief.recovery is not None
    assert runtime.belief.recovery.failure.code == FailureCode.RIM_CONTACT
    decision = runtime._control_from_route(
        _route_evidence(
            PlacementRelation.RIM_CONTACT.value,
            2,
            [{"token": "MV_UP", "cosine": 1.0}],
        )
    )
    assert decision is not None and decision.action_token == MOVE_UP


def test_unconfirmed_visual_contact_cannot_lift_or_release() -> None:
    runtime = VerifiedCapabilityRuntime(placement_v22_enabled=True, placement_confirm_frames=1)
    runtime._pending_critical_kind = "VERIFY_SEATED"
    risk = runtime.apply_critical_decision(PlacementRelation.RIM_CONTACT.value)
    assert risk["next_action"] == STOP
    assert runtime.belief.recovery is None
    runtime._pending_critical_kind = "VERIFY_SEATED"
    seated = runtime.apply_critical_decision(PlacementRelation.SEATED_HELD.value)
    assert seated["next_action"] == STOP
    assert runtime.belief.seated.truth == TruthValue.UNKNOWN


def test_release_requires_fresh_contained_supported_held_payload() -> None:
    runtime = VerifiedCapabilityRuntime(placement_v22_enabled=True)
    runtime.belief.frame_id = 8
    runtime.belief.grasp_epoch = 4
    runtime.belief.route_epoch = 3
    runtime.belief.target = SimpleNamespace(instance_id="target-1")
    runtime.belief.eef_xyz = EvidenceValue((0.1, 0.2, 0.3), TruthValue.TRUE, "eef", 1.0, 8)
    runtime.belief.held = EvidenceValue(True, TruthValue.TRUE, "verified_hold", 1.0, 8)
    runtime.placement_belief = PlacementBelief(
        relation=PlacementRelation.RIM_CONTACT,
        frame_id=8,
        instance_id="target-1",
        grasp_epoch=4,
        containment_margin_m=0.05,
        rim_clearance_m=0.001,
        uncertainty_m=(0.001, 0.001, 0.001),
        fresh=True,
    )
    runtime._placement_contact_verified = True
    runtime._placement_support_stability_count = 1
    runtime._placement_support_candidate = {
        "frame_id": 7, "route_epoch": 3, "grasp_epoch": 4,
        "instance_id": "target-1", "route_id": "route-1",
        "eef_xyz": (0.1, 0.2, 0.3), "containment_margin_m": 0.05,
        "rim_clearance_m": 0.001,
    }
    runtime._pending_critical_kind = "VERIFY_SEATED"
    denied = runtime.apply_critical_decision(PlacementRelation.SEATED_HELD.value)
    assert denied["next_action"] == STOP
    runtime.belief.frame_id = 9
    runtime.belief.eef_xyz = EvidenceValue((0.1005, 0.2, 0.3), TruthValue.TRUE, "eef", 1.0, 9)
    runtime.belief.held = EvidenceValue(True, TruthValue.TRUE, "verified_hold", 1.0, 9)
    runtime.placement_belief = PlacementBelief(
        relation=PlacementRelation.SEATED_HELD, frame_id=9, instance_id="target-1",
        grasp_epoch=4, containment_margin_m=0.05, rim_clearance_m=0.001,
        uncertainty_m=(0.001, 0.001, 0.001), fresh=True,
    )
    support_evidence = {
        "placement_belief": {
            "relation": "SEATED_HELD", "frame_id": 9, "grasp_epoch": 4,
            "containment_margin_m": 0.05, "rim_clearance_m": 0.001,
            "uncertainty_m": [0.001, 0.001, 0.001], "fresh": True,
        },
        "route": {"route_id": "route-1"},
    }
    runtime._update_placement_support_evidence(
        frame_id=9, previous_action=STOP, eef_z_stalled=False,
        progress={}, route_evidence=support_evidence,
    )
    assert runtime.belief.seated.truth == TruthValue.TRUE
    assert runtime._placement_release_ready()
    released = runtime._control_from_route(
        _route_evidence(PlacementRelation.SEATED_HELD.value, 9)
    )
    assert released is not None and released.action_token == "RELEASE"
    assert runtime.belief.seated.truth == TruthValue.TRUE


def test_contact_candidate_requires_an_authorized_executed_descent_receipt() -> None:
    runtime = VerifiedCapabilityRuntime(placement_v22_enabled=True)
    runtime.belief.frame_id = 2
    runtime.belief.grasp_epoch = 4
    runtime.belief.target = SimpleNamespace(instance_id="target-1")
    runtime.belief.eef_xyz = EvidenceValue((0.1, 0.2, 0.3), TruthValue.TRUE, "eef", 1.0, 2)
    runtime.belief.held = EvidenceValue(True, TruthValue.TRUE, "verified_hold", 1.0, 2)
    runtime.placement_belief = PlacementBelief(
        relation=PlacementRelation.RIM_CONTACT, frame_id=2, instance_id="target-1",
        grasp_epoch=4, containment_margin_m=0.05, rim_clearance_m=0.001,
        uncertainty_m=(0.001, 0.001, 0.001), fresh=True,
    )
    route_evidence = {
        "placement_belief": {"frame_id": 2},
        "route": {"route_id": "route-1"},
    }
    progress = {"contact_or_stall": True}
    runtime.last_event = {"action_receipt": {"executed_action": STOP, "authorized_action": STOP}}
    runtime._update_placement_support_evidence(
        frame_id=2, previous_action=MOVE_DOWN, eef_z_stalled=True,
        progress=progress, route_evidence=route_evidence,
    )
    assert not runtime._placement_contact_verified
    runtime.last_event = {"action_receipt": {"executed_action": MOVE_DOWN, "authorized_action": MOVE_DOWN}}
    runtime._update_placement_support_evidence(
        frame_id=2, previous_action=MOVE_DOWN, eef_z_stalled=True,
        progress=progress, route_evidence=route_evidence,
    )
    assert runtime._placement_contact_verified


def test_runtime_v22_above_unaligned_verifier_resumes_metric_descent_once() -> None:
    runtime = VerifiedCapabilityRuntime(
        placement_v22_enabled=True, placement_confirm_frames=1
    )
    route = _route_evidence(PlacementRelation.RIM_CONTACT.value, 1)
    runtime._control_from_route(route)
    runtime._pending_critical_kind = "VERIFY_SEATED"
    runtime.placement_belief = PlacementBelief(
        relation=PlacementRelation.ABOVE_ALIGNED, frame_id=1, grasp_epoch=4,
        containment_margin_m=0.04, rim_clearance_m=0.01,
        uncertainty_m=(0.001, 0.001, 0.001), fresh=True,
    )
    response = runtime.apply_critical_decision(
        PlacementRelation.ABOVE_ALIGNED.value,
        details={"next_step": "CONTINUE_DESCENT"},
    )
    assert response["accepted"] is True
    assert response["next_action"] == MOVE_DOWN
    first = runtime._control_from_route(_route_evidence(PlacementRelation.ABOVE_ALIGNED.value, 2))
    second = runtime._control_from_route(_route_evidence(PlacementRelation.RIM_CONTACT.value, 3))
    assert first is not None and first.action_token == MOVE_DOWN
    assert second is not None and second.action_token == STOP


def test_runtime_v22_provider_conflict_cannot_become_a_motion_candidate() -> None:
    runtime = VerifiedCapabilityRuntime(placement_v22_enabled=True, placement_confirm_frames=1)
    evidence = _route_evidence(
        PlacementRelation.ABOVE_UNALIGNED.value,
        1,
        [{"token": "MV_RIGHT", "cosine": 0.9}],
    )
    evidence["placement_belief"]["conflicts"] = ["moge2_vs_active_parallax"]
    decision = runtime._control_from_route(evidence)
    assert decision is not None and decision.action_token == STOP


def test_v22_visual_route_is_geometry_only_and_does_not_call_qwen_intent() -> None:
    class Client:
        calls = 0

        def complete_json(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("V2.2 VisualRoute must not own a placement intent")

    client = Client()
    plugin = VisualRoutePlugin(
        enabled=True,
        mode="active",
        client=client,
        placement_v22_enabled=True,
    )
    output = plugin.update(
        agentview=np.zeros((256, 256, 3), dtype=np.uint8),
        wrist=None,
        stage="TRANSPORT",
        subgoals=[],
        current_index=0,
        frame_id=0,
        eef_world=[0.06744, -0.09998, 0.11518],
        geometry=_geometry(),
        held_evidence={"bbox_xyxy": [148, 70, 167, 95], "confidence": 0.9},
        destination_evidence={
            "bbox_xyxy": [17, 103, 77, 160],
            "opening_bbox_xyxy": [30, 103, 74, 125],
            "opening_mask": _mask(),
            "confidence": 0.9,
        },
        gripper_closed=True,
    )
    assert client.calls == 0
    assert output["evidence"]["intent"] is None
    assert plugin.ready_to_release() is False
    assert plugin.metadata()["placement_authority"] == "vcr_runtime"


def test_v22_route_without_opening_mask_fails_closed() -> None:
    plugin = VisualRoutePlugin(enabled=True, mode="active", placement_v22_enabled=True)
    output = plugin.update(
        agentview=np.zeros((256, 256, 3), dtype=np.uint8),
        wrist=None,
        stage="TRANSPORT",
        subgoals=[],
        current_index=0,
        frame_id=0,
        eef_world=[0.06744, -0.09998, 0.11518],
        geometry=_geometry(),
        held_evidence={"bbox_xyxy": [148, 70, 167, 95], "confidence": 0.9},
        destination_evidence={
            "bbox_xyxy": [17, 103, 77, 160],
            "opening_bbox_xyxy": [30, 103, 74, 125],
            "confidence": 0.9,
        },
        gripper_closed=True,
    )
    assert output["evidence"]["route"]["valid"] is False
    assert output["evidence"]["placement_belief"]["relation"] == "UNKNOWN"


def test_v22_opening_mask_remains_valid_when_outer_container_touches_frame_edge() -> None:
    plugin = VisualRoutePlugin(
        enabled=True,
        mode="active",
        placement_v22_enabled=True,
    )
    output = plugin.update(
        agentview=np.zeros((256, 256, 3), dtype=np.uint8),
        wrist=None,
        stage="TRANSPORT",
        subgoals=[],
        current_index=0,
        frame_id=0,
        eef_world=[0.06744, -0.09998, 0.11518],
        geometry=_geometry(),
        held_evidence={"bbox_xyxy": [148, 70, 167, 95], "confidence": 0.9},
        destination_evidence={
            "bbox_xyxy": [0, 95, 77, 160],
            "opening_bbox_xyxy": [0, 95, 77, 160],
            "opening_mask": _mask(30, 74, 103, 125),
            "outer_mask": _mask(0, 77, 95, 160),
            "confidence": 0.9,
        },
        gripper_closed=True,
    )
    route = output["evidence"]["route"]
    assert route["valid"] is True
    assert route["opening_bbox_xyxy"] == [30, 103, 74, 125]
    assert route["destination_bbox_xyxy"] == [0, 95, 77, 160]


def test_v22_clipped_but_non_degenerate_opening_mask_remains_geometric_evidence() -> None:
    plugin = VisualRoutePlugin(
        enabled=True,
        mode="active",
        placement_v22_enabled=True,
    )
    output = plugin.update(
        agentview=np.zeros((256, 256, 3), dtype=np.uint8),
        wrist=None,
        stage="TRANSPORT",
        subgoals=[],
        current_index=0,
        frame_id=0,
        eef_world=[0.06744, -0.09998, 0.11518],
        geometry=_geometry(),
        held_evidence={"bbox_xyxy": [148, 70, 167, 95], "confidence": 0.9},
        destination_evidence={
            "bbox_xyxy": [0, 95, 62, 160],
            "opening_bbox_xyxy": [0, 103, 47, 125],
            "opening_mask": _mask(0, 47, 103, 125),
            "outer_mask": _mask(0, 62, 95, 160),
            "confidence": 0.9,
        },
        gripper_closed=True,
    )
    assert output["evidence"]["route"]["valid"] is True
    assert output["evidence"]["route"]["opening_bbox_xyxy"] == [0, 103, 47, 125]


def test_anyplace_shadow_preserves_rotation_and_never_marks_unsupported_pose_reachable() -> None:
    provider = AnyPlaceShadowProvider(enabled=True, supports_orientation=False)
    candidates = provider._parse_candidates(
        {
            "candidates": [
                {
                    "candidate_id": "pose-0",
                    "object_pose_world": [1.0, 2.0, 3.0],
                    "rotation": [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                    "reachable": True,
                }
            ]
        },
        frame_id=7,
        instance_id="object-1",
    )
    assert len(candidates) == 1
    assert candidates[0].orientation_supported is False
    assert candidates[0].reachable is False
    assert candidates[0].diagnostics["rejection"] == "UNSUPPORTED_ORIENTATION"
    assert candidates[0].diagnostics["predicted_rotation"][0][1] == -1.0


def test_anyplace_default_worker_isolated_and_fail_closed() -> None:
    provider = AnyPlaceShadowProvider(
        enabled=True,
        python=sys.executable,
        worker="core/runtime_v2/anyplace_worker.py",
        timeout_s=5.0,
    )
    result = provider.infer(
        parent_points=np.zeros((8, 3), dtype=np.float32),
        child_points=np.ones((8, 3), dtype=np.float32),
        frame_id=1,
        instance_id="x",
    )
    assert result["health"] == "UNKNOWN"
    assert result["candidates"] == []
    assert result["reason"] == "worker_returned_no_candidates"

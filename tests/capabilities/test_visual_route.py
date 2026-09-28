import numpy as np

from core.capabilities.camera_geometry import (
    CameraCalibration,
    estimate_vertical_line_height,
)
from core.v0_types import Subgoal
from plugins.visual_route import (
    RoutePhase,
    RoutePlan,
    RouteReview,
    TransportIntent,
    VisualRoutePlugin,
)


def _calibration():
    angle = np.deg2rad(30.0)
    camera_to_world = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    return CameraCalibration(
        name="agentview",
        width=256,
        height=256,
        fovy_deg=60.0,
        position_world=np.array([0.0, 0.0, 1.0]),
        camera_to_world=camera_to_world,
    )


def _route(phase=RoutePhase.CLEARANCE):
    return RoutePlan(
        route_id="route-test",
        phase=phase,
        waypoints_world=[[0.0, 0.0, 0.30], [0.1, 0.1, 0.30], [0.12, 0.12, 0.30]],
        waypoints_px=[[50, 50], [100, 100], [120, 120]],
        safe_transport_z_m=0.30,
        destination_xy_world=[0.12, 0.12],
        confidence=0.9,
        valid=True,
    )


def _geometry_meta(calibration):
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


def _libero_geometry():
    calibration = CameraCalibration(
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
    return _geometry_meta(calibration)


class _Response:
    def __init__(self, payload):
        self.payload = {"json": payload, "latency_s": 0.01}
        self.raw_text = str(payload)


class _IntentClient:
    def __init__(self, *, held="HELD", intent="TRANSFER", route="VALID"):
        self.held = held
        self.intent = intent
        self.route = route
        self.calls = 0

    def complete_json(self, *args, **kwargs):
        self.calls += 1
        expected = {
            "CLEAR": "MORE_CLEARANCE",
            "TRANSFER": "MORE_ROUTE_PROGRESS",
            "ALIGN": "LESS_OPENING_ERROR",
            "DESCEND": "LOWER_STABLE",
            "RECOVER_CLEAR": "LESS_CONTACT",
            "REACQUIRE": "REACQUIRE_HOLD",
            "READY_TO_RELEASE": "SEATED",
        }[self.intent]
        return _Response(
            {
                "intent": self.intent,
                "route_assessment": self.route,
                "held_assessment": self.held,
                "expected_change": expected,
                "confidence": "HIGH",
                "reasoning": "fresh visual relation supports this short plan",
            }
        )


def test_vertical_line_height_uses_camera_ray_and_known_xy():
    calibration = _calibration()
    estimate = estimate_vertical_line_height(calibration, (128, 128), (0.0, 0.2))
    assert estimate is not None
    assert abs(estimate["height_m"] - 0.6535898) < 1e-3
    assert estimate["residual_m"] < 1e-6


def test_route_gate_uses_geometry_and_current_risk_not_phase_alone():
    plugin = VisualRoutePlugin(enabled=True, mode="active")
    plugin.route = _route(RoutePhase.CLEARANCE)
    decision = plugin.gate("DONE", stage="LIFT", eef_world=[0.0, 0.0, 0.1])
    assert decision.executed_token == "DONE"
    assert decision.allowed

    decision = plugin.gate("DONE", stage="MOVE", held_evidence={
        "known": True,
        "aligned": False,
    })
    assert decision.executed_token == "DONE"
    assert decision.allowed

    plugin.route.phase = RoutePhase.CLEARANCE
    plugin.last_review = RouteReview(verdict="NO", issue="LOW_CLEARANCE")
    decision = plugin.gate("DONE", stage="LIFT", eef_world=[0.0, 0.0, 0.4])
    assert decision.executed_token == "DONE"
    assert decision.allowed
    assert decision.executed_token == decision.requested_token

    plugin.route.phase = RoutePhase.TRANSFER
    decision = plugin.gate("MV_DOWN", stage="MOVE", held_evidence={
        "known": False,
        "destination_minus_held_center_px": [18, 2],
        "correction_candidates": {"horizontal": "MV_LEFT", "vertical": "MV_FWD"},
    })
    assert decision.executed_token == "MV_DOWN"
    assert decision.allowed

    decision = plugin.gate("MV_DOWN", stage="PLACE", held_evidence={
        "known": True,
        "aligned": False,
        "destination_minus_held_center_px": [18, 2],
        "correction_candidates": {"horizontal": "MV_LEFT", "vertical": "MV_FWD"},
    })
    assert decision.executed_token == "MV_DOWN"
    assert decision.allowed
    assert "not_aligned" in decision.reason

    plugin.route.phase = RoutePhase.RECOVER_CLEAR
    decision = plugin.gate("MV_DOWN", stage="PLACE", held_evidence={"known": False})
    assert decision.executed_token == "MV_DOWN"
    assert decision.allowed

    decision = plugin.gate("MV_DOWN", stage="PLACE", held_evidence={
        "known": True,
        "aligned": True,
        "rim_contact_risk": True,
        "clearance_progress_ready": False,
    })
    assert decision.executed_token == "MV_DOWN"
    assert decision.allowed
    assert "rim_contact_risk" in decision.reason


def test_low_clearance_review_cannot_latch_route_phase():
    plugin = VisualRoutePlugin(enabled=True, mode="active")
    plugin.route = _route(RoutePhase.CLEARANCE)
    plugin.last_review = RouteReview(verdict="NO", issue="LOW_CLEARANCE")

    # A stale/incorrect review is evidence, not a host-written phase lock.
    plugin._update_phase(
        "LIFT",
        [0.0, 0.0, 0.5],
        held={},
        destination={},
        previous_action="DONE",
    )
    assert plugin.route.phase == RoutePhase.TRANSFER
    assert plugin.last_review.issue == "LOW_CLEARANCE"

    # The old review type remains readable in archived telemetry, but it is not
    # consulted by the adaptive intent loop.
    assert plugin.last_review.issue == "LOW_CLEARANCE"


def test_alignment_scales_with_detected_opening_not_fixed_pixel_offset():
    plugin = VisualRoutePlugin(enabled=True, mode="active")
    held = {"bbox_xyxy": [90, 90, 110, 130]}
    large_opening = {"opening_bbox_xyxy": [70, 70, 150, 150]}
    small_opening = {"opening_bbox_xyxy": [100, 90, 120, 120]}

    large = plugin._alignment_signal(held, large_opening)
    small = plugin._alignment_signal(held, small_opening)
    assert large["source"] == "opening_relative_geometry"
    assert large["aligned"]
    assert not small["aligned"]
    assert large["limits_px"] != small["limits_px"]


def test_missing_opening_does_not_fabricate_a_route_from_outer_box():
    calibration = _calibration()
    plugin = VisualRoutePlugin(enabled=True, mode="active")
    output = plugin.update(
        agentview=np.zeros((256, 256, 3), dtype=np.uint8),
        wrist=None,
        stage="TRANSPORT",
        subgoals=[],
        current_index=0,
        frame_id=0,
        eef_world=[0.0, 0.0, 0.3],
        geometry=_geometry_meta(calibration),
        held_evidence={"bbox_xyxy": [70, 80, 90, 150], "confidence": 0.9},
        destination_evidence={"bbox_xyxy": [150, 100, 220, 200], "confidence": 0.9},
        gripper_closed=True,
    )
    assert output["evidence"]["route"]["valid"] is False
    assert output["evidence"]["route"]["last_reason"] == "incomplete_visual_route_evidence"


def test_route_is_inert_during_approach_even_if_a_stale_plan_exists():
    plugin = VisualRoutePlugin(enabled=True, mode="active")
    plugin.route = _route(RoutePhase.RECOVER_CLEAR)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    output = plugin.update(
        agentview=frame,
        wrist=None,
        stage="APPROACH",
        subgoals=[],
        current_index=0,
        frame_id=4,
        eef_world=[0.0, 0.0, 0.2],
        geometry={},
        held_evidence=None,
        destination_evidence=None,
        gripper_closed=False,
    )
    assert output["agentview"] is frame
    assert output["context"] == ""
    assert output["evidence"]["route"] is None
    assert output["evidence"]["route_discarded"] == "stage_outside_transport"


def test_renderer_preserves_raw_frame_and_draws_route_overlay():
    plugin = VisualRoutePlugin(enabled=True, mode="shadow")
    plugin.route = _route(RoutePhase.TRANSFER)
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    rendered = plugin.render(
        frame,
        held_evidence={"bbox_xyxy": [40, 40, 65, 90]},
        destination_evidence={
            "bbox_xyxy": [150, 140, 220, 220],
            "opening_bbox_xyxy": [165, 155, 205, 195],
        },
    )
    assert np.array_equal(frame, np.zeros_like(frame))
    assert rendered.shape == frame.shape
    assert np.any(rendered != frame)
    # The new overlay has no red/orange receptacle boxes or text/legend.
    red_dominant = (
        (rendered[..., 0] > 180)
        & (rendered[..., 0] > rendered[..., 1] * 1.5)
        & (rendered[..., 0] > rendered[..., 2] * 1.5)
    )
    assert not np.any(red_dominant)


def test_safe_transport_height_is_fixed_for_a_grasp_epoch():
    plugin = VisualRoutePlugin(enabled=True, mode="active", client=None)
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    held = {"bbox_xyxy": [148, 127, 167, 162], "confidence": 0.8}
    destination = {
        "bbox_xyxy": [17, 103, 77, 160],
        "opening_bbox_xyxy": [30, 103, 74, 125],
        "confidence": 0.8,
    }
    first = plugin.update(
        agentview=frame,
        wrist=None,
        stage="TRANSPORT",
        subgoals=[],
        current_index=0,
        frame_id=0,
        eef_world=[0.06744, -0.09998, 0.11518],
        geometry=_libero_geometry(),
        held_evidence=held,
        destination_evidence=destination,
        gripper_closed=True,
    )
    safe_z = first["evidence"]["route"]["safe_transport_z_m"]
    assert first["evidence"]["route"]["valid"]
    assert safe_z > 0.11518

    # A periodic refresh at a much higher EEF must not turn current_z into a
    # moving clearance target.
    plugin._replace_route(
        frame_id=8,
        eef_world=[0.06744, -0.09998, 0.40],
        geometry=_libero_geometry(),
        held_evidence=held,
        destination_evidence=destination,
        reason="periodic_8_frames",
    )
    assert plugin.route.safe_transport_z_m == safe_z
    assert plugin.route.payload_below_eef_m == first["evidence"]["route"]["payload_below_eef_m"]


def test_transport_collapse_removes_static_lift_move_place_conflict():
    plan = [
        Subgoal("a", "object", "body", "APPROACH", "approach", "aligned"),
        Subgoal("g", "object", "body", "GRASP", "grasp", "held"),
        Subgoal("l", "object", "body", "LIFT", "lift", "clear"),
        Subgoal("m", "container", "opening", "MOVE", "move", "above"),
        Subgoal("p", "container", "opening", "PLACE", "place", "seated"),
        Subgoal("r", "container", "opening", "RELEASE", "release", "open"),
    ]
    collapsed = VisualRoutePlugin.collapse_transport_subgoals(plan)
    assert [item.motion for item in collapsed] == [
        "APPROACH", "GRASP", "TRANSPORT", "RELEASE"
    ]
    assert collapsed[2].target == "container"


def test_qwen_not_held_is_rejected_by_latched_visual_evidence():
    client = _IntentClient(held="LOST", intent="REACQUIRE")
    plugin = VisualRoutePlugin(enabled=True, mode="active", client=client)
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    output = plugin.update(
        agentview=frame,
        wrist=frame,
        stage="TRANSPORT",
        subgoals=[],
        current_index=0,
        frame_id=0,
        eef_world=[0.06744, -0.09998, 0.11518],
        geometry=_libero_geometry(),
        held_evidence={"bbox_xyxy": [148, 127, 167, 162], "confidence": 0.8},
        destination_evidence={
            "bbox_xyxy": [17, 103, 77, 160],
            "opening_bbox_xyxy": [30, 103, 74, 125],
            "confidence": 0.8,
        },
        gripper_closed=True,
    )
    assert output["evidence"]["holding_arbiter"]["state"] == "HELD"
    assert output["evidence"]["intent"]["held_assessment"] == "LOST"
    assert not output["evidence"]["intent"]["held_assessment_accepted"]


def test_object_eef_separation_triggers_qwen_loss_review():
    client = _IntentClient(held="LOST", intent="REACQUIRE")
    plugin = VisualRoutePlugin(enabled=True, mode="active", client=client)
    plugin._held_latched = True
    plugin._current_eef_px = [100.0, 100.0]
    held = {"bbox_xyxy": [90, 105, 110, 145], "confidence": 0.9}
    plugin._update_holding(held, True, 1.0)
    assert plugin._holding_arbiter["state"] == "HELD"

    # The tracked object remains behind while the hand moves. One frame is only
    # suspicion; two consecutive relation violations are review evidence.
    plugin._current_eef_px = [125.0, 100.0]
    plugin._update_holding(held, True, 0.2)
    assert plugin._holding_arbiter["state"] == "HELD"
    plugin._current_eef_px = [130.0, 100.0]
    plugin._update_holding(held, True, 0.2)
    assert plugin._holding_arbiter["state"] == "SUSPECTED_LOST"
    assert plugin._holding_arbiter["separation_frames"] == 2


def test_geometry_and_qwen_intent_refresh_counters_are_independent():
    client = _IntentClient()
    plugin = VisualRoutePlugin(
        enabled=True,
        mode="active",
        client=client,
        replan_interval=8,
        intent_interval=6,
    )
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    kwargs = dict(
        agentview=frame,
        wrist=frame,
        stage="TRANSPORT",
        subgoals=[],
        current_index=0,
        eef_world=[0.06744, -0.09998, 0.11518],
        geometry=_libero_geometry(),
        held_evidence={"bbox_xyxy": [148, 127, 167, 162], "confidence": 0.8},
        destination_evidence={"bbox_xyxy": [17, 103, 77, 160], "opening_bbox_xyxy": [30, 103, 74, 125]},
        gripper_closed=True,
    )
    first = plugin.update(frame_id=0, **kwargs)
    assert first["evidence"]["geometry_refresh_count"] == 1
    # The fixed mock proposes TRANSFER while calibrated clearance is still
    # positive, so the prediction-error loop performs one immediate rethink.
    # That extra semantic call must not refresh CPU geometry.
    assert first["evidence"]["intent_refresh_count"] == 2
    assert client.calls == 2
    for frame_id in range(1, 7):
        output = plugin.update(frame_id=frame_id, previous_action="MV_UP", **kwargs)
    assert output["evidence"]["intent_refresh_count"] >= 2
    assert output["evidence"]["geometry_refresh_count"] == 1


def test_clearance_route_candidate_comes_from_configured_action_vectors():
    plugin = VisualRoutePlugin(
        enabled=True,
        mode="active",
        move_vectors={
            "MV_UP": [0.0, 0.0, 1.0],
            "MV_DOWN": [0.0, 0.0, -1.0],
            "MV_LEFT": [0.0, 1.0, 0.0],
        },
    )
    plugin.route = _route(RoutePhase.CLEARANCE)
    progress = plugin._compute_progress(
        eef_world=[0.0, 0.0, 0.20],
        held={"bbox_xyxy": [40, 40, 60, 80]},
        destination={"opening_bbox_xyxy": [100, 100, 140, 130]},
        previous_action=None,
    )
    assert progress.active_waypoint_residual_world_m == [0.0, 0.0, 0.1]
    assert [item["token"] for item in progress.route_direction_candidates] == [
        "MV_UP"
    ]


def test_disabled_plugin_is_identity():
    plugin = VisualRoutePlugin(enabled=False, mode="active")
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    output = plugin.update(
        agentview=frame,
        wrist=None,
        stage="LIFT",
        subgoals=[],
        current_index=0,
        frame_id=0,
        eef_world=[0.0, 0.0, 0.2],
        geometry={},
        held_evidence=None,
        destination_evidence=None,
        gripper_closed=True,
    )
    assert output["agentview"] is frame
    assert output["context"] == ""
    assert output["evidence"] == {}

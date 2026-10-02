from __future__ import annotations

import numpy as np

from core.agent.stage_control import Controller

from core.runtime_v2 import (
    EvidenceValue,
    EntityTrack,
    FailureCode,
    ObservationHealth,
    OptionName,
    TruthValue,
    VerifiedCapabilityRuntime,
)
from core.runtime_v2.runtime import RuntimeLimits
from core.runtime_v2.control import MOVE_UP, ResidualController
from core.runtime_v2.tracking import EntityTracker, candidate_dicts


def _evidence(
    frame: int,
    bbox=(113, 122, 130, 156),
    *,
    dx=20.0,
    dy=4.0,
    source="sam3",
    tool=None,
):
    return {
        "frame_id": frame,
        "camera": "agentview",
        "stage": "APPROACH",
        "target": "test target",
        "bbox_xyxy": list(bbox) if bbox is not None else None,
        "confidence": 0.8 if bbox is not None else 0.0,
        "source": source,
        "visible": bbox is not None,
        "tool": tool or {},
        "geometry": {
            "target_minus_eef_px": [dx, dy],
            "calibrated_correction_candidates": {
                "horizontal": "MV_RIGHT",
                "vertical": "MV_FWD",
            },
        },
    }


def test_stage_controller_exposes_pregrasp_resolver() -> None:
    class Agent:
        def resolve_pregrasp(self, **kwargs):
            return {"selected": kwargs["choice"]}

    controller = Controller(Agent())
    assert controller.resolve_pregrasp(choice="MV_BACK") == {
        "selected": "MV_BACK"
    }


def test_secondary_grasp_association_is_same_camera_and_read_only_until_commit() -> None:
    runtime = VerifiedCapabilityRuntime(semantic_pregrasp_enabled=True)
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    seeded = runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(1, bbox=(113, 122, 130, 156)),
        previous_action=None,
        agentview=image,
        wrist=image,
        eef_xyz=(0.0, 0.0, 0.2),
    )
    original_id = runtime.belief.target.instance_id
    original_frame = runtime.belief.target.last_confirmed_frame
    secondary = {
        "camera": "agentview",
        "source": "sam3_abstain",
        "visible": False,
        "tool": {
            "candidates": [
                {"bbox_xyxy": [113, 122, 130, 156], "score": 0.71},
                {"bbox_xyxy": [20, 20, 38, 48], "score": 0.70},
            ]
        },
    }

    preview = runtime.associate_secondary_target_view(
        evidence={**secondary, "stage": "GRASP"},
        image=image,
        camera="agentview",
        frame_id=2,
        commit=False,
    )
    assert preview["health"] == "VALID"
    assert preview["bbox_xyxy"] == [113.0, 122.0, 130.0, 156.0]
    assert runtime.belief.target.instance_id == original_id
    assert runtime.belief.target.last_confirmed_frame == original_frame

    wrong_camera = runtime.associate_secondary_target_view(
        evidence={**secondary, "stage": "GRASP"},
        image=image,
        camera="wrist",
        frame_id=2,
        commit=False,
    )
    assert wrong_camera["health"] == "UNKNOWN"
    assert wrong_camera["reason"] == "no_existing_target_identity"

    committed = runtime.associate_secondary_target_view(
        evidence={**secondary, "stage": "GRASP"},
        image=image,
        camera="agentview",
        frame_id=2,
        commit=True,
    )
    assert committed["committed"] is True
    assert runtime.belief.target.instance_id == original_id
    assert runtime.belief.target.camera == "agentview"
    assert runtime.belief.target.last_confirmed_frame == 2


def test_occluded_primary_can_refresh_same_instance_from_secondary_without_becoming_valid() -> None:
    runtime = VerifiedCapabilityRuntime(semantic_pregrasp_enabled=True)
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(1, bbox=(113, 122, 130, 156)),
        previous_action=None,
        agentview=image,
        wrist=image,
        eef_xyz=(0.0, 0.0, 0.2),
    )
    target_id = runtime.belief.target.instance_id
    wrist_evidence = {
        "frame_id": 2,
        "camera": "wrist",
        "stage": "GRASP",
        "target": "target",
        "bbox_xyxy": None,
        "confidence": 0.0,
        "source": "sam3_abstain",
        "visible": False,
        "tool": {},
        "secondary_view": {
            "camera": "agentview",
            "stage": "GRASP",
            "source": "sam3_abstain",
            "bbox_xyxy": [113, 122, 130, 156],
            "confidence": 0.71,
            "visible": True,
            "tool": {
                "candidates": [
                    {"bbox_xyxy": [113, 122, 130, 156], "score": 0.71},
                    {"bbox_xyxy": [20, 20, 38, 48], "score": 0.70},
                ]
            },
        },
    }
    result = runtime.observe_frame(
        stage="GRASP",
        evidence=wrist_evidence,
        previous_action=None,
        agentview=image,
        wrist=image,
        eef_xyz=(0.0, 0.0, 0.2),
    )
    assert result["observation_health"] == ObservationHealth.OCCLUDED.value
    assert runtime.belief.target.instance_id == target_id
    assert runtime.belief.target.last_confirmed_frame == 2
    assert runtime.last_event["evidence"]["secondary_view"]["instance_association"]["health"] == "VALID"


def test_failed_grasp_reobserve_gets_one_new_frame_probe_decision_then_stops() -> None:
    runtime = VerifiedCapabilityRuntime(
        semantic_pregrasp_enabled=True,
        require_spatial_ready_for_grasp=True,
        pregrasp_reflection_mode="double",
        approach_min_height_m=0.10,
        approach_max_height_m=0.14,
    )
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    target_box = [113, 122, 130, 156]
    runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(1, bbox=tuple(target_box)),
        previous_action=None,
        agentview=image,
        wrist=image,
        image_refs={"agentview": "images/raw_agentview/0001.png", "wrist": "images/raw_wrist/0001.png"},
        eef_xyz=(0.0, 0.0, 0.135),
    )

    def occluded_frame(frame: int, z: float, target_runtime=None) -> dict:
        target_runtime = target_runtime or runtime
        return target_runtime.observe_frame(
            stage="GRASP",
            evidence={
                "frame_id": frame,
                "camera": "wrist",
                "stage": "GRASP",
                "target": "target",
                "bbox_xyxy": None,
                "confidence": 0.0,
                "source": "sam3_abstain",
                "visible": False,
                "tool": {},
                "secondary_view": {
                    "camera": "agentview",
                    "stage": "GRASP",
                    "source": "sam3_abstain",
                    "bbox_xyxy": target_box,
                    "confidence": 0.8,
                    "visible": True,
                    "tool": {
                        "candidates": [
                            {"bbox_xyxy": target_box, "score": 0.8},
                            {"bbox_xyxy": [20, 20, 38, 48], "score": 0.79},
                        ]
                    },
                },
            },
            previous_action=None,
            agentview=image,
            wrist=image,
            image_refs={
                "agentview": f"images/raw_agentview/{frame:04d}.png",
                "wrist": f"images/raw_wrist/{frame:04d}.png",
            },
            eef_xyz=(0.0, 0.0, z),
        )

    first = occluded_frame(2, 0.135)
    assert first["critical_decision"]["allowed_answers"] == ["REOBSERVE", "UNKNOWN"]
    lifted = runtime.apply_critical_decision("REOBSERVE")
    assert lifted["next_action"] == MOVE_UP
    runtime.commit_executed_action(executed_action=MOVE_UP, authorized_action=MOVE_UP)

    # A real bounded REOBSERVE lift can cross the configured approach upper
    # band. The follow-up gate must accept the action's authorized envelope.
    after_lift = occluded_frame(3, 0.145)
    followup = after_lift["critical_decision"]
    assert followup is not None
    assert followup["reflection_trigger"] == "grasp_reobserve_outcome"
    assert followup["reflection_mode"] == "double"
    assert followup["allowed_answers"] == ["PROBE_DEPTH", "UNKNOWN"]
    assert "REOBSERVE" not in followup["allowed_answers"]
    assert followup["evidence_frame_ids"][-1] == 3

    probed = runtime.apply_critical_decision("PROBE_DEPTH")
    assert probed["next_action"] == runtime.axis_actions["probe_depth"]
    probe_action = probed["next_action"]
    runtime.commit_executed_action(executed_action=probe_action, authorized_action=probe_action)

    exhausted = occluded_frame(4, 0.145)
    assert exhausted["critical_decision"] is None
    assert exhausted["action_token"] == "STOP"

    # A measured pose beyond the lift envelope still cannot trigger Qwen or
    # authorize a follow-up probe.
    out_of_envelope = VerifiedCapabilityRuntime(
        semantic_pregrasp_enabled=True,
        require_spatial_ready_for_grasp=True,
        pregrasp_reflection_mode="double",
        approach_min_height_m=0.10,
        approach_max_height_m=0.14,
    )
    out_of_envelope.observe_frame(
        stage="APPROACH",
        evidence=_evidence(1, bbox=tuple(target_box)),
        previous_action=None,
        agentview=image,
        wrist=image,
        image_refs={"agentview": "images/raw_agentview/0001.png", "wrist": "images/raw_wrist/0001.png"},
        eef_xyz=(0.0, 0.0, 0.135),
    )
    review = occluded_frame(2, 0.135, out_of_envelope)
    assert review["critical_decision"] is not None
    assert out_of_envelope.apply_critical_decision("REOBSERVE")["next_action"] == MOVE_UP
    out_of_envelope.commit_executed_action(executed_action=MOVE_UP, authorized_action=MOVE_UP)
    overshot = occluded_frame(3, 0.16, out_of_envelope)
    assert overshot["critical_decision"] is None
    assert overshot["action_token"] == "STOP"


def test_historical_step_161_rejects_high_score_identity_switch() -> None:
    tracker = EntityTracker(role="target")
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    first = tracker.update(
        evidence=_evidence(157),
        image=image,
        frame_id=157,
        semantic_label="green-capped bottle",
    )
    original_id = first.track.instance_id

    replay = _evidence(
        161,
        bbox=(181, 96, 199, 134),
        tool={
            "candidates": [
                {"score": 0.9023, "bbox_xyxy": [181, 96, 199, 134]},
                {"score": 0.5625, "bbox_xyxy": [105, 122, 116, 132]},
                {"score": 0.4570, "bbox_xyxy": [142, 100, 157, 125]},
                {"score": 0.4473, "bbox_xyxy": [113, 122, 130, 156]},
            ]
        },
    )
    result = tracker.update(
        evidence=replay,
        image=image,
        frame_id=161,
        semantic_label="green-capped bottle",
    )

    assert result.health == ObservationHealth.VALID
    assert result.track.instance_id == original_id
    assert result.track.bbox_xyxy == (113.0, 122.0, 130.0, 156.0)
    assert result.track.source == "runtime_v2_instance_associated"


def test_close_candidates_abstain_without_changing_identity() -> None:
    tracker = EntityTracker(role="target", ambiguity_margin=0.25)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    seeded = tracker.update(
        evidence=_evidence(1, bbox=(20, 20, 40, 50)),
        image=image,
        frame_id=1,
        semantic_label="object",
    )
    old_bbox = seeded.track.bbox_xyxy
    result = tracker.update(
        evidence=_evidence(
            2,
            bbox=(21, 20, 41, 50),
            tool={
                "candidates": [
                    {"score": 0.8, "bbox_xyxy": [21, 20, 41, 50]},
                    {"score": 0.79, "bbox_xyxy": [19, 20, 39, 50]},
                ]
            },
        ),
        image=image,
        frame_id=2,
        semantic_label="object",
    )
    assert result.health == ObservationHealth.AMBIGUOUS
    assert tracker.track.bbox_xyxy == old_bbox


def test_camera_handoff_requires_candidate_resolution_and_preserves_id() -> None:
    runtime = VerifiedCapabilityRuntime()
    seeded = runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(1),
        previous_action=None,
        agentview=np.zeros((256, 256, 3), dtype=np.uint8),
    )
    identity = seeded["belief"]["target"]["instance_id"]
    wrist = _evidence(
        2,
        bbox=(0, 102, 56, 169),
        tool={
            "candidates": [
                {"score": 0.65, "bbox_xyxy": [0, 102, 56, 169]},
                {"score": 0.58, "bbox_xyxy": [103, 41, 159, 82]},
            ]
        },
    )
    wrist["camera"] = "wrist"
    ambiguous = runtime.observe_frame(
        stage="GRASP",
        evidence=wrist,
        previous_action="DONE",
        agentview=np.zeros((256, 256, 3), dtype=np.uint8),
        wrist=np.zeros((256, 256, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.12),
    )
    assert ambiguous["critical_decision"]["kind"] == "SELECT_INSTANCE"
    # Candidate IDs use deterministic top-to-bottom spatial ordering, not the
    # detector's fluctuating confidence order.
    chosen = runtime.apply_critical_decision("candidate-0")
    assert chosen["instance_id"] == identity
    assert runtime.belief.target.camera == "wrist"
    assert runtime.belief.target.bbox_xyxy == (103.0, 41.0, 159.0, 82.0)


def test_candidate_ids_never_mix_primary_and_secondary_camera_coordinates() -> None:
    tool = {
        "probe_camera": "wrist",
        "selected_camera": "wrist",
        "probe": {
            "candidates": [
                {"score": 0.69, "bbox_xyxy": [100, 41, 161, 76]},
                {"score": 0.53, "bbox_xyxy": [0, 100, 49, 166]},
            ]
        },
        "secondary_camera": "agentview",
        "secondary": {
            "candidates": [
                {"score": 0.88, "bbox_xyxy": [148, 131, 167, 162]},
                {"score": 0.72, "bbox_xyxy": [183, 95, 199, 134]},
            ]
        },
    }
    wrist = candidate_dicts(tool, camera="wrist")
    assert [item["bbox_xyxy"] for item in wrist] == [
        [100, 41, 161, 76],
        [0, 100, 49, 166],
    ]
    assert all(item["camera"] == "wrist" for item in wrist)


def test_sensor_fault_fails_closed_and_becomes_typed_failure() -> None:
    runtime = VerifiedCapabilityRuntime(sensor_fault_limit=3)
    broken = _evidence(
        1,
        bbox=None,
        source="sam3_abstain",
        tool={"error": "sam3_client: Broken pipe"},
    )
    results = []
    for frame in range(1, 4):
        broken["frame_id"] = frame
        results.append(
            runtime.observe_frame(
                stage="APPROACH",
                evidence=broken,
                previous_action=None,
                agentview=np.zeros((64, 64, 3), dtype=np.uint8),
            )
        )
    assert all(item["action_token"] == "STOP" for item in results)
    assert results[-1]["status"] == "FAILED"
    assert results[-1]["failure"]["code"] == "SENSOR_FAULT"


def test_runtime_controller_learns_effect_and_stops_no_progress() -> None:
    control = ResidualController(no_progress_limit=3)
    first = control.decide(
        context="ALIGN_PREGRASP",
        error=(30.0, 2.0),
        horizontal_prior="MV_RIGHT",
        depth_prior="MV_FWD",
        previous_action=None,
        tolerance_px=10.0,
    )
    assert first.action_token == "MV_RIGHT"

    action = first.action_token
    decision = None
    for _ in range(3):
        decision = control.decide(
            context="ALIGN_PREGRASP",
            error=(30.0, 2.0),
            horizontal_prior="MV_RIGHT",
            depth_prior="MV_FWD",
            previous_action=action,
            tolerance_px=10.0,
        )
        action = decision.action_token
    assert decision.action_token == "STOP"
    assert decision.failure == "NO_PROGRESS"
    control.start_recovery("ALIGN_PREGRASP")
    alternative = control.decide(
        context="RELOCALIZE",
        error=(30.0, 2.0),
        horizontal_prior="MV_RIGHT",
        depth_prior="MV_FWD",
        previous_action="STOP",
        tolerance_px=10.0,
    )
    assert alternative.action_token == "MV_LEFT"
    assert "opposite action hypothesis" in alternative.reason


def test_single_outlier_cannot_overwrite_calibrated_action_prior() -> None:
    control = ResidualController()
    first = control.decide(
        context="MOVE_TO_HOVER",
        error=(5.0, 70.0),
        horizontal_prior="MV_RIGHT",
        depth_prior="MV_FWD",
        previous_action=None,
        tolerance_px=10.0,
    )
    assert first.action_token == "MV_FWD"
    outlier = control.decide(
        context="MOVE_TO_HOVER",
        error=(5.0, 75.0),
        horizontal_prior="MV_RIGHT",
        depth_prior="MV_FWD",
        previous_action="MV_FWD",
        tolerance_px=10.0,
    )
    assert outlier.action_token == "MV_FWD"
    recovered = control.decide(
        context="MOVE_TO_HOVER",
        error=(5.0, 74.0),
        horizontal_prior="MV_RIGHT",
        depth_prior="MV_FWD",
        previous_action="MV_FWD",
        tolerance_px=10.0,
    )
    assert recovered.action_token == "MV_FWD"


def test_axis_hysteresis_prevents_stepwise_cross_axis_chatter() -> None:
    control = ResidualController(axis_hold_steps=3)
    first = control.decide(
        context="MOVE_TO_HOVER",
        error=(50.0, 49.0),
        horizontal_prior="MV_RIGHT",
        depth_prior="MV_FWD",
        previous_action=None,
        tolerance_px=8.0,
    )
    assert first.action_token == "MV_RIGHT"

    # The other residual becomes slightly larger, but a single observation
    # must not make the controller swap axes and produce visible chatter.
    second = control.decide(
        context="MOVE_TO_HOVER",
        error=(48.0, 49.5),
        horizontal_prior="MV_RIGHT",
        depth_prior="MV_FWD",
        previous_action="MV_RIGHT",
        tolerance_px=8.0,
    )
    third = control.decide(
        context="MOVE_TO_HOVER",
        error=(46.0, 50.0),
        horizontal_prior="MV_RIGHT",
        depth_prior="MV_FWD",
        previous_action="MV_RIGHT",
        tolerance_px=8.0,
    )
    assert second.action_token == "MV_RIGHT"
    assert third.action_token == "MV_RIGHT"


def test_action_effect_progress_ignores_orthogonal_residual_growth() -> None:
    control = ResidualController(axis_hold_steps=3)
    first = control.decide(
        context="ALIGN_PREGRASP",
        error=(30.0, 20.0),
        horizontal_prior="MV_RIGHT",
        depth_prior="MV_FWD",
        previous_action=None,
        tolerance_px=8.0,
    )
    assert first.action_token == "MV_RIGHT"
    decision = control.decide(
        context="ALIGN_PREGRASP",
        error=(28.0, 40.0),
        horizontal_prior="MV_RIGHT",
        depth_prior="MV_FWD",
        previous_action="MV_RIGHT",
        tolerance_px=8.0,
    )
    assert decision.action_token == "MV_RIGHT"
    assert control.snapshot()["transition"]["status"] == "IMPROVING"
    assert control.snapshot()["no_progress_count"] == 0


def test_repeated_wrong_external_action_is_removed_from_bounded_choices() -> None:
    control = ResidualController(wrong_direction_limit=2)
    control.prime_external_action(
        context="QWEN_PREGRASP", error=(2.0, 10.0), action="MV_LEFT"
    )
    control.observe_transition(
        context="QWEN_PREGRASP", error=(6.0, 10.0), previous_action="MV_LEFT"
    )
    control.prime_external_action(
        context="QWEN_PREGRASP", error=(6.0, 10.0), action="MV_LEFT"
    )
    control.observe_transition(
        context="QWEN_PREGRASP", error=(10.0, 10.0), previous_action="MV_LEFT"
    )
    assert control.contradicted_actions("QWEN_PREGRASP") == ("MV_LEFT",)


def test_semantic_depth_probe_cannot_repeat_a_contradicted_direction() -> None:
    runtime = VerifiedCapabilityRuntime(semantic_pregrasp_enabled=True)
    runtime._pending_critical_kind = "PREGRASP_DECISION"
    runtime._pending_allowed_answers = (
        "PROBE_DEPTH", "CORRECT_DEPTH", "UNKNOWN"
    )
    runtime._pending_control_error = (0.0, 12.0)
    runtime.controller.effect_direction_streaks[("QWEN_PREGRASP", "MV_BACK")] = (-1, 2)

    decision = runtime.apply_critical_decision("PROBE_DEPTH")

    assert decision["next_action"] == "MV_FWD"
    assert "wrong-way effect" in decision["reason"]
    assert runtime.controller.pending_external["requested_action"] == "MV_FWD"


def test_semantic_depth_probe_stops_after_both_directions_fail() -> None:
    runtime = VerifiedCapabilityRuntime(semantic_pregrasp_enabled=True)
    runtime._pending_critical_kind = "PREGRASP_DECISION"
    runtime._pending_allowed_answers = ("PROBE_DEPTH", "UNKNOWN")
    runtime._pending_control_error = (0.0, 12.0)
    runtime.controller.effect_direction_streaks[("QWEN_PREGRASP", "MV_BACK")] = (-1, 2)
    runtime.controller.effect_direction_streaks[("QWEN_PREGRASP", "MV_FWD")] = (-1, 2)

    decision = runtime.apply_critical_decision("PROBE_DEPTH")

    assert decision["next_action"] == "STOP"
    assert "no uncontradicted reverse remains" in decision["reason"]
    assert runtime.controller.pending_external is None


def test_semantic_depth_probe_reverses_once_after_observed_wrong_way_effect() -> None:
    runtime = VerifiedCapabilityRuntime(semantic_pregrasp_enabled=True)
    runtime._pending_critical_kind = "PREGRASP_DECISION"
    runtime._pending_allowed_answers = ("PROBE_DEPTH", "UNKNOWN")
    runtime._pending_control_error = (0.0, 12.0)
    runtime.controller.last_transition = {
        "status": "WRONG_DIRECTION",
        "action": "MV_BACK",
        "improvement_px": -3.0,
    }

    decision = runtime.apply_critical_decision("PROBE_DEPTH")

    assert decision["next_action"] == "MV_FWD"
    assert "testing the opposite direction once" in decision["reason"]

    runtime.controller.last_transition = {
        "status": "WRONG_DIRECTION",
        "action": "MV_FWD",
        "improvement_px": -2.0,
    }
    runtime._pending_critical_kind = "PREGRASP_DECISION"
    runtime._pending_allowed_answers = ("PROBE_DEPTH", "UNKNOWN")
    stopped = runtime.apply_critical_decision("PROBE_DEPTH")

    assert stopped["next_action"] == "STOP"
    assert "opposite depth probe also failed" in stopped["reason"]


def test_semantic_depth_probe_keeps_improving_reverse_after_first_wrong_way_sample() -> None:
    runtime = VerifiedCapabilityRuntime(semantic_pregrasp_enabled=True)
    runtime._pending_critical_kind = "PREGRASP_DECISION"
    runtime._pending_allowed_answers = ("PROBE_DEPTH", "UNKNOWN")
    runtime._pending_control_error = (0.0, 12.0)
    runtime.controller.effect_direction_streaks[("QWEN_PREGRASP", "MV_BACK")] = (-1, 1)
    runtime.controller.effect_direction_streaks[("QWEN_PREGRASP", "MV_FWD")] = (1, 1)
    # A fresh improving reverse observation must not make the next generic
    # PROBE_DEPTH answer bounce back to the original direction.
    runtime.controller.last_transition = {
        "status": "IMPROVING",
        "action": "MV_FWD",
        "improvement_px": 3.0,
    }

    decision = runtime.apply_critical_decision("PROBE_DEPTH")

    assert decision["next_action"] == "MV_FWD"
    assert "keep using the observed reverse direction" in decision["reason"]


def _fresh_secondary_agentview_evidence(*, frame=5, instance_id="target-1"):
    return {
        "camera": "wrist",
        "geometry": {"valid": True},
        "secondary_view": {
            "camera": "agentview",
            "instance_association": {
                "health": "VALID",
                "instance_id": instance_id,
                "frame_id": frame,
            },
            "geometry": {
                "valid": True,
                "in_frame": True,
                "frame_id": frame,
                "camera_calibration": {"width": 512, "height": 512},
                "target_minus_eef_px": [63.0, -110.0],
                "calibrated_correction_candidates": {
                    "horizontal": "MV_RIGHT",
                    "vertical": "MV_BACK",
                },
            },
        },
    }


def test_visual_align_compiles_one_fresh_same_instance_calibrated_action() -> None:
    runtime = VerifiedCapabilityRuntime(
        semantic_pregrasp_enabled=True,
        alignment_px=13.5,
    )
    runtime.belief.frame_id = 5
    runtime.belief.observation_health = ObservationHealth.VALID
    runtime.belief.target = EntityTrack(
        instance_id="target-1",
        semantic_label="target",
        bbox_xyxy=(20, 30, 40, 90),
        confidence=0.95,
        source="runtime_v2_instance_associated",
        last_confirmed_frame=5,
        camera="agentview",
    )
    evidence = _fresh_secondary_agentview_evidence()
    alignment = runtime._current_visual_alignment(evidence)
    assert alignment is not None
    assert alignment["target_minus_eef_px"] == [63.0, -110.0]
    assert runtime._compile_semantic_pregrasp_raw(
        "VISUAL_ALIGN",
        evidence={"visual_alignment": alignment},
        eef_xyz=None,
    ).action_token == "MV_BACK"

    runtime.last_event = {"evidence": evidence}
    runtime._pending_critical_kind = "PREGRASP_DECISION"
    runtime._pending_allowed_answers = ("VISUAL_ALIGN", "UNKNOWN")
    decision = runtime.apply_critical_decision("VISUAL_ALIGN")
    assert decision["accepted"] is True
    assert decision["next_action"] == "MV_BACK"
    assert runtime.controller.pending_external["error"] == (63.0, -110.0)


def test_visual_align_is_unavailable_for_stale_or_cross_instance_secondary_view() -> None:
    runtime = VerifiedCapabilityRuntime(semantic_pregrasp_enabled=True)
    runtime.belief.frame_id = 5
    runtime.belief.observation_health = ObservationHealth.VALID
    runtime.belief.target = EntityTrack(
        instance_id="target-1",
        semantic_label="target",
        bbox_xyxy=(20, 30, 40, 90),
        confidence=0.95,
        source="runtime_v2_instance_associated",
        last_confirmed_frame=5,
        camera="agentview",
    )
    stale = _fresh_secondary_agentview_evidence(frame=4)
    mismatch = _fresh_secondary_agentview_evidence(instance_id="other-target")
    assert runtime._current_visual_alignment(stale) is None
    assert runtime._current_visual_alignment(mismatch) is None

    valid_but_aligned = _fresh_secondary_agentview_evidence()
    valid_but_aligned["secondary_view"]["geometry"]["target_minus_eef_px"] = [4, -5]
    aligned = runtime._current_visual_alignment(valid_but_aligned)
    assert aligned is not None  # retained for measuring the previous action effect
    assert runtime._compile_semantic_pregrasp_raw(
        "VISUAL_ALIGN",
        evidence={"visual_alignment": aligned},
        eef_xyz=None,
    ).action_token == "STOP"


def test_visual_align_skips_a_measured_failed_axis_and_uses_next_current_axis() -> None:
    runtime = VerifiedCapabilityRuntime(semantic_pregrasp_enabled=True)
    runtime.belief.frame_id = 5
    runtime.belief.observation_health = ObservationHealth.VALID
    runtime.belief.target = EntityTrack(
        instance_id="target-1",
        semantic_label="target",
        bbox_xyxy=(20, 30, 40, 90),
        confidence=0.95,
        source="runtime_v2_instance_associated",
        last_confirmed_frame=5,
        camera="agentview",
    )
    runtime.controller.effect_direction_streaks[("QWEN_PREGRASP", "MV_BACK")] = (-1, 2)
    alignment = runtime._current_visual_alignment(
        _fresh_secondary_agentview_evidence()
    )
    assert alignment is not None
    decision = runtime._compile_semantic_pregrasp(
        "VISUAL_ALIGN",
        evidence={"visual_alignment": alignment},
        eef_xyz=None,
    )
    assert decision.action_token == "MV_RIGHT"
    assert "next fresh uncontradicted axis" in decision.reason


def test_v22_allows_visual_grasp_when_spatial_fusion_is_unknown_but_evidence_is_fresh() -> None:
    runtime = VerifiedCapabilityRuntime(
        semantic_pregrasp_enabled=True,
        require_spatial_ready_for_grasp=False,
        placement_v22_enabled=True,
        approach_min_height_m=0.10,
        approach_max_height_m=0.14,
    )
    runtime.belief.frame_id = 5
    runtime.belief.observation_health = ObservationHealth.VALID
    runtime.belief.target = EntityTrack(
        instance_id="target-1",
        semantic_label="target",
        bbox_xyxy=(90.0, 40.0, 150.0, 80.0),
        confidence=0.9,
        source="runtime_v2_instance_associated",
        last_confirmed_frame=5,
        camera="wrist",
    )
    runtime.belief.eef_xyz = EvidenceValue(
        value=(0.0, 0.0, 0.12),
        truth=TruthValue.TRUE,
        source="robot_proprioception",
        confidence=1.0,
        frame_id=5,
    )
    semantic_evidence = {
        "spatial_belief": {"health": "UNKNOWN", "relations": ["UNKNOWN"]}
    }

    choices = runtime._available_semantic_pregrasp_actions(
        evidence=semantic_evidence, eef_xyz=runtime.belief.eef_xyz.value
    )
    assert "GRASP" in choices

    runtime._pending_critical_kind = "PREGRASP_DECISION"
    runtime._pending_allowed_answers = choices
    runtime._pending_control_error = (0.0, 8.0)
    runtime._pending_evidence_frame_ids = [5]
    decision = runtime.apply_critical_decision(
        "GRASP",
        details={
            "agent_decision": {
                "selected": "GRASP",
                "validated": True,
                "evidence_for": [
                    {
                        "frame_id": 5,
                        "camera": "wrist",
                        "observation": "the target is visible between the open fingers",
                    }
                ],
                "evidence_against": [],
                "summary": "fresh wrist view supports one bounded close",
            }
        },
    )
    assert decision["next_action"] == "GRASP"


def test_v22_visual_grasp_stays_blocked_without_fresh_wrist_identity() -> None:
    runtime = VerifiedCapabilityRuntime(
        semantic_pregrasp_enabled=True,
        require_spatial_ready_for_grasp=False,
        placement_v22_enabled=True,
        approach_min_height_m=0.10,
        approach_max_height_m=0.14,
    )
    runtime.belief.frame_id = 5
    runtime.belief.observation_health = ObservationHealth.VALID
    runtime.belief.target = EntityTrack(
        instance_id="target-1",
        semantic_label="target",
        bbox_xyxy=(90.0, 40.0, 150.0, 80.0),
        confidence=0.9,
        source="runtime_v2_instance_associated",
        last_confirmed_frame=4,
        camera="wrist",
    )
    runtime.belief.eef_xyz = EvidenceValue(
        value=(0.0, 0.0, 0.12),
        truth=TruthValue.TRUE,
        source="robot_proprioception",
        confidence=1.0,
        frame_id=5,
    )

    choices = runtime._available_semantic_pregrasp_actions(
        evidence={"spatial_belief": {"health": "UNKNOWN", "relations": ["UNKNOWN"]}},
        eef_xyz=runtime.belief.eef_xyz.value,
    )
    assert "GRASP" not in choices


def test_no_progress_recovery_is_one_reobserve_then_resumes_option() -> None:
    runtime = VerifiedCapabilityRuntime(alignment_px=10.0)
    runtime.controller.last_action = "MV_BACK"
    failure = runtime._failure(
        code=FailureCode.NO_PROGRESS,
        option=OptionName.MOVE_TO_HOVER,
        frame_id=1,
        reason="test stall",
    )
    runtime._begin_recovery(failure, OptionName.MOVE_TO_HOVER)

    stopped_reobserve = runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(2, dx=20, dy=50),
        previous_action="STOP",
        agentview=np.zeros((100, 100, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.2),
    )
    assert stopped_reobserve["option"] == "RELOCALIZE"
    assert stopped_reobserve["belief"]["recovery"] is not None

    resumed = runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(3, dx=20, dy=49),
        previous_action=stopped_reobserve["action_token"],
        agentview=np.zeros((100, 100, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.2),
    )
    assert resumed["option"] == "MOVE_TO_HOVER"
    assert resumed["belief"]["recovery"] is None


def test_recovery_attempt_count_survives_local_context_completion() -> None:
    runtime = VerifiedCapabilityRuntime()
    failure = runtime._failure(
        code=FailureCode.NO_PROGRESS,
        option=OptionName.DESCEND_TO_GRASP,
        frame_id=1,
        reason="repeated local stall",
    )
    runtime._begin_recovery(failure, OptionName.DESCEND_TO_GRASP)
    runtime.belief.recovery = None
    runtime._begin_recovery(failure, OptionName.DESCEND_TO_GRASP)
    runtime.belief.recovery = None
    runtime._begin_recovery(failure, OptionName.DESCEND_TO_GRASP)
    assert runtime.belief.recovery.attempt == 3


def test_visual_hold_loss_creates_recovery_context_and_release() -> None:
    runtime = VerifiedCapabilityRuntime()
    runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(10),
        previous_action=None,
        agentview=np.zeros((100, 100, 3), dtype=np.uint8),
    )
    runtime.belief.held = EvidenceValue(
        True, TruthValue.TRUE, "prior verified hold", 0.9, 10
    )
    evidence = _evidence(11, bbox=(60, 20, 80, 60))
    evidence["target"] = "basket"
    evidence["held_object"] = {
        "bbox_xyxy": [20, 20, 40, 60],
        "confidence": 0.9,
        "source": "tracker",
    }
    evidence["visual_route"] = {
        "holding_arbiter": {"state": "LOST"},
        "progress": {},
    }
    result = runtime.observe_frame(
        stage="TRANSPORT",
        evidence=evidence,
        previous_action="MV_LEFT",
        agentview=np.zeros((100, 100, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.2),
        gripper_closed=True,
        gripper_width_m=0.02,
    )
    assert result["action_token"] == "RELEASE"
    assert result["failure"]["code"] == "LOST_HOLD"
    assert result["belief"]["recovery"]["resume_option"] == "LIFT_CLEAR"
    assert result["belief"]["route_epoch"] == 1


def test_relocalize_preserves_target_id() -> None:
    runtime = VerifiedCapabilityRuntime(alignment_px=10.0)
    first = runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(1, dx=20, dy=0),
        previous_action=None,
        agentview=np.zeros((256, 256, 3), dtype=np.uint8),
    )
    target_id = first["belief"]["target"]["instance_id"]
    failure = runtime._failure(
        code=FailureCode.LOST_HOLD,
        option=OptionName.TRANSFER,
        frame_id=2,
        reason="test drop",
    )
    runtime._begin_recovery(failure, OptionName.LIFT_CLEAR)
    recovered = runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(3, dx=2, dy=2),
        previous_action=first["action_token"],
        agentview=np.zeros((256, 256, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.106),
    )
    assert recovered["option"] == "RELOCALIZE"
    assert recovered["belief"]["target"]["instance_id"] == target_id
    assert recovered["status"] == "SUCCEEDED"
    assert recovered["belief"]["recovery"] is None


def test_alignment_cannot_skip_pregrasp_height_or_close() -> None:
    runtime = VerifiedCapabilityRuntime(
        alignment_px=10.0,
        approach_min_height_m=0.105,
        approach_max_height_m=0.14,
    )
    high = runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(1, dx=2, dy=2),
        previous_action=None,
        agentview=np.zeros((100, 100, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.20),
    )
    assert high["action_token"] == "MV_DOWN"
    descend = runtime.observe_frame(
        stage="GRASP",
        evidence=_evidence(2, dx=2, dy=2),
        previous_action="MV_DOWN",
        agentview=np.zeros((100, 100, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.12),
    )
    assert descend["action_token"] == "MV_DOWN"
    wrist_evidence = _evidence(3, bbox=(100, 40, 156, 100), dx=2, dy=2)
    wrist_evidence["camera"] = "wrist"
    final_approach = runtime.observe_frame(
        stage="GRASP",
        evidence=wrist_evidence,
        previous_action="MV_DOWN",
        agentview=np.zeros((100, 100, 3), dtype=np.uint8),
        wrist=np.zeros((100, 100, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.114),
    )
    assert final_approach["action_token"] == "STOP"
    assert final_approach["status"] == "NEED_DECISION"
    assert final_approach["critical_decision"]["kind"] == "PREGRASP_DECISION"
    correction = runtime.apply_critical_decision("MV_DOWN")
    assert correction["next_action"] == "MV_DOWN"
    wrist_evidence["frame_id"] = 4
    aligned_grasp = runtime.observe_frame(
        stage="GRASP",
        evidence=wrist_evidence,
        previous_action="MV_DOWN",
        agentview=np.zeros((100, 100, 3), dtype=np.uint8),
        wrist=np.zeros((100, 100, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.11),
    )
    assert aligned_grasp["action_token"] == "STOP"
    close = runtime.apply_critical_decision("GRASP")
    assert close["next_action"] == "GRASP"


def test_agentview_occlusion_cannot_close_without_wrist_evidence() -> None:
    runtime = VerifiedCapabilityRuntime(
        alignment_px=10.0,
        approach_min_height_m=0.105,
        approach_max_height_m=0.14,
    )
    visible = runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(1, dx=2, dy=2),
        previous_action=None,
        agentview=np.zeros((100, 100, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.106),
    )
    occluded = _evidence(
        2,
        bbox=None,
        dx=50,
        dy=-40,
        source="occlusion_memory",
    )
    close = runtime.observe_frame(
        stage="GRASP",
        evidence=occluded,
        previous_action=visible["action_token"],
        agentview=np.zeros((100, 100, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.106),
    )
    assert close["observation_health"] == "OCCLUDED"
    assert close["action_token"] == "STOP"
    assert close["failure"]["code"] == "NO_PROGRESS"
    assert close["belief"]["alignment_residual"]["value"] == [2.0, 2.0]


def test_wrist_vertical_pixels_cannot_destroy_verified_planar_depth() -> None:
    runtime = VerifiedCapabilityRuntime(
        approach_min_height_m=0.105,
        approach_max_height_m=0.14,
    )
    # Seed a globally aligned target, then hand over to Wrist.  Its target is
    # deliberately far from the old vertical pixel anchor; that coordinate is
    # perspective/height evidence, not a forward/back residual.
    runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(1, dx=1, dy=1),
        previous_action=None,
        agentview=np.zeros((256, 256, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.12),
    )
    wrist_evidence = _evidence(2, bbox=(96, 5, 160, 35), dx=0, dy=-50)
    wrist_evidence["camera"] = "wrist"
    decision = runtime.observe_frame(
        stage="GRASP",
        evidence=wrist_evidence,
        previous_action="DONE",
        agentview=np.zeros((256, 256, 3), dtype=np.uint8),
        wrist=np.zeros((256, 256, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.114),
    )
    assert decision["action_token"] == "STOP"
    assert decision["critical_decision"]["kind"] == "PREGRASP_DECISION"
    bounded = runtime.apply_critical_decision("MV_DOWN")
    assert bounded["next_action"] == "MV_DOWN"
    rejected = runtime.apply_critical_decision("MOVE_TO_MAGIC_POSE")
    assert rejected["accepted"] is False


def test_close_range_descent_uses_locked_track_not_rejected_detector_bbox() -> None:
    runtime = VerifiedCapabilityRuntime(
        alignment_px=6.0,
        approach_min_height_m=0.105,
        approach_max_height_m=0.14,
    )
    seeded = runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(1, dx=1, dy=1),
        previous_action=None,
        agentview=np.zeros((256, 256, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.139),
    )
    hidden = _evidence(
        2,
        bbox=(181, 96, 199, 134),
        dx=68,
        dy=-25,
        source="sam3",
    )
    hidden["geometry"]["pixel_xy"] = [121.5, 139.0]
    descending = runtime.observe_frame(
        stage="GRASP",
        evidence=hidden,
        previous_action="DONE",
        agentview=np.zeros((256, 256, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.13),
    )
    assert descending["observation_health"] == "OCCLUDED"
    assert descending["action_token"] == "MV_DOWN"

    hidden["frame_id"] = 3
    still_descending = runtime.observe_frame(
        stage="GRASP",
        evidence=hidden,
        previous_action="MV_DOWN",
        agentview=np.zeros((256, 256, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.115),
    )
    assert still_descending["action_token"] == "MV_DOWN"

    wrist_evidence = _evidence(4, bbox=(100, 40, 156, 100), dx=80, dy=-60)
    wrist_evidence["camera"] = "wrist"
    final_approach = runtime.observe_frame(
        stage="GRASP",
        evidence=wrist_evidence,
        previous_action="MV_DOWN",
        agentview=np.zeros((256, 256, 3), dtype=np.uint8),
        wrist=np.zeros((256, 256, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.114),
    )
    assert final_approach["action_token"] == "STOP"
    assert runtime.apply_critical_decision("MV_DOWN")["next_action"] == "MV_DOWN"
    wrist_evidence["frame_id"] = 5
    closing = runtime.observe_frame(
        stage="GRASP",
        evidence=wrist_evidence,
        previous_action="MV_DOWN",
        agentview=np.zeros((256, 256, 3), dtype=np.uint8),
        wrist=np.zeros((256, 256, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.11),
    )
    assert closing["action_token"] == "STOP"
    assert runtime.apply_critical_decision("GRASP")["next_action"] == "GRASP"
    assert closing["belief"]["target"]["instance_id"] == seeded["belief"]["target"]["instance_id"]
    assert closing["belief"]["target"]["camera"] == "wrist"


def test_table_support_backprojection_overrides_misleading_pixel_center() -> None:
    runtime = VerifiedCapabilityRuntime(
        support_plane_z_m=0.0,
        world_alignment_tolerance_m=0.008,
    )
    evidence = _evidence(1, bbox=(40, 40, 60, 50), dx=50, dy=0)
    evidence["geometry"]["camera_calibration"] = {
        "width": 100,
        "height": 100,
        "fovy_deg": 90.0,
        "position_world": [0.0, 0.0, 1.0],
        "camera_to_world": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "rotation_degrees": 0,
        "flip": "none",
    }
    result = runtime.observe_frame(
        stage="APPROACH",
        evidence=evidence,
        previous_action=None,
        agentview=np.zeros((100, 100, 3), dtype=np.uint8),
        eef_xyz=(0.1, 0.0, 0.2),
    )
    # Pixel-center control would move right; support-plane world geometry says
    # the EEF is 10 cm beyond the target in X and must move back.
    assert result["action_token"] == "MV_BACK"
    assert result["belief"]["world_xy_residual_m"]["truth"] == "TRUE"


def test_occlusion_cannot_bypass_option_budget() -> None:
    limits = RuntimeLimits(commit=2)
    runtime = VerifiedCapabilityRuntime(limits=limits)
    runtime.observe_frame(
        stage="GRASP",
        evidence=_evidence(1, bbox=None, source="occlusion_memory"),
        previous_action=None,
        agentview=np.zeros((100, 100, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.12),
    )
    runtime.observe_frame(
        stage="GRASP",
        evidence=_evidence(2, bbox=None, source="occlusion_memory"),
        previous_action="STOP",
        agentview=np.zeros((100, 100, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.12),
    )
    exhausted = runtime.observe_frame(
        stage="GRASP",
        evidence=_evidence(3, bbox=None, source="occlusion_memory"),
        previous_action="STOP",
        agentview=np.zeros((100, 100, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.12),
    )
    assert exhausted["status"] == "FAILED"
    assert exhausted["failure"]["code"] == "OPTION_BUDGET_EXCEEDED"


def test_critical_candidate_choice_keeps_instance_id_and_unlocks_health() -> None:
    runtime = VerifiedCapabilityRuntime()
    first = runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(1, bbox=(20, 20, 40, 50)),
        previous_action=None,
        agentview=np.zeros((100, 100, 3), dtype=np.uint8),
    )
    identity = first["belief"]["target"]["instance_id"]
    ambiguous = runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(
            2,
            bbox=(21, 20, 41, 50),
            tool={
                "candidates": [
                    {"score": 0.8, "bbox_xyxy": [21, 20, 41, 50]},
                    {"score": 0.79, "bbox_xyxy": [19, 20, 39, 50]},
                ]
            },
        ),
        previous_action=first["action_token"],
        agentview=np.zeros((100, 100, 3), dtype=np.uint8),
    )
    assert ambiguous["critical_decision"]["kind"] == "SELECT_INSTANCE"
    commit = runtime.apply_critical_decision("candidate-0")
    assert commit["accepted"] is True
    assert runtime.belief.target.instance_id == identity
    assert runtime.belief.observation_health == ObservationHealth.VALID


def test_v21_temporal_track_resolves_detector_tie_after_qwen_selection() -> None:
    runtime = VerifiedCapabilityRuntime(
        temporal_identity_lock_on_detector_ties=True
    )
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    first = runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(1, bbox=(20, 20, 40, 50)),
        previous_action=None,
        agentview=image,
    )
    identity = first["belief"]["target"]["instance_id"]

    tied_detector = _evidence(
        2,
        bbox=None,
        source="sam3_abstain",
        tool={
            "abstain_reason": "ambiguous_top_detections",
            "candidates": [
                # Detector confidence deliberately favors the distractor.
                {"score": 0.99, "bbox_xyxy": [80, 20, 100, 50]},
                {"score": 0.80, "bbox_xyxy": [21, 20, 41, 50]},
            ],
        },
    )
    result = runtime.observe_frame(
        stage="APPROACH",
        evidence=tied_detector,
        previous_action=first["action_token"],
        agentview=image,
    )

    assert result["belief"]["observation_health"] == ObservationHealth.VALID
    assert result["belief"]["target"]["instance_id"] == identity
    assert result["critical_decision"] is None
    assert result["event"]["health_reason"] == "temporal_identity_disambiguated_detector_tie"


def test_v21_recent_qwen_identity_lock_survives_subthreshold_motion_match() -> None:
    runtime = VerifiedCapabilityRuntime(
        temporal_identity_lock_on_detector_ties=True
    )
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    first = runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(
            1,
            bbox=None,
            source="sam3_abstain",
            tool={
                "abstain_reason": "ambiguous_top_detections",
                "candidates": [
                    {"score": 0.80, "bbox_xyxy": [20, 20, 40, 50]},
                    {"score": 0.79, "bbox_xyxy": [80, 20, 100, 50]},
                ],
            },
        ),
        previous_action=None,
        agentview=image,
    )
    assert first["critical_decision"]["kind"] == "SELECT_INSTANCE"
    selected = runtime.apply_critical_decision("candidate-0")
    assert selected["accepted"] is True
    target_id = runtime.belief.target.instance_id

    # The same object moved seven pixels. The unique in-gate track match is
    # valid and inside EntityTracker's post-selection lock, although its
    # normalized confidence is below the separate 0.90 general cutoff.
    continued = runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(
            2,
            bbox=None,
            source="sam3_abstain",
            tool={
                "abstain_reason": "ambiguous_top_detections",
                "candidates": [
                    {"score": 0.80, "bbox_xyxy": [27, 20, 47, 50]},
                    {"score": 0.79, "bbox_xyxy": [80, 20, 100, 50]},
                ],
            },
        ),
        previous_action=first["action_token"],
        agentview=image,
    )
    assert runtime.belief.target.association_confidence < 0.90
    assert continued["observation_health"] == ObservationHealth.VALID
    assert continued["belief"]["target"]["instance_id"] == target_id
    assert continued["critical_decision"] is None
    assert continued["event"]["health_reason"] == "temporal_identity_disambiguated_detector_tie"


def test_v21_detector_tie_lock_does_not_accept_stale_ambiguous_track() -> None:
    runtime = VerifiedCapabilityRuntime(
        temporal_identity_lock_on_detector_ties=True
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    first = runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(1, bbox=(20, 20, 40, 50)),
        previous_action=None,
        agentview=image,
    )
    tied_detector = _evidence(
        2,
        bbox=None,
        source="sam3_abstain",
        tool={
            "abstain_reason": "ambiguous_top_detections",
            "candidates": [
                {"score": 0.80, "bbox_xyxy": [21, 20, 41, 50]},
                {"score": 0.79, "bbox_xyxy": [19, 20, 39, 50]},
            ],
        },
    )
    result = runtime.observe_frame(
        stage="APPROACH",
        evidence=tied_detector,
        previous_action=first["action_token"],
        agentview=image,
    )

    assert result["belief"]["observation_health"] == ObservationHealth.AMBIGUOUS
    assert result["critical_decision"]["kind"] == "SELECT_INSTANCE"
    assert result["action_token"] == "STOP"


def test_qwen_candidate_ids_are_spatially_stable_not_detector_score_order() -> None:
    runtime = VerifiedCapabilityRuntime()
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    ambiguous = runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(
            1,
            bbox=None,
            source="sam3_abstain",
            tool={
                "abstain_reason": "ambiguous_top_detections",
                "candidates": [
                    {"score": 0.99, "bbox_xyxy": [80, 20, 100, 50]},
                    {"score": 0.80, "bbox_xyxy": [20, 20, 40, 50]},
                ],
            },
        ),
        previous_action=None,
        agentview=image,
    )

    assert ambiguous["critical_decision"]["candidates"][0]["bbox_xyxy"] == [20, 20, 40, 50]
    runtime.apply_critical_decision("candidate-0")
    assert runtime.belief.target.bbox_xyxy == (20.0, 20.0, 40.0, 50.0)


def test_v21_metric_approach_selects_budget_from_tracked_support_residual() -> None:
    from core.capabilities.camera_geometry import (
        CameraCalibration,
        backproject_pixel_to_plane,
    )

    calibration = CameraCalibration(
        name="agentview",
        width=256,
        height=256,
        fovy_deg=45.0,
        position_world=np.asarray([0.896577, 1e-6, 0.65]),
        camera_to_world=np.asarray([
            [-2e-6, -0.52877, 0.848765],
            [1.0, -1e-6, 2e-6],
            [0.0, 0.848765, 0.52877],
        ]),
        rotation_degrees=180,
        flip="none",
    )
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    evidence = _evidence(1, bbox=(181, 96, 199, 134), dx=20.0, dy=4.0)
    evidence["geometry"]["camera_calibration"] = {
        "width": 256,
        "height": 256,
        "fovy_deg": 45.0,
        "position_world": [0.896577, 1e-6, 0.65],
        "camera_to_world": [
            [-2e-6, -0.52877, 0.848765],
            [1.0, -1e-6, 2e-6],
            [0.0, 0.848765, 0.52877],
        ],
        "rotation_degrees": 180,
        "flip": "none",
    }

    far_runtime = VerifiedCapabilityRuntime(
        metric_approach_uses_hover_budget=True
    )
    far = far_runtime.observe_frame(
        stage="APPROACH",
        evidence=evidence,
        previous_action=None,
        agentview=image,
        eef_xyz=(-0.1473, 0.0041, 0.2612),
    )
    assert far["belief"]["current_option"] == OptionName.MOVE_TO_HOVER

    support = backproject_pixel_to_plane(
        calibration, [190.0, 134.0], 0.015
    )
    assert support is not None
    local_runtime = VerifiedCapabilityRuntime(
        metric_approach_uses_hover_budget=True
    )
    local = local_runtime.observe_frame(
        stage="APPROACH",
        evidence=evidence,
        previous_action=None,
        agentview=image,
        eef_xyz=(float(support[0]), float(support[1]), 0.12),
    )
    assert local["belief"]["current_option"] == OptionName.ALIGN_PREGRASP


def test_v21_occluded_grasp_entry_uses_current_episode_memory_before_clearance() -> None:
    runtime = VerifiedCapabilityRuntime(
        semantic_pregrasp_enabled=True,
        pregrasp_reflection_mode="double",
        approach_min_height_m=0.105,
        approach_max_height_m=0.14,
    )
    runtime.reset(episode_id="episode-visual-memory-test")
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    refs_1 = {
        "agentview": "images/raw_agentview/0001.png",
        "wrist": "images/raw_wrist/0001.png",
    }
    refs_2 = {
        "agentview": "images/raw_agentview/0002.png",
        "wrist": "images/raw_wrist/0002.png",
    }
    first = runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(1),
        previous_action=None,
        agentview=image,
        wrist=image,
        image_refs=refs_1,
        eef_xyz=(0.0, 0.0, 0.12),
    )
    identity = first["belief"]["target"]["instance_id"]

    wrist_evidence = _evidence(
        2,
        bbox=None,
        source="sam3_abstain",
        tool={"abstain_reason": "no_detections_after_backend_threshold"},
    )
    wrist_evidence["camera"] = "wrist"
    occluded = runtime.observe_frame(
        stage="GRASP",
        evidence=wrist_evidence,
        previous_action="DONE",
        agentview=image,
        wrist=image,
        image_refs=refs_2,
        eef_xyz=(0.0, 0.0, 0.12),
    )

    request = occluded["critical_decision"]
    assert request["kind"] == "PREGRASP_DECISION"
    assert request["reflection_trigger"] == "grasp_entry"
    assert request["reflection_mode"] == "double"
    assert request["allowed_answers"] == ["REOBSERVE", "UNKNOWN"]
    assert request["allowed_options"] == ["REOBSERVE"]
    assert request["visual_memory_refs"] == [
        "images/raw_agentview/0001.png",
        "images/raw_wrist/0001.png",
        "images/raw_agentview/0002.png",
        "images/raw_wrist/0002.png",
    ]
    assert occluded["belief"]["target"]["instance_id"] == identity
    assert occluded["action_token"] == "STOP"

    decision = runtime.apply_critical_decision(
        "REOBSERVE",
        details={
            "agent_decision": {
                "selected": "REOBSERVE",
                "validated": True,
                "state_hypothesis": "The current Wrist view does not resolve target-to-gripper relation.",
                "evidence_for": [{
                    "frame_id": 2,
                    "camera": "agentview",
                    "observation": "The gripper is near the target while Wrist does not show the target.",
                }],
                "evidence_against": [],
                "missing_observation": "A fresh dual view after a small clearance change.",
                "expected_effect": "The new frame may reveal target-to-gripper relation.",
                "failure_condition": "The target remains occluded after the one bounded view move.",
                "summary": "Request one new observation before deciding whether to grasp.",
            }
        },
    )
    assert decision["accepted"] is True
    assert decision["answer"] == "REOBSERVE"
    assert decision["semantic_choice"] == "REOBSERVE"
    assert decision["next_action"] == "MV_UP"
    assert decision["reobserve_for_view"] is True
    assert abs(decision["reobserve_lift_step_m"] - 0.02) < 1e-9

    repeated_occlusion = runtime.observe_frame(
        stage="GRASP",
        evidence={**wrist_evidence, "frame_id": 3},
        previous_action="MV_UP",
        agentview=image,
        wrist=image,
        image_refs={
            "agentview": "images/raw_agentview/0003.png",
            "wrist": "images/raw_wrist/0003.png",
        },
        eef_xyz=(0.0, 0.0, 0.13),
    )
    assert repeated_occlusion["critical_decision"] is None
    assert repeated_occlusion["action_token"] == "STOP"
    assert "stale geometry" in repeated_occlusion["reason"]


def test_v21_occluded_grasp_entry_rejects_stale_target_identity() -> None:
    runtime = VerifiedCapabilityRuntime(
        semantic_pregrasp_enabled=True,
        pregrasp_reflection_mode="double",
        semantic_evidence_max_age_frames=0,
        approach_min_height_m=0.105,
        approach_max_height_m=0.14,
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    refs = {
        "agentview": "images/raw_agentview/0001.png",
        "wrist": "images/raw_wrist/0001.png",
    }
    runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(1),
        previous_action=None,
        agentview=image,
        wrist=image,
        image_refs=refs,
        eef_xyz=(0.0, 0.0, 0.12),
    )
    wrist_evidence = _evidence(
        2,
        bbox=None,
        source="sam3_abstain",
        tool={"abstain_reason": "no_detections_after_backend_threshold"},
    )
    wrist_evidence["camera"] = "wrist"
    result = runtime.observe_frame(
        stage="GRASP",
        evidence=wrist_evidence,
        previous_action="DONE",
        agentview=image,
        wrist=image,
        image_refs={
            "agentview": "images/raw_agentview/0002.png",
            "wrist": "images/raw_wrist/0002.png",
        },
        eef_xyz=(0.0, 0.0, 0.12),
    )
    assert result["critical_decision"] is None
    assert result["action_token"] == "STOP"


def test_v21_memory_marks_grasp_entry_and_keeps_raw_frame_receipts() -> None:
    runtime = VerifiedCapabilityRuntime(semantic_pregrasp_enabled=True)
    images = np.zeros((100, 100, 3), dtype=np.uint8)
    runtime.observe_frame(
        stage="APPROACH",
        evidence=_evidence(1),
        previous_action=None,
        agentview=images,
        wrist=images,
        image_refs={
            "agentview": "images/raw_agentview/0001.png",
            "wrist": "images/raw_wrist/0001.png",
        },
        eef_xyz=(0.0, 0.0, 0.12),
    )
    runtime.observe_frame(
        stage="GRASP",
        evidence=_evidence(2),
        previous_action="MV_DOWN",
        agentview=images,
        wrist=images,
        image_refs={
            "agentview": "images/raw_agentview/0002.png",
            "wrist": "images/raw_wrist/0002.png",
        },
        eef_xyz=(0.0, 0.0, 0.11),
    )

    target_id = runtime.belief.target.instance_id
    bundle = runtime.visual_memory.placement_bundle(
        instance_id=target_id, grasp_epoch=runtime.belief.grasp_epoch, limit=3
    )
    assert [entry["frame_id"] for entry in bundle] == [1, 2]
    assert "STAGE_ENTRY_GRASP" in bundle[-1]["tags"]
    assert bundle[-1]["agentview_ref"] == "images/raw_agentview/0002.png"
    assert bundle[-1]["wrist_ref"] == "images/raw_wrist/0002.png"


def test_unknown_hold_requests_bounded_qwen_verdict() -> None:
    runtime = VerifiedCapabilityRuntime()
    result = runtime.observe_frame(
        stage="LIFT",
        evidence=_evidence(4),
        previous_action="GRASP",
        agentview=np.zeros((100, 100, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.12),
        gripper_closed=True,
        gripper_width_m=0.02,
    )
    assert result["action_token"] == "STOP"
    assert result["critical_decision"]["kind"] == "VERIFY_HOLD"
    committed = runtime.apply_critical_decision("YES")
    assert committed["accepted"] is True
    assert runtime.belief.held.truth == TruthValue.TRUE
    assert runtime.belief.grasp_epoch == 1


def test_release_requires_seated_verdict_and_rejected_seat_holds_without_contact_evidence() -> None:
    runtime = VerifiedCapabilityRuntime()
    runtime.belief.held = EvidenceValue(
        True, TruthValue.TRUE, "verified hold", 0.9, 8
    )
    result = runtime.observe_frame(
        stage="RELEASE",
        evidence=_evidence(9),
        previous_action="MV_DOWN",
        agentview=np.zeros((100, 100, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.12),
        gripper_closed=True,
        gripper_width_m=0.02,
    )
    assert result["action_token"] == "STOP"
    assert result["critical_decision"]["kind"] == "VERIFY_SEATED"
    rejected = runtime.apply_critical_decision("NO")
    # A negative semantic answer does not establish rim contact or authorize
    # an automatic lift. The runtime waits for a bounded, evidence-backed
    # recovery option instead of treating every failed seat as a collision.
    assert rejected["next_action"] == "STOP"
    assert rejected["rollback_stage"] == "PLACE"
    assert runtime.belief.seated.truth == TruthValue.FALSE


def test_post_close_verdict_commits_hold_or_typed_empty_grasp_recovery() -> None:
    runtime = VerifiedCapabilityRuntime()
    runtime.observe_frame(
        stage="GRASP",
        evidence=_evidence(1, dx=2, dy=2),
        previous_action=None,
        agentview=np.zeros((100, 100, 3), dtype=np.uint8),
        eef_xyz=(0.0, 0.0, 0.12),
    )
    rejected = runtime.report_grasp_verdict(
        verdict="NO",
        frame_id=1,
        mechanically_empty=True,
        reasoning="object remains supported",
    )
    assert rejected["failure"]["code"] == "EMPTY_GRASP"
    assert rejected["recovery_context"]["resume_option"] == "MOVE_TO_HOVER"
    assert runtime.belief.held.truth == TruthValue.FALSE

    accepted = runtime.report_grasp_verdict(
        verdict="YES",
        frame_id=2,
        mechanically_empty=False,
        reasoning="stable enclosure",
    )
    assert accepted["verdict"] == "PASS"
    assert runtime.belief.held.truth == TruthValue.TRUE
    assert runtime.belief.recovery is None

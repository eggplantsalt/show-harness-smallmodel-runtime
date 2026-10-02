from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest

from core.capabilities.camera_geometry import CameraCalibration
from core.runtime_v3.arbiter import Arbiter
from core.runtime_v3.executor import Executor
from core.runtime_v3.metric_entity import MetricEntityReference
from core.runtime_v3.object_relative import (
    DIRECTION_ORDER, MultiScaleAlignmentOptionGenerator,
    make_multiscale_alignment_option, resolve_object_relative_geometry,
)
from core.runtime_v3.options import BoundedMicroMotionSpec
from core.runtime_v3.selector import Selection
from core.runtime_v3.state import BeliefState, ObjectRelativeState
from scripts.runtime_v3_multiscale_alignment import (
    TICK_BUDGETS, _post_action_stop, candidate_lattice_records,
)


MOVE_VECTORS = {
    "MV_FWD": [1.0, 0.0, 0.0], "MV_BACK": [-1.0, 0.0, 0.0],
    "MV_LEFT": [0.0, 1.0, 0.0], "MV_RIGHT": [0.0, -1.0, 0.0],
    "MV_UP": [0.0, 0.0, 1.0], "MV_DOWN": [0.0, 0.0, -1.0],
}
CAMERA = CameraCalibration(
    name="agentview", width=512, height=512, fovy_deg=60.0,
    position_world=np.zeros(3), camera_to_world=np.eye(3),
)


def _resolve(scales=(0.003, 0.006, 0.009), *, contracts=None,
             target=(300.0, 256.0), eef=(0.0, 0.0, -1.0), bounds=(-2.0, 1.0),
             camera=CAMERA):
    if contracts is None:
        contracts = {scale: {"verified": True, "max_ticks": TICK_BUDGETS[scale]}
                     for scale in scales}
    return resolve_object_relative_geometry(
        target_reference_px=target, eef_position_xyz_m=eef, calibration=camera,
        move_vectors=MOVE_VECTORS, workspace_z_bounds_m=bounds,
        candidate_scales_m=scales, scale_contracts=contracts,
    )


def _state_for(candidate, *, frame_id=7, extras=None):
    relative = ObjectRelativeState(
        target_phrase="salad dressing", target_visible=True,
        target_identity_status="ANCHORED", target_reference_point_px=(300.0, 256.0),
        target_reference_valid=True, eef_projection_px=(256.0, 256.0),
    )
    geometry = {
        "camera_projection_valid": True, "workspace_valid": True,
        "multiscale_alignment_valid": True, "pixel_error_before_px": 44.0,
        "workspace_z_bounds_m": (-2.0, 1.0),
        "chosen_lattice_candidate": candidate,
    }
    geometry.update(extras or {})
    return BeliefState(
        object_relative_state=relative,
        end_effector_state={"position_xyz": (0.0, 0.0, -1.0)},
        relevant_geometry=geometry, frame_id=frame_id, observation_fresh=True,
    )


def _picked_geometry():
    return _resolve()["chosen_lattice_candidate"]


def test_bounded_specs_allow_all_three_calibrated_displacement_scales():
    for scale in (0.003, 0.006, 0.009):
        spec = BoundedMicroMotionSpec("FWD", (1, 0, 0), scale, TICK_BUDGETS[scale])
        assert spec.requested_displacement_m == scale


def test_three_millimeter_contract_keeps_the_existing_five_tick_budget():
    spec = BoundedMicroMotionSpec("DOWN", (0, 0, -1), 0.003, TICK_BUDGETS[0.003])
    assert spec.max_ticks == 5


def test_six_millimeter_contract_uses_its_larger_seven_tick_budget():
    spec = BoundedMicroMotionSpec("RIGHT", (0, -1, 0), 0.006, TICK_BUDGETS[0.006])
    assert spec.max_ticks == 7


def test_nine_millimeter_contract_uses_its_larger_finite_ten_tick_budget():
    spec = BoundedMicroMotionSpec("UP", (0, 0, 1), 0.009, TICK_BUDGETS[0.009])
    assert spec.max_ticks == 10


def test_scale_or_budget_beyond_the_calibrated_contract_is_rejected():
    with pytest.raises(ValueError, match="9 mm"):
        BoundedMicroMotionSpec("FWD", (1, 0, 0), 0.012, 10)
    with pytest.raises(ValueError, match="1 and 10"):
        BoundedMicroMotionSpec("FWD", (1, 0, 0), 0.009, 11)


def test_candidate_enumeration_is_direction_times_verified_scales():
    result = _resolve((0.003, 0.006))
    assert len(result["candidate_lattice"]) == len(DIRECTION_ORDER) * 2
    assert {(row["direction"], row["displacement_m"]) for row in result["candidate_lattice"]} == {
        (direction, scale) for direction in DIRECTION_ORDER for scale in (0.003, 0.006)
    }


def test_unverified_scale_contract_is_recorded_invalid_and_never_selected():
    result = _resolve((0.003, 0.006), contracts={
        0.003: {"verified": True, "max_ticks": 5},
        0.006: {"verified": False, "max_ticks": 7},
    })
    invalid = [row for row in result["candidate_lattice"] if row["displacement_m"] == 0.006]
    assert len(invalid) == 6
    assert all(not row["valid"] and row["reason"] == "scale_contract_not_verified" for row in invalid)
    assert all(row["workspace_valid"] for row in invalid)
    assert result["chosen_lattice_candidate"]["displacement_m"] == 0.003


def test_workspace_invalid_lattice_point_is_retained_but_not_selected():
    camera = CameraCalibration(
        name="agentview", width=512, height=512, fovy_deg=60.0,
        position_world=np.asarray((0.0, 0.0, 2.0)), camera_to_world=np.eye(3),
    )
    result = _resolve((0.003, 0.006, 0.009), eef=(0.0, 0.0, 0.995),
                      bounds=(0.02, 1.0), camera=camera)
    up = [row for row in result["candidate_lattice"] if row["direction"] == "UP"]
    assert len(up) == 3
    assert up[0]["workspace_valid"]
    assert all(not row["workspace_valid"] and row["reason"] == "workspace_boundary" for row in up[1:])
    assert result["chosen_lattice_candidate"]["direction"] != "UP"


def test_runtime_picks_the_valid_candidate_with_minimum_predicted_error():
    result = _resolve()
    valid = [row for row in result["candidate_lattice"] if row["valid"]]
    chosen = result["chosen_lattice_candidate"]
    assert chosen["predicted_error_px"] == min(row["predicted_error_px"] for row in valid)


def test_each_fresh_state_recomputes_the_sealed_scale_and_direction():
    first = {"valid": True, "scale_contract_valid": True, "workspace_valid": True,
             "direction": "FWD", "direction_unit": [1, 0, 0], "displacement_m": 0.003,
             "max_ticks": 5, "predicted_error_px": 10, "predicted_improvement_px": 2}
    second = {"valid": True, "scale_contract_valid": True, "workspace_valid": True,
              "direction": "RIGHT", "direction_unit": [0, -1, 0], "displacement_m": 0.009,
              "max_ticks": 10, "predicted_error_px": 8, "predicted_improvement_px": 4}
    generator = MultiScaleAlignmentOptionGenerator()
    option_1 = generator.generate(_state_for(first, frame_id=10))[0]
    option_2 = generator.generate(_state_for(second, frame_id=11))[0]
    assert option_1.primitive.micro_motion_spec.direction == "FWD"
    assert option_1.primitive.micro_motion_spec.requested_displacement_m == 0.003
    assert option_2.primitive.micro_motion_spec.direction == "RIGHT"
    assert option_2.primitive.micro_motion_spec.requested_displacement_m == 0.009
    assert option_1.evidence_frame_id != option_2.evidence_frame_id


def test_generated_option_has_one_semantic_align_id_and_seals_physical_scale():
    option = make_multiscale_alignment_option(_state_for(_picked_geometry()))
    assert option.option_id == "ALIGN_TO_TARGET_BOUNDED"
    assert "DOWN_9MM" not in option.option_id
    assert option.primitive.micro_motion_spec.requested_displacement_m == _picked_geometry()["displacement_m"]


def test_executor_cannot_change_arbiter_approved_scale_and_stops_at_target():
    candidate = {"valid": True, "scale_contract_valid": True, "workspace_valid": True,
                 "direction": "FWD", "direction_unit": [1, 0, 0], "displacement_m": 0.009,
                 "max_ticks": 10, "predicted_error_px": 2, "predicted_improvement_px": 1}
    state = _state_for(candidate, frame_id=12)
    option = make_multiscale_alignment_option(state)
    approved = Arbiter().authorize(state, [option], Selection(option.option_id)).action
    assert approved.primitive.micro_motion_spec.requested_displacement_m == 0.009
    with pytest.raises(FrozenInstanceError):
        approved.primitive.micro_motion_spec.requested_displacement_m = 0.003

    class Backend:
        ticks = 0
        def execute_approved_micro_tick(self, _action):
            self.ticks += 1
        def execute_approved_action(self, _action):
            raise AssertionError("unexpected primitive")

    backend, arbiter = Backend(), Arbiter()
    # Re-authorize through the same Arbiter used by Executor.
    action = arbiter.authorize(state, [option], Selection(option.option_id)).action
    position = [0.0, 0.0, -1.0]
    def observe():
        position[0] += 0.0025
        return SimpleNamespace(proprioception={"end_effector_state": {"position_xyz": tuple(position)}})
    result = Executor(backend, arbiter).execute(action, tick_observer=observe).result
    assert result["requested_mm"] == 9.0
    assert result["termination"] == "TARGET_REACHED"
    assert result["ticks_executed"] == backend.ticks == 4
    assert result["actual_projection_mm"] <= 9.0 + 2.5


def test_executor_honors_scale_specific_max_tick_budget_when_projection_is_unreachable():
    candidate = {"valid": True, "scale_contract_valid": True, "workspace_valid": True,
                 "direction": "FWD", "direction_unit": [1, 0, 0], "displacement_m": 0.009,
                 "max_ticks": 10, "predicted_error_px": 2, "predicted_improvement_px": 1}
    state = _state_for(candidate, frame_id=13)
    option = make_multiscale_alignment_option(state)
    arbiter = Arbiter()
    action = arbiter.authorize(state, [option], Selection(option.option_id)).action

    class Backend:
        ticks = 0
        def execute_approved_micro_tick(self, _action):
            self.ticks += 1
        def execute_approved_action(self, _action):
            raise AssertionError("unexpected primitive")

    backend = Backend()
    position = [0.0, 0.0, -1.0]
    def observe():
        position[0] += 0.0001
        return SimpleNamespace(proprioception={"end_effector_state": {"position_xyz": tuple(position)}})
    result = Executor(backend, arbiter).execute(action, tick_observer=observe).result
    assert result["termination"] == "MAX_TICKS_REACHED"
    assert result["max_ticks"] == backend.ticks == 10
    assert result["actual_projection_mm"] < 9.0


def test_one_semantic_align_decision_gets_exactly_one_arbiter_approval():
    state = _state_for(_picked_geometry())
    option = MultiScaleAlignmentOptionGenerator().generate(state)[0]
    arbiter = Arbiter()
    decision = arbiter.authorize(state, [option], Selection(option.option_id))
    assert decision.action is not None
    assert arbiter.is_approved(decision.action)


def test_nonpositive_observed_effect_stops_without_a_retry():
    state = _state_for(_picked_geometry())
    assert _post_action_stop(1, 0.0, state) == "EFFECT_NOT_IMPROVED"
    assert _post_action_stop(1, -0.2, state) == "EFFECT_NOT_IMPROVED"


def test_candidate_lattice_artifact_keeps_every_candidate_and_marks_the_winner():
    result = _resolve((0.003, 0.006))
    records = candidate_lattice_records(result["candidate_lattice"], result["chosen_lattice_candidate"])
    assert len(records) == 12
    assert sum(row["selected"] for row in records) == 1
    assert all({"direction", "scale_mm", "predicted_pixel", "predicted_error_px",
                "predicted_improvement_px", "workspace_valid", "contract_valid",
                "valid", "selected"} <= set(row) for row in records)


def test_qwen_oracle_diagnostic_fields_cannot_change_runtime_candidate_choice():
    selected = _picked_geometry()
    baseline_state = _state_for(selected)
    baseline = make_multiscale_alignment_option(baseline_state)
    annotated = make_multiscale_alignment_option(_state_for(
        selected, extras={"qwen_requested_direction": "UP", "qwen_requested_scale_m": 0.009,
                          "oracle_target_world_position_m": [99, 99, 99],
                          "runtime_metric_distance_m": 0.001,
                          "metric_entity_reference_world_m": [-99, 99, -99]}))
    formal_reference = MetricEntityReference(
        entity_key="semantic entity", camera="agentview", coordinate_frame="world",
        reference_world_m=(-99.0, 99.0, -99.0), valid_depth_count=4, mask_pixel_count=4,
        valid_depth_ratio=1.0, depth_median_m=1.0, depth_spread_m=0.0,
        depth_source="monocular_metric", source_frame_id="frame-1", valid=True,
    )
    state_with_estimated_3d_evidence = replace(
        baseline_state,
        object_relative_state=replace(
            baseline_state.object_relative_state,
            metric_entity_reference=formal_reference,
        ),
    )
    with_metric = make_multiscale_alignment_option(state_with_estimated_3d_evidence)
    assert annotated.primitive.micro_motion_spec == baseline.primitive.micro_motion_spec
    assert with_metric.primitive.micro_motion_spec == baseline.primitive.micro_motion_spec
    assert annotated.expected_effect["predicted_improvement_px"] == baseline.expected_effect["predicted_improvement_px"]

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.runtime_v3.adapters.libero_env import LiberoEnvironmentAdapter
from core.runtime_v3.adapters.libero_observation import LiberoObservationAdapter
from core.runtime_v3.adapters.qwen_selector import QwenSelectorAdapter
from core.runtime_v3.arbiter import Arbiter, DecisionKind
from core.runtime_v3.effects import EffectObserver
from core.runtime_v3.observer import RobotObservation
from core.runtime_v3.options import PrimitiveCommand, RuntimeOption
from core.runtime_v3.smoke_options import SmokeOptionGenerator
from core.runtime_v3.state import BeliefState, StateBuilder
from core.sim.libero_task import LiberoTaskHandle
from core.vlm.vlm_client import VLMResponse


ROOT = Path(__file__).resolve().parents[2]


def _mock_observation():
    return {
        "agentview_image": np.zeros((12, 16, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((12, 16, 3), 127, dtype=np.uint8),
        "robot0_eef_pos": np.asarray([0.25, -0.10, 0.35]),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.asarray([0.02, -0.02]),
    }


def _state():
    observation = RobotObservation(
        "obs-1", 1,
        proprioception={"end_effector_state": {"position_xyz": [0.25, -0.1, 0.35]},
                        "gripper_state": "UNKNOWN", "gripper_width_m": 0.04},
        evidence={"stage": "SMOKE", "relevant_geometry": {
            "workspace_valid": True, "workspace_z_bounds_m": [0.02, 0.60],
        }},
        evidence_refs=("frame-1",), fresh=True,
    )
    return StateBuilder().update(StateBuilder().initialize("LIBERO_OBJECT:0"), observation)


def test_a_libero_observation_adapter_reads_mock_environment():
    raw = _mock_observation()

    class Env:
        _showharness_last_obs = raw

    env = Env()
    handle = LiberoTaskHandle(env, "LIBERO_OBJECT", 0, "mock task", "move upward",
                              "mock.bddl", np.zeros((1, 7)), 0)
    environment = LiberoEnvironmentAdapter(handle)
    adapter = LiberoObservationAdapter()
    v3_observation = adapter.observe(environment)
    assert adapter.last_raw is not None
    assert adapter.last_raw.agentview_rgb.shape == (12, 16, 3)
    assert adapter.last_raw.wrist_rgb.shape == (12, 16, 3)
    assert adapter.last_raw.eef_position_xyz == (0.25, -0.1, 0.35)
    assert adapter.last_raw.eef_quaternion == (0.0, 0.0, 0.0, 1.0)
    assert adapter.last_raw.gripper_width_m > 0
    assert v3_observation.proprioception["gripper_state"] == "UNKNOWN"
    assert v3_observation.evidence["relevant_geometry"]["workspace_valid"]


class FakeQwenClient:
    model = "Qwen/mock"

    def __init__(self, raw_output: str, parsed_payload: dict | None = None):
        self.raw_output = raw_output
        self.parsed_payload = parsed_payload
        self.kwargs = None

    def complete_json(self, prompt, **kwargs):
        self.kwargs = {"prompt": prompt, **kwargs}
        if self.parsed_payload is None:
            content = self.raw_output
        else:
            content = json.dumps(self.parsed_payload)
        payload = {"raw": {"choices": [{"message": {"content": self.raw_output}}]},
                   "usage": {"prompt_tokens": 91, "completion_tokens": 8},
                   "latency_s": 0.01}
        return VLMResponse("", content, payload)


def _option(option_id="OPTION_X"):
    return RuntimeOption(
        option_id=option_id, option_type="bounded_vertical_probe", description="small upward move",
        preconditions={"observation_fresh": True},
        expected_effect={"end_effector_delta_xyz": [0.0, 0.0, 0.005]},
        primitive=PrimitiveCommand("move", "MV_UP", {"step_m": 0.005}, 1, 1.0),
        confidence=1.0, evidence=("frame-1",), evidence_frame_id=1,
    )


def test_b_qwen_adapter_parses_legal_option_id():
    client = FakeQwenClient('{"selection":"OPTION_X"}')
    selector = QwenSelectorAdapter(client, "move the end effector safely")
    selection = selector.select(_state(), [_option()])
    assert selection.status == "SELECTED"
    assert selection.option_id == "OPTION_X"
    assert selector.last_record["selection_valid"]
    assert "OPTION_X" in client.kwargs["prompt"]


def test_qwen_valid_structured_output_matches_exact_bounded_schema():
    client = FakeQwenClient('{"selection":"OPTION_X"}')
    selector = QwenSelectorAdapter(client, "instruction")
    selection = selector.select(_state(), [_option()])
    assert selection.status == "SELECTED"
    assert selection.option_id == "OPTION_X"
    assert client.kwargs["schema"]["additionalProperties"] is False
    assert client.kwargs["schema"]["required"] == ["selection"]


def test_instruct_selector_parses_structured_choice_and_records_output_tokens():
    client = FakeQwenClient('{"selection":"OPTION_X"}')
    state = _state()
    state = BeliefState(**{
        **state.__dict__,
        "relevant_geometry": {"target_offset_xyz_m": [0.02, 0.0, 0.0], "target_visible": True},
    })
    selector = QwenSelectorAdapter(client, "align to the target", max_tokens=64)
    selection = selector.select(state, [_option()])
    assert selection.status == "SELECTED"
    assert client.kwargs["max_tokens"] == 64
    assert client.kwargs["schema"]["additionalProperties"] is False
    assert selector.last_record["output_tokens"] == 8
    assert selector.last_record["prompt_tokens"] == 91
    assert "target_offset_xyz_m" in client.kwargs["prompt"]


@pytest.mark.parametrize("raw", ["MV_LEFT", "move left"])
def test_qwen_invalid_raw_motion_and_natural_language_fail_closed(raw):
    selection = QwenSelectorAdapter(FakeQwenClient(raw), "instruction").select(
        _state(), [_option()]
    )
    assert selection.status == "INVALID_SELECTION"
    assert selection.option_id == "INVALID_SELECTION"


def test_c_qwen_adapter_rejects_nonexistent_option():
    client = FakeQwenClient('{"selection":"OPTION_999"}')
    selection = QwenSelectorAdapter(client, "instruction").select(_state(), [_option()])
    assert selection.status == "INVALID_SELECTION"
    assert selection.option_id == "INVALID_SELECTION"


def test_d_qwen_adapter_rejects_raw_motion_command():
    client = FakeQwenClient("MV_LEFT")
    selection = QwenSelectorAdapter(client, "instruction").select(_state(), [_option()])
    assert selection.status == "INVALID_SELECTION"
    assert selector_not_raw_action(selection.option_id)


def selector_not_raw_action(option_id: str) -> bool:
    return option_id not in {"MV_LEFT", "MV_RIGHT", "MV_FWD", "MV_BACK", "MV_UP", "MV_DOWN"}


def test_e_smoke_option_is_bounded_and_evidence_scoped():
    state = _state()
    # Add the same evidence shape supplied by the actual LIBERO observer.
    state = BeliefState(**{**state.__dict__, "evidence_refs": ("frame-1",)})
    option = SmokeOptionGenerator().generate(state)[0]
    assert option.option_id == "OPTION_SAFE_LIFT"
    assert option.primitive.token == "MV_UP"
    assert option.primitive.parameters["step_m"] == 0.005
    assert option.primitive.max_steps == 1
    assert option.evidence_frame_id == state.frame_id
    assert option.evidence == state.evidence_refs


def test_f_arbiter_approves_legal_smoke_option():
    state = _state()
    state = BeliefState(**{**state.__dict__, "evidence_refs": ("frame-1",)})
    option = SmokeOptionGenerator().generate(state)[0]
    decision = Arbiter().authorize(state, [option],
                                   SimpleNamespace(option_id=option.option_id, status="SELECTED"))
    assert decision.kind == DecisionKind.APPROVED
    assert decision.action is not None


def test_g_effect_observer_measures_fresh_eef_delta():
    before = BeliefState(end_effector_state={"position_xyz": [0.2, 0.1, 0.35]})
    after = BeliefState(end_effector_state={"position_xyz": [0.2, 0.1, 0.351]},
                        observation_fresh=True)
    effect = EffectObserver().compare(
        before,
        {"end_effector_delta_xyz": [0.0, 0.0, 0.005], "minimum_delta_projection_m": 0.0001},
        after,
    )
    assert effect.before_eef_position == (0.2, 0.1, 0.35)
    assert effect.after_eef_position == (0.2, 0.1, 0.351)
    assert effect.expected_delta == (0.0, 0.0, 0.005)
    assert effect.observed_delta == pytest.approx((0.0, 0.0, 0.001))
    assert effect.achieved is True


def test_h_runner_source_has_no_environment_step_bypass():
    source = (ROOT / "core/runtime_v3/runner.py").read_text()
    assert "environment.step(" not in source
    assert "env.step(" not in source


def test_i_qwen_adapter_holds_no_environment_or_controller_reference():
    selector = QwenSelectorAdapter(FakeQwenClient('{"selection":"REOBSERVE"}'), "instruction")
    forbidden = {"env", "environment", "controller", "backend", "executor"}
    assert not forbidden.intersection(vars(selector))


def test_selector_evaluation_script_has_no_executor_construction_or_import():
    source = (ROOT / "scripts/runtime_v3_qwen_selector_eval.py").read_text()
    assert "from core.runtime_v3.executor" not in source
    assert "Executor(" not in source
    assert "LiberoEnvironmentAdapter" not in source


def test_task_conditioned_selector_cases_reuse_same_state_and_options_with_different_answers():
    from scripts.runtime_v3_qwen_selector_eval import _choices_for_case

    first = _choices_for_case(8)
    second = _choices_for_case(9)
    assert first[1] == second[1]
    assert first[2] == second[2]
    assert first[3] != second[3]
    assert first[5] == second[5] == "PAIR_1"

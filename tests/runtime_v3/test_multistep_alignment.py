from __future__ import annotations

import base64
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from core.capabilities.camera_geometry import CameraCalibration
from core.runtime_v3.arbiter import Arbiter, DecisionKind
from core.runtime_v3.effects import EffectObserver
from core.runtime_v3.executor import Executor
from core.runtime_v3.micro_motion import BoundedMicroMotionOptionGenerator
from core.runtime_v3.object_relative import (
    ObjectRelativeAlignmentOptionGenerator,
    ObjectRelativePerceptionObserver,
)
from core.runtime_v3.observer import RobotObservation
from core.runtime_v3.options import BoundedMicroMotionSpec, PrimitiveCommand, RuntimeOption
from core.runtime_v3.runner import RuntimeV3Runner
from core.runtime_v3.selector import Selection
from core.runtime_v3.state import BeliefState, ObjectRelativeState, StateBuilder
from scripts import runtime_v3_multistep_alignment as multi

ROOT = Path(__file__).resolve().parents[2]
UNITS = {
    "FWD": (1.0, 0.0, 0.0), "BACK": (-1.0, 0.0, 0.0),
    "LEFT": (0.0, 1.0, 0.0), "RIGHT": (0.0, -1.0, 0.0),
    "UP": (0.0, 0.0, 1.0), "DOWN": (0.0, 0.0, -1.0),
}


def _candidate(direction, improvement=1.0, *, valid=True):
    return {
        "direction": direction, "valid": valid,
        "hypothetical_projection_px": [10.0, 20.0],
        "predicted_error_after_px": 99.0,
        "predicted_improvement_px": improvement,
        "reason": None if valid else "workspace_boundary",
    }


def _state(*, visible=True, identity="SAME_TARGET", reference_valid=True,
           invalidation=None, candidates=None, scene_ready=True):
    relative = ObjectRelativeState(
        target_phrase="salad dressing", target_visible=visible,
        target_identity_status=identity, target_reference_point_px=(20.0, 30.0),
        target_reference_valid=reference_valid,
        target_reference_invalidation_reason=invalidation,
        eef_projection_px=(120.0, 30.0), image_error_norm_px=100.0,
        scene_ready=scene_ready, scene_ready_gate_enabled=True,
    )
    return BeliefState(
        step_id=1, object_relative_state=relative,
        end_effector_state={"position_xyz": [0.2, 0.1, 0.3]},
        relevant_geometry={
            "workspace_valid": True, "workspace_z_bounds_m": [0.02, 0.60],
            "camera_projection_valid": True,
            "sam3_candidate_count": (1 if identity == "TARGET_IDENTITY_LOST" else 0),
            "candidate_directions": list(candidates if candidates is not None else
                                          [_candidate("FWD")]),
        }, evidence_refs=("frame-1",), frame_id=1, observation_fresh=True,
    )


class _Backend:
    def __init__(self, position=(0.2, 0.1, 0.3)):
        self.position = list(position)
        self.directions = []
        self.calls = 0

    def execute_approved_micro_tick(self, action):
        direction = action.primitive.micro_motion_spec.direction
        self.directions.append(direction)
        self.calls += 1
        unit = action.primitive.micro_motion_spec.direction_unit
        for i in range(3):
            self.position[i] += unit[i] * 0.0031
        return {"tick": self.calls}

    def execute_approved_action(self, _action):
        raise AssertionError("unexpected non-micro-motion action")


def _approved(arbiter=None, direction="LEFT", position=(0.2, 0.1, 0.3)):
    state = BeliefState(
        task_id="LIBERO_OBJECT:2", step_id=1,
        end_effector_state={"position_xyz": list(position)},
        relevant_geometry={"workspace_valid": True, "workspace_z_bounds_m": [0.02, 0.60]},
        evidence_refs=("frame-1",), frame_id=1, observation_fresh=True,
    )
    option_generator = BoundedMicroMotionOptionGenerator(direction, UNITS[direction])
    selected_arbiter = arbiter if arbiter is not None else Arbiter()
    decision = selected_arbiter.authorize(
        state, option_generator.generate(state), Selection(option_generator.option_id),
    )
    assert decision.kind == DecisionKind.APPROVED and decision.action is not None
    return selected_arbiter, decision.action


def _approved_alignment(arbiter, direction, position=(0.2, 0.1, 0.3)):
    candidate = {
        "direction": direction, "direction_unit": list(UNITS[direction]),
        "predicted_error_after_px": 99.0, "predicted_improvement_px": 1.0,
        "hypothetical_projection_px": [21.0, 30.0], "valid": True,
    }
    relative = ObjectRelativeState(
        "salad dressing", True, target_centroid_px=(20.0, 30.0),
        eef_projection_px=(120.0, 30.0), target_identity_status="SAME_TARGET",
        target_reference_point_px=(20.0, 30.0), target_reference_valid=True,
    )
    state = BeliefState(
        step_id=1, object_relative_state=relative,
        end_effector_state={"position_xyz": list(position)},
        relevant_geometry={"workspace_valid": True,
                           "workspace_z_bounds_m": [0.02, 0.60],
                           "camera_projection_valid": True,
                           "object_relative_alignment_valid": True,
                           "pixel_error_before_px": 100.0,
                           "candidate_directions": [candidate],
                           "chosen_candidate": candidate},
        evidence_refs=("frame-1",), frame_id=1, observation_fresh=True,
    )
    options = ObjectRelativeAlignmentOptionGenerator().generate(state)
    decision = arbiter.authorize(state, options, Selection("ALIGN_TO_TARGET_SMALL"))
    assert decision.kind == DecisionKind.APPROVED and decision.action is not None
    return decision.action


def test_each_alignment_step_requires_a_new_arbiter_authorization():
    arbiter = multi.CountingArbiter(before_authorize=lambda *_args: None)
    backend = _Backend()
    executor = Executor(backend, arbiter)
    for direction in ("DOWN", "RIGHT"):
        action = _approved_alignment(arbiter, direction)
        executor.execute(
            action,
            tick_observer=lambda: RobotObservation(
                "tick", backend.calls,
                proprioception={"end_effector_state": {"position_xyz": backend.position}},
                fresh=True,
            ),
        )
    assert arbiter.alignment_authorization_calls == 2
    assert arbiter.approval_count == 2
    assert backend.directions == ["DOWN", "RIGHT"]


def test_executor_rejects_a_second_semantic_execution_under_the_same_approval():
    arbiter, action = _approved()
    backend = _Backend()
    executor = Executor(backend, arbiter)
    observe = lambda: RobotObservation(
        "tick", backend.calls,
        proprioception={"end_effector_state": {"position_xyz": backend.position}}, fresh=True,
    )
    executor.execute(action, tick_observer=observe)
    calls_after_first_execution = backend.calls
    replay_executor = Executor(backend, arbiter)
    with pytest.raises(TypeError, match="only once"):
        replay_executor.execute(action, tick_observer=observe)
    assert backend.calls == calls_after_first_execution


def test_geometry_is_recomputed_after_each_fresh_observation(monkeypatch):
    import core.runtime_v3.object_relative as module

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:30, 20:30] = True
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    class BaseObserver:
        def __init__(self):
            self.index = 0
            self.last_raw = None

        def observe(self, _environment):
            self.index += 1
            position = (0.0 + self.index * 0.001, 0.0, -1.0)
            self.last_raw = SimpleNamespace(eef_position_xyz=position)
            return RobotObservation(
                f"obs-{self.index}", self.index, images={"agentview": image},
                proprioception={"end_effector_state": {"position_xyz": position}},
                evidence={"relevant_geometry": {"workspace_valid": True,
                                                   "workspace_z_bounds_m": [-2.0, 1.0]}},
                fresh=True,
            )

    class Sam:
        def segment(self, _image, _phrase, **_kwargs):
            return {"success": True, "details": {"metadata": {"image_size": [64, 64]},
                    "detections": [{"mask": {"format": "png", "base64": encoded}}]}}

    calibration = CameraCalibration(
        name="agentview", width=64, height=64, fovy_deg=60.0,
        position_world=np.zeros(3), camera_to_world=np.eye(3),
    )
    monkeypatch.setattr(module, "make_mujoco_calibrations",
                        lambda *_args, **_kwargs: {"agentview": calibration})
    original = module.resolve_object_relative_geometry
    seen = []

    def track_geometry(**kwargs):
        seen.append(tuple(kwargs["eef_position_xyz_m"]))
        return original(**kwargs)

    monkeypatch.setattr(module, "resolve_object_relative_geometry", track_geometry)
    observer = ObjectRelativePerceptionObserver(
        BaseObserver(), Sam(), target_phrase="salad dressing", move_vectors=UNITS,
    )
    observer.observe(object())
    observer.observe(object())
    assert len(seen) == 2
    assert seen[0] != seen[1]
    assert len(observer.last_resolution["candidate_directions"]) == 6


def test_direction_can_change_between_successive_runtime_steps():
    class Environment:
        def __init__(self):
            self.position = [0.2, 0.1, 0.3]

    class Observer:
        def __init__(self, backend):
            self.backend = backend
            self.frame = 0

        def observe(self, _environment):
            self.frame += 1
            return RobotObservation(
                f"obs-{self.frame}", self.frame,
                proprioception={"end_effector_state": {"position_xyz": self.backend.position}},
                evidence={"relevant_geometry": {
                    "workspace_valid": True, "workspace_z_bounds_m": [0.02, 0.60],
                }}, evidence_refs=(f"frame-{self.frame}",), fresh=True,
            )

    class SequenceGenerator:
        def __init__(self):
            self.directions = iter(("DOWN", "RIGHT"))

        def generate(self, state):
            direction = next(self.directions)
            spec = BoundedMicroMotionSpec(direction, UNITS[direction])
            return [RuntimeOption(
                option_id="ALIGN_TO_TARGET_SMALL", option_type="verified_alignment",
                description="one test-only bounded alignment step", preconditions={},
                expected_effect={},
                primitive=PrimitiveCommand(kind="micro_motion", max_steps=1,
                                           max_duration_s=5.0, micro_motion_spec=spec),
                confidence=1.0,
            )]

    class Selector:
        def select(self, _state, options):
            return Selection("ALIGN_TO_TARGET_SMALL")

    environment = Environment()
    backend = _Backend()
    observer = Observer(backend)
    arbiter = multi.CountingArbiter(before_authorize=lambda *_args: None)
    executor = Executor(backend, arbiter)
    generator = SequenceGenerator()
    for _ in range(2):
        runner = RuntimeV3Runner(
            observer=observer, state_builder=StateBuilder(), option_generator=generator,
            selector=Selector(), arbiter=arbiter, executor=executor,
            effect_observer=EffectObserver(),
        )
        result = runner.run_episode(environment, task_id="task", max_steps=1, reset=False)
        assert result["actions"] == 1
    assert backend.directions == ["DOWN", "RIGHT"]
    assert arbiter.approval_count == 2


def test_fixed_reference_comparison_does_not_follow_sam_centroid():
    frozen_reference = (30.5, 40.5)
    post_sam_centroid = (35.5, 41.5)
    assert not multi.same_fixed_reference(frozen_reference, post_sam_centroid)
    assert multi.same_fixed_reference(frozen_reference, (30.5, 40.5))


def test_post_sam_centroid_cannot_overwrite_observer_reference_anchor(monkeypatch):
    import core.runtime_v3.object_relative as module

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    masks = []
    for top in (20, 22):
        mask = np.zeros((64, 64), dtype=bool)
        mask[top:top + 10, 20:30] = True
        buffer = io.BytesIO()
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(buffer, format="PNG")
        masks.append(base64.b64encode(buffer.getvalue()).decode("ascii"))

    class Base:
        last_raw = SimpleNamespace(eef_position_xyz=(0.0, 0.0, -1.0))
        frame = 0

        def observe(self, _environment):
            self.frame += 1
            self.last_raw = SimpleNamespace(eef_position_xyz=(0.0, 0.0, -1.0))
            return RobotObservation(
                f"obs-{self.frame}", self.frame, images={"agentview": image},
                proprioception={"end_effector_state": {"position_xyz": [0.0, 0.0, -1.0]}},
                evidence={"relevant_geometry": {"workspace_valid": True,
                                                   "workspace_z_bounds_m": [-2.0, 1.0]}},
                fresh=True,
            )

    class Sam:
        index = 0

        def segment(self, _image, _phrase, **_kwargs):
            value = masks[min(self.index, len(masks) - 1)]
            self.index += 1
            return {"success": True, "details": {"metadata": {"image_size": [64, 64]},
                    "detections": [{"mask": {"format": "png", "base64": value}}]}}

    calibration = CameraCalibration(
        name="agentview", width=64, height=64, fovy_deg=60.0,
        position_world=np.zeros(3), camera_to_world=np.eye(3),
    )
    monkeypatch.setattr(module, "make_mujoco_calibrations",
                        lambda *_args, **_kwargs: {"agentview": calibration})
    observer = ObjectRelativePerceptionObserver(Base(), Sam(), target_phrase="salad dressing",
                                                 move_vectors=UNITS)
    first = observer.observe(object())
    second = observer.observe(object())
    initial_reference = first.evidence["object_relative_state"].target_reference_point_px
    assert second.evidence["object_relative_state"].target_centroid_px != initial_reference
    assert second.evidence["object_relative_state"].target_reference_point_px == initial_reference
    assert observer.reference_anchor.reference_point_px == initial_reference


def test_nonpositive_effect_stops_the_episode():
    state = _state(candidates=[_candidate("FWD", improvement=0.5)])
    assert multi.post_action_termination(
        alignment_step=1, actual_improvement_px=0.0, state_after=state,
    ) == "EFFECT_NOT_IMPROVED"
    assert multi.post_action_termination(
        alignment_step=1, actual_improvement_px=-0.1, state_after=state,
    ) == "EFFECT_NOT_IMPROVED"


def test_no_positive_candidate_stops_without_claiming_alignment():
    state = _state(candidates=[_candidate("DOWN", improvement=0.0),
                               _candidate("UP", improvement=-0.1)])
    assert multi.classify_pre_action_stop(state) == "NO_POSITIVE_OPTION"


def test_six_step_budget_is_enforced_without_completion_threshold():
    assert multi.MAX_ALIGNMENT_STEPS == 6
    assert len(multi.INIT_STATES) * multi.MAX_ALIGNMENT_STEPS == 36
    state = _state(candidates=[_candidate("FWD", improvement=0.5)])
    assert multi.post_action_termination(
        alignment_step=6, actual_improvement_px=0.2, state_after=state,
    ) == "MAX_ALIGNMENT_STEPS"
    assert not hasattr(multi, "ALIGNMENT_COMPLETE_THRESHOLD_PX")


def test_target_identity_loss_stops_the_episode():
    state = _state(visible=False, identity="TARGET_IDENTITY_LOST",
                   candidates=[_candidate("FWD")])
    assert multi.classify_pre_action_stop(state) == "TARGET_IDENTITY_LOST"


def test_target_visibility_loss_stops_the_episode():
    state = _state(visible=False, identity="SAME_TARGET",
                   candidates=[_candidate("FWD")])
    assert multi.classify_pre_action_stop(state) == "TARGET_VISIBILITY_LOST"


def test_invalid_target_reference_stops_the_episode():
    state = _state(reference_valid=False, invalidation="camera_changed",
                   candidates=[_candidate("FWD")])
    assert multi.classify_pre_action_stop(state) == "REFERENCE_INVALID"


def test_qwen_selector_has_no_controller_or_backend_access():
    selector = multi.AlignmentSelector()
    assert not {"controller", "backend", "executor", "environment"}.intersection(vars(selector))
    assert "QwenSelectorAdapter" not in Path(multi.__file__).read_text()


def test_runtime_runner_routes_actions_through_executor_only():
    source = (ROOT / "core/runtime_v3/runner.py").read_text()
    assert "self.executor.execute(" in source
    assert "environment.step(" not in source
    assert "env.step(" not in source


def test_oracle_diagnostic_cannot_change_runtime_candidate_selection():
    state = _state(candidates=[_candidate("DOWN", improvement=0.3),
                               _candidate("RIGHT", improvement=0.2)])
    before = multi.classify_pre_action_stop(state)
    unused_oracle_a = {"world_position_m": [0.1, 0.2, 0.3]}
    unused_oracle_b = {"world_position_m": [9.0, -3.0, 2.0]}
    assert unused_oracle_a != unused_oracle_b
    assert multi.classify_pre_action_stop(state) == before is None
    assert "oracle" not in multi.ObjectRelativeAlignmentOptionGenerator.generate.__code__.co_names


def test_error_trajectory_and_normalization_are_logged_correctly():
    errors = [100.0, 98.0, 96.0]
    assert multi.normalized_trajectory(errors) == [1.0, 0.98, 0.96]
    assert multi.normalized_trajectory([None, 2.0]) == [None, None]


def test_candidate_record_keeps_all_six_directions_and_predictions():
    directions = ("FWD", "BACK", "LEFT", "RIGHT", "UP", "DOWN")
    records = multi.candidate_records([_candidate(direction, i * 0.1, valid=i != 2)
                                       for i, direction in enumerate(directions)])
    assert [row["direction"] for row in records] == list(directions)
    assert all("predicted_eef_projection_px" in row for row in records)
    assert all("predicted_error_px" in row for row in records)
    assert all("predicted_improvement_px" in row for row in records)
    assert records[2]["valid"] is False


def test_direction_reversal_and_transition_metrics_are_explicit():
    sequence = ["DOWN", "DOWN", "UP", "LEFT", "RIGHT"]
    assert multi.count_direction_reversals(sequence) == 2
    assert sum(left != right for left, right in zip(sequence, sequence[1:])) == 3


def test_aggregate_trajectory_plot_is_written(tmp_path):
    episodes = [
        {"init_state_index": 0, "error_trajectory_px": [100.0, 90.0]},
        {"init_state_index": 1, "error_trajectory_px": [200.0, 180.0, 160.0]},
    ]
    result = multi.save_error_trajectory(episodes, tmp_path / "trajectory.png")
    assert Path(result["path"]).is_file()
    assert result["mean_normalized_error_by_step"] == [1.0, 0.9, 0.8]


def test_reference_error_uses_frozen_reference_and_eef_projection():
    state = _state()
    assert multi.error_from_state(state) == 100.0
    invalid = _state(reference_valid=False)
    assert multi.error_from_state(invalid) is None

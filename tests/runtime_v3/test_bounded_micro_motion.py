from __future__ import annotations

from pathlib import Path

from core.runtime_v3.arbiter import Arbiter, DecisionKind
from core.runtime_v3.effects import EffectObserver
from core.runtime_v3.executor import Executor
from core.runtime_v3.micro_motion import BoundedMicroMotionOptionGenerator
from core.runtime_v3.observer import RobotObservation
from core.runtime_v3.runner import RuntimeV3Runner
from core.runtime_v3.selector import DeterministicSelector, Selection
from core.runtime_v3.state import BeliefState, StateBuilder


ROOT = Path(__file__).resolve().parents[2]


def _state(position=(0.2, 0.1, 0.3)):
    return BeliefState(
        task_id="LIBERO_OBJECT:2",
        step_id=1,
        end_effector_state={"position_xyz": list(position)},
        relevant_geometry={
            "workspace_valid": True,
            "workspace_z_bounds_m": [0.02, 0.60],
        },
        evidence_refs=("frame-1",),
        frame_id=1,
        observation_fresh=True,
    )


def _approved(direction="LEFT", position=(0.2, 0.1, 0.3), *, max_ticks=5):
    state = _state(position)
    units = {
        "FWD": (1.0, 0.0, 0.0), "BACK": (-1.0, 0.0, 0.0),
        "LEFT": (0.0, 1.0, 0.0), "RIGHT": (0.0, -1.0, 0.0),
        "UP": (0.0, 0.0, 1.0), "DOWN": (0.0, 0.0, -1.0),
    }
    generator = BoundedMicroMotionOptionGenerator(
        direction, units[direction], max_ticks=max_ticks,
    )
    options = generator.generate(state)
    arbiter = Arbiter()
    decision = arbiter.authorize(state, options, Selection(generator.option_id))
    assert decision.kind == DecisionKind.APPROVED
    assert decision.action is not None
    return arbiter, decision.action, generator


class _MotionBackend:
    def __init__(self, start, deltas):
        self.position = list(start)
        self.deltas = [tuple(delta) for delta in deltas]
        self.directions = []
        self.action_ids = []
        self.calls = 0

    def execute_approved_micro_tick(self, action):
        self.calls += 1
        self.directions.append(action.primitive.micro_motion_spec.direction)
        self.action_ids.append(id(action))
        delta = self.deltas[min(self.calls - 1, len(self.deltas) - 1)]
        self.position = [self.position[i] + delta[i] for i in range(3)]
        return {"tick": self.calls}

    def execute_approved_action(self, _action):
        raise AssertionError("micro-motion bypassed the bounded tick loop")


def _observe(backend, frame):
    return RobotObservation(
        observation_id=f"obs-{frame}",
        frame_id=frame,
        proprioception={"end_effector_state": {"position_xyz": list(backend.position)}},
        evidence={"relevant_geometry": {
            "workspace_valid": True,
            "workspace_z_bounds_m": [0.02, 0.60],
        }},
        evidence_refs=(f"frame-{frame}",),
        fresh=True,
    )


def test_arbiter_approves_one_bounded_micro_motion_with_start_state():
    _arbiter, action, generator = _approved()
    assert action.option_id == "MOVE_LEFT_SMALL"
    assert action.primitive.micro_motion_spec.requested_displacement_m == 0.003
    assert action.primitive.micro_motion_spec.max_ticks == 5
    assert action.start_eef_position_xyz_m == (0.2, 0.1, 0.3)
    assert generator.option_id == "MOVE_LEFT_SMALL"


def test_executor_repeats_only_the_approved_direction():
    arbiter, action, _ = _approved()
    backend = _MotionBackend((0.2, 0.1, 0.3), [(0.0, 0.0011, 0.0)] * 5)
    frame = 1

    def observe():
        nonlocal frame
        frame += 1
        return _observe(backend, frame)

    result = Executor(backend, arbiter).execute(action, tick_observer=observe).result
    assert backend.directions == ["LEFT", "LEFT", "LEFT"]
    assert len(set(backend.action_ids)) == 1
    assert result["termination"] == "TARGET_REACHED"


def test_executor_stops_immediately_when_target_projection_is_reached():
    arbiter, action, _ = _approved()
    backend = _MotionBackend((0.2, 0.1, 0.3), [(0.0, 0.0031, 0.0)] * 5)
    frame = 1
    result = Executor(backend, arbiter).execute(
        action,
        tick_observer=lambda: _observe(backend, 2),
    ).result
    assert backend.calls == 1
    assert result["termination"] == "TARGET_REACHED"
    assert result["actual_projection_mm"] >= 3.0


def test_executor_stops_at_max_ticks_below_requested_displacement():
    arbiter, action, _ = _approved()
    backend = _MotionBackend((0.2, 0.1, 0.3), [(0.0, 0.0004, 0.0)] * 5)
    frame = iter(range(2, 8))
    result = Executor(backend, arbiter).execute(
        action, tick_observer=lambda: _observe(backend, next(frame)),
    ).result
    assert backend.calls == 5
    assert result["termination"] == "MAX_TICKS_REACHED"
    assert result["actual_projection_m"] < 0.003


def test_negative_progress_stops_without_opposite_direction_correction():
    arbiter, action, _ = _approved()
    backend = _MotionBackend(
        (0.2, 0.1, 0.3), [(0.0, 0.001, 0.0), (0.0, -0.0005, 0.0)],
    )
    frame = iter((2, 3, 4))
    result = Executor(backend, arbiter).execute(
        action, tick_observer=lambda: _observe(backend, next(frame)),
    ).result
    assert result["termination"] == "NEGATIVE_PROGRESS"
    assert backend.directions == ["LEFT", "LEFT"]
    assert backend.calls == 2


def test_runner_source_has_no_direct_environment_step_path():
    source = (ROOT / "core/runtime_v3/runner.py").read_text()
    assert "environment.step(" not in source
    assert "env.step(" not in source
    assert "self.executor.execute(" in source


def test_each_executor_tick_observation_and_diagnostics_are_recorded():
    arbiter, action, _ = _approved()
    backend = _MotionBackend((0.2, 0.1, 0.3), [(0.0, 0.0011, 0.0002)] * 5)
    frame = iter((2, 3, 4))
    result = Executor(backend, arbiter).execute(
        action, tick_observer=lambda: _observe(backend, next(frame)),
    ).result
    ticks = result["tick_observations"]
    assert len(ticks) == result["ticks_executed"] == 3
    assert [row["frame_id"] for row in ticks] == [2, 3, 4]
    assert all("incremental_projection_mm" in row for row in ticks)
    assert all("off_axis_magnitude_m" in row for row in ticks)
    assert all("eef_position_xyz_m" in row for row in ticks)
    assert result["execution_duration_s"] >= 0


def test_one_bounded_motion_uses_exactly_one_arbiter_approval():
    class CountingArbiter(Arbiter):
        def __init__(self):
            super().__init__()
            self.approvals = 0

        def authorize(self, *args, **kwargs):
            self.approvals += 1
            return super().authorize(*args, **kwargs)

    class Environment:
        def reset(self):
            self.position = [0.2, 0.1, 0.3]

    class Observer:
        def __init__(self):
            self.frame = 0

        def observe(self, environment):
            self.frame += 1
            return RobotObservation(
                f"obs-{self.frame}", self.frame,
                proprioception={"end_effector_state": {"position_xyz": environment.position}},
                evidence={"relevant_geometry": {
                    "workspace_valid": True,
                    "workspace_z_bounds_m": [0.02, 0.60],
                }},
                evidence_refs=(f"frame-{self.frame}",), fresh=True,
            )

    class Backend:
        def __init__(self, environment):
            self.environment = environment
            self.calls = 0
            self.directions = []

        def execute_approved_micro_tick(self, action):
            self.calls += 1
            self.directions.append(action.primitive.micro_motion_spec.direction)
            self.environment.position[1] += 0.0011
            return {"tick": self.calls}

        def execute_approved_action(self, _action):
            raise AssertionError("unexpected non-micro action")

    environment = Environment()
    arbiter = CountingArbiter()
    backend = Backend(environment)
    generator = BoundedMicroMotionOptionGenerator("LEFT", (0.0, 1.0, 0.0))
    runner = RuntimeV3Runner(
        observer=Observer(), state_builder=StateBuilder(), option_generator=generator,
        selector=DeterministicSelector(generator.option_id), arbiter=arbiter,
        executor=Executor(backend, arbiter), effect_observer=EffectObserver(),
    )
    result = runner.run_episode(environment, task_id="task", max_steps=1)
    assert result["actions"] == 1
    assert arbiter.approvals == 1
    assert backend.calls == 3
    assert backend.directions == ["LEFT", "LEFT", "LEFT"]


def test_qwen_selector_chooses_semantic_option_without_controller_access():
    from core.runtime_v3.adapters.qwen_selector import QwenSelectorAdapter
    from core.vlm.vlm_client import VLMResponse

    class Client:
        model = "Qwen/mock"

        def complete_json(self, _prompt, **_kwargs):
            return VLMResponse(
                "", '{"selection":"MOVE_LEFT_SMALL"}',
                {"raw": {"choices": [{"message": {"content": '{"selection":"MOVE_LEFT_SMALL"}'}}]}},
            )

    state = _state()
    generator = BoundedMicroMotionOptionGenerator("LEFT", (0.0, 1.0, 0.0))
    selector = QwenSelectorAdapter(Client(), "move left")
    selection = selector.select(state, generator.generate(state))
    assert selection.option_id == "MOVE_LEFT_SMALL"
    assert selection.status == "SELECTED"
    assert not {"controller", "backend", "executor", "environment"}.intersection(vars(selector))


def test_workspace_boundary_stops_before_issuing_the_next_control_tick():
    state = BeliefState(
        step_id=1,
        end_effector_state={"position_xyz": [0.2, 0.1, 0.599]},
        relevant_geometry={"workspace_valid": True, "workspace_z_bounds_m": [0.02, 0.60]},
        evidence_refs=("frame-1",), frame_id=1, observation_fresh=True,
    )
    generator = BoundedMicroMotionOptionGenerator("UP", (0.0, 0.0, 1.0))
    arbiter = Arbiter()
    action = arbiter.authorize(state, generator.generate(state), Selection(generator.option_id)).action
    assert action is not None
    backend = _MotionBackend((0.2, 0.1, 0.599), [(0.0, 0.0, 0.001)])
    result = Executor(backend, arbiter).execute(
        action, tick_observer=lambda: _observe(backend, 2),
    ).result
    assert result["termination"] == "BOUNDARY_STOP"
    assert result["ticks_executed"] == 0
    assert backend.calls == 0

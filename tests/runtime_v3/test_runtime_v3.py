from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from core.runtime_v3.arbiter import Arbiter, ApprovedAction, DecisionKind
from core.runtime_v3.effects import EffectObserver
from core.runtime_v3.executor import Executor
from core.runtime_v3.observer import RobotObservation
from core.runtime_v3.options import OptionGenerator, PrimitiveCommand, RuntimeOption
from core.runtime_v3.runner import RuntimeV3Runner
from core.runtime_v3.selector import Selection
from core.runtime_v3.state import BeliefState, StateBuilder


ROOT = Path(__file__).resolve().parents[2]


class FakeBackend:
    def __init__(self):
        self.calls = []

    def execute_primitive(self, primitive):
        self.calls.append(primitive)
        return {"ok": True}


def test_option_generator_has_no_executor_dependency():
    tree = ast.parse((ROOT / "core/runtime_v3/options.py").read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all("executor" not in (node.module or "") for node in imports if isinstance(node, ast.ImportFrom))
    assert "Executor" not in (ROOT / "core/runtime_v3/options.py").read_text()


def test_invalid_selector_id_never_executes():
    class Environment:
        def reset(self):
            pass

    class ObserverStub:
        def observe(self, _environment):
            candidate = {"option_id": "OPTION_0", "option_type": "move",
                         "description": "move", "primitive": {"kind": "move", "token": "MV_LEFT"},
                         "evidence_frame_id": 1}
            return RobotObservation("obs", 1,
                                    evidence={"relevant_geometry": {"option_candidates": [candidate]}},
                                    fresh=True)

    class InvalidSelector:
        def select(self, _state, _options):
            return Selection("BOGUS", status="INVALID_SELECTION")

    arbiter = Arbiter()
    backend = FakeBackend()
    runner = RuntimeV3Runner(
        observer=ObserverStub(), state_builder=StateBuilder(), option_generator=OptionGenerator(),
        selector=InvalidSelector(), arbiter=arbiter, executor=Executor(backend, arbiter),
        effect_observer=EffectObserver(),
    )
    result = runner.run_episode(Environment(), task_id="invalid", max_steps=2)
    assert result["status"] == DecisionKind.INVALID_SELECTION.value
    assert backend.calls == []


def test_executor_rejects_unapproved_action():
    arbiter = Arbiter()
    backend = FakeBackend()
    executor = Executor(backend, arbiter)
    forged = ApprovedAction("OPTION_0", PrimitiveCommand("move", "MV_LEFT"), {}, 0, 1)
    with pytest.raises(TypeError):
        executor.execute(forged)
    assert backend.calls == []


def test_robot_controller_is_called_only_after_arbiter_authorize():
    state = BeliefState(frame_id=4, observation_fresh=True, evidence_refs=("frame-4",))
    option = RuntimeOption("OPTION_0", "move", "one move", {}, {"stage": "next"},
                           PrimitiveCommand("move", "MV_LEFT"), 0.9,
                           evidence=("frame-4",), evidence_frame_id=4)
    arbiter = Arbiter()
    backend = FakeBackend()
    executor = Executor(backend, arbiter)
    assert backend.calls == []
    decision = arbiter.authorize(state, [option], Selection("OPTION_0"))
    assert decision.kind == DecisionKind.APPROVED
    executor.execute(decision.action)
    assert len(backend.calls) == 1


def test_state_builder_returns_one_canonical_belief_state():
    builder = StateBuilder()
    state = builder.initialize("task")
    observation = RobotObservation("obs-1", 1, evidence={"stage": "approach"}, fresh=True)
    updated = builder.update(state, observation)
    assert isinstance(updated, BeliefState)
    assert updated.stage == "approach"
    assert updated.step_id == state.step_id + 1


def test_runtime_v3_has_no_legacy_policy_imports():
    forbidden = ("visual_route", "recursive", "recovery", "verified_runtime",
                 "runtime_v2", "mem_text", "placement_review")
    for path in (ROOT / "core/runtime_v3").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported = (node.module or "").lower()
                assert not any(name in imported for name in forbidden), (path, imported)
            elif isinstance(node, ast.Import):
                assert not any(name in alias.name.lower() for alias in node.names for name in forbidden), path


def test_legacy_runner_and_core_sources_match_frozen_tag():
    paths = ["scripts/run_libero_zeroshot.py", "core/sim/zeroshot_libero_runner.py",
             "core/capabilities/visual_harness.py", "core/capabilities/verified_runtime.py",
             "plugins/visual_route/plugin.py", "plugins/recovery/plugin.py"]
    subprocess.run(["git", "diff", "--exit-code", "legacy-full-harness-0928", "--", *paths],
                   cwd=ROOT, check=True, capture_output=True, text=True)


def test_mock_one_step_smoke_cycle():
    class Environment:
        def reset(self):
            self.reset_called = True

    class ObserverStub:
        def __init__(self):
            self.frame = 0

        def observe(self, _environment):
            self.frame += 1
            candidates = []
            if self.frame == 1:
                candidates = [{
                    "option_id": "OPTION_0", "option_type": "probe",
                    "description": "bounded mock action", "confidence": 1.0,
                    "evidence_frame_id": 1,
                    "primitive": {"kind": "hold", "max_steps": 1},
                    "expected_effect": {"stage": "observed"},
                }]
            return RobotObservation(
                f"obs-{self.frame}", self.frame,
                evidence={"stage": "observed", "relevant_geometry": {"option_candidates": candidates}},
                evidence_refs=(f"frame-{self.frame}",), fresh=True,
            )

    class OneStepSelector:
        def select(self, _state, options):
            return Selection(options[0].option_id if options else "REOBSERVE")

    environment = Environment()
    arbiter = Arbiter()
    backend = FakeBackend()
    runner = RuntimeV3Runner(
        observer=ObserverStub(), state_builder=StateBuilder(),
        option_generator=OptionGenerator(), selector=OneStepSelector(),
        arbiter=arbiter, executor=Executor(backend, arbiter),
        effect_observer=EffectObserver(),
    )
    result = runner.run_episode(environment, task_id="mock", max_steps=1)
    assert environment.reset_called
    assert result["actions"] == 1
    assert len(backend.calls) == 1
    assert result["state"].last_action == "OPTION_0"

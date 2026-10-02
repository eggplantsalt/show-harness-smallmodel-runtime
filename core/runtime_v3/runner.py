"""Short orchestration loop for a single Runtime V3 episode."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Optional

from .arbiter import Arbiter, DecisionKind
from .effects import EffectObserver
from .executor import Executor
from .observer import Observer
from .options import OptionGenerator
from .selector import DeterministicSelector
from .state import BeliefState, StateBuilder


class RuntimeV3Runner:
    def __init__(
        self,
        *,
        observer: Observer,
        state_builder: StateBuilder,
        option_generator: OptionGenerator,
        selector: Any,
        arbiter: Arbiter,
        executor: Executor,
        effect_observer: EffectObserver,
        logger: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        self.observer = observer
        self.state_builder = state_builder
        self.option_generator = option_generator
        self.selector = selector
        self.arbiter = arbiter
        self.executor = executor
        self.effect_observer = effect_observer
        self.logger = logger
        self.state: BeliefState | None = None

    def run_episode(
        self,
        environment: Any,
        *,
        task_id: str,
        max_steps: int = 100,
        reset: bool = True,
        done_fn: Optional[Callable[[Any], bool]] = None,
    ) -> dict[str, Any]:
        if reset:
            try:
                environment.reset()
            except Exception as exc:
                return {"status": "ENV_INIT_FAILED", "reason": f"{type(exc).__name__}: {exc}", "actions": 0}
        try:
            self.state = self.state_builder.initialize(task_id)
            observation = self.observer.observe(environment)
        except Exception as exc:
            return {"status": "OBSERVATION_FAILED", "reason": f"{type(exc).__name__}: {exc}", "actions": 0}
        try:
            self.state = self.state_builder.update(self.state, observation)
        except Exception as exc:
            return {"status": "STATE_BUILD_FAILED", "reason": f"{type(exc).__name__}: {exc}", "actions": 0}
        actions = 0
        reobservations = 0

        while not self.state.done and actions + reobservations < max_steps:
            try:
                options = self.option_generator.generate(self.state)
            except Exception as exc:
                return {"status": "OPTION_GENERATION_FAILED", "reason": f"{type(exc).__name__}: {exc}",
                        "actions": actions, "state": self.state}
            try:
                selection = self.selector.select(self.state, options)
                decision = self.arbiter.authorize(self.state, options, selection)
            except Exception as exc:
                return {"status": "ARBITER_REJECTED", "reason": f"{type(exc).__name__}: {exc}",
                        "actions": actions, "state": self.state}
            if decision.kind == DecisionKind.REOBSERVE:
                reobservations += 1
                try:
                    observation = self.observer.observe(environment)
                    self.state = self.state_builder.update(self.state, observation)
                except Exception as exc:
                    return {"status": "OBSERVATION_FAILED", "reason": f"{type(exc).__name__}: {exc}",
                            "actions": actions, "state": self.state}
                continue
            if decision.kind in {DecisionKind.ABORT, DecisionKind.INVALID_SELECTION}:
                return {"status": "ARBITER_REJECTED", "decision": decision.kind.value,
                        "reason": decision.reason,
                        "actions": actions, "state": self.state}

            action = decision.action
            if action is None:
                return {"status": "ABORT", "reason": "arbiter returned no approved action",
                        "actions": actions, "state": self.state}
            before = self.state
            try:
                if action.primitive.kind == "micro_motion":
                    execution = self.executor.execute(
                        action,
                        tick_observer=lambda: self._observe_execution_tick(environment),
                    )
                else:
                    execution = self.executor.execute(action)
            except Exception as exc:
                return {"status": "EXECUTION_FAILED", "reason": f"{type(exc).__name__}: {exc}",
                        "actions": actions, "state": self.state}
            actions += 1

            # Every approved primitive is followed by a new observation before
            # the next option selection.
            try:
                observation = self.observer.observe(environment)
                after = self.state_builder.update(
                    before,
                    observation,
                    action=action.option_id,
                    expected_effect=action.expected_effect,
                )
                effect = self.effect_observer.compare(before, action.expected_effect, after)
            except Exception as exc:
                return {"status": "EFFECT_NOT_OBSERVED", "reason": f"{type(exc).__name__}: {exc}",
                        "actions": actions, "state": before}
            self.state = replace(after, last_observed_effect={
                "expected": dict(effect.expected),
                "observed": dict(effect.observed),
                "achieved": effect.achieved,
                "unexpected_motion": effect.unexpected_motion,
                "no_effect": effect.no_effect,
                "uncertainty": effect.uncertainty,
                "before_eef_position": effect.before_eef_position,
                "after_eef_position": effect.after_eef_position,
                "expected_delta": effect.expected_delta,
                "observed_delta": effect.observed_delta,
            })
            if self.logger is not None:
                self.logger({
                    "step_id": before.step_id,
                    "state_before": before,
                    "options": options,
                    "selection": selection,
                    "arbiter_result": {"kind": decision.kind.value, "reason": decision.reason},
                    "approved_action": action,
                    "execution": execution,
                    "state_after": self.state,
                    "effect": effect,
                })
            if done_fn is not None and done_fn(environment):
                return {"status": "DONE", "actions": actions, "state": self.state}

        status = "DONE" if self.state.done else "STEP_LIMIT"
        return {"status": status, "actions": actions, "state": self.state}

    def _observe_execution_tick(self, environment: Any) -> Any:
        """Allow perceptual observers to keep every low-level tick lightweight."""
        observe = getattr(self.observer, "observe_for_execution_tick", None)
        if callable(observe):
            return observe(environment)
        return self.observer.observe(environment)


def make_deterministic_stub_runner(**kwargs: Any) -> RuntimeV3Runner:
    """Small explicit constructor for wiring and dry-run checks."""
    kwargs.setdefault("selector", DeterministicSelector())
    kwargs.setdefault("effect_observer", EffectObserver())
    return RuntimeV3Runner(**kwargs)

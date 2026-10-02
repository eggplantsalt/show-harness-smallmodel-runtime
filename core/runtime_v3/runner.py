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
            environment.reset()
        self.state = self.state_builder.initialize(task_id)
        observation = self.observer.observe(environment)
        self.state = self.state_builder.update(self.state, observation)
        actions = 0
        reobservations = 0

        while not self.state.done and actions + reobservations < max_steps:
            options = self.option_generator.generate(self.state)
            selection = self.selector.select(self.state, options)
            decision = self.arbiter.authorize(self.state, options, selection)
            if decision.kind == DecisionKind.REOBSERVE:
                reobservations += 1
                observation = self.observer.observe(environment)
                self.state = self.state_builder.update(self.state, observation)
                continue
            if decision.kind in {DecisionKind.ABORT, DecisionKind.INVALID_SELECTION}:
                return {"status": decision.kind.value, "reason": decision.reason,
                        "actions": actions, "state": self.state}

            action = decision.action
            if action is None:
                return {"status": "ABORT", "reason": "arbiter returned no approved action",
                        "actions": actions, "state": self.state}
            before = self.state
            execution = self.executor.execute(action)
            actions += 1

            # Every approved primitive is followed by a new observation before
            # the next option selection.
            observation = self.observer.observe(environment)
            after = self.state_builder.update(
                before,
                observation,
                action=action.option_id,
                expected_effect=action.expected_effect,
            )
            effect = self.effect_observer.compare(before, action.expected_effect, after)
            self.state = replace(after, last_observed_effect={
                "expected": dict(effect.expected),
                "observed": dict(effect.observed),
                "achieved": effect.achieved,
                "unexpected_motion": effect.unexpected_motion,
                "no_effect": effect.no_effect,
                "uncertainty": effect.uncertainty,
            })
            if self.logger is not None:
                self.logger({
                    "state_before": before,
                    "options": options,
                    "selection": selection,
                    "approved_action": action,
                    "execution": execution,
                    "state_after": self.state,
                    "effect": effect,
                })
            if done_fn is not None and done_fn(environment):
                return {"status": "DONE", "actions": actions, "state": self.state}

        status = "DONE" if self.state.done else "STEP_LIMIT"
        return {"status": status, "actions": actions, "state": self.state}


def make_deterministic_stub_runner(**kwargs: Any) -> RuntimeV3Runner:
    """Small explicit constructor for wiring and dry-run checks."""
    kwargs.setdefault("selector", DeterministicSelector())
    kwargs.setdefault("effect_observer", EffectObserver())
    return RuntimeV3Runner(**kwargs)

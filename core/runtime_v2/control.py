"""Residual controller and online action-effect verification for VCR-v2."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional


MOVE_LEFT = "MV_LEFT"
MOVE_RIGHT = "MV_RIGHT"
MOVE_FWD = "MV_FWD"
MOVE_BACK = "MV_BACK"
MOVE_UP = "MV_UP"
MOVE_DOWN = "MV_DOWN"
STOP = "STOP"
DONE = "DONE"

OPPOSITE = {
    MOVE_LEFT: MOVE_RIGHT,
    MOVE_RIGHT: MOVE_LEFT,
    MOVE_FWD: MOVE_BACK,
    MOVE_BACK: MOVE_FWD,
    MOVE_UP: MOVE_DOWN,
    MOVE_DOWN: MOVE_UP,
}


@dataclass(frozen=True)
class ControlDecision:
    action_token: str
    reason: str
    predicted_effect: Optional[tuple[float, float]]
    failure: Optional[str] = None


class ResidualController:
    """Choose one bounded move, then verify it from the next observation."""

    def __init__(
        self,
        *,
        min_progress_px: float = 1.0,
        wrong_direction_px: float = 1.5,
        no_progress_limit: int = 3,
        wrong_direction_limit: int = 2,
        cycle_window: int = 8,
        effect_alpha: float = 0.5,
        axis_hold_steps: int = 3,
    ) -> None:
        self.min_progress_px = max(0.0, float(min_progress_px))
        self.wrong_direction_px = max(self.min_progress_px, float(wrong_direction_px))
        self.no_progress_limit = max(1, int(no_progress_limit))
        self.wrong_direction_limit = max(1, int(wrong_direction_limit))
        self.cycle_window = max(4, int(cycle_window))
        self.effect_alpha = min(1.0, max(0.05, float(effect_alpha)))
        self.axis_hold_steps = max(1, int(axis_hold_steps))
        self.reset()

    def reset(self) -> None:
        self.effects: dict[tuple[str, str], tuple[float, float]] = {}
        self.effect_counts: dict[tuple[str, str], int] = {}
        self.effect_direction_streaks: dict[tuple[str, str], tuple[int, int]] = {}
        self.forced_next: dict[str, str] = {}
        self.reset_transient()

    def reset_transient(self) -> None:
        """Reset option-local watchdog state while retaining learned effects."""
        self.last_error: Optional[tuple[float, float]] = None
        self.last_action: Optional[str] = None
        self.last_context: Optional[str] = None
        self.no_progress_count = 0
        self.wrong_direction_count = 0
        self.cumulative_progress = 0.0
        self.history: deque[tuple[tuple[int, int], str]] = deque(maxlen=self.cycle_window)
        self.axis_lock: Optional[int] = None
        self.axis_hold_remaining = 0
        self.last_transition: dict[str, Any] = {"status": "UNINITIALIZED"}
        self.pending_external: Optional[dict[str, Any]] = None

    @staticmethod
    def norm(error: tuple[float, float]) -> float:
        return max(abs(float(error[0])), abs(float(error[1])))

    def observe_transition(
        self,
        *,
        context: str,
        error: tuple[float, float],
        previous_action: Optional[str],
    ) -> dict[str, Any]:
        action = str(previous_action or "").strip().upper()
        if self.last_context is not None and context != self.last_context:
            self.no_progress_count = 0
            self.wrong_direction_count = 0
            self.cumulative_progress = 0.0
            self.history.clear()
            self.last_error = None
            self.last_action = None
            self.last_context = context
        if (
            self.last_error is None
            or self.last_action is None
            or action != self.last_action
            or context != self.last_context
        ):
            self.last_transition = {"status": "NO_PREVIOUS_EFFECT"}
            return self.last_transition

        delta = (error[0] - self.last_error[0], error[1] - self.last_error[1])
        key = (context, action)
        old = self.effects.get(key)
        if old is None:
            self.effects[key] = delta
        else:
            a = self.effect_alpha
            self.effects[key] = (
                (1.0 - a) * old[0] + a * delta[0],
                (1.0 - a) * old[1] + a * delta[1],
            )
        self.effect_counts[key] = self.effect_counts.get(key, 0) + 1

        action_axis = (
            0
            if action in {MOVE_LEFT, MOVE_RIGHT}
            else (1 if action in {MOVE_FWD, MOVE_BACK} else None)
        )
        improvement = (
            abs(self.last_error[action_axis]) - abs(error[action_axis])
            if action_axis is not None
            else self.norm(self.last_error) - self.norm(error)
        )
        direction = 1 if improvement > 0.0 else (-1 if improvement < 0.0 else 0)
        old_direction, old_count = self.effect_direction_streaks.get(key, (0, 0))
        self.effect_direction_streaks[key] = (
            direction,
            old_count + 1 if direction != 0 and direction == old_direction else (1 if direction != 0 else 0),
        )
        if improvement >= self.min_progress_px:
            status = "IMPROVING"
            self.no_progress_count = 0
            self.wrong_direction_count = 0
            self.cumulative_progress = 0.0
        elif improvement <= -self.wrong_direction_px:
            status = "WRONG_DIRECTION"
            self.wrong_direction_count += 1
            # A wrong-direction observation is useful calibration evidence,
            # not a stall.  The controller reverses/re-estimates it separately.
            self.no_progress_count = 0
            self.cumulative_progress = 0.0
        elif improvement > 0.0:
            self.cumulative_progress += improvement
            if self.cumulative_progress >= self.min_progress_px:
                status = "IMPROVING_CUMULATIVE"
                self.no_progress_count = 0
                self.wrong_direction_count = 0
                self.cumulative_progress = 0.0
            else:
                status = "SMALL_PROGRESS"
                self.no_progress_count += 1
                self.wrong_direction_count = 0
        else:
            status = "NO_PROGRESS"
            self.no_progress_count += 1
            self.wrong_direction_count = 0
            self.cumulative_progress = 0.0
        self.last_transition = {
            "status": status,
            "action": action,
            "error_before_px": list(self.last_error),
            "error_after_px": list(error),
            "observed_delta_px": list(delta),
            "improvement_px": round(improvement, 3),
            "no_progress_count": self.no_progress_count,
            "wrong_direction_count": self.wrong_direction_count,
            "cumulative_progress_px": round(self.cumulative_progress, 3),
        }
        return self.last_transition

    def _predicted_norm(
        self, context: str, action: str, error: tuple[float, float]
    ) -> Optional[float]:
        effect = self.effects.get((context, action))
        _direction, streak = self.effect_direction_streaks.get(
            (context, action), (0, 0)
        )
        if (
            effect is None
            or self.effect_counts.get((context, action), 0) < 2
            or streak < 2
        ):
            return None
        return self.norm((error[0] + effect[0], error[1] + effect[1]))

    def _oscillating(self) -> bool:
        if len(self.history) < 4:
            return False
        recent = list(self.history)
        actions = [item[1] for item in recent]
        if len(set(actions)) != 2:
            return False
        first, second = actions[-2], actions[-1]
        if OPPOSITE.get(first) != second:
            return False
        alternating = all(
            actions[index] == actions[-1 - ((len(actions) - 1 - index) % 2)]
            for index in range(len(actions))
        )
        states = [item[0] for item in recent]
        repeated_state = len(set(states)) <= max(2, len(states) // 2)
        return alternating and repeated_state

    def decide(
        self,
        *,
        context: str,
        error: tuple[float, float],
        horizontal_prior: Optional[str],
        depth_prior: Optional[str],
        previous_action: Optional[str],
        tolerance_px: float,
    ) -> ControlDecision:
        self.observe_transition(
            context=context, error=error, previous_action=previous_action
        )
        if self._oscillating():
            return ControlDecision(STOP, "detected bounded action cycle", None, "OSCILLATION")
        if self.no_progress_count >= self.no_progress_limit:
            return ControlDecision(STOP, "residual made no progress", None, "NO_PROGRESS")
        if self.norm(error) <= float(tolerance_px):
            return ControlDecision(DONE, "residual inside option tolerance", (0.0, 0.0))

        forced = self.forced_next.pop(context, None) or self.forced_next.pop("*", None)
        if forced:
            signature = (int(round(error[0] / 4.0)), int(round(error[1] / 4.0)))
            self.history.append((signature, forced))
            self.last_error = error
            self.last_action = forced
            self.last_context = context
            return ControlDecision(
                forced,
                "typed recovery tests the opposite action hypothesis",
                self.effects.get((context, forced)),
            )

        natural_axis = 0 if abs(error[0]) >= abs(error[1]) else 1
        if (
            self.axis_lock is not None
            and self.axis_hold_remaining > 0
            and abs(error[self.axis_lock]) > float(tolerance_px)
        ):
            axis = self.axis_lock
            self.axis_hold_remaining -= 1
        else:
            axis = natural_axis
            if axis != self.axis_lock:
                self.axis_lock = axis
                self.axis_hold_remaining = self.axis_hold_steps - 1
        pair = (MOVE_LEFT, MOVE_RIGHT) if axis == 0 else (MOVE_FWD, MOVE_BACK)
        prior = str(horizontal_prior if axis == 0 else depth_prior or "").upper()
        if prior not in pair:
            prior = pair[0]

        predictions = [
            (value, action)
            for action in pair
            if (value := self._predicted_norm(context, action, error)) is not None
        ]
        if len(predictions) == 2:
            predictions.sort()
            action = predictions[0][1]
            reason = "minimum learned residual"
        else:
            # A single action's EMA is not enough to reject the calibrated
            # camera prior: actuator latency can make the first sample look
            # reversed and poison that estimate.  Reversing direction requires
            # two consecutive wrong-direction transitions, as specified by the
            # runtime contract, or comparable predictions for both directions.
            if (
                self.last_transition.get("status") == "WRONG_DIRECTION"
                and self.wrong_direction_count >= self.wrong_direction_limit
                and str(previous_action or "").upper() in pair
            ):
                action = OPPOSITE[str(previous_action).upper()]
                reason = "reverse last wrong-direction action"
            else:
                action = prior
                reason = "calibrated geometry prior"

        predicted_effect = self.effects.get((context, action))
        signature = (int(round(error[0] / 4.0)), int(round(error[1] / 4.0)))
        self.history.append((signature, action))
        self.last_error = error
        self.last_action = action
        self.last_context = context
        return ControlDecision(action, reason, predicted_effect)

    def start_recovery(self, context: str) -> None:
        """Clear a stalled option and schedule one bounded alternative probe."""
        last = self.last_action
        self.reset_transient()
        if last in OPPOSITE:
            self.forced_next["*"] = OPPOSITE[last]

    def prime_external_action(
        self,
        *,
        context: str,
        error: tuple[float, float],
        action: str,
        defer: bool = False,
    ) -> None:
        """Record a requested action without pretending it was executed.

        The runner must call :meth:`commit_executed_action` after all safety
        layers and adapters have returned the actual token.
        """
        action = str(action or "").strip().upper()
        if action not in OPPOSITE:
            return
        if not defer:
            signature = (int(round(error[0] / 4.0)), int(round(error[1] / 4.0)))
            self.history.append((signature, action))
            self.last_error = error
            self.last_action = action
            self.last_context = context
            return
        self.pending_external = {"context": context, "error": tuple(error), "requested_action": action}

    def commit_executed_action(self, *, executed_action: str, authorized_action: Optional[str] = None) -> dict[str, Any]:
        """Commit only the token that reached the robot adapter."""
        pending = self.pending_external
        self.pending_external = None
        executed = str(executed_action or "").strip().upper()
        if pending is None:
            return {"committed": False, "executed_action": executed, "reason": "no pending external action"}
        context = str(pending["context"])
        error = tuple(pending["error"])
        if executed not in OPPOSITE:
            return {"committed": False, "executed_action": executed, "reason": "executed action is not a motion atom"}
        signature = (int(round(error[0] / 4.0)), int(round(error[1] / 4.0)))
        self.history.append((signature, executed))
        self.last_error = error
        self.last_action = executed
        self.last_context = context
        return {
            "committed": True,
            "context": context,
            "requested_action": pending["requested_action"],
            "authorized_action": authorized_action or pending["requested_action"],
            "executed_action": executed,
        }

    def contradicted_actions(self, context: str) -> tuple[str, ...]:
        """Actions whose measured axis effect was repeatedly counterproductive."""
        return tuple(
            sorted(
                action
                for (item_context, action), (direction, count) in self.effect_direction_streaks.items()
                if item_context == context
                and direction < 0
                and count >= self.wrong_direction_limit
            )
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "action_effects": {
                f"{context}:{action}": list(effect)
                for (context, action), effect in self.effects.items()
            },
            "action_effect_counts": {
                f"{context}:{action}": count
                for (context, action), count in self.effect_counts.items()
            },
            "action_effect_direction_streaks": {
                f"{context}:{action}": {"direction": value[0], "count": value[1]}
                for (context, action), value in self.effect_direction_streaks.items()
            },
            "transition": dict(self.last_transition),
            "no_progress_count": self.no_progress_count,
            "wrong_direction_count": self.wrong_direction_count,
            "cumulative_progress_px": round(self.cumulative_progress, 3),
            "axis_lock": self.axis_lock,
            "axis_hold_remaining": self.axis_hold_remaining,
            "history": [
                {"state_signature": list(state), "action": action}
                for state, action in self.history
            ],
            "pending_external": dict(self.pending_external) if self.pending_external else None,
        }

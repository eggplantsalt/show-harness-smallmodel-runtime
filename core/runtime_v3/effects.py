"""Small effect records; interpretation remains based on the new observation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .state import BeliefState


@dataclass(frozen=True)
class ExpectedEffect:
    values: Mapping[str, Any]


@dataclass(frozen=True)
class EffectRecord:
    expected: Mapping[str, Any]
    observed: Mapping[str, Any]
    achieved: bool | None
    unexpected_motion: bool = False
    no_effect: bool = False
    uncertainty: str = "unknown"
    before_eef_position: tuple[float, float, float] | None = None
    after_eef_position: tuple[float, float, float] | None = None
    expected_delta: tuple[float, float, float] | None = None
    observed_delta: tuple[float, float, float] | None = None


class EffectObserver:
    """Compare explicitly expected state fields with a newly observed state."""

    def compare(self, before: BeliefState, expected: Mapping[str, Any], after: BeliefState) -> EffectRecord:
        observed = {
            "target_identity": after.target_identity,
            "holding_state": after.holding_state,
            "contact_state": after.contact_state,
            "gripper_state": after.gripper_state,
            "stage": after.stage,
        }
        before_position = self._eef_position(before)
        after_position = self._eef_position(after)
        expected_delta_raw = expected.get("end_effector_delta_xyz")
        expected_delta = self._vector(expected_delta_raw)
        observed_delta = (
            tuple(after_position[i] - before_position[i] for i in range(3))
            if before_position is not None and after_position is not None else None
        )
        if expected_delta is not None:
            if observed_delta is None:
                achieved = None
            else:
                expected_norm = sum(v * v for v in expected_delta) ** 0.5
                projection = (sum(expected_delta[i] * observed_delta[i] for i in range(3))
                              / expected_norm if expected_norm > 0 else 0.0)
                threshold = float(expected.get("minimum_delta_projection_m", 0.0))
                achieved = projection > threshold
            return EffectRecord(
                expected=dict(expected),
                observed={
                    **observed,
                    "before_eef_position": before_position,
                    "after_eef_position": after_position,
                    "observed_delta": observed_delta,
                },
                achieved=achieved,
                unexpected_motion=bool(observed_delta is not None and not achieved
                                       and any(abs(v) > 1e-6 for v in observed_delta)),
                no_effect=observed_delta is not None and all(abs(v) <= 1e-6 for v in observed_delta),
                uncertainty="observed" if after.observation_fresh and observed_delta is not None
                else "eef_position_unavailable_or_stale",
                before_eef_position=before_position,
                after_eef_position=after_position,
                expected_delta=expected_delta,
                observed_delta=observed_delta,
            )
        if not expected:
            return EffectRecord({}, observed, None, uncertainty="no_expected_fields")
        matches = all(observed.get(key) == value for key, value in expected.items())
        changed = any((
            before.end_effector_state != after.end_effector_state,
            before.gripper_state != after.gripper_state,
            before.holding_state != after.holding_state,
            before.contact_state != after.contact_state,
            before.stage != after.stage,
        ))
        return EffectRecord(
            expected=dict(expected),
            observed=observed,
            achieved=matches,
            unexpected_motion=bool(changed and not matches),
            no_effect=not changed,
            uncertainty="observed" if after.observation_fresh else "stale_observation",
        )

    @staticmethod
    def _eef_position(state: BeliefState) -> tuple[float, float, float] | None:
        if not isinstance(state.end_effector_state, Mapping):
            return None
        return EffectObserver._vector(state.end_effector_state.get("position_xyz"))

    @staticmethod
    def _vector(value: Any) -> tuple[float, float, float] | None:
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            return None
        try:
            vector = tuple(float(v) for v in value)
        except (TypeError, ValueError):
            return None
        if not all(abs(v) < float("inf") for v in vector):
            return None
        return vector

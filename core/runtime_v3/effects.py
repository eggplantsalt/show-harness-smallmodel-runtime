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

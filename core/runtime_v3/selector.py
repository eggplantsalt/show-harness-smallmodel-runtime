"""Compact bounded semantic selection; no raw physical action vocabulary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .options import RuntimeOption
from .state import BeliefState


@dataclass(frozen=True)
class Selection:
    option_id: str
    status: str = "SELECTED"
    raw_output: str = ""


class DeterministicSelector:
    """Safe stub used until a compact model adapter is configured."""

    def select(self, state: BeliefState, options: Sequence[RuntimeOption]) -> Selection:
        return Selection("REOBSERVE", status="REOBSERVE", raw_output="stub")


class CompactVLMSelector:
    """Send only compact state fields and the current bounded option list."""

    def __init__(self, adapter: Callable[[Mapping[str, Any]], Any]) -> None:
        self.adapter = adapter

    @staticmethod
    def payload(state: BeliefState, options: Sequence[RuntimeOption]) -> dict[str, Any]:
        return {
            "task_id": state.task_id,
            "step_id": state.step_id,
            "stage": state.stage,
            "target_identity": state.target_identity,
            "target_confidence": state.target_confidence,
            "holding_state": state.holding_state,
            "contact_state": state.contact_state,
            "uncertainty": state.uncertainty,
            "options": [
                {"option_id": o.option_id, "option_type": o.option_type,
                 "description": o.description, "expected_effect": dict(o.expected_effect),
                 "confidence": o.confidence}
                for o in options
            ],
            "allowed_non_option_choices": ["REOBSERVE", "ABORT"],
        }

    def select(self, state: BeliefState, options: Sequence[RuntimeOption]) -> Selection:
        raw = self.adapter(self.payload(state, options))
        if isinstance(raw, Mapping):
            if set(raw) != {"option_id"}:
                return Selection("INVALID_SELECTION", status="INVALID_SELECTION",
                                 raw_output=json.dumps(raw, ensure_ascii=False))
            choice = raw.get("option_id")
            encoded = json.dumps(raw, ensure_ascii=False)
        else:
            encoded = str(raw)
            choice = encoded.strip()
            if choice.startswith("{"):
                try:
                    parsed = json.loads(choice)
                    choice = (parsed.get("option_id")
                              if isinstance(parsed, dict) and set(parsed) == {"option_id"}
                              else None)
                except (json.JSONDecodeError, AttributeError):
                    choice = None
        if not isinstance(choice, str):
            return Selection("INVALID_SELECTION", status="INVALID_SELECTION", raw_output=encoded)
        choice = choice.strip()
        valid = {o.option_id for o in options} | {"REOBSERVE", "ABORT"}
        if choice not in valid:
            return Selection("INVALID_SELECTION", status="INVALID_SELECTION", raw_output=encoded)
        return Selection(choice, status="SELECTED", raw_output=encoded)

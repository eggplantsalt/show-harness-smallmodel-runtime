"""Verified, bounded action candidates supplied by deterministic runtime tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .state import BeliefState


@dataclass(frozen=True)
class PrimitiveCommand:
    kind: str
    token: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    max_steps: int = 1
    max_duration_s: float = 1.0


@dataclass(frozen=True)
class RuntimeOption:
    option_id: str
    option_type: str
    description: str
    preconditions: Mapping[str, Any]
    expected_effect: Mapping[str, Any]
    primitive: PrimitiveCommand
    confidence: float
    evidence: tuple[str, ...] = ()
    evidence_frame_id: int | None = None


class OptionGenerator:
    """Convert explicit geometry candidates into typed options.

    A geometry provider may place candidate dictionaries under
    ``relevant_geometry['option_candidates']``.  This adapter does not infer
    new actions or contact the environment.
    """

    def generate(self, state: BeliefState) -> list[RuntimeOption]:
        candidates = state.relevant_geometry.get("option_candidates", ())
        options: list[RuntimeOption] = []
        if not isinstance(candidates, (list, tuple)):
            return options
        for raw in candidates:
            if not isinstance(raw, Mapping):
                continue
            primitive_raw = raw.get("primitive")
            if not isinstance(primitive_raw, Mapping):
                continue
            primitive = PrimitiveCommand(
                kind=str(primitive_raw.get("kind", "")),
                token=primitive_raw.get("token"),
                parameters=dict(primitive_raw.get("parameters", {}) or {}),
                max_steps=int(primitive_raw.get("max_steps", 1)),
                max_duration_s=float(primitive_raw.get("max_duration_s", 1.0)),
            )
            try:
                option = RuntimeOption(
                    option_id=str(raw["option_id"]),
                    option_type=str(raw["option_type"]),
                    description=str(raw["description"]),
                    preconditions=dict(raw.get("preconditions", {}) or {}),
                    expected_effect=dict(raw.get("expected_effect", {}) or {}),
                    primitive=primitive,
                    confidence=float(raw.get("confidence", 0.0)),
                    evidence=tuple(str(v) for v in raw.get("evidence", ())),
                    evidence_frame_id=(int(raw["evidence_frame_id"])
                                       if raw.get("evidence_frame_id") is not None else None),
                )
            except (KeyError, TypeError, ValueError):
                continue
            options.append(option)
        return options

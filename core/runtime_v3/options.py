"""Verified, bounded action candidates supplied by deterministic runtime tools."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from .state import BeliefState


@dataclass(frozen=True)
class BoundedMicroMotionSpec:
    """One semantic direction realized by a strictly bounded sequence of ticks."""

    direction: str
    direction_unit: tuple[float, float, float]
    requested_displacement_m: float = 0.003
    max_ticks: int = 5
    control_tick_step_m: float = 0.005

    def __post_init__(self) -> None:
        direction = str(self.direction).upper()
        if direction not in {"FWD", "BACK", "LEFT", "RIGHT", "UP", "DOWN"}:
            raise ValueError("direction must be one of FWD/BACK/LEFT/RIGHT/UP/DOWN")
        unit = tuple(float(value) for value in self.direction_unit)
        if len(unit) != 3 or not all(math.isfinite(value) for value in unit):
            raise ValueError("direction_unit must contain three finite values")
        norm = math.sqrt(sum(value * value for value in unit))
        if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("direction_unit must have unit length")
        requested = float(self.requested_displacement_m)
        tick_step = float(self.control_tick_step_m)
        if not math.isfinite(requested) or requested <= 0:
            raise ValueError("requested_displacement_m must be finite and positive")
        if requested > 0.009:
            raise ValueError("requested_displacement_m cannot exceed the calibrated 9 mm V3 scale")
        if not math.isfinite(tick_step) or tick_step <= 0:
            raise ValueError("control_tick_step_m must be finite and positive")
        if tick_step > 0.005:
            raise ValueError("control_tick_step_m cannot exceed the calibrated 5 mm control tick")
        if isinstance(self.max_ticks, bool):
            raise ValueError("max_ticks must be between 1 and 10")
        try:
            ticks = int(self.max_ticks)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("max_ticks must be between 1 and 10") from exc
        if ticks != self.max_ticks or not 1 <= ticks <= 10:
            raise ValueError("max_ticks must be between 1 and 10")
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "direction_unit", unit)
        object.__setattr__(self, "requested_displacement_m", requested)
        object.__setattr__(self, "control_tick_step_m", tick_step)
        object.__setattr__(self, "max_ticks", ticks)


@dataclass(frozen=True)
class PrimitiveCommand:
    kind: str
    token: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    max_steps: int = 1
    max_duration_s: float = 1.0
    micro_motion_spec: BoundedMicroMotionSpec | None = None


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
                micro_motion_spec=(
                    BoundedMicroMotionSpec(
                        direction=str(raw_spec["direction"]),
                        direction_unit=tuple(raw_spec["direction_unit"]),
                        requested_displacement_m=float(
                            raw_spec.get("requested_displacement_m", 0.003)
                        ),
                        max_ticks=int(raw_spec.get("max_ticks", 5)),
                        control_tick_step_m=float(raw_spec.get("control_tick_step_m", 0.005)),
                    )
                    if isinstance((raw_spec := primitive_raw.get("micro_motion_spec")), Mapping)
                    else None
                ),
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

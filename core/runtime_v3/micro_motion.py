"""Semantic option construction for one direction-bounded micro-motion."""

from __future__ import annotations

from typing import Sequence

from .options import BoundedMicroMotionSpec, PrimitiveCommand, RuntimeOption
from .state import BeliefState


class BoundedMicroMotionOptionGenerator:
    """Offer one configured semantic direction as a bounded RuntimeOption."""

    def __init__(
        self,
        direction: str,
        direction_unit: Sequence[float],
        *,
        requested_displacement_m: float = 0.003,
        max_ticks: int = 5,
        control_tick_step_m: float = 0.005,
    ) -> None:
        self.spec = BoundedMicroMotionSpec(
            direction=direction,
            direction_unit=tuple(float(value) for value in direction_unit),
            requested_displacement_m=requested_displacement_m,
            max_ticks=max_ticks,
            control_tick_step_m=control_tick_step_m,
        )
        self.option_id = f"MOVE_{self.spec.direction}_SMALL"

    def generate(self, state: BeliefState) -> list[RuntimeOption]:
        if (
            not state.observation_fresh
            or state.frame_id is None
            or state.end_effector_state is None
            or not bool(state.relevant_geometry.get("workspace_valid"))
        ):
            return []
        direction_delta = [
            component * self.spec.requested_displacement_m
            for component in self.spec.direction_unit
        ]
        return [RuntimeOption(
            option_id=self.option_id,
            option_type="bounded_micro_motion",
            description=(
                f"Move {self.spec.direction} by up to "
                f"{self.spec.requested_displacement_m * 1000:.1f} mm, then observe."
            ),
            preconditions={
                "observation_fresh": True,
                "relevant_geometry.workspace_valid": True,
            },
            expected_effect={
                "end_effector_delta_xyz": direction_delta,
                "minimum_delta_projection_m": self.spec.requested_displacement_m - 1e-12,
            },
            primitive=PrimitiveCommand(
                kind="micro_motion",
                max_steps=1,
                max_duration_s=1.0,
                micro_motion_spec=self.spec,
            ),
            confidence=1.0,
            evidence=state.evidence_refs,
            evidence_frame_id=state.frame_id,
        )]

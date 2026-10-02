"""Options reserved for the one-action infrastructure smoke test."""

from __future__ import annotations

from core.runtime_v3.options import OptionGenerator, PrimitiveCommand, RuntimeOption
from core.runtime_v3.state import BeliefState


class SmokeOptionGenerator(OptionGenerator):
    """Offer one small upward primitive only when observed workspace bounds allow it."""

    def __init__(self, step_m: float = 0.005) -> None:
        self.step_m = float(step_m)

    def generate(self, state: BeliefState) -> list[RuntimeOption]:
        geometry = state.relevant_geometry
        if not state.observation_fresh or not bool(geometry.get("workspace_valid")):
            return []
        if state.end_effector_state is None or state.frame_id is None:
            return []
        return [self._make_option("OPTION_SAFE_LIFT", self.step_m, state)]

    def offline_options(self, state: BeliefState) -> list[RuntimeOption]:
        """Build three non-executed bounded candidates for Qwen selector inspection."""
        if not state.observation_fresh or not bool(state.relevant_geometry.get("workspace_valid")):
            return []
        if state.end_effector_state is None or state.frame_id is None:
            return []
        bounds = state.relevant_geometry.get("workspace_z_bounds_m", ())
        position = state.end_effector_state.get("position_xyz")
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            return []
        if not isinstance(position, (tuple, list)) or len(position) != 3:
            return []
        try:
            upper = float(bounds[1])
            current_z = float(position[2])
        except (TypeError, ValueError):
            return []
        return [
            self._make_option(f"OPTION_SAFE_LIFT_{millimeters}MM", millimeters / 1000.0, state)
            for millimeters in (2, 5, 8)
            if current_z + millimeters / 1000.0 <= upper
        ]

    @staticmethod
    def _make_option(option_id: str, step_m: float, state: BeliefState) -> RuntimeOption:
        return RuntimeOption(
            option_id=option_id,
            option_type="bounded_vertical_probe",
            description=f"Raise the end effector by {step_m * 1000:.0f} mm, then observe again.",
            preconditions={
                "observation_fresh": True,
                "relevant_geometry.workspace_valid": True,
            },
            expected_effect={
                "end_effector_delta_xyz": [0.0, 0.0, step_m],
                "minimum_delta_projection_m": 0.0001,
            },
            primitive=PrimitiveCommand(
                kind="move",
                token="MV_UP",
                parameters={"step_m": step_m},
                max_steps=1,
                max_duration_s=1.0,
            ),
            confidence=1.0,
            evidence=state.evidence_refs,
            evidence_frame_id=state.frame_id,
        )

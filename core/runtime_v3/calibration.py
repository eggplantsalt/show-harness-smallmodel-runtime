"""Bounded, one-action actuation-contract calibration helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .arbiter import Arbiter
from .effects import EffectObserver
from .executor import Executor, LiberoPrimitiveBackend
from .options import PrimitiveCommand, RuntimeOption
from .runner import RuntimeV3Runner
from .selector import DeterministicSelector
from .state import BeliefState, StateBuilder


def compute_contract_metrics(
    before_xyz: Sequence[float],
    after_xyz: Sequence[float],
    direction_unit: Sequence[float],
    commanded_step_m: float,
    *,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Measure one observed EEF displacement against its commanded axis."""

    before = np.asarray(before_xyz, dtype=float).reshape(-1)
    after = np.asarray(after_xyz, dtype=float).reshape(-1)
    direction = np.asarray(direction_unit, dtype=float).reshape(-1)
    if before.shape != (3,) or after.shape != (3,) or direction.shape != (3,):
        raise ValueError("positions and direction must each have exactly three values")
    if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
        raise ValueError("positions must be finite")
    if not np.all(np.isfinite(direction)):
        raise ValueError("direction must be finite")
    direction_norm = float(np.linalg.norm(direction))
    if not np.isclose(direction_norm, 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("direction_unit must have unit length")
    commanded_step_m = float(commanded_step_m)
    if not np.isfinite(commanded_step_m) or commanded_step_m <= 0:
        raise ValueError("commanded_step_m must be finite and positive")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")

    delta = after - before
    projection = float(np.dot(delta, direction))
    off_axis = delta - projection * direction
    observed_norm = float(np.linalg.norm(delta))
    direction_cosine = projection / (observed_norm + epsilon)
    return {
        "delta_xyz_m": delta.tolist(),
        "delta_xyz_mm": (delta * 1000.0).tolist(),
        "projection_m": projection,
        "projection_mm": projection * 1000.0,
        "realization_ratio": projection / commanded_step_m,
        "off_axis_vector_m": off_axis.tolist(),
        "off_axis_vector_mm": (off_axis * 1000.0).tolist(),
        "off_axis_magnitude_m": float(np.linalg.norm(off_axis)),
        "off_axis_magnitude_mm": float(np.linalg.norm(off_axis)) * 1000.0,
        "direction_cosine": direction_cosine,
        "observed_norm_m": observed_norm,
        "observed_norm_mm": observed_norm * 1000.0,
    }


class CalibrationOptionGenerator:
    """Offer only the requested bounded move, scoped to the fresh frame."""

    def __init__(
        self,
        token: str,
        direction_unit: Sequence[float],
        step_m: float,
        *,
        workspace_z_bounds_m: tuple[float, float] | None = None,
    ) -> None:
        self.token = str(token)
        direction = np.asarray(direction_unit, dtype=float).reshape(-1)
        if direction.shape != (3,) or not np.isclose(np.linalg.norm(direction), 1.0):
            raise ValueError("direction_unit must be a 3D unit vector")
        if not np.isfinite(step_m) or float(step_m) <= 0:
            raise ValueError("step_m must be finite and positive")
        self.direction_unit = direction
        self.step_m = float(step_m)
        self.option_id = f"CALIBRATION_{self.token}"
        self.workspace_z_bounds_m = workspace_z_bounds_m
        self.skip_reason: str | None = None

    def generate(self, state: BeliefState) -> list[RuntimeOption]:
        if (
            not state.observation_fresh
            or state.frame_id is None
            or state.end_effector_state is None
        ):
            return []
        if self.workspace_z_bounds_m is not None:
            position = state.end_effector_state.get("position_xyz")
            if not isinstance(position, (tuple, list)) or len(position) != 3:
                return []
            target_z = float(position[2]) + float(self.direction_unit[2]) * self.step_m
            z_min, z_max = self.workspace_z_bounds_m
            if target_z < z_min or target_z > z_max:
                self.skip_reason = "SKIPPED_WORKSPACE_BOUNDARY"
                return []
        expected_delta = (self.direction_unit * self.step_m).tolist()
        return [
            RuntimeOption(
                option_id=self.option_id,
                option_type="single_step_actuation_calibration",
                description=f"Issue one {self.token} command at {self.step_m * 1000:.1f} mm.",
                preconditions={"observation_fresh": True},
                expected_effect={
                    "end_effector_delta_xyz": expected_delta,
                    # Retained only so the legacy EffectObserver result can be
                    # recorded beside, and distinguished from, contract metrics.
                    "minimum_delta_projection_m": 0.0001,
                },
                primitive=PrimitiveCommand(
                    kind="move",
                    token=self.token,
                    parameters={"step_m": self.step_m},
                    max_steps=1,
                    max_duration_s=1.0,
                ),
                confidence=1.0,
                evidence=state.evidence_refs,
                evidence_frame_id=state.frame_id,
            )
        ]


def _eef_xyz(state: BeliefState) -> tuple[float, float, float] | None:
    if not isinstance(state.end_effector_state, Mapping):
        return None
    raw = state.end_effector_state.get("position_xyz")
    if not isinstance(raw, (tuple, list)) or len(raw) != 3:
        return None
    return tuple(float(value) for value in raw)


def run_calibration_trial(
    environment: Any,
    observer: Any,
    controller: Any,
    *,
    task_id: str,
    token: str,
    direction_unit: Sequence[float],
    commanded_step_m: float,
    workspace_z_bounds_m: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Reset once and run one fresh-frame option through the full V3 authority path."""

    events: list[dict[str, Any]] = []
    arbiter = Arbiter()
    option_generator = CalibrationOptionGenerator(
        token,
        direction_unit,
        commanded_step_m,
        workspace_z_bounds_m=workspace_z_bounds_m,
    )
    backend = LiberoPrimitiveBackend(environment, controller, arbiter)
    runner = RuntimeV3Runner(
        observer=observer,
        state_builder=StateBuilder(),
        option_generator=option_generator,
        selector=DeterministicSelector(option_generator.option_id),
        arbiter=arbiter,
        executor=Executor(backend, arbiter),
        effect_observer=EffectObserver(),
        logger=events.append,
    )
    result = runner.run_episode(
        environment,
        task_id=task_id,
        max_steps=1,
        reset=True,
    )
    event = events[-1] if events else {}
    before = event.get("state_before") or result.get("state")
    after = event.get("state_after")
    before_xyz = _eef_xyz(before) if isinstance(before, BeliefState) else None
    after_xyz = _eef_xyz(after) if isinstance(after, BeliefState) else None
    metrics = None
    if before_xyz is not None and after_xyz is not None:
        metrics = compute_contract_metrics(
            before_xyz, after_xyz, direction_unit, commanded_step_m
        )
    effect = event.get("effect")
    return {
        "run_status": result.get("status"),
        "actions": int(result.get("actions", 0)),
        "reason": result.get("reason"),
        "skip_reason": option_generator.skip_reason,
        "state_before_eef_xyz_m": list(before_xyz) if before_xyz is not None else None,
        "state_after_eef_xyz_m": list(after_xyz) if after_xyz is not None else None,
        "effect_achieved_legacy": getattr(effect, "achieved", None),
        "contract_metrics": metrics,
        "selection": getattr(event.get("selection"), "option_id", None),
        "approved_action": getattr(event.get("approved_action"), "option_id", None),
        "backend_execution": event.get("execution") is not None,
    }

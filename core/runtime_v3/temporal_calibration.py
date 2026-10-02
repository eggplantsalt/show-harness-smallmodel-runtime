"""Bounded repeated-tick measurements through Runtime V3's single-tick path."""

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


def compute_vector_metrics(
    delta_xyz_m: Sequence[float],
    direction_unit: Sequence[float],
    commanded_step_m: float,
) -> dict[str, Any]:
    """Apply the contract metrics to an already measured displacement vector."""

    from .calibration import compute_contract_metrics

    delta = np.asarray(delta_xyz_m, dtype=float).reshape(-1)
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        raise ValueError("delta_xyz_m must contain three finite values")
    return compute_contract_metrics([0.0, 0.0, 0.0], delta, direction_unit, commanded_step_m)


def baseline_corrected_effect(
    action_delta_xyz_m: Sequence[float],
    hold_delta_xyz_m: Sequence[float],
    direction_unit: Sequence[float],
    commanded_step_m: float,
) -> dict[str, Any]:
    """Subtract a matched HOLD displacement and report the remaining effect."""

    action_delta = np.asarray(action_delta_xyz_m, dtype=float).reshape(-1)
    hold_delta = np.asarray(hold_delta_xyz_m, dtype=float).reshape(-1)
    if action_delta.shape != (3,) or hold_delta.shape != (3,):
        raise ValueError("action and HOLD displacements must each contain three values")
    if not np.all(np.isfinite(action_delta)) or not np.all(np.isfinite(hold_delta)):
        raise ValueError("action and HOLD displacements must be finite")
    corrected = action_delta - hold_delta
    metrics = compute_vector_metrics(corrected, direction_unit, commanded_step_m)
    return {
        "delta_xyz_m": corrected.tolist(),
        "delta_xyz_mm": (corrected * 1000.0).tolist(),
        "hold_delta_xyz_m": hold_delta.tolist(),
        "hold_delta_xyz_mm": (hold_delta * 1000.0).tolist(),
        "metrics": metrics,
    }


def aggregate_temporal_response(
    action_delta_by_tick: Sequence[Sequence[float]],
    hold_delta_by_tick: Sequence[Sequence[float]],
    direction_unit: Sequence[float],
    per_tick_command_m: float,
) -> list[dict[str, Any]]:
    """Aggregate raw and tick-matched corrected cumulative response curves."""

    if len(action_delta_by_tick) != len(hold_delta_by_tick):
        raise ValueError("action and matched HOLD curves must have the same number of ticks")
    result = []
    for tick, (action_delta, hold_delta) in enumerate(
        zip(action_delta_by_tick, hold_delta_by_tick), start=1
    ):
        raw = compute_vector_metrics(action_delta, direction_unit, per_tick_command_m * tick)
        corrected = baseline_corrected_effect(
            action_delta, hold_delta, direction_unit, per_tick_command_m * tick
        )
        result.append({
            "tick": tick,
            "raw_delta_xyz_m": raw["delta_xyz_m"],
            "raw_metrics": raw,
            "hold_delta_xyz_m": np.asarray(hold_delta, dtype=float).reshape(3).tolist(),
            "baseline_corrected_delta_xyz_m": corrected["delta_xyz_m"],
            "baseline_corrected_delta_xyz_mm": corrected["delta_xyz_mm"],
            "baseline_corrected_metrics": corrected["metrics"],
        })
    return result


def opposite_pair_metric(
    effect_a_xyz_m: Sequence[float], effect_b_xyz_m: Sequence[float]
) -> dict[str, Any]:
    """Measure how closely two baseline-corrected opposite effects cancel."""

    first = np.asarray(effect_a_xyz_m, dtype=float).reshape(-1)
    second = np.asarray(effect_b_xyz_m, dtype=float).reshape(-1)
    if first.shape != (3,) or second.shape != (3,):
        raise ValueError("opposite effects must each contain three values")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("opposite effects must be finite")
    residual = first + second
    return {
        "residual_xyz_m": residual.tolist(),
        "residual_xyz_mm": (residual * 1000.0).tolist(),
        "residual_norm_m": float(np.linalg.norm(residual)),
        "residual_norm_mm": float(np.linalg.norm(residual)) * 1000.0,
    }


def settling_curve(trials: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-tick drift from independent HOLD horizon trials."""

    max_tick = max(
        (len(row.get("points_xyz_m", ())) - 1 for row in trials), default=0
    )
    curve: list[dict[str, Any]] = []
    for tick in range(1, max_tick + 1):
        deltas = []
        cumulative = []
        for row in trials:
            points = row.get("points_xyz_m")
            if not isinstance(points, (list, tuple)) or len(points) <= tick:
                continue
            start = np.asarray(points[0], dtype=float)
            previous = np.asarray(points[tick - 1], dtype=float)
            current = np.asarray(points[tick], dtype=float)
            deltas.append(current - previous)
            cumulative.append(current - start)
        if not deltas:
            continue
        per_tick = np.asarray(deltas, dtype=float)
        total = np.asarray(cumulative, dtype=float)
        norms = np.linalg.norm(per_tick, axis=1) * 1000.0
        curve.append({
            "tick": tick,
            "n": int(len(deltas)),
            "mean_delta_xyz_mm": (per_tick.mean(axis=0) * 1000.0).tolist(),
            "mean_delta_norm_mm": float(norms.mean()),
            "std_delta_norm_mm": float(norms.std(ddof=0)),
            "mean_cumulative_delta_xyz_mm": (total.mean(axis=0) * 1000.0).tolist(),
            "mean_cumulative_norm_mm": float(
                (np.linalg.norm(total, axis=1) * 1000.0).mean()
            ),
        })
    return curve


class TemporalOptionGenerator:
    """Offer one fresh-frame HOLD or one bounded move for a calibration tick."""

    def __init__(
        self,
        *,
        token: str | None,
        direction_unit: Sequence[float] | None,
        step_m: float,
        workspace_z_bounds_m: tuple[float, float] | None = None,
    ) -> None:
        self.token = token
        self.direction_unit = (
            None if direction_unit is None
            else np.asarray(direction_unit, dtype=float).reshape(-1)
        )
        if token is None:
            self.option_id = "CALIBRATION_HOLD"
            if direction_unit is not None:
                raise ValueError("HOLD cannot have a movement direction")
        else:
            if self.direction_unit is None or self.direction_unit.shape != (3,):
                raise ValueError("a move token requires a 3D direction")
            if not np.isclose(np.linalg.norm(self.direction_unit), 1.0):
                raise ValueError("direction_unit must have unit length")
            self.option_id = f"CALIBRATION_{token}"
        self.step_m = float(step_m)
        if not np.isfinite(self.step_m) or self.step_m <= 0:
            raise ValueError("step_m must be finite and positive")
        self.workspace_z_bounds_m = workspace_z_bounds_m
        self.skip_reason: str | None = None

    def generate(self, state: BeliefState) -> list[RuntimeOption]:
        if not state.observation_fresh or state.frame_id is None or state.end_effector_state is None:
            return []
        if self.workspace_z_bounds_m is not None and self.token is not None:
            raw_position = state.end_effector_state.get("position_xyz")
            if not isinstance(raw_position, (tuple, list)) or len(raw_position) != 3:
                return []
            target_z = float(raw_position[2]) + float(self.direction_unit[2]) * self.step_m
            z_min, z_max = self.workspace_z_bounds_m
            if target_z < z_min or target_z > z_max:
                self.skip_reason = "SKIPPED_WORKSPACE_BOUNDARY"
                return []
        primitive = PrimitiveCommand(
            kind="hold" if self.token is None else "move",
            token=self.token,
            parameters={} if self.token is None else {"step_m": self.step_m},
            max_steps=1,
            max_duration_s=1.0,
        )
        expected_delta = (
            [0.0, 0.0, 0.0]
            if self.token is None
            else (self.direction_unit * self.step_m).tolist()
        )
        return [RuntimeOption(
            option_id=self.option_id,
            option_type="temporal_actuation_calibration",
            description=(
                "Maintain the current end-effector target and preserve gripper state."
                if self.token is None
                else f"Issue one {self.token} control-tick command."
            ),
            preconditions={"observation_fresh": True},
            expected_effect={"end_effector_delta_xyz": expected_delta},
            primitive=primitive,
            confidence=1.0,
            evidence=state.evidence_refs,
            evidence_frame_id=state.frame_id,
        )]


def run_v3_tick(
    environment: Any,
    observer: Any,
    controller: Any,
    *,
    task_id: str,
    token: str | None,
    direction_unit: Sequence[float] | None,
    commanded_step_m: float,
    reset: bool,
    workspace_z_bounds_m: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Run exactly one fresh observation/option/approval/Executor/control cycle."""

    events: list[dict[str, Any]] = []
    arbiter = Arbiter()
    generator = TemporalOptionGenerator(
        token=token,
        direction_unit=direction_unit,
        step_m=commanded_step_m,
        workspace_z_bounds_m=workspace_z_bounds_m,
    )
    backend = LiberoPrimitiveBackend(environment, controller, arbiter)
    runner = RuntimeV3Runner(
        observer=observer,
        state_builder=StateBuilder(),
        option_generator=generator,
        selector=DeterministicSelector(generator.option_id),
        arbiter=arbiter,
        executor=Executor(backend, arbiter),
        effect_observer=EffectObserver(),
        logger=events.append,
    )
    result = runner.run_episode(environment, task_id=task_id, max_steps=1, reset=reset)
    event = events[-1] if events else {}
    before = event.get("state_before") or result.get("state")
    after = event.get("state_after")
    before_position = _eef_xyz(before) if isinstance(before, BeliefState) else None
    after_position = _eef_xyz(after) if isinstance(after, BeliefState) else None
    return {
        "run_status": result.get("status"),
        "actions": int(result.get("actions", 0)),
        "reason": result.get("reason"),
        "skip_reason": generator.skip_reason,
        "state_before_eef_xyz_m": list(before_position) if before_position else None,
        "state_after_eef_xyz_m": list(after_position) if after_position else None,
        "selection": getattr(event.get("selection"), "option_id", None),
        "approved_action": getattr(event.get("approved_action"), "option_id", None),
        "backend_execution": event.get("execution") is not None,
    }


def _eef_xyz(state: BeliefState) -> tuple[float, float, float] | None:
    if not isinstance(state.end_effector_state, Mapping):
        return None
    raw = state.end_effector_state.get("position_xyz")
    if not isinstance(raw, (tuple, list)) or len(raw) != 3:
        return None
    return tuple(float(value) for value in raw)

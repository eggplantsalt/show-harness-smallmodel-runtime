#!/usr/bin/env python3
"""Measure LIBERO HOLD settling and repeated atomic control-tick response."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_yaml
from core.runtime_v3.adapters.libero_env import LiberoEnvironmentAdapter
from core.runtime_v3.adapters.libero_observation import LiberoObservationAdapter
from core.runtime_v3.temporal_calibration import (
    aggregate_temporal_response,
    compute_vector_metrics,
    opposite_pair_metric,
    run_v3_tick,
    settling_curve,
)
from interpreters.libero_atomic_controller import LiberoAtomicController
from core.action_units import MOVE_ATOMS


HOLD_HORIZONS = (1, 2, 4, 5)
PRE_SETTLE_VALUES = (0, 1, 2, 4)
ACTION_HORIZONS = (1, 2, 3, 4)
HOLD_TRIALS = 3
ACTION_TRIALS = 1  # One independent reset for each requested primitive/horizon.
OPPOSITE_PAIRS = (("MV_LEFT", "MV_RIGHT"), ("MV_FWD", "MV_BACK"), ("MV_UP", "MV_DOWN"))


class _RecordingController:
    """Record the exact controller vector created inside Executor's backend call."""

    def __init__(self, controller: LiberoAtomicController) -> None:
        self.controller = controller
        self.action_vectors: list[list[float]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.controller, name)

    def _record(self, vector: Any) -> Any:
        self.action_vectors.append(np.asarray(vector, dtype=float).reshape(-1).tolist())
        return vector

    def action_for_atomic(self, token: str, *, step_m: float | None = None) -> Any:
        return self._record(self.controller.action_for_atomic(token, step_m=step_m))

    def hold_action(self) -> Any:
        return self._record(self.controller.hold_action())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/robot_libero_clean_qwen3vl.yaml"))
    parser.add_argument("--suite", default=None)
    parser.add_argument("--task", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--init-state-index", type=int, default=None)
    parser.add_argument("--commanded-step-mm", type=float, default=5.0)
    parser.add_argument("--workspace-z-min-m", type=float, default=0.02)
    parser.add_argument("--workspace-z-max-m", type=float, default=0.60)
    parser.add_argument("--output-dir", default=str(ROOT / "rollouts/runtime_v3_response"))
    parser.add_argument("--run-dir", default=None, help="Resume the baseline artifact for --phase actions")
    parser.add_argument("--phase", choices=("baseline", "actions"), default="baseline")
    parser.add_argument("--pre-settle-ticks", type=int, choices=PRE_SETTLE_VALUES, default=None)
    return parser


def _new_run_dir(base: str | Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(base).expanduser() / f"run_{timestamp}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _controller(cfg: dict[str, Any], vectors: dict[str, list[float]]) -> _RecordingController:
    base = LiberoAtomicController(
        move_vectors=vectors,
        step_m=float(cfg.get("step_m", 0.02)),
        sim_steps_per_decision=1,
        position_scale_m=float(cfg.get("position_scale_m", 0.05)),
    )
    return _RecordingController(base)


def _one_tick(
    environment: LiberoEnvironmentAdapter,
    observer: LiberoObservationAdapter,
    controller: _RecordingController,
    *,
    task_id: str,
    token: str | None,
    direction: list[float] | None,
    step_m: float,
    reset: bool,
    workspace: tuple[float, float],
) -> dict[str, Any]:
    outcome = run_v3_tick(
        environment,
        observer,
        controller,
        task_id=task_id,
        token=token,
        direction_unit=direction,
        commanded_step_m=step_m,
        reset=reset,
        workspace_z_bounds_m=workspace,
    )
    if outcome["actions"] != 1 or not outcome["backend_execution"]:
        raise RuntimeError(f"Runtime V3 calibration tick did not execute: {outcome}")
    if outcome["state_before_eef_xyz_m"] is None or outcome["state_after_eef_xyz_m"] is None:
        raise RuntimeError(f"Runtime V3 calibration tick did not produce EEF states: {outcome}")
    return outcome


def _fresh_observer(workspace: tuple[float, float]) -> LiberoObservationAdapter:
    return LiberoObservationAdapter(max_eef_z_m=workspace[1], min_eef_z_m=workspace[0])


def _base_row(
    metadata: dict[str, Any], *, condition: str, horizon: int, trial_index: int,
    settle_ticks: int,
) -> dict[str, Any]:
    return {
        "run_id": metadata["run_id"],
        "suite": metadata["suite"],
        "task_id": metadata["task_id"],
        "task_name": metadata.get("task_name"),
        "seed": metadata["seed"],
        "init_state_index": metadata["init_state_index"],
        "condition": condition,
        "horizon": horizon,
        "trial_index": trial_index,
        "pre_settle_ticks": settle_ticks,
        "points_xyz_m": [],
        "timed_points_xyz_m": [],
        "action_vectors": [],
        "runtime_cycles": [],
    }


def _run_hold_sequence(
    environment: LiberoEnvironmentAdapter,
    *,
    metadata: dict[str, Any],
    workspace: tuple[float, float],
    settle_ticks: int,
    horizon: int,
    condition: str,
    trial_index: int,
    vectors: dict[str, list[float]],
    step_m: float,
) -> dict[str, Any]:
    controller = _controller(metadata["controller_config"], vectors)
    observer = _fresh_observer(workspace)
    row = _base_row(metadata, condition=condition, horizon=horizon, trial_index=trial_index,
                    settle_ticks=settle_ticks)
    first = True
    initial_position: list[float] | None = None
    for _tick in range(settle_ticks):
        outcome = _one_tick(environment, observer, controller, task_id=metadata["runtime_task_id"],
                            token=None, direction=None, step_m=step_m, reset=first,
                            workspace=workspace)
        first = False
        if initial_position is None:
            initial_position = outcome["state_before_eef_xyz_m"]
            row["points_xyz_m"].append(initial_position)
        row["points_xyz_m"].append(outcome["state_after_eef_xyz_m"])
        row["runtime_cycles"].append(outcome)
    if settle_ticks == 0:
        environment.reset()
        observation = observer.observe(environment)
        initial_position = list(observer.last_raw.eef_position_xyz)  # type: ignore[union-attr]
        row["points_xyz_m"].append(initial_position)
        first = False

    calibrated_start = row["points_xyz_m"][-1]
    row["timed_points_xyz_m"].append(calibrated_start)
    for _tick in range(horizon):
        outcome = _one_tick(environment, observer, controller, task_id=metadata["runtime_task_id"],
                            token=None, direction=None, step_m=step_m, reset=first,
                            workspace=workspace)
        first = False
        row["timed_points_xyz_m"].append(outcome["state_after_eef_xyz_m"])
        row["points_xyz_m"].append(outcome["state_after_eef_xyz_m"])
        row["runtime_cycles"].append(outcome)
    row["calibrated_start_eef_xyz_m"] = calibrated_start
    row["action_vectors"] = controller.action_vectors
    row["trial_status"] = "EXECUTED"
    return row


def _run_pre_settle_trial(
    environment: LiberoEnvironmentAdapter,
    *, metadata: dict[str, Any], workspace: tuple[float, float], settle_ticks: int,
    trial_index: int, vectors: dict[str, list[float]], step_m: float,
) -> dict[str, Any]:
    row = _run_hold_sequence(
        environment, metadata=metadata, workspace=workspace, settle_ticks=settle_ticks,
        horizon=0, condition="pre_settle", trial_index=trial_index, vectors=vectors,
        step_m=step_m,
    )
    points = row["points_xyz_m"]
    end_position = points[-1]
    start_position = points[0]
    delta = (np.asarray(end_position, dtype=float) - np.asarray(start_position, dtype=float)).tolist()
    row["pre_settle_drift_xyz_m"] = delta
    row["pre_settle_drift_norm_mm"] = float(np.linalg.norm(delta)) * 1000.0
    return row


def _pre_settle_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for k in PRE_SETTLE_VALUES:
        group = [
            row for row in rows
            if row["condition"] == "pre_settle" and row["pre_settle_ticks"] == k
        ]
        vectors = np.asarray([row["pre_settle_drift_xyz_m"] for row in group], dtype=float)
        norms = np.linalg.norm(vectors, axis=1) * 1000.0 if len(vectors) else np.asarray([])
        results.append({
            "k": k,
            "n": len(group),
            "mean_drift_xyz_mm": (vectors.mean(axis=0) * 1000.0).tolist() if len(vectors) else [0, 0, 0],
            "mean_drift_norm_mm": float(norms.mean()) if len(norms) else 0.0,
            "std_drift_norm_mm": float(norms.std(ddof=0)) if len(norms) else 0.0,
            "per_tick_curve": settling_curve(group),
        })
    return results


def _hold_horizon_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for horizon in HOLD_HORIZONS:
        group = [row for row in rows if row["condition"] == "settling_horizon" and row["horizon"] == horizon]
        endpoints = np.asarray([
            np.asarray(row["points_xyz_m"][-1], dtype=float) - np.asarray(row["points_xyz_m"][0], dtype=float)
            for row in group
        ], dtype=float)
        results.append({
            "ticks": horizon,
            "n": len(group),
            "mean_delta_xyz_mm": (endpoints.mean(axis=0) * 1000.0).tolist() if len(endpoints) else [0, 0, 0],
            "mean_norm_mm": float((np.linalg.norm(endpoints, axis=1) * 1000.0).mean()) if len(endpoints) else 0.0,
        })
    return results


def _write_baseline_summary(run_dir: Path, metadata: dict[str, Any]) -> None:
    hold_rows = _read_jsonl(run_dir / "hold_trials.jsonl")
    settle_rows = [row for row in hold_rows if row["condition"] == "settling_horizon"]
    pre_rows = [row for row in hold_rows if row["condition"] == "pre_settle"]
    matched = [row for row in hold_rows if row["condition"] == "matched_hold"]
    summary = {
        "metadata": metadata,
        "hold_action_audit": metadata.get("hold_action_audit"),
        "hold_settling_curve": settling_curve(settle_rows),
        "hold_horizon_endpoints": _hold_horizon_summary(hold_rows),
        "pre_settle_response": _pre_settle_summary(pre_rows),
        "matched_post_settle_hold_trials": len(matched),
        "action_trials_completed": len(_read_jsonl(run_dir / "action_trials.jsonl")),
        "recommended_pre_settle_ticks": metadata.get("standardized_pre_settle_ticks"),
        "response_curves": {},
        "opposite_pair_residuals": {},
        "real_robot_actions_executed": 0,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "baseline_summary.json", summary)


def _run_baseline(args: argparse.Namespace, run_dir: Path, metadata: dict[str, Any], vectors: dict[str, list[float]], step_m: float) -> None:
    if (run_dir / "hold_trials.jsonl").exists():
        raise SystemExit(f"baseline artifacts already exist in {run_dir}; refusing to overwrite")
    workspace = (args.workspace_z_min_m, args.workspace_z_max_m)
    environment = LiberoEnvironmentAdapter.create(
        suite_name=metadata["suite"], task_id=metadata["task_id"],
        init_state_index=metadata["init_state_index"], seed=metadata["seed"],
        camera_height=int(metadata["camera_height"]), camera_width=int(metadata["camera_width"]),
        horizon=16,
    )
    try:
        metadata["task_name"] = environment.task_name
        metadata["task_description"] = environment.task_description
        hold_audit = _controller(metadata["controller_config"], vectors)
        metadata["hold_action_audit"] = {
            "action_vector": hold_audit.hold_action().tolist(),
            "position_xyz_command": [0.0, 0.0, 0.0],
            "rotation_axis_angle_command": [0.0, 0.0, 0.0],
            "gripper_command": float(hold_audit.controller.state.gripper_command),
            "construction": "LiberoAtomicController.hold_action -> _action(zero translation), preserving controller gripper state",
            "osc_pose": {
                "control_delta": True,
                "control_ori": True,
                "zero_axis_angle_holds_current_orientation": True,
            },
        }
        for horizon in HOLD_HORIZONS:
            for trial in range(HOLD_TRIALS):
                row = _run_hold_sequence(
                    environment, metadata=metadata, workspace=workspace, settle_ticks=0,
                    horizon=horizon, condition="settling_horizon", trial_index=trial,
                    vectors=vectors, step_m=step_m,
                )
                _append_jsonl(run_dir / "hold_trials.jsonl", row)
        for settle_ticks in PRE_SETTLE_VALUES:
            for trial in range(HOLD_TRIALS):
                row = _run_pre_settle_trial(
                    environment, metadata=metadata, workspace=workspace, settle_ticks=settle_ticks,
                    trial_index=trial, vectors=vectors, step_m=step_m,
                )
                _append_jsonl(run_dir / "hold_trials.jsonl", row)
    finally:
        environment.close()
    metadata["baseline_trials_per_condition"] = HOLD_TRIALS
    _write_json(run_dir / "metadata.json", metadata)
    _write_baseline_summary(run_dir, metadata)


def _mean_matched_hold_delta(rows: list[dict[str, Any]], horizon: int, tick: int) -> np.ndarray:
    group = [row for row in rows if row["condition"] == "matched_hold" and row["horizon"] == horizon]
    deltas = [
        np.asarray(row["timed_points_xyz_m"][tick], dtype=float)
        - np.asarray(row["timed_points_xyz_m"][0], dtype=float)
        for row in group if len(row["timed_points_xyz_m"]) > tick
    ]
    if not deltas:
        raise RuntimeError(f"no matched HOLD response for horizon={horizon}, tick={tick}")
    return np.mean(np.asarray(deltas, dtype=float), axis=0)


def _run_action_trial(
    environment: LiberoEnvironmentAdapter,
    *, metadata: dict[str, Any], workspace: tuple[float, float], token: str,
    direction: list[float], horizon: int, settle_ticks: int, trial_index: int,
    vectors: dict[str, list[float]], step_m: float,
) -> dict[str, Any]:
    controller = _controller(metadata["controller_config"], vectors)
    observer = _fresh_observer(workspace)
    row = _base_row(metadata, condition="action", horizon=horizon, trial_index=trial_index,
                    settle_ticks=settle_ticks)
    row.update({"token": token, "direction_unit": direction})
    first = True
    initial_position: list[float] | None = None
    for _tick in range(settle_ticks):
        outcome = _one_tick(environment, observer, controller, task_id=metadata["runtime_task_id"],
                            token=None, direction=None, step_m=step_m, reset=first,
                            workspace=workspace)
        first = False
        if initial_position is None:
            initial_position = outcome["state_before_eef_xyz_m"]
            row["points_xyz_m"].append(initial_position)
        row["points_xyz_m"].append(outcome["state_after_eef_xyz_m"])
        row["runtime_cycles"].append(outcome)
    if settle_ticks == 0:
        # The first actual action cycle performs reset and captures the calibrated start.
        initial_position = None
    else:
        initial_position = row["points_xyz_m"][-1]
    row["settle_end_eef_xyz_m"] = row["points_xyz_m"][-1] if row["points_xyz_m"] else None
    row["timed_points_xyz_m"] = [initial_position] if settle_ticks > 0 else []
    for _tick in range(horizon):
        outcome = _one_tick(environment, observer, controller, task_id=metadata["runtime_task_id"],
                            token=token, direction=direction, step_m=step_m, reset=first,
                            workspace=workspace)
        first = False
        if initial_position is None:
            initial_position = outcome["state_before_eef_xyz_m"]
            row["points_xyz_m"].append(initial_position)
            row["settle_end_eef_xyz_m"] = initial_position
            row["timed_points_xyz_m"].append(initial_position)
        row["points_xyz_m"].append(outcome["state_after_eef_xyz_m"])
        row["timed_points_xyz_m"].append(outcome["state_after_eef_xyz_m"])
        row["runtime_cycles"].append(outcome)
    row["calibrated_start_eef_xyz_m"] = initial_position
    row["action_vectors"] = controller.action_vectors
    row["trial_status"] = "EXECUTED"
    row["raw_tick_metrics"] = []
    for tick in range(1, horizon + 1):
        delta = (
            np.asarray(row["timed_points_xyz_m"][tick], dtype=float)
            - np.asarray(row["timed_points_xyz_m"][0], dtype=float)
        )
        row["raw_tick_metrics"].append({
            "tick": tick,
            "delta_xyz_m": delta.tolist(),
            "metrics": compute_vector_metrics(delta, direction, step_m * tick),
        })
    return row


def _metrics_for_action_rows(
    action_rows: list[dict[str, Any]], hold_rows: list[dict[str, Any]], step_m: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    response_curves: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []
    corrected_by_token_tick: dict[tuple[str, int], list[float]] = {}
    for row in action_rows:
        token = row["token"]
        horizon = int(row["horizon"])
        direction = row["direction_unit"]
        token_curve = response_curves.setdefault(token, {"by_tick_endpoint": {}, "independent_horizon_trials": []})
        matched_hold_deltas = [
            _mean_matched_hold_delta(hold_rows, tick, tick)
            for tick in range(1, horizon + 1)
        ]
        aggregated = aggregate_temporal_response(
            [raw["delta_xyz_m"] for raw in row["raw_tick_metrics"]],
            matched_hold_deltas,
            direction,
            step_m,
        )
        trial_curve = {"independent_horizon": horizon, "ticks": []}
        for point in aggregated:
            tick = int(point["tick"])
            corrected_delta = point["baseline_corrected_delta_xyz_m"]
            corrected_metrics = point["baseline_corrected_metrics"]
            hold_delta = np.asarray(point["hold_delta_xyz_m"], dtype=float)
            raw_delta = point["raw_delta_xyz_m"]
            raw_metrics = point["raw_metrics"]
            tick_record = {
                "horizon_trial": horizon,
                "n_action_trials": 1,
                "raw_delta_xyz_m": raw_delta,
                "raw_delta_xyz_mm": (np.asarray(raw_delta) * 1000.0).tolist(),
                "raw_metrics": raw_metrics,
                "matched_hold_mean_delta_xyz_m": hold_delta.tolist(),
                "baseline_corrected_delta_xyz_m": corrected_delta,
                "baseline_corrected_delta_xyz_mm": point["baseline_corrected_delta_xyz_mm"],
                "baseline_corrected_metrics": corrected_metrics,
            }
            trial_curve["ticks"].append(tick_record)
            # The endpoint curve uses the independently reset horizon matching its tick.
            # Longer trajectories remain available alongside it as full tick traces.
            if horizon != tick:
                continue
            corrected_by_token_tick[(token, tick)] = corrected_delta
            token_curve["by_tick_endpoint"][str(tick)] = tick_record
            table_rows.append({
                "token": token,
                "tick": tick,
                "independent_horizon": horizon,
                "action_trials": 1,
                "raw_projection_mm": raw_metrics["projection_mm"],
                "raw_realization_ratio": raw_metrics["realization_ratio"],
                "raw_off_axis_magnitude_mm": raw_metrics["off_axis_magnitude_mm"],
                "raw_direction_cosine": raw_metrics["direction_cosine"],
                "raw_delta_xyz_mm": json.dumps(tick_record["raw_delta_xyz_mm"]),
                "hold_delta_xyz_mm": json.dumps((hold_delta * 1000.0).tolist()),
                "corrected_projection_mm": corrected_metrics["projection_mm"],
                "corrected_realization_ratio": corrected_metrics["realization_ratio"],
                "corrected_off_axis_magnitude_mm": corrected_metrics["off_axis_magnitude_mm"],
                "corrected_direction_cosine": corrected_metrics["direction_cosine"],
                "corrected_delta_xyz_mm": json.dumps(point["baseline_corrected_delta_xyz_mm"]),
            })
        token_curve["independent_horizon_trials"].append(trial_curve)

    pair_summary: dict[str, Any] = {}
    for token_a, token_b in OPPOSITE_PAIRS:
        pair_summary[f"{token_a}_vs_{token_b}"] = {}
        for tick in ACTION_HORIZONS:
            if (token_a, tick) not in corrected_by_token_tick or (token_b, tick) not in corrected_by_token_tick:
                continue
            pair_summary[f"{token_a}_vs_{token_b}"][str(tick)] = opposite_pair_metric(
                corrected_by_token_tick[(token_a, tick)], corrected_by_token_tick[(token_b, tick)]
            )
    return response_curves, table_rows, pair_summary


def _write_response_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "token", "tick", "independent_horizon", "action_trials", "raw_projection_mm",
        "raw_realization_ratio", "raw_off_axis_magnitude_mm", "raw_direction_cosine",
        "raw_delta_xyz_mm", "hold_delta_xyz_mm", "corrected_projection_mm",
        "corrected_realization_ratio", "corrected_off_axis_magnitude_mm",
        "corrected_direction_cosine", "corrected_delta_xyz_mm",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_actions(args: argparse.Namespace, run_dir: Path, metadata: dict[str, Any], vectors: dict[str, list[float]], step_m: float) -> None:
    if args.pre_settle_ticks is None:
        raise SystemExit("--phase actions requires evidence-based --pre-settle-ticks")
    if not (run_dir / "baseline_summary.json").exists():
        raise SystemExit("action phase requires a completed --phase baseline in the same --run-dir")
    if (run_dir / "action_trials.jsonl").exists():
        raise SystemExit(f"action artifacts already exist in {run_dir}; refusing to rerun")
    metadata["standardized_pre_settle_ticks"] = args.pre_settle_ticks
    hold_rows = _read_jsonl(run_dir / "hold_trials.jsonl")
    workspace = (args.workspace_z_min_m, args.workspace_z_max_m)
    environment = LiberoEnvironmentAdapter.create(
        suite_name=metadata["suite"], task_id=metadata["task_id"],
        init_state_index=metadata["init_state_index"], seed=metadata["seed"],
        camera_height=int(metadata["camera_height"]), camera_width=int(metadata["camera_width"]),
        horizon=16,
    )
    try:
        # Matched HOLD trajectories use the chosen pre-settle and each exact action horizon.
        existing_matched_horizons = {
            int(row["horizon"]) for row in hold_rows if row["condition"] == "matched_hold"
        }
        for horizon in ACTION_HORIZONS:
            if horizon in existing_matched_horizons:
                continue
            row = _run_hold_sequence(
                environment, metadata=metadata, workspace=workspace,
                settle_ticks=args.pre_settle_ticks, horizon=horizon,
                condition="matched_hold", trial_index=0, vectors=vectors, step_m=step_m,
            )
            _append_jsonl(run_dir / "hold_trials.jsonl", row)
        for token in MOVE_ATOMS:
            if token not in vectors:
                continue
            for horizon in ACTION_HORIZONS:
                row = _run_action_trial(
                    environment, metadata=metadata, workspace=workspace, token=token,
                    direction=vectors[token], horizon=horizon,
                    settle_ticks=args.pre_settle_ticks, trial_index=0,
                    vectors=vectors, step_m=step_m,
                )
                _append_jsonl(run_dir / "action_trials.jsonl", row)
    finally:
        environment.close()
    metadata["action_trials_per_primitive_horizon"] = ACTION_TRIALS
    metadata["real_robot_actions_executed"] = 0
    _write_json(run_dir / "metadata.json", metadata)
    hold_rows = _read_jsonl(run_dir / "hold_trials.jsonl")
    action_rows = _read_jsonl(run_dir / "action_trials.jsonl")
    response_curves, table_rows, pair_summary = _metrics_for_action_rows(action_rows, hold_rows, step_m)
    _write_response_csv(run_dir / "response_table.csv", table_rows)
    incomplete_attempts = _read_jsonl(run_dir / "calibration_failures.jsonl")
    recorded_sim_ticks = sum(len(row.get("runtime_cycles", ())) for row in hold_rows + action_rows)
    discarded_sim_ticks = sum(int(row.get("actions_executed", 0)) for row in incomplete_attempts)
    summary = {
        "metadata": metadata,
        "hold_action_audit": metadata.get("hold_action_audit"),
        "hold_settling_curve": settling_curve([
            row for row in hold_rows if row["condition"] == "settling_horizon"
        ]),
        "hold_horizon_endpoints": _hold_horizon_summary(hold_rows),
        "pre_settle_response": _pre_settle_summary([
            row for row in hold_rows if row["condition"] == "pre_settle"
        ]),
        "matched_post_settle_hold_trials": sum(row["condition"] == "matched_hold" for row in hold_rows),
        "recommended_pre_settle_ticks": args.pre_settle_ticks,
        "response_curves": response_curves,
        "opposite_pair_residuals": pair_summary,
        "incomplete_attempts": incomplete_attempts,
        "recorded_simulated_control_ticks": recorded_sim_ticks,
        "discarded_incomplete_simulated_control_ticks": discarded_sim_ticks,
        "total_simulated_control_ticks_executed": recorded_sim_ticks + discarded_sim_ticks,
        "real_robot_actions_executed": 0,
        "metric_notes": {
            "projection": "signed displacement projection on the configured token direction",
            "realization_ratio": "projection divided by commanded cumulative displacement (5 mm per repeated tick)",
            "off_axis_magnitude": "norm of displacement after removing the token-axis projection",
            "direction_cosine": "signed projection divided by total displacement norm",
            "baseline_correction": "action cumulative delta minus mean matched HOLD cumulative delta at same pre-settle and tick count",
            "opposite_pair_residual": "norm of the sum of the pair's corrected vectors; no pass threshold imposed",
            "action_replication": "one independent reset per token/horizon, as requested; HOLD matched conditions have one trajectory each",
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_readme(run_dir, metadata)


def _write_readme(run_dir: Path, metadata: dict[str, Any]) -> None:
    text = (
        f"Runtime V3 temporal response calibration: {metadata['run_id']}\n"
        f"Suite/task/init/seed: {metadata['suite']} / {metadata['task_id']} / "
        f"{metadata['init_state_index']} / {metadata['seed']}\n"
        f"Standardized pre-settle: {metadata.get('standardized_pre_settle_ticks', 'pending action phase')} HOLD ticks\n"
        "Every physical control tick was a separate Runtime V3 observe -> option -> selector -> arbiter -> Executor -> LIBERO backend cycle.\n"
        "Repeated-token trajectories were measurement procedures; Executor and RuntimeOption remain single-tick.\n"
        "HOLD is zero translation and zero axis-angle under robosuite OSC_POSE control_delta=True, control_ori=True, preserving gripper command.\n"
        "No movement mapping or controller parameter was changed. No real robot action was executed.\n"
    )
    (run_dir / "README.txt").write_text(text, encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    if not np.isfinite(args.commanded_step_mm) or args.commanded_step_mm <= 0:
        raise SystemExit("--commanded-step-mm must be finite and positive")
    if args.workspace_z_min_m >= args.workspace_z_max_m:
        raise SystemExit("workspace Z minimum must be below its maximum")
    cfg = load_yaml(args.config)
    suite = str(args.suite or cfg.get("suite_name", "LIBERO_OBJECT"))
    task_id = int(args.task if args.task is not None else cfg.get("task_id", 0))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    init_index = int(args.init_state_index if args.init_state_index is not None else cfg.get("init_state_index", 0))
    if cfg.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(cfg["libero_dir"])
    raw_vectors = cfg.get("move_vectors")
    if not isinstance(raw_vectors, dict):
        raise SystemExit("config must define move_vectors")
    vectors: dict[str, list[float]] = {}
    for token in MOVE_ATOMS:
        if token not in raw_vectors:
            continue
        direction = np.asarray(raw_vectors[token], dtype=float).reshape(-1)
        if direction.shape != (3,) or not np.isclose(np.linalg.norm(direction), 1.0):
            raise SystemExit(f"{token} mapping must be a 3D unit vector")
        vectors[token] = direction.tolist()
    if set(vectors) != set(MOVE_ATOMS):
        raise SystemExit(f"all six configured translations are required; got {sorted(vectors)}")
    step_m = args.commanded_step_mm / 1000.0
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"--run-dir does not exist: {run_dir}")
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            raise SystemExit(f"run metadata missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        if args.phase != "baseline":
            raise SystemExit("--phase actions requires --run-dir from a completed baseline phase")
        run_dir = _new_run_dir(args.output_dir)
        metadata = {
            "run_id": run_dir.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": str(Path(args.config).resolve()),
            "suite": suite,
            "task_id": task_id,
            "runtime_task_id": f"{suite}:{task_id}",
            "seed": seed,
            "init_state_index": init_index,
            "commanded_step_m": step_m,
            "hold_horizons": list(HOLD_HORIZONS),
            "pre_settle_values": list(PRE_SETTLE_VALUES),
            "action_horizons": list(ACTION_HORIZONS),
            "hold_trials_per_condition": HOLD_TRIALS,
            "action_trials_per_primitive_horizon": ACTION_TRIALS,
            "move_vectors": vectors,
            "controller_config": {
                "step_m": float(cfg.get("step_m", 0.02)),
                "position_scale_m": float(cfg.get("position_scale_m", 0.05)),
                "sim_steps_per_decision": 1,
            },
            "controller_sim_steps_per_tick": 1,
            "camera_height": int(cfg.get("camera_height", 256)),
            "camera_width": int(cfg.get("camera_width", 256)),
            "workspace_z_bounds_m": [args.workspace_z_min_m, args.workspace_z_max_m],
            "runtime_authority": "one independent RuntimeV3Runner max_steps=1 cycle per environment control tick",
            "real_robot_actions_executed": 0,
        }
        _write_json(run_dir / "metadata.json", metadata)
    if metadata["suite"] != suite or metadata["task_id"] != task_id or metadata["seed"] != seed or metadata["init_state_index"] != init_index:
        # A resumed action phase uses the recorded baseline's settings as its source of truth.
        args.suite, args.task, args.seed, args.init_state_index = (
            metadata["suite"], metadata["task_id"], metadata["seed"], metadata["init_state_index"]
        )
    if args.phase == "baseline":
        _run_baseline(args, run_dir, metadata, vectors, step_m)
    else:
        if metadata["move_vectors"] != vectors:
            raise SystemExit("resumed action phase config has different move vectors from baseline")
        _run_actions(args, run_dir, metadata, vectors, step_m)
    print(json.dumps({"phase": args.phase, "run_dir": str(run_dir), "summary": str(run_dir / 'summary.json')}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

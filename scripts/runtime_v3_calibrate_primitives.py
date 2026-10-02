#!/usr/bin/env python3
"""Measure one-step Runtime V3 effects for every configured LIBERO move token."""

from __future__ import annotations

import argparse
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
from core.runtime_v3.calibration import run_calibration_trial
from interpreters.libero_atomic_controller import LiberoAtomicController
from core.action_units import MOVE_ATOMS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/robot_libero_clean_qwen3vl.yaml"),
    )
    parser.add_argument("--suite", default=None)
    parser.add_argument("--task", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--init-state-index", type=int, default=None)
    parser.add_argument("--commanded-step-mm", type=float, default=5.0)
    parser.add_argument("--trials-per-primitive", type=int, choices=(3,), default=3)
    parser.add_argument(
        "--output-dir", default=str(ROOT / "rollouts/runtime_v3_calibration")
    )
    parser.add_argument("--workspace-z-min-m", type=float, default=0.02)
    parser.add_argument("--workspace-z-max-m", type=float, default=0.60)
    return parser


def _runtime_timing(environment: LiberoEnvironmentAdapter) -> dict[str, Any]:
    current = environment.env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "control_freq") and hasattr(current, "model_timestep"):
            control_freq = float(current.control_freq)
            model_timestep = float(current.model_timestep)
            control_timestep = float(current.control_timestep)
            return {
                "control_frequency_hz": control_freq,
                "control_timestep_s": control_timestep,
                "model_timestep_s": model_timestep,
                "simulation_substeps_per_env_step": int(
                    control_timestep / model_timestep
                ),
            }
        current = getattr(current, "env", None)
    return {}


def _trial_aggregate(rows: list[dict[str, Any]], step_m: float) -> dict[str, Any]:
    metrics_rows = [row["contract_metrics"] for row in rows if row.get("contract_metrics")]
    if not metrics_rows:
        return {
            "n": 0,
            "commanded_step_m": step_m,
            "status_counts": _status_counts(rows),
        }
    summary: dict[str, Any] = {
        "n": len(metrics_rows),
        "commanded_step_m": step_m,
        "status_counts": _status_counts(rows),
    }
    columns = {
        "projection_m": "projection_m",
        "realization_ratio": "realization_ratio",
        "off_axis_magnitude_m": "off_axis_magnitude_m",
        "direction_cosine": "direction_cosine",
        "observed_norm_m": "observed_norm_m",
    }
    for label, key in columns.items():
        values = np.asarray([float(row[key]) for row in metrics_rows], dtype=float)
        summary[f"mean_{label}"] = float(values.mean())
        summary[f"std_{label}"] = float(values.std(ddof=0))
    summary["mean_projection_mm"] = summary["mean_projection_m"] * 1000.0
    summary["std_projection_mm"] = summary["std_projection_m"] * 1000.0
    summary["mean_off_axis_magnitude_mm"] = summary["mean_off_axis_magnitude_m"] * 1000.0
    summary["std_off_axis_magnitude_mm"] = summary["std_off_axis_magnitude_m"] * 1000.0
    summary["mean_observed_norm_mm"] = summary["mean_observed_norm_m"] * 1000.0
    return summary


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("trial_status", "UNKNOWN"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _new_run_dir(base: str | Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(base).expanduser() / f"run_{timestamp}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def main() -> int:
    args = _parser().parse_args()
    if not np.isfinite(args.commanded_step_mm) or args.commanded_step_mm <= 0:
        raise SystemExit("--commanded-step-mm must be finite and positive")
    if args.workspace_z_min_m >= args.workspace_z_max_m:
        raise SystemExit("workspace Z minimum must be below its maximum")

    cfg = load_yaml(args.config)
    suite_name = str(args.suite or cfg.get("suite_name", "LIBERO_OBJECT"))
    task_id = int(args.task if args.task is not None else cfg.get("task_id", 0))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    init_state_index = int(
        args.init_state_index
        if args.init_state_index is not None
        else cfg.get("init_state_index", 0)
    )
    if cfg.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(cfg["libero_dir"])

    move_vectors = cfg.get("move_vectors")
    if not isinstance(move_vectors, dict):
        raise SystemExit("config must define move_vectors")
    tokens = [token for token in MOVE_ATOMS if token in move_vectors]
    if not tokens:
        raise SystemExit("no configured MOVE_ATOMS found in move_vectors")
    directions: dict[str, list[float]] = {}
    for token in tokens:
        direction = np.asarray(move_vectors[token], dtype=float).reshape(-1)
        norm = float(np.linalg.norm(direction))
        if direction.shape != (3,) or not np.isclose(norm, 1.0, rtol=1e-6, atol=1e-6):
            raise SystemExit(f"{token} mapping must be a 3D unit vector; got {direction}")
        directions[token] = direction.tolist()

    step_m = float(args.commanded_step_mm) / 1000.0
    output_dir = _new_run_dir(args.output_dir)
    trials_path = output_dir / "trials.jsonl"
    summary_path = output_dir / "summary.json"
    readme_path = output_dir / "README.txt"
    environment = None
    rows_by_token: dict[str, list[dict[str, Any]]] = {token: [] for token in tokens}
    total_actions = 0
    metadata: dict[str, Any] = {
        "run_id": output_dir.name,
        "suite": suite_name,
        "task_id": task_id,
        "seed": seed,
        "init_state_index": init_state_index,
        "commanded_step_m": step_m,
        "trials_per_primitive_requested": args.trials_per_primitive,
        "total_actions_executed": 0,
        "tokens_from_MOVE_ATOMS_and_config": tokens,
        "move_vectors": directions,
        "controller_sim_steps_per_decision": 1,
        "position_scale_m": float(cfg.get("position_scale_m", 0.05)),
        "workspace_z_bounds_m": [args.workspace_z_min_m, args.workspace_z_max_m],
    }
    try:
        environment = LiberoEnvironmentAdapter.create(
            suite_name=suite_name,
            task_id=task_id,
            init_state_index=init_state_index,
            seed=seed,
            camera_height=int(cfg.get("camera_height", 256)),
            camera_width=int(cfg.get("camera_width", 256)),
            horizon=8,
        )
        metadata.update(
            {
                "task_name": environment.task_name,
                "task_description": environment.task_description,
                "timing": _runtime_timing(environment),
            }
        )
        with trials_path.open("w", encoding="utf-8") as stream:
            for token in tokens:
                for trial_index in range(args.trials_per_primitive):
                    # A new controller drops any previous controller-side state;
                    # run_calibration_trial resets to the same fixed init state.
                    controller = LiberoAtomicController(
                        move_vectors=move_vectors,
                        step_m=float(cfg.get("step_m", 0.02)),
                        sim_steps_per_decision=1,
                        position_scale_m=float(cfg.get("position_scale_m", 0.05)),
                    )
                    observer = LiberoObservationAdapter(
                        max_eef_z_m=args.workspace_z_max_m,
                        min_eef_z_m=args.workspace_z_min_m,
                        safe_lift_step_m=step_m,
                    )
                    outcome = run_calibration_trial(
                        environment,
                        observer,
                        controller,
                        task_id=f"{suite_name}:{task_id}",
                        token=token,
                        direction_unit=directions[token],
                        commanded_step_m=step_m,
                        workspace_z_bounds_m=(
                            args.workspace_z_min_m,
                            args.workspace_z_max_m,
                        ),
                    )
                    total_actions += int(outcome["actions"])
                    row = {
                        "suite": suite_name,
                        "task_id": task_id,
                        "seed": seed,
                        "init_state_index": init_state_index,
                        "primitive": token,
                        "trial_index": trial_index,
                        "trials_per_primitive": args.trials_per_primitive,
                        "commanded_direction_unit": directions[token],
                        "commanded_step_m": step_m,
                        "commanded_step_mm": args.commanded_step_mm,
                        "trial_status": (
                            outcome["skip_reason"]
                            or (
                                "COMPLETED"
                                if outcome["actions"] == 1
                                and outcome["contract_metrics"] is not None
                                and outcome["backend_execution"]
                                else "FAILED"
                            )
                        ),
                        **outcome,
                    }
                    rows_by_token[token].append(row)
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    stream.flush()
                    print(
                        f"{token} trial {trial_index + 1}/{args.trials_per_primitive}: "
                        f"{row['trial_status']} actions={outcome['actions']} "
                        f"metrics={json.dumps(outcome['contract_metrics'])}"
                    )

        metadata["total_actions_executed"] = total_actions
        summary: dict[str, Any] = {"_metadata": metadata}
        for token, rows in rows_by_token.items():
            summary[token] = _trial_aggregate(rows, step_m)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        timing = metadata.get("timing", {})
        readme_path.write_text(
            "Runtime V3 Actuation Contract Calibration\n"
            "===========================================\n\n"
            f"Run: {output_dir.name}\n"
            f"Task: {suite_name}:{task_id} — {metadata['task_name']}\n"
            f"Seed / init state: {seed} / {init_state_index}\n"
            f"Commanded step: {args.commanded_step_mm:.3f} mm\n"
            f"Trials per configured translation token: {args.trials_per_primitive}\n"
            f"Tokens: {', '.join(tokens)}\n"
            f"Real actions executed: {total_actions}\n"
            f"Controller setting: sim_steps_per_decision=1, position_scale_m="
            f"{metadata['position_scale_m']}\n"
            f"LIBERO control frequency: {timing.get('control_frequency_hz', 'unknown')} Hz; "
            f"one adapter step advances {timing.get('control_timestep_s', 'unknown')} s "
            f"using {timing.get('simulation_substeps_per_env_step', 'unknown')} simulation substeps.\n\n"
            "Each trial resets the environment to the same fixed init state, observes once, "
            "selects one deterministic option, authorizes it through the Arbiter, and sends "
            "one action through Executor and LiberoPrimitiveBackend. There is no action loop "
            "or repeat within a trial. `effect_achieved_legacy` is preserved only as the "
            "old 0.1 mm projection check; it is not a contract metric. No new verification "
            "threshold is selected here.\n\n"
            "Summary statistics use completed trials only and population standard deviation. "
            "See trials.jsonl for every reset, EEF before/after position, legacy effect flag, "
            "and per-trial contract metrics. Workspace Z boundary skips are kept in the log.\n",
            encoding="utf-8",
        )
        print(f"Calibration artifacts: {output_dir}")
        all_trials_accounted_for = all(
            row["trial_status"] in {"COMPLETED", "SKIPPED_WORKSPACE_BOUNDARY"}
            for token_rows in rows_by_token.values()
            for row in token_rows
        )
        return 0 if all_trials_accounted_for else 2
    except Exception as exc:
        print(f"Calibration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"Partial trial log: {trials_path}", file=sys.stderr)
        return 1
    finally:
        if environment is not None:
            environment.close()


if __name__ == "__main__":
    raise SystemExit(main())

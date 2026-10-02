#!/usr/bin/env python3
"""Run the fixed task-2, seed-0 cross-state bounded micro-motion calibration."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.action_units import MOVE_ATOMS
from core.config import load_yaml
from core.runtime_v3.adapters.libero_env import LiberoEnvironmentAdapter
from core.runtime_v3.adapters.libero_observation import LiberoObservationAdapter, RawObservation
from core.runtime_v3.arbiter import Arbiter
from core.runtime_v3.effects import EffectObserver
from core.runtime_v3.executor import Executor, LiberoPrimitiveBackend
from core.runtime_v3.micro_motion import BoundedMicroMotionOptionGenerator
from core.runtime_v3.runner import RuntimeV3Runner
from core.runtime_v3.selector import DeterministicSelector
from core.runtime_v3.state import StateBuilder
from core.runtime_v3.temporal_calibration import run_v3_tick
from interpreters.libero_atomic_controller import LiberoAtomicController


TASK_ID = 2
SEED = 0
INIT_STATES = (0, 1, 2)
PRE_SETTLE_TICKS = 4
REQUESTED_MM = 3.0
MAX_TICKS = 5
CONTROL_TICK_MM = 5.0
DIRECTION_ORDER = ("FWD", "BACK", "LEFT", "RIGHT", "UP", "DOWN")


class _CountingArbiter(Arbiter):
    def __init__(self) -> None:
        super().__init__()
        self.authorization_calls = 0

    def authorize(self, *args, **kwargs):
        self.authorization_calls += 1
        return super().authorize(*args, **kwargs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(ROOT / "configs/robot_libero_clean_qwen3vl.yaml"),
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "rollouts/runtime_v3_micro_motion"),
    )
    return parser


def _new_run_dir(base: str | Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(base).expanduser() / f"run_{timestamp}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_video(trial_dir: Path, frames: list[RawObservation]) -> None:
    import cv2

    agent_dir = trial_dir / "images" / "agentview"
    wrist_dir = trial_dir / "images" / "wrist"
    agent_dir.mkdir(parents=True, exist_ok=True)
    wrist_dir.mkdir(parents=True, exist_ok=True)
    video_frames = []
    for index, frame in enumerate(frames):
        agent = cv2.cvtColor(frame.agentview_rgb, cv2.COLOR_RGB2BGR)
        wrist = frame.wrist_rgb
        if wrist is None:
            wrist_bgr = np.zeros_like(agent)
        else:
            wrist_bgr = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(agent_dir / f"{index:04d}.png"), agent)
        cv2.imwrite(str(wrist_dir / f"{index:04d}.png"), wrist_bgr)
        video_frames.append(np.concatenate((agent, wrist_bgr), axis=1))
    if not video_frames:
        return
    height, width = video_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(trial_dir / "visualization.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height),
    )
    try:
        for frame in video_frames:
            writer.write(frame)
    finally:
        writer.release()


def _vector(state: Any) -> list[float] | None:
    if state is None or not isinstance(state.end_effector_state, dict):
        return None
    raw = state.end_effector_state.get("position_xyz")
    if not isinstance(raw, (tuple, list)) or len(raw) != 3:
        return None
    return [float(value) for value in raw]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def group_metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        executions = [row["execution"] for row in group]
        projections = [float(item["actual_projection_m"]) for item in executions]
        errors = [float(item["absolute_displacement_error_m"]) for item in executions]
        ticks = [int(item["ticks_executed"]) for item in executions]
        off_axis = [float(item["off_axis_magnitude_m"]) for item in executions]
        cosines = [
            float(item["direction_cosine"])
            for item in executions if item["direction_cosine"] is not None
        ]
        return {
            "n": len(group),
            "target_reached_rate": sum(
                projection >= REQUESTED_MM / 1000.0 for projection in projections
            ) / len(group) if group else 0.0,
            "mean_requested_displacement_mm": REQUESTED_MM if group else None,
            "mean_actual_projection_mm": float(np.mean(projections)) * 1000.0 if group else None,
            "mean_absolute_displacement_error_mm": float(np.mean(errors)) * 1000.0 if group else None,
            "mean_ticks": float(np.mean(ticks)) if group else None,
            "mean_off_axis_magnitude_mm": float(np.mean(off_axis)) * 1000.0 if group else None,
            "mean_direction_cosine": float(np.mean(cosines)) if cosines else None,
            "termination_reason_counts": dict(Counter(item["termination"] for item in executions)),
        }

    by_direction = {
        direction: group_metrics([row for row in rows if row["direction"] == direction])
        for direction in DIRECTION_ORDER
    }
    by_init_state = {
        str(index): group_metrics([row for row in rows if row["init_state_index"] == index])
        for index in INIT_STATES
    }
    overall_termination = Counter(row["execution"]["termination"] for row in rows)
    return {
        "overall_bounded_executions": len(rows),
        "target_reached_rate": sum(
            row["execution"]["actual_projection_m"] >= REQUESTED_MM / 1000.0 for row in rows
        ) / len(rows) if rows else 0.0,
        "termination_reason_counts": dict(overall_termination),
        "by_direction": by_direction,
        "by_init_state": by_init_state,
    }


def _calibrate_one(
    environment: LiberoEnvironmentAdapter,
    *,
    direction: str,
    direction_unit: list[float],
    move_vectors: dict[str, list[float]],
    config: dict[str, Any],
    workspace: tuple[float, float],
    trial_dir: Path,
    init_state_index: int,
) -> dict[str, Any]:
    raw_frames: list[RawObservation] = []
    controller = LiberoAtomicController(
        move_vectors=move_vectors,
        step_m=float(config.get("step_m", 0.02)),
        sim_steps_per_decision=1,
        position_scale_m=float(config.get("position_scale_m", 0.05)),
    )
    observer = LiberoObservationAdapter(
        max_eef_z_m=workspace[1],
        min_eef_z_m=workspace[0],
        safe_lift_step_m=CONTROL_TICK_MM / 1000.0,
        on_raw_observation=raw_frames.append,
    )
    pre_settle_cycles = []
    reset = True
    for _ in range(PRE_SETTLE_TICKS):
        outcome = run_v3_tick(
            environment,
            observer,
            controller,
            task_id=f"LIBERO_OBJECT:{TASK_ID}",
            token=None,
            direction_unit=None,
            commanded_step_m=CONTROL_TICK_MM / 1000.0,
            reset=reset,
            workspace_z_bounds_m=workspace,
        )
        reset = False
        if outcome["actions"] != 1 or not outcome["backend_execution"]:
            raise RuntimeError(f"V3 HOLD pre-settle tick failed: {outcome}")
        pre_settle_cycles.append(outcome)

    arbiter = _CountingArbiter()
    generator = BoundedMicroMotionOptionGenerator(
        direction,
        direction_unit,
        requested_displacement_m=REQUESTED_MM / 1000.0,
        max_ticks=MAX_TICKS,
        control_tick_step_m=CONTROL_TICK_MM / 1000.0,
    )
    backend = LiberoPrimitiveBackend(environment, controller, arbiter)
    events: list[dict[str, Any]] = []
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
    run_result = runner.run_episode(
        environment,
        task_id=f"LIBERO_OBJECT:{TASK_ID}",
        max_steps=1,
        reset=False,
    )
    if (run_result.get("actions") != 1 or len(events) != 1
            or arbiter.authorization_calls != 1):
        raise RuntimeError(f"V3 bounded execution did not complete exactly once: {run_result}")
    event = events[0]
    execution_record = event["execution"]
    execution = execution_record.result
    if not isinstance(execution, dict):
        raise RuntimeError("V3 Executor returned no bounded execution record")
    _write_video(trial_dir, raw_frames)
    initial_position = execution["tick_observations"][0]["eef_position_xyz_m"] if execution["tick_observations"] else _vector(event["state_before"])
    return {
        "trial_id": trial_dir.name,
        "suite": "LIBERO_OBJECT",
        "task_id": TASK_ID,
        "task_name": environment.task_name,
        "seed": SEED,
        "init_state_index": init_state_index,
        "direction": direction,
        "runtime_option": generator.option_id,
        "arbiter_approvals_for_bounded_motion": arbiter.authorization_calls,
        "runner_motion_actions": int(run_result["actions"]),
        "pre_settle_ticks": PRE_SETTLE_TICKS,
        "pre_settle_control_cycles": pre_settle_cycles,
        "state_before_eef_xyz_m": _vector(event["state_before"]),
        "first_tick_observed_eef_xyz_m": initial_position,
        "execution": execution,
        "video": str((trial_dir / "visualization.mp4").name),
        "observed_frame_count": len(raw_frames),
        "runner_status": run_result.get("status"),
        "effect_observer_achieved_diagnostic": getattr(event.get("effect"), "achieved", None),
    }


def main() -> int:
    args = _parser().parse_args()
    config = load_yaml(args.config)
    if int(config.get("task_id", TASK_ID)) != TASK_ID or int(config.get("seed", SEED)) != SEED:
        raise SystemExit("This calibration is fixed to LIBERO_OBJECT task 2 and seed 0.")
    if config.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(config["libero_dir"])

    raw_vectors = config.get("move_vectors")
    if not isinstance(raw_vectors, dict):
        raise SystemExit("config must define move_vectors")
    move_vectors: dict[str, list[float]] = {}
    direction_vectors: dict[str, list[float]] = {}
    for direction in DIRECTION_ORDER:
        token = f"MV_{direction}"
        if token not in MOVE_ATOMS or token not in raw_vectors:
            raise SystemExit(f"missing configured movement atom {token}")
        vector = np.asarray(raw_vectors[token], dtype=float).reshape(-1)
        if vector.shape != (3,) or not np.isclose(np.linalg.norm(vector), 1.0):
            raise SystemExit(f"{token} must map to a 3D unit vector, got {vector.tolist()}")
        move_vectors[token] = vector.tolist()
        direction_vectors[direction] = vector.tolist()

    workspace = (0.02, 0.60)
    output_dir = _new_run_dir(args.output_dir)
    trials_path = output_dir / "trials.jsonl"
    summary_path = output_dir / "summary.json"
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "run_id": output_dir.name,
        "suite": "LIBERO_OBJECT",
        "task_id": TASK_ID,
        "seed": SEED,
        "init_state_indices": list(INIT_STATES),
        "directions": list(DIRECTION_ORDER),
        "requested_displacement_mm": REQUESTED_MM,
        "max_ticks": MAX_TICKS,
        "control_tick_step_mm": CONTROL_TICK_MM,
        "standardized_pre_settle_hold_ticks": PRE_SETTLE_TICKS,
        "controller_sim_steps_per_decision": 1,
        "workspace_z_bounds_m": list(workspace),
        "move_vectors": direction_vectors,
        "total_bounded_executions_expected": len(INIT_STATES) * len(DIRECTION_ORDER),
    }
    with trials_path.open("w", encoding="utf-8") as stream:
        for init_state_index in INIT_STATES:
            environment = LiberoEnvironmentAdapter.create(
                suite_name="LIBERO_OBJECT",
                task_id=TASK_ID,
                init_state_index=init_state_index,
                seed=SEED,
                camera_height=int(config.get("camera_height", 256)),
                camera_width=int(config.get("camera_width", 256)),
                horizon=32,
            )
            try:
                available = len(environment.handle.init_states)
                if available <= max(INIT_STATES):
                    raise RuntimeError(
                        f"task {TASK_ID} has {available} init states; cannot execute fixed states 0,1,2"
                    )
                metadata.setdefault("task_name", environment.task_name)
                metadata.setdefault("task_description", environment.task_description)
                for direction in DIRECTION_ORDER:
                    trial_dir = output_dir / f"state_{init_state_index}" / direction.lower()
                    trial_dir.mkdir(parents=True, exist_ok=False)
                    row = _calibrate_one(
                        environment,
                        direction=direction,
                        direction_unit=direction_vectors[direction],
                        move_vectors=move_vectors,
                        config=config,
                        workspace=workspace,
                        trial_dir=trial_dir,
                        init_state_index=init_state_index,
                    )
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
            finally:
                environment.close()

    summary = {
        "metadata": metadata,
        **_summarize(rows),
        "by_state_and_direction": {
            f"state_{state_index}_{direction}": _summarize([
                row for row in rows
                if row["init_state_index"] == state_index and row["direction"] == direction
            ])["by_direction"][direction]
            for state_index in INIT_STATES for direction in DIRECTION_ORDER
        },
        "trials_path": str(trials_path.relative_to(ROOT)),
    }
    _write_json(summary_path, summary)
    (output_dir / "README.txt").write_text(
        "Runtime V3 bounded micro-motion calibration.\n"
        "Fixed LIBERO_OBJECT task 2, seed 0, init states 0/1/2. Each trial uses four\n"
        "Arbiter-approved HOLD pre-settle ticks, then exactly one Arbiter-approved\n"
        "semantic bounded motion with a 3 mm target and five-tick maximum.\n"
        "Target reached is projection >= 3 mm; off-axis, overshoot, and cosine are\n"
        "diagnostic fields. See trials.jsonl, summary.json, and per-trial videos.\n",
        encoding="utf-8",
    )
    print(json.dumps({"run_dir": str(output_dir), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

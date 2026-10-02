#!/usr/bin/env python3
"""Run one bounded Runtime V3 LIBERO action or a no-action Qwen selector check."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_secrets_env, load_yaml
from core.runtime_v3.adapters.libero_env import LiberoEnvironmentAdapter
from core.runtime_v3.adapters.libero_observation import LiberoObservationAdapter, RawObservation
from core.runtime_v3.adapters.qwen_selector import QwenSelectorAdapter
from core.runtime_v3.arbiter import Arbiter
from core.runtime_v3.effects import EffectObserver
from core.runtime_v3.executor import Executor, LiberoPrimitiveBackend
from core.runtime_v3.runner import RuntimeV3Runner
from core.runtime_v3.selector import DeterministicSelector
from core.runtime_v3.smoke_options import SmokeOptionGenerator
from core.runtime_v3.state import StateBuilder
from interpreters.libero_atomic_controller import LiberoAtomicController


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


class SmokeRunLog:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._steps = (output_dir / "steps.jsonl").open("a", encoding="utf-8")
        self._observations = (output_dir / "observations.jsonl").open("a", encoding="utf-8")

    def __call__(self, record: dict[str, Any]) -> None:
        self._write(self._steps, record)

    def save_observation(self, record: RawObservation) -> None:
        image_dir = self.output_dir / "images" / f"observation_{record.observation_index:04d}"
        image_dir.mkdir(parents=True, exist_ok=True)
        agentview_path = image_dir / "agentview.png"
        Image.fromarray(record.agentview_rgb).save(agentview_path)
        wrist_path = None
        if record.wrist_rgb is not None:
            wrist_path = image_dir / "wrist.png"
            Image.fromarray(record.wrist_rgb).save(wrist_path)
        self._write(self._observations, {
            "observation_index": record.observation_index,
            "environment_step": record.environment_step,
            "timestamp_monotonic": record.timestamp_monotonic,
            "agentview_path": str(agentview_path.relative_to(self.output_dir)),
            "wrist_path": str(wrist_path.relative_to(self.output_dir)) if wrist_path else None,
            "eef_position_xyz": record.eef_position_xyz,
            "eef_quaternion": record.eef_quaternion,
            "gripper_width_m": record.gripper_width_m,
            "raw_keys": record.raw_keys,
        })

    def write_record(self, filename: str, record: dict[str, Any]) -> None:
        path = self.output_dir / filename
        path.write_text(json.dumps(_jsonable(record), indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    @staticmethod
    def _write(stream, record: dict[str, Any]) -> None:
        stream.write(json.dumps(_jsonable(record), ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()

    def close(self) -> None:
        self._steps.close()
        self._observations.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/robot_libero_clean_qwen3vl.yaml"))
    parser.add_argument("--task", "--task-id", dest="task", type=int, default=None)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--init-state-index", type=int, default=None)
    parser.add_argument("--mode", choices=("deterministic", "qwen-offline"), default="deterministic")
    parser.add_argument("--max-real-actions", type=int, default=1)
    parser.add_argument("--output-dir", default=str(ROOT / "rollouts/runtime_v3_smoke"))
    parser.add_argument("--workspace-z-min-m", type=float, default=0.02)
    parser.add_argument("--workspace-z-max-m", type=float, default=0.60)
    parser.add_argument("--lift-step-mm", type=float, default=5.0)
    parser.add_argument("--vlm-backend", default=None)
    parser.add_argument("--vlm-url", default=os.environ.get("VLM_URL") or os.environ.get("VLLM_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL"))
    return parser


def _new_output_dir(base: str | Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(base).expanduser() / f"run_{stamp}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _make_vlm_client(args: argparse.Namespace, cfg: dict[str, Any]):
    from core.sim.launch import build_config, make_vlm_client

    vlm_args = argparse.Namespace(
        task_suite_name=None,
        task_id=None,
        episode_index=None,
        max_steps=None,
        loop_period_s=None,
        log_dir=None,
        vlm_backend=args.vlm_backend,
        vlm_url=args.vlm_url,
        model=args.model,
    )
    resolved = build_config(vlm_args, cfg)
    return make_vlm_client(vlm_args, resolved)


def main() -> int:
    args = _parser().parse_args()
    if args.max_real_actions not in (0, 1):
        raise SystemExit("--max-real-actions must be 0 or 1 for this single-step smoke")
    if args.mode == "qwen-offline" and args.max_real_actions != 0:
        raise SystemExit("qwen-offline mode requires --max-real-actions 0")
    if args.mode == "deterministic" and args.max_real_actions == 0:
        raise SystemExit("deterministic mode requires one approved action; use qwen-offline for zero-action mode")

    cfg = load_yaml(args.config)
    suite_name = str(args.suite or cfg.get("suite_name", "LIBERO_OBJECT"))
    task_id = int(args.task if args.task is not None else cfg.get("task_id", 0))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    init_state_index = int(args.init_state_index if args.init_state_index is not None
                           else cfg.get("init_state_index", 0))
    if cfg.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(cfg["libero_dir"])

    output_dir = _new_output_dir(args.output_dir)
    log = SmokeRunLog(output_dir)
    environment = None
    final: dict[str, Any] = {
        "mode": args.mode,
        "suite": suite_name,
        "task_id": task_id,
        "seed": seed,
        "init_state_index": init_state_index,
        "max_real_actions": args.max_real_actions,
        "status": "ENV_INIT_FAILED",
        "output_dir": str(output_dir),
    }
    try:
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
        except Exception as exc:
            final.update({"status": "ENV_INIT_FAILED", "reason": f"{type(exc).__name__}: {exc}"})
            return _finish(log, final, 1)

        observer = LiberoObservationAdapter(
            max_eef_z_m=args.workspace_z_max_m,
            min_eef_z_m=args.workspace_z_min_m,
            safe_lift_step_m=args.lift_step_mm / 1000.0,
            on_raw_observation=log.save_observation,
        )
        state_builder = StateBuilder()
        if args.mode == "deterministic":
            arbiter = Arbiter()
            controller = LiberoAtomicController(
                move_vectors=cfg["move_vectors"],
                step_m=args.lift_step_mm / 1000.0,
                sim_steps_per_decision=1,
                position_scale_m=float(cfg.get("position_scale_m", 0.05)),
            )
            backend = LiberoPrimitiveBackend(environment, controller, arbiter)
            runner = RuntimeV3Runner(
                observer=observer,
                state_builder=state_builder,
                option_generator=SmokeOptionGenerator(args.lift_step_mm / 1000.0),
                selector=DeterministicSelector("OPTION_SAFE_LIFT"),
                arbiter=arbiter,
                executor=Executor(backend, arbiter),
                effect_observer=EffectObserver(),
                logger=log,
            )
            result = runner.run_episode(
                environment,
                task_id=f"{suite_name}:{task_id}",
                max_steps=1,
                reset=False,
            )
            state = result.get("state")
            effect = state.last_observed_effect if state is not None else None
            effect_achieved = effect.get("achieved") if isinstance(effect, dict) else None
            if result.get("actions") == 1 and effect_achieved is True:
                status = "REAL_SMOKE_PASSED"
            elif result.get("actions") == 1:
                status = "EFFECT_NOT_OBSERVED"
            else:
                status = str(result.get("status", "EXECUTION_FAILED"))
            final.update({
                "status": status,
                "task_name": environment.task_name,
                "task_instruction": environment.task_description,
                "runtime_status": result.get("status"),
                "actions": result.get("actions", 0),
                "state_after": state,
                "effect": effect,
            })
            log.write_record("result.json", final)
            print(json.dumps(_jsonable(final), indent=2, ensure_ascii=False))
            return 0 if status == "REAL_SMOKE_PASSED" else 1

        # Offline selector path: real observation/state, bounded options, no
        # controller or backend constructed, and no Arbiter/Executor call.
        try:
            observation = observer.observe(environment)
            state = state_builder.update(state_builder.initialize(f"{suite_name}:{task_id}"), observation)
            options = SmokeOptionGenerator(args.lift_step_mm / 1000.0).offline_options(state)
            if not 2 <= len(options) <= 4:
                final.update({"status": "OPTION_GENERATION_FAILED",
                              "reason": f"expected 2-4 options, received {len(options)}"})
                return _finish(log, final, 1)
            load_secrets_env()
            client = _make_vlm_client(args, cfg)
            client.health_check(wait_s=0.0)
            selector = QwenSelectorAdapter(client, environment.task_description)
            selection = selector.select(state, options)
            final.update({
                "status": "QWEN_OFFLINE_SMOKE_COMPLETED",
                "task_name": environment.task_name,
                "task_instruction": environment.task_description,
                "model": selector.model,
                "option_count": len(options),
                "options": options,
                "selection": selection,
                "qwen_record": selector.last_record,
                "real_robot_action_executed": False,
            })
            log.write_record("qwen_selector.json", final)
            print(json.dumps(_jsonable(final), indent=2, ensure_ascii=False))
            return 0 if selection.status != "INVALID_SELECTION" else 1
        except Exception as exc:
            final.update({"status": "QWEN_ADAPTER_BLOCKED",
                          "reason": f"{type(exc).__name__}: {exc}",
                          "real_robot_action_executed": False})
            return _finish(log, final, 1)
    finally:
        if environment is not None:
            environment.close()
        log.close()


def _finish(log: SmokeRunLog, record: dict[str, Any], exit_code: int) -> int:
    log.write_record("result.json", record)
    print(json.dumps(_jsonable(record), indent=2, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

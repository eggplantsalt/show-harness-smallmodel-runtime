#!/usr/bin/env python3
"""Run three one-motion object-relative alignment trials and one Qwen no-action smoke."""

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
from urllib.parse import urlparse

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.capabilities.sam3_client import Sam3Client
from core.config import load_secrets_env, load_yaml
from core.runtime_v3.adapters.libero_env import LiberoEnvironmentAdapter
from core.runtime_v3.adapters.libero_observation import LiberoObservationAdapter
from core.runtime_v3.adapters.qwen_selector import QwenSelectorAdapter
from core.runtime_v3.arbiter import Arbiter, DecisionKind
from core.runtime_v3.effects import EffectObserver
from core.runtime_v3.executor import Executor, LiberoPrimitiveBackend
from core.runtime_v3.object_relative import (
    ObjectRelativeAlignmentOptionGenerator,
    ObjectRelativePerceptionObserver,
)
from core.runtime_v3.runner import RuntimeV3Runner
from core.runtime_v3.selector import DeterministicSelector
from core.runtime_v3.state import BeliefState, StateBuilder
from core.runtime_v3.temporal_calibration import run_v3_tick
from core.sim.launch import build_config, make_vlm_client
from interpreters.libero_atomic_controller import LiberoAtomicController


SUITE = "LIBERO_OBJECT"
TASK_ID = 2
TARGET_PHRASE = "salad dressing"
INIT_STATES = (0, 1, 2)
PRE_SETTLE_TICKS = 4
CAMERA_HEIGHT = 512
CAMERA_WIDTH = 512
REQUESTED_MM = 3.0
MAX_TICKS = 5
CONTROL_TICK_MM = 5.0


class CountingArbiter(Arbiter):
    def __init__(self) -> None:
        super().__init__()
        self.authorization_calls = 0
        self.approval_count = 0

    def authorize(self, *args, **kwargs):
        self.authorization_calls += 1
        decision = super().authorize(*args, **kwargs)
        if decision.kind == DecisionKind.APPROVED:
            self.approval_count += 1
        return decision


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/robot_libero_clean_qwen3vl.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "rollouts/runtime_v3_object_relative_alignment"))
    parser.add_argument("--sam3-url", default="http://127.0.0.1:8773/sse")
    parser.add_argument("--sam3-python", default="/root/autodl-tmp/openeta-services/sam3/.venv/bin/python")
    parser.add_argument("--sam3-timeout-s", type=float, default=120.0)
    parser.add_argument("--vlm-backend", default=None)
    parser.add_argument("--vlm-url", default=os.environ.get("VLM_URL") or os.environ.get("VLLM_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL"))
    return parser


def _new_run_dir(base: str | Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(base).expanduser() / f"run_{stamp}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(path: Path, record: Any) -> None:
    path.write_text(json.dumps(_jsonable(record), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def _make_vlm_client(args: argparse.Namespace, config: dict[str, Any]):
    vlm_args = argparse.Namespace(
        task_suite_name=None, task_id=None, episode_index=None, max_steps=None,
        loop_period_s=None, log_dir=None, vlm_backend=args.vlm_backend,
        vlm_url=args.vlm_url, model=args.model,
    )
    resolved = build_config(vlm_args, config)
    return make_vlm_client(vlm_args, resolved)


def _configure_local_sam3_proxy_bypass(url: str) -> None:
    """Configure this V3 process so its existing MCP bridge reaches loopback."""
    if urlparse(str(url)).hostname not in {"127.0.0.1", "localhost", "::1"}:
        return
    entries: list[str] = []
    for name in ("NO_PROXY", "no_proxy"):
        entries.extend(item.strip() for item in os.environ.get(name, "").split(",") if item.strip())
    entries.extend(("127.0.0.1", "localhost", "::1"))
    value = ",".join(dict.fromkeys(entries))
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def _raw_resolution(observer: LiberoObservationAdapter) -> dict[str, Any]:
    raw = observer.last_raw
    if raw is None:
        return {"agentview": None, "wrist": None}
    return {
        "agentview": {"width": int(raw.agentview_rgb.shape[1]),
                      "height": int(raw.agentview_rgb.shape[0])},
        "wrist": ({"width": int(raw.wrist_rgb.shape[1]), "height": int(raw.wrist_rgb.shape[0])}
                  if raw.wrist_rgb is not None else None),
    }


def _run_trial(
    *,
    init_state_index: int,
    run_dir: Path,
    config: dict[str, Any],
    sam3: Sam3Client,
    workspace: tuple[float, float],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    trial_dir = run_dir / f"init_state_{init_state_index}"
    trial_dir.mkdir(parents=True, exist_ok=False)
    environment = LiberoEnvironmentAdapter.create(
        suite_name=SUITE,
        task_id=TASK_ID,
        init_state_index=init_state_index,
        seed=0,
        camera_height=CAMERA_HEIGHT,
        camera_width=CAMERA_WIDTH,
        horizon=32,
    )
    try:
        if TARGET_PHRASE.casefold() not in environment.task_description.casefold():
            raise RuntimeError(
                f"task metadata does not contain requested target phrase {TARGET_PHRASE!r}: "
                f"{environment.task_description!r}"
            )
        controller = LiberoAtomicController(
            move_vectors=config["move_vectors"],
            step_m=CONTROL_TICK_MM / 1000.0,
            sim_steps_per_decision=1,
            position_scale_m=float(config.get("position_scale_m", 0.05)),
        )
        base_observer = LiberoObservationAdapter(
            max_eef_z_m=workspace[1],
            min_eef_z_m=workspace[0],
            safe_lift_step_m=CONTROL_TICK_MM / 1000.0,
        )
        pre_settle = []
        reset = True
        for _ in range(PRE_SETTLE_TICKS):
            outcome = run_v3_tick(
                environment,
                base_observer,
                controller,
                task_id=f"{SUITE}:{TASK_ID}",
                token=None,
                direction_unit=None,
                commanded_step_m=CONTROL_TICK_MM / 1000.0,
                reset=reset,
                workspace_z_bounds_m=workspace,
            )
            reset = False
            if outcome["actions"] != 1 or not outcome["backend_execution"]:
                raise RuntimeError(f"V3 HOLD pre-settle failed: {outcome}")
            pre_settle.append(outcome)

        observer = ObjectRelativePerceptionObserver(
            base_observer,
            sam3,
            target_phrase=TARGET_PHRASE,
            move_vectors=config["move_vectors"],
        )
        arbiter = CountingArbiter()
        events: list[dict[str, Any]] = []
        backend = LiberoPrimitiveBackend(environment, controller, arbiter)
        runner = RuntimeV3Runner(
            observer=observer,
            state_builder=StateBuilder(),
            option_generator=ObjectRelativeAlignmentOptionGenerator(),
            selector=DeterministicSelector("ALIGN_TO_TARGET_SMALL"),
            arbiter=arbiter,
            executor=Executor(backend, arbiter),
            effect_observer=EffectObserver(),
            logger=events.append,
        )
        result = runner.run_episode(
            environment,
            task_id=f"{SUITE}:{TASK_ID}",
            max_steps=1,
            reset=False,
        )
        history = observer.perception_history
        initial = history[0] if history else None
        final = history[-1] if len(history) > 1 else None
        if initial is not None:
            before_artifacts = observer.save_visual_artifacts(
                str(trial_dir / "artifacts"), image=initial["image"], prefix="before",
                segmentation=initial["segmentation"], resolution=initial["resolution"],
            )
        else:
            before_artifacts = {}
        if final is not None:
            after_artifacts = observer.save_visual_artifacts(
                str(trial_dir / "artifacts"), image=final["image"], prefix="after",
                segmentation=final["segmentation"], resolution=final["resolution"],
            )
        else:
            after_artifacts = {}
        event = events[0] if events else {}
        state_before = event.get("state_before")
        state_after = event.get("state_after")
        execution_record = event.get("execution")
        execution = getattr(execution_record, "result", None)
        before_state = initial.get("object_relative_state") if initial else None
        after_state = final.get("object_relative_state") if final else None
        error_before = (before_state.image_error_norm_px if before_state is not None else None)
        error_after = (after_state.image_error_norm_px if after_state is not None else None)
        expected_effect = event.get("approved_action").expected_effect if event.get("approved_action") else {}
        chosen_spec = (event.get("approved_action").primitive.micro_motion_spec
                       if event.get("approved_action") else None)
        predicted_after = expected_effect.get("predicted_image_error_after_px")
        predicted_improvement = expected_effect.get("predicted_improvement_px")
        actual_improvement = (float(error_before) - float(error_after)
                              if error_before is not None and error_after is not None else None)
        centroid_shift = None
        if (before_state is not None and after_state is not None
                and before_state.target_centroid_px is not None
                and after_state.target_centroid_px is not None):
            centroid_shift = float(np.linalg.norm(
                np.asarray(after_state.target_centroid_px, dtype=float)
                - np.asarray(before_state.target_centroid_px, dtype=float)
            ))
        prediction_error = (float(predicted_improvement) - actual_improvement
                            if predicted_improvement is not None and actual_improvement is not None
                            else None)
        segmentation_before = initial.get("segmentation") if initial else None
        segmentation_after = final.get("segmentation") if final else None
        calibration = initial.get("calibration") if initial else None
        resolution_before = _raw_resolution(base_observer)
        record = {
            "suite": SUITE,
            "task_id": TASK_ID,
            "task_name": environment.task_name,
            "task_instruction": environment.task_description,
            "target_phrase": TARGET_PHRASE,
            "seed": 0,
            "init_state_index": init_state_index,
            "pre_settle_hold_ticks": PRE_SETTLE_TICKS,
            "pre_settle_cycles": pre_settle,
            "source_resolution_before": {
                "agentview": ([int(initial["image"].shape[1]), int(initial["image"].shape[0])]
                              if initial else None),
                "wrist": ([resolution_before["wrist"]["width"], resolution_before["wrist"]["height"]]
                          if resolution_before["wrist"] else None),
            },
            "sam3_input_resolution": ([int(initial["image"].shape[1]), int(initial["image"].shape[0])]
                                      if initial else None),
            "sam3_confidence_threshold": observer.confidence_threshold,
            "sam3_detection_count_before": ((segmentation_before.response.get("details", {}).get("detection_count"))
                                             if segmentation_before else None),
            "sam3_detection_count_after": ((segmentation_after.response.get("details", {}).get("detection_count"))
                                            if segmentation_after else None),
            "sam3_reported_source_resolution": (
                initial["segmentation"].response.get("details", {}).get("metadata", {}).get("image_size")
                if initial else None
            ),
            "sam3_response_error_before": (
                segmentation_before.response.get("error")
                if segmentation_before is not None and not segmentation_before.visible else None
            ),
            "sam3_response_error_after": (
                segmentation_after.response.get("error")
                if segmentation_after is not None and not segmentation_after.visible else None
            ),
            "camera_calibration": ({
                "camera": calibration.name,
                "width": calibration.width,
                "height": calibration.height,
                "fovy_deg": calibration.fovy_deg,
                "position_world": calibration.position_world.tolist(),
                "camera_to_world": calibration.camera_to_world.tolist(),
                "rotation_degrees": calibration.rotation_degrees,
                "flip": calibration.flip,
            } if calibration is not None else None),
            "target_visible_before": bool(segmentation_before and segmentation_before.visible),
            "sam3_quality_score_before": (segmentation_before.quality_score if segmentation_before else None),
            "target_centroid_before_px": before_state.target_centroid_px if before_state else None,
            "eef_projection_before_px": before_state.eef_projection_px if before_state else None,
            "pixel_error_before": error_before,
            "runtime_option": event.get("approved_action").option_id if event.get("approved_action") else None,
            "decision_owner": "runtime",
            "decision_reason": (state_before.relevant_geometry.get("object_relative_decision_reason")
                                if state_before else (initial["resolution"].get("reason")
                                                      if initial else None)),
            "candidate_directions": (state_before.relevant_geometry.get("candidate_directions", [])
                                     if state_before else (initial["resolution"].get("candidate_directions", [])
                                                           if initial else [])),
            "chosen_direction": chosen_spec.direction if chosen_spec else None,
            "chosen_physical_vector_xyz": chosen_spec.direction_unit if chosen_spec else None,
            "predicted_pixel_error_after": predicted_after,
            "predicted_improvement_px": predicted_improvement,
            "authorization_calls": arbiter.authorization_calls,
            "arbiter_approved_alignment_actions": arbiter.approval_count,
            "runner_actions": int(result.get("actions", 0)),
            "runner_status": result.get("status"),
            "execution": execution,
            "target_visible_after": bool(segmentation_after and segmentation_after.visible),
            "sam3_quality_score_after": (segmentation_after.quality_score if segmentation_after else None),
            "target_centroid_after_px": after_state.target_centroid_px if after_state else None,
            "target_centroid_shift_px": centroid_shift,
            "eef_projection_after_px": after_state.eef_projection_px if after_state else None,
            "pixel_error_after": error_after,
            "actual_improvement_px": actual_improvement,
            "prediction_error_predicted_minus_actual_px": prediction_error,
            "alignment_improved": (bool(float(error_after) < float(error_before))
                                   if error_before is not None and error_after is not None else None),
            "artifacts": {"before": before_artifacts, "after": after_artifacts},
            "source_changed_by_resize": False,
        }
        _write_json(trial_dir / "trial.json", record)
        qwen_input = None
        if init_state_index == INIT_STATES[0] and initial is not None and before_state is not None:
            qwen_input = {"image": initial["image"], "object_relative_state": before_state,
                          "task_instruction": environment.task_description}
        return record, qwen_input
    finally:
        environment.close()


def _qwen_visual_smoke(
    *, args: argparse.Namespace, config: dict[str, Any], qwen_input: dict[str, Any], run_dir: Path,
) -> dict[str, Any]:
    load_secrets_env()
    client = _make_vlm_client(args, config)
    client.health_check(wait_s=0.0)
    selector = QwenSelectorAdapter(client, qwen_input["task_instruction"], max_tokens=96)
    state = BeliefState(
        task_id=f"{SUITE}:{TASK_ID}",
        target_identity=TARGET_PHRASE,
        object_relative_state=qwen_input["object_relative_state"],
    )
    options = [
        {"option_id": "OPTION_A", "description": "Align to the visible target."},
        {"option_id": "OPTION_B", "description": "Reobserve the target."},
        {"option_id": "OPTION_C", "description": "Abort this attempt."},
    ]
    selection = selector.select_visual_semantic(state, qwen_input["image"], options)
    record = {
        "mode": "NO_ACTION_VISUAL_SEMANTIC_SELECTOR_SMOKE",
        "model": selector.model,
        "task_instruction": qwen_input["task_instruction"],
        "target_phrase": TARGET_PHRASE,
        "source_width": int(qwen_input["image"].shape[1]),
        "source_height": int(qwen_input["image"].shape[0]),
        "model_input_width": selector.last_record.get("model_input_width"),
        "model_input_height": selector.last_record.get("model_input_height"),
        "semantic_options": options,
        "selection": selection,
        "qwen_record": selector.last_record,
        "robot_actions_executed": 0,
    }
    _write_json(run_dir / "qwen_visual_no_action.json", record)
    close = getattr(client, "close", None)
    if callable(close):
        close()
    return record


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    visible_before = [row for row in rows if row.get("pixel_error_before") is not None]
    visible_after = [row for row in rows if row.get("pixel_error_after") is not None]
    valid = [row for row in rows if row.get("pixel_error_before") is not None
             and row.get("pixel_error_after") is not None]
    improved = [row for row in valid if row.get("alignment_improved")]

    def mean(group: list[dict[str, Any]], key: str):
        values = [float(row[key]) for row in group if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    return {
        "trials": len(rows),
        "real_bounded_executions": sum(int(row.get("runner_actions", 0)) for row in rows),
        "max_bounded_motions_per_trial": 1,
        "visible_before_count": sum(bool(row.get("target_visible_before")) for row in rows),
        "alignment_evaluable_trials": len(valid),
        "alignment_improved_count": len(improved),
        "alignment_improvement_rate": len(improved) / len(rows) if rows else None,
        "alignment_improvement_rate_among_evaluable": len(improved) / len(valid) if valid else None,
        "mean_error_before_px": mean(visible_before, "pixel_error_before"),
        "mean_error_before_px_among_evaluable": mean(valid, "pixel_error_before"),
        "mean_error_after_px": mean(visible_after, "pixel_error_after"),
        "mean_actual_improvement_px": mean(valid, "actual_improvement_px"),
        "mean_predicted_improvement_px": mean(rows, "predicted_improvement_px"),
        "mean_prediction_error_predicted_minus_actual_px": mean(valid, "prediction_error_predicted_minus_actual_px"),
    }


def main() -> int:
    args = _parser().parse_args()
    config = load_yaml(args.config)
    _configure_local_sam3_proxy_bypass(args.sam3_url)
    if config.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(config["libero_dir"])
    run_dir = _new_run_dir(args.output_dir)
    workspace = (0.02, 0.60)
    sam3 = Sam3Client(url=args.sam3_url, python=args.sam3_python,
                      timeout_s=args.sam3_timeout_s, max_attempts=1)
    rows: list[dict[str, Any]] = []
    qwen_input = None
    blockers: list[str] = []
    try:
        for init_state_index in INIT_STATES:
            try:
                record, candidate_qwen_input = _run_trial(
                    init_state_index=init_state_index,
                    run_dir=run_dir,
                    config=config,
                    sam3=sam3,
                    workspace=workspace,
                )
                rows.append(record)
                if not record.get("target_visible_before"):
                    blockers.append(
                        f"init_state_{init_state_index}: SAM3 returned no usable target mask; no alignment was authorized"
                    )
                elif not record.get("target_visible_after"):
                    blockers.append(
                        f"init_state_{init_state_index}: post-motion SAM3 returned no usable target mask"
                    )
                elif record.get("runner_actions") != 1 or record.get("arbiter_approved_alignment_actions") != 1:
                    blockers.append(
                        f"init_state_{init_state_index}: expected exactly one authorized alignment execution"
                    )
                if candidate_qwen_input is not None:
                    qwen_input = candidate_qwen_input
            except Exception as exc:
                blockers.append(f"init_state_{init_state_index}: {type(exc).__name__}: {exc}")
                _write_json(run_dir / f"init_state_{init_state_index}" / "failure.json", {
                    "init_state_index": init_state_index,
                    "error": f"{type(exc).__name__}: {exc}",
                })
    finally:
        sam3.close()

    qwen_record = None
    if qwen_input is not None:
        try:
            qwen_record = _qwen_visual_smoke(
                args=args, config=config, qwen_input=qwen_input, run_dir=run_dir,
            )
        except Exception as exc:
            blockers.append(f"qwen_visual_smoke: {type(exc).__name__}: {exc}")
            qwen_record = {"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}",
                           "robot_actions_executed": 0}
            _write_json(run_dir / "qwen_visual_no_action.json", qwen_record)
    summary = _summary(rows)
    summary_record = {
        "status": ("COMPLETED" if len(rows) == len(INIT_STATES) and not blockers
                   else "BLOCKED" if blockers else "PARTIAL"),
        "branch": "runtime-v3",
        "baseline_commit": "d0581189ad8c12d82bf0bf15171e6cf4c006c437",
        "suite": SUITE,
        "task_id": TASK_ID,
        "target_phrase": TARGET_PHRASE,
        "camera_render_width": CAMERA_WIDTH,
        "camera_render_height": CAMERA_HEIGHT,
        "resolution_policy": "source RGB rendered at 512x512; no resize or upsample in SAM3/Qwen client path",
        "sam3_interface": "existing OpenETA SAM3 MCP via core.capabilities.sam3_client.Sam3Client",
        "sam3_checkpoint_path": os.environ.get("OPENETA_SAM3_CHECKPOINT_PATH",
                                               "/root/autodl-tmp/openeta-services/models/sam3/sam3.pt"),
        "rows": rows,
        "summary": summary,
        "qwen_visual_no_action": qwen_record,
        "blockers": blockers,
        "legacy_config_modified": False,
        "robot_actions_from_qwen": 0,
    }
    _write_json(run_dir / "summary.json", summary_record)
    print(json.dumps(_jsonable(summary_record), indent=2, ensure_ascii=False))
    return 0 if len(rows) == len(INIT_STATES) and qwen_record is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())

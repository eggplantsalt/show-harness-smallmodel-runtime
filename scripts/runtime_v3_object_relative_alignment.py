#!/usr/bin/env python3
"""Run three one-motion, same-target-verified Runtime V3 alignment trials."""

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
from core.config import load_yaml
from core.runtime_v3.adapters.libero_env import LiberoEnvironmentAdapter
from core.runtime_v3.adapters.libero_observation import LiberoObservationAdapter
from core.runtime_v3.arbiter import Arbiter, DecisionKind
from core.runtime_v3.effects import EffectObserver
from core.runtime_v3.executor import Executor, LiberoPrimitiveBackend
from core.runtime_v3.object_relative import (
    alignment_verification_metrics,
    ObjectRelativeAlignmentOptionGenerator,
    ObjectRelativePerceptionObserver,
)
from core.runtime_v3.runner import RuntimeV3Runner
from core.runtime_v3.selector import DeterministicSelector
from core.runtime_v3.state import StateBuilder
from core.runtime_v3.temporal_calibration import run_v3_tick
from interpreters.libero_atomic_controller import LiberoAtomicController


SUITE = "LIBERO_OBJECT"
TASK_ID = 2
TARGET_PHRASE = "salad dressing"
INIT_STATES = (0, 1, 2)
PRE_SETTLE_TICKS = 4
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
    parser.add_argument("--camera-resolution", type=int, default=512)
    return parser


def _new_run_dir(base: str | Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(base).expanduser() / f"run_{stamp}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(path: Path, record: Any) -> None:
    path.write_text(json.dumps(_jsonable(record), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


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
    camera_resolution: int,
) -> dict[str, Any]:
    trial_dir = run_dir / f"init_state_{init_state_index}"
    trial_dir.mkdir(parents=True, exist_ok=False)
    environment = LiberoEnvironmentAdapter.create(
        suite_name=SUITE,
        task_id=TASK_ID,
        init_state_index=init_state_index,
        seed=0,
        camera_height=camera_resolution,
        camera_width=camera_resolution,
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
        event = events[0] if events else {}
        history = observer.perception_history
        initial = history[0] if history else None
        final = history[-1] if len(history) > 1 else None
        if initial is not None:
            before_artifacts = observer.save_visual_artifacts(
                str(trial_dir / "artifacts"), image=initial["image"], prefix="before",
                segmentation=initial["segmentation"], resolution=initial["resolution"],
                selected_direction=(event.get("approved_action").primitive.micro_motion_spec.direction
                                    if event.get("approved_action") else None),
            )
        else:
            before_artifacts = {}
        if final is not None:
            after_artifacts = observer.save_visual_artifacts(
                str(trial_dir / "artifacts"), image=final["image"], prefix="after",
                segmentation=final["segmentation"], resolution=final["resolution"],
                selected_direction=(event.get("approved_action").primitive.micro_motion_spec.direction
                                    if event.get("approved_action") else None),
            )
        else:
            after_artifacts = {}
        state_before = event.get("state_before")
        state_after = event.get("state_after")
        execution_record = event.get("execution")
        execution = getattr(execution_record, "result", None)
        before_state = initial.get("object_relative_state") if initial else None
        after_state = final.get("object_relative_state") if final else None
        error_before = (before_state.image_error_norm_px
                        if before_state is not None and before_state.target_identity_status == "ANCHORED"
                        else None)
        post_identity_status = (final["segmentation"].identity_status if final else "TARGET_IDENTITY_LOST")
        raw_error_after = (after_state.image_error_norm_px if after_state is not None else None)
        verification = alignment_verification_metrics(
            error_before, raw_error_after,
            identity_status=post_identity_status,
        )
        error_after = verification["error_after_px"]
        expected_effect = event.get("approved_action").expected_effect if event.get("approved_action") else {}
        chosen_spec = (event.get("approved_action").primitive.micro_motion_spec
                       if event.get("approved_action") else None)
        predicted_after = expected_effect.get("predicted_image_error_after_px")
        predicted_improvement = expected_effect.get("predicted_improvement_px")
        actual_improvement = verification["actual_improvement_px"]
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
        association_metrics = segmentation_after.association_metrics if segmentation_after else None
        selected_association = None
        if isinstance(association_metrics, dict):
            selected_id = association_metrics.get("selected_candidate_id")
            selected_association = next((item for item in association_metrics.get("candidates", [])
                                         if item.get("candidate_id") == selected_id), None)
        predicted_eef_shift = None
        observed_eef_shift = None
        geometry_consistent = None
        if before_state is not None and before_state.eef_projection_px is not None and chosen_spec is not None:
            chosen_candidate = next((item for item in (state_before.relevant_geometry.get("candidate_directions", [])
                                                        if state_before else [])
                                     if item.get("direction") == chosen_spec.direction and item.get("valid")), None)
            if chosen_candidate is not None:
                predicted_eef_shift = (np.asarray(chosen_candidate["hypothetical_projection_px"], dtype=float)
                                       - np.asarray(before_state.eef_projection_px, dtype=float)).tolist()
                if after_state is not None and after_state.eef_projection_px is not None:
                    observed_eef_shift = (np.asarray(after_state.eef_projection_px, dtype=float)
                                          - np.asarray(before_state.eef_projection_px, dtype=float)).tolist()
                    geometry_consistent = bool(float(np.dot(predicted_eef_shift, observed_eef_shift)) > 0.0)
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
            "target_identity_status_before": (segmentation_before.identity_status if segmentation_before else None),
            "target_identity_anchor": ({
                "target_phrase": observer.identity_anchor.target_phrase,
                "candidate_id": observer.identity_anchor.candidate_id,
                "frame_id": observer.identity_anchor.frame_id,
                "centroid_px": observer.identity_anchor.centroid_px,
                "bbox_xyxy": observer.identity_anchor.bbox_xyxy,
                "mask_area": observer.identity_anchor.mask_area,
            } if observer.identity_anchor is not None else None),
            "sam3_candidates_before": ([{
                "candidate_id": item.candidate_id, "rank": item.rank,
                "backend_index": item.backend_index, "score": item.score,
                "area_px": item.area_px, "centroid_px": item.centroid_px,
                "bbox_xyxy": item.bbox_xyxy,
            } for item in segmentation_before.candidates] if segmentation_before else []),
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
            "target_identity_status_after": post_identity_status,
            "same_target_identity": post_identity_status == "SAME_TARGET",
            "identity_retained": post_identity_status == "SAME_TARGET",
            "target_association_metrics": association_metrics,
            "selected_association_metrics": selected_association,
            "sam3_candidates_after": ([{
                "candidate_id": item.candidate_id, "rank": item.rank,
                "backend_index": item.backend_index, "score": item.score,
                "area_px": item.area_px, "centroid_px": item.centroid_px,
                "bbox_xyxy": item.bbox_xyxy,
            } for item in segmentation_after.candidates] if segmentation_after else []),
            "sam3_quality_score_after": (segmentation_after.quality_score if segmentation_after else None),
            "target_centroid_after_px": after_state.target_centroid_px if after_state else None,
            "target_centroid_shift_px": centroid_shift,
            "eef_projection_after_px": after_state.eef_projection_px if after_state else None,
            "pixel_error_after": error_after,
            "actual_improvement_px": actual_improvement,
            "prediction_error_predicted_minus_actual_px": prediction_error,
            **verification,
            "predicted_eef_pixel_shift": predicted_eef_shift,
            "observed_eef_pixel_shift": observed_eef_shift,
            "geometry_direction_consistent": geometry_consistent,
            "artifacts": {"before": before_artifacts, "after": after_artifacts},
            "source_changed_by_resize": False,
        }
        _write_json(trial_dir / "trial.json", record)
        return record
    finally:
        environment.close()


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("same_target_identity")
             and row.get("pixel_error_before") is not None and row.get("pixel_error_after") is not None]
    improved = [row for row in valid if row.get("alignment_improved")]

    def mean(group: list[dict[str, Any]], key: str):
        values = [float(row[key]) for row in group if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    return {
        "trials": len(rows),
        "real_bounded_executions": sum(int(row.get("runner_actions", 0)) for row in rows),
        "max_bounded_motions_per_trial": 1,
        "visible_before_count": sum(bool(row.get("target_visible_before")) for row in rows),
        "identity_retained_count": sum(bool(row.get("identity_retained")) for row in rows),
        "identity_retention_rate": (sum(bool(row.get("identity_retained")) for row in rows) / len(rows)
                                     if rows else None),
        "alignment_evaluable_trials": len(valid),
        "alignment_improved_count": len(improved),
        "alignment_improvement_rate_among_evaluable": len(improved) / len(valid) if valid else None,
        "conditional_alignment_improvement_rate": len(improved) / len(valid) if valid else None,
        "mean_error_before_px_among_evaluable": mean(valid, "pixel_error_before"),
        "mean_error_after_px_among_evaluable": mean(valid, "pixel_error_after"),
        "mean_actual_improvement_px": mean(valid, "actual_improvement_px"),
        "mean_predicted_improvement_px": mean(rows, "predicted_improvement_px"),
        "mean_prediction_error_predicted_minus_actual_px": mean(valid, "prediction_error_predicted_minus_actual_px"),
        "geometry_direction_consistency_count": sum(row.get("geometry_direction_consistent") is True for row in rows),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.camera_resolution not in {512, 768}:
        raise SystemExit("--camera-resolution must match an audited source renderer size: 512 or 768")
    config = load_yaml(args.config)
    _configure_local_sam3_proxy_bypass(args.sam3_url)
    if config.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(config["libero_dir"])
    run_dir = _new_run_dir(args.output_dir)
    workspace = (0.02, 0.60)
    sam3 = Sam3Client(url=args.sam3_url, python=args.sam3_python,
                      timeout_s=args.sam3_timeout_s, max_attempts=1)
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    try:
        for init_state_index in INIT_STATES:
            try:
                record = _run_trial(
                    init_state_index=init_state_index,
                    run_dir=run_dir,
                    config=config,
                    sam3=sam3,
                    workspace=workspace,
                    camera_resolution=args.camera_resolution,
                )
                rows.append(record)
                if not record.get("target_visible_before"):
                    blockers.append(
                        f"init_state_{init_state_index}: SAM3 returned no usable target mask; no alignment was authorized"
                    )
                if record.get("runner_actions") != 1 or record.get("arbiter_approved_alignment_actions") != 1:
                    blockers.append(
                        f"init_state_{init_state_index}: expected exactly one authorized alignment execution"
                    )
            except Exception as exc:
                blockers.append(f"init_state_{init_state_index}: {type(exc).__name__}: {exc}")
                _write_json(run_dir / f"init_state_{init_state_index}" / "failure.json", {
                    "init_state_index": init_state_index,
                    "error": f"{type(exc).__name__}: {exc}",
                })
    finally:
        sam3.close()

    summary = _summary(rows)
    summary_record = {
        "status": ("COMPLETED" if len(rows) == len(INIT_STATES) and not blockers
                   else "BLOCKED" if blockers else "PARTIAL"),
        "branch": "runtime-v3",
        "baseline_commit": "d8b52dbb5ddaa0f4417aa6a0ee0de4b97ab80ba1",
        "suite": SUITE,
        "task_id": TASK_ID,
        "target_phrase": TARGET_PHRASE,
        "camera_render_width": args.camera_resolution,
        "camera_render_height": args.camera_resolution,
        "canonical_transform": "vertical_flip",
        "resolution_policy": f"source RGB rendered directly at {args.camera_resolution}x{args.camera_resolution}; no resize or upsample",
        "sam3_interface": "existing OpenETA SAM3 MCP via core.capabilities.sam3_client.Sam3Client",
        "sam3_checkpoint_path": os.environ.get("OPENETA_SAM3_CHECKPOINT_PATH",
                                               "/root/autodl-tmp/openeta-services/models/sam3/sam3.pt"),
        "rows": rows,
        "summary": summary,
        "blockers": blockers,
        "legacy_config_modified": False,
        "robot_actions_from_qwen": 0,
        "oracle_diagnostic_only": False,
        "oracle_used_by_runtime": False,
    }
    _write_json(run_dir / "summary.json", summary_record)
    print(json.dumps(_jsonable(summary_record), indent=2, ensure_ascii=False))
    return 0 if len(rows) == len(INIT_STATES) and not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())

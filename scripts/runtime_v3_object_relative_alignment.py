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
from PIL import Image, ImageDraw

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
    compare_alignment_improvements,
    frozen_reference_error,
    ObjectRelativeAlignmentOptionGenerator,
    ObjectRelativePerceptionObserver,
)
from core.runtime_v3.runner import RuntimeV3Runner
from core.runtime_v3.scene_initialization import run_scene_ready_holds
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
    def __init__(self, *, before_authorize=None) -> None:
        super().__init__()
        self.authorization_calls = 0
        self.approval_count = 0
        self.before_authorize = before_authorize
        self.pre_action_ready_written = False

    def authorize(self, state, options, selection):
        self.authorization_calls += 1
        if selection.option_id == "ALIGN_TO_TARGET_SMALL":
            if self.before_authorize is None:
                raise RuntimeError("PRE_ACTION_READY writer is required before alignment authorization")
            self.before_authorize(state, options, selection)
            self.pre_action_ready_written = True
        decision = super().authorize(state, options, selection)
        if (decision.kind == DecisionKind.APPROVED and decision.action is not None
                and decision.action.option_id == "ALIGN_TO_TARGET_SMALL"
                and not self.pre_action_ready_written):
            raise RuntimeError("alignment approval attempted without PRE_ACTION_READY")
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
    payload = json.dumps(_jsonable(record), indent=2, ensure_ascii=False) + "\n"
    with path.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _ensure_output_writable(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / f".write_probe_{uuid.uuid4().hex}"
    with probe.open("x", encoding="utf-8") as stream:
        stream.write("logger output check\n")
        stream.flush()
        os.fsync(stream.fileno())
    probe.unlink()


def _oracle_target_world_position(environment: Any) -> dict[str, Any]:
    """Read simulator target pose for a diagnostic record only."""
    target_body = "salad_dressing_1_main"
    try:
        sim = environment.env.sim
        model, data = sim.model, sim.data
        if callable(getattr(model, "body_name2id", None)):
            body_id = int(model.body_name2id(target_body))
        elif callable(getattr(model, "body", None)):
            body_id = int(model.body(target_body).id)
        else:
            raise AttributeError("MuJoCo model has no body-name lookup")
        position = np.asarray(data.xpos[body_id], dtype=float).reshape(3)
        if not np.all(np.isfinite(position)):
            raise ValueError("body position is not finite")
        return {"available": True, "body": target_body, "body_id": body_id,
                "world_position_m": position.tolist(), "source": "simulator_body_pose_diagnostic_only"}
    except Exception as exc:
        return {"available": False, "body": target_body,
                "error": f"{type(exc).__name__}: {exc}",
                "source": "simulator_body_pose_diagnostic_only"}


def _save_reference_comparison_overlay(
    *, observer: ObjectRelativePerceptionObserver, before: dict[str, Any] | None,
    after: dict[str, Any] | None, predicted_projection_px: Any, output_path: Path,
) -> str | None:
    if before is None or after is None:
        return None
    image = Image.fromarray(np.ascontiguousarray(after["image"], dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)

    def point(value: Any) -> tuple[float, float] | None:
        try:
            coords = np.asarray(value, dtype=float).reshape(2)
        except (TypeError, ValueError):
            return None
        return (float(coords[0]), float(coords[1])) if np.all(np.isfinite(coords)) else None

    reference = point(after["object_relative_state"].target_reference_point_px)
    eef_before = point(before["object_relative_state"].eef_projection_px)
    eef_after = point(after["object_relative_state"].eef_projection_px)
    predicted = point(predicted_projection_px)
    sam_centroid = point(after["object_relative_state"].target_centroid_px)

    def marker(position: tuple[float, float] | None, color: tuple[int, int, int], shape: str):
        if position is None:
            return
        x, y = position
        r = 9
        if shape == "diamond":
            draw.polygon(((x, y-r), (x+r, y), (x, y+r), (x-r, y)), outline=color, fill=(20, 20, 20))
        elif shape == "triangle":
            draw.polygon(((x, y-r), (x+r, y+r), (x-r, y+r)), outline=color, fill=(20, 20, 20))
        elif shape == "cross":
            draw.line((x-r, y-r, x+r, y+r), fill=color, width=3)
            draw.line((x-r, y+r, x+r, y-r), fill=color, width=3)
        else:
            draw.ellipse((x-r, y-r, x+r, y+r), outline=color, width=3)

    entries = [
        ("FIXED TARGET REFERENCE", (255, 215, 0), reference, "diamond"),
        ("EEF BEFORE", (0, 190, 255), eef_before, "circle"),
        ("EEF AFTER", (70, 255, 255), eef_after, "cross"),
        ("PREDICTED EEF", (255, 0, 255), predicted, "triangle"),
        ("SAM CENTROID: DIAGNOSTIC ONLY", (255, 60, 60), sam_centroid, "circle"),
    ]
    for label, color, position, shape in entries:
        marker(position, color, shape)
    legend_draw = ImageDraw.Draw(image)
    for index, (label, color, _position, shape) in enumerate(entries):
        y = 8 + index * 19
        legend_draw.rectangle((6, y - 2, 205, y + 14), fill=(0, 0, 0), outline=(70, 70, 70))
        if shape == "diamond":
            legend_draw.polygon(((14, y+2), (19, y+7), (14, y+12), (9, y+7)), outline=color)
        elif shape == "triangle":
            legend_draw.polygon(((14, y+1), (19, y+12), (9, y+12)), outline=color)
        elif shape == "cross":
            legend_draw.line((9, y+2, 19, y+12), fill=color, width=2)
            legend_draw.line((9, y+12, 19, y+2), fill=color, width=2)
        else:
            legend_draw.ellipse((9, y+2, 19, y+12), outline=color, width=2)
        legend_draw.text((24, y), label, fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)


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
    artifacts_dir = trial_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=False)
    _ensure_output_writable(trial_dir)
    _ensure_output_writable(artifacts_dir)
    _write_json(trial_dir / "TRIAL_SETUP_READY.json", {
        "status": "TRIAL_SETUP_READY",
        "init_state_index": init_state_index,
        "trial_directory_writable": True,
        "artifact_directory_writable": True,
        "pre_action_ready_required_before_arbiter": True,
    })
    pre_settle: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    before_artifacts: dict[str, str] = {}
    after_artifacts: dict[str, str] = {}
    comparison_overlay: str | None = None
    oracle_before: dict[str, Any] | None = None
    oracle_after: dict[str, Any] | None = None
    pre_action_ready_record: dict[str, Any] | None = None
    formal_initial_frame: dict[str, Any] | None = None
    scene_ready_result: dict[str, Any] | None = None
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
            scene_ready_required=True,
        )
        scene_ready_result = run_scene_ready_holds(
            environment, observer, controller, task_id=f"{SUITE}:{TASK_ID}",
            commanded_step_m=CONTROL_TICK_MM / 1000.0,
            workspace_z_bounds_m=workspace, max_hold_ticks=40,
        )
        if not scene_ready_result.get("ready") or not observer.scene_ready:
            raise RuntimeError(f"visual SceneReady was not established: {scene_ready_result}")

        def write_pre_action_ready(state, options, selection) -> None:
            nonlocal oracle_before, pre_action_ready_record, formal_initial_frame
            _ensure_output_writable(trial_dir)
            _ensure_output_writable(artifacts_dir)
            if (state.object_relative_state is None
                    or not state.object_relative_state.target_reference_valid
                    or state.object_relative_state.target_reference_point_px is None):
                raise RuntimeError("no valid frozen visual reference; alignment authorization blocked")
            selected = next((option for option in options
                             if option.option_id == selection.option_id), None)
            if selected is None or selected.primitive.micro_motion_spec is None:
                raise RuntimeError("selected alignment option has no bounded micro-motion spec")
            if not observer.perception_history:
                raise RuntimeError("pre-action perception artifacts are unavailable")
            initial_frame = observer.perception_history[-1]
            formal_initial_frame = initial_frame
            selected_direction = selected.primitive.micro_motion_spec.direction
            before_artifacts.update(observer.save_visual_artifacts(
                str(artifacts_dir), image=initial_frame["image"], prefix="before",
                segmentation=initial_frame["segmentation"], resolution=initial_frame["resolution"],
                selected_direction=selected_direction,
            ))
            oracle_before = _oracle_target_world_position(environment)
            chosen = next((candidate for candidate in state.relevant_geometry.get("candidate_directions", [])
                           if candidate.get("direction") == selected_direction and candidate.get("valid")), None)
            pre_action_ready_record = {
                "status": "PRE_ACTION_READY",
                "init_state_index": init_state_index,
                "task": f"{SUITE}:{TASK_ID}",
                "frame_id": state.frame_id,
                "selected_option": selection.option_id,
                "direction": selected_direction,
                "target_reference_px": state.object_relative_state.target_reference_point_px,
                "target_reference_valid": state.object_relative_state.target_reference_valid,
                "predicted_eef_projection_px": (chosen.get("hypothetical_projection_px")
                                                 if chosen else None),
                "before_artifacts": dict(before_artifacts),
                "scene_ready": observer.scene_ready,
                "scene_ready_evidence": (observer.scene_ready_evidence.to_record()
                                          if observer.scene_ready_evidence else None),
                "logger_writable": True,
                "artifact_variables_initialized": True,
                "oracle_target_pose_diagnostic_only": oracle_before,
                "oracle_used_by_runtime": False,
            }
            marker_path = trial_dir / "PRE_ACTION_READY.json"
            _write_json(marker_path, pre_action_ready_record)
            written = json.loads(marker_path.read_text(encoding="utf-8"))
            if written.get("status") != "PRE_ACTION_READY":
                raise RuntimeError("PRE_ACTION_READY record failed read-back verification")

        arbiter = CountingArbiter(before_authorize=write_pre_action_ready)
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
        oracle_after = _oracle_target_world_position(environment)
        event = events[0] if events else {}
        history = observer.perception_history
        initial = formal_initial_frame or (history[0] if history else None)
        final = history[-1] if len(history) > 1 else None
        if initial is not None and not before_artifacts:
            before_artifacts = observer.save_visual_artifacts(
                str(artifacts_dir), image=initial["image"], prefix="before",
                segmentation=initial["segmentation"], resolution=initial["resolution"],
                selected_direction=(event.get("approved_action").primitive.micro_motion_spec.direction
                                    if event.get("approved_action") else None),
            )
        if final is not None:
            after_artifacts = observer.save_visual_artifacts(
                str(artifacts_dir), image=final["image"], prefix="after",
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
        reference_point = (before_state.target_reference_point_px if before_state is not None else None)
        reference_valid_before = bool(before_state and before_state.target_reference_valid)
        reference_valid_after = bool(after_state and after_state.target_reference_valid)
        error_before = frozen_reference_error(
            reference_point,
            before_state.eef_projection_px if before_state is not None else None,
            reference_valid=reference_valid_before,
        )
        error_after = frozen_reference_error(
            reference_point,
            after_state.eef_projection_px if after_state is not None else None,
            reference_valid=(reference_valid_before and reference_valid_after),
        )
        post_identity_status = (final["segmentation"].identity_status if final else "TARGET_IDENTITY_LOST")
        frozen_actual_improvement = (error_before - error_after
                                     if error_before is not None and error_after is not None else None)
        frozen_verification = {
            "verification_status": "FROZEN_REFERENCE_VALID" if error_after is not None
            else "FROZEN_REFERENCE_INVALID",
            "error_after_px": error_after,
            "actual_improvement_px": frozen_actual_improvement,
            "alignment_improved": (error_after < error_before
                                   if error_before is not None and error_after is not None else None),
        }
        dynamic_error_before = frozen_reference_error(
            before_state.target_centroid_px if before_state is not None else None,
            before_state.eef_projection_px if before_state is not None else None,
        )
        dynamic_error_after = frozen_reference_error(
            after_state.target_centroid_px if after_state is not None else None,
            after_state.eef_projection_px if after_state is not None else None,
        )
        dynamic_verification = alignment_verification_metrics(
            dynamic_error_before, dynamic_error_after,
            identity_status=post_identity_status,
        )
        expected_effect = event.get("approved_action").expected_effect if event.get("approved_action") else {}
        chosen_spec = (event.get("approved_action").primitive.micro_motion_spec
                       if event.get("approved_action") else None)
        predicted_after = expected_effect.get("predicted_image_error_after_px")
        predicted_improvement = expected_effect.get("predicted_improvement_px")
        actual_improvement = frozen_actual_improvement
        centroid_shift = None
        if (before_state is not None and after_state is not None
                and before_state.target_centroid_px is not None
                and after_state.target_centroid_px is not None):
            centroid_shift = float(np.linalg.norm(
                np.asarray(after_state.target_centroid_px, dtype=float)
                - np.asarray(before_state.target_centroid_px, dtype=float)
            ))
        prediction_comparison = compare_alignment_improvements(
            predicted_improvement_px=predicted_improvement,
            actual_improvement_px=actual_improvement,
        )
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
        predicted_eef_projection = None
        geometry_consistent = None
        if before_state is not None and before_state.eef_projection_px is not None and chosen_spec is not None:
            chosen_candidate = next((item for item in (state_before.relevant_geometry.get("candidate_directions", [])
                                                        if state_before else [])
                                     if item.get("direction") == chosen_spec.direction and item.get("valid")), None)
            if chosen_candidate is not None:
                predicted_eef_projection = list(chosen_candidate["hypothetical_projection_px"])
                predicted_eef_shift = (np.asarray(predicted_eef_projection, dtype=float)
                                       - np.asarray(before_state.eef_projection_px, dtype=float)).tolist()
                if after_state is not None and after_state.eef_projection_px is not None:
                    observed_eef_shift = (np.asarray(after_state.eef_projection_px, dtype=float)
                                          - np.asarray(before_state.eef_projection_px, dtype=float)).tolist()
                    geometry_consistent = bool(float(np.dot(predicted_eef_shift, observed_eef_shift)) > 0.0)
        comparison_overlay = _save_reference_comparison_overlay(
            observer=observer,
            before=initial,
            after=final,
            predicted_projection_px=predicted_eef_projection,
            output_path=artifacts_dir / "reference_comparison.png",
        )
        oracle_displacement_vector = None
        oracle_displacement_norm = None
        if (oracle_before and oracle_before.get("available") and oracle_after
                and oracle_after.get("available")):
            delta = (np.asarray(oracle_after["world_position_m"], dtype=float)
                     - np.asarray(oracle_before["world_position_m"], dtype=float))
            oracle_displacement_vector = delta.tolist()
            oracle_displacement_norm = float(np.linalg.norm(delta))
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
            "scene_ready_initialization": scene_ready_result,
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
            "target_reference_anchor": (_jsonable(initial.get("reference_anchor"))
                                        if initial else None),
            "target_reference_px": reference_point,
            "target_reference_valid_before": reference_valid_before,
            "target_reference_valid_after": reference_valid_after,
            "target_reference_invalidation_reason": (
                after_state.target_reference_invalidation_reason if after_state else None
            ),
            "sam3_candidates_before": ([{
                "candidate_id": item.candidate_id, "rank": item.rank,
                "backend_index": item.backend_index, "score": item.score,
                "area_px": item.area_px, "centroid_px": item.centroid_px,
                "bbox_xyxy": item.bbox_xyxy,
            } for item in segmentation_before.candidates] if segmentation_before else []),
            "sam3_quality_score_before": (segmentation_before.quality_score if segmentation_before else None),
            "target_centroid_before_px": before_state.target_centroid_px if before_state else None,
            "eef_projection_before_px": before_state.eef_projection_px if before_state else None,
            "error_before_frozen_px": error_before,
            "pixel_error_before": error_before,
            "dynamic_sam_error_before_px": dynamic_error_before,
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
            "predicted_eef_projection_after_px": predicted_eef_projection,
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
            "error_after_frozen_px": error_after,
            "pixel_error_after": error_after,
            "actual_improvement_px": actual_improvement,
            "frozen_reference_improvement_px": actual_improvement,
            **frozen_verification,
            "dynamic_sam_error_after_px": dynamic_error_after,
            "dynamic_sam_improvement_px": dynamic_verification["actual_improvement_px"],
            "dynamic_sam_alignment_improved": dynamic_verification["alignment_improved"],
            "dynamic_sam_verification_status": dynamic_verification["verification_status"],
            "prediction_residual_actual_minus_predicted_px": prediction_comparison[
                "prediction_residual_actual_minus_predicted_px"],
            "predicted_eef_pixel_shift": predicted_eef_shift,
            "observed_eef_pixel_shift": observed_eef_shift,
            "geometry_direction_consistent": geometry_consistent,
            "pre_action_ready": bool(pre_action_ready_record),
            "scene_ready_before_anchor": bool(pre_action_ready_record and pre_action_ready_record.get("scene_ready")),
            "scene_ready_evidence": (pre_action_ready_record or {}).get("scene_ready_evidence"),
            "pre_action_ready_record": (str(trial_dir / "PRE_ACTION_READY.json")
                                         if pre_action_ready_record else None),
            "oracle_target_motion_diagnostic": {
                "target_body": "salad_dressing_1_main",
                "before_world_position_m": (oracle_before or {}).get("world_position_m"),
                "after_world_position_m": (oracle_after or {}).get("world_position_m"),
                "displacement_vector_m": oracle_displacement_vector,
                "displacement_norm_m": oracle_displacement_norm,
                "before_available": bool(oracle_before and oracle_before.get("available")),
                "after_available": bool(oracle_after and oracle_after.get("available")),
                "used_by_runtime": False,
                "fed_to_runtime_state": False,
                "fed_to_option_generator": False,
                "fed_to_arbiter": False,
                "fed_to_executor": False,
                "fed_to_qwen": False,
            },
            "camera_unchanged": bool(initial and final
                                     and initial.get("camera_signature") == final.get("camera_signature")),
            "reference_invalidation_signals_before": (
                initial.get("reference_invalidation_signals") if initial else None),
            "reference_invalidation_signals_after": (
                final.get("reference_invalidation_signals") if final else None),
            "artifacts": {"before": before_artifacts, "after": after_artifacts,
                          "reference_comparison": comparison_overlay},
            "source_changed_by_resize": False,
        }
        _write_json(trial_dir / "trial.json", record)
        return record
    finally:
        environment.close()


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frozen_valid = [row for row in rows if row.get("target_reference_valid_before")
                    and row.get("target_reference_valid_after")
                    and row.get("error_before_frozen_px") is not None
                    and row.get("error_after_frozen_px") is not None]
    dynamic_valid = [row for row in rows if row.get("dynamic_sam_improvement_px") is not None]
    improved = [row for row in frozen_valid if row.get("alignment_improved")]
    dynamic_improved = [row for row in dynamic_valid
                        if row.get("dynamic_sam_alignment_improved") is True]

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
        "alignment_evaluable_trials": len(frozen_valid),
        "alignment_improved_count": len(improved),
        "alignment_improvement_rate_among_evaluable": len(improved) / len(frozen_valid) if frozen_valid else None,
        "frozen_reference_success_rate": len(improved) / len(frozen_valid) if frozen_valid else None,
        "mean_error_before_frozen_px": mean(frozen_valid, "error_before_frozen_px"),
        "mean_error_after_frozen_px": mean(frozen_valid, "error_after_frozen_px"),
        "mean_actual_improvement_px": mean(frozen_valid, "frozen_reference_improvement_px"),
        "dynamic_sam_evaluable_trials": len(dynamic_valid),
        "dynamic_sam_success_count": len(dynamic_improved),
        "dynamic_sam_success_rate": len(dynamic_improved) / len(dynamic_valid) if dynamic_valid else None,
        "mean_dynamic_sam_improvement_px": mean(dynamic_valid, "dynamic_sam_improvement_px"),
        "mean_predicted_improvement_px": mean(rows, "predicted_improvement_px"),
        "mean_prediction_residual_actual_minus_predicted_px": mean(
            frozen_valid, "prediction_residual_actual_minus_predicted_px"),
        "geometry_direction_consistency_count": sum(row.get("geometry_direction_consistent") is True for row in rows),
        "pre_action_ready_count": sum(row.get("pre_action_ready") is True for row in rows),
        "oracle_motion_diagnostic_available_count": sum(
            row.get("oracle_target_motion_diagnostic", {}).get("before_available")
            and row.get("oracle_target_motion_diagnostic", {}).get("after_available") for row in rows),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.camera_resolution != 512:
        raise SystemExit("this phase is frozen to direct 512x512 source rendering")
    config = load_yaml(args.config)
    _configure_local_sam3_proxy_bypass(args.sam3_url)
    if config.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(config["libero_dir"])
    run_dir = _new_run_dir(args.output_dir)
    _ensure_output_writable(run_dir)
    _write_json(run_dir / "RUN_SETUP_READY.json", {
        "status": "RUN_SETUP_READY", "branch": "runtime-v3",
        "camera_resolution": args.camera_resolution,
        "logger_writable": True, "oracle_used_by_runtime": False,
    })
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
                failure_dir = run_dir / f"init_state_{init_state_index}"
                failure_dir.mkdir(parents=True, exist_ok=True)
                _write_json(failure_dir / "failure.json", {
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
        "baseline_commit": "030f20fdca1385adb93f281560db1ffa78e8cba3",
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
        "oracle_diagnostic_only": True,
        "oracle_used_by_runtime": False,
    }
    _write_json(run_dir / "summary.json", summary_record)
    print(json.dumps(_jsonable(summary_record), indent=2, ensure_ascii=False))
    return 0 if len(rows) == len(INIT_STATES) and not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())

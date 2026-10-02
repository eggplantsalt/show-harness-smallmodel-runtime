#!/usr/bin/env python3
"""Diagnose LIBERO_OBJECT target settling versus one bounded V3 alignment.

Oracle poses are written to diagnostic artifacts only. The action episode uses
Runtime V3's normal Runner, Arbiter and Executor authority path.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.capabilities.camera_geometry import make_mujoco_calibrations
from core.capabilities.sam3_client import Sam3Client
from core.config import load_yaml
from core.runtime_v3.adapters.libero_env import LiberoEnvironmentAdapter
from core.runtime_v3.adapters.libero_observation import LiberoObservationAdapter
from core.runtime_v3.arbiter import Arbiter
from core.runtime_v3.canonical_image import CanonicalImageAdapter
from core.runtime_v3.effects import EffectObserver
from core.runtime_v3.executor import Executor, LiberoPrimitiveBackend
from core.runtime_v3.object_relative import ObjectRelativeAlignmentOptionGenerator, ObjectRelativePerceptionObserver
from core.runtime_v3.runner import RuntimeV3Runner
from core.runtime_v3.scene_settling import (
    action_excess_motion,
    compare_oracle_and_sam_pixel_motion,
    displacement,
    project_world_motion_to_canonical_pixels,
    require_matched_duration,
    target_motion_curve,
    validate_no_action_commands,
)
from core.runtime_v3.selector import DeterministicSelector
from core.runtime_v3.state import StateBuilder
from core.runtime_v3.temporal_calibration import run_v3_tick
from interpreters.libero_atomic_controller import LiberoAtomicController


SUITE = "LIBERO_OBJECT"
TASK_ID = 2
TARGET_PHRASE = "salad dressing"
TARGET_BODY = "salad_dressing_1_main"
INIT_STATES = (0, 1, 2)
PRE_SETTLE_TICKS = 4
SETTLING_TICKS = (0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20)
CONTROL_TICK_MM = 5.0
CANONICAL = CanonicalImageAdapter()


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _write_json(path: Path, record: Any) -> None:
    payload = json.dumps(_jsonable(record), indent=2, ensure_ascii=False) + "\n"
    with path.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _append_jsonl(path: Path, record: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _ensure_writable(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / f".write_probe_{uuid.uuid4().hex}"
    with probe.open("x", encoding="utf-8") as stream:
        stream.write("logger write check\n")
        stream.flush()
        os.fsync(stream.fileno())
    probe.unlink()


def _configure_proxy_bypass(url: str) -> None:
    if urlparse(str(url)).hostname not in {"127.0.0.1", "localhost", "::1"}:
        return
    entries = [part.strip() for key in ("NO_PROXY", "no_proxy")
               for part in os.environ.get(key, "").split(",") if part.strip()]
    entries.extend(("127.0.0.1", "localhost", "::1"))
    value = ",".join(dict.fromkeys(entries))
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def _name(model: Any, kind: str, index: int) -> str | None:
    lookup = getattr(model, f"{kind}_id2name", None)
    if callable(lookup):
        try:
            return lookup(int(index))
        except Exception:
            return None
    try:
        return str(getattr(model, kind)(int(index)).name)
    except Exception:
        return None


def _target_body_audit(environment: Any) -> dict[str, Any]:
    sim = environment.env.sim
    model, data = sim.model, sim.data
    if not callable(getattr(model, "body_name2id", None)):
        raise RuntimeError("MuJoCo model has no body_name2id lookup")
    body_id = int(model.body_name2id(TARGET_BODY))
    parent_id = int(model.body_parentid[body_id])
    geom_ids = [idx for idx, owner in enumerate(model.geom_bodyid) if int(owner) == body_id]
    site_ids = [idx for idx, owner in enumerate(getattr(model, "site_bodyid", []))
                if int(owner) == body_id]
    joint_ids = []
    first_joint = int(getattr(model, "body_jntadr", np.full(body_id + 1, -1))[body_id])
    joint_count = int(getattr(model, "body_jntnum", np.zeros(body_id + 1))[body_id])
    if first_joint >= 0:
        joint_ids = list(range(first_joint, first_joint + joint_count))
    return {
        "body": TARGET_BODY,
        "body_id": body_id,
        "parent_body_id": parent_id,
        "parent_body": _name(model, "body", parent_id),
        "world_frame_source": "environment.env.sim.data.xpos[body_id]",
        "queried_quantity": "MuJoCo body-frame origin in world coordinates",
        "not_queried": ["site_xpos", "geom_xpos", "visual geometry offset", "task success predicate"],
        "model_body_local_position_m": np.asarray(model.body_pos[body_id], dtype=float).tolist(),
        "joint_ids": joint_ids,
        "joint_names": [_name(model, "joint", idx) for idx in joint_ids],
        "geom_ids": geom_ids,
        "geom_names": [_name(model, "geom", idx) for idx in geom_ids],
        "geom_local_positions_m": [np.asarray(model.geom_pos[idx], dtype=float).tolist()
                                    for idx in geom_ids],
        "site_ids": site_ids,
        "site_names": [_name(model, "site", idx) for idx in site_ids],
        "site_local_positions_m": [np.asarray(model.site_pos[idx], dtype=float).tolist()
                                    for idx in site_ids],
        "task_instruction": environment.task_description,
        "task_instruction_mentions_target": TARGET_PHRASE.casefold() in environment.task_description.casefold(),
        "body_name_matches_target_phrase": "salad_dressing" in TARGET_BODY,
        "body_count": int(model.nbody),
        "initial_world_position_m": np.asarray(data.xpos[body_id], dtype=float).reshape(3).tolist(),
    }


def _oracle_pose(environment: Any, expected_body_id: int) -> dict[str, Any]:
    sim = environment.env.sim
    model, data = sim.model, sim.data
    body_id = int(model.body_name2id(TARGET_BODY))
    if body_id != int(expected_body_id):
        raise RuntimeError(f"target body ID changed: expected {expected_body_id}, received {body_id}")
    position = np.asarray(data.xpos[body_id], dtype=float).reshape(3)
    if not np.all(np.isfinite(position)):
        raise ValueError("target body world position is not finite")
    return {
        "body": TARGET_BODY,
        "body_id": body_id,
        "world_position_m": position.tolist(),
        "source": "environment.env.sim.data.xpos[body_id]",
        "frame": "MuJoCo world frame",
        "diagnostic_only": True,
    }


def _descendant_body_ids(model: Any, root_id: int) -> set[int]:
    parents = np.asarray(model.body_parentid, dtype=int)
    result = {int(root_id)}
    changed = True
    while changed:
        changed = False
        for body_id, parent_id in enumerate(parents):
            if int(parent_id) in result and body_id not in result:
                result.add(body_id)
                changed = True
    return result


def _is_robot_body(model: Any, body_id: int) -> bool:
    current = int(body_id)
    while current > 0:
        body_name = (_name(model, "body", current) or "").casefold()
        if body_name.startswith(("robot", "gripper", "panda", "franka")):
            return True
        current = int(model.body_parentid[current])
    return False


def _contacts(environment: Any, target_body_id: int) -> dict[str, Any]:
    sim = environment.env.sim
    model, data = sim.model, sim.data
    target_ids = _descendant_body_ids(model, target_body_id)
    target_contacts = []
    robot_target = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom_a, geom_b = int(contact.geom1), int(contact.geom2)
        body_a, body_b = int(model.geom_bodyid[geom_a]), int(model.geom_bodyid[geom_b])
        item = {
            "geom_ids": [geom_a, geom_b],
            "geom_names": [_name(model, "geom", geom_a), _name(model, "geom", geom_b)],
            "body_ids": [body_a, body_b],
            "body_names": [_name(model, "body", body_a), _name(model, "body", body_b)],
            "distance_m": float(contact.dist),
            "robot_target_contact": bool(
                (body_a in target_ids and _is_robot_body(model, body_b))
                or (body_b in target_ids and _is_robot_body(model, body_a))
            ),
        }
        if body_a in target_ids or body_b in target_ids:
            target_contacts.append(item)
        if item["robot_target_contact"]:
            robot_target.append(item)
    return {
        "simulator_contact_count": int(data.ncon),
        "target_contact_count": len(target_contacts),
        "robot_target_contact_count": len(robot_target),
        "robot_target_contact": bool(robot_target),
        "target_contacts": target_contacts,
        "diagnostic_only": True,
    }


def _camera_calibration(environment: Any, image: np.ndarray):
    height, width = image.shape[:2]
    return make_mujoco_calibrations(
        environment.env, {"agentview": "agentview"},
        image_shapes={"agentview": (height, width)},
        rotations={"agentview": 0}, flips={"agentview": "none"},
    )["agentview"]


def _save_frame(image: np.ndarray, segmentation: Any, out_dir: Path, stem: str) -> tuple[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_path, overlay_path = out_dir / f"{stem}_rgb.png", out_dir / f"{stem}_overlay.png"
    Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8), mode="RGB").save(rgb_path)
    base = Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8), mode="RGB").convert("RGBA")
    if segmentation.mask is not None:
        mask = np.asarray(segmentation.mask, dtype=bool)
        red = Image.new("RGBA", base.size, (255, 20, 40, 0))
        red.putalpha(Image.fromarray(np.where(mask, 105, 0).astype(np.uint8), mode="L"))
        base = Image.alpha_composite(base, red)
    overlay = base.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    if segmentation.bbox_xyxy is not None:
        x0, y0, x1, y1 = [float(x) for x in segmentation.bbox_xyxy]
        draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 80), width=2)
    if segmentation.centroid_px is not None:
        x, y = [float(v) for v in segmentation.centroid_px]
        draw.ellipse((x-4, y-4, x+4, y+4), fill=(255, 230, 0), outline=(0, 0, 0))
    draw.text((8, 8), f"{stem} | {segmentation.identity_status}", fill=(255, 255, 0), stroke_width=1,
              stroke_fill=(0, 0, 0))
    overlay.save(overlay_path)
    return str(rgb_path), str(overlay_path)


def _sample_from_perception(
    *, environment: Any, base_observer: LiberoObservationAdapter, observer: ObjectRelativePerceptionObserver,
    body_id: int, tick: int, phase: str, artifact_dir: Path, save_frame: bool,
) -> dict[str, Any]:
    if not observer.perception_history:
        raise RuntimeError("perception observer has no observation to sample")
    frame = observer.perception_history[-1]
    segmentation = frame["segmentation"]
    raw = base_observer.last_raw
    if raw is None:
        raise RuntimeError("base observation adapter has no raw observation")
    image = frame["image"]
    pose = _oracle_pose(environment, body_id)
    eef = np.asarray(raw.eef_position_xyz, dtype=float).reshape(3)
    rgb_path = overlay_path = None
    if save_frame:
        rgb_path, overlay_path = _save_frame(image, segmentation, artifact_dir,
                                             f"{phase}_tick_{tick:02d}")
    sample = {
        "phase": phase,
        "tick": int(tick),
        "environment_step": int(environment.step_count),
        "simulation_time_s": float(environment.env.sim.data.time),
        "timestamp_monotonic": float(raw.timestamp_monotonic),
        "eef_world_position_m": list(eef),
        "target_body_pose": pose,
        "target_world_position_m": pose["world_position_m"],
        "eef_target_distance_m": float(np.linalg.norm(
            eef - np.asarray(pose["world_position_m"], dtype=float)
        )),
        "sam_visible": bool(segmentation.visible),
        "sam_identity_status": segmentation.identity_status,
        "sam_candidate_id": segmentation.selected_candidate_id,
        "sam_centroid_px": list(segmentation.centroid_px) if segmentation.centroid_px is not None else None,
        "sam_bbox_xyxy": list(segmentation.bbox_xyxy) if segmentation.bbox_xyxy is not None else None,
        "sam_mask_area_px": int(segmentation.area_px) if segmentation.area_px is not None else None,
        "sam_quality_score": segmentation.quality_score,
        "camera_calibration": frame["calibration"],
        "camera_signature": frame["camera_signature"],
        "contact": _contacts(environment, body_id),
        "image_path": rgb_path,
        "overlay_path": overlay_path,
    }
    return sample


def _save_plot(path: Path, series: list[tuple[str, list[tuple[float, float]]]], *,
               x_label: str, y_label: str) -> None:
    width, height = 900, 500
    left, top, right, bottom = 76, 36, 30, 64
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    plot_w, plot_h = width - left - right, height - top - bottom
    points = [p for _name, values in series for p in values if np.isfinite(p[0]) and np.isfinite(p[1])]
    if not points:
        draw.text((left, top), "No valid samples", fill="black")
        image.save(path)
        return
    xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
    ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    ymin -= (ymax - ymin) * 0.08
    ymax += (ymax - ymin) * 0.08
    draw.line((left, top, left, top + plot_h), fill="black", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="black", width=2)
    draw.text((left, 8), y_label, fill="black")
    draw.text((left + plot_w // 2, height - 24), x_label, fill="black")
    colors = [(210, 40, 45), (35, 110, 210), (20, 145, 80), (180, 100, 15)]
    for index, (label, values) in enumerate(series):
        color = colors[index % len(colors)]
        coords = [(left + (x-xmin)/(xmax-xmin)*plot_w,
                   top + (ymax-y)/(ymax-ymin)*plot_h) for x, y in values]
        if len(coords) >= 2:
            draw.line(coords, fill=color, width=3)
        for x, y in coords:
            draw.ellipse((x-4, y-4, x+4, y+4), fill=color)
        legend_x = left + index * 190
        draw.line((legend_x, top + 14, legend_x + 24, top + 14), fill=color, width=4)
        draw.text((legend_x + 30, top + 6), label, fill=color)
    image.save(path)


def _controller(config: Mapping[str, Any]) -> LiberoAtomicController:
    return LiberoAtomicController(
        move_vectors=config["move_vectors"], step_m=CONTROL_TICK_MM / 1000.0,
        sim_steps_per_decision=1,
        position_scale_m=float(config.get("position_scale_m", 0.05)),
    )


def _base_observer(workspace: tuple[float, float]) -> LiberoObservationAdapter:
    return LiberoObservationAdapter(
        max_eef_z_m=workspace[1], min_eef_z_m=workspace[0],
        safe_lift_step_m=CONTROL_TICK_MM / 1000.0,
    )


def _run_action_trial(
    *, init_state: int, run_dir: Path, config: Mapping[str, Any], sam3: Sam3Client,
    workspace: tuple[float, float], camera_resolution: int,
) -> dict[str, Any]:
    trial_id = f"init_state_{init_state}_ACTION"
    out = run_dir / trial_id
    artifact_dir = out / "artifacts"
    _ensure_writable(out)
    _ensure_writable(artifact_dir)
    environment = LiberoEnvironmentAdapter.create(
        suite_name=SUITE, task_id=TASK_ID, init_state_index=init_state, seed=0,
        camera_height=camera_resolution, camera_width=camera_resolution, horizon=48,
    )
    body_audit = _target_body_audit(environment)
    if not body_audit["task_instruction_mentions_target"] or not body_audit["body_name_matches_target_phrase"]:
        environment.close()
        raise RuntimeError("target body/task-instruction identity audit failed")
    base = _base_observer(workspace)
    controller = _controller(config)
    pre_settle = []
    try:
        reset = True
        for _ in range(PRE_SETTLE_TICKS):
            result = run_v3_tick(
                environment, base, controller, task_id=f"{SUITE}:{TASK_ID}", token=None,
                direction_unit=None, commanded_step_m=CONTROL_TICK_MM / 1000.0,
                reset=reset, workspace_z_bounds_m=workspace,
            )
            reset = False
            if (result["actions"] != 1 or result["approved_action"] != "CALIBRATION_HOLD"
                    or not result["backend_execution"]):
                raise RuntimeError(f"required pre-settle HOLD was not executed: {result}")
            pre_settle.append(result)
        if environment.step_count != PRE_SETTLE_TICKS:
            raise RuntimeError(f"pre-settle used {environment.step_count} env.step calls, expected 4")

        observer = ObjectRelativePerceptionObserver(
            base, sam3, target_phrase=TARGET_PHRASE, move_vectors=config["move_vectors"],
        )
        trace: list[dict[str, Any]] = []
        pre_ready: dict[str, Any] = {}
        def write_pre_action_ready(state, options, selection):
            if (state.object_relative_state is None
                    or not state.object_relative_state.target_reference_valid
                    or state.object_relative_state.target_reference_point_px is None):
                raise RuntimeError("no valid frozen visual reference at the formal pre-action boundary")
            selected = next((option for option in options if option.option_id == selection.option_id), None)
            if selected is None or selected.primitive.micro_motion_spec is None:
                raise RuntimeError("selected action is missing its bounded micro-motion contract")
            t0 = _sample_from_perception(
                environment=environment, base_observer=base, observer=observer,
                body_id=body_audit["body_id"], tick=0, phase="formal_t0", artifact_dir=artifact_dir,
                save_frame=True,
            )
            trace.append(t0)
            payload = {
                "status": "PRE_ACTION_READY",
                "trial_id": trial_id,
                "init_state_index": init_state,
                "env_step_before_action": int(environment.step_count),
                "pre_settle_hold_ticks": PRE_SETTLE_TICKS,
                "runtime_option": selection.option_id,
                "direction": selected.primitive.micro_motion_spec.direction,
                "max_ticks": selected.primitive.micro_motion_spec.max_ticks,
                "requested_displacement_m": selected.primitive.micro_motion_spec.requested_displacement_m,
                "formal_t0": t0,
                "target_identity": observer.identity_anchor,
                "target_reference": observer.reference_anchor,
                "logger_writable": True,
                "artifact_variables_initialized": True,
                "oracle_used_by_runtime": False,
                "oracle_pose_diagnostic_only": True,
                "translation_command_authorized": True,
            }
            marker = out / "PRE_ACTION_READY.json"
            _write_json(marker, payload)
            read_back = json.loads(marker.read_text(encoding="utf-8"))
            if read_back.get("status") != "PRE_ACTION_READY" or not read_back.get("logger_writable"):
                raise RuntimeError("PRE_ACTION_READY record did not pass read-back verification")
            pre_ready.update(read_back)

        class ReadyArbiter(Arbiter):
            def authorize(self, state, options, selection):
                if selection.option_id == "ALIGN_TO_TARGET_SMALL":
                    write_pre_action_ready(state, options, selection)
                return super().authorize(state, options, selection)

        arbiter = ReadyArbiter()
        backend = LiberoPrimitiveBackend(environment, controller, arbiter)
        events: list[dict[str, Any]] = []

        def observe_action_tick(current_environment):
            observation = observer.observe(current_environment)
            tick_number = len(trace)
            trace.append(_sample_from_perception(
                environment=current_environment, base_observer=base, observer=observer,
                body_id=body_audit["body_id"], tick=tick_number, phase="action",
                artifact_dir=artifact_dir, save_frame=True,
            ))
            return observation

        # Executor invokes this callback exactly once after every physical tick.
        observer.observe_for_execution_tick = observe_action_tick
        runner = RuntimeV3Runner(
            observer=observer, state_builder=StateBuilder(),
            option_generator=ObjectRelativeAlignmentOptionGenerator(),
            selector=DeterministicSelector("ALIGN_TO_TARGET_SMALL"), arbiter=arbiter,
            executor=Executor(backend, arbiter), effect_observer=EffectObserver(), logger=events.append,
        )
        result = runner.run_episode(
            environment, task_id=f"{SUITE}:{TASK_ID}", max_steps=1, reset=False,
        )
        event = events[0] if events else {}
        execution = getattr(event.get("execution"), "result", None)
        execution_ticks = int(execution.get("ticks_executed", 0)) if isinstance(execution, dict) else 0
        if not pre_ready:
            raise RuntimeError(f"action trial did not write PRE_ACTION_READY: {result}")
        if result.get("actions") != 1 or event.get("approved_action") is None:
            raise RuntimeError(f"bounded alignment did not execute exactly once: {result}")
        if execution_ticks < 1 or len(trace) != execution_ticks + 1:
            raise RuntimeError(f"tickwise action trace mismatch: ticks={execution_ticks}, samples={len(trace)}")
        for previous, current in zip(trace, trace[1:]):
            if current["environment_step"] - previous["environment_step"] != 1:
                raise RuntimeError("action samples are not one env.step apart")
        _write_json(out / "trial.json", {
            "trial_id": trial_id, "trial_type": "ACTION", "init_state_index": init_state,
            "suite": SUITE, "task_id": TASK_ID, "seed": 0,
            "task_instruction": environment.task_description,
            "body_audit": body_audit,
            "pre_settle": {"requested_hold_ticks": PRE_SETTLE_TICKS, "env_steps": PRE_SETTLE_TICKS,
                           "runtime_results": pre_settle},
            "pre_action_ready": str(out / "PRE_ACTION_READY.json"),
            "runner_status": result.get("status"), "runner_actions": result.get("actions"),
            "runtime_option": event.get("approved_action").option_id if event.get("approved_action") else None,
            "execution": execution,
            "bounded_translation_actions": 1,
            "qwen_action_count": 0,
            "oracle_used_by_runtime": False,
            "diagnostic_samples": trace,
        })
        return {
            "trial_id": trial_id, "trial_type": "ACTION", "init_state_index": init_state,
            "body_audit": body_audit, "diagnostic_samples": trace,
            "execution_ticks": execution_ticks,
            "action_direction": (event["approved_action"].primitive.micro_motion_spec.direction
                                 if event.get("approved_action") else None),
            "runtime_option": event.get("approved_action").option_id if event.get("approved_action") else None,
            "runner_status": result.get("status"), "runner_actions": result.get("actions"),
            "execution": execution,
        }
    finally:
        environment.close()


def _run_no_action_trial(
    *, init_state: int, matched_ticks: int, run_dir: Path, config: Mapping[str, Any], sam3: Sam3Client,
    workspace: tuple[float, float], camera_resolution: int,
) -> dict[str, Any]:
    trial_id = f"init_state_{init_state}_NO_ACTION"
    out = run_dir / trial_id
    artifact_dir = out / "artifacts"
    _ensure_writable(out)
    _ensure_writable(artifact_dir)
    environment = LiberoEnvironmentAdapter.create(
        suite_name=SUITE, task_id=TASK_ID, init_state_index=init_state, seed=0,
        camera_height=camera_resolution, camera_width=camera_resolution, horizon=48,
    )
    body_audit = _target_body_audit(environment)
    base, controller = _base_observer(workspace), _controller(config)
    observer = ObjectRelativePerceptionObserver(
        base, sam3, target_phrase=TARGET_PHRASE, move_vectors=config["move_vectors"],
    )
    trace: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    try:
        reset = True
        pre_settle = []
        for _ in range(PRE_SETTLE_TICKS):
            result = run_v3_tick(
                environment, base, controller, task_id=f"{SUITE}:{TASK_ID}", token=None,
                direction_unit=None, commanded_step_m=CONTROL_TICK_MM / 1000.0,
                reset=reset, workspace_z_bounds_m=workspace,
            )
            reset = False
            if (result["actions"] != 1 or result["approved_action"] != "CALIBRATION_HOLD"
                    or not result["backend_execution"]):
                raise RuntimeError(f"NO_ACTION pre-settle did not execute HOLD: {result}")
            pre_settle.append(result)
        if environment.step_count != PRE_SETTLE_TICKS:
            raise RuntimeError("NO_ACTION pre-settle did not use exactly four environment steps")
        observer.observe(environment)
        trace.append(_sample_from_perception(
            environment=environment, base_observer=base, observer=observer,
            body_id=body_audit["body_id"], tick=0, phase="no_action", artifact_dir=artifact_dir,
            save_frame=True,
        ))
        for tick in range(1, int(matched_ticks) + 1):
            before_steps = environment.step_count
            result = run_v3_tick(
                environment, base, controller, task_id=f"{SUITE}:{TASK_ID}", token=None,
                direction_unit=None, commanded_step_m=CONTROL_TICK_MM / 1000.0,
                reset=False, workspace_z_bounds_m=workspace,
            )
            if (result["actions"] != 1 or result["approved_action"] != "CALIBRATION_HOLD"
                    or not result["backend_execution"] or environment.step_count != before_steps + 1):
                raise RuntimeError(f"NO_ACTION step was not exactly one Executor HOLD: {result}")
            observer.observe(environment)
            commands.append({"kind": "HOLD", "translation_command_count": 0,
                             "approved_option": result["approved_action"],
                             "environment_step": int(environment.step_count)})
            trace.append(_sample_from_perception(
                environment=environment, base_observer=base, observer=observer,
                body_id=body_audit["body_id"], tick=tick, phase="no_action", artifact_dir=artifact_dir,
                save_frame=True,
            ))
        validate_no_action_commands(commands)
        _write_json(out / "trial.json", {
            "trial_id": trial_id, "trial_type": "NO_ACTION", "init_state_index": init_state,
            "suite": SUITE, "task_id": TASK_ID, "seed": 0,
            "body_audit": body_audit,
            "pre_settle_hold_ticks": PRE_SETTLE_TICKS,
            "matched_duration_ticks": len(commands), "commands": commands,
            "zero_translation_commands": len(commands),
            "oracle_used_by_runtime": False,
            "diagnostic_samples": trace,
        })
        return {
            "trial_id": trial_id, "trial_type": "NO_ACTION", "init_state_index": init_state,
            "body_audit": body_audit, "diagnostic_samples": trace,
            "execution_ticks": len(commands), "commands": commands,
        }
    finally:
        environment.close()


def _run_settling_curve(
    *, init_state: int, run_dir: Path, config: Mapping[str, Any], sam3: Sam3Client,
    workspace: tuple[float, float], camera_resolution: int,
) -> dict[str, Any]:
    trial_id = f"init_state_{init_state}_SETTLING"
    out = run_dir / trial_id
    artifact_dir = out / "artifacts"
    _ensure_writable(out)
    _ensure_writable(artifact_dir)
    environment = LiberoEnvironmentAdapter.create(
        suite_name=SUITE, task_id=TASK_ID, init_state_index=init_state, seed=0,
        camera_height=camera_resolution, camera_width=camera_resolution, horizon=64,
    )
    body_audit = _target_body_audit(environment)
    base, controller = _base_observer(workspace), _controller(config)
    observer = ObjectRelativePerceptionObserver(
        base, sam3, target_phrase=TARGET_PHRASE, move_vectors=config["move_vectors"],
    )
    try:
        environment.reset()
        if environment.step_count != 0:
            raise RuntimeError("settling curve must start at environment step zero")
        observer.observe(environment)
        samples = [_sample_from_perception(
            environment=environment, base_observer=base, observer=observer,
            body_id=body_audit["body_id"], tick=0, phase="settling", artifact_dir=artifact_dir,
            save_frame=True,
        )]
        selected = set(SETTLING_TICKS)
        for tick in range(1, max(SETTLING_TICKS) + 1):
            before_steps = environment.step_count
            result = run_v3_tick(
                environment, base, controller, task_id=f"{SUITE}:{TASK_ID}", token=None,
                direction_unit=None, commanded_step_m=CONTROL_TICK_MM / 1000.0,
                reset=False, workspace_z_bounds_m=workspace,
            )
            if (result["actions"] != 1 or result["approved_action"] != "CALIBRATION_HOLD"
                    or not result["backend_execution"] or environment.step_count != before_steps + 1):
                raise RuntimeError(f"settling curve step was not one Executor HOLD: {result}")
            if tick in selected:
                observer.observe(environment)
                samples.append(_sample_from_perception(
                    environment=environment, base_observer=base, observer=observer,
                    body_id=body_audit["body_id"], tick=tick, phase="settling", artifact_dir=artifact_dir,
                    save_frame=True,
                ))
        curve = target_motion_curve(samples)
        record = {
            "trial_id": trial_id, "trial_type": "SETTLING_DIAGNOSTIC_ONLY",
            "init_state_index": init_state, "suite": SUITE, "task_id": TASK_ID, "seed": 0,
            "body_audit": body_audit, "selected_ticks": list(SETTLING_TICKS),
            "hold_environment_steps": max(SETTLING_TICKS),
            "runtime_translation_commands": 0,
            "oracle_used_by_runtime": False,
            "samples": curve,
        }
        _write_json(out / "trial.json", record)
        return record
    finally:
        environment.close()


def _endpoint_pair(record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    samples = record["diagnostic_samples"]
    return samples[0], samples[-1]


def _matched_row(action: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    require_matched_duration(action["execution_ticks"], control["execution_ticks"])
    action_before, action_after = _endpoint_pair(action)
    control_before, control_after = _endpoint_pair(control)
    action_delta = displacement(action_before["target_world_position_m"],
                                action_after["target_world_position_m"])
    control_delta = displacement(control_before["target_world_position_m"],
                                 control_after["target_world_position_m"])
    excess = action_excess_motion(action_delta["delta_xyz_m"], control_delta["delta_xyz_m"])
    calibration = action_before["camera_calibration"]
    camera_same = action_before["camera_signature"] == action_after["camera_signature"]
    if not camera_same:
        raise RuntimeError("agentview camera calibration changed within the action window")
    projected = project_world_motion_to_canonical_pixels(
        action_before["target_world_position_m"], action_after["target_world_position_m"],
        calibration, CANONICAL,
    )
    pixel_comparison = compare_oracle_and_sam_pixel_motion(
        projected.get("oracle_pixel_shift") if projected.get("available") else None,
        action_before["sam_centroid_px"], action_after["sam_centroid_px"],
    )
    return {
        "init_state_index": action["init_state_index"],
        "matched_ticks": action["execution_ticks"],
        "action_direction": action.get("action_direction"),
        "action_window_env_steps": [action_before["environment_step"], action_after["environment_step"]],
        "no_action_window_env_steps": [control_before["environment_step"], control_after["environment_step"]],
        "action_target_displacement": action_delta,
        "no_action_target_displacement": control_delta,
        "action_excess_motion": excess,
        "eef_action_displacement": displacement(action_before["eef_world_position_m"],
                                                  action_after["eef_world_position_m"]),
        "eef_no_action_displacement": displacement(control_before["eef_world_position_m"],
                                                     control_after["eef_world_position_m"]),
        "action_contact_events": [sample["contact"] for sample in action["diagnostic_samples"]],
        "action_any_robot_target_contact": any(
            sample["contact"]["robot_target_contact"] for sample in action["diagnostic_samples"]
        ),
        "action_eef_target_distance_m_by_tick": [
            {"tick": sample["tick"], "environment_step": sample["environment_step"],
             "distance_m": sample["eef_target_distance_m"]}
            for sample in action["diagnostic_samples"]
        ],
        "oracle_pixel_projection": projected,
        "sam_oracle_pixel_comparison": pixel_comparison,
        "sam_no_action_centroid_shift_px": (
            (np.asarray(control_after["sam_centroid_px"], dtype=float)
             - np.asarray(control_before["sam_centroid_px"], dtype=float)).tolist()
            if control_before["sam_centroid_px"] is not None and control_after["sam_centroid_px"] is not None
            else None
        ),
        "camera_unchanged_within_action": camera_same,
        "oracle_used_by_runtime": False,
        "action_trial_id": action["trial_id"],
        "no_action_trial_id": control["trial_id"],
    }


def _write_csv(path: Path, headers: list[str], rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_jsonable(row.get(key)), ensure_ascii=False)
                             if isinstance(row.get(key), (dict, list, tuple)) else row.get(key)
                             for key in headers})
        stream.flush()
        os.fsync(stream.fileno())


def _root_cause(matched: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not matched or any(not row["oracle_pixel_projection"].get("available")
                          for row in matched):
        return {"classification": "ORACLE_DIAGNOSTIC_BUG",
                "evidence": "At least one required same-body world-to-image measurement was unavailable."}
    natural_magnitude = sum(float(row["no_action_target_displacement"]["norm_m"]) for row in matched)
    action_excess_magnitude = sum(float(row["action_excess_motion"]["norm_m"]) for row in matched)
    same_direction_pairs = []
    for row in matched:
        action_vec = np.asarray(row["action_target_displacement"]["delta_xyz_m"], dtype=float)
        hold_vec = np.asarray(row["no_action_target_displacement"]["delta_xyz_m"], dtype=float)
        same_direction_pairs.append(float(np.dot(action_vec, hold_vec)) > 0.0)
    # Compare measured paired-control motion to the per-pair action excess. No
    # absolute displacement cut-off is imposed before inspecting these trials.
    if all(same_direction_pairs) and natural_magnitude > action_excess_magnitude:
        return {
            "classification": "NATURAL_SCENE_SETTLING",
            "evidence": "All three ACTION vectors point with their same-init-state NO_ACTION vectors, and summed NO_ACTION displacement norm exceeds summed paired action-excess norm.",
            "summed_no_action_norm_m": natural_magnitude,
            "summed_action_excess_norm_m": action_excess_magnitude,
            "secondary_uncertainty": "Three one-shot matched pairs do not estimate between-reset variance.",
        }
    if action_excess_magnitude > natural_magnitude:
        return {
            "classification": "ACTION_INDUCED_TARGET_MOTION",
            "evidence": "Summed paired action-excess displacement norm exceeds summed same-init-state NO_ACTION displacement norm.",
            "summed_no_action_norm_m": natural_magnitude,
            "summed_action_excess_norm_m": action_excess_magnitude,
            "secondary_uncertainty": "Matched pairs are one-shot; contact and direction are reported separately.",
        }
    return {
        "classification": "MIXED / UNRESOLVED",
        "evidence": "Same-init-state ACTION and NO_ACTION displacement vectors do not meet a consistent attribution rule across all three init states.",
        "secondary_uncertainty": "Only one matched pair per init state was run, as specified.",
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/robot_libero_clean_qwen3vl.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "rollouts/runtime_v3_scene_settling"))
    parser.add_argument("--sam3-url", default="http://127.0.0.1:8773/sse")
    parser.add_argument("--sam3-python", default="/root/autodl-tmp/openeta-services/sam3/.venv/bin/python")
    parser.add_argument("--sam3-timeout-s", type=float, default=120.0)
    parser.add_argument("--camera-resolution", type=int, default=512)
    args = parser.parse_args()
    if args.camera_resolution != 512:
        raise SystemExit("M3.1c freezes direct source rendering to 512x512")
    config = load_yaml(args.config)
    if config.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(config["libero_dir"])
    _configure_proxy_bypass(args.sam3_url)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_dir).expanduser() / f"run_{stamp}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    _ensure_writable(run_dir)
    _write_json(run_dir / "RUN_SETUP_READY.json", {
        "status": "RUN_SETUP_READY", "branch": "runtime-v3",
        "starting_commit": "5ee4ddef8ee55f3bbd048dcc589f1b290cca9525",
        "camera_render": [512, 512], "canonical_transform": "vertical_flip",
        "sam3_confidence_threshold": 0.05, "qwen_action_count": 0,
        "logger_writable": True, "oracle_used_by_runtime": False,
        "planned_trials": {"action": 3, "no_action": 3, "settling_diagnostic_only": 3},
    })
    _ensure_writable(run_dir / "ACTION_preflight")
    workspace = (0.02, 0.60)
    sam3 = Sam3Client(url=args.sam3_url, python=args.sam3_python,
                      timeout_s=args.sam3_timeout_s, max_attempts=1)
    action_rows, control_rows, curves = [], [], []
    blockers: list[str] = []
    try:
        for init_state in INIT_STATES:
            try:
                action = _run_action_trial(
                    init_state=init_state, run_dir=run_dir, config=config, sam3=sam3,
                    workspace=workspace, camera_resolution=args.camera_resolution,
                )
                action_rows.append(action)
                control = _run_no_action_trial(
                    init_state=init_state, matched_ticks=action["execution_ticks"],
                    run_dir=run_dir, config=config, sam3=sam3,
                    workspace=workspace, camera_resolution=args.camera_resolution,
                )
                control_rows.append(control)
            except Exception as exc:
                blockers.append(f"init_state_{init_state} matched trials: {type(exc).__name__}: {exc}")
                failure_dir = run_dir / f"init_state_{init_state}_failure"
                failure_dir.mkdir(parents=True, exist_ok=True)
                _write_json(failure_dir / "failure.json", {"error": blockers[-1]})
        if len(action_rows) == 3 and len(control_rows) == 3:
            for action, control in zip(action_rows, control_rows):
                try:
                    _append_jsonl(run_dir / "matched_trials.jsonl", _matched_row(action, control))
                except Exception as exc:
                    blockers.append(f"init_state_{action['init_state_index']} attribution: {type(exc).__name__}: {exc}")
        for init_state in INIT_STATES:
            try:
                curves.append(_run_settling_curve(
                    init_state=init_state, run_dir=run_dir, config=config, sam3=sam3,
                    workspace=workspace, camera_resolution=args.camera_resolution,
                ))
            except Exception as exc:
                blockers.append(f"init_state_{init_state} settling curve: {type(exc).__name__}: {exc}")
                failure_dir = run_dir / f"init_state_{init_state}_settling_failure"
                failure_dir.mkdir(parents=True, exist_ok=True)
                _write_json(failure_dir / "failure.json", {"error": blockers[-1]})
    finally:
        sam3.close()

    matched_path = run_dir / "matched_trials.jsonl"
    matched = [json.loads(line) for line in matched_path.read_text(encoding="utf-8").splitlines()
               if line.strip()] if matched_path.exists() else []
    settling_curves = {str(row["init_state_index"]): row["samples"] for row in curves}
    _write_json(run_dir / "settling_curves.json", {"curves": curves})
    z_rows = []
    for init, curve in settling_curves.items():
        for sample in curve:
            z_rows.append({
                "init_state_index": int(init), "tick": sample["tick"],
                "simulation_time_s": sample["simulation_time_s"],
                "target_x_m": sample["target_world_position_m"][0],
                "target_y_m": sample["target_world_position_m"][1],
                "target_z_m": sample["target_world_position_m"][2],
                "target_delta_z_from_tick0_m": sample["target_cumulative_delta_from_tick0_m"][2],
                "eef_x_m": sample["eef_world_position_m"][0],
                "eef_y_m": sample["eef_world_position_m"][1],
                "eef_z_m": sample["eef_world_position_m"][2],
                "eef_delta_z_from_tick0_m": sample["eef_cumulative_delta_from_tick0_m"][2],
                "eef_target_distance_m": sample.get("eef_target_distance_m"),
                "sam_centroid_x_px": (sample["sam_centroid_px"][0] if sample["sam_centroid_px"] else None),
                "sam_centroid_y_px": (sample["sam_centroid_px"][1] if sample["sam_centroid_px"] else None),
                "sam_bbox_xyxy": sample["sam_bbox_xyxy"], "sam_mask_area_px": sample["sam_mask_area_px"],
            })
    _write_csv(run_dir / "target_z_curve.csv", list(z_rows[0]) if z_rows else ["init_state_index", "tick"], z_rows)
    pixel_rows = []
    for row in matched:
        pixel_rows.append({
            "init_state_index": row["init_state_index"],
            "oracle_pixel_shift": row["oracle_pixel_projection"].get("oracle_pixel_shift"),
            "sam_centroid_shift": row["sam_oracle_pixel_comparison"].get("sam_centroid_shift"),
            "residual_sam_minus_oracle_px": row["sam_oracle_pixel_comparison"].get("residual_sam_minus_oracle_px"),
            "residual_norm_px": row["sam_oracle_pixel_comparison"].get("residual_norm_px"),
            "no_action_sam_centroid_shift_px": row["sam_no_action_centroid_shift_px"],
        })
    _write_csv(run_dir / "pixel_motion_comparison.csv", [
        "init_state_index", "oracle_pixel_shift", "sam_centroid_shift",
        "residual_sam_minus_oracle_px", "residual_norm_px", "no_action_sam_centroid_shift_px",
    ], pixel_rows)

    curve_plot = []
    target_z_plot = []
    eef_displacement_plot = []
    centroid_plot = []
    for init, samples in settling_curves.items():
        target_z_series = [(s["tick"], s["target_world_position_m"][2]) for s in samples]
        eef_z_series = [(s["tick"], s["eef_world_position_m"][2]) for s in samples]
        target_z_plot.append((f"state {init}", target_z_series))
        curve_plot.append((f"state {init} target z", target_z_series))
        curve_plot.append((f"state {init} EEF z", eef_z_series))
        eef_displacement_plot.append((f"state {init}", [
            (s["tick"], float(np.linalg.norm(s["eef_cumulative_delta_from_tick0_m"])) * 1000.0)
            for s in samples
        ]))
        centroid_plot.append((f"state {init} SAM y", [(s["tick"], s["sam_centroid_px"][1])
                                                       for s in samples if s["sam_centroid_px"] is not None]))
    _save_plot(run_dir / "target_and_eef_z_vs_tick.png", curve_plot,
               x_label="HOLD tick", y_label="World z (m)")
    _save_plot(run_dir / "target_z_vs_tick.png", target_z_plot,
               x_label="HOLD tick", y_label="Target body world z (m)")
    _save_plot(run_dir / "eef_displacement_vs_tick.png", eef_displacement_plot,
               x_label="HOLD tick", y_label="EEF displacement from tick 0 (mm)")
    _save_plot(run_dir / "sam_centroid_y_vs_tick.png", centroid_plot,
               x_label="HOLD tick", y_label="Canonical centroid y (px)")
    root_cause = _root_cause(matched)
    body_ids = [row["body_audit"]["body_id"] for row in action_rows + control_rows + curves]
    final = {
        "status": "COMPLETED" if len(action_rows) == 3 and len(control_rows) == 3
        and len(curves) == 3 and not blockers else "PARTIAL",
        "branch": "runtime-v3", "starting_commit": "5ee4ddef8ee55f3bbd048dcc589f1b290cca9525",
        "suite": SUITE, "task_id": TASK_ID, "target_phrase": TARGET_PHRASE,
        "task_seed": 0, "init_state_indices": list(INIT_STATES),
        "canonical_transform": CANONICAL.orientation, "source_resolution": [512, 512],
        "sam3_confidence_threshold": 0.05, "qwen_action_count": 0,
        "trial_counts": {"ACTION": len(action_rows), "NO_ACTION": len(control_rows),
                         "SETTLING_DIAGNOSTIC_ONLY": len(curves)},
        "oracle_audit": {
            "body": TARGET_BODY, "world_position_source": "environment.env.sim.data.xpos[body_id]",
            "body_ids_observed": body_ids, "body_id_stable_across_trials": len(set(body_ids)) <= 1,
            "target_instruction_and_name_audited": all(
                row["body_audit"]["task_instruction_mentions_target"]
                and row["body_audit"]["body_name_matches_target_phrase"]
                for row in action_rows + control_rows + curves
            ),
            "before_after_sampling": "same body ID at formal t0 after exactly 4 HOLD env.step calls, and after each action/control env.step; ACTION ticks captured by Executor tick_observer",
            "oracle_used_by_runtime": False,
        },
        "matched_trials": matched,
        "settling_curves": settling_curves,
        "root_cause": root_cause,
        "contact_diagnostic": {
            "action_trial_count": len(action_rows),
            "trials_with_robot_target_contact": sum(bool(row.get("diagnostic_samples")
                and any(sample["contact"]["robot_target_contact"] for sample in row["diagnostic_samples"]))
                for row in action_rows),
            "events_by_trial": {str(row["init_state_index"]): [sample["contact"]
                for sample in row["diagnostic_samples"]] for row in action_rows},
        },
        "scene_ready_implemented": True,
        "scene_ready_applied_to_matched_trials": False,
        "scene_ready_design": {
            "runtime_inputs": ["same associated target identity", "SAM centroid stability",
                               "SAM bbox stability", "SAM mask-area stability"],
            "oracle_used": "MUST BE NO",
            "window_observations": 3,
            "thresholds": {"centroid_shift_px_max": 0.02,
                            "bbox_edge_shift_px_max": 0.0,
                            "mask_area_change_px_max": 1},
            "threshold_source": (
                "Stable HOLD samples at ticks 10/15/20 in init states 0/1/2: maximum paired "
                "centroid variation 0.0162061 px, bbox-edge variation 0 px, mask-area variation "
                "1 px; centroid limit rounded up to 0.02 px."
            ),
            "promotion": "The pre-ready identity is provisional association for settling evidence; "
                        "the formal TargetIdentityAnchor and TargetReferenceAnchor use the current "
                        "associated mask after the stable window passes.",
            "initialization_hold_bound_ticks": 40,
        },
        "artifacts": {
            "matched_trials_jsonl": str(matched_path),
            "settling_curves_json": str(run_dir / "settling_curves.json"),
            "target_z_curve_csv": str(run_dir / "target_z_curve.csv"),
            "pixel_motion_comparison_csv": str(run_dir / "pixel_motion_comparison.csv"),
            "target_z_plot": str(run_dir / "target_z_vs_tick.png"),
            "eef_displacement_plot": str(run_dir / "eef_displacement_vs_tick.png"),
            "sam_centroid_y_plot": str(run_dir / "sam_centroid_y_vs_tick.png"),
        },
        "blockers": blockers, "legacy_config_modified": False, "legacy_policy_modified": False,
        "runtime_oracle_leak": False,
    }
    _write_json(run_dir / "summary.json", final)
    print(json.dumps(_jsonable(final), indent=2, ensure_ascii=False))
    return 0 if final["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(_main())

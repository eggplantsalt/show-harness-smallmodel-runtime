#!/usr/bin/env python3
"""Validate Runtime V3 SceneReady online, then optionally run one-step trials.

Stage A is intentionally initialization-only. It runs four RobotReady HOLDs,
then holds until the existing visual SceneReady gate fires (or reaches its
40-HOLD bound), and runs three post-ready diagnostic HOLDs. Oracle pose data is
written only to experiment artifacts and never enters Runtime observations.

Stage B is kept behind an explicit second invocation so raw Stage A curves can
be reviewed before any bounded alignment action is authorized.
"""

from __future__ import annotations

import argparse
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

from core.capabilities.camera_geometry import CameraCalibration
from core.capabilities.sam3_client import Sam3Client
from core.config import load_yaml
from core.runtime_v3.adapters.libero_env import LiberoEnvironmentAdapter
from core.runtime_v3.adapters.libero_observation import LiberoObservationAdapter
from core.runtime_v3.canonical_image import CanonicalImageAdapter
from core.runtime_v3.object_relative import ObjectRelativePerceptionObserver
from core.runtime_v3.scene_initialization import scene_ready_status
from core.runtime_v3.scene_settling import (
    SCENE_READY_MAX_BBOX_EDGE_SHIFT_PX,
    SCENE_READY_MAX_CENTROID_SHIFT_PX,
    SCENE_READY_MAX_MASK_AREA_CHANGE_PX,
    SCENE_READY_WINDOW_OBSERVATIONS,
)
from scripts.runtime_v3_scene_settling_diagnostic import (
    compare_oracle_and_sam_pixel_motion,
    project_world_motion_to_canonical_pixels,
)
from core.runtime_v3.temporal_calibration import run_v3_tick
from interpreters.libero_atomic_controller import LiberoAtomicController


SUITE = "LIBERO_OBJECT"
TASK_ID = 2
TASK_SEED = 0
TARGET_PHRASE = "salad dressing"
ROBOT_READY_HOLD_TICKS = 4
SCENE_READY_MAX_HOLD_TICKS = 40
POST_READY_DIAGNOSTIC_HOLDS = 3
CONTROL_TICK_MM = 5.0
FRAME_TICKS = {0, 4, 6, 8}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable({name: getattr(value, name) for name in value.__dataclass_fields__})
    return repr(value)


def _write_json(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(record), stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _new_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = base / f"run_{stamp}_{uuid.uuid4().hex[:8]}"
    result.mkdir(parents=True, exist_ok=False)
    return result


def _configure_local_proxy_bypass(url: str) -> None:
    if urlparse(url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        return
    names = ("NO_PROXY", "no_proxy")
    entries = [item.strip() for name in names for item in os.environ.get(name, "").split(",") if item.strip()]
    entries.extend(("127.0.0.1", "localhost", "::1"))
    value = ",".join(dict.fromkeys(entries))
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def _target_body_id(environment: Any) -> int:
    model = environment.env.sim.model
    body_name = "salad_dressing_1_main"
    if callable(getattr(model, "body_name2id", None)):
        return int(model.body_name2id(body_name))
    return int(model.body(body_name).id)


def _oracle_pose(environment: Any, body_id: int) -> dict[str, Any]:
    """Simulator-only target pose diagnostic; never passed to Runtime."""
    position = np.asarray(environment.env.sim.data.xpos[body_id], dtype=float).reshape(3)
    if not np.all(np.isfinite(position)):
        raise ValueError("oracle target position is not finite")
    return {
        "body": "salad_dressing_1_main",
        "body_id": int(body_id),
        "world_position_m": position.tolist(),
        "z_m": float(position[2]),
        "source": "simulator_body_pose_diagnostic_only",
        "diagnostic_only": True,
        "oracle_used_by_runtime": False,
    }


def _save_overlay(frame: Mapping[str, Any], path: Path, label: str) -> None:
    image = Image.fromarray(np.ascontiguousarray(frame["image"], dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    segmentation = frame["segmentation"]
    if segmentation.mask is not None:
        mask = np.asarray(segmentation.mask, dtype=bool)
        outline = mask & ~np.pad(mask[1:, :], ((0, 1), (0, 0)), constant_values=False)
        outline |= mask & ~np.pad(mask[:-1, :], ((1, 0), (0, 0)), constant_values=False)
        outline |= mask & ~np.pad(mask[:, 1:], ((0, 0), (0, 1)), constant_values=False)
        outline |= mask & ~np.pad(mask[:, :-1], ((0, 0), (1, 0)), constant_values=False)
        ys, xs = np.nonzero(outline)
        for x, y in zip(xs.tolist(), ys.tolist()):
            draw.point((x, y), fill=(255, 50, 50))
    if segmentation.centroid_px is not None:
        x, y = segmentation.centroid_px
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(255, 230, 0), width=2)
    draw.rectangle((0, 0, 245, 44), fill=(0, 0, 0))
    draw.text((6, 5), label, fill=(255, 255, 255))
    draw.text((6, 24), f"SAM {segmentation.identity_status} / ready={frame['scene_ready']}", fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _contact_sheet(paths: list[Path], output: Path) -> None:
    unique = list(dict.fromkeys(path for path in paths if path.exists()))
    if not unique:
        return
    thumb_w, thumb_h, label_h = 256, 256, 28
    columns = min(3, len(unique))
    rows = (len(unique) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(unique):
        image = Image.open(path).convert("RGB").resize((thumb_w, thumb_h))
        x, y = (index % columns) * thumb_w, (index // columns) * (thumb_h + label_h)
        sheet.paste(image, (x, y))
        draw.text((x + 5, y + thumb_h + 4), path.stem, fill=(0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def _save_oracle_curve_plot(rows: list[Mapping[str, Any]], output: Path) -> None:
    """Plot raw per-state target height and log displacement for human review."""
    width, panel_h = 1200, 330
    top_margin, left, right = 40, 80, 35
    colors = [(205, 45, 45), (35, 100, 200), (20, 145, 80),
              (170, 90, 180), (205, 130, 15), (20, 145, 155)]
    image = Image.new("RGB", (width, 2 * panel_h + 30), "white")
    draw = ImageDraw.Draw(image)
    panels = (
        ("Oracle target z by environment tick (m)", "z_m", False),
        ("Oracle per-tick displacement (log10 meters)",
         "target_displacement_from_previous_tick_norm_m", True),
    )
    for panel_index, (title, field, log_scale) in enumerate(panels):
        y0 = panel_index * (panel_h + 30)
        plot_top, plot_bottom = y0 + top_margin, y0 + panel_h - 42
        plot_left, plot_right = left, width - right
        values: list[tuple[int, float, tuple[int, int, int], int]] = []
        for index, row in enumerate(rows):
            color = colors[index % len(colors)]
            for sample in row.get("trace", []):
                value = (sample.get("oracle_target", {}).get(field)
                         if field == "z_m" else sample.get(field))
                if value is None:
                    continue
                number = float(value)
                if log_scale:
                    number = float(np.log10(max(number, 1e-18)))
                values.append((int(sample["tick"]), number, color, int(row["init_state_index"])))
        if not values:
            continue
        xmin, xmax = min(item[0] for item in values), max(item[0] for item in values)
        ymin, ymax = min(item[1] for item in values), max(item[1] for item in values)
        if xmax <= xmin:
            xmax = xmin + 1
        if ymax <= ymin:
            ymax = ymin + 1.0
        pad = (ymax - ymin) * 0.08
        ymin, ymax = ymin - pad, ymax + pad
        draw.text((plot_left, y0 + 8), title, fill=(0, 0, 0))
        draw.text((5, plot_top - 4), f"{ymax:.3g}", fill=(0, 0, 0))
        draw.text((5, plot_bottom - 8), f"{ymin:.3g}", fill=(0, 0, 0))
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(0, 0, 0), width=2)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(0, 0, 0), width=2)
        for tick in range(xmin, xmax + 1, max(1, (xmax - xmin) // 8 or 1)):
            x = plot_left + (tick - xmin) / (xmax - xmin) * (plot_right - plot_left)
            draw.line((x, plot_bottom, x, plot_bottom + 5), fill=(0, 0, 0))
            draw.text((x - 8, plot_bottom + 8), str(tick), fill=(0, 0, 0))
        for index, row in enumerate(rows):
            color = colors[index % len(colors)]
            series = [(tick, value) for tick, value, sample_color, state in values
                      if sample_color == color and state == int(row["init_state_index"])]
            points = [(
                plot_left + (tick - xmin) / (xmax - xmin) * (plot_right - plot_left),
                plot_bottom - (value - ymin) / (ymax - ymin) * (plot_bottom - plot_top),
            ) for tick, value in series]
            if len(points) > 1:
                draw.line(points, fill=color, width=2)
            for x, y in points:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            if not log_scale:
                trigger_tick = row.get("scene_ready", {}).get("trigger_environment_tick")
                if trigger_tick is not None:
                    marker_x = plot_left + (int(trigger_tick) - xmin) / (xmax - xmin) * (plot_right - plot_left)
                    for marker_y in range(plot_top, plot_bottom, 10):
                        draw.line((marker_x, marker_y, marker_x, min(marker_y + 4, plot_bottom)),
                                  fill=color, width=1)
            legend_x = plot_left + index * 150
            draw.line((legend_x, y0 + 28, legend_x + 20, y0 + 28), fill=color, width=3)
            draw.text((legend_x + 25, y0 + 20), f"init {row['init_state_index']}", fill=(0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _one_hold(environment: Any, base: LiberoObservationAdapter, controller: LiberoAtomicController,
              *, reset: bool, workspace: tuple[float, float]) -> dict[str, Any]:
    before = int(environment.step_count)
    result = run_v3_tick(
        environment, base, controller, task_id=f"{SUITE}:{TASK_ID}", token=None,
        direction_unit=None, commanded_step_m=CONTROL_TICK_MM / 1000.0,
        reset=reset, workspace_z_bounds_m=workspace,
    )
    if (int(result.get("actions", 0)) != 1
            or result.get("approved_action") != "CALIBRATION_HOLD"
            or not result.get("backend_execution")
            or int(environment.step_count) != before + 1):
        raise RuntimeError(f"initialization expected exactly one Executor HOLD: {result}")
    return result


def _sample(environment: Any, base: LiberoObservationAdapter, observer: Any, body_id: int,
            previous_pose: list[float] | None, *, phase: str, trigger_tick: int | None,
            selected_frame: Path | None = None) -> tuple[dict[str, Any], list[float]]:
    frame = observer.perception_history[-1]
    segmentation = frame["segmentation"]
    raw = base.last_raw
    pose = _oracle_pose(environment, body_id)
    xyz = pose["world_position_m"]
    delta = None if previous_pose is None else (np.asarray(xyz) - np.asarray(previous_pose)).tolist()
    evidence = (observer.scene_ready_evidence.to_record()
                if observer.scene_ready_evidence is not None else {})
    record = {
        "tick": int(environment.step_count),
        "environment_step": int(environment.step_count),
        "phase": phase,
        "simulation_time_s": float(environment.env.sim.data.time),
        "eef_pose": {
            "position_xyz_m": list(raw.eef_position_xyz) if raw is not None else None,
            "quaternion_xyzw": list(raw.eef_quaternion) if raw is not None else None,
        },
        "sam_target": {
            "visible": bool(segmentation.visible),
            "identity_status": segmentation.identity_status,
            "candidate_id": segmentation.selected_candidate_id,
            "centroid_px": list(segmentation.centroid_px) if segmentation.centroid_px is not None else None,
            "bbox_xyxy": list(segmentation.bbox_xyxy) if segmentation.bbox_xyxy is not None else None,
            "mask_area_px": int(segmentation.area_px) if segmentation.area_px is not None else None,
            "quality_score": segmentation.quality_score,
        },
        "scene_ready": bool(observer.scene_ready),
        "scene_ready_internal": evidence,
        "stable_window_length": int(evidence.get("stable_observation_count", 0)),
        "stable_interval_flags": evidence.get("last_visual_interval"),
        "scene_ready_triggered_this_tick": bool(observer.scene_ready and trigger_tick == environment.step_count),
        "scene_ready_trigger_tick": trigger_tick,
        "oracle_target": pose,
        "target_delta_from_previous_tick_m": delta,
        "target_displacement_from_previous_tick_norm_m": (
            float(np.linalg.norm(delta)) if delta is not None else None
        ),
        "image_path": str(selected_frame) if selected_frame else None,
        "oracle_used_by_runtime": False,
    }
    if selected_frame is not None:
        _save_overlay(frame, selected_frame, f"{phase} | tick {environment.step_count}")
    return record, xyz


def _run_stage_a_trial(*, init_state: int, run_dir: Path, config: Mapping[str, Any],
                       sam3: Sam3Client, workspace: tuple[float, float], resolution: int) -> dict[str, Any]:
    trial_dir = run_dir / f"init_state_{init_state}"
    artifacts = trial_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=False)
    environment = LiberoEnvironmentAdapter.create(
        suite_name=SUITE, task_id=TASK_ID, init_state_index=init_state, seed=TASK_SEED,
        camera_height=resolution, camera_width=resolution, horizon=128,
    )
    base = LiberoObservationAdapter(max_eef_z_m=workspace[1], min_eef_z_m=workspace[0],
                                   safe_lift_step_m=CONTROL_TICK_MM / 1000.0)
    controller = LiberoAtomicController(
        move_vectors=config["move_vectors"], step_m=CONTROL_TICK_MM / 1000.0,
        sim_steps_per_decision=1, position_scale_m=float(config.get("position_scale_m", 0.05)),
    )
    observer = ObjectRelativePerceptionObserver(
        base, sam3, target_phrase=TARGET_PHRASE, move_vectors=config["move_vectors"],
        scene_ready_required=True,
    )
    try:
        if TARGET_PHRASE.casefold() not in environment.task_description.casefold():
            raise RuntimeError(f"requested target is absent from task instruction: {environment.task_description!r}")
        body_id = _target_body_id(environment)
        trace: list[dict[str, Any]] = []
        saved: list[Path] = []
        previous_pose: list[float] | None = None
        trigger_tick: int | None = None
        initial = environment.reset()
        del initial
        # Tick 0 records the reset state. The evidence window starts only after
        # the four RobotReady holds; the reset observation cannot trigger it.
        observer.observe(environment)
        observer.scene_ready_evidence.reset()
        record, previous_pose = _sample(
            environment, base, observer, body_id, previous_pose, phase="RESET_DIAGNOSTIC",
            trigger_tick=None, selected_frame=artifacts / "tick_000_overlay.png",
        )
        trace.append(record)
        saved.append(artifacts / "tick_000_overlay.png")

        for robot_tick in range(1, ROBOT_READY_HOLD_TICKS + 1):
            _one_hold(environment, base, controller, reset=False, workspace=workspace)
            observer.observe(environment)
            if robot_tick < ROBOT_READY_HOLD_TICKS:
                observer.scene_ready_evidence.reset()
            path = artifacts / f"tick_{environment.step_count:03d}_overlay.png" if environment.step_count in FRAME_TICKS else None
            record, previous_pose = _sample(
                environment, base, observer, body_id, previous_pose, phase="ROBOT_READY",
                trigger_tick=None, selected_frame=path,
            )
            trace.append(record)
            if path:
                saved.append(path)
        robot_ready_tick = int(environment.step_count)

        for _scene_hold in range(SCENE_READY_MAX_HOLD_TICKS):
            _one_hold(environment, base, controller, reset=False, workspace=workspace)
            observer.observe(environment)
            if observer.scene_ready and trigger_tick is None:
                trigger_tick = int(environment.step_count)
            tick = int(environment.step_count)
            should_save = tick in FRAME_TICKS or tick == trigger_tick
            path = artifacts / f"tick_{tick:03d}_overlay.png" if should_save else None
            record, previous_pose = _sample(
                environment, base, observer, body_id, previous_pose, phase="SCENE_READY_WAIT",
                trigger_tick=trigger_tick, selected_frame=path,
            )
            trace.append(record)
            if path:
                saved.append(path)
            if trigger_tick is not None:
                break

        if trigger_tick is not None:
            for diagnostic_index in range(1, POST_READY_DIAGNOSTIC_HOLDS + 1):
                _one_hold(environment, base, controller, reset=False, workspace=workspace)
                observer.observe(environment)
                tick = int(environment.step_count)
                path = artifacts / f"tick_{tick:03d}_overlay.png"
                record, previous_pose = _sample(
                    environment, base, observer, body_id, previous_pose,
                    phase=f"POST_READY_DIAGNOSTIC_{diagnostic_index}",
                    trigger_tick=trigger_tick, selected_frame=path,
                )
                trace.append(record)
                saved.append(path)

        scene_holds = max(0, int(environment.step_count) - robot_ready_tick - (
            POST_READY_DIAGNOSTIC_HOLDS if trigger_tick is not None else 0
        ))
        status = scene_ready_status(
            ready=trigger_tick is not None,
            hold_ticks=scene_holds,
            max_hold_ticks=SCENE_READY_MAX_HOLD_TICKS,
        )
        sheet = artifacts / "scene_ready_contact_sheet.png"
        _contact_sheet(saved, sheet)
        # Raw curves are saved before any qualitative false-ready assessment.
        result = {
            "status": status,
            "suite": SUITE,
            "task_id": TASK_ID,
            "seed": TASK_SEED,
            "init_state_index": init_state,
            "task_instruction": environment.task_description,
            "target_body_id": body_id,
            "render_resolution": [resolution, resolution],
            "canonical_orientation": "vertical_flip",
            "robot_ready": {"completed": True, "hold_ticks": ROBOT_READY_HOLD_TICKS,
                            "completed_environment_tick": robot_ready_tick},
            "scene_ready": {
                "triggered": trigger_tick is not None,
                "trigger_environment_tick": trigger_tick,
                "hold_ticks_after_robot_ready": scene_holds,
                "timeout": trigger_tick is None,
                "timeout_status": "SCENE_READY_TIMEOUT" if trigger_tick is None else None,
                "max_hold_ticks": SCENE_READY_MAX_HOLD_TICKS,
                "window_observations": SCENE_READY_WINDOW_OBSERVATIONS,
                "max_centroid_shift_px": SCENE_READY_MAX_CENTROID_SHIFT_PX,
                "max_bbox_edge_shift_px": SCENE_READY_MAX_BBOX_EDGE_SHIFT_PX,
                "max_mask_area_change_px": SCENE_READY_MAX_MASK_AREA_CHANGE_PX,
            },
            "diagnostic_post_ready_hold_ticks": POST_READY_DIAGNOSTIC_HOLDS if trigger_tick is not None else 0,
            "false_ready_assessment": "RAW_ORACLE_CURVE_REVIEW_REQUIRED" if trigger_tick is not None else "NOT_APPLICABLE_TIMEOUT",
            "oracle_used_by_runtime": False,
            "qwen_action_count": 0,
            "trace": trace,
            "contact_sheet": str(sheet),
        }
        _write_json(trial_dir / "scene_ready_trace.json", result)
        return result
    finally:
        environment.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/robot_libero_clean_qwen3vl.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "rollouts/runtime_v3_scene_ready_validation"))
    parser.add_argument("--sam3-url", default="http://127.0.0.1:8773/sse")
    parser.add_argument("--sam3-python", default="/root/autodl-tmp/openeta-services/sam3/.venv/bin/python")
    parser.add_argument("--sam3-timeout-s", type=float, default=120.0)
    parser.add_argument("--camera-resolution", type=int, default=512)
    parser.add_argument("--stage", choices=("a", "b"), default="a")
    parser.add_argument("--stage-a-dir", type=Path)
    return parser


def _stage_a_gate(stage_a_dir: Path) -> tuple[dict[str, Any], list[int]]:
    summary_path = stage_a_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Stage A summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("false_ready_review_complete") is not True:
        raise RuntimeError("Stage A raw-curve review is not complete; Stage B remains blocked")
    rows = summary.get("rows", [])
    states = [int(row["init_state_index"]) for row in rows]
    expected_states = list(range(min(6, int(summary.get("available_init_state_count", 0)))))
    if (summary.get("failures") or states != expected_states
            or len(states) > 6 or len(set(states)) != len(states)
            or summary.get("ready_rate") != 1.0 or summary.get("timeout_rate") != 0.0
            or summary.get("false_ready_rate") != 0.0
            or any(row.get("false_ready_assessment") != "NO_FALSE_READY" for row in rows)):
        raise RuntimeError("Stage A did not pass: require every selected state ready with no timeout or false-ready")
    return summary, states


def _run_stage_b(args: argparse.Namespace) -> int:
    if args.stage_a_dir is None:
        raise SystemExit("--stage b requires --stage-a-dir with reviewed Stage A raw curves")
    stage_a_dir = args.stage_a_dir.expanduser().resolve()
    stage_a_summary, init_states = _stage_a_gate(stage_a_dir)
    _configure_local_proxy_bypass(args.sam3_url)
    config = load_yaml(args.config)
    if config.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(config["libero_dir"])
    stage_b_dir = stage_a_dir / "stage_b"
    stage_b_dir.mkdir(parents=False, exist_ok=False)
    _write_json(stage_b_dir / "RUN_SETUP_READY.json", {
        "status": "RUN_SETUP_READY", "stage": "B_ONE_BOUNDED_ALIGNMENT_PER_INIT_STATE",
        "init_states": init_states, "max_alignment_actions": len(init_states),
        "scene_ready_false_ready_rate": stage_a_summary["false_ready_rate"],
        "scene_ready_timeout_rate": stage_a_summary["timeout_rate"],
        "oracle_used_by_runtime": False, "qwen_action_count": 0,
    })
    workspace = (0.02, 0.60)
    sam3 = Sam3Client(url=args.sam3_url, python=args.sam3_python,
                      timeout_s=args.sam3_timeout_s, max_attempts=1)
    from scripts.runtime_v3_object_relative_alignment import _run_trial as run_alignment_trial
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        for init_state in init_states:
            prior_pose: list[float] | None = None
            origin_pose: list[float] | None = None

            def diagnostic_callback(phase: str, environment: Any,
                                    *, _prior: list[float] | None = None) -> dict[str, Any]:
                nonlocal prior_pose, origin_pose
                sample = _oracle_pose(environment, _target_body_id(environment))
                xyz = list(sample["world_position_m"])
                if origin_pose is None:
                    origin_pose = xyz
                delta_previous = None if prior_pose is None else (np.asarray(xyz) - np.asarray(prior_pose)).tolist()
                delta_origin = (np.asarray(xyz) - np.asarray(origin_pose)).tolist()
                prior_pose = xyz
                return {
                    "phase": phase,
                    "environment_step": int(environment.step_count),
                    "target_world_position_m": xyz,
                    "target_z_m": float(xyz[2]),
                    "delta_from_previous_sample_m": delta_previous,
                    "delta_from_pre_action_m": delta_origin,
                    "norm_from_pre_action_m": float(np.linalg.norm(delta_origin)),
                    "diagnostic_only": True,
                    "oracle_used_by_runtime": False,
                }

            try:
                record = run_alignment_trial(
                    init_state_index=init_state,
                    run_dir=stage_b_dir,
                    config=config,
                    sam3=sam3,
                    workspace=workspace,
                    camera_resolution=512,
                    diagnostic_callback=diagnostic_callback,
                )
                oracle = record.get("oracle_target_motion_diagnostic", {})
                samples = oracle.get("tick_samples", [])
                if len(samples) >= 2:
                    before_xyz = samples[0]["target_world_position_m"]
                    after_xyz = samples[-1]["target_world_position_m"]
                    camera = record.get("camera_calibration")
                    projected = {"available": False, "reason": "camera_calibration_missing"}
                    if isinstance(camera, Mapping):
                        calibration = CameraCalibration(
                            name=str(camera["camera"]), width=int(camera["width"]),
                            height=int(camera["height"]), fovy_deg=float(camera["fovy_deg"]),
                            position_world=np.asarray(camera["position_world"], dtype=float),
                            camera_to_world=np.asarray(camera["camera_to_world"], dtype=float),
                            rotation_degrees=int(camera.get("rotation_degrees", 0)),
                            flip=str(camera.get("flip", "none")),
                        )
                        projected = project_world_motion_to_canonical_pixels(
                            before_xyz, after_xyz, calibration, CanonicalImageAdapter(),
                        )
                    pixel_comparison = compare_oracle_and_sam_pixel_motion(
                        projected.get("oracle_pixel_shift") if projected.get("available") else None,
                        record.get("target_centroid_before_px"), record.get("target_centroid_after_px"),
                    )
                    oracle["pre_action_to_action_end_displacement_xyz_m"] = (
                        np.asarray(after_xyz, dtype=float) - np.asarray(before_xyz, dtype=float)
                    ).tolist()
                    oracle["pre_action_to_action_end_displacement_norm_m"] = float(
                        np.linalg.norm(np.asarray(after_xyz, dtype=float) - np.asarray(before_xyz, dtype=float))
                    )
                    oracle["projected_pixel_motion"] = projected
                    oracle["sam_oracle_pixel_comparison"] = pixel_comparison
                    record["oracle_target_motion_diagnostic"] = oracle
                record["post_ready_target_motion_assessment"] = "RAW_CONTROL_WINDOW_RECORDED"
                record["runtime_received_oracle_motion"] = False
                record["qwen_action_count"] = 0
                trial_path = stage_b_dir / f"init_state_{init_state}" / "trial.json"
                _write_json(trial_path, record)
                rows.append(record)
            except Exception as exc:
                failure = {"init_state_index": init_state,
                           "error": f"{type(exc).__name__}: {exc}",
                           "execution_state": "UNKNOWN_IF_FAILURE_OCCURRED_AFTER_AUTHORIZATION"}
                failure_dir = stage_b_dir / f"init_state_{init_state}"
                failure_dir.mkdir(parents=True, exist_ok=True)
                _write_json(failure_dir / "failure.json", failure)
                failures.append(failure)
    finally:
        sam3.close()
    evaluable = [row for row in rows if row.get("error_before_frozen_px") is not None
                 and row.get("error_after_frozen_px") is not None]
    improved = [row for row in evaluable if row.get("actual_improvement_px") is not None
                and row["actual_improvement_px"] > 0]
    summary = {
        "status": "STAGE_B_RAW_CURVES_READY_FOR_REVIEW" if not failures and len(rows) == len(init_states) else "STAGE_B_PARTIAL",
        "stage_a_summary": str(stage_a_dir / "summary.json"),
        "suite": SUITE, "task_id": TASK_ID, "seed": TASK_SEED,
        "init_states": init_states,
        "bounded_alignment_actions": sum(int(row.get("runner_actions", 0)) for row in rows),
        "maximum_bounded_alignment_actions": len(init_states),
        "alignment_evaluable_trials_before_motion_review": len(evaluable),
        "raw_alignment_improvement_rate_before_motion_review": len(improved) / len(evaluable) if evaluable else None,
        "valid_stable_trials": None,
        "alignment_improvement_rate": None,
        "mean_actual_improvement_px": None,
        "rows": rows,
        "failures": failures,
        "qwen_action_count": 0,
        "oracle_used_by_runtime": False,
    }
    _write_json(stage_b_dir / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))
    print(f"STAGE_B_DIR={stage_b_dir}")
    return 0 if summary["status"] == "STAGE_B_RAW_CURVES_READY_FOR_REVIEW" else 1


def main() -> int:
    args = _parser().parse_args()
    if args.stage == "b":
        if args.camera_resolution != 512:
            raise SystemExit("this validation is fixed to the existing 512x512 Runtime V3 path")
        return _run_stage_b(args)
    if args.camera_resolution != 512:
        raise SystemExit("this validation is fixed to the existing 512x512 Runtime V3 path")
    _configure_local_proxy_bypass(args.sam3_url)
    config = load_yaml(args.config)
    if config.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(config["libero_dir"])
    workspace_cfg = config.get("workspace_z_bounds_m", {})
    workspace = (float(workspace_cfg.get("min", 0.02)), float(workspace_cfg.get("max", 0.60)))
    run_dir = _new_run_dir(Path(args.output_dir).expanduser())
    _write_json(run_dir / "RUN_SETUP_READY.json", {
        "status": "RUN_SETUP_READY", "stage": "A_INITIALIZATION_ONLY",
        "trial_output_directory_writable": True, "alignment_actions_authorized": False,
        "branch": "runtime-v3", "camera_resolution": 512, "seed": TASK_SEED,
    })
    # Probe the task's real init-state inventory, then run the first six in order.
    inventory = LiberoEnvironmentAdapter.create(
        suite_name=SUITE, task_id=TASK_ID, init_state_index=0, seed=TASK_SEED,
        camera_height=512, camera_width=512, horizon=128,
    )
    try:
        count = len(inventory.handle.init_states)
    finally:
        inventory.close()
    init_states = list(range(min(6, count)))
    if not init_states:
        raise RuntimeError("LIBERO_OBJECT task 2 exposes no valid init states")
    sam3 = Sam3Client(url=args.sam3_url, python=args.sam3_python,
                      timeout_s=args.sam3_timeout_s)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        for init_state in init_states:
            try:
                rows.append(_run_stage_a_trial(
                    init_state=init_state, run_dir=run_dir, config=config, sam3=sam3,
                    workspace=workspace, resolution=512,
                ))
            except Exception as exc:
                failure_dir = run_dir / f"init_state_{init_state}"
                failure_dir.mkdir(parents=True, exist_ok=True)
                failure = {"init_state_index": init_state, "error": f"{type(exc).__name__}: {exc}"}
                _write_json(failure_dir / "failure.json", failure)
                failures.append(failure)
    finally:
        sam3.close()
    ready_rows = [row for row in rows if row["scene_ready"]["triggered"]]
    timeout_rows = [row for row in rows if row["scene_ready"]["timeout"]]
    triggers = [row["scene_ready"]["trigger_environment_tick"] for row in ready_rows]
    summary = {
        "status": "STAGE_A_RAW_CURVES_READY_FOR_REVIEW" if len(rows) == len(init_states) and not failures else "STAGE_A_INCOMPLETE",
        "branch": "runtime-v3",
        "starting_commit": "218d2e0e30e4bed5854fef175dd8955ff88760a3",
        "suite": SUITE,
        "task_id": TASK_ID,
        "seed": TASK_SEED,
        "target_phrase": TARGET_PHRASE,
        "available_init_state_count": count,
        "init_states_attempted": init_states,
        "stage_a_trial_count": len(rows),
        "ready_rate": len(ready_rows) / len(rows) if rows else None,
        "timeout_rate": len(timeout_rows) / len(rows) if rows else None,
        "trigger_tick_distribution": triggers,
        "mean_trigger_tick": float(np.mean(triggers)) if triggers else None,
        "false_ready_rate": None,
        "false_ready_assessment": "REVIEW_RAW_ORACLE_CURVES_BEFORE_ASSIGNING",
        "thresholds_modified": False,
        "alignment_stage_started": False,
        "oracle_used_by_runtime": False,
        "qwen_action_count": 0,
        "rows": rows,
        "failures": failures,
        "raw_curve_review_required_before_stage_b": True,
    }
    curve_plot = run_dir / "oracle_motion_curves.png"
    _save_oracle_curve_plot(rows, curve_plot)
    summary["oracle_motion_curve_plot"] = str(curve_plot)
    _write_json(run_dir / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))
    print(f"RUN_DIR={run_dir}")
    return 0 if len(rows) == len(init_states) and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

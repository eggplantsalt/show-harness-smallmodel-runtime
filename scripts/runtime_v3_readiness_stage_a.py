#!/usr/bin/env python3
"""Formal HOLD-only Stage A for the frozen M3.7 readiness task set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping
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
from core.runtime_v3.object_relative import ObjectRelativePerceptionObserver
from core.runtime_v3.scene_initialization import run_scene_ready_holds
from core.runtime_v3.temporal_calibration import run_v3_tick
from interpreters.libero_atomic_controller import LiberoAtomicController


SUITE = "LIBERO_OBJECT"
ROBOT_READY_HOLD_TICKS = 4
POST_READY_ORACLE_HOLDS = 3
MATERIAL_MOTION_THRESHOLD_M = 0.001
TASKS = {
    0: ("Pick the alphabet soup and place it in the basket", "alphabet soup"),
    2: ("Pick the salad dressing and place it in the basket", "salad dressing"),
    6: ("Pick the butter and place it in the basket", "butter"),
    7: ("Pick the milk and place it in the basket", "milk"),
    1: ("Pick the cream cheese and place it in the basket", "cream cheese"),
    8: ("Pick the chocolate pudding and place it in the basket", "chocolate pudding"),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_record") and callable(value.to_record):
        return _jsonable(value.to_record())
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable({name: getattr(value, name) for name in value.__dataclass_fields__})
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(value), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configure_proxy(url: str) -> None:
    if urlparse(url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        return
    values = [item.strip() for key in ("NO_PROXY", "no_proxy")
              for item in os.environ.get(key, "").split(",") if item.strip()]
    values.extend(("127.0.0.1", "localhost", "::1"))
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(dict.fromkeys(values))


def _body_name(environment: LiberoEnvironmentAdapter) -> str:
    text = Path(environment.handle.bddl_file).read_text(encoding="utf-8")
    match = re.search(r"\(:obj_of_interest\s+([^\s()]+)", text, flags=re.IGNORECASE)
    if match is None:
        raise RuntimeError("BDDL has no :obj_of_interest entry")
    return f"{match.group(1)}_main"


def _body_id(environment: LiberoEnvironmentAdapter, name: str) -> int:
    model = environment.env.sim.model
    if callable(getattr(model, "body_name2id", None)):
        return int(model.body_name2id(name))
    return int(model.body(name).id)


def _target_pose(environment: LiberoEnvironmentAdapter, body_id: int) -> list[float]:
    return np.asarray(environment.env.sim.data.xpos[body_id], dtype=float).reshape(3).tolist()


def _hold(environment, observer, controller, *, task_id: int, workspace, reset=False) -> None:
    before = int(environment.step_count)
    outcome = run_v3_tick(
        environment, observer, controller, task_id=f"{SUITE}:{task_id}",
        token=None, direction_unit=None, commanded_step_m=0.005,
        reset=reset, workspace_z_bounds_m=workspace,
    )
    if (outcome.get("actions") != 1 or outcome.get("approved_action") != "CALIBRATION_HOLD"
            or not outcome.get("backend_execution")
            or int(environment.step_count) != before + 1):
        raise RuntimeError(f"Stage A expected one approved zero-translation HOLD: {outcome}")


def _contact_sheet(observer, output: Path, *, title: str) -> str:
    selected = observer.perception_history
    ticks = sorted(set((0, 1, 2, 3, 4, 10, 20, 40, len(selected) - 1)))
    ticks = [value for value in ticks if 0 <= value < len(selected)]
    width, height, label_h = 320, 320, 58
    columns = 3
    rows = (len(ticks) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * (height + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, tick in enumerate(ticks):
        sample = selected[tick]
        image = Image.fromarray(sample["image"], mode="RGB")
        segmentation = sample["segmentation"]
        mask = segmentation.mask
        if mask is not None:
            rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            rgba[mask] = (255, 35, 35, 110)
            image = Image.alpha_composite(image.convert("RGBA"), Image.fromarray(rgba)).convert("RGB")
        draw_image = ImageDraw.Draw(image)
        box = segmentation.bbox_xyxy
        centroid = segmentation.centroid_px
        if box is not None:
            draw_image.rectangle(tuple(box), outline=(255, 240, 0), width=3)
        if centroid is not None:
            x, y = centroid
            draw_image.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(0, 255, 255), width=3)
        image.thumbnail((width, height))
        x0, y0 = (index % columns) * width, (index // columns) * (height + label_h)
        sheet.paste(image, (x0, y0))
        evidence = sample.get("entity_observation_evidence") or {}
        draw.text((x0 + 3, y0 + height + 2),
                  f"{title} | observation {tick} | {segmentation.identity_status}\n"
                  f"candidate={len(segmentation.candidates)} selected={segmentation.selected_candidate_id} "
                  f"score={segmentation.quality_score} "
                  f"motion={sample.get('scene_motion_score')} "
                  f"ready={sample.get('scene_motion_ready')}/{sample.get('entity_observation_ready')}",
                  fill=(0, 0, 0))
        if evidence.get("last_visual_interval"):
            draw.text((x0 + 3, y0 + height + 32), str(evidence["last_visual_interval"]), fill=(0, 0, 0))
    sheet_path = output / "contact_sheets" / f"{title.replace(' ', '_')}.png"
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path)
    return str(sheet_path)


def _run_episode(*, task_id: int, init_state: int, phrase: str, config: Mapping[str, Any],
                 sam3: Sam3Client, output: Path, resolution: int) -> dict[str, Any]:
    workspace_cfg = config.get("workspace_z_bounds_m", {})
    workspace = (float(workspace_cfg.get("min", 0.02)), float(workspace_cfg.get("max", 0.60)))
    environment = LiberoEnvironmentAdapter.create(
        suite_name=SUITE, task_id=task_id, init_state_index=init_state, seed=0,
        camera_height=resolution, camera_width=resolution, horizon=256,
    )
    base_observer = LiberoObservationAdapter(
        max_eef_z_m=workspace[1], min_eef_z_m=workspace[0], safe_lift_step_m=0.005,
    )
    controller = LiberoAtomicController(
        move_vectors=config["move_vectors"], step_m=0.005, sim_steps_per_decision=1,
        position_scale_m=float(config.get("position_scale_m", 0.05)),
    )
    episode_dir = output / f"task_{task_id}" / f"init_state_{init_state}"
    episode_dir.mkdir(parents=True, exist_ok=False)
    oracle_samples: list[dict[str, Any]] = []
    robot_trace: list[dict[str, Any]] = []
    try:
        body_name = _body_name(environment)
        body_id = _body_id(environment, body_name)
        reset = True
        for index in range(ROBOT_READY_HOLD_TICKS):
            _hold(environment, base_observer, controller, task_id=task_id,
                  workspace=workspace, reset=reset)
            reset = False
            base_observer.observe(environment)
            robot_trace.append({
                "environment_step": int(environment.step_count),
                "target_world_position_m": _target_pose(environment, body_id),
                "hold_index": index + 1,
            })

        observer = ObjectRelativePerceptionObserver(
            base_observer, sam3, target_phrase=phrase,
            move_vectors=config["move_vectors"], scene_ready_required=True,
        )
        observe = observer.observe

        def observe_with_oracle_log(current_environment):
            observation = observe(current_environment)
            # This post-observation recorder is not an argument to Runtime and
            # writes only a separate diagnostic trace for subsequent grading.
            oracle_samples.append({
                "environment_step": int(current_environment.step_count),
                "target_world_position_m": _target_pose(current_environment, body_id),
                "diagnostic_only_oracle": True,
            })
            return observation

        observer.observe = observe_with_oracle_log
        ready_result = run_scene_ready_holds(
            environment, observer, controller, task_id=f"{SUITE}:{task_id}",
            commanded_step_m=0.005, workspace_z_bounds_m=workspace, max_hold_ticks=40,
        )
        ready_step = int(environment.step_count) if ready_result["ready"] else None
        for _ in range(POST_READY_ORACLE_HOLDS):
            if not ready_result["ready"]:
                break
            _hold(environment, base_observer, controller, task_id=task_id, workspace=workspace)
            observer.observe(environment)

        poses = np.asarray([row["target_world_position_m"] for row in oracle_samples], dtype=float)
        speeds = np.linalg.norm(np.diff(poses, axis=0), axis=1) if len(poses) > 1 else np.asarray([])
        readiness_samples = ready_result.get("samples", [])
        ready_index = next((index for index, row in enumerate(readiness_samples)
                            if row.get("scene_ready")), None)
        # Oracle uses only a preregistered 1 mm/tick diagnostic materiality
        # level. It never changes or retries the Runtime threshold.
        false_ready = bool(
            ready_result["ready"] and ready_index is not None
            and np.any(speeds[ready_index:ready_index + POST_READY_ORACLE_HOLDS]
                       > MATERIAL_MOTION_THRESHOLD_M)
        )
        settled_tail = bool(speeds.size >= 3 and np.all(speeds[-3:] <= MATERIAL_MOTION_THRESHOLD_M))
        false_not_ready = bool(not ready_result["ready"] and settled_tail)
        oracle_diagnostic = {
            "pose_samples": oracle_samples,
            "per_observation_translation_m": speeds.tolist(),
            "material_motion_threshold_m_per_observation": MATERIAL_MOTION_THRESHOLD_M,
            "settled_at_tail": settled_tail,
            "false_ready": false_ready,
            "false_not_ready": false_not_ready,
            "oracle_used_by_runtime": False,
        }
        contact_sheet = None
        if init_state == 0 and task_id in {0, 6, 1, 8}:
            contact_sheet = _contact_sheet(observer, episode_dir, title=f"task_{task_id}_{phrase}")
        record = {
            "task_id": task_id,
            "task_phrase": phrase,
            "init_state_index": init_state,
            "seed": 0,
            "robot_ready": {"completed": True, "hold_ticks": ROBOT_READY_HOLD_TICKS,
                            "trace": robot_trace},
            "scene_motion_ready": ready_result["scene_motion_ready"],
            "entity_observation_ready": ready_result["entity_observation_ready"],
            "grounding_success": ready_result["grounding_success"],
            "identity_valid": ready_result["identity_valid"],
            "reference_valid": ready_result["reference_valid"],
            "ticks_to_ready": ready_result["ticks_to_ready"],
            "hold_ticks": ready_result["hold_ticks"],
            "termination_reason": ready_result["termination_reason"],
            "ready": ready_result["ready"],
            "readiness_trace": ready_result["samples"],
            "runtime_evidence": {
                "scene_motion": observer.scene_motion_evidence.to_record(),
                "entity_observation": observer.entity_observation_evidence.to_record(),
                "grounding_query": observer.grounding_query,
                "raw_semantic_phrase_preserved": observer.entity_spec.semantic_phrase,
            },
            "oracle_diagnostic_only": oracle_diagnostic,
            "alignment_actions": 0,
            "contact_sheet": contact_sheet,
            "ready_environment_step": ready_step,
        }
        _write_json(episode_dir / "readiness_episode.json", record)
        return record
    finally:
        environment.close()


def _summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for episode in episodes:
        grouped.setdefault(int(episode["task_id"]), []).append(episode)
    tasks = {}
    for task_id, rows in grouped.items():
        ready = [int(row["ready"]) for row in rows]
        tasks[str(task_id)] = {
            "episodes": len(rows),
            "robot_ready": sum(bool(row["robot_ready"]["completed"]) for row in rows),
            "scene_motion_ready": sum(bool(row["scene_motion_ready"]) for row in rows),
            "entity_observation_ready": sum(bool(row["entity_observation_ready"]) for row in rows),
            "grounding_success": sum(bool(row["grounding_success"]) for row in rows),
            "identity_valid": sum(bool(row["identity_valid"]) for row in rows),
            "reference_valid": sum(bool(row["reference_valid"]) for row in rows),
            "ready": sum(ready),
            "timeouts": len(rows) - sum(ready),
            "ticks_to_ready": [row["ticks_to_ready"] for row in rows],
            "termination_reasons": dict(Counter(row["termination_reason"] for row in rows)),
            "false_ready_count": sum(bool(row["oracle_diagnostic_only"]["false_ready"]) for row in rows),
            "false_not_ready_count": sum(bool(row["oracle_diagnostic_only"]["false_not_ready"])
                                          for row in rows),
        }
    return {
        "phase": "M3.7_STAGE_A_READINESS_ONLY",
        "tasks": tasks,
        "episode_count": len(episodes),
        "alignment_actions": sum(int(row["alignment_actions"]) for row in episodes),
        "false_ready_count": sum(bool(row["oracle_diagnostic_only"]["false_ready"]) for row in episodes),
        "false_not_ready_count": sum(bool(row["oracle_diagnostic_only"]["false_not_ready"])
                                     for row in episodes),
        "oracle_used_by_runtime": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "experiments/runtime_v3/m3_7/readiness_heldout_manifest.json"))
    parser.add_argument("--config", default=str(ROOT / "configs/robot_libero_clean_qwen3vl.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "rollouts/runtime_v3_m3_7_readiness_stage_a"))
    parser.add_argument("--sam3-url", default="http://127.0.0.1:8773/sse")
    parser.add_argument("--sam3-python", default="/root/autodl-tmp/openeta-services/sam3/.venv/bin/python")
    parser.add_argument("--sam3-timeout-s", type=float, default=120.0)
    parser.add_argument("--camera-resolution", type=int, default=512)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_FORMAL_POSTFIX_EVALUATION":
        raise RuntimeError("held-out selection manifest is not frozen")
    if _git("status", "--porcelain"):
        raise RuntimeError("Stage A requires the frozen Runtime refactor commit and a clean worktree")
    start_head = _git("rev-parse", "HEAD")
    config_path = Path(args.config).resolve()
    config = load_yaml(str(config_path))
    if config.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(config["libero_dir"])
    _configure_proxy(args.sam3_url)
    sam3 = Sam3Client(url=args.sam3_url, python=args.sam3_python,
                      timeout_s=args.sam3_timeout_s, max_attempts=1)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    episodes = []
    try:
        for task_id in (0, 2, 6, 7, 1, 8):
            instruction, phrase = TASKS[task_id]
            if task_id in {1, 8}:
                frozen = next(row for row in manifest["heldout_tasks"]
                              if int(row["task_id"]) == task_id)
                if (frozen["instruction"] != instruction
                        or phrase not in frozen["instruction"].casefold()):
                    raise RuntimeError("formal task binding differs from frozen held-out manifest")
            for init_state in manifest["selected_init_states"]:
                episode = _run_episode(
                    task_id=task_id, init_state=int(init_state), phrase=phrase,
                    config=config, sam3=sam3, output=output,
                    resolution=args.camera_resolution,
                )
                episodes.append(episode)
                print(f"task={task_id} init={init_state} ready={episode['ready']} "
                      f"motion={episode['scene_motion_ready']} entity={episode['entity_observation_ready']} "
                      f"ground={episode['grounding_success']} identity={episode['identity_valid']} "
                      f"reference={episode['reference_valid']} reason={episode['termination_reason']}",
                      flush=True)
    finally:
        sam3.close()
    if _git("rev-parse", "HEAD") != start_head:
        raise RuntimeError("Runtime HEAD changed during formal Stage A")
    result = _summary(episodes)
    result.update({
        "status": "COMPLETED",
        "suite": SUITE,
        "init_states": list(manifest["selected_init_states"]),
        "seed": int(manifest["seed"]),
        "heldout_manifest": str(manifest_path),
        "heldout_manifest_sha256": _sha256(manifest_path),
        "starting_commit": start_head,
        "final_commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "runtime_core_changed_between_episodes": False,
        "align_changed_between_episodes": False,
        "alignment_actions": 0,
        "physical_config_sha256": _sha256(config_path),
        "formal_privileged_inputs": {
            "target_pose": False, "simulator_depth": False,
            "ground_truth_contact": False, "task_success": False,
        },
        "episodes": episodes,
    })
    _write_json(output / "summary.json", result)
    print(f"STAGE_A_SUMMARY={output / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

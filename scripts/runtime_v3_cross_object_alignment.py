#!/usr/bin/env python3
"""Run the frozen reference-binding M3.6 cross-object ALIGN gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime_v3.task_spec import ReferenceTaskCompiler
from scripts.runtime_v3_multiscale_alignment import (
    DIRECTION_ORDER, MAX_ALIGNMENT_STEPS,
)
from scripts.runtime_v3_multiscale_alignment import (
    new_run_dir, run_stage_b_episode, write_json,
)


MANIFEST = ROOT / "experiments/runtime_v3/cross_object_align_manifest.json"
SCALE_EVIDENCE = ROOT / (
    "rollouts/runtime_v3_multiscale_alignment/"
    "run_20261002T133108Z_c21b2a84/stage_a/calibration_summary.json"
)
PHYSICAL_FILES = (
    "core/runtime_v3/object_relative.py", "core/runtime_v3/micro_motion.py",
    "core/runtime_v3/options.py", "core/runtime_v3/arbiter.py",
    "core/runtime_v3/executor.py", "core/runtime_v3/effects.py",
    "scripts/runtime_v3_multiscale_alignment.py",
    "scripts/runtime_v3_cross_object_alignment.py",
)


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True,
                           capture_output=True, text=True)
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_signature() -> dict[str, str]:
    result = {}
    for path in sorted((ROOT / "core/runtime_v3").rglob("*.py")):
        result[path.relative_to(ROOT).as_posix()] = _sha256(path)
    for relative in PHYSICAL_FILES:
        path = ROOT / relative
        result[relative] = _sha256(path)
    return result


def _configure_local_proxy(url: str) -> None:
    if urlparse(url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        return
    values = [value.strip() for key in ("NO_PROXY", "no_proxy")
              for value in os.environ.get(key, "").split(",") if value.strip()]
    value = ",".join(dict.fromkeys((*values, "127.0.0.1", "localhost", "::1")))
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = value


def _failure_layer(reason: str | None) -> str | None:
    if not reason:
        return None
    value = str(reason).upper()
    if value in {"MAX_ALIGNMENT_STEPS", "TARGET_REACHED", "STEP_LIMIT", "COMPLETED", "DONE"}:
        return None
    compact = "".join(character for character in value if character.isalnum())
    if "INSTRUCTION" in value or "SEMANTIC" in value:
        return "SEMANTIC_BINDING_FAILURE"
    if "SCENE_READY" in value or "SCENE_NOT_READY" in value or "SCENEREADY" in compact:
        return "SCENE_NOT_READY"
    if "IDENTITY" in value:
        return "IDENTITY_FAILURE"
    if "REFERENCE" in value:
        return "REFERENCE_FAILURE"
    if "VISIBILITY" in value or "TARGET_VISIBLE" in value or "SAM" in value:
        return "PERCEPTION_FAILURE"
    if "NO_POSITIVE_OPTION" in value or "NO_VALID" in value:
        return "NO_VALID_PHYSICAL_OPTION"
    if "ARBITER" in value or "AUTHORIZATION" in value:
        return "ARBITER_AUTHORIZATION_FAILURE"
    if "EXECUTION" in value or "EXECUTOR" in value:
        return "PHYSICAL_EXECUTION_FAILURE"
    if "EFFECT_NOT_IMPROVED" in value or "EFFECT_NOT_OBSERVED" in value or "EFFECT_VERIFICATION" in value:
        return "EFFECT_VERIFICATION_FAILURE"
    return "UNCLASSIFIED_FAILURE"


def _episode_layers(episode: Mapping[str, Any]) -> list[str]:
    layers = []
    if episode.get("failure_layer"):
        layers.append(str(episode["failure_layer"]))
    for step in episode.get("alignment_steps", []):
        if step.get("failure_layer"):
            layers.append(str(step["failure_layer"]))
        reason = step.get("termination_reason")
        layer = _failure_layer(reason)
        if layer:
            layers.append(layer)
    fallback_reason = episode.get("error")
    if fallback_reason is None and not episode.get("failure_layer"):
        fallback_reason = episode.get("termination_reason")
    layer = _failure_layer(fallback_reason)
    if layer:
        layers.append(layer)
    return list(dict.fromkeys(layers))


def _task_metrics(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    steps = [step for episode in episodes for step in episode.get("alignment_steps", [])]
    executed = [step for step in steps if step.get("executed")]
    improvements = [float(step["actual_improvement_px"]) for step in executed
                    if step.get("actual_improvement_px") is not None]
    errors = [episode for episode in episodes if episode.get("initial_error_px") is not None
              and episode.get("final_error_px") is not None]
    normalized = [
        (float(item["initial_error_px"]) - float(item["final_error_px"])) /
        float(item["initial_error_px"])
        for item in errors if float(item["initial_error_px"]) > 0.0
    ]
    layers = Counter(layer for episode in episodes for layer in _episode_layers(episode))
    grounded = [episode for episode in episodes if episode.get("alignment_steps")]
    identity = [episode for episode in episodes if episode.get("alignment_steps")]
    reference = [episode for episode in episodes if episode.get("alignment_steps")]
    scene_attempts = [episode for episode in episodes
                      if episode.get("scene_ready_initialization")
                      or episode.get("failure_layer") == "SCENE_NOT_READY"]
    scene_ready_count = sum(bool(episode.get("scene_ready_initialization", {}).get("ready"))
                            for episode in scene_attempts)
    net_positive = [episode for episode in episodes
                    if episode.get("initial_error_px") is not None
                    and episode.get("final_error_px") is not None
                    and float(episode["initial_error_px"]) > float(episode["final_error_px"])]
    ticks = sum(int(episode.get("control_ticks", 0)) for episode in episodes)
    directions = Counter(str(step.get("direction")) for step in executed if step.get("direction"))
    scales = Counter(f"{int(round(float(step['scale_mm'])))}mm"
                     for step in executed if step.get("scale_mm") is not None)
    episode_values = [float(item["initial_error_px"]) - float(item["final_error_px"])
                      for item in errors]
    positive_effects = sum(value > 0.0 for value in improvements)
    count = len(episodes)
    grounding_rate = (sum(any(bool(step.get("target_visible")) for step in episode.get("alignment_steps", []))
                          for episode in grounded) / len(grounded)) if grounded else None
    identity_rate = (sum(any(bool(step.get("identity_valid")) for step in episode.get("alignment_steps", []))
                         for episode in identity) / len(identity)) if identity else None
    reference_rate = (sum(any(bool(step.get("reference_valid")) for step in episode.get("alignment_steps", []))
                           for episode in reference) / len(reference)) if reference else None
    scene_rate = scene_ready_count / len(scene_attempts) if scene_attempts else None
    if (len(net_positive) == count and count and grounding_rate == identity_rate == reference_rate == scene_rate == 1.0):
        judgment = "PASS"
    elif net_positive:
        judgment = "PARTIAL"
    else:
        judgment = "FAIL"
    return {
        "episodes": count,
        "grounding_success_rate": grounding_rate,
        "grounding_observed_episodes": len(grounded),
        "identity_success_rate": identity_rate,
        "identity_observed_episodes": len(identity),
        "reference_validity_rate": reference_rate,
        "reference_observed_episodes": len(reference),
        "scene_ready_success_rate": scene_rate,
        "scene_ready_attempts": len(scene_attempts),
        "align_actions_executed": len(executed),
        "positive_effect_fraction": positive_effects / len(improvements) if improvements else None,
        "monotonic_episodes": sum(bool(item.get("all_steps_positive")) for item in episodes),
        "initial_error_px_mean": float(np.mean([float(item["initial_error_px"]) for item in errors])) if errors else None,
        "final_error_px_mean": float(np.mean([float(item["final_error_px"]) for item in errors])) if errors else None,
        "normalized_error_reduction_mean": float(np.mean(normalized)) if normalized else None,
        "net_improving_episodes": len(net_positive),
        "mean_improvement_per_semantic_step_px": float(np.mean(improvements)) if improvements else None,
        "control_ticks": ticks,
        "mean_control_ticks_per_semantic_step": ticks / len(executed) if executed else None,
        "direction_distribution": dict(directions),
        "scale_distribution": dict(scales),
        "failure_layers": dict(layers),
        "judgment": judgment,
    }


def _write_cross_task_montage(episodes: Sequence[Mapping[str, Any]], run_dir: Path) -> str | None:
    panels = []
    for episode in episodes:
        task_id = episode.get("task_id")
        state = episode.get("init_state_index")
        contact_sheet = episode.get("contact_sheet")
        if contact_sheet and Path(contact_sheet).is_file():
            panels.append((f"task {task_id} / state {state}", Path(contact_sheet)))
        for step in episode.get("alignment_steps", []):
            paths = step.get("visual_artifacts", {})
            before = paths.get("before", {}).get("alignment_overlay")
            after = paths.get("after", {}).get("alignment_overlay")
            if before and Path(before).is_file():
                panels.append((f"task {task_id} / state {state} / step {step.get('semantic_step')} before",
                               Path(before)))
            if after and Path(after).is_file():
                panels.append((f"task {task_id} / state {state} / step {step.get('semantic_step')} after",
                               Path(after)))
    if not panels:
        return None
    thumb_w, thumb_h = 320, 280
    columns = 4
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * thumb_h), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(panels):
        with Image.open(path) as image:
            tile = image.convert("RGB")
            tile.thumbnail((thumb_w - 10, thumb_h - 32))
            x0 = (index % columns) * thumb_w
            y0 = (index // columns) * thumb_h
            sheet.paste(tile, (x0 + (thumb_w - tile.width) // 2, y0 + 25))
        draw.text(((index % columns) * thumb_w + 5,
                   (index // columns) * thumb_h + 5), label, fill=(245, 245, 245))
    output = run_dir / "cross_task_contact_sheet.png"
    sheet.save(output)
    return str(output)


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if not manifest.get("selection_frozen_before_rollout"):
        raise RuntimeError("task manifest must be frozen before formal rollout")
    if _git("status", "--porcelain"):
        raise RuntimeError("formal rollout requires a clean worktree")
    starting_head = _git("rev-parse", "HEAD")
    frozen_branch = _git("branch", "--show-current")
    source_signature = _runtime_signature()
    manifest_path = Path(args.manifest).resolve()
    manifest_hash = _sha256(manifest_path)
    config_path = Path(args.config).resolve()
    config_hash = _sha256(config_path)

    calibration = json.loads(Path(args.scale_evidence).read_text(encoding="utf-8"))
    verified_scales = []
    contracts = {}
    for scale_m, max_ticks in zip((0.003, 0.006, 0.009), (5, 7, 10)):
        label = f"{int(scale_m * 1000)}mm"
        if calibration.get("scale_decisions", {}).get(label, {}).get("status") == "VERIFIED":
            verified_scales.append(scale_m)
            contracts[scale_m] = {"verified": True, "max_ticks": max_ticks}
    if tuple(verified_scales) != (0.003, 0.006, 0.009):
        raise RuntimeError("the frozen anchor evidence does not verify the existing 3/6/9mm contracts")

    from core.capabilities.sam3_client import Sam3Client
    from core.config import load_yaml

    config = load_yaml(str(config_path))
    if config.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(config["libero_dir"])
    _configure_local_proxy(args.sam3_url)
    sam3 = Sam3Client(url=args.sam3_url, python=args.sam3_python,
                      timeout_s=args.sam3_timeout_s, max_attempts=1)
    compiler = ReferenceTaskCompiler()
    run_dir = new_run_dir(args.output_dir)
    all_episodes = []
    task_records = []
    workspace = (0.02, 0.60)
    try:
        for task_entry in manifest["tasks"]:
            if _sha256(manifest_path) != manifest_hash:
                raise RuntimeError("frozen task manifest changed during formal rollout")
            task_id = int(task_entry["task_id"])
            task_spec = compiler.compile({
                "instruction": task_entry["instruction"],
                "entities": [{
                    "key": task_entry["entity_key"],
                    "semantic_phrase": task_entry["reference_entity_phrase"],
                    "role": task_entry["role"],
                }],
                "focus_entity_key": task_entry["entity_key"],
                "goal_kind": manifest["goal_kind"],
            })
            task_dir = run_dir / f"task_{task_id}"
            task_dir.mkdir(parents=True, exist_ok=False)
            head_before = _git("rev-parse", "HEAD")
            signature_before = _runtime_signature()
            config_hash_before = _sha256(config_path)
            episodes = []

            def run_indices(indices: Sequence[int]) -> None:
                for init_state in indices:
                    try:
                        episode = run_stage_b_episode(
                            init_state=int(init_state), run_dir=task_dir, config=config,
                            sam3=sam3, workspace=workspace,
                            camera_resolution=int(args.camera_resolution),
                            verified_scales=verified_scales, contracts=contracts,
                            suite_name=str(manifest["suite"]), task_id=task_id,
                            entity_spec=task_spec.focus_entity, diagnostic_oracle=False,
                            seed=int(manifest["seed"]),
                        )
                        episode["failure_layer"] = None
                    except Exception as exc:
                        episode = {
                            "init_state_index": int(init_state), "suite": manifest["suite"],
                            "task_id": task_id, "seed": int(manifest["seed"]),
                            "task_instruction": task_spec.instruction,
                            "entity_key": task_spec.focus_entity.key,
                            "entity_phrase": task_spec.focus_entity.semantic_phrase,
                            "entity_role": task_spec.focus_entity.role,
                            "status": "FAILED", "termination_reason": "TRIAL_FAILED",
                            "error": f"{type(exc).__name__}: {exc}",
                            "failure_layer": _failure_layer(str(exc)),
                            "alignment_steps": [], "executed_alignment_steps": 0,
                            "oracle_used_by_runtime": False,
                        }
                        failure_dir = task_dir / "stage_b" / f"init_state_{int(init_state)}"
                        failure_dir.mkdir(parents=True, exist_ok=True)
                        write_json(failure_dir / "failure.json", episode)
                    episode["task_id"] = task_id
                    episode["entity_phrase"] = task_spec.focus_entity.semantic_phrase
                    episodes.append(episode)
                    all_episodes.append(episode)
                    print(f"task={task_id} init={init_state} status={episode.get('status')} "
                          f"termination={episode.get('termination_reason')}", flush=True)

            initial_states = [int(value) for value in manifest["initial_gate"]["init_states"]]
            run_indices(initial_states)
            initial_metrics = _task_metrics(episodes)
            expanded_states = []
            if (task_id != 2 and len(episodes) == len(initial_states)
                    and initial_metrics["net_improving_episodes"] == len(initial_states)
                    and initial_metrics["grounding_success_rate"] == 1.0
                    and initial_metrics["identity_success_rate"] == 1.0
                    and initial_metrics["reference_validity_rate"] == 1.0
                    and initial_metrics["scene_ready_success_rate"] == 1.0):
                expanded_states = [3, 4, 5]
                run_indices(expanded_states)
            if _git("rev-parse", "HEAD") != head_before:
                raise RuntimeError("HEAD changed while a task's formal rollout was in progress")
            if _runtime_signature() != signature_before or _sha256(config_path) != config_hash_before:
                raise RuntimeError("Runtime source or physical configuration changed between task rollouts")
            task_record = {
                "task_id": task_id, "instruction": task_spec.instruction,
                "entity_key": task_spec.focus_entity.key,
                "entity_phrase": task_spec.focus_entity.semantic_phrase,
                "role": task_spec.focus_entity.role,
                "selection_reason": task_entry["reason_for_inclusion"],
                "episodes": episodes,
                "metrics": _task_metrics(episodes),
                "initial_gate_metrics": initial_metrics,
                "expanded_init_states": expanded_states,
                "commit_before_task": head_before,
                "commit_after_task": _git("rev-parse", "HEAD"),
                "runtime_source_signature_unchanged": True,
                "config_sha256": config_hash_before,
            }
            task_records.append(task_record)
            write_json(task_dir / "task_summary.json", task_record)
            print(f"task={task_id} judgment={task_record['metrics']['judgment']} "
                  f"net_positive={task_record['metrics']['net_improving_episodes']}/"
                  f"{task_record['metrics']['episodes']}", flush=True)
    finally:
        sam3.close()

    if _git("rev-parse", "HEAD") != starting_head:
        raise RuntimeError("HEAD changed during the frozen cross-object rollout")
    source_unchanged = _runtime_signature() == source_signature
    if not source_unchanged:
        raise RuntimeError("Runtime or ALIGN experiment source changed during formal rollout")

    montage = _write_cross_task_montage(all_episodes, run_dir)
    unseen = [row for row in task_records if int(row["task_id"]) != 2]
    overall = ("YES" if unseen and all(row["metrics"]["judgment"] == "PASS" for row in unseen)
               else "PARTIAL" if any(row["metrics"]["net_improving_episodes"] > 0 for row in unseen)
               else "NO")
    audit = {
        "starting_commit": starting_head, "final_commit": _git("rev-parse", "HEAD"),
        "branch": frozen_branch,
        "task_commits": {str(row["task_id"]): {
            "before": row["commit_before_task"], "after": row["commit_after_task"]
        } for row in task_records},
        "same_commit_across_tasks": all(
            row["commit_before_task"] == row["commit_after_task"] == starting_head
            for row in task_records
        ),
        "runtime_core_changed_between_tasks": False,
        "align_contract_changed_between_tasks": False,
        "per_object_thresholds": False,
        "per_task_direction_rules": False,
        "per_task_scale_rules": False,
        "runtime_source_signature_unchanged": source_unchanged,
        "physical_config_sha256": config_hash,
        "scale_evidence_path": str(Path(args.scale_evidence).resolve()),
        "scale_evidence_sha256": _sha256(Path(args.scale_evidence).resolve()),
        "verified_scales_mm": [scale * 1000.0 for scale in verified_scales],
    }
    summary = {
        "phase": "M3.6 Zero-Code-Change Cross-Object Runtime Transfer",
        "status": "COMPLETED", "suite": manifest["suite"], "seed": manifest["seed"],
        "manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": manifest_hash,
        "starting_commit": starting_head, "final_commit": _git("rev-parse", "HEAD"),
        "branch": frozen_branch,
        "physical_contract": {
            "goal_kind": "ALIGN", "directions": list(DIRECTION_ORDER),
            "scales_mm": [scale * 1000 for scale in verified_scales],
            "max_semantic_steps_per_episode": MAX_ALIGNMENT_STEPS,
            "same_selector": "MultiscaleSelector", "same_option_generator": "MultiScaleAlignmentOptionGenerator",
        },
        "tasks": task_records,
        "cross_object_judgment": overall,
        "failure_taxonomy": dict(Counter(
            layer for episode in all_episodes for layer in _episode_layers(episode)
        )),
        "code_change_audit": audit,
        "privileged_information": {
            "formal_runtime_target_pose": False, "formal_runtime_gt_depth": False,
            "formal_runtime_gt_contact": False, "formal_runtime_task_success": False,
        },
        "visual_audit": {"cross_task_contact_sheet": montage,
                         "episode_contact_sheets": [row.get("contact_sheet") for row in all_episodes
                                                     if row.get("contact_sheet")]},
        "qwen_binding": "NOT_RUN_BEFORE_REFERENCE_BINDING_JUDGMENT",
        "rollout_commit_unchanged": audit["same_commit_across_tasks"] and source_unchanged,
    }
    write_json(run_dir / "generalization_code_change_audit.json", audit)
    write_json(run_dir / "summary.json", summary)
    print(f"M3.6 reference-binding {overall}: {run_dir / 'summary.json'}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--scale-evidence", default=str(SCALE_EVIDENCE))
    parser.add_argument("--config", default=str(ROOT / "configs/robot_libero_clean_qwen3vl.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "rollouts/runtime_v3_cross_object_alignment"))
    parser.add_argument("--sam3-url", default="http://127.0.0.1:8773/sse")
    parser.add_argument("--sam3-python", default="/root/autodl-tmp/openeta-services/sam3/.venv/bin/python")
    parser.add_argument("--sam3-timeout-s", type=float, default=120.0)
    parser.add_argument("--camera-resolution", type=int, default=512)
    args = parser.parse_args()
    if args.camera_resolution != 512:
        raise SystemExit("M3.6 uses the existing 512x512 canonical observation path")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

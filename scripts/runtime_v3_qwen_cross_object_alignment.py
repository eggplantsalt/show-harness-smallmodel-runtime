#!/usr/bin/env python3
"""Run the frozen ALIGN contract using previously audited Qwen TaskSpecs."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime_v3.task_spec import ReferenceTaskCompiler
from scripts.runtime_v3_cross_object_alignment import (
    _episode_layers, _runtime_signature, _sha256, _task_metrics,
    _write_cross_task_montage, _failure_layer, DIRECTION_ORDER, MAX_ALIGNMENT_STEPS,
)
from scripts.runtime_v3_multiscale_alignment import (
    new_run_dir, run_stage_b_episode, write_json,
)
from scripts.runtime_v3_qwen_task_binding import _normalise_phrase


MANIFEST = ROOT / "experiments/runtime_v3/cross_object_align_manifest.json"
CONFIG = ROOT / "configs/robot_libero_clean_qwen3vl.yaml"
SCALE_EVIDENCE = ROOT / (
    "rollouts/runtime_v3_multiscale_alignment/"
    "run_20261002T133108Z_c21b2a84/stage_a/calibration_summary.json"
)
REFERENCE_STAGE_A = ROOT / (
    "rollouts/runtime_v3_cross_object_alignment/"
    "run_20261003T083700Z_9865926b/summary.json"
)
ALIGN_PHYSICAL_FILES = (
    "core/runtime_v3/object_relative.py", "core/runtime_v3/micro_motion.py",
    "core/runtime_v3/options.py", "core/runtime_v3/arbiter.py",
    "core/runtime_v3/executor.py", "core/runtime_v3/effects.py",
    "core/runtime_v3/state.py", "core/runtime_v3/scene_settling.py",
    "core/runtime_v3/scene_initialization.py",
    "scripts/runtime_v3_multiscale_alignment.py",
)


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True,
                           capture_output=True, text=True)
    return result.stdout.strip()


def _git_file_sha(commit: str, relative: str) -> str:
    result = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=ROOT,
                            check=True, capture_output=True)
    return hashlib.sha256(result.stdout).hexdigest()


def _binding_by_task(binding_data: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = binding_data.get("tasks", [])
    return {int(row["task_id"]): row for row in rows if isinstance(row, Mapping)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    config_path = Path(args.config).resolve()
    binding_path = Path(args.bindings).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding_data = json.loads(binding_path.read_text(encoding="utf-8"))
    reference_data = json.loads(Path(args.reference_summary).read_text(encoding="utf-8"))
    if not manifest.get("selection_frozen_before_rollout"):
        raise RuntimeError("Qwen physical transfer requires the frozen task manifest")
    if _git("status", "--porcelain"):
        raise RuntimeError("formal Qwen ALIGN rollout requires a clean worktree")
    if binding_data.get("manifest_sha256") != _sha256(manifest_path):
        raise RuntimeError("offline Qwen binding was produced for a different task manifest")
    if binding_data.get("status") != "COMPLETED":
        raise RuntimeError("offline Qwen semantic binding did not complete")
    if _git("rev-parse", "HEAD") == "":
        raise RuntimeError("could not identify the frozen Qwen rollout commit")

    starting_commit = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    source_signature = {
        **_runtime_signature(),
        "scripts/runtime_v3_qwen_cross_object_alignment.py": _sha256(Path(__file__).resolve()),
    }
    manifest_hash, binding_hash, config_hash = (
        _sha256(manifest_path), _sha256(binding_path), _sha256(config_path)
    )
    qwen_rows = _binding_by_task(binding_data)
    expected_ids = {int(row["task_id"]) for row in manifest["tasks"]}
    if set(qwen_rows) != expected_ids:
        raise RuntimeError("offline Qwen output task set does not match the frozen manifest")

    reference_commit = str(reference_data["starting_commit"])
    physical_hashes = {relative: _sha256(ROOT / relative) for relative in ALIGN_PHYSICAL_FILES}
    physical_matches_reference = {
        relative: digest == _git_file_sha(reference_commit, relative)
        for relative, digest in physical_hashes.items()
    }
    if not all(physical_matches_reference.values()):
        changed = [name for name, same in physical_matches_reference.items() if not same]
        raise RuntimeError(f"ALIGN physical source changed since reference Stage A: {changed}")

    calibration = json.loads(Path(args.scale_evidence).read_text(encoding="utf-8"))
    verified_scales = []
    contracts = {}
    for scale_m, max_ticks in zip((0.003, 0.006, 0.009), (5, 7, 10)):
        label = f"{int(scale_m * 1000)}mm"
        if calibration.get("scale_decisions", {}).get(label, {}).get("status") == "VERIFIED":
            verified_scales.append(scale_m)
            contracts[scale_m] = {"verified": True, "max_ticks": max_ticks}
    if tuple(verified_scales) != (0.003, 0.006, 0.009):
        raise RuntimeError("the reference Stage A 3/6/9mm contracts are not verified")

    from core.config import load_yaml
    from core.capabilities.sam3_client import Sam3Client

    config = load_yaml(str(config_path))
    if config.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(config["libero_dir"])
    sam3 = Sam3Client(url=args.sam3_url, python=args.sam3_python,
                      timeout_s=args.sam3_timeout_s, max_attempts=1)
    compiler = ReferenceTaskCompiler()
    run_dir = new_run_dir(args.output_dir)
    task_records = []
    all_episodes = []
    workspace = (0.02, 0.60)
    try:
        for task_entry in manifest["tasks"]:
            if (_git("rev-parse", "HEAD") != starting_commit
                    or {
                        **_runtime_signature(),
                        "scripts/runtime_v3_qwen_cross_object_alignment.py": _sha256(Path(__file__).resolve()),
                    } != source_signature
                    or _sha256(config_path) != config_hash
                    or _sha256(binding_path) != binding_hash
                    or _sha256(manifest_path) != manifest_hash):
                raise RuntimeError("code, config, task manifest, or Qwen binding changed during rollout")

            task_id = int(task_entry["task_id"])
            binding = qwen_rows[task_id]
            bound_entity = {
                "key": binding.get("entity_key", ""),
                "semantic_phrase": binding.get("semantic_phrase", ""),
                "role": binding.get("role", ""),
            }
            binding_matches = bool(
                binding.get("schema_valid")
                and binding.get("binding_matches_reference")
                and _normalise_phrase(str(bound_entity["semantic_phrase"]))
                    == _normalise_phrase(str(task_entry["reference_entity_phrase"]))
                and str(bound_entity["key"]) == str(task_entry["entity_key"])
                and str(bound_entity["role"]) == str(task_entry["role"])
            )
            if binding_matches:
                task_spec = compiler.compile({
                    "instruction": task_entry["instruction"],
                    "entities": [bound_entity],
                    "focus_entity_key": str(binding["focus_entity_key"]),
                    "goal_kind": str(binding["goal_kind"]),
                })
            else:
                task_spec = None

            task_dir = run_dir / f"task_{task_id}"
            task_dir.mkdir(parents=True, exist_ok=False)
            head_before = _git("rev-parse", "HEAD")
            signature_before = {
                **_runtime_signature(),
                "scripts/runtime_v3_qwen_cross_object_alignment.py": _sha256(Path(__file__).resolve()),
            }
            config_hash_before = _sha256(config_path)
            binding_hash_before = _sha256(binding_path)
            episodes: list[dict[str, Any]] = []
            initial_states = [int(value) for value in manifest["initial_gate"]["init_states"]]

            def run_indices(indices: Sequence[int]) -> None:
                for init_state in indices:
                    if not binding_matches or task_spec is None:
                        episode = {
                            "init_state_index": int(init_state), "suite": manifest["suite"],
                            "task_id": task_id, "seed": int(manifest["seed"]),
                            "task_instruction": str(task_entry["instruction"]),
                            "entity_phrase": binding.get("semantic_phrase"),
                            "status": "SKIPPED_SEMANTIC_BINDING_FAILURE",
                            "termination_reason": "SEMANTIC_BINDING_FAILURE",
                            "failure_layer": "SEMANTIC_BINDING_FAILURE",
                            "executed_alignment_steps": 0, "alignment_steps": [],
                            "oracle_used_by_runtime": False,
                        }
                    else:
                        try:
                            episode = run_stage_b_episode(
                                init_state=init_state, run_dir=task_dir, config=config,
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
                                "init_state_index": init_state, "suite": manifest["suite"],
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
                            failure_dir = task_dir / "stage_b" / f"init_state_{init_state}"
                            failure_dir.mkdir(parents=True, exist_ok=True)
                            write_json(failure_dir / "failure.json", episode)
                    episode["task_id"] = task_id
                    episode["entity_phrase"] = bound_entity["semantic_phrase"]
                    episodes.append(episode)
                    all_episodes.append(episode)
                    print(f"task={task_id} init={init_state} status={episode.get('status')} "
                          f"termination={episode.get('termination_reason')}", flush=True)

            run_indices(initial_states)
            initial_metrics = _task_metrics(episodes)
            expanded_states = []
            if (binding_matches and task_spec is not None and task_id != 2
                    and len(episodes) == len(initial_states)
                    and initial_metrics["net_improving_episodes"] == len(initial_states)
                    and initial_metrics["grounding_success_rate"] == 1.0
                    and initial_metrics["identity_success_rate"] == 1.0
                    and initial_metrics["reference_validity_rate"] == 1.0
                    and initial_metrics["scene_ready_success_rate"] == 1.0):
                expanded_states = [3, 4, 5]
                run_indices(expanded_states)
            if (_git("rev-parse", "HEAD") != head_before
                    or {
                        **_runtime_signature(),
                        "scripts/runtime_v3_qwen_cross_object_alignment.py": _sha256(Path(__file__).resolve()),
                    } != signature_before
                    or _sha256(config_path) != config_hash_before
                    or _sha256(binding_path) != binding_hash_before):
                raise RuntimeError("Runtime source or Qwen task binding changed during a task rollout")
            task_record = {
                "task_id": task_id, "instruction": str(task_entry["instruction"]),
                "reference_entity_phrase": str(task_entry["reference_entity_phrase"]),
                "entity_key": bound_entity["key"], "entity_phrase": bound_entity["semantic_phrase"],
                "role": bound_entity["role"], "binding_matches_reference": binding_matches,
                "qwen_binding_status": binding.get("status"),
                "qwen_model": binding_data.get("model"),
                "episodes": episodes, "metrics": _task_metrics(episodes),
                "initial_gate_metrics": initial_metrics,
                "expanded_init_states": expanded_states,
                "commit_before_task": head_before,
                "commit_after_task": _git("rev-parse", "HEAD"),
                "runtime_source_signature_unchanged": True,
                "config_sha256": config_hash_before,
                "qwen_binding_sha256": binding_hash_before,
            }
            task_records.append(task_record)
            write_json(task_dir / "task_summary.json", task_record)
            print(f"task={task_id} judgment={task_record['metrics']['judgment']} "
                  f"net_positive={task_record['metrics']['net_improving_episodes']}/"
                  f"{task_record['metrics']['episodes']}", flush=True)
    finally:
        sam3.close()

    if _git("rev-parse", "HEAD") != starting_commit:
        raise RuntimeError("HEAD changed during Qwen physical transfer")
    final_signature = {
        **_runtime_signature(),
        "scripts/runtime_v3_qwen_cross_object_alignment.py": _sha256(Path(__file__).resolve()),
    }
    if (final_signature != source_signature or _sha256(config_path) != config_hash
            or _sha256(manifest_path) != manifest_hash or _sha256(binding_path) != binding_hash):
        raise RuntimeError("source, config, manifest, or binding changed during Qwen physical transfer")

    montage = _write_cross_task_montage(all_episodes, run_dir)
    unseen = [row for row in task_records if int(row["task_id"]) != 2]
    bound_unseen = [row for row in unseen if row["binding_matches_reference"]]
    physical_judgment = (
        "YES" if len(bound_unseen) >= 2 and all(row["metrics"]["judgment"] == "PASS"
                                                 for row in bound_unseen)
        else "PARTIAL" if any(row["metrics"]["net_improving_episodes"] > 0
                              for row in bound_unseen)
        else "NO"
    )
    all_bindings_match = all(bool(row.get("binding_matches_reference")) for row in qwen_rows.values())
    qwen_runtime_judgment = (
        "YES" if all_bindings_match and physical_judgment == "YES"
        else "PARTIAL" if any(row["binding_matches_reference"] for row in task_records)
        and any(row["metrics"]["net_improving_episodes"] > 0 for row in task_records)
        else "NO"
    )
    audit = {
        "starting_commit": starting_commit, "final_commit": _git("rev-parse", "HEAD"),
        "branch": branch,
        "task_commits": {str(row["task_id"]): {
            "before": row["commit_before_task"], "after": row["commit_after_task"]
        } for row in task_records},
        "same_commit_across_tasks": all(
            row["commit_before_task"] == row["commit_after_task"] == starting_commit
            for row in task_records
        ),
        "runtime_core_changed_between_tasks": False,
        "align_contract_changed_between_tasks": False,
        "physical_files_unchanged_from_reference_stage_a": physical_matches_reference,
        "physical_control_code_unchanged_from_reference_stage_a": all(physical_matches_reference.values()),
        "per_object_thresholds": False,
        "per_task_direction_rules": False,
        "per_task_scale_rules": False,
        "qwen_selected_physical_direction": False,
        "qwen_selected_physical_scale": False,
        "physical_config_sha256": config_hash,
        "reference_stage_a_commit": reference_commit,
        "reference_stage_a_config_sha256": reference_data["code_change_audit"]["physical_config_sha256"],
        "qwen_binding_sha256": binding_hash,
        "runtime_source_signature_unchanged": True,
        "verified_scales_mm": [scale * 1000.0 for scale in verified_scales],
    }
    summary = {
        "phase": "M3.6 Stage B Qwen Semantic Coprocessor Cross-Object ALIGN",
        "status": "COMPLETED", "binding_mode": "qwen",
        "suite": manifest["suite"], "seed": manifest["seed"],
        "manifest_sha256": manifest_hash, "qwen_binding": str(binding_path),
        "qwen_binding_model": binding_data.get("model"),
        "qwen_binding_matches": sum(bool(row.get("binding_matches_reference")) for row in qwen_rows.values()),
        "semantic_binding_accuracy": sum(bool(row.get("binding_matches_reference")) for row in qwen_rows.values()) / len(qwen_rows) if qwen_rows else None,
        "qwen_selected_physical_direction": False, "qwen_selected_physical_scale": False,
        "starting_commit": starting_commit, "final_commit": _git("rev-parse", "HEAD"),
        "branch": branch,
        "physical_contract": {
            "goal_kind": "ALIGN", "directions": list(DIRECTION_ORDER),
            "scales_mm": [scale * 1000 for scale in verified_scales],
            "max_semantic_steps_per_episode": MAX_ALIGNMENT_STEPS,
            "same_selector": "MultiscaleSelector",
            "same_option_generator": "MultiScaleAlignmentOptionGenerator",
        },
        "tasks": task_records,
        "cross_object_physical_judgment": physical_judgment,
        "qwen_semantic_coprocessor_judgment": qwen_runtime_judgment,
        "failure_taxonomy": dict(Counter(
            layer for episode in all_episodes for layer in _episode_layers(episode)
        )),
        "code_change_audit": audit,
        "privileged_information": {
            "formal_runtime_target_pose": False, "formal_runtime_gt_depth": False,
            "formal_runtime_gt_contact": False, "formal_runtime_task_success": False,
        },
        "visual_audit": {"cross_task_contact_sheet": montage},
        "legacy_modified": False,
    }
    write_json(run_dir / "generalization_code_change_audit.json", audit)
    write_json(run_dir / "summary.json", summary)
    print(f"M3.6 Qwen binding {qwen_runtime_judgment}: {run_dir / 'summary.json'}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--scale-evidence", default=str(SCALE_EVIDENCE))
    parser.add_argument("--reference-summary", default=str(REFERENCE_STAGE_A))
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--output-dir", default=str(ROOT / "rollouts/runtime_v3_qwen_cross_object_alignment"))
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

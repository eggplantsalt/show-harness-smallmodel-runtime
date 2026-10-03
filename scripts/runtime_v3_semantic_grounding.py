#!/usr/bin/env python3
"""M3.8 development diagnostics and frozen held-out grounding/ALIGN runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.capabilities.sam3_client import Sam3Client
from core.config import load_yaml, resolve_vlm_config
from core.runtime_v3.grounding import (
    DEFAULT_GROUNDING_POOL_K,
    QwenSemanticCandidateSelector,
    QwenSemanticRegionProposer,
    SemanticGroundingBinder,
    candidate_from_detection,
    make_candidate_contact_sheet,
    GroundingCandidate,
)
from core.runtime_v3.task_spec import QwenTaskCompiler, ReferenceTaskCompiler
from core.vlm.vlm_client import VLMClient
from scripts.runtime_v3_multiscale_alignment import (
    _save_semantic_grounding_audit,
    _setup_trial,
    run_stage_b_episode,
    write_json,
)

CONFIG = ROOT / "configs/robot_libero_clean_qwen3vl.yaml"
MANIFEST = ROOT / "experiments/runtime_v3/m3_8/grounding_heldout_manifest.json"
DEV_TASKS = (
    (2, "salad dressing"), (7, "milk"), (0, "alphabet soup"),
    (6, "butter"), (1, "cream cheese"), (8, "chocolate pudding"),
)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _require_clean_worktree(phase: str) -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError(f"{phase} requires a committed, clean worktree")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_signature() -> dict[str, str]:
    files = list((ROOT / "core/runtime_v3").rglob("*.py")) + [
        ROOT / "core/capabilities/sam3_client.py",
        ROOT / "scripts/runtime_v3_multiscale_alignment.py",
        ROOT / "scripts/runtime_v3_semantic_grounding.py",
    ]
    return {path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in sorted(set(files))}


def _configure_local_proxy(url: str) -> None:
    if urlparse(url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        return
    existing = [item.strip() for key in ("NO_PROXY", "no_proxy")
                for item in os.environ.get(key, "").split(",") if item.strip()]
    value = ",".join(dict.fromkeys((*existing, "127.0.0.1", "localhost", "::1")))
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = value


def _clients(config: Mapping[str, Any], args: argparse.Namespace):
    vlm = resolve_vlm_config(dict(config))
    if str(vlm.get("provider", "vllm")).casefold() != "vllm":
        raise RuntimeError("M3.8 requires the configured local Qwen vLLM endpoint")
    if urlparse(str(vlm.get("base_url", ""))).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("M3.8 only permits the configured local Qwen endpoint")
    qwen = VLMClient(
        base_url=str(vlm["base_url"]), model=str(vlm["model"]),
        api_key=str(vlm.get("api_key", "EMPTY")),
        timeout_s=float(vlm.get("timeout_s", 180.0)), max_tokens=256,
        temperature=0.0, chat_template_kwargs={"enable_thinking": False, "thinking": False},
        provider="vllm", api_dialect=str(vlm.get("api_dialect", "vllm")),
        reasoning_effort=None, max_retries=0,
    )
    qwen.health_check(wait_s=0.0)
    response = qwen.session.get(f"{qwen.base_url}/models", timeout=10.0)
    response.raise_for_status()
    served = [str(item.get("id", "")) for item in response.json().get("data", [])
              if isinstance(item, dict)]
    if qwen.model not in served:
        raise RuntimeError(f"local endpoint serves {served!r}, expected {qwen.model!r}")
    _configure_local_proxy(args.sam3_url)
    sam = Sam3Client(url=args.sam3_url, python=args.sam3_python,
                     timeout_s=args.sam3_timeout_s, max_attempts=1)
    return qwen, sam


def _new_binder(qwen: Any, sam: Any, instruction: str, pool_k: int):
    return SemanticGroundingBinder(
        qwen_client=qwen, sam_client=sam, task_instruction=instruction,
        confidence_threshold=0.05, max_candidates=pool_k,
    )


def _close_qwen(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()
        return
    session = getattr(client, "session", None)
    if callable(getattr(session, "close", None)):
        session.close()


def _new_output(parent: Path, label: str) -> Path:
    import datetime
    import uuid
    output = parent / (
        f"{label}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    output.mkdir(parents=True, exist_ok=False)
    return output


def run_development(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(str(Path(args.config).resolve()))
    qwen, sam = _clients(config, args)
    output = _new_output(Path(args.output_dir).resolve(), "development")
    rows = []
    try:
        for task_id, phrase in DEV_TASKS:
            frame = ROOT / "experiments/runtime_v3/m3_8/development_frames" / (
                f"task_{task_id}_{phrase.replace(' ', '_')}_canonical.png"
            )
            if not frame.is_file():
                raise RuntimeError(f"missing saved canonical development frame: {frame}")
            image = np.asarray(Image.open(frame).convert("RGB"), dtype=np.uint8)
            instruction = f"Pick the {phrase} and place it in the basket"
            entity = ReferenceTaskCompiler().compile({
                "instruction": instruction,
                "entities": [{"key": "target", "semantic_phrase": phrase, "role": "MANIPULAND"}],
                "focus_entity_key": "target", "goal_kind": "ALIGN",
            }).focus_entity
            binder = _new_binder(qwen, sam, instruction, args.pool_k)
            result = binder.ground(
                image, entity_key=entity.key, semantic_phrase=entity.semantic_phrase,
                semantic_query=entity.semantic_phrase,
            )
            task_out = output / f"task_{task_id}_{phrase.replace(' ', '_')}"
            task_out.mkdir()
            if result.candidate_sheet is not None:
                Image.fromarray(result.candidate_sheet, mode="RGB").save(task_out / "candidate_sheet.png")
            baseline_details = result.baseline_response.get("details", {})
            detections = baseline_details.get("detections", []) if isinstance(baseline_details, Mapping) else []
            baseline = []
            for index, item in enumerate(detections if isinstance(detections, list) else []):
                if isinstance(item, Mapping):
                    candidate = candidate_from_detection(
                        item, candidate_id=f"C{index}", image_shape=image.shape[:2], source="text_sam",
                    )
                    if candidate is not None:
                        baseline.append(candidate)
            if baseline:
                sheet = make_candidate_contact_sheet(image, baseline[:args.pool_k])
                Image.fromarray(sheet, mode="RGB").save(task_out / "direct_text_sam_baseline.png")
            row = {
                "task_id": task_id, "semantic_phrase": phrase,
                "canonical_frame": str(frame.resolve()),
                "raw_proposals": result.pool_counts.get("raw_proposals"),
                "filtered_proposals": result.pool_counts.get("filtered_proposals"),
                "deduplicated_proposals": result.pool_counts.get("deduplicated_proposals"),
                "final_candidates": len(result.candidates), "candidate_pool_k": args.pool_k,
                "region_prompt_version": QwenSemanticRegionProposer.PROMPT_VERSION,
                "selector_prompt_version": QwenSemanticCandidateSelector.PROMPT_VERSION,
                "proposal_regions": result.evidence.proposal_region_count,
                "proposal_coverage_human_review": None,
                "selected_candidate_id": result.selected.candidate_id if result.selected else None,
                "semantic_decision": result.evidence.decision,
                "semantic_selection_correct_human_review": None,
                "direct_text_sam_detection_count": len(detections) if isinstance(detections, list) else 0,
                "direct_text_sam_top_candidate_human_review": None,
                "entity_observation_ready": None, "agent_calls": result.evidence.agent_calls,
                "sources": {item.candidate_id: list(item.sources) for item in result.candidates},
                "candidate_sheet": str((task_out / "candidate_sheet.png").resolve())
                    if (task_out / "candidate_sheet.png").exists() else None,
                "direct_baseline_sheet": str((task_out / "direct_text_sam_baseline.png").resolve())
                    if (task_out / "direct_text_sam_baseline.png").exists() else None,
                "proposal_response": result.proposal_response_text,
                "selector_response": result.selector_response_text,
            }
            rows.append(row)
            write_json(task_out / "development_episode.json", row)
            print(f"dev task={task_id} candidates={row['final_candidates']} "
                  f"regions={row['proposal_regions']} selected={row['selected_candidate_id']} "
                  f"calls={row['agent_calls']}", flush=True)
    finally:
        sam.close()
        _close_qwen(qwen)
    report = {
        "phase": "M3.8 development-only semantic proposal diagnostics",
        "status": "COMPLETED", "tasks": rows, "align_actions": 0,
        "robot_actions": 0, "oracle_used": False, "candidate_pool_k": args.pool_k,
        "region_prompt_version": QwenSemanticRegionProposer.PROMPT_VERSION,
        "selector_prompt_version": QwenSemanticCandidateSelector.PROMPT_VERSION,
        "model_calls": dict(qwen.metrics), "artifacts": str(output.resolve()),
    }
    write_json(output / "development_summary.json", report)
    print(output / "development_summary.json", flush=True)
    return report


def run_development_runtime(args: argparse.Namespace) -> dict[str, Any]:
    """Run bounded zero-motion readiness checks on the six development tasks."""
    config = load_yaml(str(Path(args.config).resolve()))
    if config.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(config["libero_dir"])
    qwen, sam = _clients(config, args)
    output = _new_output(Path(args.output_dir).resolve(), "development_runtime")
    rows = []
    try:
        for task_id, phrase in DEV_TASKS:
            instruction = f"Pick the {phrase} and place it in the basket"
            entry = {
                "task_id": task_id, "instruction": instruction,
                "reference_entity_phrase": phrase, "seed": 0,
            }
            spec = _reference_spec(entry, "ALIGN")
            task_dir = output / f"task_{task_id}_{phrase.replace(' ', '_')}"
            task_dir.mkdir(parents=True, exist_ok=False)
            binder = _new_binder(qwen, sam, instruction, args.pool_k)
            environment = None
            scene = {}
            try:
                environment, _controller, _base, observer, holds, scene, _trigger = _setup_trial(
                    init_state=0, config=config, sam3=sam, workspace=(0.02, 0.60),
                    camera_resolution=args.camera_resolution, suite_name="LIBERO_OBJECT",
                    task_id=task_id, entity_spec=spec.focus_entity, seed=0,
                    semantic_grounding_binder=binder, require_scene_ready=False,
                )
                grounding = observer.grounding_evidence
                result = observer.semantic_grounding_result
                audit = _save_semantic_grounding_audit(observer, task_dir)
                entity_readiness = (
                    observer.entity_observation_evidence.to_record()
                    if observer.entity_observation_evidence is not None else {}
                )
                motion_readiness = (
                    observer.scene_motion_evidence.to_record()
                    if observer.scene_motion_evidence is not None else {}
                )
                row = {
                    "task_id": task_id, "init_state_index": 0, "seed": 0,
                    "semantic_phrase": phrase,
                    "robot_ready": len(holds) == 4,
                    "robot_ready_hold_ticks": len(holds),
                    "scene_motion_ready": bool(observer.scene_motion_ready),
                    "entity_observation_ready": bool(observer.entity_observation_ready),
                    "identity_valid": bool(observer.identity_valid),
                    "reference_valid": bool(observer.reference_valid),
                    "grounding_to_reference_success": bool(
                        grounding and grounding.valid and observer.entity_observation_ready
                        and observer.identity_valid and observer.reference_valid
                    ),
                    "proposal_region_count": (
                        grounding.proposal_region_count if grounding else 0
                    ),
                    "raw_proposal_count": result.pool_counts.get("raw_proposals", 0) if result else 0,
                    "filtered_proposal_count": result.pool_counts.get("filtered_proposals", 0) if result else 0,
                    "deduplicated_proposal_count": result.pool_counts.get("deduplicated_proposals", 0) if result else 0,
                    "final_candidate_count": len(result.candidates) if result else 0,
                    "candidate_pool_k": args.pool_k,
                    "selected_candidate_id": grounding.candidate_id if grounding else None,
                    "semantic_decision": grounding.decision if grounding else "INVALID",
                    "no_match": bool(grounding and grounding.decision == "NO_MATCH"),
                    "readiness_ticks": len(scene.get("samples", [])),
                    "entity_observation_samples": entity_readiness.get(
                        "stable_observation_count", 0
                    ),
                    "scene_motion_stable_frame_pairs": motion_readiness.get(
                        "stable_frame_pair_count", 0
                    ),
                    "agent_calls": grounding.agent_calls if grounding else 0,
                    "alignment_actions": 0,
                    "robot_motion_actions": 0,
                    "formal_object_pose": False,
                    "formal_simulator_segmentation": False,
                    "formal_simulator_depth": False,
                    "formal_ground_truth_contact": False,
                    "formal_task_success": False,
                    "oracle_used_by_runtime": False,
                    "grounding_audit": audit,
                }
            except Exception as exc:
                row = {
                    "task_id": task_id, "init_state_index": 0, "seed": 0,
                    "status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                    "alignment_actions": 0, "oracle_used_by_runtime": False,
                }
            finally:
                if environment is not None:
                    environment.close()
            rows.append(row)
            write_json(task_dir / "development_runtime_episode.json", row)
            print(f"dev-runtime task={task_id} ready={row.get('grounding_to_reference_success')} "
                  f"selected={row.get('selected_candidate_id')} "
                  f"termination={row.get('error', scene.get('termination_reason'))}",
                  flush=True)
    finally:
        sam.close()
        _close_qwen(qwen)
    report = {
        "phase": "M3.8 development-only Runtime grounding/readiness",
        "status": "COMPLETED", "tasks": rows, "episode_count": len(rows),
        "candidate_pool_k": args.pool_k,
        "region_prompt_version": QwenSemanticRegionProposer.PROMPT_VERSION,
        "selector_prompt_version": QwenSemanticCandidateSelector.PROMPT_VERSION,
        "alignment_actions": 0, "robot_motion_actions": 0,
        "formal_privileged_inputs": {
            "object_pose": False, "simulator_segmentation": False, "simulator_depth": False,
            "ground_truth_contact": False, "task_success": False,
        },
        "model_calls": dict(qwen.metrics), "artifacts": str(output.resolve()),
    }
    write_json(output / "development_runtime_summary.json", report)
    print(output / "development_runtime_summary.json", flush=True)
    return report


def _frozen_manifest(path: Path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_FORMAL_ROLLOUT":
        raise RuntimeError("M3.8 held-out manifest is not frozen")
    if manifest.get("formal_privileged_inputs", {}).get("object_pose") is not False:
        raise RuntimeError("formal manifest must disable privileged object pose")
    return manifest, _sha256(path)


def _validate_grounding_configuration(manifest: Mapping[str, Any], pool_k: int) -> None:
    if int(manifest.get("candidate_pool_k", -1)) != int(pool_k):
        raise RuntimeError("candidate pool K differs from the value frozen in the manifest")
    if manifest.get("region_prompt_version") != QwenSemanticRegionProposer.PROMPT_VERSION:
        raise RuntimeError("region proposer prompt differs from the frozen manifest")
    if manifest.get("selector_prompt_version") != QwenSemanticCandidateSelector.PROMPT_VERSION:
        raise RuntimeError("candidate selector prompt differs from the frozen manifest")


def _reference_spec(entry: Mapping[str, Any], goal_kind: str):
    return ReferenceTaskCompiler().compile({
        "instruction": entry["instruction"],
        "entities": [{"key": "target", "semantic_phrase": entry["reference_entity_phrase"],
                      "role": "MANIPULAND"}],
        "focus_entity_key": "target", "goal_kind": goal_kind,
    })


def run_stage_a(args: argparse.Namespace) -> dict[str, Any]:
    _require_clean_worktree("Stage A")
    manifest_path = Path(args.manifest).resolve()
    manifest, manifest_hash = _frozen_manifest(manifest_path)
    _validate_grounding_configuration(manifest, args.pool_k)
    config_path = Path(args.config).resolve()
    config_hash = _sha256(config_path)
    config = load_yaml(str(config_path))
    if config.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(config["libero_dir"])
    qwen, sam = _clients(config, args)
    output = _new_output(Path(args.output_dir).resolve(), "stage_a")
    starting_head, branch = _git("rev-parse", "HEAD"), _git("branch", "--show-current")
    source_signature = _runtime_signature()
    episodes = []
    try:
        for entry in manifest["heldout_tasks"]:
            spec = _reference_spec(entry, "ALIGN")
            for init_state in entry["init_states"]:
                task_dir = output / f"task_{int(entry['task_id'])}" / f"init_state_{int(init_state)}"
                task_dir.mkdir(parents=True, exist_ok=False)
                binder = _new_binder(qwen, sam, spec.instruction, args.pool_k)
                environment = None
                try:
                    environment, _controller, _base, observer, holds, scene, trigger = _setup_trial(
                        init_state=int(init_state), config=config, sam3=sam,
                        workspace=(0.02, 0.60), camera_resolution=args.camera_resolution,
                        suite_name=manifest["suite"], task_id=int(entry["task_id"]),
                        entity_spec=spec.focus_entity, seed=int(entry["seed"]),
                        semantic_grounding_binder=binder, require_scene_ready=False,
                    )
                    grounding = observer.grounding_evidence
                    result = observer.semantic_grounding_result
                    audit = _save_semantic_grounding_audit(observer, task_dir)
                    entity_readiness = (
                        observer.entity_observation_evidence.to_record()
                        if observer.entity_observation_evidence is not None else {}
                    )
                    motion_readiness = (
                        observer.scene_motion_evidence.to_record()
                        if observer.scene_motion_evidence is not None else {}
                    )
                    episode = {
                        "task_id": int(entry["task_id"]), "init_state_index": int(init_state),
                        "seed": int(entry["seed"]), "task_instruction": spec.instruction,
                        "entity_key": spec.focus_entity.key, "entity_phrase": spec.focus_entity.semantic_phrase,
                        "robot_ready": len(holds) == 4, "robot_ready_hold_ticks": len(holds),
                        "scene_motion_ready": bool(observer.scene_motion_ready),
                        "proposal_region_count": (
                            grounding.proposal_region_count if grounding else 0
                        ),
                        "candidate_proposal_count": len(result.candidates) if result else 0,
                        "final_candidate_count": len(result.candidates) if result else 0,
                        "raw_proposal_count": result.pool_counts.get("raw_proposals", 0) if result else 0,
                        "filtered_proposal_count": result.pool_counts.get("filtered_proposals", 0) if result else 0,
                        "deduplicated_proposal_count": result.pool_counts.get("deduplicated_proposals", 0) if result else 0,
                        "target_candidate_present_human_review": None,
                        "selected_candidate_id": grounding.candidate_id if grounding else None,
                        "candidate_pool_k": args.pool_k,
                        "semantic_decision": grounding.decision if grounding else "INVALID",
                        "selection_correct_human_review": None,
                        "no_match": bool(grounding and grounding.decision == "NO_MATCH"),
                        "entity_observation_ready": bool(observer.entity_observation_ready),
                        "identity_valid": bool(observer.identity_valid),
                        "reference_valid": bool(observer.reference_valid),
                        "grounding_to_reference_success": bool(
                            grounding and grounding.valid and observer.identity_valid
                            and observer.entity_observation_ready and observer.reference_valid
                        ),
                        "readiness_ticks": len(scene.get("samples", [])),
                        "entity_observation_samples": entity_readiness.get(
                            "stable_observation_count", 0
                        ),
                        "scene_motion_stable_frame_pairs": motion_readiness.get(
                            "stable_frame_pair_count", 0
                        ),
                        "agent_calls": grounding.agent_calls if grounding else 0,
                        "alignment_actions": 0, "termination_reason": scene.get("termination_reason"),
                        "grounding_audit": audit,
                        "formal_input_channels": ["canonical_agentview_rgb", "robot_proprioception"],
                        "formal_object_pose": False, "formal_simulator_segmentation": False,
                        "formal_simulator_depth": False, "formal_ground_truth_contact": False,
                        "formal_task_success": False, "oracle_used_by_runtime": False,
                    }
                except Exception as exc:
                    episode = {
                        "task_id": int(entry["task_id"]), "init_state_index": int(init_state),
                        "seed": int(entry["seed"]), "status": "FAILED",
                        "error": f"{type(exc).__name__}: {exc}", "alignment_actions": 0,
                        "oracle_used_by_runtime": False,
                    }
                    write_json(task_dir / "failure.json", episode)
                finally:
                    if environment is not None:
                        environment.close()
                episodes.append(episode)
                write_json(task_dir / "stage_a_episode.json", episode)
                print(f"StageA task={entry['task_id']} init={init_state} "
                      f"ready={episode.get('grounding_to_reference_success')} "
                      f"selected={episode.get('selected_candidate_id')} "
                      f"reason={episode.get('termination_reason', episode.get('error'))}", flush=True)
    finally:
        sam.close()
        _close_qwen(qwen)
    report = {
        "phase": "M3.8 held-out grounding only", "status": "COMPLETED",
        "manifest": str(manifest_path), "manifest_sha256": manifest_hash,
        "starting_head": starting_head, "final_head": _git("rev-parse", "HEAD"),
        "branch": branch, "source_signature": source_signature, "config_sha256": config_hash,
        "candidate_pool_k": args.pool_k, "episodes": episodes, "episode_count": len(episodes),
        "region_prompt_version": QwenSemanticRegionProposer.PROMPT_VERSION,
        "selector_prompt_version": QwenSemanticCandidateSelector.PROMPT_VERSION,
        "alignment_actions": sum(int(row.get("alignment_actions", 0)) for row in episodes),
        "formal_privileged_inputs": {
            "object_pose": False, "simulator_segmentation": False, "simulator_depth": False,
            "ground_truth_contact": False, "task_success": False,
        },
        "model_calls": dict(qwen.metrics), "artifacts": str(output.resolve()),
    }
    if manifest_hash != _sha256(manifest_path) or config_hash != _sha256(config_path):
        raise RuntimeError("frozen manifest or configuration changed during Stage A")
    if starting_head != _git("rev-parse", "HEAD") or source_signature != _runtime_signature():
        raise RuntimeError("source changed during Stage A")
    write_json(output / "stage_a_summary.json", report)
    print(output / "stage_a_summary.json", flush=True)
    return report


def _eligible_episodes(stage_a: Mapping[str, Any], review: Mapping[str, Any]):
    review_rows = {}
    for row in review.get("episodes", []):
        key = (int(row["task_id"]), int(row["init_state_index"]))
        if key in review_rows:
            raise RuntimeError(f"duplicate visual-review row for {key}")
        if not isinstance(row.get("selected_matches_target"), bool):
            raise RuntimeError(f"visual review lacks selected_matches_target for {key}")
        if not isinstance(row.get("target_candidate_present"), bool):
            raise RuntimeError(f"visual review lacks target_candidate_present for {key}")
        if row["selected_matches_target"] and not row["target_candidate_present"]:
            raise RuntimeError(f"visual review marks an absent target proposal as selected for {key}")
        review_rows[key] = row
    stage_keys = {
        (int(row["task_id"]), int(row["init_state_index"]))
        for row in stage_a.get("episodes", []) if row.get("init_state_index") is not None
    }
    if stage_keys != set(review_rows):
        raise RuntimeError("visual review must cover every Stage A episode exactly once")
    return [
        row for row in stage_a.get("episodes", [])
        if row.get("grounding_to_reference_success")
        and review_rows.get((int(row["task_id"]), int(row["init_state_index"])), {})
            .get("selected_matches_target") is True
        and review_rows[(int(row["task_id"]), int(row["init_state_index"]))]
            .get("selected_candidate_id") == row.get("selected_candidate_id")
    ]


def _run_alignment(args: argparse.Namespace, *, full_semantic: bool):
    _require_clean_worktree("Stage B/C")
    manifest_path = Path(args.manifest).resolve()
    manifest, manifest_hash = _frozen_manifest(manifest_path)
    _validate_grounding_configuration(manifest, args.pool_k)
    stage_a = json.loads(Path(args.stage_a_summary).read_text(encoding="utf-8"))
    if stage_a.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("Stage A does not match the frozen manifest")
    review = json.loads(Path(args.visual_audit).read_text(encoding="utf-8"))
    if review.get("review_complete") is not True:
        raise RuntimeError("manual visual audit is not marked complete")
    accepted = _eligible_episodes(stage_a, review)
    if not accepted:
        raise RuntimeError("no visually correct, Runtime-reference-valid episodes qualify for ALIGN")
    if stage_a.get("source_signature") != _runtime_signature():
        raise RuntimeError("Stage A and ALIGN stages must use the same frozen source signature")
    if full_semantic:
        if not args.stage_b_summary:
            raise RuntimeError("Stage C requires the completed Stage B summary")
        stage_b = json.loads(Path(args.stage_b_summary).read_text(encoding="utf-8"))
        if (stage_b.get("manifest_sha256") != manifest_hash
                or int(stage_b.get("alignment_steps", 0)) <= 0
                or int(stage_b.get("positive_effects", 0)) <= 0
                or stage_b.get("source_signature") != _runtime_signature()):
            raise RuntimeError("Stage C requires positive-effect evidence from the same Stage B manifest")
    config_path = Path(args.config).resolve()
    config_hash = _sha256(config_path)
    config = load_yaml(str(config_path))
    if config.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(config["libero_dir"])
    qwen, sam = _clients(config, args)
    output = _new_output(Path(args.output_dir).resolve(), "stage_c" if full_semantic else "stage_b")
    starting_head, branch = _git("rev-parse", "HEAD"), _git("branch", "--show-current")
    signature = _runtime_signature()
    compiler = QwenTaskCompiler(qwen, max_tokens=128) if full_semantic else None
    compiler_specs = {}
    compiler_calls = 0
    tasks = []
    try:
        if compiler is not None:
            for task_id in sorted({int(row["task_id"]) for row in accepted}):
                entry = next(row for row in manifest["heldout_tasks"] if int(row["task_id"]) == task_id)
                before = int(qwen.metrics["model_calls"])
                spec = compiler.compile(str(entry["instruction"]))
                compiler_calls += int(qwen.metrics["model_calls"]) - before
                actual = " ".join(spec.focus_entity.semantic_phrase.casefold().split())
                if actual.startswith(("the ", "a ", "an ")):
                    actual = actual.split(" ", 1)[1]
                if actual != entry["reference_entity_phrase"]:
                    raise RuntimeError(f"QwenTaskCompiler phrase mismatch for task {task_id}: {actual!r}")
                compiler_specs[task_id] = spec

        calibration = json.loads(Path(args.scale_evidence).read_text(encoding="utf-8"))
        scales, contracts = [], {}
        for scale, max_ticks in zip((0.003, 0.006, 0.009), (5, 7, 10)):
            label = f"{int(scale * 1000)}mm"
            if calibration.get("scale_decisions", {}).get(label, {}).get("status") == "VERIFIED":
                scales.append(scale)
                contracts[scale] = {"verified": True, "max_ticks": max_ticks}
        if tuple(scales) != (0.003, 0.006, 0.009):
            raise RuntimeError("existing 3/6/9 mm ALIGN contracts are not all verified")

        for stage_row in accepted:
            task_id, init_state = int(stage_row["task_id"]), int(stage_row["init_state_index"])
            entry = next(row for row in manifest["heldout_tasks"] if int(row["task_id"]) == task_id)
            spec = compiler_specs.get(task_id) or _reference_spec(entry, "ALIGN")
            episode_dir = output / f"task_{task_id}" / f"init_state_{init_state}"
            episode_dir.mkdir(parents=True, exist_ok=False)
            binder = _new_binder(qwen, sam, spec.instruction, args.pool_k)
            try:
                episode = run_stage_b_episode(
                    init_state=init_state, run_dir=episode_dir, config=config, sam3=sam,
                    workspace=(0.02, 0.60), camera_resolution=args.camera_resolution,
                    verified_scales=scales, contracts=contracts,
                    suite_name=manifest["suite"], task_id=task_id,
                    entity_spec=spec.focus_entity, diagnostic_oracle=False, seed=int(entry["seed"]),
                    semantic_grounding_binder=binder,
                    expected_grounding_candidate_id=str(stage_row["selected_candidate_id"]),
                )
            except Exception as exc:
                episode = {
                    "task_id": task_id, "init_state_index": init_state, "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}", "alignment_steps": [],
                    "executed_alignment_steps": 0, "oracle_used_by_runtime": False,
                }
            episode["stage_a_visually_correct"] = True
            episode["qwen_physical_actions"] = 0
            episode["physical_contract_changes"] = 0
            tasks.append(episode)
            write_json(episode_dir / "m3_8_episode_summary.json", episode)
            print(f"{'StageC' if full_semantic else 'StageB'} task={task_id} init={init_state} "
                  f"steps={episode.get('executed_alignment_steps', 0)} "
                  f"termination={episode.get('termination_reason', episode.get('error'))}", flush=True)
    finally:
        sam.close()
        _close_qwen(qwen)

    executed = [step for row in tasks for step in row.get("alignment_steps", []) if step.get("executed")]
    effects = [float(step["actual_improvement_px"]) for step in executed
               if step.get("actual_improvement_px") is not None]
    completed = [row for row in tasks if int(row.get("executed_alignment_steps", 0)) > 0]
    report = {
        "phase": "M3.8 Stage C full semantic path" if full_semantic else "M3.8 Stage B ALIGN revalidation",
        "status": "COMPLETED", "manifest_sha256": manifest_hash,
        "starting_head": starting_head, "final_head": _git("rev-parse", "HEAD"),
        "branch": branch, "source_signature": signature, "tasks": tasks,
        "candidate_pool_k": args.pool_k,
        "region_prompt_version": QwenSemanticRegionProposer.PROMPT_VERSION,
        "selector_prompt_version": QwenSemanticCandidateSelector.PROMPT_VERSION,
        "eligible_episodes": len(accepted), "alignment_steps": len(executed),
        "positive_effects": sum(value > 0 for value in effects),
        "positive_effect_fraction": sum(value > 0 for value in effects) / len(effects) if effects else None,
        "monotonic_episode_fraction": (
            sum(bool(row.get("all_steps_positive")) for row in completed) / len(completed)
            if completed else None
        ),
        "mean_normalized_error_reduction": _normalized_reduction(tasks),
        "direction_distribution": _counts(step.get("direction") for step in executed),
        "scale_distribution_mm": _counts(step.get("scale_mm") for step in executed),
        "qwen_task_compiler_calls": compiler_calls, "qwen_physical_actions": 0,
        "physical_contract_changes": 0, "oracle_used_by_runtime": False,
        "formal_privileged_inputs": {
            "object_pose": False, "simulator_segmentation": False, "simulator_depth": False,
            "ground_truth_contact": False, "task_success": False,
        },
        "model_calls": dict(qwen.metrics), "artifacts": str(output.resolve()),
    }
    if manifest_hash != _sha256(manifest_path) or config_hash != _sha256(config_path):
        raise RuntimeError("manifest or config changed during ALIGN evaluation")
    if starting_head != _git("rev-parse", "HEAD") or signature != _runtime_signature():
        raise RuntimeError("source changed during ALIGN evaluation")
    name = "stage_c_summary.json" if full_semantic else "stage_b_summary.json"
    write_json(output / name, report)
    print(output / name, flush=True)
    return report


def _normalized_reduction(rows):
    values = []
    for row in rows:
        before, after = row.get("initial_error_px"), row.get("final_error_px")
        if before is not None and after is not None and float(before) > 0:
            values.append((float(before) - float(after)) / float(before))
    return float(np.mean(values)) if values else None


def _counts(values):
    result = {}
    for value in values:
        if value is None:
            continue
        key = str(round(float(value), 4) if isinstance(value, (int, float)) else value)
        result[key] = result.get(key, 0) + 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "development-runtime", "stage-a", "stage-b", "stage-c"), required=True)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--output-dir", default="rollouts/runtime_v3_m3_8")
    parser.add_argument("--stage-a-summary")
    parser.add_argument("--stage-b-summary")
    parser.add_argument("--visual-audit")
    parser.add_argument("--scale-evidence", default="rollouts/runtime_v3_multiscale_alignment/run_20261002T133108Z_c21b2a84/stage_a/calibration_summary.json")
    parser.add_argument("--sam3-url", default="http://127.0.0.1:8773/sse")
    parser.add_argument("--sam3-python", default="/root/autodl-tmp/openeta-services/sam3/.venv/bin/python")
    parser.add_argument("--sam3-timeout-s", type=float, default=90.0)
    parser.add_argument("--camera-resolution", type=int, default=512)
    parser.add_argument("--pool-k", type=int, default=DEFAULT_GROUNDING_POOL_K)
    args = parser.parse_args()
    if args.phase == "development":
        run_development(args)
    elif args.phase == "development-runtime":
        run_development_runtime(args)
    elif args.phase == "stage-a":
        run_stage_a(args)
    else:
        if not args.stage_a_summary or not args.visual_audit:
            parser.error("Stage B/C requires --stage-a-summary and --visual-audit")
        if args.phase == "stage-c" and not args.stage_b_summary:
            parser.error("Stage C requires --stage-b-summary")
        _run_alignment(args, full_semantic=args.phase == "stage-c")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

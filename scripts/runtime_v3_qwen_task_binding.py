#!/usr/bin/env python3
"""Run one offline Qwen semantic binding call per frozen M3.6 instruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_yaml, resolve_vlm_config
from core.runtime_v3.task_spec import QwenTaskCompiler
from core.vlm.vlm_client import VLMClient
from scripts.runtime_v3_multistep_alignment import new_run_dir, write_json


MANIFEST = ROOT / "experiments/runtime_v3/cross_object_align_manifest.json"
CONFIG = ROOT / "configs/robot_libero_clean_qwen3vl.yaml"


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True,
                           capture_output=True, text=True)
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalise_phrase(value: str) -> str:
    phrase = " ".join(value.casefold().split())
    return re.sub(r"^(?:the|a|an)\s+", "", phrase)


def _raw_has_physical_fields(raw_text: str | None) -> bool:
    if not raw_text:
        return False
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return False
    forbidden = {
        "direction", "distance", "pixel", "pixels", "xyz", "coordinate",
        "coordinates", "scale", "threshold", "grasp_point", "action",
        "controller_action", "controller", "trajectory",
    }
    def has_key(value: Any) -> bool:
        if isinstance(value, dict):
            return any(str(key).casefold() in forbidden or has_key(child)
                       for key, child in value.items())
        if isinstance(value, list):
            return any(has_key(item) for item in value)
        return False
    return has_key(parsed)


def _make_client(config: dict[str, Any]) -> VLMClient:
    vlm = resolve_vlm_config(config)
    if str(vlm.get("provider", "vllm")).casefold() != "vllm":
        raise RuntimeError("M3.6 Qwen binding requires the configured local vLLM backend")
    if urlparse(str(vlm.get("base_url", ""))).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("M3.6 Qwen binding only permits the configured local endpoint")
    return VLMClient(
        base_url=str(vlm["base_url"]), model=str(vlm["model"]),
        api_key=str(vlm.get("api_key", "EMPTY")),
        timeout_s=float(vlm.get("timeout_s", 180.0)), max_tokens=128,
        temperature=0.0,
        chat_template_kwargs={"enable_thinking": False, "thinking": False},
        provider="vllm", api_dialect=str(vlm.get("api_dialect", "vllm")),
        reasoning_effort=None, max_retries=0,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    config_path = Path(args.config).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("selection_frozen_before_rollout"):
        raise RuntimeError("semantic binding requires the frozen M3.6 task manifest")
    if _git("status", "--porcelain"):
        raise RuntimeError("Qwen binding requires a clean worktree")
    starting_commit = _git("rev-parse", "HEAD")
    manifest_hash, config_hash = _sha256(manifest_path), _sha256(config_path)
    config = load_yaml(str(config_path))
    client = _make_client(config)
    client.health_check(wait_s=0.0)
    model_response = client.session.get(f"{client.base_url}/models", timeout=10.0)
    model_response.raise_for_status()
    served_models = [str(item.get("id", "")) for item in
                     model_response.json().get("data", []) if isinstance(item, dict)]
    if client.model not in served_models:
        raise RuntimeError(
            f"local endpoint serves {served_models!r}, expected configured model {client.model!r}"
        )

    compiler = QwenTaskCompiler(client, max_tokens=128)
    rows = []
    try:
        for entry in manifest["tasks"]:
            calls_before = int(client.metrics["model_calls"])
            row: dict[str, Any] = {
                "task_id": int(entry["task_id"]),
                "instruction": str(entry["instruction"]),
                "reference_entity_phrase": str(entry["reference_entity_phrase"]),
                "model_calls_for_instruction": 0,
                "schema_valid": False,
                "binding_matches_reference": False,
                "qwen_selected_direction": False,
                "qwen_selected_scale": False,
                "robot_actions": 0,
            }
            try:
                task_spec = compiler.compile(row["instruction"])
                entity = task_spec.focus_entity
                row.update({
                    "schema_valid": True,
                    "entity_key": entity.key,
                    "semantic_phrase": entity.semantic_phrase,
                    "role": entity.role,
                    "focus_entity_key": task_spec.focus_entity_key,
                    "goal_kind": task_spec.goal_kind.value,
                    "phrase_matches_reference": (
                        _normalise_phrase(entity.semantic_phrase)
                        == _normalise_phrase(row["reference_entity_phrase"])
                    ),
                    "entity_key_matches_reference": entity.key == str(entry["entity_key"]),
                    "role_matches_reference": entity.role == str(entry["role"]),
                    "qwen_raw_physical_fields_detected": False,
                    "raw_response": compiler.last_raw_text,
                })
                row["binding_matches_reference"] = bool(
                    row["phrase_matches_reference"]
                    and row["entity_key_matches_reference"]
                    and row["role_matches_reference"]
                    and task_spec.goal_kind.value == manifest["goal_kind"]
                )
                row["status"] = "MATCH" if row["binding_matches_reference"] else "MISMATCH"
            except Exception as exc:
                row.update({
                    "status": "INVALID_SCHEMA_OR_CALL",
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_response": compiler.last_raw_text or getattr(exc, "raw_text", None),
                })
                row["qwen_raw_physical_fields_detected"] = _raw_has_physical_fields(
                    row["raw_response"]
                )
            finally:
                row["model_calls_for_instruction"] = (
                    int(client.metrics["model_calls"]) - calls_before
                )
            rows.append(row)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    if _git("rev-parse", "HEAD") != starting_commit:
        raise RuntimeError("HEAD changed during Qwen offline semantic binding")
    if _sha256(manifest_path) != manifest_hash or _sha256(config_path) != config_hash:
        raise RuntimeError("task manifest or VLM configuration changed during Qwen binding")
    run_dir = new_run_dir(args.output_dir)
    matches = sum(bool(row["binding_matches_reference"]) for row in rows)
    audit = {
        "phase": "M3.6 Stage B offline Qwen semantic binding",
        "status": "COMPLETED", "binding_mode": "qwen",
        "starting_commit": starting_commit, "final_commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "manifest": str(manifest_path), "manifest_sha256": manifest_hash,
        "config_sha256": config_hash,
        "model": client.model, "endpoint": client.base_url,
        "temperature": 0.0, "max_tokens": 128,
        "one_model_call_per_instruction": all(row["model_calls_for_instruction"] == 1 for row in rows),
        "schema_valid_count": sum(bool(row["schema_valid"]) for row in rows),
        "binding_match_count": matches, "task_count": len(rows),
        "semantic_binding_accuracy": matches / len(rows) if rows else None,
        "qwen_selected_physical_direction": False,
        "qwen_selected_physical_scale": False,
        "qwen_emitted_forbidden_physical_fields": any(
            bool(row.get("qwen_raw_physical_fields_detected")) for row in rows
        ),
        "robot_actions": 0,
        "tasks": rows,
        "client_metrics": dict(client.metrics),
    }
    write_json(run_dir / "qwen_binding.json", audit)
    print(f"Qwen binding {matches}/{len(rows)} matched: {run_dir / 'qwen_binding.json'}", flush=True)
    return audit


def reconcile_saved(args: argparse.Namespace) -> dict[str, Any]:
    """Re-audit saved Qwen text without making another model call."""
    manifest_path = Path(args.manifest).resolve()
    source_path = Path(args.reconcile_existing).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not manifest.get("selection_frozen_before_rollout"):
        raise RuntimeError("semantic reconciliation requires the frozen M3.6 task manifest")
    if source.get("status") != "COMPLETED" or source.get("binding_mode") != "qwen":
        raise RuntimeError("saved report is not a completed Qwen binding audit")
    if source.get("manifest_sha256") != _sha256(manifest_path):
        raise RuntimeError("saved Qwen output belongs to a different task manifest")

    rows_by_id = {int(row["task_id"]): dict(row) for row in source.get("tasks", [])}
    expected = {int(row["task_id"]): row for row in manifest["tasks"]}
    if set(rows_by_id) != set(expected):
        raise RuntimeError("saved Qwen output task set does not match the frozen manifest")
    rows = []
    for task_id, entry in expected.items():
        row = rows_by_id[task_id]
        row["literal_phrase_matches_reference"] = bool(row.get("phrase_matches_reference"))
        raw = row.get("raw_response")
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else None
            entity = parsed["entities"][0] if isinstance(parsed, dict) else {}
            phrase = entity.get("semantic_phrase", "") if isinstance(entity, dict) else ""
            role = entity.get("role", "") if isinstance(entity, dict) else ""
            key = entity.get("key", "") if isinstance(entity, dict) else ""
            focus_key = parsed.get("focus_entity_key", "") if isinstance(parsed, dict) else ""
            goal = parsed.get("goal_kind", "") if isinstance(parsed, dict) else ""
            schema_valid = bool(
                row.get("schema_valid") and len(parsed.get("entities", [])) == 1
                and key == str(entry["entity_key"]) and role == str(entry["role"])
                and focus_key == key and goal == str(manifest["goal_kind"])
            ) if isinstance(parsed, dict) else False
        except (json.JSONDecodeError, KeyError, TypeError, IndexError):
            phrase, role, key, goal, schema_valid = "", "", "", "", False
        phrase_matches = bool(
            schema_valid
            and _normalise_phrase(str(phrase))
                == _normalise_phrase(str(entry["reference_entity_phrase"]))
        )
        row.update({
            "schema_valid": schema_valid,
            "entity_key": key,
            "semantic_phrase": phrase,
            "role": role,
            "focus_entity_key": key,
            "goal_kind": goal,
            "phrase_matches_reference": phrase_matches,
            "binding_matches_reference": phrase_matches,
            "match_rule": "casefold, collapse whitespace, strip one leading English article",
            "status": "MATCH" if phrase_matches else "MISMATCH",
        })
        rows.append(row)

    matches = sum(bool(row["binding_matches_reference"]) for row in rows)
    audit = dict(source)
    audit.update({
        "phase": "M3.6 Stage B offline Qwen semantic binding audit (saved-response reconciliation)",
        "status": "COMPLETED", "tasks": rows,
        "source_binding_audit": str(source_path),
        "source_binding_audit_sha256": _sha256(source_path),
        "audit_commit": _git("rev-parse", "HEAD"),
        "model_calls_this_reconciliation": 0,
        "model_calls_for_original_outputs": sum(
            int(row.get("model_calls_for_instruction", 0)) for row in rows
        ),
        "match_rule": "casefold, collapse whitespace, strip one leading English article",
        "binding_match_count": matches,
        "task_count": len(rows),
        "semantic_binding_accuracy": matches / len(rows) if rows else None,
        "one_model_call_per_instruction": all(
            int(row.get("model_calls_for_instruction", 0)) == 1 for row in rows
        ),
        "qwen_emitted_forbidden_physical_fields": any(
            bool(row.get("qwen_raw_physical_fields_detected")) for row in rows
        ),
        "robot_actions": 0,
    })
    run_dir = new_run_dir(args.output_dir)
    write_json(run_dir / "qwen_binding.json", audit)
    print(f"Reconciled saved Qwen binding {matches}/{len(rows)}: "
          f"{run_dir / 'qwen_binding.json'}", flush=True)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--output-dir", default=str(ROOT / "rollouts/runtime_v3_qwen_task_binding"))
    parser.add_argument("--reconcile-existing", help="re-audit saved raw responses without model calls")
    args = parser.parse_args()
    return 0 if (reconcile_saved(args) if args.reconcile_existing else run(args)) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate bounded Qwen option selection without constructing robot authority."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_secrets_env, load_yaml
from core.runtime_v3.adapters.qwen_selector import QwenSelectorAdapter
from core.runtime_v3.options import PrimitiveCommand, RuntimeOption
from core.runtime_v3.state import BeliefState
from core.sim.launch import build_config, make_vlm_client


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/robot_libero_clean_qwen3vl.yaml"),
    )
    parser.add_argument("--vlm-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    return parser


def _option(option_id: str, description: str, primitive_kind: str = "move") -> RuntimeOption:
    token = "MV_UP" if primitive_kind == "move" else None
    return RuntimeOption(
        option_id=option_id,
        option_type="bounded_selector_eval",
        description=description,
        preconditions={},
        expected_effect={},
        primitive=PrimitiveCommand(primitive_kind, token, max_steps=1),
        confidence=1.0,
        evidence=("selector-contract-case",),
        evidence_frame_id=1,
    )


def _state(
    task_id: str,
    *,
    stage: str,
    target_identity: str,
    confidence: float,
    uncertainty: dict[str, Any] | None = None,
    geometry: dict[str, Any] | None = None,
    contact_state: str | None = None,
) -> BeliefState:
    return BeliefState(
        task_id=task_id,
        step_id=1,
        stage=stage,
        target_identity=target_identity,
        target_confidence=confidence,
        contact_state=contact_state,
        uncertainty=uncertainty or {"target": "low"},
        relevant_geometry=geometry or {},
        evidence_refs=("selector-contract-case",),
        frame_id=1,
        observation_fresh=True,
    )


def _move_choice_options() -> list[RuntimeOption]:
    candidates = [
        ("OPTION_0", "Translate 5 mm along +X.", "MV_FWD", [0.005, 0.0, 0.0]),
        ("OPTION_1", "Translate 5 mm along -X.", "MV_BACK", [-0.005, 0.0, 0.0]),
        ("OPTION_2", "Translate 5 mm along +Y.", "MV_LEFT", [0.0, 0.005, 0.0]),
        ("OPTION_3", "Translate 5 mm along -Y.", "MV_RIGHT", [0.0, -0.005, 0.0]),
    ]
    result = []
    for option_id, description, token, delta in candidates:
        option = _option(option_id, description)
        result.append(RuntimeOption(
            option_id=option.option_id,
            option_type=option.option_type,
            description=option.description,
            preconditions={"observation_fresh": True},
            expected_effect={"end_effector_delta_xyz": delta},
            primitive=PrimitiveCommand("move", token, {"step_m": 0.005}, 1, 1.0),
            confidence=1.0,
            evidence=("selector-contract-case",),
            evidence_frame_id=1,
        ))
    result.append(_option("REOBSERVE", "Take one new observation without moving."))
    return result


def _choices_for_case(index: int) -> tuple[str, BeliefState, list[RuntimeOption], str, str, str | None]:
    clear_offsets = (
        [0.035, 0.002, 0.0],
        [-0.028, 0.004, 0.0],
        [0.002, -0.041, 0.0],
        [0.003, 0.032, 0.0],
    )
    if index < 4:
        offset = clear_offsets[index]
        candidates = [([1.0, 0.0, 0.0], "OPTION_0"), ([-1.0, 0.0, 0.0], "OPTION_1"),
                      ([0.0, 1.0, 0.0], "OPTION_2"), ([0.0, -1.0, 0.0], "OPTION_3")]
        expected = max(candidates, key=lambda pair: float(np.dot(pair[0], offset)))[1]
        state = _state(
            f"selector-clear-{index + 1}", stage="ALIGN", target_identity="visible target",
            confidence=0.97,
            geometry={"target_offset_xyz_m": offset, "target_visible": True},
        )
        return (
            "Align the end effector to the visible target. Select the bounded move that reduces the target offset.",
            state, _move_choice_options(), expected, "A_CLEAR_GEOMETRIC_CHOICE", None,
        )
    if index < 6:
        state = _state(
            f"selector-ambiguous-{index - 3}", stage="UNCERTAIN", target_identity="uncertain target",
            confidence=0.04 + 0.01 * (index - 4),
            uncertainty={"target": "identity_ambiguous", "position": "unreliable"},
            geometry={"target_visible": False},
        )
        options = [
            _option("OPTION_0", "Move 5 mm toward the presumed target."),
            _option("OPTION_1", "Close the gripper at the presumed target.", "grasp"),
            _option("REOBSERVE", "Take one new observation without moving."),
        ]
        return (
            "Identify the requested object, then move toward it safely.", state, options,
            "REOBSERVE", "B_AMBIGUOUS_STATE", None,
        )
    if index < 8:
        grasp_stage = index == 7
        stage = "GRASP" if grasp_stage else "APPROACH"
        state = _state(
            f"selector-stage-{index - 5}", stage=stage, target_identity="requested block",
            confidence=0.98,
            geometry={"target_visible": True, "approach_complete": grasp_stage,
                      "grasp_preconditions_met": grasp_stage},
            contact_state="CONTACT_CONFIRMED" if grasp_stage else "NO_CONTACT",
        )
        options = [
            _option("OPTION_0", "Continue the bounded approach toward the requested block."),
            _option("OPTION_1", "Close the gripper on the requested block.", "grasp"),
            _option("REOBSERVE", "Take one new observation without moving."),
        ]
        return (
            "Pick up the requested block and keep the next action within the current stage.",
            state, options, "OPTION_1" if grasp_stage else "OPTION_0", "C_STAGE_DECISION", None,
        )

    pair_index = (index - 8) // 2
    choose_second = (index - 8) % 2 == 1
    names = (("red block", "blue bowl"), ("green cup", "yellow plate"))[pair_index]
    state = _state(
        f"selector-task-pair-{pair_index + 1}", stage="SELECT_TARGET",
        target_identity=f"{names[0]} and {names[1]} candidates", confidence=0.98,
        geometry={"target_visible": True},
    )
    options = [
        _option("OPTION_0", f"Approach the {names[0]} candidate."),
        _option("OPTION_1", f"Approach the {names[1]} candidate."),
        _option("REOBSERVE", "Take one new observation without moving."),
    ]
    requested = names[1] if choose_second else names[0]
    task = f"For this task, manipulate the {requested}. Select the option that approaches that requested object."
    return task, state, options, "OPTION_1" if choose_second else "OPTION_0", "D_TASK_CONDITIONED_CHOICE", f"PAIR_{pair_index + 1}"


class _FixedRawResponseClient:
    """Safety fixture used to prove malformed/raw commands fail closed."""

    model = "injected-invalid-response-fixture"

    def __init__(self, raw: str) -> None:
        self.raw = raw

    def complete_json(self, _prompt: str, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            raw_text=self.raw,
            payload={"raw": {"choices": [{"message": {"content": self.raw}}]}},
        )


def _raw_selection(raw: str) -> tuple[bool, str | None]:
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return False, None
    if not isinstance(parsed, dict) or set(parsed) != {"selection"}:
        return False, None
    choice = parsed.get("selection")
    return isinstance(choice, str), choice if isinstance(choice, str) else None


def _safety_cases() -> list[dict[str, Any]]:
    malformed = [
        ("MV_LEFT", "raw_action_token"),
        ("move left", "natural_language_action"),
        ('{"selection":"MV_LEFT"}', "json_with_disallowed_action"),
        ('{"selection":"OPTION_0","extra":"MV_LEFT"}', "extra_schema_key"),
    ]
    state = BeliefState(
        task_id="selector-safety-fixture",
        frame_id=1,
        observation_fresh=True,
        evidence_refs=("selector-contract-case",),
    )
    options = [_option("OPTION_0", "One bounded move option.")]
    cases = []
    for index, (raw, label) in enumerate(malformed, start=13):
        selector = QwenSelectorAdapter(_FixedRawResponseClient(raw), "safety fixture")
        started = time.monotonic()
        selection = selector.select(state, options)
        structural_schema_valid, raw_choice = _raw_selection(raw)
        allowed = {option.option_id for option in options} | {"REOBSERVE", "ABORT"}
        cases.append(
            {
                "case_id": f"CASE_{index:02d}",
                "category": "C_INVALID_RAW_ACTION_SAFETY",
                "fixture": label,
                "model_invoked": False,
                "model": "injected-invalid-response-fixture",
                "temperature": 0.0,
                "prompt_chars": selector.last_record.get("prompt_chars"),
                "latency_s": time.monotonic() - started,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "raw_response": raw,
                "parsed_selection": selection.option_id,
                "raw_selection_value": raw_choice,
                "valid_schema": structural_schema_valid,
                "valid_option": raw_choice in allowed,
                "expected_option": "INVALID_SELECTION",
                "semantic_correct": selection.option_id == "INVALID_SELECTION",
                "selection_status": selection.status,
            }
        )
    return cases


def main() -> int:
    args = _parser().parse_args()
    if not 1 <= args.max_output_tokens <= 512:
        raise SystemExit("--max-output-tokens must be between 1 and 512")
    cfg = load_yaml(args.config)
    load_secrets_env()
    vlm_args = argparse.Namespace(
        task_suite_name=None,
        task_id=None,
        episode_index=None,
        max_steps=None,
        loop_period_s=None,
        log_dir=None,
        vlm_backend=None,
        vlm_url=args.vlm_url,
        model=args.model,
    )
    resolved = build_config(vlm_args, cfg)
    client = make_vlm_client(vlm_args, resolved)
    client.health_check(wait_s=0.0)
    selector = QwenSelectorAdapter(client, "", max_tokens=args.max_output_tokens)
    temperature = float(resolved["vlm"].get("temperature", 0.0))
    cases: list[dict[str, Any]] = []
    for index in range(12):
        task, state, options, expected, category, pair_id = _choices_for_case(index)
        selector.task_instruction = task
        selection = selector.select(state, options)
        allowed = {option.option_id for option in options} | {"REOBSERVE", "ABORT"}
        selected = selection.option_id
        record = dict(selector.last_record)
        raw_response = record.get("raw_model_response", "")
        schema_valid, raw_choice = _raw_selection(raw_response)
        option_valid = raw_choice in allowed
        cases.append(
            {
                "case_id": f"CASE_{index + 1:02d}",
                "category": category,
                "pair_id": pair_id,
                "task_instruction": task,
                "compact_state": {
                    "task_id": state.task_id,
                    "stage": state.stage,
                    "target_identity": state.target_identity,
                    "target_confidence": state.target_confidence,
                    "contact_state": state.contact_state,
                    "uncertainty": state.uncertainty,
                    "relevant_geometry": dict(state.relevant_geometry),
                },
                "options": [
                    {"option_id": option.option_id, "description": option.description,
                     "expected_effect": dict(option.expected_effect)}
                    for option in options
                ],
                "model_invoked": True,
                "model": record.get("model"),
                "temperature": temperature,
                "prompt_chars": record.get("prompt_chars"),
                "latency_s": record.get("latency_s"),
                "max_output_tokens": record.get("max_tokens"),
                "output_tokens": record.get("output_tokens"),
                "reasoning_tokens": record.get("reasoning_tokens"),
                "finish_reason": record.get("finish_reason"),
                "raw_response": raw_response,
                "error": record.get("error"),
                "parsed_selection": selected,
                "raw_selection_value": raw_choice,
                "valid_schema": schema_valid,
                "valid_option": option_valid,
                "expected_option": expected,
                "semantic_correct": selected == expected,
                "selection_status": selection.status,
            }
        )
    cases.extend(_safety_cases())

    live_cases = [case for case in cases if case["model_invoked"]]
    denominator = max(1, len(live_cases))
    latencies = [float(case["latency_s"]) for case in live_cases if case.get("latency_s") is not None]
    output_tokens = [int(case["output_tokens"]) for case in live_cases
                     if isinstance(case.get("output_tokens"), int)]
    reasoning_tokens = [int(case["reasoning_tokens"]) for case in live_cases
                        if isinstance(case.get("reasoning_tokens"), int)]
    summary = {
        "model": resolved["vlm"]["model"],
        "temperature": temperature,
        "max_output_tokens": args.max_output_tokens,
        "endpoint": resolved["vlm"]["base_url"],
        "live_model_cases": len(live_cases),
        "interface_safety_cases": len(cases) - len(live_cases),
        "total_cases": len(cases),
        "schema_valid_rate": sum(case["valid_schema"] for case in live_cases) / denominator,
        "option_valid_rate": sum(case["valid_option"] for case in live_cases) / denominator,
        "selection_accuracy": sum(case["semantic_correct"] for case in live_cases) / denominator,
        "average_latency_s": sum(latencies) / len(latencies) if latencies else None,
        "average_output_tokens": sum(output_tokens) / len(output_tokens) if output_tokens else None,
        "average_reasoning_tokens": sum(reasoning_tokens) / len(reasoning_tokens) if reasoning_tokens else None,
        "metric_definitions": {
            "schema_valid": "raw output parses as exactly one JSON object with the single string key 'selection'",
            "option_valid": "raw selection value is a current option ID, REOBSERVE, or ABORT",
            "selection_accuracy": "adapter's fail-closed selected ID equals the expected ID",
        },
        "real_robot_actions_executed": 0,
        "executor_constructed": False,
        "executor_constructed": False,
        "cases": cases,
    }
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "cases"},
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Selector evaluation: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Strict text-only adapter from the existing VLMClient to bounded V3 options."""

from __future__ import annotations

import json
import time
from typing import Any, Sequence

from core.runtime_v3.options import RuntimeOption
from core.runtime_v3.selector import Selection
from core.runtime_v3.state import BeliefState


class QwenSelectorAdapter:
    """Keep only the VLM client and instruction; never retain env/controller/backend."""

    def __init__(self, client: Any, task_instruction: str) -> None:
        self.client = client
        self.task_instruction = str(task_instruction)
        self.model = str(getattr(client, "model", "unknown"))
        self.last_record: dict[str, Any] = {}

    def select(self, state: BeliefState, options: Sequence[RuntimeOption]) -> Selection:
        allowed = [option.option_id for option in options] + ["REOBSERVE", "ABORT"]
        compact = {
            "task_id": state.task_id,
            "step_id": state.step_id,
            "stage": state.stage,
            "target_identity": state.target_identity,
            "target_confidence": state.target_confidence,
            "holding_state": state.holding_state,
            "contact_state": state.contact_state,
            "uncertainty": state.uncertainty,
            "options": [
                {
                    "option_id": option.option_id,
                    "option_type": option.option_type,
                    "description": option.description,
                    "preconditions": dict(option.preconditions),
                    "expected_effect": dict(option.expected_effect),
                    "confidence": option.confidence,
                }
                for option in options
            ],
        }
        prompt = (
            "Select one bounded runtime option using the task instruction and current state. "
            "Do not output robot commands, trajectories, or prose. Return exactly one JSON "
            "object with the single key selection. Its value must be one of the allowed IDs.\n"
            f"Task instruction: {self.task_instruction}\n"
            f"Compact state and options: {json.dumps(compact, ensure_ascii=False, separators=(',', ':'))}\n"
            f"Allowed selections: {json.dumps(allowed)}"
        )
        schema = {
            "type": "object",
            "properties": {"selection": {"type": "string", "enum": allowed}},
            "required": ["selection"],
            "additionalProperties": False,
        }
        started = time.monotonic()
        raw_model_response = ""
        parsed_selection = None
        error = None
        try:
            response = self.client.complete_json(
                prompt,
                agentview_image=None,
                wrist_image=None,
                schema=schema,
                max_tokens=64,
                temperature=0.0,
                chat_template_kwargs={"enable_thinking": False, "thinking": False},
                debug=True,
            )
            raw_model_response = self._raw_content(response)
            parsed = json.loads(response.raw_text)
            if isinstance(parsed, dict) and set(parsed) == {"selection"}:
                parsed_selection = parsed.get("selection")
            if isinstance(parsed_selection, str) and parsed_selection in allowed:
                choice = parsed_selection
                status = "REOBSERVE" if choice == "REOBSERVE" else (
                    "ABORT" if choice == "ABORT" else "SELECTED"
                )
            else:
                choice, status = "INVALID_SELECTION", "INVALID_SELECTION"
        except Exception as exc:
            raw_model_response = str(getattr(exc, "raw_text", "") or raw_model_response)
            error = f"{type(exc).__name__}: {exc}"
            choice, status = "INVALID_SELECTION", "INVALID_SELECTION"
        latency_s = time.monotonic() - started
        self.last_record = {
            "model": self.model,
            "raw_model_response": raw_model_response,
            "parsed_selection": parsed_selection,
            "selection_valid": status != "INVALID_SELECTION",
            "latency_s": latency_s,
            "prompt_chars": len(prompt),
            "option_count": len(options),
            "error": error,
        }
        return Selection(
            option_id=choice,
            status=status,
            raw_output=raw_model_response,
            parsed_selection=parsed_selection if isinstance(parsed_selection, str) else None,
            prompt_chars=len(prompt),
            latency_s=latency_s,
        )

    @staticmethod
    def _raw_content(response: Any) -> str:
        payload = getattr(response, "payload", {}) or {}
        raw = payload.get("raw", {}) if isinstance(payload, dict) else {}
        try:
            content = raw["choices"][0]["message"]["content"]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(str(part.get("text", "")) for part in content
                                  if isinstance(part, dict))
        except (KeyError, IndexError, TypeError):
            pass
        return str(getattr(response, "raw_text", ""))

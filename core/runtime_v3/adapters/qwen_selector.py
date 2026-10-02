"""Strict text-only adapter from the existing VLMClient to bounded V3 options."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Mapping, Sequence

import numpy as np

from core.runtime_v3.options import RuntimeOption
from core.runtime_v3.selector import Selection
from core.runtime_v3.state import BeliefState


class QwenSelectorAdapter:
    """Keep only the VLM client and instruction; never retain env/controller/backend."""

    def __init__(self, client: Any, task_instruction: str, *, max_tokens: int = 64) -> None:
        self.client = client
        self.task_instruction = str(task_instruction)
        self.model = str(getattr(client, "model", "unknown"))
        self.max_tokens = int(max_tokens)
        if not 1 <= self.max_tokens <= 512:
            raise ValueError("max_tokens must be in the bounded range [1, 512]")
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
            "relevant_geometry": {
                key: state.relevant_geometry[key]
                for key in (
                    "target_offset_xyz_m", "target_relation", "target_visible",
                    "approach_complete", "grasp_preconditions_met",
                )
                if key in state.relevant_geometry
            },
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
        response_payload: dict[str, Any] = {}
        try:
            response = self.client.complete_json(
                prompt,
                agentview_image=None,
                wrist_image=None,
                schema=schema,
                max_tokens=self.max_tokens,
                temperature=0.0,
                chat_template_kwargs={"enable_thinking": False, "thinking": False},
                debug=True,
            )
            response_payload = getattr(response, "payload", {}) or {}
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
            response_payload = getattr(exc, "payload", {}) or {}
            error = f"{type(exc).__name__}: {exc}"
            choice, status = "INVALID_SELECTION", "INVALID_SELECTION"
        latency_s = time.monotonic() - started
        usage = response_payload.get("usage", {}) if isinstance(response_payload, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
        details = usage.get("completion_tokens_details", {})
        if not isinstance(details, dict):
            details = {}
        raw_payload = response_payload.get("raw", {}) if isinstance(response_payload, dict) else {}
        try:
            finish_reason = raw_payload["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError):
            finish_reason = None
        self.last_record = {
            "model": self.model,
            "raw_model_response": raw_model_response,
            "parsed_selection": parsed_selection,
            "selection_valid": status != "INVALID_SELECTION",
            "latency_s": latency_s,
            "prompt_chars": len(prompt),
            "option_count": len(options),
            "max_tokens": self.max_tokens,
            "output_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens", details.get("reasoning_tokens")),
            "prompt_tokens": usage.get("prompt_tokens"),
            "finish_reason": finish_reason,
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

    def select_visual_semantic(
        self,
        state: BeliefState,
        agentview_image: Any,
        semantic_options: Sequence[Mapping[str, str]],
    ) -> Selection:
        """No-action high-resolution visual smoke for semantic option selection.

        This method only retains the VLM client and task instruction. It accepts
        semantic option labels, sends the supplied image without resizing, and
        returns a selection value; it has no controller or backend path.
        """
        image = np.asarray(agentview_image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("agentview_image must be an HxWx3 RGB array")
        options = [
            {"option_id": str(item.get("option_id", "")),
             "description": str(item.get("description", ""))}
            for item in semantic_options
        ]
        allowed = [item["option_id"] for item in options]
        if not 2 <= len(options) <= 3 or any(not value for value in allowed) or len(set(allowed)) != len(allowed):
            raise ValueError("visual semantic smoke requires two or three unique option IDs")
        physical = re.compile(r"\b(left|right|forward|back(?:ward)?|up|down)\b|MV_(?:LEFT|RIGHT|FWD|BACK|UP|DOWN)", re.I)
        if any(physical.search(item["option_id"] + " " + item["description"]) for item in options):
            raise ValueError("visual semantic options cannot expose physical directions")
        relative = state.object_relative_state
        compact = {
            "task_id": state.task_id,
            "step_id": state.step_id,
            "stage": state.stage,
            "target_phrase": relative.target_phrase if relative else state.target_identity,
            "target_visible": relative.target_visible if relative else None,
            "target_centroid_px": relative.target_centroid_px if relative else None,
            "eef_projection_px": relative.eef_projection_px if relative else None,
            "image_error_px": relative.image_error_px if relative else None,
            "image_error_norm_px": relative.image_error_norm_px if relative else None,
            "options": options,
        }
        prompt = (
            "Choose one semantic option using the task instruction, image, and compact state. "
            "The options describe semantic actions only. Do not provide or infer any robot "
            "direction, trajectory, or actuator command. Return only JSON with key selection.\n"
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
        error = None
        payload: dict[str, Any] = {}
        parsed_selection = None
        try:
            response = self.client.complete_json(
                prompt,
                agentview_image=image,
                wrist_image=None,
                schema=schema,
                max_tokens=self.max_tokens,
                temperature=0.0,
                chat_template_kwargs={"enable_thinking": False, "thinking": False},
                debug=True,
                agentview_label="Image A: 512-class raw LIBERO agentview RGB",
                image_detail="high",
            )
            payload = getattr(response, "payload", {}) or {}
            raw_model_response = self._raw_content(response)
            parsed = json.loads(response.raw_text)
            if isinstance(parsed, dict) and set(parsed) == {"selection"}:
                parsed_selection = parsed.get("selection")
            if isinstance(parsed_selection, str) and parsed_selection in allowed:
                choice, status = parsed_selection, "SELECTED"
            else:
                choice, status = "INVALID_SELECTION", "INVALID_SELECTION"
        except Exception as exc:
            raw_model_response = str(getattr(exc, "raw_text", "") or raw_model_response)
            payload = getattr(exc, "payload", {}) or {}
            error = f"{type(exc).__name__}: {exc}"
            choice, status = "INVALID_SELECTION", "INVALID_SELECTION"
        audit = payload.get("request_audit", {}) if isinstance(payload, dict) else {}
        audited_images = audit.get("images", []) if isinstance(audit, dict) else []
        latency_s = time.monotonic() - started
        self.last_record = {
            "model": self.model,
            "raw_model_response": raw_model_response,
            "parsed_selection": parsed_selection,
            "selection_valid": status == "SELECTED",
            "latency_s": latency_s,
            "prompt_chars": len(prompt),
            "option_count": len(options),
            "semantic_options": options,
            "max_tokens": self.max_tokens,
            "source_width": int(image.shape[1]),
            "source_height": int(image.shape[0]),
            "model_input_width": (int(audited_images[0]["size"][0])
                                  if audited_images and audited_images[0].get("size") else None),
            "model_input_height": (int(audited_images[0]["size"][1])
                                   if audited_images and audited_images[0].get("size") else None),
            "request_audit": audit,
            "robot_action_executed": 0,
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

from __future__ import annotations

import json
from typing import Any, Sequence

from core.prompting.wrist_marker import parse_wrist_marker, wrist_marker_prompt

from .vlm_client import VLMParseError, VLMResponse, recover_allowed_token


DIRECTION_TOKENS = (
    "MV_FWD",
    "MV_BACK",
    "MV_LEFT",
    "MV_RIGHT",
    "MV_UP",
    "MV_DOWN",
)
CONTROLLER_TOKENS = DIRECTION_TOKENS + ("GRASP", "RELEASE", "DONE")
TOKEN_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False, "thinking": False}


def _make_pregrasp_temporal_panel(agentview, wrist, previous_agentview=None, previous_wrist=None):
    """Return a small labeled temporal panel while preserving raw wrist input."""
    try:
        import numpy as np
        from PIL import Image, ImageDraw

        def as_image(value):
            if value is None:
                return Image.new("RGB", (256, 256), "#777777")
            image = Image.fromarray(np.asarray(value).astype("uint8"))
            return image.convert("RGB").resize((256, 256))

        tiles = [as_image(previous_agentview), as_image(agentview), as_image(previous_wrist), as_image(wrist)]
        panel = Image.new("RGB", (512, 512), "white")
        labels = ("AGENTVIEW BEFORE", "AGENTVIEW NOW", "WRIST BEFORE", "WRIST NOW")
        for index, (tile, label) in enumerate(zip(tiles, labels)):
            x = (index % 2) * 256
            y = (index // 2) * 256
            panel.paste(tile, (x, y))
            ImageDraw.Draw(panel).rectangle((x + 2, y + 2, x + 150, y + 18), fill="white")
            ImageDraw.Draw(panel).text((x + 4, y + 4), label, fill="black")
        return np.asarray(panel)
    except Exception:
        return agentview


def append_pregrasp_target_crop(panel, current_image, bbox_xyxy, camera: str):
    """Append a proportional target-region crop to the exact panel sent to Qwen.

    The crop is only an enlargement of current raw pixels. It uses the live
    detector box and scales its context margin with that box; it adds no
    synthetic evidence or scene-specific coordinates.
    """
    try:
        import math
        import numpy as np
        from PIL import Image, ImageDraw

        source = np.asarray(current_image)
        if source.ndim != 3 or source.shape[2] != 3:
            return panel, None
        x0, y0, x1, y1 = (float(value) for value in bbox_xyxy)
        height, width = source.shape[:2]
        if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
            return panel, None
        x0, x1 = max(0.0, x0), min(float(width), x1)
        y0, y1 = max(0.0, y0), min(float(height), y1)
        box_width, box_height = x1 - x0, y1 - y0
        if box_width < 2.0 or box_height < 2.0:
            return panel, None
        # Keep context around the contact area, with crop scale following the
        # current object box instead of a fixed pixel/object-size rule.
        side = int(math.ceil(max(box_width, box_height) * 2.5))
        center_x, center_y = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        left, top = int(math.floor(center_x - side / 2)), int(math.floor(center_y - side / 2))
        right, bottom = left + side, top + side
        crop_canvas = Image.new("RGB", (side, side), "#555555")
        clipped_left, clipped_top = max(0, left), max(0, top)
        clipped_right, clipped_bottom = min(width, right), min(height, bottom)
        if clipped_right <= clipped_left or clipped_bottom <= clipped_top:
            return panel, None
        crop = Image.fromarray(source.astype("uint8"), mode="RGB").crop(
            (clipped_left, clipped_top, clipped_right, clipped_bottom)
        )
        crop_canvas.paste(crop, (clipped_left - left, clipped_top - top))
        tile = crop_canvas.resize((256, 256), Image.Resampling.LANCZOS)

        base = Image.fromarray(np.asarray(panel).astype("uint8")).convert("RGB")
        if base.width < 512:
            padded = Image.new("RGB", (512, base.height), "white")
            padded.paste(base, ((512 - base.width) // 2, 0))
            base = padded
        output = Image.new("RGB", (base.width, base.height + 256), "white")
        output.paste(base, (0, 0))
        output.paste(tile, (0, base.height))
        note = Image.new("RGB", (base.width - 256, 256), "white")
        draw = ImageDraw.Draw(note)
        draw.text((8, 12), f"CURRENT {str(camera).upper()} ROI", fill="black")
        draw.text((8, 32), "Proportional raw-pixel crop; no details added.", fill="black")
        draw.text((8, 52), "Use the full views above for gripper context.", fill="black")
        output.paste(note, (256, base.height))
        metadata = {
            "camera": str(camera).lower(),
            "bbox_xyxy": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
            "crop_xyxy_unclipped": [left, top, right, bottom],
            "source_image_size": [int(width), int(height)],
            "crop_scale_from_bbox": 2.5,
        }
        return np.asarray(output), metadata
    except (TypeError, ValueError, IndexError, OSError):
        return panel, None


def append_fresh_pregrasp_target_crop(
    panel,
    target_track: dict[str, Any] | None,
    current_frame_id: int,
    *,
    agentview_image=None,
    wrist_image=None,
):
    """Append a target crop only when the live identity and bbox share this frame."""
    if not isinstance(target_track, dict):
        return panel, None
    try:
        if (
            not target_track.get("instance_id")
            or int(target_track.get("last_confirmed_frame", -1)) != int(current_frame_id)
        ):
            return panel, None
    except (TypeError, ValueError):
        return panel, None
    camera = str(target_track.get("camera") or "").lower()
    source = wrist_image if camera == "wrist" else agentview_image if camera == "agentview" else None
    if source is None or target_track.get("bbox_xyxy") is None:
        return panel, None
    return append_pregrasp_target_crop(panel, source, target_track["bbox_xyxy"], camera)


def _compact_spatial_prompt(value: Any) -> dict[str, Any]:
    """Keep provider diagnostics out of the small VLM context window."""
    if not isinstance(value, dict):
        return {"health": "UNKNOWN", "relations": ["UNKNOWN"]}
    compact = {
        key: value.get(key)
        for key in (
            "health",
            "instance_id",
            "frame_id",
            "fused_relative_xyz",
            "relations",
            "uncertainty",
            "agreeing_sources",
            "conflicting_sources",
        )
        if key in value
    }
    compact.setdefault("health", "UNKNOWN")
    compact.setdefault("relations", ["UNKNOWN"])
    return compact


def _compact_memory_prompt(entries: Sequence[Any], limit: int = 3) -> list[Any]:
    """Summarize only the temporal/action fields needed for a correction."""
    result: list[dict[str, Any]] = []
    for entry in list(entries or [])[-max(1, int(limit)) :]:
        if not isinstance(entry, dict):
            result.append(str(entry)[:120])
            continue
        row = {
            key: entry.get(key)
            for key in (
                "frame_id", "before_frame_id", "route_phase",
                "requested_action", "authorized_action", "executed_action",
                "motion_delta", "predicted_effect", "observed_effect",
                "effect_status", "tags",
            )
            if key in entry
        }
        placement = entry.get("placement_summary")
        if isinstance(placement, dict):
            row["placement_summary"] = {
                key: placement.get(key)
                for key in ("expected_effect", "route_uncertainty_m", "evidence_sources")
                if key in placement
            }
        decision = entry.get("decision_summary")
        if isinstance(decision, dict):
            row["prior_agent_hypothesis_unverified"] = {
                key: decision.get(key)
                for key in (
                    "status", "relation", "selected_option", "reasoning", "missing_observation",
                    "expected_effect", "failure_condition", "frame_id",
                    "expires_after_frame",
                )
                if key in decision
            }
        result.append(row)
    return result


def _critical_reasoning_settings(
    client: Any,
    enabled: bool,
    default_tokens: int,
    *,
    budget_cap: int | None = None,
    final_reserve_tokens: int | None = None,
) -> tuple[dict[str, bool], int, float, int | None]:
    """Enable deliberate reasoning only on critical events and a Thinking model."""
    model = str(getattr(client, "model", ""))
    is_thinking_checkpoint = model.rstrip("/").endswith("Qwen3-VL-8B-Thinking")
    active = bool(enabled and is_thinking_checkpoint)
    if not active:
        return TOKEN_CHAT_TEMPLATE_KWARGS, int(default_tokens), 0.0, None
    budget = int(getattr(client, "cot_max_tokens", 0) or 0) or int(default_tokens)

    if budget_cap is not None:
        budget = min(budget, max(1, int(budget_cap)))
    configured_thinking_budget = int(
        getattr(client, "thinking_token_budget", 0) or 1024
    )
    # vLLM's thinking budget ends the reasoning section while leaving room for
    # the schema-constrained final answer within max_tokens.
    final_reserve = (
        min(512, max(128, budget // 4))
        if final_reserve_tokens is None
        else max(1, int(final_reserve_tokens))
    )
    thinking_budget = min(
        configured_thinking_budget, max(1, budget - final_reserve)
    )
    return {"enable_thinking": True}, budget, 1.0, thinking_budget


def _join_prompt_parts(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def _temporal_pair_image(before, after, *, before_frame_id=None, after_frame_id=None):
    """Make a labeled, lossless side-by-side image pair for one camera."""
    from PIL import Image, ImageDraw
    import numpy as np

    def to_rgb(value):
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError("temporal grasp panel needs an RGB image")
        return Image.fromarray(np.asarray(array[..., :3], dtype=np.uint8), mode="RGB")

    left = to_rgb(before)
    right = to_rgb(after)
    height = max(left.height, right.height)
    width = max(left.width, right.width)
    if left.size != (width, height):
        left = left.resize((width, height), Image.Resampling.BILINEAR)
    if right.size != (width, height):
        right = right.resize((width, height), Image.Resampling.BILINEAR)
    header = 24
    panel = Image.new("RGB", (width * 2, height + header), (18, 22, 28))
    panel.paste(left, (0, header))
    panel.paste(right, (width, header))
    draw = ImageDraw.Draw(panel)
    draw.text((5, 5), f"BEFORE close · frame {before_frame_id}", fill=(240, 240, 240))
    draw.text((width + 5, 5), f"AFTER close · frame {after_frame_id}", fill=(240, 240, 240))
    return panel


def _default_output_contract(extra_tokens: Sequence[str] = ()) -> str:
    """The controller's default answer protocol: one atomic-action token as JSON.

    This is the output contract that used to live inside ``prompts/controller.txt``; it
    now lives in the role so an answer-protocol tool (e.g. ``plugins.mcq``) can swap it
    without editing the prompt body. Protocol plugins own their own contract text.

    ``extra_tokens`` appends optional action tokens contributed by a tool (e.g.
    ``plugins.rotation``'s ROTATE_CW/CCW), so the offered set matches ``allowed_tokens``.
    """
    tokens = tuple(CONTROLLER_TOKENS) + tuple(extra_tokens)
    return (
        "Choose exactly one action:\n"
        + ", ".join(tokens)
        + '\nReturn JSON only: {"decision":"ONE_ACTION","reasoning":"one visual sentence"}'
    )


def _without_json_output_contract(prompt: str) -> str:
    lines = []
    skip_fields_line = False
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("Return JSON only"):
            skip_fields_line = True
            continue
        if skip_fields_line and stripped:
            skip_fields_line = False
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _token_only_prompt(prompt: str, allowed_tokens: Sequence[str]) -> str:
    clean_prompt = _without_json_output_contract(prompt)
    return (
        "Answer with exactly one token from this list and nothing else:\n"
        + " ".join(allowed_tokens)
        + "\n\nUse the task, stage, image, and rules below only to choose that token.\n\n"
        + clean_prompt
        + "\n\nFinal answer: exactly one allowed token, no JSON, no markdown, no prose."
    )


def _decision_schema(allowed_tokens: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": list(allowed_tokens)},
            "reasoning": {"type": "string"},
        },
        "required": ["decision", "reasoning"],
        "additionalProperties": False,
    }


def _complete_decision_json(
    client: Any,
    prompt: str,
    allowed_tokens: Sequence[str],
    agentview_image,
    wrist_image=None,
    fallback_token: str | None = None,
    fallback_reason: str = "fallback after invalid decision JSON and token retry",
    debug: bool = False,
) -> VLMResponse:
    json_prompt = (
        prompt
        + '\n\nReturn JSON only. Put "decision" first, then "reasoning". '
        + 'Use one short sentence for "reasoning".'
    )
    try:
        response = client.complete_json(
            json_prompt,
            agentview_image,
            wrist_image=wrist_image,
            schema=_decision_schema(allowed_tokens),
            max_tokens=None,
            temperature=0.0,
            chat_template_kwargs=TOKEN_CHAT_TEMPLATE_KWARGS,
            debug=debug,
        )
        payload = response.payload.get("json")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Decision JSON missing object payload: {response.raw_text!r}")
        decision = str(payload.get("decision", "")).strip()
        if decision not in allowed_tokens:
            raise RuntimeError(
                f"Decision JSON has invalid decision {decision!r}; "
                f"allowed tokens are {list(allowed_tokens)}"
            )
        reasoning = str(payload.get("reasoning") or "").strip()
        normalized = {"decision": decision, "reasoning": reasoning}
        merged_payload = dict(response.payload)
        merged_payload["json"] = normalized
        return VLMResponse(
            token=decision,
            raw_text=json.dumps(normalized, ensure_ascii=False, sort_keys=True),
            payload=merged_payload,
        )
    except RuntimeError as exc:
        recovered = _recover_decision_from_malformed_json(exc, allowed_tokens)
        if recovered:
            decision, raw_text = recovered
            normalized = {
                "decision": decision,
                "reasoning": (
                    f"Recovered {decision} from malformed VLM JSON output. "
                    f"Raw output: {_one_line(raw_text)}"
                ),
            }
            payload = {
                "json": normalized,
                "recovered_from_malformed_json": True,
                "json_error": str(exc),
                "latency_s": 0.0,
            }
            return VLMResponse(
                token=decision,
                raw_text=json.dumps(normalized, ensure_ascii=False, sort_keys=True),
                payload=payload,
            )
        return _token_retry_or_fallback(
            client=client,
            prompt=prompt,
            allowed_tokens=allowed_tokens,
            fallback_token=fallback_token,
            fallback_reason=fallback_reason,
            agentview_image=agentview_image,
            wrist_image=wrist_image,
            debug=debug,
            json_error=exc,
            primary_reasoning=_reasoning_from_exception(exc),
        )


def _complete_cot_decision(
    client: Any,
    prompt: str,
    allowed_tokens: Sequence[str],
    agentview_image,
    wrist_image=None,
    fallback_token: str | None = None,
    fallback_reason: str = "cot fallback after no recoverable token",
    debug: bool = False,
) -> VLMResponse:
    """Reasoner path: a reasoner only grounds spatially when it reasons BEFORE
    answering, so this does NOT use guided JSON (which forces the answer at token
    0). The prompt makes the model write a localisation paragraph and end with
    'FINAL: <TOKEN>'; we recover the token from that prose. On failure, fall back
    to the strict token retry."""
    # Chain-of-thought needs headroom, but an unbounded budget lets a verbose thinker
    # (Gemma) decode ~1024 tokens every step -- the dominant latency cost. A backend can
    # cap it via `cot_max_tokens`; otherwise keep the old 1024 floor. chat_template_kwargs={}
    # keeps the backend's own thinking setting (Gemma: enable_thinking=true -> <think>).
    # strip_reasoning=False preserves the CoT for analysis.
    cot_budget = int(getattr(client, "cot_max_tokens", 0) or 0) or max(
        1024, int(getattr(client, "max_tokens", 0) or 0)
    )
    # Per-backend reasoning directive controls how much the model thinks -- length is
    # prompt-driven, not a token cap. Brief for a verbose thinker (Gemma), more thorough
    # for a concise one. Empty -> use the backend prompt as-is.
    directive = str(getattr(client, "reasoning_directive", "") or "").strip()
    cot_prompt = f"{prompt}\n\n{directive}" if directive else prompt
    response = client.complete_text(
        cot_prompt,
        agentview_image,
        wrist_image=wrist_image,
        max_tokens=cot_budget,
        temperature=0.0,
        chat_template_kwargs={},
        strip_reasoning=False,
        debug=debug,
    )
    raw_text = response.raw_text or ""
    token = recover_allowed_token(raw_text, allowed_tokens)
    if token in allowed_tokens:
        # Keep the complete chain-of-thought; steps.json logs it in full and
        # steps.jsonl truncates its own copy for compactness.
        reasoning = _one_line(raw_text)
        normalized = {"decision": token, "reasoning": reasoning}
        payload = dict(response.payload)
        payload["json"] = normalized
        payload["cot"] = True
        return VLMResponse(
            token=token,
            raw_text=json.dumps(normalized, ensure_ascii=False, sort_keys=True),
            payload=payload,
        )
    return _token_retry_or_fallback(
        client=client,
        prompt=prompt,
        allowed_tokens=allowed_tokens,
        fallback_token=fallback_token,
        fallback_reason=fallback_reason,
        agentview_image=agentview_image,
        wrist_image=wrist_image,
        debug=debug,
        primary_reasoning=_one_line(raw_text),
    )


def _token_retry_or_fallback(
    client: Any,
    prompt: str,
    allowed_tokens: Sequence[str],
    fallback_token: str | None,
    fallback_reason: str,
    agentview_image,
    wrist_image=None,
    debug: bool = False,
    json_error: RuntimeError | None = None,
    extra_fields: dict[str, str] | None = None,
    primary_reasoning: str = "",
) -> VLMResponse:
    tokens = tuple(allowed_tokens)
    if not tokens:
        raise RuntimeError("No allowed tokens available for VLM fallback")
    fallback = fallback_token if fallback_token in tokens else None
    try:
        retry = client.complete_token(
            _token_only_prompt(prompt, tokens),
            tokens,
            agentview_image,
            wrist_image=wrist_image,
            chat_template_kwargs=TOKEN_CHAT_TEMPLATE_KWARGS,
            debug=debug,
        )
        # Keep the model's own reasoning from the first pass; the token just came from
        # a strict-retry call. Only fall back to the bare label when no reasoning exists.
        normalized = {
            "decision": retry.token,
            "reasoning": _retry_reasoning(
                primary_reasoning, "strict token retry after invalid JSON"
            ),
        }
        if extra_fields:
            normalized.update(extra_fields)
        payload: dict[str, Any] = {"json": normalized, "retry": retry.payload}
        if "latency_s" in retry.payload:
            payload["latency_s"] = retry.payload["latency_s"]
        if json_error is not None:
            payload["json_error"] = str(json_error)
        return VLMResponse(
            token=retry.token,
            raw_text=json.dumps(normalized, ensure_ascii=False, sort_keys=True),
            payload=payload,
        )
    except RuntimeError as retry_exc:
        if fallback is None:
            raise RuntimeError(
                "Invalid decision JSON and strict token retry failed; "
                f"json_error={json_error}; retry_error={retry_exc}"
            ) from retry_exc
        normalized = {
            "decision": fallback,
            "reasoning": _retry_reasoning(primary_reasoning, fallback_reason),
        }
        if extra_fields:
            normalized.update(extra_fields)
        payload = {
            "json": normalized,
            "fallback": True,
            "json_error": "" if json_error is None else str(json_error),
            "retry_error": str(retry_exc),
            "latency_s": 0.0,
        }
        return VLMResponse(
            token=fallback,
            raw_text=json.dumps(normalized, ensure_ascii=False, sort_keys=True),
            payload=payload,
        )


def _retry_reasoning(primary_reasoning: str, label: str) -> str:
    """The reasoning to log when the token came from a strict retry / commit fallback.
    Prefer the model's own first-pass reasoning, tagging how the token was recovered so
    the provenance is still visible; fall back to the bare label when none exists."""
    primary = (primary_reasoning or "").strip()
    if primary:
        return f"{primary} [{label}]"
    return label


def _reasoning_from_exception(exc: RuntimeError | None) -> str:
    """Best-effort recovery of the model's raw output text from a parse failure, so it
    can still be logged as the reasoning instead of a generic label."""
    raw = getattr(exc, "raw_text", "") or ""
    return _one_line(raw) if raw else ""


def _recover_decision_from_malformed_json(
    exc: RuntimeError, allowed_tokens: Sequence[str]
) -> tuple[str, str] | None:
    raw_text = str(getattr(exc, "raw_text", "") or "")
    if not raw_text and isinstance(exc, VLMParseError):
        raw_text = str(exc.raw_text or "")
    if not raw_text:
        return None
    token = recover_allowed_token(raw_text, allowed_tokens)
    if not token:
        return None
    return token, raw_text


def _one_line(value: Any) -> str:
    return " ".join(str(value).split())


class ControllerAgent:
    """Single per-step role: returns the executed base/grasp token directly."""

    def __init__(
        self,
        client: Any,
        prompt_template: str,
        common_context: str,
        transport_prompt_template: str | None = None,
        cot_mode: bool = False,
        gripper_color: str = "black",
        proprio_plugin: Any = None,
        mcq_plugin: Any = None,
        mem_text_plugin: Any = None,
        variable_step_plugin: Any = None,
        action_chunk_plugin: Any = None,
        rotation_plugin: Any = None,
        affordance_plugin: Any = None,
        action_ablation_plugin: Any = None,
        table_height_m: float | None = None,
    ) -> None:
        self.client = client
        self.prompt_template = prompt_template
        self.transport_prompt_template = transport_prompt_template
        self.common_context = common_context
        self.cot_mode = cot_mode
        # Physical color of the gripper as seen in the camera views, substituted for
        # the prompt's {gripper_color} placeholder (default: "black").
        self.gripper_color = str(gripper_color or "black")
        # Optional controller plugins. proprio_plugin is a context provider (renders the
        # {proprio} block); mcq_plugin is an answer protocol (swaps {output_contract} and
        # the allowed answer tokens). Both default to off -> today's behaviour, so callers
        # that don't pass them are unaffected. table_height_m is
        # the constant the proprio tool needs and is fixed for the episode.
        self.proprio_plugin = proprio_plugin
        self.mcq_plugin = mcq_plugin
        # mem_text_plugin (context provider) owns the move-memory text: the {mem_text}
        # "Recent moves" line and the {mem_text_rules} history bullets. Injected like the
        # others; None -> those placeholders render empty (no move history in the prompt).
        self.mem_text_plugin = mem_text_plugin
        # Wrist-visibility consumers. variable_step_plugin (shared with the controller) sizes
        # the step from the TARGET's wrist visibility; action_chunk_plugin repeats a move while
        # the TARGET is far. Both read the shared "WRIST: YES/NO" judgment, so whenever EITHER
        # is enabled we render the marker (core.prompting.wrist_marker) and parse the VLM's reply into
        # response.payload["target_in_wrist"] for the runner to forward.
        self.variable_step_plugin = variable_step_plugin
        self.action_chunk_plugin = action_chunk_plugin
        # Optional rotation plugin: offers ROTATE_CW/CCW as extra controller tokens and, once
        # the gripper has yawed, compensates wrist-judged MV_* moves for that yaw (the
        # compensation lives in the controller; here we only surface the tokens + prompt).
        # It also needs the shared WRIST: YES/NO judgment, so it joins _wants_wrist().
        self.rotation_plugin = rotation_plugin
        # Optional affordance tool (context provider): the runner grounds each stage's
        # contact point and annotates the AgentView; here it only mediates the AFFORD
        # field ("<part> = RED dot in AgentView" while a dot is active). None/disabled
        # -> the planner's affordance text passes through unchanged.
        self.affordance_plugin = affordance_plugin
        # Action-type ablation (plugins.action_ablation). Its letters modes are an answer
        # protocol on the same duck interface as mcq (and win over mcq when both are
        # on); every enabled mode also funnels the FULLY ASSEMBLED prompt through
        # filter_final so run-time text obeys the setting.
        self.action_ablation_plugin = action_ablation_plugin
        self.table_height_m = table_height_m
        # The most recent fully-rendered controller prompt, for periodic logging.
        self.last_prompt = ""

    def _active_protocol(self) -> Any:
        """The active answer-protocol tool (ablation letters > mcq > None)."""
        ablation = self.action_ablation_plugin
        if (
            ablation is not None
            and getattr(ablation, "enabled", False)
            and getattr(ablation, "answer_protocol", False)
        ):
            return ablation
        return (
            self.mcq_plugin
            if (self.mcq_plugin is not None and self.mcq_plugin.enabled)
            else None
        )

    def _wants_wrist(self) -> bool:
        """Whether any consumer needs the shared WRIST: YES/NO wrist-visibility judgment."""
        return bool(
            getattr(self.variable_step_plugin, "enabled", False)
            or getattr(self.action_chunk_plugin, "enabled", False)
            or getattr(self.rotation_plugin, "enabled", False)
        )

    def verify_grasp(
        self,
        *,
        task: str,
        target: str,
        affordance: str,
        agentview_image,
        wrist_image=None,
        before_agentview_image=None,
        before_wrist_image=None,
        before_frame_id: Optional[int] = None,
        after_frame_id: Optional[int] = None,
        reasoning_enabled: bool = False,
        debug: bool = False,
    ) -> dict[str, Any]:
        """Ask the VLM once whether the just-commanded close visually holds.

        This is deliberately a semantic visual check, separate from the atomic action
        policy. It does not use a gripper-width threshold and does not require Wrist to
        show the object: AgentView is the global view, while Wrist is supplementary and
        may be occluded by the gripper mount.
        """
        temporal = bool(before_agentview_image is not None or before_wrist_image is not None)
        if temporal:
            agentview_image = _temporal_pair_image(
                before_agentview_image if before_agentview_image is not None else agentview_image,
                agentview_image,
                before_frame_id=before_frame_id,
                after_frame_id=after_frame_id,
            )
            if wrist_image is not None:
                wrist_image = _temporal_pair_image(
                    before_wrist_image if before_wrist_image is not None else wrist_image,
                    wrist_image,
                    before_frame_id=before_frame_id,
                    after_frame_id=after_frame_id,
                )
        v22 = bool(reasoning_enabled)
        prompt = (
            "You are the high-level post-action grasp verifier for a robot task.\n"
            f"TASK: {task}\n"
            f"TARGET: {target}\n"
            f"AFFORDANCE: {affordance}\n\n"
            "A GRASP command has just closed the gripper. Inspect both live images. "
            "AgentView is authoritative for the global relation between the EEF, target, "
            "and scene; Wrist is supplementary local evidence. The Wrist camera is mounted "
            "on/above the gripper and can be occluded at contact. Do not answer NO merely "
            "because the target is absent or clipped in Wrist. Answer YES when the target "
            "is visibly enclosed between the fingers and its body is supported by the closed "
            "gripper rather than merely appearing underneath it. Do not treat a bottle that "
            "is still on the table below the hand, or a coincidental 2D overlap, as a held "
            "grasp. Answer NO only when the target is visibly outside/on the table or the "
            "closed gripper is clearly empty. Answer UNKNOWN when the views cannot separate "
            "a real hold from contact/occlusion; do not convert uncertainty into YES. "
            "Answer UNKNOWN when the views do not contain enough evidence. Do not use any "
            "fixed gripper-width or object-size threshold."
            + (
                (
                    "Image A and B are temporal panels (left is before close, right is after close). "
                    if temporal else "These are current dual-camera views without a pre-close panel. "
                )
                + "A YES is only a grasp candidate, not proof of a stable hold. Also assess whether "
                "the short vertical path above the closed fingers is visibly clear for exactly "
                "one small diagnostic lift; say CLEAR only if both views support clearance, "
                "otherwise BLOCKED or UNKNOWN. Do not prescribe coordinates or other actions. "
                f"Cite exact frame ids ({before_frame_id}, {after_frame_id}) and camera names; "
                "a YES must cite current post-close evidence.\n\n"
                'Return JSON only: {"grasped":"YES|NO|UNKNOWN",'
                '"diagnostic_lift":"CLEAR|BLOCKED|UNKNOWN",'
                '"evidence_for":[{"frame_id":int,"camera":"agentview|wrist","observation":"..."}],'
                '"evidence_against":[{"frame_id":int,"camera":"agentview|wrist","observation":"..."}],'
                '"reasoning":"brief decision summary"}'
                if v22
                else "\n\nReturn JSON only: {\"grasped\":\"YES|NO|UNKNOWN\","
                "\"reasoning\":\"one short visual sentence\"}"
            )
        )
        schema = {
            "type": "object",
            "properties": {
                "grasped": {"type": "string", "enum": ["YES", "NO", "UNKNOWN"]},
                **(
                    {
                        "diagnostic_lift": {"type": "string", "enum": ["CLEAR", "BLOCKED", "UNKNOWN"]},
                        "evidence_for": {
                            "type": "array", "items": {
                                "type": "object", "properties": {
                                    "frame_id": {"type": "integer"},
                                    "camera": {"type": "string", "enum": ["agentview", "wrist"]},
                                    "observation": {"type": "string"},
                                }, "required": ["frame_id", "camera", "observation"],
                                "additionalProperties": False,
                            },
                        },
                        "evidence_against": {
                            "type": "array", "items": {
                                "type": "object", "properties": {
                                    "frame_id": {"type": "integer"},
                                    "camera": {"type": "string", "enum": ["agentview", "wrist"]},
                                    "observation": {"type": "string"},
                                }, "required": ["frame_id", "camera", "observation"],
                                "additionalProperties": False,
                            },
                        },
                        "reasoning": {"type": "string"},
                    }
                    if v22 else {"reasoning": {"type": "string"}}
                ),
            },
            "required": (
                ["grasped", "diagnostic_lift", "evidence_for", "evidence_against", "reasoning"]
                if v22 else ["grasped", "reasoning"]
            ),
            "additionalProperties": False,
        }
        try:
            reasoning_kwargs, max_tokens, temperature, thinking_budget = _critical_reasoning_settings(
                self.client, v22, 128
            )
            response = self.client.complete_json(
                prompt,
                agentview_image,
                wrist_image=wrist_image,
                schema=schema,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking_token_budget=thinking_budget,
                chat_template_kwargs=reasoning_kwargs,
                debug=debug,
                agentview_label=("AgentView before | after GRASP" if temporal else "AgentView after GRASP"),
                wrist_label=("Wrist before | after GRASP (supplementary; may be occluded)" if temporal else "Wrist after GRASP (supplementary; may be occluded)"),
            )
            payload = response.payload.get("json")
            if not isinstance(payload, dict):
                raise RuntimeError("grasp verifier returned no JSON object")
            verdict = str(payload.get("grasped", "UNKNOWN")).strip().upper()
            if verdict not in {"YES", "NO", "UNKNOWN"}:
                verdict = "UNKNOWN"
            evidence_for = payload.get("evidence_for", []) if v22 else []
            evidence_against = payload.get("evidence_against", []) if v22 else []
            citations_valid = False
            current_cited = False
            if v22:
                allowed_frames = {
                    int(value) for value in (before_frame_id, after_frame_id)
                    if value is not None
                }

                def valid_citations(items):
                    if not isinstance(items, list):
                        return False
                    for item in items:
                        if not isinstance(item, dict):
                            return False
                        try:
                            cited_frame = int(item.get("frame_id", -1))
                        except (TypeError, ValueError):
                            return False
                        if (
                            cited_frame not in allowed_frames
                            or str(item.get("camera", "")).lower() not in {"agentview", "wrist"}
                            or not str(item.get("observation", "")).strip()
                        ):
                            return False
                    return True

                citations_valid = valid_citations(evidence_for) and valid_citations(evidence_against)
                if after_frame_id is not None and isinstance(evidence_for, list):
                    for item in evidence_for:
                        if isinstance(item, dict):
                            try:
                                current_cited = current_cited or int(item.get("frame_id", -1)) == int(after_frame_id)
                            except (TypeError, ValueError):
                                pass
                if not citations_valid:
                    verdict = "UNKNOWN"
                if verdict == "YES" and not current_cited:
                    verdict = "UNKNOWN"
            return {
                "decision": verdict,
                "reasoning": str(payload.get("reasoning") or "").strip(),
                "diagnostic_lift": str(payload.get("diagnostic_lift", "UNKNOWN")).strip().upper(),
                "diagnostic_lift_clear": bool(
                    v22
                    and verdict == "YES"
                    and str(payload.get("diagnostic_lift", "UNKNOWN")).strip().upper() == "CLEAR"
                    and citations_valid
                    and current_cited
                ),
                "evidence_for": evidence_for,
                "evidence_against": evidence_against,
                "raw_text": response.raw_text,
                "completion": {
                    key: (response.payload or {}).get(key)
                    for key in ("finish_reason", "reasoning_present", "reasoning_chars", "usage", "request_audit")
                    if (response.payload or {}).get(key) is not None
                },
                "latency_s": (response.payload or {}).get("latency_s"),
            }
        except Exception as exc:
            # A verifier transport/parse failure is uncertainty, not evidence of an
            # empty grasp. The runner keeps the original Agent decision inspectable.
            return {
                "decision": "UNKNOWN",
                "reasoning": "grasp verifier unavailable; visual verdict is unknown",
                "error": f"{type(exc).__name__}: {exc}",
                "completion": getattr(exc, "payload", {}) if isinstance(exc, VLMParseError) else {},
            }

    def resolve_instance(
        self,
        *,
        task: str,
        target: str,
        candidates: Sequence[dict[str, Any]],
        agentview_image,
        other_view_image=None,
        candidate_camera: str = "agentview",
        debug: bool = False,
    ) -> dict[str, Any]:
        """Choose one enumerated candidate on ambiguity; never return an action."""
        import numpy as np  # Rare critical path only.
        from PIL import Image, ImageDraw

        frame = np.asarray(agentview_image, dtype=np.uint8)
        view_name = "Wrist" if str(candidate_camera).lower() == "wrist" else "AgentView"
        full_views = Image.fromarray(frame[..., :3]).convert("RGB")
        draw = ImageDraw.Draw(full_views)
        draw.rectangle((0, 0, full_views.width, 18), fill="white")
        draw.text((4, 3), f"FULL {view_name}", fill="black")
        other_full_view = None
        if other_view_image is not None:
            other_full_view = Image.fromarray(
                np.asarray(other_view_image, dtype=np.uint8)[..., :3]
            ).convert("RGB")
            if other_full_view.size != full_views.size:
                other_full_view = other_full_view.resize(full_views.size, Image.Resampling.BILINEAR)
            other_name = "AgentView" if view_name == "Wrist" else "Wrist"
            draw = ImageDraw.Draw(other_full_view)
            draw.rectangle((0, 0, other_full_view.width, 18), fill="white")
            draw.text((4, 3), f"FULL {other_name} (same frame)", fill="black")
        tiles = []
        candidate_lines = []
        valid_ids = []
        candidate_colors = [
            (255, 96, 48), (30, 190, 255), (255, 205, 40),
            (220, 70, 220), (60, 210, 120), (255, 140, 210),
        ]
        for index, candidate in enumerate(candidates):
            candidate_id = f"candidate-{index}"
            color = candidate_colors[index % len(candidate_colors)]
            bbox = candidate.get("bbox_xyxy") if isinstance(candidate, dict) else None
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            raw_x1, raw_y1, raw_x2, raw_y2 = [int(round(float(v))) for v in bbox]
            bbox_text = [raw_x1, raw_y1, raw_x2, raw_y2]
            box_w, box_h = max(1, raw_x2 - raw_x1), max(1, raw_y2 - raw_y1)
            # Enlarge the live provider box with context proportional to its
            # own dimensions, and preserve its aspect ratio in the evidence
            # tile.  This avoids distorting small objects into a square crop.
            pad_x, pad_y = max(2, int(round(box_w * 0.25))), max(2, int(round(box_h * 0.18)))
            crop_x1, crop_x2 = max(0, raw_x1 - pad_x), min(frame.shape[1], raw_x2 + pad_x)
            crop_y1, crop_y2 = max(0, raw_y1 - pad_y), min(frame.shape[0], raw_y2 + pad_y)
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue
            # Link each enlarged crop to its exact same-frame region. The
            # runtime gives these IDs a deterministic spatial order, never a
            # detector-score order.
            draw = ImageDraw.Draw(full_views)
            draw.rectangle((raw_x1, raw_y1, raw_x2, raw_y2), outline=color, width=3)
            label_top = max(18, raw_y1 - 15)
            draw.rectangle((raw_x1, label_top, raw_x1 + 78, label_top + 15), fill=color)
            draw.text((raw_x1 + 2, label_top + 1), candidate_id, fill="black")
            crop = Image.fromarray(frame[crop_y1:crop_y2, crop_x1:crop_x2, :3]).convert("RGB")
            canvas = Image.new("RGB", (160, 192), "white")
            tile = crop.copy()
            tile.thumbnail((152, 164), Image.Resampling.LANCZOS)
            canvas.paste(tile, ((160 - tile.width) // 2, 24 + (164 - tile.height) // 2))
            tile_draw = ImageDraw.Draw(canvas)
            tile_draw.rectangle((0, 0, 159, 191), outline=color, width=3)
            tile_draw.text((4, 3), candidate_id, fill=color)
            tiles.append(np.asarray(canvas))
            valid_ids.append(candidate_id)
            candidate_lines.append(
                f"{candidate_id}: bbox_xyxy={bbox_text} pixels in {view_name}; its enlarged "
                "crop matches the same-color outlined region in the full view"
            )
        if not tiles:
            return {"selected": "UNKNOWN", "reason": "no valid candidate crops"}
        # Keep the candidate image panel bounded while retaining every labeled
        # choice. Up to three columns are placed per row; the number of choices
        # itself is controlled by the upstream detector budget.
        columns = min(3, len(tiles))
        rows = (len(tiles) + columns - 1) // columns
        sheet_image = Image.new("RGB", (160 * columns, 192 * rows), "white")
        for index, tile in enumerate(tiles):
            sheet_image.paste(Image.fromarray(tile), ((index % columns) * 160, (index // columns) * 192))
        sheet = np.asarray(sheet_image)
        if other_full_view is not None:
            pair = Image.new(
                "RGB",
                (full_views.width + other_full_view.width, full_views.height),
                "white",
            )
            pair.paste(full_views, (0, 0))
            pair.paste(other_full_view, (full_views.width, 0))
            full_views = pair
        allowed = valid_ids + ["UNKNOWN"]
        prompt = (
            "Resolve target identity after the temporal tracker found multiple plausible "
            "segmentations. Inspect both same-frame camera views and the labeled candidate "
            f"crops from {view_name}; the crops are enlarged while preserving their aspect ratio. "
            "Candidate IDs use top-to-bottom then left-to-right spatial order for this frame, "
            "not detector rank; each crop is linked to a same-color outlined box in the full "
            "view. Compare visible appearance and task semantics, not detector score or "
            "generic assumptions about typical object colors. "
            "Select exactly one candidate only when its identity is supported by the views; "
            "otherwise select UNKNOWN. This is an identity decision, not a robot action. "
            "Use the reason field to cite visible distinguishing features or explain what "
            "prevents a reliable match.\n"
            f"TASK: {task}\nTARGET: {target}\n"
            + "\n".join(candidate_lines)
            + '\nReturn JSON only: {"selected":"candidate-N|UNKNOWN",'
            '"reason":"one short visual sentence"}'
        )
        schema = {
            "type": "object",
            "properties": {
                "selected": {"type": "string", "enum": allowed},
                "reason": {"type": "string"},
            },
            "required": ["selected", "reason"],
            "additionalProperties": False,
        }
        chat_kwargs, output_tokens, temperature, thinking_budget = (
            _critical_reasoning_settings(
                self.client,
                True,
                160,
                budget_cap=2048,
                # Identity selection is a semantic decision across multiple
                # images. Give a Thinking checkpoint its configured reasoning
                # budget while reserving space for the final visual justification.
                final_reserve_tokens=128,
            )
        )
        try:
            response = self.client.complete_json(
                prompt,
                np.asarray(full_views),
                wrist_image=sheet,
                schema=schema,
                max_tokens=output_tokens,
                temperature=temperature,
                thinking_token_budget=thinking_budget,
                chat_template_kwargs=chat_kwargs,
                debug=debug,
                agentview_label="Same-frame full AgentView and Wrist views for target identity",
                wrist_label=f"Labeled {view_name} candidate crops, aspect ratio preserved",
            )
            payload = response.payload.get("json")
            if not isinstance(payload, dict):
                raise RuntimeError("instance resolver returned no JSON object")
            return {
                "selected": str(payload.get("selected") or "UNKNOWN"),
                "reason": str(payload.get("reason") or ""),
                "raw_text": response.raw_text,
                "latency_s": (response.payload or {}).get("latency_s"),
                "completion": {
                    key: (response.payload or {}).get(key)
                    for key in (
                        "finish_reason", "reasoning_present", "reasoning_chars",
                        "usage", "request_audit",
                    )
                    if (response.payload or {}).get(key) is not None
                },
            }
        except Exception as exc:
            return {
                "selected": "UNKNOWN",
                "reason": "instance resolver unavailable",
                "error": f"{type(exc).__name__}: {exc}",
                "completion": (
                    getattr(exc, "payload", {})
                    if isinstance(exc, VLMParseError)
                    else {}
                ),
            }

    def resolve_pregrasp(
        self,
        *,
        task: str,
        target: str,
        affordance: str,
        allowed_actions: Sequence[str],
        runtime_reason: str,
        agentview_image,
        wrist_image,
        previous_agentview_image=None,
        previous_wrist_image=None,
        spatial_belief: dict[str, Any] | None = None,
        visual_alignment: dict[str, Any] | None = None,
        executed_action: str | None = None,
        visual_memory: Sequence[dict[str, Any]] | None = None,
        visual_memory_bundle: Sequence[dict[str, Any]] | None = None,
        reflection_mode: str = "off",
        reflection_trigger: str | None = None,
        allow_visual_grasp_without_spatial: bool = False,
        target_roi_included: bool = False,
        prebuilt_memory_panel=None,
        current_frame_id: int | None = None,
        previous_frame_id: int | None = None,
        debug: bool = False,
    ) -> dict[str, Any]:
        """Make one semantic pregrasp choice from raw dual views.

        The runtime constrains the action alphabet and executes exactly one
        result before asking again.  Qwen cannot commit HELD or task success.
        """
        allowed = [str(item).strip().upper() for item in allowed_actions]
        allowed = list(dict.fromkeys(item for item in allowed if item))
        if "UNKNOWN" not in allowed:
            allowed.append("UNKNOWN")
        semantic_mode = any(item in {"VISUAL_ALIGN", "CORRECT_DEPTH", "CORRECT_LATERAL", "CORRECT_HEIGHT", "PROBE_DEPTH", "SELECT_GRASP"} for item in allowed)
        if "REOBSERVE" in allowed:
            choices = (
                "REOBSERVE asks the runtime for one bounded upward clearance move, then a fresh "
                "dual-view observation; it does not authorize grasp or claim contact. Choose it "
                "when that new view can resolve the current occlusion."
            )
        elif semantic_mode:
            choices = (
                "VISUAL_ALIGN asks the runtime for one bounded correction from the fresh, "
                "same-instance AgentView target-to-EEF pixel residual; the runtime chooses "
                "the largest calibrated axis not already contradicted by measured effects "
                "and requires a new observation afterward. "
                "CORRECT_DEPTH selects a signed depth correction supplied by the harness; "
                "CORRECT_LATERAL and CORRECT_HEIGHT work similarly. PROBE_DEPTH requests "
                "one reversible parallax probe."
            )
        else:
            choices = "Use MV_* only in legacy profiles; do not infer a Wrist vertical pixel offset as forward/back motion."
        spatial_text = json.dumps(
            _compact_spatial_prompt(spatial_belief), ensure_ascii=False, sort_keys=True
        )
        thinking_checkpoint = str(getattr(self.client, "model", "")).rstrip("/").endswith(
            "Qwen3-VL-8B-Thinking"
        )
        v22_decision = bool(allow_visual_grasp_without_spatial)
        structured_decision = bool(v22_decision or thinking_checkpoint)
        unknown_spatial_choice = (
            "At this grasp-entry occlusion review, choose REOBSERVE if the single bounded view-clearance "
            "option is safe and likely to provide a new useful view. Choose UNKNOWN if no safe, useful "
            "observation option is supported; UNKNOWN means stop without moving."
            if "REOBSERVE" in allowed
            else
            "If SPATIAL BELIEF is UNKNOWN, decide from the raw temporal views: choose GRASP "
            "when the target is visibly aligned between the open fingers; choose PROBE_DEPTH "
            "only when relative depth remains unclear and a bounded probe is needed. Do not "
            "probe repeatedly after the measured effect is clear."
            if allow_visual_grasp_without_spatial
            else "If SPATIAL BELIEF is UNKNOWN, inspect the raw temporal views and same-episode "
            "action effects. Choose VISUAL_ALIGN when the current same-instance AgentView "
            "projection is fresh and clearly outside its alignment tolerance. Choose PROBE_DEPTH "
            "only when the unresolved relation is specifically depth and one bounded probe is "
            "safe; otherwise choose UNKNOWN. UNKNOWN does not make a probe mandatory."
        )
        visual_alignment_text = json.dumps(
            visual_alignment if isinstance(visual_alignment, dict) else {"valid": False},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        temporal = prebuilt_memory_panel
        if temporal is None:
            temporal = _make_pregrasp_temporal_panel(
                agentview_image,
                wrist_image,
                previous_agentview_image,
                previous_wrist_image,
            )
        memory_instruction = (
            "The attached visual-memory panel contains raw dual-view frames from this same "
            "episode, ordered oldest to newest; its last row is the frozen current observation. "
            "If a labeled ROI strip appears below the frame rows, it is a magnification of the "
            "current frame, not a later time step and not another memory observation. "
            "Use the recorded executed action and observed effect to identify what changed. "
            "Earlier model judgments are explicitly unverified hypotheses, not facts.\n"
            if visual_memory_bundle and prebuilt_memory_panel is not None
            else ""
        )
        reflection_mode = str(reflection_mode or "off").lower()
        if reflection_mode not in {"off", "single", "double"}:
            reflection_mode = "off"
        memory_frames = [
            int(item["frame_id"])
            for item in (visual_memory_bundle or ())
            if isinstance(item, dict) and item.get("frame_id") is not None
        ]
        allowed_frame_ids = sorted(set(
            memory_frames
            + ([int(current_frame_id)] if current_frame_id is not None else [])
            + ([int(previous_frame_id)] if previous_frame_id is not None else [])
        ))
        prompt = (
            "You are the semantic pregrasp decision component inside a verified robot "
            "runtime. Inspect the raw live AgentView and Wrist images. The harness has "
            "kept the target identity and will execute exactly ONE action, capture fresh "
            "images, and ask again. Numeric overlays or detector boxes are evidence, not "
            "proof of physical alignment.\n"
            "CURRENT STAGE: GRASP. The destination/receptacle/basket is irrelevant now. "
            "Do not compare the target to the destination; compare only the target object "
            "to the open gripper fingers.\n"
            f"TARGET: {target}\nGRASPABLE PART: {affordance}\n"
            f"RUNTIME EVIDENCE: {runtime_reason}\n\n"
            "Choose GRASP only when the intended graspable part is visibly between the "
            "open fingers at a usable depth in Wrist and AgentView is consistent with the "
            "same object. Seeing the object merely near the camera center is insufficient. "
            "Reason from the newest dual views, compare the most relevant earlier frame with "
            "the action that actually executed, and check whether the visual relation agrees "
            "with the spatial tool output. Resolve contradictions by requesting only a runtime-"
            "allowed observation or bounded correction. The complete final JSON is mandatory; "
            "keep each field concise and do not restate the whole scene. "
            "If it is not ready, choose exactly one semantic correction from ALLOWED. "
            f"{choices} If a previous close was empty, do not repeat the same pose; choose "
            f"a correction visible in the raw views. {unknown_spatial_choice} Do not use "
            "CORRECT_DEPTH without a signed FRONT/BACK relation. Select UNKNOWN when evidence "
            "is insufficient or contradictory; do not force a physical guess. Do not claim "
            "held/success.\n"
            "The runtime evidence reports measured image motion after the previous action. "
            "Respect it: an action removed from ALLOWED has repeatedly moved the target in "
            "the wrong direction and must not be repeated.\n"
            f"{memory_instruction}"
            + (
                "The bottom panel row includes an enlarged crop of the current detected target. "
                "It contains only raw pixels at a larger scale; use the full views above to "
                "check finger placement and recover spatial context.\n"
                if target_roi_included else ""
            )
            + f"SPATIAL BELIEF (tool output, may be UNKNOWN): {spatial_text}\n"
            + f"FRESH VISUAL ALIGNMENT (runtime-calibrated, may be unavailable): {visual_alignment_text}\n"
            f"LAST EXECUTED ACTION: {executed_action or 'NONE'}\n"
            f"VISUAL MEMORY SUMMARY: {json.dumps(_compact_memory_prompt(visual_memory_bundle or visual_memory), ensure_ascii=False, separators=(',', ':'))}\n"
            f"ALLOWED: {', '.join(allowed)}\n"
            + (
                f"Evidence may cite only these in-episode frame ids: {allowed_frame_ids}. "
                "Cite the camera and visible observation; do not infer hidden contact. "
                "State the likely relation, evidence for and against it, what observation is "
                "missing, what should change after the selected option, and what result would "
                "show the option failed. Keep the summary concise; use UNKNOWN for unsupported "
                "claims.\n"
                'Return one final JSON object with selected, state_hypothesis, evidence_for, '
                'evidence_against, missing_observation, expected_effect, failure_condition, summary.'
                if structured_decision
                else 'Return JSON only: {"selected":"ONE_ALLOWED_TOKEN",'
                '"reason":"one short visual sentence"}'
            )
        )
        if structured_decision:
            citation_schema = {
                "type": "object",
                "properties": {
                    "frame_id": {"type": "integer", "enum": allowed_frame_ids},
                    "camera": {"type": "string", "enum": ["agentview", "wrist"]},
                    "observation": {"type": "string"},
                },
                "required": ["frame_id", "camera", "observation"],
                "additionalProperties": False,
            }
            schema = {
                "type": "object",
                "properties": {
                    "selected": {"type": "string", "enum": allowed},
                    "state_hypothesis": {"type": "string"},
                    # The runtime already downgrades an affirmative action to UNKNOWN
                    # without a valid citation. Make that contract explicit to the
                    # constrained decoder so it cannot spend its final tokens on a
                    # confident label while leaving the evidence list empty.
                    "evidence_for": {
                        "type": "array", "items": citation_schema,
                        "minItems": 1, "maxItems": 2,
                    },
                    "evidence_against": {"type": "array", "items": citation_schema, "maxItems": 2},
                    "missing_observation": {"type": "string"},
                    "expected_effect": {"type": "string"},
                    "failure_condition": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": [
                    "selected", "state_hypothesis", "evidence_for", "evidence_against",
                    "missing_observation", "expected_effect", "failure_condition", "summary",
                ],
                "additionalProperties": False,
            }
        else:
            schema = {
                "type": "object",
                "properties": {
                    "selected": {"type": "string", "enum": allowed},
                    "reason": {"type": "string"},
                },
                "required": ["selected", "reason"],
                "additionalProperties": False,
            }
        reflection_audit: dict[str, Any] = {
            "mode": reflection_mode,
            "trigger": reflection_trigger,
            "scene_description": None,
            "first_pass": None,
            "same_image_payload": None,
        }
        try:
            final_prompt = prompt
            first_response = None
            scene_description = None
            if reflection_mode == "double":
                if not allowed_frame_ids:
                    reflection_audit["mode"] = "off"
                    reflection_audit["skipped_reason"] = "no frame ids available for reflection evidence"
                else:
                    required_current_frame_id = (
                        int(current_frame_id)
                        if current_frame_id is not None
                        else int(max(allowed_frame_ids))
                    )
                    first_schema = {
                        "type": "object",
                        "properties": {
                            "current_observation": {
                                "type": "object",
                                "properties": {
                                    "frame_id": {
                                        "type": "integer",
                                        "enum": [required_current_frame_id],
                                    },
                                    "camera": {
                                        "type": "string",
                                        "enum": ["agentview", "wrist"],
                                    },
                                    "observation": {"type": "string"},
                                },
                                "required": ["frame_id", "camera", "observation"],
                                "additionalProperties": False,
                            },
                            "visible_state": {"type": "string"},
                            "target_gripper_relation": {
                                "type": "string",
                                "enum": [
                                    "BETWEEN", "IN_FRONT", "BEHIND", "LATERAL_OFFSET",
                                    "NOT_VISIBLE", "OCCLUDED", "UNKNOWN",
                                ],
                            },
                            "action_effect": {"type": "string"},
                            "uncertainties": {
                                "type": "array", "items": {"type": "string"},
                                "maxItems": 4,
                            },
                            "evidence": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "frame_id": {"type": "integer", "enum": allowed_frame_ids},
                                        "camera": {"type": "string", "enum": ["agentview", "wrist"]},
                                        "observation": {"type": "string"},
                                    },
                                    "required": ["frame_id", "camera", "observation"],
                                    "additionalProperties": False,
                                },
                                "minItems": 1, "maxItems": 4,
                            },
                        },
                        "required": [
                            "current_observation", "visible_state", "target_gripper_relation", "action_effect",
                            "uncertainties", "evidence",
                        ],
                        "additionalProperties": False,
                    }
                    first_prompt = (
                        "First pass: describe evidence only. Inspect the attached same-episode "
                        "raw dual-view panel; rows are oldest to newest and the last row is the "
                        "labeled current-frame row is the frozen observation. If a separate "
                        "labeled ROI strip is below it, that strip enlarges pixels from this same "
                        "current frame and is not a later observation. Compare target-to-gripper relation and what the "
                        "actual executed action changed. Do not choose an action or claim hidden "
                        "contact/holding. The required current_observation must cite exactly the "
                        f"frozen current frame {required_current_frame_id} and one camera tile; "
                        "write UNKNOWN in its observation if that tile does not resolve the claim. "
                        "Then cite supporting temporal evidence with frame ids and camera.\n"
                        f"TASK: {task}\nTARGET: {target}\nAFFORDANCE: {affordance}\n"
                        f"TRIGGER: {reflection_trigger or 'UNKNOWN'}\n"
                        f"ACTION MEMORY: {json.dumps(_compact_memory_prompt(visual_memory_bundle or visual_memory), ensure_ascii=False, separators=(',', ':'))}\n"
                        f"AVAILABLE FRAME IDS: {allowed_frame_ids}"
                    )
                    first_kwargs, first_max_tokens, first_temperature, first_thinking_budget = (
                        _critical_reasoning_settings(
                            self.client, True, 1024, budget_cap=1024,
                            final_reserve_tokens=512,
                        )
                    )
                    first_response = self.client.complete_json(
                        first_prompt,
                        temporal,
                        wrist_image=None,
                        schema=first_schema,
                        max_tokens=first_max_tokens,
                        temperature=first_temperature,
                        thinking_token_budget=first_thinking_budget,
                        chat_template_kwargs=first_kwargs,
                        debug=debug,
                        agentview_label="Frozen same-episode dual-view memory panel",
                    )
                    scene_description = first_response.payload.get("json")
                    if not isinstance(scene_description, dict):
                        raise RuntimeError("pregrasp reflection returned no structured scene description")
                    cited = scene_description.get("evidence")
                    current_observation = scene_description.get("current_observation")
                    cited_current_frame_id = None
                    if isinstance(current_observation, dict):
                        candidate_frame_id = current_observation.get("frame_id")
                        if isinstance(candidate_frame_id, int) and not isinstance(candidate_frame_id, bool):
                            cited_current_frame_id = candidate_frame_id
                    valid_current_observation = (
                        isinstance(current_observation, dict)
                        and cited_current_frame_id == required_current_frame_id
                        and str(current_observation.get("camera", "")).lower()
                        in {"agentview", "wrist"}
                        and bool(str(current_observation.get("observation", "")).strip())
                    )
                    valid_evidence = (
                        valid_current_observation and isinstance(cited, list) and bool(cited)
                    )
                    if valid_evidence:
                        for item in cited:
                            cited_frame_id = None
                            if isinstance(item, dict):
                                candidate_frame_id = item.get("frame_id")
                                if isinstance(candidate_frame_id, int) and not isinstance(candidate_frame_id, bool):
                                    cited_frame_id = candidate_frame_id
                            if (
                                not isinstance(item, dict)
                                or cited_frame_id not in allowed_frame_ids
                                or str(item.get("camera", "")).lower() not in {"agentview", "wrist"}
                                or not str(item.get("observation", "")).strip()
                            ):
                                valid_evidence = False
                                break
                    if not valid_evidence:
                        reflection_audit.update({
                            "scene_description": scene_description,
                            "first_pass": {
                                "latency_s": first_response.payload.get("latency_s"),
                                "usage": first_response.payload.get("usage"),
                                "completion": {
                                    key: first_response.payload.get(key)
                                    for key in ("finish_reason", "reasoning_present", "reasoning_chars")
                                },
                                "request_audit": first_response.payload.get("request_audit"),
                            },
                            "valid_evidence": False,
                        })
                        return {
                            "selected": "UNKNOWN",
                            "reason": "first-pass reflection lacked valid current-frame evidence",
                            "reflection": reflection_audit,
                        }
                    reflection_audit.update({
                        "scene_description": scene_description,
                        "first_pass": {
                            "latency_s": first_response.payload.get("latency_s"),
                            "usage": first_response.payload.get("usage"),
                            "completion": {
                                key: first_response.payload.get(key)
                                for key in ("finish_reason", "reasoning_present", "reasoning_chars")
                            },
                            "request_audit": first_response.payload.get("request_audit"),
                        },
                        "valid_evidence": True,
                    })
                    final_prompt = (
                        prompt
                        + "\nFIRST-PASS STRUCTURED OBSERVATION (unverified model hypothesis; "
                        "check it against the original frozen image panel and action receipts; "
                        "contradictions override these notes):\n"
                        + json.dumps(scene_description, ensure_ascii=False, separators=(",", ":"))
                        + "\nBefore selecting, independently re-check each specific first-pass claim "
                        "against the labeled current-frame tiles and any bottom ROI. The ROI is "
                        "the same current image, not a new time step. Correct or withdraw a claim "
                        "that the pixels do not support; do not accept a claim just because the "
                        "first pass stated it. Then select one allowed option. Do not treat the "
                        "repeated analysis as an independent physical sensor."
                    )
            critical_reasoning_enabled = bool(v22_decision or thinking_checkpoint)
            if v22_decision:
                reasoning_kwargs, max_tokens, temperature, thinking_budget = _critical_reasoning_settings(
                    self.client, True, 1024
                )
            elif thinking_checkpoint:
                reasoning_kwargs, max_tokens, temperature, thinking_budget = _critical_reasoning_settings(
                    self.client, True, 4096,
                )
            else:
                reasoning_kwargs, max_tokens, temperature, thinking_budget = _critical_reasoning_settings(
                    self.client, critical_reasoning_enabled, 1024
                )
            response = self.client.complete_json(
                final_prompt,
                temporal,
                # The temporal panel already contains the current Wrist tile.
                # Sending it a second time can exceed Qwen8B's 4096-token
                # context limit once evidence and memory are included.
                wrist_image=None,
                schema=schema,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking_token_budget=thinking_budget,
                chat_template_kwargs=reasoning_kwargs,
                debug=debug,
                agentview_label="Raw live AgentView",
                wrist_label="Raw live Wrist view",
            )
            if first_response is not None:
                first_images = (
                    first_response.payload.get("request_audit", {}).get("images")
                    if isinstance(first_response.payload.get("request_audit"), dict)
                    else None
                )
                final_images = (
                    response.payload.get("request_audit", {}).get("images")
                    if isinstance(response.payload.get("request_audit"), dict)
                    else None
                )
                same_images = bool(first_images is not None and first_images == final_images)
                reflection_audit["same_image_payload"] = same_images
                if not same_images:
                    return {
                        "selected": "UNKNOWN",
                        "reason": "two-pass reflection image payloads did not match",
                        "reflection": reflection_audit,
                        "completion": response.payload.get("request_audit"),
                    }
            payload = response.payload.get("json")
            if not isinstance(payload, dict):
                raise RuntimeError("pregrasp resolver returned no JSON object")
            selected = str(payload.get("selected") or "UNKNOWN").upper()
            if selected not in allowed:
                selected = "UNKNOWN"
            if structured_decision:
                valid_citations = True
                citations = []
                for field in ("evidence_for", "evidence_against"):
                    values = payload.get(field)
                    if not isinstance(values, list):
                        valid_citations = False
                        break
                    for item in values:
                        if not isinstance(item, dict):
                            valid_citations = False
                            break
                        if int(item.get("frame_id", -1)) not in allowed_frame_ids or str(item.get("camera", "")).lower() not in {"agentview", "wrist"} or not str(item.get("observation", "")).strip():
                            valid_citations = False
                            break
                        citations.append(dict(item))
                if selected != "UNKNOWN" and (not valid_citations or not payload.get("evidence_for")):
                    selected = "UNKNOWN"
                if selected != "UNKNOWN" and current_frame_id is not None and not any(
                    isinstance(item, dict)
                    and int(item.get("frame_id", -1)) == int(current_frame_id)
                    for item in (payload.get("evidence_for") or [])
                ):
                    selected = "UNKNOWN"
                decision = {
                    "selected": selected,
                    "state_hypothesis": str(payload.get("state_hypothesis") or "UNKNOWN"),
                    "evidence_for": payload.get("evidence_for") if valid_citations else [],
                    "evidence_against": payload.get("evidence_against") if valid_citations else [],
                    "missing_observation": str(payload.get("missing_observation") or ""),
                    "expected_effect": str(payload.get("expected_effect") or ""),
                    "failure_condition": str(payload.get("failure_condition") or ""),
                    "summary": str(payload.get("summary") or ""),
                    "frame_ids": allowed_frame_ids,
                    "validated": bool(valid_citations),
                }
                return {
                    "selected": selected,
                    "reason": decision["summary"],
                    "agent_decision": decision,
                    "raw_text": response.raw_text,
                    "latency_s": (response.payload or {}).get("latency_s"),
                    "usage": (response.payload or {}).get("usage"),
                    "request_audit": (response.payload or {}).get("request_audit"),
                    "reflection": reflection_audit,
                    "completion_metadata": {
                        key: (response.payload or {}).get(key)
                        for key in (
                            "finish_reason", "reasoning_present", "reasoning_chars",
                            "final_content_chars",
                        )
                    },
                }
            return {
                "selected": selected,
                "reason": str(payload.get("reason") or ""),
                "raw_text": response.raw_text,
                "latency_s": (response.payload or {}).get("latency_s"),
                "usage": (response.payload or {}).get("usage"),
                "request_audit": (response.payload or {}).get("request_audit"),
                "completion_metadata": {
                    key: (response.payload or {}).get(key)
                    for key in ("finish_reason", "reasoning_present", "reasoning_chars", "final_content_chars")
                },
                "reflection": reflection_audit,
            }
        except Exception as exc:
            return {
                "selected": "UNKNOWN",
                "reason": "pregrasp resolver unavailable",
                "error": f"{type(exc).__name__}: {exc}",
                "completion": getattr(exc, "payload", {}) if isinstance(exc, VLMParseError) else {},
                "reflection": reflection_audit,
            }

    def verify_place(
        self,
        *,
        task: str,
        target: str,
        affordance: str,
        agentview_image,
        wrist_image=None,
        debug: bool = False,
        placement_v22: bool = False,
        placement_evidence: dict[str, Any] | None = None,
        memory_panel=None,
        memory_bundle: Sequence[dict[str, Any]] | None = None,
        reflection_mode: str = "off",
        reflection_trigger: str | None = None,
        allowed_options: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Verify placement; V2.2 returns a physical relation, not an action."""
        if placement_v22:
            evidence = placement_evidence if isinstance(placement_evidence, dict) else {}
            allowed_options = tuple(allowed_options or ("REOBSERVE", "CHANGE_VIEW", "HOLD"))
            compact_evidence = {
                key: evidence.get(key)
                for key in (
                    "frame_id", "eef_residual_world",
                    "rim_clearance_m", "containment_margin_m", "uncertainty_m",
                    "evidence_sources", "conflicts", "fresh",
                )
                if key in evidence
            }
            if reflection_mode == "legacy" and "relation" in evidence:
                compact_evidence["relation"] = evidence["relation"]
            timeline = [
                {
                    key: entry.get(key)
                    for key in ("frame_id", "executed_action", "motion_delta", "route_phase", "route_epoch", "placement_summary", "predicted_effect", "observed_effect", "effect_status", "decision_summary", "tags")
                    if key in entry
                }
                for entry in list(memory_bundle or [])[-3:]
                if isinstance(entry, dict)
            ]
            if reflection_mode == "legacy":
                timeline = []
            visual_timeline = [
                {key: entry.get(key) for key in (
                    "frame_id", "executed_action", "motion_delta",
                    "observed_effect", "effect_status", "tags",
                ) if key in entry}
                for entry in timeline
            ]
            allowed_frame_ids = sorted({
                int(item["frame_id"])
                for item in timeline
                if item.get("frame_id") is not None
            })
            scene_description = None
            first_latency = None
            first_usage = None
            first_request_audit = None
            if reflection_mode == "double" and memory_panel is not None:
                first_prompt = (
                    "Describe only what the ordered raw AgentView/Wrist frame pairs establish. "
                    "Rows are oldest to newest. Distinguish visible object/opening geometry from "
                    "physical contact or support, which may be hidden. Do not infer contact from "
                    "a single bounding-box overlap. Use UNKNOWN for unsupported fields. "
                    "Do not select an action or placement relation.\n"
                    f"TASK: {task}\nTARGET: {target}\nTRIGGER: {reflection_trigger or 'UNKNOWN'}\n"
                    f"EXECUTED ACTION TIMELINE: {json.dumps(visual_timeline, ensure_ascii=False, separators=(',', ':'))}\n"
                    'Return JSON only: {"held_object":"text or UNKNOWN","opening":"text or UNKNOWN",'
                    '"relative_position":"text or UNKNOWN","contact_or_support":"text or UNKNOWN",'
                    '"action_effect":"text or UNKNOWN","uncertainty":"text","evidence_frame_ids":[0]}'
                )
                first_schema = {
                    "type": "object",
                    "properties": {
                        key: {"type": "string"}
                        for key in ("held_object", "opening", "relative_position", "contact_or_support", "action_effect", "uncertainty")
                    } | {"evidence_frame_ids": {"type": "array", "items": {"type": "integer"}}},
                    "required": ["held_object", "opening", "relative_position", "contact_or_support", "action_effect", "uncertainty", "evidence_frame_ids"],
                    "additionalProperties": False,
                }
                try:
                    (
                        reflection_kwargs,
                        reflection_tokens,
                        reflection_temperature,
                        reflection_thinking_budget,
                    ) = (
                        _critical_reasoning_settings(
                            self.client, placement_v22, 220, budget_cap=1024
                        )
                    )
                    first = self.client.complete_json(
                        first_prompt, memory_panel, wrist_image=None,
                        schema=first_schema, max_tokens=reflection_tokens,
                        temperature=reflection_temperature,
                        thinking_token_budget=reflection_thinking_budget,
                        chat_template_kwargs=reflection_kwargs, debug=debug,
                        agentview_label="Ordered raw placement memory; oldest row first",
                    )
                    scene_description = first.payload.get("json")
                    first_latency = (first.payload or {}).get("latency_s")
                    first_usage = (first.payload or {}).get("usage")
                    first_request_audit = (first.payload or {}).get("request_audit")
                    if not isinstance(scene_description, dict):
                        raise ValueError("scene description is absent")
                    valid_frames = {int(item.get("frame_id")) for item in timeline}
                    if not set(scene_description.get("evidence_frame_ids") or []).issubset(valid_frames):
                        raise ValueError("scene description cites unavailable frames")
                except Exception as exc:
                    return {"relation": "UNKNOWN", "decision": "UNKNOWN", "reasoning": "scene reflection unavailable", "error": f"{type(exc).__name__}: {exc}"}
            for key in ("opening_free_space_polygon_world", "object_footprint_world"):
                points = evidence.get(key)
                if isinstance(points, (list, tuple)):
                    compact_evidence[key] = [
                        [round(float(value), 4) for value in point[:2]]
                        for point in list(points)[:16]
                        if isinstance(point, (list, tuple)) and len(point) >= 2
                    ]
            view_instruction = (
                "Inspect the latest row of the raw image panel and compare earlier rows"
                if reflection_mode in {"single", "double"} and memory_panel is not None
                else "Inspect the two live raw images"
            )
            temporal_evidence_text = (
                "" if reflection_mode == "legacy" else
                f"EXECUTED ACTION TIMELINE: {json.dumps(timeline, ensure_ascii=False, separators=(',', ':'))}\n"
                f"VISUAL SCENE DESCRIPTION (hypothesis, may contain errors): {json.dumps(scene_description, ensure_ascii=False, separators=(',', ':'))}\n"
            )
            prompt = (
                "You are the visual seating verifier for a robot placement transaction.\n"
                f"TASK: {task}\nRECEPTACLE: {target}\nAFFORDANCE: {affordance}\n\n"
                f"{view_instruction} to classify the physical relation of the still-held "
                "object to the receptacle opening. Use the object's visible footprint, rim, "
                "and any stable support/contact evidence; do not use a gripper-center pixel "
                "offset as a substitute for the object footprint. Return only a relation. "
                "ABOVE_UNALIGNED means it is above but not safely contained; ABOVE_ALIGNED "
                "means it is above the free space with a visible margin; DESCENDING_CLEAR "
                "means a bounded descent can continue; RIM_CONTACT means a rim/contact "
                "boundary is visible; SEATED_HELD means the payload is stably seated while "
                "the gripper still holds it; LOST means the held instance is not visible; "
                "UNKNOWN means the views do not establish the relation. Do not choose a "
                "movement token and do not claim success from a single ambiguous frame.\n\n"
                "GEOMETRIC HARNESS EVIDENCE (measurement aid, not a success proof):\n"
                f"{json.dumps(compact_evidence, ensure_ascii=False, separators=(',', ':'))}\n"
                f"{temporal_evidence_text}"
                "Interpret a positive rim_clearance_m as the payload lowest point being "
                "below the estimated rim plane. A positive containment_margin_m means the "
                "measured payload footprint is inside the opening with that margin; null or "
                "conflicting values are not alignment evidence. The footprint is a lower "
                "contour estimate, so distinguish a held body still visible above the rim "
                "from its lower footprint being seated. Combine this evidence with both live "
                "views and return SEATED_HELD only when the held payload is stably supported.\n\n"
                "Any prior decision_summary is an unverified model hypothesis tied to its "
                "listed frame and route epoch. Treat it as a question to recheck, not as a fact; "
                "ignore it if its frame, route, or current images disagree. The action-effect "
                "record describes observed execution and motion, not proof of contact.\n\n"
                f"Cite only these visible frame ids in your final decision: {allowed_frame_ids}. "
                "For each citation identify AgentView or Wrist and a visible fact. Separate "
                "evidence for from evidence against your relation. State the next observation "
                "needed and what effect would falsify the chosen next step.\n\n"
                'Return JSON only: {"relation":"ABOVE_UNALIGNED|ABOVE_ALIGNED|'
                'DESCENDING_CLEAR|RIM_CONTACT|SEATED_HELD|LOST|UNKNOWN",'
                '"reasoning":"brief final rationale","next_step":"allowed option",'
                '"evidence_for":[],"evidence_against":[],"missing_observation":"text",'
                '"expected_effect":"text","failure_condition":"text"}'
            )
            prompt = prompt.replace(
                '"reasoning":"one short visual sentence"}',
                '"reasoning":"one short visual sentence","next_step":"'
                + "|".join(allowed_options)
                + '"}',
            )
            schema = {
                "type": "object",
                "properties": {
                    "relation": {
                        "type": "string",
                        "enum": [
                            "ABOVE_UNALIGNED", "ABOVE_ALIGNED", "DESCENDING_CLEAR",
                            "RIM_CONTACT", "SEATED_HELD", "LOST", "UNKNOWN",
                        ],
                    },
                    "reasoning": {"type": "string"},
                    "next_step": {"type": "string", "enum": list(allowed_options)},
                    "evidence_for": {
                        "type": "array", "maxItems": 3,
                        "items": {
                            "type": "object",
                            "properties": {
                                "frame_id": {"type": "integer", "enum": allowed_frame_ids},
                                "camera": {"type": "string", "enum": ["agentview", "wrist"]},
                                "observation": {"type": "string"},
                            },
                            "required": ["frame_id", "camera", "observation"],
                            "additionalProperties": False,
                        },
                    },
                    "evidence_against": {
                        "type": "array", "maxItems": 3,
                        "items": {
                            "type": "object",
                            "properties": {
                                "frame_id": {"type": "integer", "enum": allowed_frame_ids},
                                "camera": {"type": "string", "enum": ["agentview", "wrist"]},
                                "observation": {"type": "string"},
                            },
                            "required": ["frame_id", "camera", "observation"],
                            "additionalProperties": False,
                        },
                    },
                    "missing_observation": {"type": "string"},
                    "expected_effect": {"type": "string"},
                    "failure_condition": {"type": "string"},
                },
                "required": [
                    "relation", "reasoning", "next_step", "evidence_for", "evidence_against",
                    "missing_observation", "expected_effect", "failure_condition",
                ],
                "additionalProperties": False,
            }
        else:
            prompt = (
                "You are the high-level post-action placement verifier for a robot task.\n"
                f"TASK: {task}\n"
                f"RECEPTACLE: {target}\n"
                f"AFFORDANCE: {affordance}\n\n"
                "The robot has attempted to place a still-held object. Inspect "
                "both live images. AgentView is authoritative for the global relation between "
                "the held object and the receptacle; Wrist is supplementary and may be blocked "
                "by the gripper mount. Answer YES only when the target object is visibly inside "
                "the receptacle boundary or clearly seated in its opening. Answer NO only when "
                "the attempted placement is visibly off-center/failed or the object is clearly "
                "outside/fallen away. If it is merely still above the receptacle and no final "
                "placement attempt is visually complete, answer UNKNOWN. For NO, choose the "
                "single safest immediate recovery action from the image: lift with MV_UP to "
                "clear the rim before horizontal correction when needed, otherwise choose the "
                "appropriate MV_LEFT/MV_RIGHT/MV_FWD/MV_BACK correction. Do not release on NO.\n\n"
                'Return JSON only: {"placed":"YES|NO|UNKNOWN",'
                '"recovery_action":"MV_UP|MV_LEFT|MV_RIGHT|MV_FWD|MV_BACK|HOLD",'
                '"reasoning":"one short visual sentence"}'
            )
            schema = {
                "type": "object",
                "properties": {
                    "placed": {"type": "string", "enum": ["YES", "NO", "UNKNOWN"]},
                    "recovery_action": {
                        "type": "string",
                        "enum": ["MV_UP", "MV_LEFT", "MV_RIGHT", "MV_FWD", "MV_BACK", "HOLD"],
                    },
                    "reasoning": {"type": "string"},
                },
                "required": ["placed", "recovery_action", "reasoning"],
                "additionalProperties": False,
            }
        try:
            reasoning_kwargs, max_tokens, temperature, thinking_budget = _critical_reasoning_settings(
                self.client, placement_v22, 128
            )
            response = self.client.complete_json(
                prompt,
                memory_panel if placement_v22 and reflection_mode in {"single", "double"} and memory_panel is not None else agentview_image,
                wrist_image=None if placement_v22 and reflection_mode in {"single", "double"} and memory_panel is not None else wrist_image,
                schema=schema,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking_token_budget=thinking_budget,
                chat_template_kwargs=reasoning_kwargs,
                debug=debug,
                agentview_label="AgentView after PLACE",
                wrist_label="Wrist after PLACE (supplementary; may be occluded)",
            )
            payload = response.payload.get("json")
            if not isinstance(payload, dict):
                raise RuntimeError("place verifier returned no JSON object")
            reasoning = str(payload.get("reasoning") or "").strip()
            if placement_v22:
                relation = str(payload.get("relation", "UNKNOWN")).strip().upper()
                allowed_relations = {
                    "ABOVE_UNALIGNED", "ABOVE_ALIGNED", "DESCENDING_CLEAR",
                    "RIM_CONTACT", "SEATED_HELD", "LOST", "UNKNOWN",
                }
                if relation not in allowed_relations:
                    relation = "UNKNOWN"
                valid_citations = True
                for field in ("evidence_for", "evidence_against"):
                    values = payload.get(field)
                    if not isinstance(values, list):
                        valid_citations = False
                        break
                    for citation in values:
                        if (
                            not isinstance(citation, dict)
                            or int(citation.get("frame_id", -1)) not in allowed_frame_ids
                            or str(citation.get("camera", "")).lower() not in {"agentview", "wrist"}
                            or not str(citation.get("observation", "")).strip()
                        ):
                            valid_citations = False
                            break
                if not valid_citations or (relation != "UNKNOWN" and not payload.get("evidence_for")):
                    relation = "UNKNOWN"
                return {
                    "relation": relation,
                    "decision": "YES" if relation == "SEATED_HELD" else "UNKNOWN",
                    "recovery_action": "HOLD",
                    "reasoning": reasoning,
                    "evidence_for": payload.get("evidence_for", []) if valid_citations else [],
                    "evidence_against": payload.get("evidence_against", []) if valid_citations else [],
                    "missing_observation": str(payload.get("missing_observation") or ""),
                    "expected_effect": str(payload.get("expected_effect") or ""),
                    "failure_condition": str(payload.get("failure_condition") or ""),
                    "next_step": str(payload.get("next_step") or "HOLD").upper(),
                    "raw_text": response.raw_text,
                    "latency_s": (response.payload or {}).get("latency_s"),
                    "usage": (response.payload or {}).get("usage"),
                    "scene_description": scene_description,
                    "first_latency_s": first_latency,
                    "first_usage": first_usage,
                    "request_audit": (response.payload or {}).get("request_audit"),
                    "first_request_audit": first_request_audit,
                }
            verdict = str(payload.get("placed", "UNKNOWN")).strip().upper()
            if verdict not in {"YES", "NO", "UNKNOWN"}:
                verdict = "UNKNOWN"
            recovery_action = str(payload.get("recovery_action", "HOLD")).strip().upper()
            if recovery_action not in {"MV_UP", "MV_LEFT", "MV_RIGHT", "MV_FWD", "MV_BACK", "HOLD"}:
                recovery_action = "HOLD"
            return {"decision": verdict, "recovery_action": recovery_action, "reasoning": reasoning, "raw_text": response.raw_text, "latency_s": (response.payload or {}).get("latency_s")}
        except Exception as exc:
            result = {
                "decision": "UNKNOWN",
                "recovery_action": "HOLD",
                "reasoning": "place verifier unavailable; visual verdict is unknown",
                "next_step": "HOLD",
                "error": f"{type(exc).__name__}: {exc}",
                "completion": getattr(exc, "payload", {}) if isinstance(exc, VLMParseError) else {},
            }
            if placement_v22:
                result["relation"] = "UNKNOWN"
            return result

    def review_place_alignment(
        self,
        *,
        task: str,
        target: str,
        affordance: str,
        agentview_image,
        wrist_image=None,
        recent_moves: str = "",
        debug: bool = False,
    ) -> dict[str, Any]:
        """Review horizontal alignment before the controller lowers or releases."""
        prompt = (
            "You are the high-level pre-placement alignment reviewer for a robot task.\n"
            f"TASK: {task}\n"
            f"RECEPTACLE: {target}\n"
            f"AFFORDANCE: {affordance}\n\n"
            "The gripper is still closed around the target object and the robot is about "
            "to place it. Inspect both live images in detail. AgentView is authoritative "
            "for the global relation between object, gripper, and receptacle; Wrist is "
            "supplementary and may be occluded by the gripper mount. Decide whether the "
            "object is horizontally over the receptacle opening with a safe visible margin "
            "for lowering. Do not use a fixed pixel, object-size, or world-height threshold; "
            "judge the actual visual relation in these frames. Explicitly locate the held "
            "object's visible body and lowest point, compare its footprint with the opening, "
            "and check whether it is tilted or contacting a rim; the gripper center alone is "
            "not proof of alignment. If not aligned, choose the single best immediate action: "
            "use MV_UP only when the fresh image shows the object is too low/near a rim and "
            "must first be cleared, otherwise choose the horizontal correction direction. "
            "If aligned, choose MV_DOWN. If the views are insufficient, choose HOLD. "
            "The following actions were executed immediately before this review, newest first: "
            f"{recent_moves or 'none'}. If the same correction has repeated without visibly "
            "improving the object/receptacle relation, do not blindly repeat it: clear any "
            "possible rim contact with MV_UP or choose the other axis, then reassess a fresh "
            "frame.\n\n"
            'Return JSON only: {"aligned":"YES|NO|UNKNOWN",'
            '"recommended_action":"MV_DOWN|MV_UP|MV_LEFT|MV_RIGHT|MV_FWD|MV_BACK|HOLD",'
            '"reasoning":"one short visual sentence"}'
        )
        schema = {
            "type": "object",
            "properties": {
                "aligned": {"type": "string", "enum": ["YES", "NO", "UNKNOWN"]},
                "recommended_action": {
                    "type": "string",
                    "enum": ["MV_DOWN", "MV_UP", "MV_LEFT", "MV_RIGHT", "MV_FWD", "MV_BACK", "HOLD"],
                },
                "reasoning": {"type": "string"},
            },
            "required": ["aligned", "recommended_action", "reasoning"],
            "additionalProperties": False,
        }
        try:
            response = self.client.complete_json(
                prompt,
                agentview_image,
                wrist_image=wrist_image,
                schema=schema,
                max_tokens=160,
                temperature=0.0,
                chat_template_kwargs=TOKEN_CHAT_TEMPLATE_KWARGS,
                debug=debug,
                agentview_label="AgentView before PLACE alignment",
                wrist_label="Wrist before PLACE alignment (supplementary; may be occluded)",
            )
            payload = response.payload.get("json")
            if not isinstance(payload, dict):
                raise RuntimeError("place alignment reviewer returned no JSON object")
            decision = str(payload.get("aligned", "UNKNOWN")).strip().upper()
            if decision not in {"YES", "NO", "UNKNOWN"}:
                decision = "UNKNOWN"
            action = str(payload.get("recommended_action", "HOLD")).strip().upper()
            if action not in {"MV_DOWN", "MV_UP", "MV_LEFT", "MV_RIGHT", "MV_FWD", "MV_BACK", "HOLD"}:
                action = "HOLD"
            return {
                "decision": decision,
                "recommended_action": action,
                "reasoning": str(payload.get("reasoning") or "").strip(),
                "raw_text": response.raw_text,
                "latency_s": (response.payload or {}).get("latency_s"),
            }
        except Exception as exc:
            return {
                "decision": "UNKNOWN",
                "recommended_action": "HOLD",
                "reasoning": "place alignment reviewer unavailable; visual alignment is unknown",
                "error": f"{type(exc).__name__}: {exc}",
            }

    def verify_task(
        self,
        *,
        task: str,
        agentview_image,
        wrist_image=None,
        debug: bool = False,
    ) -> dict[str, Any]:
        """Re-judge the whole task from a fresh, unobstructed observation.

        This is intentionally an Agent judgment rather than a runner state machine:
        the runner only uses the verdict to decide whether the planner should look at
        the current scene again.  The model must distinguish a target inside the
        receptacle from one on a rim, beside it, or visibly fallen/tilted.
        """
        prompt = (
            "You are the final visual outcome reviewer for a robot manipulation task.\n"
            f"TASK: {task}\n\n"
            "The previous controller plan has ended, but simulator success has not been "
            "confirmed. Inspect the fresh AgentView first and the Wrist view only as "
            "supplementary evidence. Decide whether the requested task is actually complete. "
            "For pick-and-place, answer complete=true only if the requested object is visibly "
            "inside/seated in the destination opening, not on its rim, beside it, tilted outside, "
            "or missing. If it is incomplete, describe what is visibly wrong and where the object "
            "currently is so a new planner call can choose a grounded recovery plan. Do not assume "
            "that a previous RELEASE succeeded, and do not use gripper width or simulator state "
            "as proof of placement.\n\n"
            'Return JSON only: {"complete":true|false,"reason":"one concise visual sentence"}'
        )
        schema = {
            "type": "object",
            "properties": {
                "complete": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["complete", "reason"],
            "additionalProperties": False,
        }
        try:
            response = self.client.complete_json(
                prompt,
                agentview_image,
                wrist_image=wrist_image,
                schema=schema,
                max_tokens=192,
                temperature=0.0,
                chat_template_kwargs=TOKEN_CHAT_TEMPLATE_KWARGS,
                debug=debug,
                agentview_label="AgentView final task check",
                wrist_label="Wrist final task check (supplementary)",
            )
            payload = response.payload.get("json")
            if not isinstance(payload, dict):
                raise RuntimeError("final task verifier returned no JSON object")
            return {
                "complete": bool(payload.get("complete", False)),
                "reason": str(payload.get("reason") or "").strip(),
                "raw_text": response.raw_text,
                "latency_s": (response.payload or {}).get("latency_s"),
            }
        except Exception as exc:
            return {
                "complete": False,
                "reason": "final task verifier unavailable; current outcome is unknown",
                "error": f"{type(exc).__name__}: {exc}",
            }

    def decide(
        self,
        task: str,
        subgoal: dict[str, Any],
        recent_moves: str,
        previous_direction: str,
        gripper_state: str,
        agentview_image,
        wrist_image=None,
        prev_agentview_image=None,
        proprio: dict[str, Any] | None = None,
        recovery_context: str = "",
        capability_context: str = "",
        debug: bool = False,
    ) -> VLMResponse:
        # Tools choose the prompt's context block and answer protocol. proprio (context
        # provider) is additive; mcq (answer protocol) is one-or-the-other with the default.
        proprio_block = (
            self.proprio_plugin.render(
                proprio, self.table_height_m, holding=(gripper_state == "CLOSED")
            )
            if self.proprio_plugin is not None
            else ""
        )
        gripper_proprio = (
            self.proprio_plugin.render_gripper(proprio)
            if self.proprio_plugin is not None
            else ""
        )
        mem_text = (
            self.mem_text_plugin.render_recent(recent_moves)
            if self.mem_text_plugin is not None
            else ""
        )
        mem_text_rules = (
            self.mem_text_plugin.render_rules() if self.mem_text_plugin is not None else ""
        )
        wrist_block = wrist_marker_prompt() if self._wants_wrist() else ""
        action_chunk_block = (
            self.action_chunk_plugin.render_prompt()
            if self.action_chunk_plugin is not None
            else ""
        )
        rotation_block = (
            self.rotation_plugin.render_prompt() if self.rotation_plugin is not None else ""
        )
        mcq = self._active_protocol()
        if mcq is not None:
            # An answer-protocol tool owns the whole alphabet; rotation (an extra action in
            # the default protocol) does not compose with it, so mcq wins if both are on.
            allowed_tokens: Sequence[str] = list(mcq.answer_tokens)
            output_contract = mcq.output_contract()
        else:
            rotation_tokens = (
                tuple(self.rotation_plugin.action_tokens())
                if self.rotation_plugin is not None
                else ()
            )
            allowed_tokens = tuple(CONTROLLER_TOKENS) + rotation_tokens
            output_contract = _default_output_contract(rotation_tokens)

        stage_name = str(subgoal.get("motion", "")).strip().upper()
        prompt_template = (
            self.transport_prompt_template
            if stage_name == "TRANSPORT" and self.transport_prompt_template
            else self.prompt_template
        )
        formatted_prompt = prompt_template.format(
                task=task,
                subgoal_json=json.dumps(subgoal, sort_keys=True),
                stage=str(subgoal.get("motion", "")),
                target=str(subgoal.get("target", "")),
                affordance=(
                    self.affordance_plugin.afford_field(
                        "arm", str(subgoal.get("affordance", ""))
                    )
                    if self.affordance_plugin is not None
                    else str(subgoal.get("affordance", ""))
                ),
                description=str(subgoal.get("description", "")),
                completion=str(subgoal.get("completion", "")),
                gripper_state=gripper_state,
                gripper_color=self.gripper_color,
                recent_moves=recent_moves or "none",
                mem_text=mem_text,
                mem_text_rules=mem_text_rules,
                variable_step=wrist_block,
                action_chunk=action_chunk_block,
                rotation=rotation_block,
                recovery=str(recovery_context or ""),
                proprio=proprio_block,
                gripper_proprio=gripper_proprio,
                output_contract=("" if stage_name == "TRANSPORT" else output_contract),
            )
        if stage_name == "TRANSPORT":
            # Keep live route/intent evidence immediately before the answer
            # contract. TRANSPORT deliberately excludes the staged controller's
            # common context: that LIBERO context contains pre-grasp height rules
            # which are correct before GRASP but directly contradict a carried
            # payload's clearance leg.
            prompt = _join_prompt_parts(
                formatted_prompt,
                str(capability_context or "").strip(),
                output_contract,
            )
        else:
            prompt = _join_prompt_parts(
                self.common_context,
                formatted_prompt,
                str(capability_context or "").strip(),
            )
        ablation = self.action_ablation_plugin
        ablation_on = ablation is not None and getattr(ablation, "enabled", False)
        # Blind mode's two-frame review: when the runner supplies the frame captured
        # BEFORE the previous direction token, the prompt asks the model to judge
        # that action from the before/after pair and update its table (notes are
        # never written blind, at decision time).
        review_symbol = (
            ablation.review_symbol(previous_direction)
            if (ablation_on and prev_agentview_image is not None)
            else None
        )
        if ablation_on:
            # Arm the blind harvest gate: only NOTE[<reviewed symbol>] is recorded
            # this step (None -> no review requested -> all NOTE lines ignored).
            ablation.begin_step(review_symbol)
            # One funnel for the whole setting: symbolize/strip the assembled prompt
            # (covers run-time injections) and substitute the blind table/review.
            prompt = ablation.filter_final(prompt, review_symbol=review_symbol)
        self.last_prompt = prompt
        if review_symbol is not None:
            # The review text promises the BEFORE frame as the LAST attached image.
            wrist_image = ([wrist_image] if wrist_image is not None else []) + [
                prev_agentview_image
            ]
        # On a rare double parse failure, commit to the last movement direction
        # instead of crashing the episode (degraded-output safeguard, not control).
        # Under MCQ the fallback must be expressed in the answer alphabet (a letter).
        fallback_token = (
            previous_direction if previous_direction in DIRECTION_TOKENS else "MV_DOWN"
        )
        if mcq is not None:
            fallback_token = mcq.fallback_answer(fallback_token)

        if self.cot_mode:
            response = _complete_cot_decision(
                self.client,
                prompt,
                allowed_tokens,
                agentview_image,
                wrist_image=wrist_image,
                fallback_token=fallback_token,
                fallback_reason="controller cot commit-fallback after no recoverable token",
                debug=debug,
            )
        else:
            response = _complete_decision_json(
                self.client,
                prompt,
                allowed_tokens,
                agentview_image,
                wrist_image=wrist_image,
                fallback_token=fallback_token,
                fallback_reason="controller commit-fallback after invalid JSON and token retry",
                debug=debug,
            )
        if mcq is not None:
            response = mcq.map_response(response)
        # Shared wrist-visibility judgment: recover the WRIST: YES/NO marker from the model's
        # reasoning/output and stash it on the payload for the runner to forward to its
        # consumers (variable_step step size, action_chunk step count). No marker -> None.
        if self._wants_wrist() and isinstance(response.payload, dict):
            json_obj = response.payload.get("json")
            reasoning = json_obj.get("reasoning") if isinstance(json_obj, dict) else ""
            text = str(reasoning or "") or (response.raw_text or "")
            if ablation is not None and getattr(ablation, "answer_protocol", False):
                # Letters modes: the model plans in ACT_* symbols (the prompt was
                # symbolized), but parse_plan reads atomic tokens.
                text = ablation.decode_symbols(text)
            target_in_wrist = parse_wrist_marker(text)
            response.payload["target_in_wrist"] = target_in_wrist
            # action_chunk: when far, recover the model's planned move sequence (PLAN: ...)
            # so the runner can execute it open-loop. [] -> single step.
            if getattr(self.action_chunk_plugin, "enabled", False):
                response.payload["chunk_plan"] = self.action_chunk_plugin.parse_plan(
                    text, target_in_wrist
                )
        return response

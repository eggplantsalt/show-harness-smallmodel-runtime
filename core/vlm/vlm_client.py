from __future__ import annotations

import re
import json
import copy
import random
import time
import base64
import hashlib
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Optional, Sequence
from urllib.parse import urlparse

import requests

from core.record.images import image_to_data_url



# 输入是一个最终定稿的请求字典（payload），里面装着要发给 VLM 的消息、模型名、温度、token 上限、图片以及可能的 JSON 格式要求；输出是一个审计字典，记录了模型名、token/温度设置、文本哈希、整个请求的哈希、每张图片的哈希与尺寸、图片数量以及 JSON schema 的哈希；
# 它的作用就是在请求发出前给这次调用生成一份轻量、可复现的“指纹存档”，让以后能快速判断两次请求是否完全相同、方便复现实验和排查问题，而无需保存原始图片和完整文本。
def _request_audit(payload: dict[str, Any]) -> dict[str, Any]:
    """Hash the exact finalized text and image bytes placed on the wire."""
    text_parts: list[str] = []
    image_parts: list[dict[str, Any]] = []
    for message in payload.get("messages", []):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            if isinstance(content, str):
                text_parts.append(content)
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                item: dict[str, Any] = {"sha256": None, "size": None}
                if isinstance(url, str) and url.startswith("data:") and "," in url:
                    try:
                        raw = base64.b64decode(url.split(",", 1)[1], validate=True)
                        item["sha256"] = hashlib.sha256(raw).hexdigest()
                        try:
                            from PIL import Image
                            with Image.open(io.BytesIO(raw)) as image:
                                item["size"] = list(image.size)
                        except Exception:
                            pass
                    except (ValueError, TypeError):
                        pass
                item["detail"] = (part.get("image_url") or {}).get("detail")
                image_parts.append(item)
    joined_text = "\n".join(text_parts)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    response_format = payload.get("response_format")
    response_format_audit = None
    if isinstance(response_format, dict):
        format_schema = response_format.get("json_schema")
        format_schema = format_schema.get("schema") if isinstance(format_schema, dict) else None
        response_format_audit = {
            "type": response_format.get("type"),
            "schema_sha256": (
                hashlib.sha256(
                    json.dumps(
                        format_schema, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest()
                if isinstance(format_schema, dict)
                else None
            ),
        }
    return {
        "model": payload.get("model"),
        "max_tokens": payload.get("max_tokens", payload.get("max_completion_tokens")),
        "thinking_token_budget": payload.get("thinking_token_budget"),
        "temperature": payload.get("temperature"),
        "prompt_sha256": hashlib.sha256(joined_text.encode("utf-8")).hexdigest(),
        "request_sha256": hashlib.sha256(canonical).hexdigest(),
        "images": image_parts,
        "image_count": len(image_parts),
        "response_format": response_format_audit,
    }


def _payload_has_images(payload: dict[str, Any]) -> bool:
    """Return whether an OpenAI chat payload contains an image content part."""
    for message in payload.get("messages", []):
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in content
        ):
            return True
    return False


def _content_to_text(content: Any) -> str:
    """Normalize OpenAI-compatible message content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""

# Rate-limit / transient-error retry defaults (e.g. Gemini free-tier RPM 429s). The
# client retries 429 and 5xx with exponential backoff, honouring a Retry-After header or
# a provider "retry in Xs" / "retryDelay" hint when present.
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BASE_S = 2.0
DEFAULT_RETRY_MAX_S = 30.0
# A provider-specified wait (Retry-After / "retry in Xs") is honoured up to this ceiling.
# It must exceed a typical per-minute RPM window (~60 s) -- clamping a hinted wait below
# the stated value would just guarantee the retry is still rate-limited. The ceiling only
# guards against a pathological hint (e.g. a daily-quota "retry in 3600s").
RETRY_HINT_CEILING_S = 90.0
# Besides 5xx: request timeout, transient conflict, too-early, and rate limit are all
# worth retrying; every other 4xx is a caller/auth error that a retry cannot fix.
RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429})


# Both vendor spellings of "do not emit a reasoning block". Every bare-token shape forces
# this: the MVTOKEN LoRAs were trained with the assistant turn starting at the token, so a
# <think> preamble is off-distribution, not just wasted decode.
_NO_THINKING = {"enable_thinking": False, "thinking": False}


@dataclass
class VLMResponse:
    token: str
    raw_text: str
    payload: dict


class VLMParseError(RuntimeError):
    def __init__(
        self,
        message: str,
        raw_text: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.payload = dict(payload or {})


class VLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout_s: float,
        max_tokens: int,
        temperature: float,
        chat_template_kwargs: Optional[dict] = None,
        cot_max_tokens: Optional[int] = None,
        thinking_token_budget: Optional[int] = None,
        token_thinking_budget: Optional[int] = None,
        reasoning_directive: Optional[str] = None,
        provider: str = "vllm",
        api_dialect: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        max_retries: Optional[int] = None,
        retry_base_delay_s: Optional[float] = None,
        retry_max_delay_s: Optional[float] = None,
        api_key_refresh=None,
        vision_model: Optional[str] = None,
        vision_max_tokens: Optional[int] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = float(timeout_s)
        self.max_tokens = int(max_tokens)
        # Decode budget for the chain-of-thought controller path; caps a verbose
        # thinker's per-step latency. None -> the role falls back to its 1024 floor.
        self.cot_max_tokens = int(cot_max_tokens) if cot_max_tokens else None
        self.thinking_token_budget = (
            max(1, int(thinking_token_budget)) if thinking_token_budget else None
        )
        self.token_thinking_budget = (
            max(1, int(token_thinking_budget)) if token_thinking_budget else None
        )
        # Appended to the CoT prompt to steer reasoning length/depth per backend
        # (e.g. brief for a fast model, more thorough for a reasoning one).
        self.reasoning_directive = str(reasoning_directive or "")
        self.temperature = float(temperature)
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.reasoning_enabled = _reasoning_enabled(self.chat_template_kwargs)
        # provider names the endpoint/auth family. api_dialect selects the request shape:
        # "vllm" (local, guided decoding + chat_template_kwargs), "openai" (hosted
        # OpenAI Chat Completions / Azure OpenAI deployments), or "gemini" (Gemini-style
        # OpenAI-compatible endpoints). Hosted providers reject vLLM-only fields and need
        # the payload rewritten (see _finalize_payload).
        self.provider = str(provider or "vllm").lower()
        self.api_dialect = str(api_dialect or self.provider).lower()
        self.reasoning_effort = reasoning_effort or None
        # Rate-limit / transient-error retry (on by default; tune via vlm_backends).
        self.max_retries = DEFAULT_MAX_RETRIES if max_retries is None else max(0, int(max_retries))
        self.retry_base_delay_s = (
            DEFAULT_RETRY_BASE_S if retry_base_delay_s is None else float(retry_base_delay_s)
        )
        self.retry_max_delay_s = (
            DEFAULT_RETRY_MAX_S if retry_max_delay_s is None else float(retry_max_delay_s)
        )
        # Zero-arg callable returning the CURRENT key (core.config.make_api_key_refresher):
        # a short-lived bearer token can expire mid-rollout while an external refresher
        # keeps a fresh one in secrets.env, so on a 401/403 the client re-resolves the
        # key and retries instead of crashing.
        self.api_key_refresh = api_key_refresh
        # Some fast reasoning models are text-only.  When configured, this companion
        # VLM converts live RGB frames into an API-produced textual observation before
        # the primary model plans or controls.  It never receives simulator state.
        self.vision_model = str(vision_model or "").strip() or None
        self.vision_max_tokens = int(vision_max_tokens or 512)
        self._api_key = api_key or "EMPTY"
        self.session = requests.Session()
        # A local vLLM server must not inherit the machine's HTTP proxy.  In this
        # environment the proxy can forward text requests but returns an empty 502
        # for large base64 image bodies, making a healthy local multimodal server
        # look broken.  Hosted endpoints intentionally keep requests' normal proxy
        # behavior.
        host = urlparse(self.base_url).hostname
        if host in {"127.0.0.1", "localhost", "::1"}:
            self.session.trust_env = False
        # Runtime accounting is deliberately provider-agnostic.  Local vLLM reports
        # zero monetary cost, while hosted backends can be priced offline from the
        # recorded token totals without putting provider-specific prices in rollout code.
        self.metrics: dict[str, float | int] = {
            "model_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_s": 0.0,
            "api_cost_usd": 0.0,
        }
        # Trust the key the caller resolved (core.config.resolve_api_key reads it from
        # the environment / secrets.env). Do NOT re-read env here: secrets.env sets
        # VLLM_API_KEY='EMPTY', a truthy string that would otherwise clobber a hosted
        # OpenAI/Gemini key and turn every request into a 401.
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
        )

    def _is_openai(self) -> bool:
        return self.api_dialect == "openai"

    def _is_gemini(self) -> bool:
        return self.api_dialect == "gemini"

    def _is_siliconflow(self) -> bool:
        return self.api_dialect == "siliconflow"

    def _is_hosted(self) -> bool:
        """Hosted OpenAI-compatible APIs reject vLLM-only fields
        (chat_template_kwargs / guided_* / logprobs) and need a rewritten payload."""
        return self.provider in ("openai", "gemini") or self.api_dialect in (
            "openai",
            "gemini",
            "deepseek",
            "siliconflow",
        )

    def _finalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Rewrite a vLLM-shaped payload for the active provider.

        api_dialect="vllm": use vLLM's current JSON-schema ``response_format`` field.
        For hosted providers the payload is rebuilt with only accepted fields
        (chat_template_kwargs / guided_choice / logprobs are dropped) and the local
        schema becomes an OpenAI-style JSON-object ``response_format``. The
        dialects differ:
          * openai: max_tokens -> max_completion_tokens; temperature only forwarded when
            it is the default 1.0 (reasoning models reject a custom value); reasoning_effort
            added when set.
          * gemini: standard max_tokens and temperature (0.0 is accepted and wanted for
            determinism); reasoning_effort forwarded when set (Gemini 2.5+/3.x thinking
            models accept it via the OpenAI-compat layer; omitted otherwise so non-thinking
            models like flash-lite are unaffected).
        """
        if not self._is_hosted():
            out = dict(payload)
            schema = out.pop("guided_json", None)
            if isinstance(schema, dict):
                out["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "harness_response",
                        "schema": schema,
                        "strict": True,
                    },
                }
            return out
        out: dict[str, Any] = {"model": payload["model"], "messages": payload["messages"]}
        if self._is_openai():
            if "max_tokens" in payload:
                out["max_completion_tokens"] = payload["max_tokens"]
            temperature = payload.get("temperature")
            if temperature is not None and float(temperature) == 1.0:
                out["temperature"] = 1.0
        else:  # Gemini / DeepSeek / SiliconFlow OpenAI-compatible APIs
            if "max_tokens" in payload:
                out["max_tokens"] = payload["max_tokens"]
            temperature = payload.get("temperature")
            if temperature is not None:
                out["temperature"] = float(temperature)
        # reasoning_effort applies to thinking models on both hosted providers; forward it
        # only when configured (chat / non-thinking models omit it).
        if self.api_dialect == "deepseek":
            # DeepSeek enables thinking by default. Its separate reasoning output can
            # consume an atomic controller's entire decode budget without an action.
            out["thinking"] = {"type": "enabled" if self.reasoning_enabled else "disabled"}
        if self._is_siliconflow():
            # SiliconFlow documents this flat field for DeepSeek-V4-Flash.
            # Sending it explicitly prevents the service's default reasoning
            # budget from consuming a one-step controller answer.
            out["enable_thinking"] = bool(self.reasoning_enabled)
        if self.reasoning_effort and not (
            self.api_dialect == "deepseek"
            and (not self.reasoning_enabled or self.reasoning_effort == "none")
        ):
            out["reasoning_effort"] = self.reasoning_effort
        if "guided_json" in payload:
            out["response_format"] = {"type": "json_object"}
        return out

    def _refresh_auth(self) -> bool:
        """Re-resolve the API key via the configured refresher and install it.

        Returns True only when a DIFFERENT, non-empty key was obtained -- retrying
        with the same rejected key would just 401 again."""
        if self.api_key_refresh is None:
            return False
        try:
            new_key = self.api_key_refresh()
        except Exception as exc:  # noqa: BLE001 - refresh is best-effort
            print(f"[vlm] API key refresh failed: {exc}")
            return False
        if not new_key or new_key == self._api_key:
            return False
        self._api_key = new_key
        self.session.headers["Authorization"] = f"Bearer {new_key}"
        return True

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to /chat/completions and return the parsed completion body, with
        retry+backoff on rate limits (HTTP 429), transient HTTP errors (408/409/425/5xx),
        network errors, and malformed "successful" bodies -- a relaying proxy (e.g. the
        showrobot Gemini tunnel) can answer 2xx with an error JSON, an HTML page, or an
        empty choices list while its upstream is rate-limited or restarting, so those are
        retried too instead of crashing the rollout in the caller's parse. Honours a
        Retry-After header or a provider 'retry in Xs' / 'retryDelay' body hint; otherwise
        exponential backoff with jitter. Raises RuntimeError on a non-retryable error
        (e.g. 400/401/404) or once ``max_retries`` is exhausted."""
        if self.vision_model and _payload_has_images(payload):
            payload = self._replace_images_with_vision_observation(payload)

        url = f"{self.base_url}/chat/completions"
        delay = self.retry_base_delay_s
        last = "unknown error"
        auth_refreshed = False  # at most one credential reload per request
        for attempt in range(self.max_retries + 1):
            # Default wait is exponential backoff, capped at retry_max_delay_s.
            wait = min(delay, self.retry_max_delay_s)
            request_started = time.monotonic()
            try:
                self.metrics["model_calls"] = int(self.metrics["model_calls"]) + 1
                resp = self.session.post(url, json=payload, timeout=self.timeout_s)
            except requests.ConnectionError as exc:
                last = f"request error: {exc}"  # network/tunnel hiccup -> retry
                if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
                    # A refused localhost connection is a DOWN TUNNEL/PROXY, not a
                    # rate limit -- without this hint the retry banner reads like one.
                    last = (
                        f"nothing listening at {self.base_url} -- the local "
                        "server or tunnel/proxy for this backend is DOWN; "
                        "start it and retry"
                    )
            except requests.RequestException as exc:
                last = f"request error: {exc}"  # network/tunnel hiccup -> retry
            else:
                if resp.status_code < 400:
                    data, problem = _chat_completion_data(resp)
                    if data is not None:
                        self.metrics["latency_s"] = float(self.metrics["latency_s"]) + (
                            time.monotonic() - request_started
                        )
                        usage = data.get("usage")
                        if isinstance(usage, dict):
                            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                                value = usage.get(key)
                                if isinstance(value, (int, float)):
                                    self.metrics[key] = int(self.metrics[key]) + int(value)
                        return data
                    last = f"bad completion body: {problem}"  # proxy glitch -> retry
                else:
                    last = f"HTTP {resp.status_code}: {resp.text[:1000]}"
                    if resp.status_code in (401, 403):
                        # Expired/rotated credential. Re-resolve it once (secrets.env
                        # may hold a fresher token) and retry immediately; without a
                        # fresh key this is a hard failure.
                        if not auth_refreshed and self._refresh_auth():
                            auth_refreshed = True
                            print("[vlm] auth rejected; retrying with the refreshed API key")
                            continue
                        raise RuntimeError(
                            f"VLM chat completion failed: {last} -- the API token is "
                            "invalid or expired and no fresher one was found in "
                            "configs/secrets.env. Refresh the key for this backend "
                            "(short-lived tokens need their refresher running), then retry."
                        )
                    if not (
                        resp.status_code in RETRYABLE_STATUS_CODES
                        or 500 <= resp.status_code < 600
                    ):
                        raise RuntimeError(f"VLM chat completion failed: {last}")
                    hinted = _retry_after_seconds(resp)
                    if hinted is not None:
                        # Honour the provider's stated wait FULLY (+1s margin) so the
                        # retry lands after the rate window resets -- but never retry
                        # SOONER than the exponential schedule: a congested shared pool
                        # keeps answering "retry after 1 seconds" while
                        # staying exhausted for a whole per-minute window, and a
                        # hint-pinned ~2s cadence burns every retry inside it.
                        wait = min(max(hinted + 1.0, wait), RETRY_HINT_CEILING_S)
            if attempt >= self.max_retries:
                break
            # Jitter de-synchronises concurrent clients (e.g. two arms sharing one
            # proxy) so they do not all land in the same rate window again.
            wait = max(wait, 0.0) * random.uniform(0.8, 1.2)
            # Collapse whitespace so a pretty-printed JSON error body
            # shows its message instead of just its opening "{" line.
            print(
                f"[vlm] rate-limited/transient ({' '.join(last.split())[:160]}); "
                f"retry {attempt + 1}/{self.max_retries} in {wait:.1f}s"
            )
            time.sleep(wait)
            delay = min(delay * 2, self.retry_max_delay_s)
        raise RuntimeError(
            f"VLM chat completion failed after {self.max_retries} retries: {last}"
        )

    def _replace_images_with_vision_observation(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Use the configured VLM only as an RGB-to-text sensor for a text model."""
        images: list[dict[str, Any]] = []
        for message in payload.get("messages", []):
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                images.extend(part for part in content if isinstance(part, dict) and part.get("type") == "image_url")
        if not images:
            return payload

        vision_prompt = (
            "You are the visual sensor for a robot manipulation controller. Analyze the live RGB images only. "
            "Describe visible target objects, gripper/fingers, their relative image positions, occlusion, and whether "
            "the gripper is above/beside/on an object. Do not invent hidden objects or use external knowledge. "
            "Give a compact factual observation for a separate text-only planner."
        )
        vision_payload = {
            "model": self.vision_model,
            "messages": [{"role": "user", "content": images + [{"type": "text", "text": vision_prompt}]}],
            "max_tokens": self.vision_max_tokens,
            "temperature": 0.0,
        }
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions", json=vision_payload, timeout=self.timeout_s
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Vision companion request failed: {exc}") from exc
        if response.status_code >= 400:
            raise RuntimeError(
                f"Vision companion chat completion failed: HTTP {response.status_code}: {response.text[:1000]}"
            )
        data, problem = _chat_completion_data(response)
        if data is None:
            raise RuntimeError(f"Vision companion returned an invalid completion: {problem}")
        observation = _content_to_text(data["choices"][0]["message"].get("content")).strip()
        if not observation:
            raise RuntimeError("Vision companion returned an empty observation")

        rewritten = copy.deepcopy(payload)
        for message in rewritten.get("messages", []):
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            text_parts = [str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"]
            message["content"] = "\n".join(text_parts + ["LIVE RGB OBSERVATION (from visual sensor):\n" + observation])
        return rewritten

    def health_check(self, wait_s: float = 0.0, poll_s: float = 5.0) -> None:
        url = f"{self.base_url}/models"
        deadline = time.monotonic() + max(0.0, float(wait_s))
        last_error = None
        while True:
            try:
                response = self.session.get(url, timeout=min(self.timeout_s, 20.0))
                if response.status_code < 400:
                    return
                last_error = RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:400]}"
                )
            except requests.RequestException as exc:
                last_error = exc
            if time.monotonic() >= deadline:
                if self._is_hosted():
                    hint = (
                        f"OpenAI-compatible endpoint not ready at {url}. Check base_url, the "
                        "model id, and the configured api_key_env in configs/secrets.env "
                        "or your shell."
                    )
                else:
                    hint = (
                        f"vLLM endpoint is not ready at {url}. Start it with "
                        "`bash scripts/serve_vlm.sh` and wait until `/v1/models` responds "
                        "before running the episode."
                    )
                raise RuntimeError(f"{hint} Last error: {last_error}") from last_error
            time.sleep(max(0.5, float(poll_s)))

    def complete_token(
        self,
        prompt: str,
        allowed_tokens: Sequence[str],
        agentview_image,
        wrist_image=None,
        chat_template_kwargs: Optional[dict] = None,
        debug: bool = False,
        agentview_label: Optional[str] = "Image A: agentview RGB",
        wrist_label: Optional[str] = "Image B: wrist RGB",
    ) -> VLMResponse:
        if not allowed_tokens:
            raise ValueError("allowed_tokens must not be empty")
        content = _message_content(
            prompt,
            agentview_image,
            wrist_image,
            agentview_label=agentview_label,
            wrist_label=wrist_label,
        )
        thinking_budget = self.token_thinking_budget
        if self.reasoning_enabled and thinking_budget is None and not self._is_hosted():
            # Thinking checkpoints can ignore an `enable_thinking=false` template
            # override. Give atomic ReAct choices a small, explicit reasoning budget
            # and reserve enough completion tokens for the selected action token.
            thinking_budget = 32
        active_chat_kwargs = self._reasoning_chat_kwargs(
            chat_template_kwargs, thinking_budget
        )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self._token_call_budget(thinking_budget),
            "guided_choice": list(allowed_tokens),
            "chat_template_kwargs": active_chat_kwargs,
        }
        if thinking_budget is not None:
            payload["thinking_token_budget"] = thinking_budget
        if debug:
            payload["logprobs"] = True
        data, raw_text, latency_s = self._post_completion(payload)
        try:
            token = _parse_single_token(raw_text, allowed_tokens)
        except RuntimeError as exc:
            retry_kwargs = dict(active_chat_kwargs)
            retry_kwargs.update({"enable_thinking": False, "thinking": False})
            retry_prompt = _strict_token_prompt(prompt, allowed_tokens)
            retry_payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": _message_content(
                            retry_prompt,
                            agentview_image,
                            wrist_image,
                            agentview_label=agentview_label,
                            wrist_label=wrist_label,
                        ),
                    }
                ],
                "temperature": 0.0,
                "max_tokens": self._token_call_budget(thinking_budget),
                "guided_choice": list(allowed_tokens),
                "chat_template_kwargs": self._reasoning_chat_kwargs(
                    retry_kwargs, thinking_budget
                ),
            }
            if thinking_budget is not None:
                retry_payload["thinking_token_budget"] = thinking_budget
            if debug:
                retry_payload["logprobs"] = True
            retry_data, retry_raw_text, retry_latency_s = self._post_completion(
                retry_payload
            )
            try:
                token = _parse_single_token(retry_raw_text, allowed_tokens)
            except RuntimeError as retry_exc:
                raise RuntimeError(
                    f"{exc}; strict no-thinking token retry also failed: {retry_exc}"
                ) from retry_exc
            raw_text = f"{retry_raw_text.strip()} [strict_no_thinking_retry]"
            data = {"first": data, "retry": retry_data} if debug else {}
            latency_s += retry_latency_s
        return VLMResponse(
            token=token,
            raw_text=raw_text,
            payload=_response_payload(data if debug else {}, latency_s),
        )

    # -- bare-token shapes (MVTOKEN fine-tuned policies) ---------------------
    # These live here, not in a wrapper, because the transport they need is the same
    # transport the zero-shot shapes above use: model/temperature/budget/chat-template
    # resolution and _post_completion. What makes them different is only the request
    # SHAPE -- no guided decoding, thinking forced off, the reply parsed as a bare token
    # -- which is a fine thing for a client to offer alongside complete_token.

    def complete_action_token(
        self,
        prompt: str,
        allowed_tokens: Sequence[str],
        agentview_image,
        wrist_image=None,
        debug: bool = False,
    ) -> VLMResponse:
        """One call -> ONE bare atomic action token, for a model fine-tuned to emit it.

        The request shape the MVTOKEN LoRAs were trained and validated under: images first
        (agentview, then any extra views), prompt text appended after them, thinking forced
        OFF, and NO ``guided_choice`` -- the model returns a clean bare token on its own and
        we parse it. That last part is the deliberate difference from :meth:`complete_token`,
        which constrains a NON-fine-tuned model's answer with guided decoding and falls back
        to a strict retry. Constraining a fine-tuned model instead hides whether it actually
        learned the vocabulary.
        """
        if not allowed_tokens:
            raise ValueError("allowed_tokens must not be empty")
        content = _message_content(
            prompt,
            agentview_image,
            wrist_image,
            agentview_label=None,
            wrist_label=None,
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self._token_call_budget(),
            "chat_template_kwargs": self._chat_template_kwargs(_NO_THINKING),
        }
        if debug:
            payload["logprobs"] = True
        data, raw_text, latency_s = self._post_completion(payload)
        token = _parse_single_token(raw_text, allowed_tokens)
        return VLMResponse(
            token=token,
            raw_text=raw_text,
            payload=_response_payload(data if debug else {}, latency_s),
        )

    def complete_action_token_pair(
        self,
        prompt: str,
        allowed_tokens: Sequence[str],
        agentview_image,
        wrist_image=None,
        debug: bool = False,
    ) -> VLMResponse:
        """One call -> TWO bare tokens (``"<left> <right>"``), the dual-arm ``once`` scheme.

        Same shape as :meth:`complete_action_token`; only the decode budget is doubled and
        the reply is parsed as a pair. The right token is conditioned on the left through
        the decoder's own autoregression, which is what makes a single call sufficient.
        """
        if not allowed_tokens:
            raise ValueError("allowed_tokens must not be empty")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": _message_content(
                        prompt,
                        agentview_image,
                        wrist_image,
                        agentview_label=None,
                        wrist_label=None,
                    ),
                }
            ],
            "temperature": self.temperature,
            # Two tokens instead of one; the single-token cap would risk a truncated pair.
            "max_tokens": self._token_call_budget() * 2,
            "chat_template_kwargs": self._chat_template_kwargs(_NO_THINKING),
        }
        if debug:
            payload["logprobs"] = True
        data, raw_text, latency_s = self._post_completion(payload)
        left, right = parse_token_pair(raw_text, allowed_tokens)
        return VLMResponse(
            token=f"{left} {right}",
            raw_text=raw_text,
            payload=_response_payload(data if debug else {}, latency_s),
        )

    def complete_action_token_chain(
        self,
        prompt: str,
        followup_prompt: str,
        allowed_tokens: Sequence[str],
        agentview_image,
        wrist_image=None,
        debug: bool = False,
    ) -> VLMResponse:
        """ONE image encoding, TWO answers -- the dual-arm ``chain`` scheme.

        Mirrors the ShareGPT training sample exactly: turn 1 carries the views and asks for
        the LEFT token, the model's answer is fed back verbatim as the assistant turn, and a
        short TEXT-ONLY follow-up asks for the RIGHT one -- which therefore SEES the left.

        Two HTTP round-trips, but only the first carries images and the second reuses that
        prefix verbatim, so vLLM's prefix cache serves it without re-encoding the views.
        That is the whole point of ``chain`` over ``twice``: the same sequential conditioning
        at half the image cost.
        """
        if not allowed_tokens:
            raise ValueError("allowed_tokens must not be empty")
        chat_kwargs = self._chat_template_kwargs(_NO_THINKING)

        def _ask(messages: list[dict[str, Any]]):
            call: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self._token_call_budget(),
                "chat_template_kwargs": chat_kwargs,
            }
            if debug:
                call["logprobs"] = True
            return self._post_completion(call)

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": _message_content(
                    prompt,
                    agentview_image,
                    wrist_image,
                    agentview_label=None,
                    wrist_label=None,
                ),
            }
        ]
        data_l, raw_l, latency_l = _ask(messages)
        left = _parse_single_token(raw_l, allowed_tokens)

        # The training sample's second user turn is plain text (no images).
        messages = messages + [
            {"role": "assistant", "content": left},
            {"role": "user", "content": followup_prompt},
        ]
        data_r, raw_r, latency_r = _ask(messages)
        right = _parse_single_token(raw_r, allowed_tokens)

        return VLMResponse(
            token=f"{left} {right}",
            raw_text=f"{raw_l.strip()} || {raw_r.strip()}",
            payload=_response_payload(
                {"left": data_l, "right": data_r} if debug else {},
                latency_l + latency_r,
            ),
        )

    def complete_text(
        self,
        prompt: str,
        agentview_image,
        wrist_image=None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        chat_template_kwargs: Optional[dict] = None,
        debug: bool = False,
        strip_reasoning: bool = True,
        agentview_label: Optional[str] = "Image A: agentview RGB",
        wrist_label: Optional[str] = "Image B: wrist RGB",
        image_detail: Optional[str] = None,
        thinking_token_budget: Optional[int] = None,
    ) -> VLMResponse:
        content = _message_content(
            prompt,
            agentview_image,
            wrist_image,
            agentview_label=agentview_label,
            wrist_label=wrist_label,
            image_detail=image_detail,
        )
        effective_thinking_budget = thinking_token_budget
        if effective_thinking_budget is None and self.reasoning_enabled:
            effective_thinking_budget = self.thinking_token_budget
        active_chat_kwargs = self._reasoning_chat_kwargs(
            chat_template_kwargs, effective_thinking_budget
        )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": (
                self.temperature if temperature is None else float(temperature)
            ),
            "max_tokens": self.max_tokens if max_tokens is None else int(max_tokens),
            "chat_template_kwargs": active_chat_kwargs,
        }
        if effective_thinking_budget is not None:
            payload["thinking_token_budget"] = max(1, int(effective_thinking_budget))
        if debug:
            payload["logprobs"] = True
        payload = self._finalize_payload(payload)
        audit = _request_audit(payload)
        started = time.monotonic()
        data = self._post_chat(payload)
        latency_s = time.monotonic() - started
        message = data["choices"][0]["message"]
        if data["choices"][0].get("finish_reason") == "length":
            failure_payload = _response_payload(
                {
                    **_completion_metadata(data, message),
                    "request_audit": audit,
                    "failure": "truncated_final_answer",
                },
                latency_s,
            )
            raise VLMParseError(
                "VLM final answer was truncated at the token limit",
                _message_text(message),
                failure_payload,
            )
        # strip_reasoning=False keeps the <think>...</think> chain-of-thought in the
        # text (for CoT logging/analysis); token recovery still finds the answer.
        message_text = _message_text(message)
        raw_text = (
            _strip_reasoning_artifacts(message_text)
            if strip_reasoning
            else message_text
        )
        return VLMResponse(
            token="",
            raw_text=raw_text.strip(),
            payload=_response_payload(
                {**(data if debug else {}), **_completion_metadata(data, message)}, latency_s
            ),
        )

    def complete_json(
        self,
        prompt: str,
        agentview_image,
        wrist_image=None,
        schema: Optional[dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        thinking_token_budget: Optional[int] = None,
        chat_template_kwargs: Optional[dict] = None,
        debug: bool = False,
        agentview_label: Optional[str] = "Image A: agentview RGB",
        wrist_label: Optional[str] = "Image B: wrist RGB",
        image_detail: Optional[str] = None,
    ) -> VLMResponse:
        content = _message_content(
            prompt,
            agentview_image,
            wrist_image,
            agentview_label=agentview_label,
            wrist_label=wrist_label,
            image_detail=image_detail,
        )
        effective_thinking_budget = thinking_token_budget
        if effective_thinking_budget is None and self.reasoning_enabled:
            effective_thinking_budget = self.thinking_token_budget
        active_chat_kwargs = self._reasoning_chat_kwargs(
            chat_template_kwargs, effective_thinking_budget
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": (
                self.temperature if temperature is None else float(temperature)
            ),
            "max_tokens": self.max_tokens if max_tokens is None else int(max_tokens),
            "chat_template_kwargs": active_chat_kwargs,
        }
        if effective_thinking_budget is not None:
            payload["thinking_token_budget"] = max(1, int(effective_thinking_budget))
        if schema is not None:
            payload["guided_json"] = schema
        if debug:
            payload["logprobs"] = True
        payload = self._finalize_payload(payload)
        audit = _request_audit(payload)
        started = time.monotonic()
        data = self._post_chat(payload)
        latency_s = time.monotonic() - started
        message = data["choices"][0]["message"]
        if data["choices"][0].get("finish_reason") == "length":
            failure_payload = _response_payload(
                {
                    **_completion_metadata(data, message),
                    "request_audit": audit,
                    "failure": "truncated_final_answer",
                },
                latency_s,
            )
            raise VLMParseError(
                "VLM final answer was truncated at the token limit",
                _message_text(message),
                failure_payload,
            )
        raw_text = _strip_reasoning_artifacts(
            _message_text(message)
        ).strip()
        parsed = _parse_json_object(raw_text)
        clean_text = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
        response_payload = {"json": parsed, "raw": data} if debug else {"json": parsed}
        if isinstance(data.get("usage"), dict):
            response_payload["usage"] = data["usage"]
        response_payload.update(_completion_metadata(data, message))
        response_payload["request_audit"] = audit
        return VLMResponse(
            token="",
            raw_text=clean_text,
            payload=_response_payload(response_payload, latency_s),
        )

    def _chat_template_kwargs(self, override: Optional[dict]) -> dict:
        kwargs = dict(self.chat_template_kwargs)
        if override:
            kwargs.update(override)
        return kwargs

    def _reasoning_chat_kwargs(
        self,
        override: Optional[dict],
        thinking_token_budget: Optional[int],
    ) -> dict:
        kwargs = self._chat_template_kwargs(override)
        if self.reasoning_enabled or thinking_token_budget is not None:
            # Qwen3-VL-8B-Thinking still emits a reasoning channel when a request
            # asks the template to disable thinking. Keep the template and the
            # explicit per-request token budget in agreement.
            kwargs["enable_thinking"] = True
            kwargs.pop("thinking", None)
        return kwargs

    def _token_call_budget(self, thinking_token_budget: Optional[int] = None) -> int:
        # vLLM guided_choice answers in a few tokens, so 24 is plenty. Hosted providers
        # have no guided_choice (the token is recovered from free text), and a reasoning
        # model also burns completion tokens on hidden reasoning before the short answer,
        # so a 24-cap returns empty content; give them the full configured budget.
        if self._is_hosted():
            return int(self.max_tokens)
        if thinking_token_budget is not None:
            budget = min(
                max(1, int(thinking_token_budget)),
                max(1, self.max_tokens - 24),
            )
            return max(8, min(self.max_tokens, budget + 24))
        return max(8, min(self.max_tokens, 24))

    def _post_completion(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str, float]:
        payload = self._finalize_payload(payload)
        started = time.monotonic()
        data = self._post_chat(payload)
        latency_s = time.monotonic() - started
        raw_text = _strip_reasoning_artifacts(
            _message_text(data["choices"][0]["message"])
        )
        return data, raw_text, latency_s


def _chat_completion_data(response) -> tuple[Optional[dict[str, Any]], str]:
    """Parse and validate a 2xx /chat/completions body.

    Returns ``(data, "")`` when the body is a usable completion, else ``(None,
    problem)`` so ``_post_chat`` retries it as transient: a relaying proxy can answer
    2xx with an error JSON, an HTML error page (non-JSON), or a choices-less body
    while its upstream is rate-limited or restarting, and indexing
    ``choices[0]["message"]`` on those used to crash the rollout."""
    try:
        data = response.json()
    except ValueError as exc:
        return None, f"non-JSON body: {exc}"
    if not isinstance(data, dict):
        return None, f"unexpected body type: {type(data).__name__}"
    if data.get("error"):
        return None, f"error payload: {json.dumps(data['error'], ensure_ascii=False)[:300]}"
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None, "response has no choices"
    if not isinstance(choices[0].get("message"), dict):
        return None, "first choice has no message"
    return data, ""


def _retry_after_seconds(response) -> Optional[float]:
    """Best-effort: how long to wait before retrying, from the ``Retry-After`` header
    (delta-seconds or HTTP-date form) or a provider body hint (Gemini: ``retry in
    23.2s`` / ``"retryDelay": "23s"``). Returns ``None`` when nothing parseable so the
    caller falls back to exponential backoff."""
    header = (response.headers.get("Retry-After") or "").strip()
    if header:
        try:
            return max(0.0, float(header))
        except ValueError:
            try:
                when = parsedate_to_datetime(header)
                return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError):
                pass
    try:
        body = response.text or ""
    except Exception:  # noqa: BLE001 - body may be unavailable
        body = ""
    match = (
        re.search(r"retry in ([0-9.]+)\s*s", body, re.IGNORECASE)
        # Azure-style phrasing: 'Rate Limit Exceeded, retry after 1 seconds.'
        or re.search(r"retry after ([0-9.]+)\s*second", body, re.IGNORECASE)
        or re.search(r'"retryDelay"\s*:\s*"([0-9.]+)s"', body)
    )
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def _response_payload(payload: dict[str, Any], latency_s: float) -> dict[str, Any]:
    result = dict(payload)
    result["latency_s"] = round(float(latency_s), 3)
    return result


def _message_content(
    prompt: str,
    agentview_image,
    wrist_image=None,
    *,
    agentview_label: Optional[str] = "Image A: agentview RGB",
    wrist_label: Optional[str] = "Image B: wrist RGB",
    image_detail: Optional[str] = None,
) -> list[dict[str, Any]]:
    def image_part(image) -> dict[str, Any]:
        image_url: dict[str, Any] = {"url": image_to_data_url(image)}
        if image_detail:
            image_url["detail"] = str(image_detail)
        return {"type": "image_url", "image_url": image_url}

    # Images come first (agentview, then any extra views), text last.
    content: list[dict[str, Any]] = []
    if agentview_image is not None:
        content.append(image_part(agentview_image))
    # wrist_image may be one image or an ordered sequence of extra views (the dual-arm
    # controller sends [wrist_left, wrist_right] -> Image B, Image C). Every caller
    # that passes a single array is unchanged.
    if wrist_image is not None:
        extra_images = (
            list(wrist_image) if isinstance(wrist_image, (list, tuple)) else [wrist_image]
        )
        for image in extra_images:
            if image is None:
                continue
            content.append(image_part(image))
    # The images deliberately remain first in the wire order, but the model still needs a
    # stable mapping from that order to the camera roles.  Without these labels the prompt
    # can say "Wrist" while the model has to infer which unlabeled image is Image B.
    text_parts = []
    if agentview_image is not None and agentview_label:
        text_parts.append(str(agentview_label))
    if wrist_image is not None and wrist_label:
        text_parts.append(str(wrist_label))
    text_parts.append(prompt)
    # A text content part's "text" must be a STRING; passing the list straight through
    # makes hosted providers reject the whole request ("invalid_request_body"). Join the
    # accumulator so re-enabling the image labels above stays valid too.
    content.append({"type": "text", "text": "\n".join(text_parts)})
    return content


def _strict_token_prompt(prompt: str, allowed_tokens: Sequence[str]) -> str:
    return (
        prompt
        + "\n\nCritical output format: return exactly one token from this list and "
        "nothing else. Do not include thought, reasoning, markdown, punctuation, or prose.\n"
        + " ".join(allowed_tokens)
    )


def _parse_single_token(raw_text: str, allowed_tokens: Sequence[str]) -> str:
    stripped = _strip_reasoning_artifacts(raw_text).strip()
    if stripped in allowed_tokens:
        return stripped
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("token", "answer", "direction", "status", "decision"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip() in allowed_tokens:
                return value.strip()
    pattern = r"^(?:" + "|".join(re.escape(token) for token in allowed_tokens) + r")$"
    if re.match(pattern, stripped):
        return stripped
    suffix_pattern = (
        r"(?:^|[^A-Z_])("
        + "|".join(re.escape(token) for token in allowed_tokens)
        + r")\s*[.。]*\s*$"
    )
    suffix_match = re.search(suffix_pattern, stripped)
    if suffix_match:
        return suffix_match.group(1)
    recovered = recover_allowed_token(stripped, allowed_tokens)
    if recovered:
        return recovered
    raise RuntimeError(
        f"VLM returned invalid token {raw_text!r}; allowed tokens are {list(allowed_tokens)}"
    )


def parse_token_pair(raw_text: str, allowed_tokens: Sequence[str]) -> tuple[str, str]:
    """Parse a dual-arm ``once`` reply ``"<left> <right>"`` into its two tokens.

    Order matters (the LoRA is trained LEFT-first), so this scans the text POSITIONALLY --
    unlike ``_tokens_in_text``, which walks ``allowed_tokens`` and would hand back vocabulary
    order instead. The first two hits win, so a trailing period or a stray word is tolerated
    the same way the single-token parser tolerates it.
    """
    stripped = _strip_reasoning_artifacts(raw_text).strip()
    pattern = r"\b(" + "|".join(re.escape(t) for t in allowed_tokens) + r")\b"
    found = [m.group(1) for m in re.finditer(pattern, stripped)]
    if len(found) >= 2:
        return found[0], found[1]
    raise RuntimeError(
        f"VLM returned {raw_text!r}; the once/pair contract needs TWO tokens "
        f"('<left> <right>'), found {found}. Allowed tokens are {list(allowed_tokens)}"
    )


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    original_text = raw_text
    raw_text = _strip_reasoning_artifacts(raw_text).strip()
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value
    if value is not None:
        raise RuntimeError(f"VLM returned JSON that is not an object: {original_text!r}")

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", raw_text):
        try:
            candidate, _ = decoder.raw_decode(raw_text[match.start() :])
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, dict):
            continue
        if "subgoals" in candidate:
            return candidate
        candidates.append(candidate)
    if candidates:
        return candidates[0]
    raise VLMParseError(
        f"VLM returned non-JSON text: {original_text!r}",
        raw_text=original_text,
    )


def recover_allowed_token(raw_text: str, allowed_tokens: Sequence[str]) -> str:
    tokens = tuple(allowed_tokens)
    if not tokens:
        return ""
    token_pattern = "|".join(re.escape(token) for token in tokens)
    explicit = re.search(
        rf'"(?:token|answer|direction|status|decision)"\s*:\s*"?\s*({token_pattern})\b',
        raw_text,
    )
    if explicit:
        return explicit.group(1)
    choice_patterns = (
        rf'\b(?:therefore|thus|so|final(?:ly)?|choose|chosen|select|selected|use|'
        rf'decision|answer)\b[^.\n]*?\b({token_pattern})\b',
        rf'\b({token_pattern})\b[^.\n]*?\b(?:best|correct|required|appropriate)\b',
    )
    for pattern in choice_patterns:
        matches = list(re.finditer(pattern, raw_text, flags=re.IGNORECASE))
        if matches:
            return matches[-1].group(1)
    sentences = [part.strip() for part in re.split(r"[.\n]+", raw_text) if part.strip()]
    if sentences:
        final_tokens = _tokens_in_text(sentences[-1], tokens)
        if len(final_tokens) == 1:
            return final_tokens[0]
    found = _tokens_in_text(raw_text, tokens)
    unique = list(dict.fromkeys(found))
    if len(unique) == 1:
        return unique[0]
    return ""


def _tokens_in_text(raw_text: str, allowed_tokens: Sequence[str]) -> list[str]:
    result: list[str] = []
    for token in allowed_tokens:
        if re.search(rf"\b{re.escape(token)}\b", raw_text):
            result.append(token)
    return result


def _message_text(message: dict) -> str:
    """Return only the final-answer channel, never reasoning-only output."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _completion_metadata(data: dict[str, Any], message: dict[str, Any]) -> dict[str, Any]:
    """Record reasoning use and token accounting without copying hidden CoT."""
    content = _message_text(message)
    # vLLM 0.24's `deepseek_r1` parser exposes its separate channel as
    # `reasoning`; OpenAI-compatible servers commonly use `reasoning_content`.
    # Count either field for audit, but never copy hidden reasoning into metadata.
    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning:
        reasoning = message.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = ""
    if not reasoning:
        tagged = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL | re.IGNORECASE)
        reasoning = tagged.group(1) if tagged else ""
    choice = (data.get("choices") or [{}])[0]
    metadata: dict[str, Any] = {
        "finish_reason": choice.get("finish_reason"),
        "reasoning_present": bool(reasoning.strip()),
        "reasoning_chars": len(reasoning),
        "final_content_chars": len(content),
    }
    usage = data.get("usage")
    if isinstance(usage, dict):
        metadata["usage"] = usage
    return metadata


def _strip_reasoning_artifacts(raw_text: str) -> str:
    text = str(raw_text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _reasoning_enabled(chat_template_kwargs: dict) -> bool:
    return bool(
        chat_template_kwargs.get("enable_thinking")
        or chat_template_kwargs.get("thinking")
    )

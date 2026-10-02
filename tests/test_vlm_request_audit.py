from __future__ import annotations

import hashlib

from PIL import Image

from core.vlm.vlm_client import VLMClient, VLMParseError, _message_text, _request_audit


def test_complete_json_audits_finalized_image_payload_sent_to_transport() -> None:
    sent = []
    client = VLMClient(
        base_url="http://localhost:8001/v1",
        model="local-qwen",
        api_key="EMPTY",
        timeout_s=1,
        max_tokens=32,
        temperature=0,
        max_retries=0,
    )

    def post_chat(payload):
        sent.append(payload)
        return {
            "choices": [{"message": {"content": '{"relation":"UNKNOWN"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }

    client._post_chat = post_chat
    first = Image.new("RGB", (24, 16), (10, 20, 30))
    second = Image.new("RGB", (16, 24), (40, 50, 60))
    response = client.complete_json(
        "Compare these two views.", first, second,
        schema={"type": "object", "properties": {"relation": {"type": "string"}}},
        agentview_label=None, wrist_label=None,
    )

    assert len(sent) == 1
    audit = response.payload["request_audit"]
    assert audit == _request_audit(sent[0])
    assert audit["image_count"] == 2
    assert [item["size"] for item in audit["images"]] == [[24, 16], [16, 24]]
    assert audit["prompt_sha256"] == hashlib.sha256(
        b"Compare these two views."
    ).hexdigest()
    assert sent[0]["response_format"]["type"] == "json_schema"
    assert sent[0]["response_format"]["json_schema"]["schema"] == {
        "type": "object", "properties": {"relation": {"type": "string"}}
    }
    assert "guided_json" not in sent[0]
    assert audit["response_format"]["type"] == "json_schema"
    assert audit["response_format"]["schema_sha256"]


def test_reasoning_channel_is_not_used_as_final_decision() -> None:
    assert _message_text({"content": "", "reasoning_content": '{"action":"GRASP"}'}) == ""
    sent = []
    client = VLMClient(
        base_url="http://localhost:8001/v1",
        model="Qwen/Qwen3-VL-8B-Thinking",
        api_key="EMPTY",
        timeout_s=1,
        max_tokens=128,
        temperature=1.0,
        max_retries=0,
    )
    def post_chat(payload):
        sent.append(payload)
        return {"choices": [{
            "finish_reason": "stop",
            "message": {
                "content": '{"action":"STOP"}',
                "reasoning_content": "The view is ambiguous; abstain.",
            },
        }], "usage": {"prompt_tokens": 12, "completion_tokens": 20}}

    client._post_chat = post_chat
    image = Image.new("RGB", (8, 8), (0, 0, 0))
    response = client.complete_json(
        "Choose one action.", image,
        schema={"type": "object", "properties": {"action": {"type": "string"}}},
        chat_template_kwargs={"enable_thinking": True},
        thinking_token_budget=64,
    )
    assert response.payload["json"] == {"action": "STOP"}
    assert response.payload["reasoning_present"] is True
    assert response.payload["reasoning_chars"] > 0
    assert "reasoning_content" not in response.payload
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert sent[0]["max_tokens"] == 128
    assert sent[0]["thinking_token_budget"] == 64
    assert response.payload["request_audit"]["thinking_token_budget"] == 64


def test_atomic_thinking_call_reserves_tokens_for_the_action() -> None:
    sent = []
    client = VLMClient(
        base_url="http://localhost:8001/v1",
        model="Qwen/Qwen3-VL-8B-Thinking",
        api_key="EMPTY",
        timeout_s=1,
        max_tokens=4096,
        temperature=1.0,
        chat_template_kwargs={"enable_thinking": True},
        token_thinking_budget=32,
        max_retries=0,
    )

    def post_chat(payload):
        sent.append(payload)
        return {
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": "MV_UP", "reasoning": "Select the free-space direction."},
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 36},
        }

    client._post_chat = post_chat
    response = client.complete_token(
        "Choose the safe vertical correction.",
        ["MV_UP", "STOP"],
        Image.new("RGB", (8, 8), (0, 0, 0)),
        chat_template_kwargs={"enable_thinking": False, "thinking": False},
    )

    assert response.token == "MV_UP"
    assert len(sent) == 1
    assert sent[0]["thinking_token_budget"] == 32
    assert sent[0]["max_tokens"] == 56
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": True}


def test_vllm_reasoning_field_is_counted_but_not_returned_as_decision() -> None:
    client = VLMClient(
        base_url="http://localhost:8001/v1",
        model="Qwen/Qwen3-VL-8B-Thinking",
        api_key="EMPTY",
        timeout_s=1,
        max_tokens=128,
        temperature=1.0,
        max_retries=0,
    )
    client._post_chat = lambda payload: {
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "content": '{"action":"STOP"}',
                "reasoning": "The image is ambiguous; request fresh evidence.",
            },
        }],
        "usage": {"prompt_tokens": 12, "completion_tokens": 20},
    }
    response = client.complete_json(
        "Choose one action.",
        Image.new("RGB", (8, 8), (0, 0, 0)),
        schema={"type": "object", "properties": {"action": {"type": "string"}}},
    )
    assert response.payload["json"] == {"action": "STOP"}
    assert response.payload["reasoning_present"] is True
    assert response.payload["reasoning_chars"] == len(
        "The image is ambiguous; request fresh evidence."
    )
    assert "reasoning" not in response.payload
    assert "The image is ambiguous" not in str(response.payload)


def test_reasoning_only_response_and_truncated_final_answer_fail_closed() -> None:
    client = VLMClient(
        base_url="http://localhost:8001/v1",
        model="Qwen/Qwen3-VL-8B-Thinking",
        api_key="EMPTY",
        timeout_s=1,
        max_tokens=16,
        temperature=1.0,
        max_retries=0,
    )
    image = Image.new("RGB", (8, 8), (0, 0, 0))
    client._post_chat = lambda payload: {
        "choices": [{"finish_reason": "stop", "message": {
            "content": "", "reasoning_content": '{"action":"GRASP"}'
        }}]
    }
    try:
        client.complete_json("Choose.", image, schema={"type": "object"})
    except VLMParseError:
        pass
    else:
        raise AssertionError("reasoning-only output must not become an action")

    client._post_chat = lambda payload: {
        "choices": [{"finish_reason": "length", "message": {
            "content": "", "reasoning_content": "Checking image evidence..."
        }}],
        "usage": {"prompt_tokens": 321, "completion_tokens": 16},
    }
    try:
        client.complete_json("Choose.", image, schema={"type": "object"})
    except VLMParseError as exc:
        assert exc.payload["failure"] == "truncated_final_answer"
        assert exc.payload["finish_reason"] == "length"
        assert exc.payload["reasoning_present"] is True
        assert exc.payload["reasoning_chars"] > 0
        assert exc.payload["usage"]["completion_tokens"] == 16
        assert exc.payload["request_audit"]["image_count"] == 1
        assert exc.payload["latency_s"] >= 0
    else:
        raise AssertionError("truncated final JSON must fail closed")

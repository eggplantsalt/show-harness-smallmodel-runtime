from __future__ import annotations

import numpy as np

from core.vlm.roles import ControllerAgent
from core.vlm.vlm_client import VLMResponse


def test_thinking_instance_resolver_reserves_json_tokens() -> None:
    class _ThinkingClient:
        model = "Qwen/Qwen3-VL-8B-Thinking"
        cot_max_tokens = 4096
        thinking_token_budget = 1024

        def __init__(self) -> None:
            self.kwargs = None
            self.images = None
            self.prompt = None

        def complete_json(self, prompt, image, **kwargs):
            self.kwargs = kwargs
            self.images = (image, kwargs.get("wrist_image"))
            self.prompt = prompt
            return VLMResponse(
                token="",
                raw_text='{"selected":"candidate-0","reason":"Matches the requested target."}',
                payload={
                    "json": {
                        "selected": "candidate-0",
                        "reason": "Matches the requested target.",
                    },
                    "finish_reason": "stop",
                    "reasoning_present": True,
                    "reasoning_chars": 41,
                },
            )

    client = _ThinkingClient()
    agent = ControllerAgent(
        client=client,
        prompt_template="",
        common_context="",
    )
    result = agent.resolve_instance(
        task="move the selected bottle",
        target="bottle",
        candidates=[
            {"bbox_xyxy": [10, 10, 30, 50], "score": 0.81},
            {"bbox_xyxy": [35, 10, 55, 50], "score": 0.80},
        ],
        agentview_image=np.zeros((80, 80, 3), dtype=np.uint8),
        other_view_image=np.ones((80, 80, 3), dtype=np.uint8),
    )

    assert result["selected"] == "candidate-0"
    assert result["completion"]["reasoning_present"] is True
    assert client.kwargs["chat_template_kwargs"] == {"enable_thinking": True}
    assert client.kwargs["max_tokens"] == 2048
    assert client.kwargs["thinking_token_budget"] == 1024
    assert client.images[0].shape[:2] == (80, 160)
    assert client.images[1].shape[:2] == (192, 320)
    assert "top-to-bottom then left-to-right spatial order" in client.prompt
    assert "same-color outlined box" in client.prompt
    # The candidate's actual detector rectangle is visible in the full view;
    # its same-color tile is sent as the second image.
    np.testing.assert_array_equal(client.images[0][10, 10], [255, 96, 48])

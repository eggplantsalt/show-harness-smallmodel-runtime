from __future__ import annotations

import numpy as np
import pytest

from core.vlm.roles import (
    ControllerAgent,
    append_fresh_pregrasp_target_crop,
    append_pregrasp_target_crop,
)
from core.vlm.vlm_client import VLMResponse


@pytest.mark.parametrize(
    (
        "model",
        "expected_thinking",
        "expected_temperature",
        "expected_budget",
        "expected_thinking_budget",
    ),
    [
        ("Qwen/Qwen3-VL-8B-Instruct", False, 0.0, 1024, None),
        ("Qwen/Qwen3-VL-8B-Thinking", True, 1.0, 4096, 1024),
    ],
)
def test_pregrasp_semantic_decision_selectively_enables_thinking(
    model: str,
    expected_thinking: bool,
    expected_temperature: float,
    expected_budget: int,
    expected_thinking_budget: int | None,
) -> None:
    class Client:
        cot_max_tokens = 4096

        def __init__(self) -> None:
            self.model = model
            self.calls = []

        def complete_json(self, prompt, image, **kwargs):
            self.calls.append({"prompt": prompt, "image": image, "kwargs": kwargs})
            payload = {
                "selected": "GRASP",
                "state_hypothesis": "target aligned between fingers",
                "evidence_for": [{
                    "frame_id": 5,
                    "camera": "wrist",
                    "observation": "target is visibly between the open fingers",
                }],
                "evidence_against": [],
                "missing_observation": "none before one bounded close",
                "expected_effect": "fingers close around the target",
                "failure_condition": "target moves out of the grasp region",
                "summary": "fresh dual-view alignment supports a close",
            }
            return VLMResponse(
                token="GRASP", raw_text="{}", payload={
                    "json": payload,
                    "usage": {},
                    "finish_reason": "stop",
                    "reasoning_present": True,
                    "reasoning_chars": 64,
                    "final_content_chars": 512,
                }
            )

    client = Client()
    agent = ControllerAgent(client, "pick and place", "")
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    result = agent.resolve_pregrasp(
        task="pick and place",
        target="target",
        affordance="body",
        allowed_actions=["GRASP", "UNKNOWN"],
        runtime_reason="fresh identity and safe approach pose",
        agentview_image=frame,
        wrist_image=frame,
        allow_visual_grasp_without_spatial=True,
        target_roi_included=True,
        prebuilt_memory_panel=np.zeros((512, 512, 3), dtype=np.uint8),
        current_frame_id=5,
    )

    assert result["selected"] == "GRASP"
    assert result["agent_decision"]["validated"] is True
    assert len(client.calls) == 1
    call = client.calls[0]["kwargs"]
    assert call["chat_template_kwargs"].get("enable_thinking", False) is expected_thinking
    assert call["temperature"] == expected_temperature
    assert call["max_tokens"] == expected_budget
    assert call["thinking_token_budget"] == expected_thinking_budget
    assert call["schema"]["properties"]["evidence_for"]["minItems"] == 1
    assert result["completion_metadata"]["reasoning_present"] is True
    assert result["completion_metadata"]["reasoning_chars"] == 64
    assert "bottom panel row includes an enlarged crop" in client.calls[0]["prompt"]


def test_v21_thinking_decision_can_abstain_without_forced_depth_probe() -> None:
    class Client:
        model = "Qwen/Qwen3-VL-8B-Thinking"
        cot_max_tokens = 4096
        thinking_token_budget = 1024

        def __init__(self) -> None:
            self.prompt = ""

        def complete_json(self, prompt, image, **kwargs):
            self.prompt = prompt
            return VLMResponse(
                token="UNKNOWN", raw_text="{}", payload={
                    "json": {
                        "selected": "UNKNOWN",
                        "state_hypothesis": "target depth relative to fingers is uncertain",
                        "evidence_for": [{
                            "frame_id": 5,
                            "camera": "wrist",
                            "observation": "the visible target edge is partially occluded",
                        }],
                        "evidence_against": [],
                        "missing_observation": "clear view of the target-to-finger depth relation",
                        "expected_effect": "no movement while abstaining",
                        "failure_condition": "a later view still hides the depth relation",
                        "summary": "current evidence does not justify a probe or grasp",
                    },
                    "finish_reason": "stop",
                    "reasoning_present": True,
                    "reasoning_chars": 80,
                    "usage": {},
                },
            )

    client = Client()
    agent = ControllerAgent(client, "pick up the target", "")
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    result = agent.resolve_pregrasp(
        task="pick up the target",
        target="target object",
        affordance="stable graspable body",
        allowed_actions=["PROBE_DEPTH", "UNKNOWN"],
        runtime_reason="spatial provider returned UNKNOWN",
        agentview_image=image,
        wrist_image=image,
        spatial_belief={"health": "UNKNOWN", "relations": ["UNKNOWN"]},
        prebuilt_memory_panel=np.zeros((128, 256, 3), dtype=np.uint8),
        current_frame_id=5,
    )

    assert result["selected"] == "UNKNOWN"
    assert result["agent_decision"]["validated"] is True
    assert result["agent_decision"]["missing_observation"]
    assert "UNKNOWN does not make a probe mandatory" in client.prompt


def test_v21_thinking_can_choose_runtime_compiled_visual_alignment() -> None:
    class Client:
        model = "Qwen/Qwen3-VL-8B-Thinking"
        cot_max_tokens = 4096
        thinking_token_budget = 1024

        def __init__(self) -> None:
            self.prompt = ""

        def complete_json(self, prompt, image, **kwargs):
            self.prompt = prompt
            return VLMResponse(
                token="VISUAL_ALIGN",
                raw_text="{}",
                payload={
                    "json": {
                        "selected": "VISUAL_ALIGN",
                        "state_hypothesis": "target remains laterally offset from the open gripper",
                        "evidence_for": [{
                            "frame_id": 8,
                            "camera": "agentview",
                            "observation": "the target is visibly to the right of the projected gripper point",
                        }],
                        "evidence_against": [],
                        "missing_observation": "a fresh view after one calibrated correction",
                        "expected_effect": "the same-frame target-to-gripper image residual should shrink",
                        "failure_condition": "the next fresh view shows no residual improvement",
                        "summary": "request one runtime-calibrated visual correction, then inspect a new frame",
                    },
                    "finish_reason": "stop",
                    "reasoning_present": True,
                    "reasoning_chars": 96,
                    "usage": {},
                },
            )

    client = Client()
    agent = ControllerAgent(client, "pick up the target", "")
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    result = agent.resolve_pregrasp(
        task="pick up the target",
        target="target object",
        affordance="stable graspable body",
        allowed_actions=["VISUAL_ALIGN", "PROBE_DEPTH", "UNKNOWN"],
        runtime_reason="fresh identity; visual residual exceeds alignment tolerance",
        agentview_image=image,
        wrist_image=image,
        spatial_belief={"health": "UNKNOWN", "relations": ["UNKNOWN"]},
        visual_alignment={
            "valid": True,
            "frame_id": 8,
            "instance_id": "target-1",
            "camera": "agentview",
            "target_minus_eef_px": [60.0, -25.0],
            "calibrated_correction_candidates": {
                "horizontal": "MV_RIGHT",
                "vertical": "MV_BACK",
            },
            "alignment_tolerance_px": 13.5,
        },
        prebuilt_memory_panel=np.zeros((128, 256, 3), dtype=np.uint8),
        current_frame_id=8,
    )

    assert result["selected"] == "VISUAL_ALIGN"
    assert result["agent_decision"]["validated"] is True
    assert "one bounded correction" in client.prompt
    assert '"target_minus_eef_px":[60.0,-25.0]' in client.prompt
    assert "ALLOWED: VISUAL_ALIGN, PROBE_DEPTH, UNKNOWN" in client.prompt


def test_pregrasp_target_crop_is_proportional_and_preserves_full_panel() -> None:
    panel = np.zeros((256, 512, 3), dtype=np.uint8)
    panel[..., 2] = 120
    current = np.zeros((128, 128, 3), dtype=np.uint8)
    current[..., 2] = 80
    current[56:72, 56:72] = (240, 20, 20)

    augmented, metadata = append_pregrasp_target_crop(
        panel, current, (56, 56, 72, 72), "wrist"
    )

    assert metadata is not None
    assert metadata["camera"] == "wrist"
    assert metadata["crop_scale_from_bbox"] == 2.5
    assert augmented.shape == (512, 512, 3)
    np.testing.assert_array_equal(augmented[:256], panel)
    # The 16px detector box occupies roughly 40% of the proportional crop,
    # making its red target area visibly larger while retaining context.
    roi = augmented[256:, :256]
    red_pixels = (roi[..., 0] > 200) & (roi[..., 1] < 60)
    assert int(red_pixels.sum()) > 5_000


def test_pregrasp_crop_rejects_stale_bbox_and_unknown_camera() -> None:
    panel = np.zeros((256, 512, 3), dtype=np.uint8)
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    track = {
        "instance_id": "episode-instance-1",
        "bbox_xyxy": [40, 40, 56, 56],
        "last_confirmed_frame": 9,
        "camera": "wrist",
    }

    stale_panel, stale_meta = append_fresh_pregrasp_target_crop(
        panel, track, 10, wrist_image=image
    )
    bad_camera_panel, bad_camera_meta = append_fresh_pregrasp_target_crop(
        panel, {**track, "last_confirmed_frame": 10, "camera": "overhead"}, 10,
        wrist_image=image,
    )
    fresh_panel, fresh_meta = append_fresh_pregrasp_target_crop(
        panel, {**track, "last_confirmed_frame": 10}, 10, wrist_image=image
    )

    assert stale_meta is None and stale_panel is panel
    assert bad_camera_meta is None and bad_camera_panel is panel
    assert fresh_meta is not None and fresh_meta["camera"] == "wrist"
    assert fresh_panel.shape == (512, 512, 3)

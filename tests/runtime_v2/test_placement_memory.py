from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from core.runtime_v2.memory import VisualMemory
from core.runtime_v2.placement_memory import build_placement_panel
from core.runtime_v2.types import VisualMemoryEntry
from core.vlm.roles import ControllerAgent
from core.vlm.vlm_client import VLMResponse


def _entry(frame: int, *, epoch: int = 2) -> dict:
    return {
        "instance_id": "target-1",
        "grasp_epoch": epoch,
        "frame_id": frame,
        "agentview_ref": f"images/raw_agentview/{frame:04d}.png",
        "wrist_ref": f"images/raw_wrist/{frame:04d}.png",
        "executed_action": "MV_DOWN",
    }


def test_pregrasp_double_reflection_labels_roi_as_same_frozen_frame() -> None:
    class Client:
        model = "Qwen/Qwen3-VL-8B-Thinking"
        cot_max_tokens = 4096
        thinking_token_budget = 1024

        def __init__(self):
            self.calls = []

        def complete_json(self, prompt, image, **kwargs):
            self.calls.append((prompt, np.asarray(image).copy(), kwargs))
            if len(self.calls) == 1:
                payload = {
                    "current_observation": {
                        "frame_id": 8,
                        "camera": "agentview",
                        "observation": "target position is unresolved",
                    },
                    "visible_state": "target position is unresolved",
                    "target_gripper_relation": "UNKNOWN",
                    "action_effect": "UNKNOWN",
                    "uncertainties": ["small target in full view"],
                    "evidence": [{
                        "frame_id": 8,
                        "camera": "agentview",
                        "observation": "the target crop comes from this frame",
                    }],
                }
            else:
                payload = {
                    "selected": "UNKNOWN",
                    "state_hypothesis": "current relation remains unclear",
                    "evidence_for": [{
                        "frame_id": 8,
                        "camera": "agentview",
                        "observation": "the current ROI and full view do not show finger contact",
                    }],
                    "evidence_against": [],
                    "missing_observation": "a clearer Wrist view",
                    "expected_effect": "no action while abstaining",
                    "failure_condition": "the target remains hidden",
                    "summary": "do not infer a grasp from the crop alone",
                }
            return VLMResponse(
                token="",
                raw_text="{}",
                payload={
                    "json": payload,
                    "request_audit": {"images": [{"sha256": "same-frozen-panel"}]},
                },
            )

    client = Client()
    panel = np.zeros((1024, 512, 3), dtype=np.uint8)
    result = ControllerAgent(client, "pick", "").resolve_pregrasp(
        task="pick the target",
        target="target",
        affordance="body",
        allowed_actions=["REOBSERVE", "UNKNOWN"],
        runtime_reason="target is occluded at grasp entry",
        agentview_image=panel[:256, :256],
        wrist_image=panel[:256, 256:],
        visual_memory_bundle=[_entry(8)],
        prebuilt_memory_panel=panel,
        current_frame_id=8,
        reflection_mode="double",
        reflection_trigger="grasp_entry",
        target_roi_included=True,
    )

    assert result["selected"] == "UNKNOWN"
    assert len(client.calls) == 2
    np.testing.assert_array_equal(client.calls[0][1], client.calls[1][1])
    assert "ROI strip is below it" in client.calls[0][0]
    assert "ROI is the same current image, not a new time step" in client.calls[1][0]
    assert result["reflection"]["same_image_payload"] is True


def test_raw_placement_panel_resolves_ordered_same_epoch_frames(tmp_path) -> None:
    for frame in (4, 7, 8):
        for camera in ("agentview", "wrist"):
            path = tmp_path / f"images/raw_{camera}/{frame:04d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (256, 256), (frame, 20, 30)).save(path)
    bundle = [_entry(frame) for frame in (4, 7, 8)]
    panel, meta = build_placement_panel(
        tmp_path, bundle, instance_id="target-1", grasp_epoch=2, current_frame=8
    )
    assert panel.shape == (768, 512, 3)
    assert meta["frame_ids"] == [4, 7, 8]
    assert len(meta["refs"]) == 6
    with pytest.raises(ValueError, match="epoch"):
        build_placement_panel(tmp_path, [bundle[0], _entry(8, epoch=3)], instance_id="target-1", grasp_epoch=2, current_frame=8)
    with pytest.raises(ValueError, match="non-raw"):
        build_placement_panel(tmp_path, [{**_entry(8), "agentview_ref": "images/provider_overlay/0008.png"}], instance_id="target-1", grasp_epoch=2, current_frame=8)


def test_placement_panel_rejects_memory_from_another_episode(tmp_path) -> None:
    for camera in ("agentview", "wrist"):
        path = tmp_path / f"images/raw_{camera}/0008.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (256, 256)).save(path)
    bundle = [{**_entry(8), "episode_id": "episode-A"}]
    with pytest.raises(ValueError, match="different episode"):
        build_placement_panel(
            tmp_path, bundle, instance_id="target-1", grasp_epoch=2,
            current_frame=8, episode_id="episode-B",
        )


def test_visual_memory_resets_at_episode_boundary_even_when_frame_ids_repeat() -> None:
    memory = VisualMemory(max_entries_per_epoch=3)
    previous = memory.append(VisualMemoryEntry("target-1", 2, 8))
    assert previous.episode_id == memory.episode_id

    memory.reset(episode_id="episode-B")
    assert memory.placement_bundle(instance_id="target-1", grasp_epoch=2) == []
    with pytest.raises(ValueError, match="different episode"):
        memory.append(previous)

    current = memory.append(VisualMemoryEntry("target-1", 2, 8))
    assert current.episode_id == "episode-B"
    assert memory.placement_bundle(instance_id="target-1", grasp_epoch=2)[-1]["episode_id"] == "episode-B"


def test_keyframes_survive_bounded_recent_history() -> None:
    memory = VisualMemory(max_entries_per_epoch=3)
    for frame in range(10):
        memory.append(VisualMemoryEntry("target-1", 2, frame, agentview_ref=f"images/raw_agentview/{frame:04d}.png"))
        if frame in (1, 5):
            memory.mark(instance_id="target-1", grasp_epoch=2, frame_id=frame, tag="SEATING_ENTRY")
    # Keep the most recent event frame plus the current action's pre-frame.
    # A single old keyframe must not displace frame 8, which immediately
    # precedes the frozen frame 9 observation.
    assert [item["frame_id"] for item in memory.placement_bundle(instance_id="target-1", grasp_epoch=2)] == [5, 8, 9]


def test_memory_does_not_mix_a_second_object_or_grasp_epoch() -> None:
    memory = VisualMemory(max_entries_per_epoch=3)
    memory.append(VisualMemoryEntry("bottle", 1, 2, agentview_ref="images/raw_agentview/0002.png"))
    memory.mark(instance_id="bottle", grasp_epoch=1, frame_id=2, tag="SEATING_ENTRY")
    memory.append(VisualMemoryEntry("milk-carton", 1, 3, agentview_ref="images/raw_agentview/0003.png"))
    memory.append(VisualMemoryEntry("bottle", 2, 4, agentview_ref="images/raw_agentview/0004.png"))
    assert [item["frame_id"] for item in memory.placement_bundle(instance_id="milk-carton", grasp_epoch=1)] == [3]
    assert [item["frame_id"] for item in memory.placement_bundle(instance_id="bottle", grasp_epoch=2)] == [4]


def test_double_reflection_uses_same_panel_and_hides_provider_relation_from_first_call() -> None:
    class Client:
        model = "Qwen/Qwen3-VL-8B-Thinking"
        cot_max_tokens = 2048

        def __init__(self):
            self.calls = []

        def complete_json(self, prompt, image, **kwargs):
            self.calls.append((prompt, np.asarray(image).copy(), dict(kwargs)))
            if len(self.calls) == 1:
                payload = {"held_object": "visible", "opening": "visible", "relative_position": "above", "contact_or_support": "UNKNOWN", "action_effect": "descended", "uncertainty": "contact hidden", "evidence_frame_ids": [8]}
            else:
                citation = {"frame_id": 8, "camera": "agentview", "observation": "object remains visibly above the opening"}
                payload = {
                    "relation": "ABOVE_ALIGNED",
                    "reasoning": "still above opening",
                    "next_step": "REOBSERVE",
                    "evidence_for": [citation],
                    "evidence_against": [],
                    "missing_observation": "support response after descent",
                    "expected_effect": "a fresh view clarifies vertical relation",
                    "failure_condition": "the object remains occluded",
                }
            return VLMResponse(token="", raw_text="{}", payload={"json": payload, "latency_s": 0.01})

    client = Client()
    agent = ControllerAgent(client, "", "")
    panel = np.zeros((256, 512, 3), dtype=np.uint8)
    current_memory = {
        **_entry(8),
        "observed_effect": {
            "visual_point_track": {
                "health": "VALID",
                "median_displacement_px": [1.5, -0.5],
            }
        },
    }
    result = agent.verify_place(
        task="place object", target="receptacle", affordance="inside",
        agentview_image=panel[:, :256], wrist_image=panel[:, 256:],
        placement_v22=True, memory_panel=panel, memory_bundle=[current_memory],
        reflection_mode="double", placement_evidence={"relation": "RIM_CONTACT", "frame_id": 8},
    )
    assert result["relation"] == "ABOVE_ALIGNED"
    assert len(client.calls) == 2
    assert np.array_equal(client.calls[0][1], client.calls[1][1])
    assert client.calls[0][2]["chat_template_kwargs"] == {"enable_thinking": True}
    assert client.calls[0][2]["max_tokens"] == 1024
    assert client.calls[1][2]["chat_template_kwargs"] == {"enable_thinking": True}
    assert client.calls[1][2]["max_tokens"] == 2048
    assert "median_displacement_px" in client.calls[0][0]
    assert "RIM_CONTACT" not in client.calls[0][0]
    assert "RIM_CONTACT" not in client.calls[1][0].split("GEOMETRIC HARNESS EVIDENCE")[1].split("EXECUTED ACTION TIMELINE")[0]


def test_pregrasp_double_reflection_reuses_frozen_panel_and_cites_current_frame() -> None:
    class Client:
        model = "Qwen/Qwen3-VL-8B-Thinking"
        cot_max_tokens = 4096
        thinking_token_budget = 1024

        def __init__(self):
            self.calls = []

        def complete_json(self, prompt, image, **kwargs):
            self.calls.append((prompt, np.asarray(image).copy(), dict(kwargs)))
            if len(self.calls) == 1:
                payload = {
                    "current_observation": {
                        "frame_id": 8, "camera": "wrist",
                        "observation": "The target body remains between the open fingers in the frozen frame.",
                    },
                    "visible_state": "Target is visible near the gripper.",
                    "target_gripper_relation": "BETWEEN",
                    "action_effect": "The prior approach reduced separation.",
                    "uncertainties": ["depth is not fully clear"],
                    "evidence": [{
                        "frame_id": 8, "camera": "wrist",
                        "observation": "Target body is inside the open-finger span.",
                    }],
                }
            else:
                payload = {
                    "selected": "GRASP",
                    "state_hypothesis": "target appears between the fingers",
                    "evidence_for": [{
                        "frame_id": 8, "camera": "wrist",
                        "observation": "target body is between the open fingers",
                    }],
                    "evidence_against": [],
                    "missing_observation": "post-close hold verification",
                    "expected_effect": "fingers close around the target",
                    "failure_condition": "target moves independently during verification",
                    "summary": "dual-view and prior action effect support a bounded grasp",
                }
            return VLMResponse(
                token="", raw_text="{}",
                payload={
                    "json": payload,
                    "latency_s": 0.01,
                    "usage": {"completion_tokens": 20},
                    "finish_reason": "stop",
                    "reasoning_present": True,
                    "reasoning_chars": 100,
                    "request_audit": {"images": [{"sha256": "frozen-panel", "size": [512, 256]}]},
                },
            )

    client = Client()
    agent = ControllerAgent(client, "", "")
    panel = np.zeros((256, 512, 3), dtype=np.uint8)
    result = agent.resolve_pregrasp(
        task="pick up the object", target="object", affordance="visible stable body",
        allowed_actions=["GRASP", "CORRECT_DEPTH", "UNKNOWN"],
        runtime_reason="one bounded semantic choice; fresh spatial evidence is available",
        agentview_image=panel[:, :256], wrist_image=panel[:, 256:],
        spatial_belief={"health": "VALID", "relations": ["INSIDE_ENVELOPE"]},
        visual_memory_bundle=[_entry(8)], prebuilt_memory_panel=panel,
        current_frame_id=8, reflection_mode="double", reflection_trigger="grasp_entry",
    )

    assert result["selected"] == "GRASP"
    assert len(client.calls) == 2
    assert np.array_equal(client.calls[0][1], client.calls[1][1])
    assert client.calls[0][2]["thinking_token_budget"] == 512
    assert client.calls[0][2]["max_tokens"] == 1024
    first_schema = client.calls[0][2]["schema"]
    assert "current_observation" in first_schema["required"]
    assert first_schema["properties"]["current_observation"]["properties"]["frame_id"]["enum"] == [8]
    assert client.calls[1][2]["thinking_token_budget"] == 1024
    assert client.calls[1][2]["max_tokens"] == 4096
    assert result["reflection"]["same_image_payload"] is True
    assert result["reflection"]["scene_description"]["target_gripper_relation"] == "BETWEEN"


def test_pregrasp_double_reflection_requires_explicit_current_frame_citation() -> None:
    class Client:
        def __init__(self):
            self.calls = []

        def complete_json(self, prompt, image, **kwargs):
            self.calls.append((prompt, np.asarray(image).copy(), dict(kwargs)))
            return VLMResponse(
                token="", raw_text="{}",
                payload={"json": {
                    "visible_state": "The target is near the gripper.",
                    "target_gripper_relation": "UNKNOWN",
                    "action_effect": "UNKNOWN",
                    "uncertainties": ["current occlusion"],
                    "evidence": [{
                        "frame_id": 7, "camera": "agentview",
                        "observation": "The target was visible before the current frame.",
                    }],
                }},
            )

    client = Client()
    agent = ControllerAgent(client, "", "")
    panel = np.zeros((256, 512, 3), dtype=np.uint8)
    result = agent.resolve_pregrasp(
        task="pick up the object", target="object", affordance="body",
        allowed_actions=["MV_UP", "UNKNOWN"], runtime_reason="inspect occlusion",
        agentview_image=panel[:, :256], wrist_image=panel[:, 256:],
        spatial_belief={"health": "UNKNOWN"}, visual_memory_bundle=[_entry(7), _entry(8)],
        prebuilt_memory_panel=panel, current_frame_id=8,
        reflection_mode="double", reflection_trigger="grasp_entry",
    )

    assert result["selected"] == "UNKNOWN"
    assert "current-frame evidence" in result["reason"]
    assert result["reflection"]["valid_evidence"] is False
    assert len(client.calls) == 1


def test_pregrasp_agent_can_request_runtime_compiled_reobservation() -> None:
    class Client:
        model = "Qwen/Qwen3-VL-8B-Thinking"
        cot_max_tokens = 4096
        thinking_token_budget = 1024

        def __init__(self):
            self.prompt = ""
            self.schema = {}

        def complete_json(self, prompt, image, **kwargs):
            self.prompt = prompt
            self.schema = kwargs["schema"]
            return VLMResponse(
                token="", raw_text="{}",
                payload={"json": {
                    "selected": "REOBSERVE",
                    "state_hypothesis": "The current Wrist view does not resolve the target relation.",
                    "evidence_for": [{
                        "frame_id": 8, "camera": "agentview",
                        "observation": "The target is near the gripper but the current Wrist view is occluded.",
                    }],
                    "evidence_against": [],
                    "missing_observation": "A fresh dual view after a small clearance move.",
                    "expected_effect": "The new image should expose the target-to-gripper relation.",
                    "failure_condition": "The target remains occluded after the view move.",
                    "summary": "Request a new observation before considering grasp.",
                }},
            )

    client = Client()
    agent = ControllerAgent(client, "", "")
    panel = np.zeros((256, 512, 3), dtype=np.uint8)
    result = agent.resolve_pregrasp(
        task="pick up the object", target="object", affordance="body",
        allowed_actions=["REOBSERVE", "UNKNOWN"],
        runtime_reason="target is occluded at grasp entry",
        agentview_image=panel[:, :256], wrist_image=panel[:, 256:],
        spatial_belief={"health": "UNKNOWN"}, visual_memory_bundle=[_entry(8)],
        prebuilt_memory_panel=panel, current_frame_id=8,
    )

    assert result["selected"] == "REOBSERVE"
    assert result["agent_decision"]["validated"] is True
    assert client.schema["properties"]["selected"]["enum"] == ["REOBSERVE", "UNKNOWN"]
    assert "one bounded upward clearance move" in client.prompt


def test_double_reflection_abstains_if_scene_description_cites_missing_frame() -> None:
    class Client:
        def complete_json(self, prompt, image, **kwargs):
            return VLMResponse(token="", raw_text="{}", payload={"json": {
                "held_object": "UNKNOWN", "opening": "UNKNOWN", "relative_position": "UNKNOWN",
                "contact_or_support": "UNKNOWN", "action_effect": "UNKNOWN",
                "uncertainty": "occluded", "evidence_frame_ids": [999],
            }})

    agent = ControllerAgent(Client(), "", "")
    panel = np.zeros((256, 512, 3), dtype=np.uint8)
    result = agent.verify_place(
        task="place", target="bowl", affordance="opening", agentview_image=panel[:, :256],
        wrist_image=panel[:, 256:], placement_v22=True, memory_panel=panel,
        memory_bundle=[_entry(8)], reflection_mode="double",
    )
    assert result["relation"] == "UNKNOWN"
    assert "unavailable frames" in result["error"]

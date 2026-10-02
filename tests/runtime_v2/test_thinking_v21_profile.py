from pathlib import Path

from core.config import load_yaml
from core.runtime_v2 import VerifiedCapabilityRuntime


def test_thinking_profile_keeps_v21_harness_and_registers_moge_v2() -> None:
    repo = Path(__file__).resolve().parents[2]
    config = load_yaml(
        repo / "configs" / "robot_libero_clean_qwen3vl_thinking_runtime_v21.yaml"
    )

    vlm = config["vlm_backends"]["qwen3vl_local"]
    runtime_cfg = config["runtime_v2"]
    spatial_cfg = runtime_cfg["spatial_tools"]
    tracking_cfg = runtime_cfg["visual_tracking"]
    runtime = VerifiedCapabilityRuntime.from_config(config)

    assert vlm["model"] == "Qwen/Qwen3-VL-8B-Thinking"
    assert vlm["chat_template_kwargs"]["enable_thinking"] is True
    assert runtime.placement_v22_enabled is False
    assert runtime.semantic_pregrasp_enabled is True
    assert runtime.require_spatial_ready_for_grasp is True
    assert runtime.temporal_identity_lock_on_detector_ties is True
    assert runtime.metric_approach_uses_hover_budget is True
    assert runtime.semantic_evidence_max_age_frames == 1
    assert runtime.pregrasp_reflection_mode == "double"
    assert spatial_cfg["device"] == "cuda:1"
    assert spatial_cfg["service_python"].endswith("moge-env/bin/python")
    assert spatial_cfg["checkpoint"].endswith("moge-2-vitl/model.pt")
    assert tracking_cfg["enabled"] is True
    assert tracking_cfg["device"] == "cuda:1"
    assert tracking_cfg["checkpoint"].endswith("cotracker3/scaled_online.pth")
    assert runtime.visual_point_tracker is not None
    assert runtime.visual_point_tracker.device == "cuda:1"
    assert config["capabilities"]["target_reference_height_m"] is None
    assert set(runtime.spatial_bus.providers) == {"moge2"}
    assert runtime.spatial_bus.providers["moge2"].model_version == "v2"

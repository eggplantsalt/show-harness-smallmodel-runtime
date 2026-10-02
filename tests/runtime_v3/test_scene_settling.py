from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from core.capabilities.camera_geometry import CameraCalibration
from core.runtime_v3.canonical_image import CanonicalImageAdapter
from core.runtime_v3.scene_settling import (
    SceneReadyEvidence,
    action_excess_motion,
    compare_oracle_and_sam_pixel_motion,
    displacement,
    project_world_motion_to_canonical_pixels,
    require_matched_duration,
    scene_readiness_sample,
    target_motion_curve,
    validate_no_action_commands,
)


ROOT = Path(__file__).resolve().parents[2]


def test_no_action_trial_contract_accepts_only_zero_translation_holds():
    validate_no_action_commands([
        {"kind": "HOLD", "translation_command_count": 0},
        {"kind": "HOLD", "translation_command_count": 0},
    ])
    with pytest.raises(ValueError, match="zero-translation HOLD"):
        validate_no_action_commands([{"kind": "MOVE", "translation_command_count": 1}])


def test_action_and_no_action_duration_must_match_exactly():
    require_matched_duration(4, 4)
    with pytest.raises(ValueError, match="must match exactly"):
        require_matched_duration(4, 3)


def test_oracle_displacement_keeps_world_xyz_and_norm_separate():
    result = displacement([1.0, 2.0, 3.0], [1.0, 2.0, 2.0])
    assert result["delta_xyz_m"] == [0.0, 0.0, -1.0]
    assert result["norm_m"] == 1.0


def test_action_excess_subtracts_matched_no_action_vector():
    result = action_excess_motion([0.001, 0.002, -0.060], [0.0, 0.0, -0.058])
    np.testing.assert_allclose(result["delta_xyz_m"], [0.001, 0.002, -0.002])
    assert result["norm_m"] == pytest.approx(np.sqrt(9e-6))
    assert result["norm_difference_m"] == pytest.approx(np.linalg.norm([0.001, 0.002, -0.060]) - 0.058)


def test_target_motion_curve_sorts_samples_and_keeps_target_eef_curves_distinct():
    curve = target_motion_curve([
        {"tick": 1, "environment_step": 1, "simulation_time_s": 0.05,
         "target_world_position_m": [0.0, 0.0, 0.4], "eef_world_position_m": [0.1, 0.0, 0.2]},
        {"tick": 0, "environment_step": 0, "simulation_time_s": 0.0,
         "target_world_position_m": [0.0, 0.0, 0.5], "eef_world_position_m": [0.0, 0.0, 0.2]},
    ])
    assert [point["tick"] for point in curve] == [0, 1]
    np.testing.assert_allclose(curve[1]["target_delta_from_previous_m"], [0.0, 0.0, -0.1])
    np.testing.assert_allclose(curve[1]["eef_delta_from_previous_m"], [0.1, 0.0, 0.0])


def test_world_motion_projection_uses_camera_and_canonical_vertical_flip():
    camera = CameraCalibration(
        name="agentview", width=512, height=512, fovy_deg=60.0,
        position_world=np.zeros(3), camera_to_world=np.eye(3),
    )
    result = project_world_motion_to_canonical_pixels(
        [0.0, 0.0, -2.0], [0.1, 0.0, -2.0], camera, CanonicalImageAdapter(),
    )
    assert result["available"] is True
    assert result["canonical_orientation"] == "vertical_flip"
    assert result["oracle_pixel_shift"][0] == pytest.approx(22.17, abs=0.02)
    assert result["oracle_pixel_shift"][1] == pytest.approx(0.0, abs=0.01)


def test_sam_oracle_pixel_residual_is_sam_shift_minus_oracle_shift():
    result = compare_oracle_and_sam_pixel_motion([3.0, 4.0], [10.0, 20.0], [14.0, 26.0])
    assert result["sam_centroid_shift"] == [4.0, 6.0]
    assert result["residual_sam_minus_oracle_px"] == [1.0, 2.0]
    assert result["residual_norm_px"] == pytest.approx(np.sqrt(5.0))


def test_scene_readiness_visual_sample_has_no_oracle_input_or_field():
    visual = scene_readiness_sample(
        target_identity_status="SAME_TARGET", centroid_px=[12, 20],
        bbox_xyxy=[2, 4, 30, 45], mask_area_px=234,
    )
    assert visual["evidence_source"] == "associated_sam_visual_observation"
    assert not any("oracle" in key.casefold() for key in visual)
    with pytest.raises(TypeError):
        scene_readiness_sample(
            target_identity_status="SAME_TARGET", centroid_px=[12, 20],
            bbox_xyxy=[2, 4, 30, 45], mask_area_px=234,
            oracle_target_world_position_m=[0, 0, 0],
        )


def test_scene_ready_requires_three_consecutive_stable_same_target_observations():
    gate = SceneReadyEvidence()
    sample = {"target_identity_status": "SAME_TARGET", "centroid_px": [10, 20],
              "bbox_xyxy": [2, 4, 30, 45], "mask_area_px": 234}
    assert gate.update(**sample) is False
    assert gate.update(**sample) is False
    assert gate.update(**sample) is True
    assert gate.to_record()["evidence_source"] == "associated_sam_visual_observation"
    assert gate.to_record()["oracle_used"] is False


def test_unstable_visual_interval_restarts_scene_ready_initialization_window():
    gate = SceneReadyEvidence()
    sample = {"target_identity_status": "SAME_TARGET", "centroid_px": [10, 20],
              "bbox_xyxy": [2, 4, 30, 45], "mask_area_px": 234}
    assert gate.update(**sample) is False
    assert gate.update(**sample) is False
    changed = {**sample, "centroid_px": [40, 50], "bbox_xyxy": [32, 34, 60, 75]}
    assert gate.update(**changed) is False
    assert gate.to_record()["stable_observation_count"] == 1


def test_scene_ready_stability_thresholds_are_recorded_with_measured_source():
    record = SceneReadyEvidence().to_record()
    assert record["max_centroid_shift_px"] == 0.02
    assert record["max_bbox_edge_shift_px"] == 0.0
    assert record["max_mask_area_change_px"] == 1
    assert "ticks 10/15/20" in record["threshold_source"]


def test_scene_initialization_helper_only_accepts_visual_gate_and_executes_hold_path():
    from core.runtime_v3 import scene_initialization

    params = inspect.signature(scene_initialization.run_scene_ready_holds).parameters
    assert "oracle_target_world_position_m" not in params
    source = inspect.getsource(scene_initialization.run_scene_ready_holds)
    assert "token=None" in source
    assert '"kind": "HOLD"' in source
    assert "run_v3_tick(" in source


def test_runtime_scene_diagnostic_module_does_not_import_or_mutate_belief_state():
    source = (ROOT / "core/runtime_v3/scene_settling.py").read_text(encoding="utf-8")
    assert "from core.runtime_v3.state import" not in source
    assert "from core.runtime_v3.observer import" not in source
    assert "oracle_target_world_position_m" not in source


def test_qwen_is_not_constructed_and_scene_diagnostic_has_zero_qwen_actions():
    source = (ROOT / "scripts/runtime_v3_scene_settling.py").read_text(encoding="utf-8")
    assert "Qwen" not in source
    assert '"qwen_action_count": 0' in source
    assert "DeterministicSelector(\"ALIGN_TO_TARGET_SMALL\")" in source


def test_runner_still_routes_physical_micro_motion_through_executor():
    from core.runtime_v3 import runner

    source = inspect.getsource(runner.RuntimeV3Runner.run_episode)
    assert "self.executor.execute(" in source
    assert "execute_approved_micro_tick" not in source
    assert "self.observer" not in source.split("self.executor.execute(", 1)[1].split("except Exception", 1)[0]


def test_no_action_trial_uses_runtime_hold_path_and_never_requests_a_move_token():
    source = (ROOT / "scripts/runtime_v3_scene_settling.py").read_text(encoding="utf-8")
    no_action = source.split("def _run_no_action_trial(", 1)[1].split("def _run_settling_curve(", 1)[0]
    assert "token=None" in no_action
    assert '"kind": "HOLD"' in no_action
    assert '"translation_command_count": 0' in no_action
    assert "validate_no_action_commands(commands)" in no_action

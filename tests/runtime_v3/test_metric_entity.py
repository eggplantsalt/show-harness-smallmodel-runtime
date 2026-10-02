from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import inspect
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from core.capabilities.camera_geometry import CameraCalibration
from core.runtime_v3.adapters.libero_env import LiberoEnvironmentAdapter
from core.runtime_v3.arbiter import Arbiter, ApprovedAction
from core.runtime_v3.depth import (
    DepthProvider,
    MetricDepthEstimate,
    MoGeMetricDepthProvider,
    metric_entity_reference_from_estimate,
)
from core.runtime_v3.executor import Executor
from core.runtime_v3.metric_entity import MetricEntityReference, metric_proximity_distance_m
from core.runtime_v3.observer import RobotObservation
from core.runtime_v3.options import PrimitiveCommand
from core.runtime_v3.state import BeliefState, ObjectRelativeState, StateBuilder
from core.sim.libero_task import LiberoTaskHandle
from core.runtime_v3.canonical_image import CanonicalImageAdapter


def _depth_buffer_for_z(z_m: float, near_m: float, far_m: float) -> float:
    return (1.0 - near_m / z_m) / (1.0 - near_m / far_m)


def _calibration(width=5, height=5, *, position=(0.0, 0.0, 0.0), fovy=90.0):
    return CameraCalibration(
        name="agentview", width=width, height=height, fovy_deg=fovy,
        position_world=np.asarray(position, dtype=float), camera_to_world=np.eye(3),
    )


def _metric_reference(point=(0.0, 0.0, -1.0), *, entity="entity"):
    return MetricEntityReference(
        entity_key=entity, camera="agentview", coordinate_frame="world",
        reference_world_m=point, valid_depth_count=4, mask_pixel_count=4,
        valid_depth_ratio=1.0, depth_median_m=1.0, depth_spread_m=0.0,
        depth_source="monocular_metric", source_frame_id="frame-1", valid=True,
    )


def test_formal_environment_adapter_cannot_enable_simulator_depth(monkeypatch):
    captured = {}
    env = SimpleNamespace()
    handle = LiberoTaskHandle(env, "LIBERO_OBJECT", 2, "task", "move an entity",
                              "task.bddl", np.zeros((1, 7)), 0)

    def fake_make_libero_task(**kwargs):
        captured.update(kwargs)
        return handle

    monkeypatch.setattr("core.runtime_v3.adapters.libero_env.make_libero_task", fake_make_libero_task)
    adapter = LiberoEnvironmentAdapter.create(
        camera_height=512, camera_width=512,
    )
    assert "diagnostic_camera_depths" not in captured
    assert captured["camera_height"] == captured["camera_width"] == 512
    assert adapter.env is env
    assert not hasattr(adapter, "check_success")


def test_libero_source_depth_render_is_only_enabled_by_diagnostic_flag(monkeypatch, tmp_path):
    import core.sim.libero_task as libero_task

    captured = {}

    class Task:
        name = "mock"
        language = "move an entity"
        problem_folder = "mock"
        init_states_file = "mock.pt"
        bddl_file = "mock.bddl"

    class Benchmark:
        def get_task(self, _task_id):
            return Task()

    class RenderEnvironment:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._showharness_last_obs = {}

        def seed(self, _seed):
            pass

        def reset(self):
            pass

        def set_init_state(self, _state):
            return {"agentview_depth": np.ones((512, 512, 1), dtype=np.float32)}

    parent = ModuleType("libero")
    parent.__path__ = []
    nested = ModuleType("libero.libero")
    nested.__path__ = []
    benchmark_module = ModuleType("libero.libero.benchmark")
    benchmark_module.get_benchmark = lambda _name: (lambda: Benchmark())
    env_module = ModuleType("libero.libero.envs")
    env_module.OffScreenRenderEnv = RenderEnvironment
    monkeypatch.setattr(libero_task, "ensure_libero_path", lambda: tmp_path)
    monkeypatch.setattr(libero_task, "_load_init_states", lambda _path: np.zeros((1, 7)))
    monkeypatch.setitem(sys.modules, "libero", parent)
    monkeypatch.setitem(sys.modules, "libero.libero", nested)
    monkeypatch.setitem(sys.modules, "libero.libero.benchmark", benchmark_module)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", env_module)

    handle = libero_task.make_libero_task(
        suite_name="LIBERO_OBJECT", task_id=2, camera_height=512,
        camera_width=512, diagnostic_camera_depths=True, settle_steps=0,
    )
    assert captured["camera_depths"] is True
    assert captured["camera_heights"] == captured["camera_widths"] == 512
    assert handle.env._showharness_last_obs["agentview_depth"].shape == (512, 512, 1)


def test_raw_simulator_depth_stays_out_of_robot_observation():
    from core.runtime_v3.adapters.libero_observation import LiberoObservationAdapter

    raw_depth = np.arange(16, dtype=np.float32).reshape(4, 4, 1) / 20 + 0.5
    raw = {
        "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.ones((4, 4, 3), dtype=np.uint8),
        "agentview_depth": raw_depth,
        "robot0_eye_in_hand_depth": raw_depth + 0.01,
        "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.asarray([0.02, -0.02]),
    }

    class Environment:
        step_count = 1

        def get_observation(self):
            return raw

    adapter = LiberoObservationAdapter()
    observation = adapter.observe(Environment())
    assert observation.images["agentview"].shape[:2] == raw_depth.shape[:2]
    assert observation.images["wrist"].shape[:2] == raw["robot0_eye_in_hand_depth"].shape[:2]
    assert not hasattr(adapter.last_raw, "diagnostic_agentview_depth")
    assert not hasattr(adapter.last_raw, "diagnostic_wrist_depth")
    assert not any(key.casefold().endswith("_depth") for key in adapter.last_raw.raw_keys)
    assert not hasattr(observation, "depths")
    assert "agentview_depth" not in observation.evidence


def test_deployable_provider_receives_only_rgb_and_preserves_source_resolution():
    seen = []

    def infer_rgb(rgb):
        seen.append(np.asarray(rgb).copy())
        depth = np.full(rgb.shape[:2], 2.0, dtype=np.float32)
        depth[0, 0] = np.nan
        return {"depth": depth, "mask": np.ones(rgb.shape[:2], dtype=bool)}

    provider: DepthProvider = MoGeMetricDepthProvider(inference=infer_rgb)
    rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    estimate = provider.estimate(rgb)
    assert len(seen) == 1
    assert np.array_equal(seen[0], rgb)
    assert estimate.source == "monocular_metric"
    assert estimate.depth_m.shape == rgb.shape[:2]
    assert np.isnan(estimate.depth_m[0, 0])
    assert not estimate.valid_mask[0, 0]
    assert list(inspect.signature(MoGeMetricDepthProvider.estimate).parameters) == ["self", "rgb"]


def test_formal_depth_estimate_rejects_simulator_gt_source():
    with pytest.raises(ValueError, match="formal depth source"):
        MetricDepthEstimate(np.ones((2, 2)), source="simulator_gt")
    with pytest.raises(ValueError, match="privileged simulator depth"):
        MetricEntityReference(
            entity_key="entity", camera="agentview", coordinate_frame="world",
            reference_world_m=(0.0, 0.0, -1.0), valid_depth_count=1, mask_pixel_count=1,
            valid_depth_ratio=1.0, depth_median_m=1.0, depth_spread_m=0.0,
            depth_source="simulator_gt", source_frame_id="f", valid=True,
        )


def test_mask_conditioned_metric_depth_robustly_unprojects_to_world():
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    depth[2, 2] = 100.0  # One outlier inside the SAM mask is rejected.
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:4, 1:4] = True
    estimate = MetricDepthEstimate(depth, source="monocular_metric")
    reference = metric_entity_reference_from_estimate(
        entity_key="button", camera="agentview", source_frame_id="frame-4",
        target_mask=mask, estimate=estimate, calibration=_calibration(width=4, height=4),
    )
    assert reference.valid
    assert reference.coordinate_frame == "world"
    assert reference.depth_source == "monocular_metric"
    assert reference.mask_pixel_count == 9
    assert reference.valid_depth_count == 8
    assert reference.valid_depth_ratio == pytest.approx(8 / 9)
    assert reference.depth_median_m == pytest.approx(2.0)
    assert reference.reference_world_m == pytest.approx((0.0, 0.0, -2.0))


def test_metric_reference_mask_resolution_must_match_estimated_depth_and_calibration():
    estimate = MetricDepthEstimate(np.ones((4, 4)), source="monocular_metric")
    reference = metric_entity_reference_from_estimate(
        entity_key="bowl", camera="agentview", source_frame_id="frame",
        target_mask=np.ones((5, 5), dtype=bool), estimate=estimate,
        calibration=_calibration(),
    )
    assert not reference.valid
    assert reference.invalid_reason == "mask_depth_shape_mismatch"


def test_mujoco_diagnostic_depth_conversion_and_orientation_are_isolated():
    from core.runtime_v3.canonical_image import CanonicalImageAdapter
    from scripts.runtime_v3_metric_entity_depth_diagnostic import (
        DiagnosticSimulatorDepthProvider,
        metric_entity_reference_from_rgbd,
        normalized_depth_to_metric,
    )

    near, far = 0.1, 20.0
    raw = np.full((4, 4), _depth_buffer_for_z(2.0, near, far), dtype=np.float32)
    raw[0, 1] = _depth_buffer_for_z(0.7, near, far)
    mask_raw = np.zeros((4, 4), dtype=bool)
    mask_raw[0, 1] = True
    canonical = np.flipud(mask_raw)
    reference, metric = metric_entity_reference_from_rgbd(
        entity_key="apple", camera="agentview", source_frame_id=3,
        target_mask_canonical=canonical, depth_buffer_raw=raw,
        near_m=near, far_m=far,
        calibration=CameraCalibration("agentview", 4, 4, 90.0, np.zeros(3), np.eye(3)),
        image_adapter=CanonicalImageAdapter("vertical_flip"),
    )
    assert reference.valid and reference.depth_median_m == pytest.approx(0.7, abs=2e-6)
    assert metric[3, 1] == pytest.approx(0.7, abs=2e-6)
    converted = normalized_depth_to_metric(
        np.asarray([_depth_buffer_for_z(0.5, near, far), _depth_buffer_for_z(2.0, near, far)]),
        near_m=near, far_m=far,
    )
    assert converted.tolist() == pytest.approx([0.5, 2.0], abs=2e-6)
    assert DiagnosticSimulatorDepthProvider.diagnostic_only is True


def test_invalid_normalized_simulator_depth_is_filtered_before_unprojection():
    from scripts.runtime_v3_metric_entity_depth_diagnostic import (
        metric_entity_reference_from_rgbd,
        normalized_depth_to_metric,
    )

    near, far = 0.1, 10.0
    raw = np.asarray([_depth_buffer_for_z(1.0, near, far), 0.0, 1.0,
                      np.nan, np.inf, -0.2], dtype=np.float32).reshape(1, 6)
    metric = normalized_depth_to_metric(raw, near_m=near, far_m=far)
    reference, _ = metric_entity_reference_from_rgbd(
        entity_key="drawer handle", camera="agentview", source_frame_id="sample",
        target_mask_canonical=np.ones((1, 6), dtype=bool), depth_buffer_raw=raw,
        near_m=near, far_m=far,
        calibration=CameraCalibration("agentview", 6, 1, 90.0, np.zeros(3), np.eye(3)),
        image_adapter=CanonicalImageAdapter("identity"),
    )
    assert np.isfinite(metric[0, 0])
    assert np.isnan(metric[0, 1:]).all()
    assert reference.valid_depth_count == 1
    assert reference.mask_pixel_count == 6
    assert reference.valid_depth_ratio == pytest.approx(1 / 6)


def test_metric_entity_reference_is_generic_immutable_and_frozen_once():
    reference = _metric_reference(entity="apple")
    assert reference.entity_key == "apple"
    assert reference.depth_source == "monocular_metric"
    assert reference.coordinate_frame == "world"
    later = _metric_reference((0.1, 0.0, -1.0))
    assert freeze_reference(None, reference) is reference
    assert freeze_reference(reference, later) is reference
    with pytest.raises(FrozenInstanceError):
        reference.reference_world_m = (9.0, 9.0, 9.0)


def freeze_reference(current, candidate):
    from core.runtime_v3.metric_entity import freeze_metric_reference
    return freeze_metric_reference(current, candidate)


def test_metric_distance_uses_only_proprioception_and_formal_reference():
    reference = _metric_reference((0.0, 0.0, -1.0))
    assert metric_proximity_distance_m((0.0, 0.0, 0.0), reference) == pytest.approx(1.0)
    assert metric_proximity_distance_m(None, reference) is None
    assert metric_proximity_distance_m((0.0, 0.0, 0.0), reference.invalidate("invalid")) is None


def test_metric_reference_is_accepted_by_the_single_existing_belief_state():
    reference = _metric_reference(entity="button")
    observation = RobotObservation(
        "frame", 1,
        evidence={"object_relative_state": ObjectRelativeState(
            target_phrase="semantic entity", target_visible=True,
            metric_entity_reference=reference,
        )},
    )
    state = StateBuilder().update(BeliefState(), observation)
    assert state.object_relative_state.metric_entity_reference is reference
    parsed = StateBuilder().update(BeliefState(), RobotObservation(
        "frame-2", 2,
        evidence={"object_relative_state": {
        "target_phrase": "entity", "target_visible": True,
        "metric_entity_reference": reference.__dict__,
        }},
    ))
    assert parsed.object_relative_state.metric_entity_reference == reference


def test_formal_runtime_has_no_simulator_depth_provider_or_target_pose_path():
    root = Path(__file__).resolve().parents[2]
    core = root / "core/runtime_v3"
    for path in core.rglob("*.py"):
        source = path.read_text(encoding="utf-8").casefold()
        assert "diagnosticsimulatordepthprovider" not in source
        assert "libero_depth" not in source
        assert "diagnostic_camera_depths" not in source
        assert "target_body_id" not in source
        assert "sim.data.xpos" not in source
        assert "oracle_contact" not in source
        assert "camera_depths" not in source
    assert "check_success" not in (core / "adapters/libero_env.py").read_text(encoding="utf-8")
    scene_source = (core / "scene_settling.py").read_text(encoding="utf-8").casefold()
    assert "target_world_position_m" not in scene_source
    assert "oracle_pixel_shift" not in scene_source


def test_simulator_diagnostic_provider_is_not_importable_from_formal_runtime():
    import core.runtime_v3.depth as formal_depth
    from scripts.runtime_v3_metric_entity_depth_diagnostic import DiagnosticSimulatorDepthProvider

    assert not hasattr(formal_depth, "DiagnosticSimulatorDepthProvider")
    assert DiagnosticSimulatorDepthProvider.diagnostic_only
    raw = {"agentview_depth": np.asarray([[0.5]], dtype=np.float32)}
    assert np.array_equal(
        DiagnosticSimulatorDepthProvider().estimate_normalized_buffer(raw, "agentview"),
        raw["agentview_depth"],
    )


def test_m35_experiment_injects_only_deployable_depth_into_formal_observer():
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/runtime_v3_metric_entity_grounding.py").read_text(
        encoding="utf-8")
    assert "depth_provider = MoGeMetricDepthProvider(" in source
    assert source.count("metric_depth_provider=depth_provider") == 2
    assert "metric_depth_provider=_SIMULATOR_DEPTH_PROVIDER" not in source
    assert "_SIMULATOR_DEPTH_PROVIDER.estimate_normalized_buffer" in source


def test_metric_construction_has_no_object_or_task_specific_branch():
    root = Path(__file__).resolve().parents[2]
    source = (root / "core/runtime_v3/metric_entity.py").read_text(encoding="utf-8").casefold()
    provider = (root / "core/runtime_v3/depth.py").read_text(encoding="utf-8").casefold()
    for text in (source, provider):
        assert "salad_dressing" not in text
        assert "task_id" not in text
        assert "if entity_key ==" not in text
        assert "simulator_gt" not in text


def test_oracle_diagnostic_cannot_affect_runtime_candidate_ranking_or_termination():
    from scripts.runtime_v3_near_target_observability import _stop_reason
    assert "contact" not in inspect.signature(_stop_reason).parameters


def test_qwen_zero_action_and_unapproved_executor_protection_remain():
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/runtime_v3_metric_entity_grounding.py").read_text(encoding="utf-8")
    assert "QwenClient" not in source and "QwenSelectorAdapter" not in source
    assert '"qwen_actions": 0' in source

    class Backend:
        def execute_approved_action(self, _action):
            raise AssertionError("unapproved action reached backend")

    arbiter = Arbiter()
    unapproved = ApprovedAction(
        option_id="hold", primitive=PrimitiveCommand("hold"),
        expected_effect={}, state_step_id=0, evidence_frame_id=0,
    )
    with pytest.raises(TypeError, match="authorized by its Arbiter"):
        Executor(Backend(), arbiter).execute(unapproved)

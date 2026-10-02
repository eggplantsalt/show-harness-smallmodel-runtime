import base64
import io
import os

import numpy as np
from PIL import Image

from core.capabilities.camera_geometry import CameraCalibration, project_point
from core.runtime_v3.adapters.qwen_selector import QwenSelectorAdapter
from core.runtime_v3.arbiter import Arbiter, DecisionKind
from core.runtime_v3.effects import EffectObserver
from core.runtime_v3.executor import Executor
from core.runtime_v3.observer import RobotObservation
from core.runtime_v3.object_relative import (
    ObjectRelativeAlignmentOptionGenerator,
    ObjectRelativePerceptionObserver,
    decode_sam3_mask,
    resolve_object_relative_geometry,
    segmentation_from_response,
)
from core.runtime_v3.runner import RuntimeV3Runner
from core.runtime_v3.selector import DeterministicSelector
from core.runtime_v3.state import BeliefState, ObjectRelativeState, StateBuilder


MOVE_VECTORS = {
    "MV_FWD": [1.0, 0.0, 0.0], "MV_BACK": [-1.0, 0.0, 0.0],
    "MV_LEFT": [0.0, 1.0, 0.0], "MV_RIGHT": [0.0, -1.0, 0.0],
    "MV_UP": [0.0, 0.0, 1.0], "MV_DOWN": [0.0, 0.0, -1.0],
}


def _calibration():
    return CameraCalibration(
        name="agentview", width=512, height=512, fovy_deg=60.0,
        position_world=np.zeros(3), camera_to_world=np.eye(3),
    )


def _encoded_mask(mask):
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _sam_response(mask, score=0.83):
    ys, xs = np.nonzero(mask)
    return {
        "success": True,
        "details": {"metadata": {"image_size": [mask.shape[1], mask.shape[0]]},
                    "detections": [{"score": score,
                                    "mask": {"format": "png", "base64": _encoded_mask(mask)}}]},
    }


def test_high_resolution_source_is_used_by_perception_without_resize():
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    mask = np.zeros((512, 512), dtype=bool)
    mask[200:220, 300:330] = True

    class BaseObserver:
        last_raw = type("Raw", (), {"eef_position_xyz": (0.0, 0.0, -1.0)})()

        def observe(self, _environment):
            return RobotObservation("obs", 1, images={"agentview": image},
                                    proprioception={}, evidence={"relevant_geometry": {
                                        "workspace_valid": True,
                                        "workspace_z_bounds_m": [-2.0, 1.0],
                                    }})

    class Sam:
        input_shape = None

        def segment(self, received, _phrase, **_kwargs):
            self.input_shape = received.shape
            return _sam_response(mask)

    class Sim:
        model = type("Model", (), {"camera_name2id": lambda _self, _name: 0,
                                    "cam_fovy": [60.0]})()
        data = type("Data", (), {"cam_xpos": [[0.0, 0.0, 0.0]],
                                  "cam_xmat": [np.eye(3).reshape(-1)]})()

    environment = type("Environment", (), {"sim": Sim()})()
    sam = Sam()
    observer = ObjectRelativePerceptionObserver(
        BaseObserver(), sam, target_phrase="salad dressing", move_vectors=MOVE_VECTORS,
    )
    observation = observer.observe(environment)
    assert sam.input_shape == image.shape
    assert observation.images["agentview"].shape == (512, 512, 3)
    assert observation.evidence["object_relative_state"].source_width == 512
    assert observation.evidence["object_relative_state"].source_height == 512


def test_source_resolution_is_not_faked_by_mask_or_sam_resize():
    mask = np.zeros((256, 256), dtype=bool)
    mask[10:30, 10:30] = True
    decoded = decode_sam3_mask({"base64": _encoded_mask(mask)}, (512, 512))
    assert decoded is None


def test_v3_local_sam3_bridge_bypasses_inherited_proxy(monkeypatch):
    from scripts.runtime_v3_object_relative_alignment import _configure_local_sam3_proxy_bypass

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7890")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    _configure_local_sam3_proxy_bypass("http://127.0.0.1:8773/sse")
    assert "127.0.0.1" in os.environ["NO_PROXY"].split(",")
    assert "127.0.0.1" in os.environ["no_proxy"].split(",")


def test_sam3_mask_centroid_bbox_area_and_quality_signal_are_extracted():
    mask = np.zeros((64, 80), dtype=bool)
    mask[11:21, 23:43] = True
    result = segmentation_from_response(_sam_response(mask, score=0.71), mask.shape)
    assert result.visible
    assert result.centroid_px == (32.5, 15.5)
    assert result.bbox_xyxy == (23, 11, 43, 21)
    assert result.area_px == 200
    assert result.quality_score == 0.71


def test_sam3_missing_detection_returns_invisible_without_fake_confidence():
    result = segmentation_from_response({"success": True, "details": {"detections": []}}, (10, 10))
    assert not result.visible
    assert result.quality_score is None
    assert result.centroid_px is None


def test_eef_projection_uses_calibrated_camera_geometry():
    projected = project_point(_calibration(), np.array([0.0, 0.0, -1.0]))
    assert projected is not None
    assert projected["pixel_xy"] == [256.0, 256.0]
    assert projected["in_frame"]


def test_hypothetical_displacement_is_projected_through_camera_calibration():
    result = resolve_object_relative_geometry(
        target_centroid_px=(300.0, 256.0), eef_position_xyz_m=(0, 0, -1),
        calibration=_calibration(), move_vectors=MOVE_VECTORS,
        workspace_z_bounds_m=(-2, 1),
    )
    forward = next(item for item in result["candidate_directions"] if item["direction"] == "FWD")
    assert forward["valid"]
    assert forward["hypothetical_projection_px"][0] > 256.0


def test_predicted_error_and_improvement_are_euclidean_pixel_distances():
    result = resolve_object_relative_geometry(
        target_centroid_px=(300.0, 256.0), eef_position_xyz_m=(0, 0, -1),
        calibration=_calibration(), move_vectors=MOVE_VECTORS,
        workspace_z_bounds_m=(-2, 1),
    )
    assert np.isclose(result["pixel_error_before_px"], 44.0)
    chosen = result["chosen_candidate"]
    assert chosen["direction"] == "FWD"
    assert np.isclose(chosen["predicted_improvement_px"],
                      44.0 - chosen["predicted_error_after_px"])


def test_runtime_selects_maximum_positive_physical_candidate():
    result = resolve_object_relative_geometry(
        target_centroid_px=(190.0, 256.0), eef_position_xyz_m=(0, 0, -1),
        calibration=_calibration(), move_vectors=MOVE_VECTORS,
        workspace_z_bounds_m=(-2, 1),
    )
    assert result["chosen_candidate"]["direction"] == "BACK"
    assert result["reason"] == "deterministic_object_relative_geometry"


def test_invalid_camera_geometry_resolves_to_reobserve_without_option():
    result = resolve_object_relative_geometry(
        target_centroid_px=(256, 256), eef_position_xyz_m=(0, 0, -1),
        calibration=None, move_vectors=MOVE_VECTORS, workspace_z_bounds_m=(-2, 1),
    )
    assert not result["camera_projection_valid"]
    state = BeliefState(object_relative_state=ObjectRelativeState("salad dressing", True),
                        relevant_geometry={"workspace_valid": True})
    assert ObjectRelativeAlignmentOptionGenerator().generate(state) == []
    assert DeterministicSelector().select(state, []).option_id == "REOBSERVE"
    assert result["reason"] == "camera_calibration_invalid"


def test_missing_object_resolves_to_reobserve_without_option():
    state = BeliefState(
        object_relative_state=ObjectRelativeState("salad dressing", False),
        relevant_geometry={"workspace_valid": True, "camera_projection_valid": True,
                           "object_relative_alignment_valid": True,
                           "chosen_candidate": {"direction": "FWD", "direction_unit": [1, 0, 0]}},
    )
    assert ObjectRelativeAlignmentOptionGenerator().generate(state) == []
    assert DeterministicSelector().select(state, []).option_id == "REOBSERVE"


def test_no_positive_candidate_resolves_to_reobserve():
    result = resolve_object_relative_geometry(
        target_centroid_px=(256, 256), eef_position_xyz_m=(0, 0, -1),
        calibration=_calibration(), move_vectors=MOVE_VECTORS,
        workspace_z_bounds_m=(-2, 1),
    )
    assert result["chosen_candidate"] is None
    assert result["reason"] == "no_candidate_predicted_to_improve_alignment"


def test_generated_direction_is_sealed_in_micro_motion_spec():
    state = _alignable_state(direction="RIGHT", unit=(0, -1, 0))
    option = ObjectRelativeAlignmentOptionGenerator().generate(state)[0]
    spec = option.primitive.micro_motion_spec
    assert option.option_id == "ALIGN_TO_TARGET_SMALL"
    assert spec.direction == "RIGHT"
    assert spec.direction_unit == (0.0, -1.0, 0.0)
    assert spec.requested_displacement_m == 0.003
    assert spec.max_ticks == 5


def test_one_alignment_option_causes_exactly_one_arbiter_approval():
    class CountingArbiter(Arbiter):
        calls = 0

        def authorize(self, *args, **kwargs):
            self.calls += 1
            return super().authorize(*args, **kwargs)

    class Observer:
        frame = 0

        def observe(self, _environment):
            self.frame += 1
            return _robot_observation(self.frame, (0.005 * (self.frame - 1), 0.0, -1.0))

        def observe_for_execution_tick(self, environment):
            return _robot_observation(10, (0.005, 0.0, -1.0))

    class Backend:
        directions = []

        def execute_approved_micro_tick(self, action):
            self.directions.append(action.primitive.micro_motion_spec.direction)
            return {"ok": True}

        def execute_approved_action(self, _action):
            raise AssertionError("not a micro-motion")

    arbiter = CountingArbiter()
    backend = Backend()
    runner = RuntimeV3Runner(
        observer=Observer(), state_builder=StateBuilder(),
        option_generator=ObjectRelativeAlignmentOptionGenerator(),
        selector=DeterministicSelector("ALIGN_TO_TARGET_SMALL"), arbiter=arbiter,
        executor=Executor(backend, arbiter), effect_observer=EffectObserver(),
    )
    result = runner.run_episode(object(), task_id="LIBERO_OBJECT:2", max_steps=1, reset=False)
    assert result["actions"] == 1
    assert arbiter.calls == 1
    assert backend.directions == ["FWD"]


def test_qwen_visual_smoke_sends_raw_image_and_only_semantic_choices():
    class Client:
        model = "qwen-test"

        def complete_json(self, prompt, agentview_image, wrist_image, **kwargs):
            self.prompt = prompt
            self.image_shape = np.asarray(agentview_image).shape
            assert wrist_image is None
            return type("Response", (), {
                "raw_text": '{"selection":"OPTION_A"}',
                "payload": {"request_audit": {"images": [{"size": [512, 512]}]}},
            })()

    image = np.zeros((512, 512, 3), dtype=np.uint8)
    state = BeliefState(
        task_id="LIBERO_OBJECT:2",
        object_relative_state=ObjectRelativeState(
            "salad dressing", True, target_centroid_px=(300, 250),
            eef_projection_px=(250, 250), image_error_px=(50, 0), image_error_norm_px=50,
        ),
    )
    client = Client()
    selector = QwenSelectorAdapter(client, "Pick up the salad dressing and place it in the basket.")
    result = selector.select_visual_semantic(state, image, [
        {"option_id": "OPTION_A", "description": "Align to the visible target."},
        {"option_id": "OPTION_B", "description": "Reobserve the target."},
        {"option_id": "OPTION_C", "description": "Abort this attempt."},
    ])
    assert result.option_id == "OPTION_A"
    assert client.image_shape == image.shape
    assert selector.last_record["model_input_width"] == selector.last_record["source_width"] == 512
    assert selector.last_record["model_input_height"] == selector.last_record["source_height"] == 512
    assert selector.last_record["robot_action_executed"] == 0
    assert not hasattr(selector, "controller")
    assert not hasattr(selector, "backend")
    assert "LEFT" not in client.prompt and "RIGHT" not in client.prompt


def _alignable_state(direction="FWD", unit=(1, 0, 0)):
    return BeliefState(
        task_id="LIBERO_OBJECT:2", step_id=1, frame_id=1, observation_fresh=True,
        evidence_refs=("frame-1",),
        end_effector_state={"position_xyz": [0.0, 0.0, -1.0]},
        object_relative_state=ObjectRelativeState(
            "salad dressing", True, target_quality_score=0.8,
            target_centroid_px=(300, 256), eef_projection_px=(256, 256),
            image_error_px=(44, 0), image_error_norm_px=44,
        ),
        relevant_geometry={
            "workspace_valid": True, "workspace_z_bounds_m": [-2.0, 1.0],
            "camera_projection_valid": True, "object_relative_alignment_valid": True,
            "pixel_error_before_px": 44.0,
            "chosen_candidate": {"direction": direction,
                                 "direction_unit": list(unit),
                                 "predicted_error_after_px": 32.0,
                                 "predicted_improvement_px": 12.0},
        },
    )


def _robot_observation(frame, position):
    state = _alignable_state()
    candidates = [{
        "option_id": "ALIGN_TO_TARGET_SMALL",
        "option_type": "object_relative_verified_alignment",
        "description": "One bounded target-relative alignment.",
        "preconditions": {
            "observation_fresh": True,
            "object_relative_state.target_visible": True,
            "relevant_geometry.camera_projection_valid": True,
            "relevant_geometry.workspace_valid": True,
            "relevant_geometry.object_relative_alignment_valid": True,
        },
        "expected_effect": {"image_error_before_px": 44.0,
                            "predicted_image_error_after_px": 32.0,
                            "predicted_improvement_px": 12.0},
        "confidence": 1.0,
        "evidence": ["frame-1"], "evidence_frame_id": frame,
        "primitive": {"kind": "micro_motion", "max_steps": 1, "max_duration_s": 5.0,
                      "micro_motion_spec": {"direction": "FWD", "direction_unit": [1, 0, 0],
                                            "requested_displacement_m": 0.003,
                                            "max_ticks": 5, "control_tick_step_m": 0.005}},
    }]
    return RobotObservation(
        f"obs-{frame}", frame,
        proprioception={"end_effector_state": {"position_xyz": list(position)}},
        evidence={"stage": "ALIGN", "object_relative_state": state.object_relative_state,
                  "relevant_geometry": {**state.relevant_geometry,
                                        "option_candidates": candidates}},
        evidence_refs=("frame-1",), fresh=True,
    )

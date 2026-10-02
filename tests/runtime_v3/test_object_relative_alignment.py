import base64
import io
import os

import numpy as np
from PIL import Image

from core.capabilities.camera_geometry import CameraCalibration, project_point
from core.runtime_v3.canonical_image import CanonicalImageAdapter
from core.runtime_v3.adapters.qwen_selector import QwenSelectorAdapter
from core.runtime_v3.arbiter import Arbiter, DecisionKind
from core.runtime_v3.effects import EffectObserver
from core.runtime_v3.executor import Executor
from core.runtime_v3.observer import RobotObservation
from core.runtime_v3.object_relative import (
    TargetCandidate,
    TargetIdentityAnchor,
    alignment_verification_metrics,
    associate_target_candidate,
    make_target_identity_anchor,
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


def _sam_response_many(masks, scores=None):
    scores = list(scores or [0.83] * len(masks))
    detections = []
    for index, (mask, score) in enumerate(zip(masks, scores)):
        ys, xs = np.nonzero(mask)
        detections.append({
            "rank": index, "backend_index": index, "score": score,
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
            "area_px": int(mask.sum()),
            "mask": {"format": "png", "base64": _encoded_mask(mask)},
        })
    return {"success": True, "details": {"metadata": {"image_size": [masks[0].shape[1], masks[0].shape[0]]},
                                            "detection_count": len(detections),
                                            "detections": detections}}


def test_canonical_vertical_flip_maps_image_mask_point_and_half_open_bbox_together():
    adapter = CanonicalImageAdapter()
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    mask = np.zeros((5, 7), dtype=bool)
    mask[1:3, 2:5] = True
    assert np.array_equal(adapter.transform_image(image), np.flipud(image))
    assert np.array_equal(adapter.transform_mask(mask), np.flipud(mask))
    assert adapter.transform_point((2, 1), width=7, height=5) == (2.0, 3.0)
    assert adapter.transform_bbox((2, 1, 5, 3), width=7, height=5) == (2.0, 2.0, 5.0, 4.0)


def test_canonical_horizontal_and_180_transforms_keep_raster_coordinates_consistent():
    image = np.arange(3 * 5, dtype=np.uint8).reshape(3, 5)
    horizontal = CanonicalImageAdapter("horizontal_flip")
    rotate = CanonicalImageAdapter("rotate_180")
    assert np.array_equal(horizontal.transform_mask(image), np.fliplr(image))
    assert horizontal.transform_point((1, 0), width=5, height=3) == (3.0, 0.0)
    assert horizontal.transform_bbox((1, 0, 3, 2), width=5, height=3) == (2.0, 0.0, 4.0, 2.0)
    assert np.array_equal(rotate.transform_mask(image), np.rot90(image, 2))
    assert rotate.transform_point((1, 0), width=5, height=3) == (3.0, 2.0)
    assert rotate.transform_bbox((1, 0, 3, 2), width=5, height=3) == (2.0, 1.0, 4.0, 3.0)


def test_canonical_eef_projection_uses_the_same_orientation_as_the_image():
    adapter = CanonicalImageAdapter()
    calibration = CameraCalibration(
        name="agentview", width=100, height=100, fovy_deg=60.0,
        position_world=np.zeros(3), camera_to_world=np.eye(3),
    )
    current_raw = project_point(calibration, np.asarray([0.0, 0.0, -1.0]))["pixel_xy"]
    fwd_raw = project_point(calibration, np.asarray([0.01, 0.0, -1.0]))["pixel_xy"]
    left_raw = project_point(calibration, np.asarray([0.0, 0.01, -1.0]))["pixel_xy"]
    current = np.asarray(adapter.transform_projected_point(current_raw, width=100, height=100))
    fwd = np.asarray(adapter.transform_projected_point(fwd_raw, width=100, height=100))
    left = np.asarray(adapter.transform_projected_point(left_raw, width=100, height=100))
    assert fwd[0] > current[0]  # horizontal-sensitive direction remains rightward
    assert left[1] < current[1]  # vertical-sensitive direction is up in top-left image coordinates


def test_libero_runtime_requests_the_selected_resolution_from_the_source_renderer(monkeypatch):
    from types import SimpleNamespace
    import core.runtime_v3.adapters.libero_env as libero_env_module

    seen = {}
    fake_handle = SimpleNamespace(
        env=object(), task_id=2, task_name="task", task_description="instruction",
        suite_name="LIBERO_OBJECT", init_states=np.zeros((1, 2)), init_state_index=0,
    )

    def fake_make_libero_task(**kwargs):
        seen.update(kwargs)
        return fake_handle

    monkeypatch.setattr(libero_env_module, "make_libero_task", fake_make_libero_task)
    environment = libero_env_module.LiberoEnvironmentAdapter.create(
        suite_name="LIBERO_OBJECT", task_id=2, init_state_index=0,
        camera_height=768, camera_width=768,
    )
    assert seen["camera_height"] == seen["camera_width"] == 768
    assert environment.env is fake_handle.env


def test_segmentation_keeps_the_entire_sam3_candidate_set():
    first = np.zeros((64, 80), dtype=bool)
    second = np.zeros_like(first)
    first[10:20, 10:24] = True
    second[31:45, 52:69] = True
    result = segmentation_from_response(_sam_response_many([first, second], [0.91, 0.77]), first.shape)
    assert result.visible
    assert [item.candidate_id for item in result.candidates] == ["0", "1"]
    assert [item.score for item in result.candidates] == [0.91, 0.77]
    assert result.selected_candidate_id == "0"


def test_target_identity_anchor_is_built_from_the_initial_selected_mask():
    mask = np.zeros((64, 80), dtype=bool)
    mask[10:20, 10:24] = True
    segmentation = segmentation_from_response(_sam_response(mask), mask.shape)
    anchor = make_target_identity_anchor(segmentation, target_phrase="salad dressing", frame_id=17)
    assert isinstance(anchor, TargetIdentityAnchor)
    assert anchor.target_phrase == "salad dressing"
    assert anchor.frame_id == 17
    assert anchor.mask_area == 140
    assert np.array_equal(anchor.initial_mask, mask)


def test_target_identity_anchor_is_not_created_without_a_valid_mask():
    segmentation = segmentation_from_response({"success": True, "details": {"detections": []}}, (32, 40))
    assert make_target_identity_anchor(segmentation, target_phrase="salad dressing", frame_id=1) is None


def test_same_target_mask_is_associated_by_overlap_even_if_another_candidate_scores_higher():
    anchor_mask = np.zeros((64, 80), dtype=bool)
    anchor_mask[10:22, 10:25] = True
    wrong_mask = np.zeros_like(anchor_mask)
    wrong_mask[31:45, 50:68] = True
    target_mask = anchor_mask.copy()
    candidates = segmentation_from_response(
        _sam_response_many([wrong_mask, target_mask], [0.99, 0.30]), anchor_mask.shape
    ).candidates
    anchor = TargetIdentityAnchor("salad dressing", anchor_mask, (17.0, 15.5),
                                  (10, 10, 25, 22), int(anchor_mask.sum()), "0", 1)
    association = associate_target_candidate(anchor, candidates)
    assert association.status == "SAME_TARGET"
    assert association.candidate.candidate_id == "1"
    assert association.candidate_metrics[0]["eligible"] is False
    assert association.candidate_metrics[1]["mask_iou"] == 1.0


def test_identity_association_rejects_a_mask_from_a_different_source_resolution():
    anchor_mask = np.zeros((64, 80), dtype=bool)
    anchor_mask[10:20, 10:20] = True
    anchor = TargetIdentityAnchor("salad dressing", anchor_mask, (14.5, 14.5),
                                  (10, 10, 20, 20), int(anchor_mask.sum()), "0", 1)
    wrong_resolution_mask = np.zeros((32, 40), dtype=bool)
    wrong_resolution_mask[5:10, 5:10] = True
    candidate = TargetCandidate("1", 0, 1, wrong_resolution_mask, (7.0, 7.0),
                                (5, 5, 10, 10), int(wrong_resolution_mask.sum()), 0.9)
    result = associate_target_candidate(anchor, [candidate])
    assert result.status == "TARGET_IDENTITY_LOST"
    assert result.candidate is None
    assert result.candidate_metrics[0]["reason"] == "candidate_mask_or_geometry_unavailable"


def test_different_object_cannot_replace_anchor_and_identity_loss_has_no_improvement_metric():
    anchor_mask = np.zeros((64, 80), dtype=bool)
    anchor_mask[10:22, 10:25] = True
    wrong_mask = np.zeros_like(anchor_mask)
    wrong_mask[31:45, 50:68] = True
    candidate = segmentation_from_response(_sam_response(wrong_mask), wrong_mask.shape).candidates[0]
    anchor = TargetIdentityAnchor("salad dressing", anchor_mask, (17.0, 15.5),
                                  (10, 10, 25, 22), int(anchor_mask.sum()), "0", 1)
    association = associate_target_candidate(anchor, [candidate])
    assert association.status == "TARGET_IDENTITY_LOST"
    assert association.candidate is None
    verification = alignment_verification_metrics(20.0, 1.0, identity_status=association.status)
    assert verification["verification_status"] == "TARGET_IDENTITY_LOST"
    assert verification["error_after_px"] is None
    assert verification["actual_improvement_px"] is None
    assert verification["alignment_improved"] is None


def test_valid_same_target_verification_computes_actual_improvement():
    verification = alignment_verification_metrics(10.0, 7.0, identity_status="SAME_TARGET")
    assert verification == {
        "verification_status": "SAME_TARGET",
        "error_after_px": 7.0,
        "actual_improvement_px": 3.0,
        "alignment_improved": True,
    }


def test_identity_lost_state_cannot_generate_alignment_option():
    state = _alignable_state()
    state = BeliefState(**{**state.__dict__, "object_relative_state": ObjectRelativeState(
        "salad dressing", False, target_identity_status="TARGET_IDENTITY_LOST")})
    assert ObjectRelativeAlignmentOptionGenerator().generate(state) == []


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
            target_identity_status="ANCHORED",
            target_reference_point_px=(300, 256), target_reference_valid=True,
            target_reference_camera="agentview",
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

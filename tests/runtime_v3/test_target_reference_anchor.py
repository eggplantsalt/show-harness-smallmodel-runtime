from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from core.runtime_v3.arbiter import Arbiter, DecisionKind
from core.runtime_v3.object_relative import (
    ObjectRelativeAlignmentOptionGenerator,
    ObjectRelativePerceptionObserver,
    TargetIdentityAnchor,
    TargetReferenceAnchor,
    _runtime_invalidation_signals,
    compare_alignment_improvements,
    frozen_reference_error,
    make_target_reference_anchor,
)
from core.runtime_v3.observer import RobotObservation
from core.runtime_v3.selector import Selection
from core.runtime_v3.state import BeliefState, ObjectRelativeState, StateBuilder
from scripts.runtime_v3_object_relative_alignment import CountingArbiter


def _encoded_mask(mask: np.ndarray) -> str:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _sam_response(mask: np.ndarray) -> dict:
    return {"success": True, "details": {"metadata": {"image_size": [mask.shape[1], mask.shape[0]]},
            "detections": [{"rank": 0, "backend_index": 0, "score": 0.9,
                            "mask": {"base64": _encoded_mask(mask)}}]}}


class _BaseObserver:
    def __init__(self, evidence_rows=None):
        self.frame = 0
        self.evidence_rows = list(evidence_rows or [{}])
        self.last_raw = SimpleNamespace(eef_position_xyz=(0.0, 0.0, -1.0))

    def observe(self, _environment):
        self.frame += 1
        evidence = self.evidence_rows[min(self.frame - 1, len(self.evidence_rows) - 1)]
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        return RobotObservation(
            f"obs-{self.frame}", self.frame,
            images={"agentview": image},
            proprioception={"end_effector_state": {"position_xyz": [0.0, 0.0, -1.0]}},
            evidence={"relevant_geometry": {"workspace_z_bounds_m": [-2.0, 1.0],
                                            "workspace_valid": True}, **evidence},
            evidence_refs=(f"frame-{self.frame}",), fresh=True,
        )


class _Sam:
    def __init__(self, masks):
        self.masks = list(masks)
        self.index = 0

    def segment(self, _image, _phrase, **_kwargs):
        mask = self.masks[min(self.index, len(self.masks) - 1)]
        self.index += 1
        return _sam_response(mask)


class _Environment:
    def __init__(self):
        self.camera_position = np.array([0.0, 0.0, 0.0])
        model = SimpleNamespace(camera_name2id=lambda _name: 0, cam_fovy=[60.0])
        data = SimpleNamespace(cam_xpos=[self.camera_position], cam_xmat=[np.eye(3).reshape(-1)])
        self.sim = SimpleNamespace(model=model, data=data)


def _observer(*, masks=None, evidence_rows=None, scene_ready_required=False):
    mask_a = np.zeros((64, 64), dtype=bool)
    mask_a[20:30, 20:30] = True
    mask_b = np.zeros_like(mask_a)
    mask_b[21:31, 20:30] = True
    base = _BaseObserver(evidence_rows)
    observer = ObjectRelativePerceptionObserver(
        base, _Sam(masks or [mask_a, mask_b]), target_phrase="target object",
        move_vectors={"MV_FWD": (1, 0, 0), "MV_BACK": (-1, 0, 0),
                      "MV_LEFT": (0, 1, 0), "MV_RIGHT": (0, -1, 0),
                      "MV_UP": (0, 0, 1), "MV_DOWN": (0, 0, -1)},
        scene_ready_required=scene_ready_required,
    )
    return observer, _Environment()


def _state(reference_valid=True):
    return BeliefState(
        task_id="LIBERO_OBJECT:2", step_id=1, frame_id=1, observation_fresh=True,
        evidence_refs=("frame-1",), end_effector_state={"position_xyz": [0.0, 0.0, -1.0]},
        object_relative_state=ObjectRelativeState(
            "target object", True, target_centroid_px=(30.0, 25.0),
            eef_projection_px=(32.0, 32.0), target_identity_status="ANCHORED",
            target_reference_point_px=(30.0, 25.0), target_reference_valid=reference_valid,
        ),
        relevant_geometry={
            "workspace_valid": True, "workspace_z_bounds_m": [-2.0, 1.0],
            "camera_projection_valid": True, "object_relative_alignment_valid": True,
            "pixel_error_before_px": 7.28,
            "chosen_candidate": {"direction": "FWD", "direction_unit": [1.0, 0.0, 0.0],
                                 "predicted_error_after_px": 5.0,
                                 "predicted_improvement_px": 2.28},
        },
    )


def test_reference_anchor_construction_records_initial_visual_geometry():
    anchor = TargetIdentityAnchor("target object", np.ones((8, 8), dtype=bool),
                                  (3.5, 4.5), (1, 2, 7, 8), 48, "0", 9)
    reference = make_target_reference_anchor(anchor, camera="agentview",
                                             camera_signature=("agentview", 64, 64))
    assert isinstance(reference, TargetReferenceAnchor)
    assert reference.reference_point_px == (3.5, 4.5)
    assert reference.reference_source == "initial_associated_sam_mask_centroid"
    assert reference.source_frame_id == 9
    assert reference.source_bbox_px == (1, 2, 7, 8)
    assert reference.source_mask_area == 48
    assert reference.valid


def test_new_sam_observations_do_not_move_reference_anchor():
    observer, env = _observer()
    observer.observe(env)
    frozen = observer.reference_anchor.reference_point_px
    observer.observe(env)
    assert observer.last_segmentation.centroid_px != frozen
    assert observer.reference_anchor.reference_point_px == frozen


def test_scene_ready_gate_defers_reference_until_three_stable_visual_observations():
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:30, 20:30] = True
    observer, env = _observer(masks=[mask, mask, mask], scene_ready_required=True)
    first = observer.observe(env)
    second = observer.observe(env)
    assert observer.reference_anchor is None
    assert first.evidence["object_relative_state"].scene_ready is False
    assert second.evidence["object_relative_state"].scene_ready is False
    third = observer.observe(env)
    assert observer.scene_ready is True
    assert observer.reference_anchor is not None and observer.reference_anchor.valid
    assert observer.reference_anchor.source_frame_id == third.frame_id
    assert third.evidence["object_relative_state"].scene_ready_gate_enabled is True


def test_scene_ready_gate_is_an_initialization_gate_not_a_target_motion_detector():
    mask_a = np.zeros((64, 64), dtype=bool)
    mask_a[20:30, 20:30] = True
    mask_b = np.zeros_like(mask_a)
    mask_b[28:38, 20:30] = True
    observer, env = _observer(masks=[mask_a, mask_a, mask_a, mask_b],
                              scene_ready_required=True)
    for _ in range(3):
        observer.observe(env)
    assert observer.scene_ready
    observer.observe(env)
    assert observer.scene_ready
    assert observer.reference_anchor is not None
    assert observer.reference_anchor.valid is True


def test_post_sam_centroid_is_diagnostic_while_runtime_error_uses_frozen_reference():
    observer, env = _observer()
    first = observer.observe(env)
    second = observer.observe(env)
    first_state = first.evidence["object_relative_state"]
    second_state = second.evidence["object_relative_state"]
    assert second_state.target_centroid_px != first_state.target_reference_point_px
    assert second_state.target_reference_point_px == first_state.target_reference_point_px
    assert second_state.image_error_norm_px == frozen_reference_error(
        first_state.target_reference_point_px, second_state.eef_projection_px,
    )


def test_reference_visual_artifact_saves_with_fixed_anchor_overlay(tmp_path):
    observer, env = _observer()
    observation = observer.observe(env)
    frame = observer.perception_history[-1]
    artifacts = observer.save_visual_artifacts(
        str(tmp_path), image=frame["image"], prefix="before",
        segmentation=frame["segmentation"], resolution=frame["resolution"],
    )
    assert (tmp_path / "before_overlay.png").is_file()
    assert artifacts["overlay"].endswith("before_overlay.png")
    assert observation.evidence["object_relative_state"].target_reference_valid


def test_explicit_reference_invalidation_preserves_reason():
    observer, env = _observer()
    observer.observe(env)
    observer.invalidate_target_reference("possible_contact")
    assert not observer.reference_anchor.valid
    assert observer.reference_anchor.invalidation_reason == "possible_contact"


def test_camera_change_invalidates_reference_anchor():
    observer, env = _observer()
    observer.observe(env)
    env.camera_position = np.array([0.01, 0.0, 0.0])
    env.sim.data.cam_xpos = [env.camera_position]
    observer.observe(env)
    assert not observer.reference_anchor.valid
    assert observer.reference_anchor.invalidation_reason == "camera_changed"


def test_possible_contact_signal_invalidates_reference_anchor():
    observer, env = _observer(evidence_rows=[{}, {"contact_state": "POSSIBLE_CONTACT"}])
    observer.observe(env)
    observer.observe(env)
    assert not observer.reference_anchor.valid
    assert observer.reference_anchor.invalidation_reason == "possible_contact"


def test_grasp_and_release_events_are_reference_invalidation_signals():
    assert _runtime_invalidation_signals({"gripper_event": "GRASPED"})["grasp_event"]
    assert _runtime_invalidation_signals({"gripper_event": "RELEASED"})["release_event"]
    assert _runtime_invalidation_signals({"target_motion_evidence": True})["target_motion_evidence"]


def test_explicit_reground_reestablishes_reference_from_new_associated_mask():
    first = np.zeros((64, 64), dtype=bool)
    first[10:20, 10:20] = True
    moved = np.zeros_like(first)
    moved[40:50, 40:50] = True
    observer, env = _observer(masks=[first, moved])
    observer.observe(env)
    old_reference = observer.reference_anchor.reference_point_px
    observer.request_target_reference_reground()
    observer.observe(env)
    assert observer.reference_anchor.valid
    assert observer.reference_anchor.reference_point_px != old_reference
    assert observer.reference_anchor.source_frame_id == 2


def test_frozen_reference_error_uses_eef_projection_not_post_mask_center():
    error = frozen_reference_error((10.0, 5.0), (7.0, 1.0))
    assert np.isclose(error, 5.0)


def test_frozen_reference_error_fails_closed_when_invalid_or_nonfinite():
    assert frozen_reference_error((1, 2), (3, 4), reference_valid=False) is None
    assert frozen_reference_error((float("nan"), 2), (3, 4)) is None
    assert frozen_reference_error((1, 2), None) is None


def test_prediction_residual_is_actual_minus_predicted():
    result = compare_alignment_improvements(predicted_improvement_px=1.25,
                                            actual_improvement_px=0.75)
    assert result["prediction_residual_actual_minus_predicted_px"] == -0.5


def test_missing_prediction_or_observation_has_no_residual():
    result = compare_alignment_improvements(predicted_improvement_px=1.0,
                                            actual_improvement_px=None)
    assert result["prediction_residual_actual_minus_predicted_px"] is None


def test_invalid_reference_cannot_generate_alignment_option():
    assert ObjectRelativeAlignmentOptionGenerator().generate(_state(reference_valid=False)) == []


def test_runtime_state_does_not_ingest_oracle_diagnostic_pose():
    state = StateBuilder().update(
        BeliefState(task_id="LIBERO_OBJECT:2"),
        RobotObservation("o", 1, evidence={"oracle_target_world_position_m": [9, 9, 9],
                                           "object_relative_state": {
                                               "target_phrase": "target object",
                                               "target_visible": True,
                                               "target_reference_point_px": [30, 25],
                                               "target_reference_valid": True,
                                           }}),
    )
    assert state.object_relative_state.target_reference_point_px == (30.0, 25.0)
    assert not hasattr(state, "oracle_target_world_position_m")
    assert "oracle_target_world_position_m" not in state.relevant_geometry


def test_qwen_selector_remains_without_controller_or_executor_access():
    from core.runtime_v3.adapters.qwen_selector import QwenSelectorAdapter

    selector = QwenSelectorAdapter(object(), "pick the target")
    assert not hasattr(selector, "controller")
    assert not hasattr(selector, "backend")
    assert not hasattr(selector, "executor")


def test_arbiter_still_issues_the_only_sealed_alignment_authority():
    state = _state(reference_valid=True)
    option = ObjectRelativeAlignmentOptionGenerator().generate(state)[0]
    decision = Arbiter().authorize(state, [option], Selection("ALIGN_TO_TARGET_SMALL", "VALID"))
    assert decision.kind == DecisionKind.APPROVED
    assert decision.action.option_id == "ALIGN_TO_TARGET_SMALL"
    assert decision.action.primitive.max_steps == 1


def test_pre_action_ready_callback_runs_before_alignment_authorization():
    events = []
    state = _state(reference_valid=True)
    options = ObjectRelativeAlignmentOptionGenerator().generate(state)
    arbiter = CountingArbiter(before_authorize=lambda *_: events.append("PRE_ACTION_READY"))
    decision = arbiter.authorize(state, options, Selection("ALIGN_TO_TARGET_SMALL", "VALID"))
    assert decision.kind == DecisionKind.APPROVED
    assert events == ["PRE_ACTION_READY"]
    assert arbiter.pre_action_ready_written


def test_pre_action_logger_failure_blocks_authorization():
    def fail_write(*_args):
        raise OSError("artifact directory is not writable")

    state = _state(reference_valid=True)
    options = ObjectRelativeAlignmentOptionGenerator().generate(state)
    arbiter = CountingArbiter(before_authorize=fail_write)
    with pytest.raises(OSError, match="not writable"):
        arbiter.authorize(state, options, Selection("ALIGN_TO_TARGET_SMALL", "VALID"))
    assert arbiter.approval_count == 0
    assert not arbiter.pre_action_ready_written

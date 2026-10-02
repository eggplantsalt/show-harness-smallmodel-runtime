import base64
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from core.runtime_v3.canonical_image import CanonicalImageAdapter
from core.runtime_v3.object_relative import (
    MultiScaleAlignmentOptionGenerator, TargetSegmentation,
    make_target_identity_anchor, segmentation_from_response,
)
from core.runtime_v3.state import BeliefState, ObjectRelativeState
from scripts import runtime_v3_near_target_observability as audit


def _response(mask):
    stream = io.BytesIO()
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(stream, format="PNG")
    ys, xs = np.nonzero(mask)
    return {"success": True, "details": {"metadata": {"image_size": [mask.shape[1], mask.shape[0]]},
            "detections": [{"score": 0.9,
                            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
                            "area_px": int(mask.sum()),
                            "mask": {"format": "png", "base64": base64.b64encode(stream.getvalue()).decode()}}]}}


def _mask(shape=(8, 10), bounds=(2, 5, 5, 8)):
    mask = np.zeros(shape, dtype=bool)
    y0, y1, x0, x1 = bounds
    mask[y0:y1, x0:x1] = True
    return mask


def _runtime_state(candidate=None, *, extras=None):
    candidate = candidate or {
        "valid": True, "scale_contract_valid": True, "workspace_valid": True,
        "direction": "DOWN", "direction_unit": [0.0, 0.0, -1.0],
        "displacement_m": 0.009, "max_ticks": 10,
        "predicted_error_px": 20.0, "predicted_improvement_px": 5.0,
    }
    geometry = {
        "camera_projection_valid": True, "workspace_valid": True,
        "multiscale_alignment_valid": True, "pixel_error_before_px": 25.0,
        "chosen_lattice_candidate": candidate,
    }
    geometry.update(extras or {})
    relative = ObjectRelativeState(
        target_phrase="salad dressing", target_visible=True,
        target_identity_status="ANCHORED", target_reference_point_px=(300.0, 256.0),
        target_reference_valid=True, eef_projection_px=(275.0, 256.0),
    )
    return BeliefState(
        task_id="LIBERO_OBJECT:2", object_relative_state=relative,
        end_effector_state={"position_xyz": (0.1, 0.2, 0.3)},
        relevant_geometry=geometry, frame_id=17, observation_fresh=True,
    )


def _fake_env():
    class Model:
        body_parentid = np.asarray([0, 0])
        geom_bodyid = np.asarray([], dtype=int)

        @staticmethod
        def body_name2id(_name):
            return 1

        @staticmethod
        def body_id2name(index):
            return "world" if int(index) == 0 else audit.TARGET_BODY

    class Data:
        xpos = np.asarray([[0.0, 0.0, 0.0], [0.4, 0.5, 0.6]])
        ncon = 0

    class Sim:
        model = Model()
        data = Data()

    class Env:
        sim = Sim()

        @staticmethod
        def step(_action):
            raise AssertionError("observation diagnostics must not execute actions")

    return SimpleNamespace(env=Env())


def test_wrist_source_keeps_the_genuine_render_resolution_without_upscaling():
    raw = np.arange(512 * 512 * 3, dtype=np.uint32).reshape(512, 512, 3).astype(np.uint8)
    canonical = audit.canonicalize_wrist(raw, "vertical_flip")
    assert canonical.shape == raw.shape == (512, 512, 3)
    assert canonical.dtype == np.uint8
    assert np.array_equal(canonical, np.flipud(raw))


def test_wrist_orientation_audit_keeps_transforms_independent_and_source_sized(tmp_path):
    raw_agent = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    raw_wrist = np.flipud(raw_agent).copy()
    result = audit.save_orientation_audit(raw_agent, raw_wrist, tmp_path)
    assert result["wrist_transform_selected_after_visual_audit"] is None
    assert result["agentview_canonical_transform"] == "vertical_flip"
    assert result["wrist_candidate_transforms"] == list(audit.ORIENTATION_CHOICES)
    assert result["wrist_raw_resolution"] == [7, 5]
    for orientation in audit.ORIENTATION_CHOICES:
        actual = audit.canonicalize_wrist(raw_wrist, orientation)
        expected = CanonicalImageAdapter(orientation).transform_image(raw_wrist)
        assert actual.shape == raw_wrist.shape
        assert np.array_equal(actual, expected)


def test_wrist_target_metrics_extract_visibility_candidates_centroid_and_bbox():
    mask = _mask()
    segmentation = segmentation_from_response(_response(mask), mask.shape)
    metrics = audit.extract_wrist_metrics(segmentation, mask.shape)
    assert metrics["target_visible"] is True
    assert metrics["candidate_count"] == 1
    assert metrics["centroid_px"] == [6.0, 3.0]
    assert metrics["bbox_xyxy"] == [5, 2, 8, 5]
    assert metrics["mask_area_px"] == 9
    assert metrics["association_status"] == "UNANCHORED"


def test_wrist_mask_area_ratio_uses_source_image_area():
    segmentation = segmentation_from_response(_response(_mask()), (8, 10))
    metrics = audit.extract_wrist_metrics(segmentation, (8, 10))
    assert metrics["mask_area_ratio"] == 9 / 80
    assert metrics["bbox_area_ratio"] == 9 / 80


def test_wrist_center_error_is_normalized_by_image_diagonal():
    mask = np.zeros((8, 10), dtype=bool)
    mask[3:5, 4:6] = True
    segmentation = segmentation_from_response(_response(mask), mask.shape)
    metrics = audit.extract_wrist_metrics(segmentation, mask.shape)
    assert metrics["center_error_normalized"] == 0.0
    moved = np.zeros_like(mask)
    moved[0:2, 0:2] = True
    far_metrics = audit.extract_wrist_metrics(segmentation_from_response(_response(moved), moved.shape), moved.shape)
    expected = np.hypot(0.5 - 4.5, 0.5 - 3.5) / np.hypot(10, 8)
    assert np.isclose(far_metrics["center_error_normalized"], expected)


def test_oracle_distance_is_euclidean_and_handles_missing_or_nonfinite_inputs():
    assert np.isclose(audit.oracle_distance((0, 0, 0), (3, 4, 0)), 5.0)
    assert audit.oracle_distance(None, (0, 0, 0)) is None
    assert audit.oracle_distance((float("nan"), 0, 0), (0, 0, 0)) is None


def test_observation_record_keeps_oracle_out_of_runtime_decision_state():
    state = _runtime_state()
    raw_wrist = np.zeros((8, 10, 3), dtype=np.uint8)
    wrist_mask = _mask()

    class Sam:
        seen_shape = None

        def segment(self, image, _phrase, confidence_threshold):
            self.seen_shape = image.shape
            assert confidence_threshold == 0.05
            return _response(wrist_mask)

    sam = Sam()
    agent_mask = _mask()
    agent_seg = segmentation_from_response(_response(agent_mask), agent_mask.shape)
    frame = {"raw_image": np.zeros((8, 10, 3), dtype=np.uint8),
             "image": np.zeros((8, 10, 3), dtype=np.uint8), "segmentation": agent_seg}
    before_geometry = dict(state.relevant_geometry)
    record, data, *_ = audit._observation_record(
        step=0, state=state, frame=frame, raw_wrist=raw_wrist, sam3=sam,
        wrist_orientation="identity", previous_wrist_anchor=None, environment=_fake_env(),
    )
    assert sam.seen_shape == raw_wrist.shape
    assert data["wrist_raw_resolution"] == data["wrist_canonical_resolution"]
    assert data["wrist_canonical_resolution"] == data["wrist_sam_input_resolution"]
    assert np.isclose(record.oracle_eef_target_distance_m, np.linalg.norm([0.3, 0.3, 0.3]))
    assert record.oracle_target_position_xyz_m == (0.4, 0.5, 0.6)
    assert state.relevant_geometry == before_geometry
    assert "oracle_target_position_xyz_m" not in state.relevant_geometry
    assert "oracle_eef_target_distance_m" not in state.relevant_geometry
    assert data["target_body_diagnostic"]["source"] == "simulator_diagnostic_only"
    assert "oracle_used_for_runtime" not in data


def test_observation_diagnostics_do_not_call_environment_actions():
    state = _runtime_state()
    seg = segmentation_from_response(_response(_mask()), (8, 10))

    class Sam:
        def segment(self, *_args, **_kwargs):
            return _response(_mask())

    frame = {"raw_image": np.zeros((8, 10, 3), dtype=np.uint8),
             "image": np.zeros((8, 10, 3), dtype=np.uint8), "segmentation": seg}
    audit._observation_record(
        step=0, state=state, frame=frame, raw_wrist=np.zeros((8, 10, 3), dtype=np.uint8),
        sam3=Sam(), wrist_orientation="identity", previous_wrist_anchor=None,
        environment=_fake_env(),
    )


def test_runtime_selected_direction_and_scale_still_come_from_runtime_geometry():
    candidate = _runtime_state().relevant_geometry["chosen_lattice_candidate"]
    plain = _runtime_state(candidate)
    annotated = _runtime_state(candidate, extras={
        "oracle_target_position_xyz_m": [99.0, 99.0, 99.0],
        "oracle_eef_target_distance_m": 0.001, "contact_diagnostic": True,
        "wrist_recommends_direction": "UP", "wrist_recommends_scale_m": 0.003,
    })
    option_a = MultiScaleAlignmentOptionGenerator().generate(plain)[0]
    option_b = MultiScaleAlignmentOptionGenerator().generate(annotated)[0]
    assert option_a.primitive.micro_motion_spec == option_b.primitive.micro_motion_spec
    assert option_b.primitive.micro_motion_spec.direction == "DOWN"
    assert option_b.primitive.micro_motion_spec.requested_displacement_m == 0.009


def test_contact_diagnostic_cannot_stop_or_select_a_different_action():
    state = _runtime_state()
    baseline = MultiScaleAlignmentOptionGenerator().generate(state)[0]
    contact_annotated_state = _runtime_state(extras={
        "oracle_contact": True, "oracle_target_position_xyz_m": [99.0, 99.0, 99.0],
        "oracle_eef_target_distance_m": 0.0,
    })
    selected = MultiScaleAlignmentOptionGenerator().generate(contact_annotated_state)[0]
    assert selected.primitive.micro_motion_spec == baseline.primitive.micro_motion_spec
    assert audit._stop_reason(contact_annotated_state) == audit._stop_reason(state)


def test_qwen_is_not_in_the_observability_runtime_and_summary_counts_zero(tmp_path):
    assert not hasattr(audit, "QwenSelector")
    assert "QwenSelector" not in audit.__dict__
    summary = audit.summarize([], Path(tmp_path))
    assert summary["qwen_actions"] == 0
    assert summary["oracle_used_for_runtime"] is False


def test_trajectory_row_logs_all_required_deployable_and_oracle_diagnostics():
    fields = {
        "step": 2, "agentview_error_px": 10.0, "agentview_normalized_error": 0.01,
        "agentview_mask_area_ratio": 0.07, "agentview_bbox_width_ratio": 0.2,
        "agentview_bbox_height_ratio": 0.3,
        "oracle_eef_target_distance_m": 0.2, "wrist_visible": True,
        "wrist_candidate_count": 1, "wrist_mask_area_ratio": 0.05,
        "wrist_bbox_width_ratio": 0.2, "wrist_bbox_height_ratio": 0.3,
        "wrist_bbox_area_ratio": 0.06, "wrist_center_error_normalized": 0.1,
        "wrist_association_status": "SAME_TARGET", "wrist_candidate_id": "0",
        "wrist_centroid_px": [256.0, 200.0], "wrist_mask_area_px": 13107,
        "oracle_target_position_xyz_m": [0.2, 0.3, 0.4],
        "selected_direction": "DOWN", "selected_scale_mm": 9.0,
        "eef_position_xyz_m": [0.1, 0.2, 0.3], "eef_quaternion": [1, 0, 0, 0],
        "gripper_state": "UNKNOWN",
        "gripper_width_m": 0.04, "oracle_contact": False,
    }
    row = audit._series_row(fields)
    assert all(row[key] == value for key, value in fields.items())
    assert {"agentview_bbox_width_ratio", "agentview_bbox_height_ratio",
            "wrist_mask_area_ratio", "wrist_bbox_area_ratio", "wrist_center_error_normalized",
            "wrist_centroid_px", "oracle_target_position_xyz_m", "selected_direction",
            "selected_scale_mm", "oracle_eef_target_distance_m"} <= set(row)


def test_near_target_correlation_reports_pearson_and_spearman():
    result = audit.correlation_pair([1, 2, 3, 4], [4, 3, 2, 1])
    assert result["n"] == 4
    assert np.isclose(result["pearson"], -1.0)
    assert np.isclose(result["spearman"], -1.0)


def test_wrist_target_identity_is_associated_only_with_its_own_temporal_anchor():
    first_mask = _mask()
    first = segmentation_from_response(_response(first_mask), first_mask.shape)
    wrist_anchor = make_target_identity_anchor(first, target_phrase="salad dressing", frame_id=1)
    wrong = np.zeros_like(first_mask)
    wrong[0:2, 0:2] = True
    response = {"success": True, "details": {"metadata": {"image_size": [10, 8]},
        "detections": [{"score": 0.99, "mask": _response(wrong)["details"]["detections"][0]["mask"]},
                       {"score": 0.4, "mask": _response(first_mask)["details"]["detections"][0]["mask"]}]}}
    associated, next_anchor = audit._selected_wrist_segmentation(response, first_mask.shape, wrist_anchor)
    assert associated.visible is True
    assert associated.identity_status == "SAME_TARGET"
    assert associated.selected_candidate_id == "1"
    assert next_anchor is not None


def test_contact_sheet_and_masks_are_written_in_the_matching_canonical_view(tmp_path):
    raw = np.zeros((8, 10, 3), dtype=np.uint8)
    canonical = np.flipud(raw).copy()
    segmentation = segmentation_from_response(_response(_mask()), (8, 10))
    frame = {"raw_image": raw, "image": canonical, "segmentation": segmentation}
    paths = audit._save_view_artifacts(
        frame=frame, raw_wrist=raw, wrist_canonical=canonical,
        wrist_segmentation=segmentation, wrist_metrics=audit.extract_wrist_metrics(segmentation, (8, 10)),
        directory=tmp_path, prefix="sample",
    )
    assert "agentview_canonical_overlay" in paths
    assert "agentview_canonical_mask" in paths
    assert "agentview_raw_mask" not in paths
    assert np.array_equal(np.asarray(Image.open(paths["wrist_raw"])), raw)
    assert np.array_equal(np.asarray(Image.open(paths["wrist_canonical"])), canonical)
    assert np.asarray(Image.open(paths["wrist_mask"])).shape == raw.shape[:2]


def test_invisible_wrist_still_saves_an_empty_source_sized_mask(tmp_path):
    raw = np.zeros((8, 10, 3), dtype=np.uint8)
    invisible = TargetSegmentation(False, None, None, None, None, None,
                                   {"success": True, "details": {"detections": []}})
    paths = audit._save_view_artifacts(
        frame={"raw_image": raw, "image": raw, "segmentation": invisible},
        raw_wrist=raw, wrist_canonical=raw, wrist_segmentation=invisible,
        wrist_metrics={"center_error_normalized": None}, directory=tmp_path, prefix="empty",
    )
    assert paths["wrist_mask"] is not None
    mask = np.asarray(Image.open(paths["wrist_mask"]))
    assert mask.shape == raw.shape[:2]
    assert not mask.any()

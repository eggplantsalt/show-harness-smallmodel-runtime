from __future__ import annotations

import base64
import inspect
import io
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from core.runtime_v3.grounding import (
    CandidatePoolBuilder,
    EntityGroundingEvidence,
    GroundingCandidate,
    QwenSemanticCandidateSelector,
    QwenSemanticRegionProposer,
    SemanticGroundingBinder,
    SemanticGroundingResult,
    make_candidate_contact_sheet,
)
from core.runtime_v3.object_relative import ObjectRelativePerceptionObserver
from core.runtime_v3.state import BeliefState, RuntimeEntityState, StateBuilder
from core.vlm.vlm_client import VLMResponse


ROOT = Path(__file__).resolve().parents[2]


def _candidate(candidate_id="C0", *, x=10, size=12, source="text_sam", score=0.8):
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:10 + size, x:x + size] = True
    return GroundingCandidate(candidate_id, mask, (x, 10, x + size, 10 + size), score, source)


class _Qwen:
    model = "Qwen/mock"

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def complete_json(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        answer = self.answers.pop(0)
        raw = answer if isinstance(answer, str) else json.dumps(answer)
        return VLMResponse("", raw, {"usage": {"prompt_tokens": 4}})


def test_candidate_pool_api_has_no_task_or_object_specific_input():
    parameters = inspect.signature(CandidatePoolBuilder.build).parameters
    assert set(parameters) == {"self", "candidates", "image_shape"}
    source = inspect.getsource(CandidatePoolBuilder)
    for name in ("task_id", "butter", "milk", "alphabet soup", "object_name"):
        assert name not in source.casefold()


def test_same_pool_inputs_produce_identical_candidates_without_task_controls():
    inputs = (_candidate("C7"), _candidate("C8", x=36))
    first = CandidatePoolBuilder().build(inputs, image_shape=(64, 64))
    second = CandidatePoolBuilder().build(inputs, image_shape=(64, 64))
    assert [item.candidate_id for item in first] == ["C0", "C1"]
    assert [item.candidate_id for item in first] == [item.candidate_id for item in second]
    assert all(np.array_equal(a.mask, b.mask) for a, b in zip(first, second))


def test_duplicate_text_and_region_masks_are_generically_deduplicated_and_keep_both_sources():
    pool = CandidatePoolBuilder().build(
        (_candidate("C8", source="text_sam"),
         _candidate("C2", source="qwen_region_sam_point", score=0.7)),
        image_shape=(64, 64),
    )
    assert len(pool) == 1
    assert pool[0].source == "both"
    assert set(pool[0].sources) == {"text_sam", "qwen_region_sam_point"}


def test_pool_filters_are_shared_normalized_and_retain_small_objects():
    small = _candidate("C8", size=3)
    large = _candidate("C9", size=62)
    kept = CandidatePoolBuilder().build((small, large), image_shape=(64, 64))
    assert len(kept) == 1
    assert kept[0].mask.sum() == 9


def test_final_candidate_ids_are_synthetic_and_carry_no_simulator_metadata():
    pool = CandidatePoolBuilder().build((_candidate("C17"),), image_shape=(64, 64))
    assert pool[0].candidate_id == "C0"
    assert re.fullmatch(r"C\d+", pool[0].candidate_id)
    assert not any(token in pool[0].candidate_id.casefold()
                   for token in ("body", "object", "instance", "segmentation"))


def test_selector_returns_only_an_existing_candidate_id():
    model = _Qwen([{"decision": "SELECT", "candidate_id": "C0"}])
    selected, decision = QwenSemanticCandidateSelector(model).select(
        np.zeros((64, 64, 3), dtype=np.uint8),
        make_candidate_contact_sheet(np.zeros((64, 64, 3), dtype=np.uint8), [_candidate()]),
        [_candidate()], task_instruction="Pick the item", semantic_phrase="item",
    )
    assert (selected, decision) == ("C0", "SELECT")


def test_selector_accepts_explicit_no_match():
    model = _Qwen([{"decision": "NO_MATCH", "candidate_id": None}])
    selected, decision = QwenSemanticCandidateSelector(model).select(
        np.zeros((64, 64, 3), dtype=np.uint8), np.zeros((64, 64, 3), dtype=np.uint8),
        [_candidate()], task_instruction="Pick item", semantic_phrase="item",
    )
    assert selected is None and decision == "NO_MATCH"


def test_selector_fails_closed_for_unknown_candidate_id():
    model = _Qwen([{"decision": "SELECT", "candidate_id": "C99"}])
    selected, decision = QwenSemanticCandidateSelector(model).select(
        np.zeros((64, 64, 3), dtype=np.uint8), np.zeros((64, 64, 3), dtype=np.uint8),
        [_candidate()], task_instruction="Pick item", semantic_phrase="item",
    )
    assert selected is None and decision == "INVALID"


def test_selector_fails_closed_on_invalid_json_without_retry():
    model = _Qwen(["not json"])
    selector = QwenSemanticCandidateSelector(model)
    selected, decision = selector.select(
        np.zeros((64, 64, 3), dtype=np.uint8), np.zeros((64, 64, 3), dtype=np.uint8),
        [_candidate()], task_instruction="Pick item", semantic_phrase="item",
    )
    assert selected is None and decision == "INVALID"
    assert len(model.calls) == 1


def test_selector_schema_and_prompt_forbid_direction_scale_and_robot_action():
    model = _Qwen([{"decision": "SELECT", "candidate_id": "C0"}])
    QwenSemanticCandidateSelector(model).select(
        np.zeros((64, 64, 3), dtype=np.uint8), np.zeros((64, 64, 3), dtype=np.uint8),
        [_candidate()], task_instruction="Pick item", semantic_phrase="item",
    )
    call = model.calls[0]
    assert call["schema"]["additionalProperties"] is False
    assert set(call["schema"]["properties"]) == {"decision", "candidate_id"}
    assert "direction" in call["prompt"] and "scale" in call["prompt"]
    assert "robot action" in call["prompt"]
    assert call["wrist_image"] is not None


def test_selector_has_no_arbiter_or_executor_reference():
    selector = QwenSemanticCandidateSelector(_Qwen([]))
    assert not hasattr(selector, "arbiter")
    assert not hasattr(selector, "executor")
    assert not hasattr(selector, "controller")


def test_region_proposer_output_is_bounded_normalized_and_semantic_only():
    model = _Qwen([{"decision": "PROPOSE", "regions": [
        {"bbox_norm_1000": [100, 200, 500, 800]},
    ]}])
    proposer = QwenSemanticRegionProposer(model)
    regions = proposer.propose(
        np.zeros((64, 64, 3), dtype=np.uint8), task_instruction="Pick item",
        semantic_phrase="item",
    )
    assert regions == ((0.1, 0.2, 0.5, 0.8),)
    call = model.calls[0]
    assert call["schema"]["additionalProperties"] is False
    assert "robot, motion, action" in call["prompt"]


def test_region_proposer_invalid_boxes_fail_closed_without_retry():
    model = _Qwen([{"decision": "PROPOSE", "regions": [
        {"bbox_norm_1000": [800, 100, 100, 900]},
    ]}])
    proposer = QwenSemanticRegionProposer(model)
    assert proposer.propose(
        np.zeros((64, 64, 3), dtype=np.uint8), task_instruction="Pick item",
        semantic_phrase="item",
    ) == ()
    assert len(model.calls) == 1


def test_oracle_target_segmentation_is_absent_from_generation_and_selection_interfaces():
    assert set(inspect.signature(CandidatePoolBuilder.build).parameters) == {
        "self", "candidates", "image_shape",
    }
    assert "oracle" not in str(inspect.signature(QwenSemanticCandidateSelector.select)).casefold()
    assert "segmentation_gt" not in str(inspect.signature(SemanticGroundingBinder.ground)).casefold()


def test_full_scene_context_is_retained_in_each_candidate_tile():
    image = np.full((64, 64, 3), 117, dtype=np.uint8)
    sheet = make_candidate_contact_sheet(image, [_candidate()])
    assert sheet.shape[0] >= 64
    assert sheet.shape[1] >= 64
    assert np.any(sheet == 117)


def test_saved_candidate_sheet_marks_qwen_choice_without_changing_selector_input():
    image = np.full((64, 64, 3), 117, dtype=np.uint8)
    selected_sheet = make_candidate_contact_sheet(
        image, [_candidate()], selected_candidate_id="C0", selection_decision="SELECT",
    )
    assert selected_sheet.shape[0] > image.shape[0]
    assert tuple(selected_sheet[32, 0]) == (30, 170, 70)
    empty_sheet = make_candidate_contact_sheet(image, (), selection_decision="NO_MATCH")
    assert empty_sheet.shape[0] == image.shape[0] + 32


class _Base:
    def __init__(self):
        self.index = 0
        self.last_raw = SimpleNamespace(eef_position_xyz=(0.0, 0.0, 0.2))

    def observe(self, _env):
        from core.runtime_v3.observer import RobotObservation
        self.index += 1
        return RobotObservation(
            f"obs-{self.index}", self.index,
            images={"agentview": np.zeros((64, 64, 3), dtype=np.uint8)},
            evidence={"relevant_geometry": {"workspace_z_bounds_m": [0.01, 0.6]}},
            fresh=True,
        )


class _Sam:
    def __init__(self):
        self.calls = 0

    def segment(self, *_args, **_kwargs):
        self.calls += 1
        return {"success": False, "details": {"detections": []}}


class _NoMatchBinder:
    def __init__(self):
        self.calls = 0

    def ground(self, _image, *, entity_key, semantic_phrase, semantic_query):
        self.calls += 1
        evidence = EntityGroundingEvidence(
            entity_key, semantic_query, None, "NO_MATCH", False, "semantic_selector",
            invalid_reason="SEMANTIC_SELECTION_NO_MATCH", candidate_count=1, agent_calls=2,
        )
        return SemanticGroundingResult(
            (_candidate(),), None, evidence, {"success": True}, None, "", "",
            {"raw_proposals": 1, "filtered_proposals": 1, "final_candidates": 1},
        )


class _SelectedBinder:
    def __init__(self):
        self.calls = 0

    def ground(self, _image, *, entity_key, semantic_phrase, semantic_query):
        self.calls += 1
        candidate = _candidate("C0")
        evidence = EntityGroundingEvidence(
            entity_key, semantic_query, "C0", "SELECT", True, "qwen_region_sam_point",
            candidate_count=1, proposal_region_count=1, agent_calls=2,
        )
        return SemanticGroundingResult(
            (candidate,), candidate, evidence, {"success": True}, None, "", "",
            {"raw_proposals": 1, "filtered_proposals": 1, "final_candidates": 1},
        )


class _PointSam(_Sam):
    def __init__(self):
        super().__init__()
        self.point_calls = 0

    def segment_points(self, image, points):
        self.point_calls += 1
        assert points and points[0]["label"] == 1
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[10:22, 10:22] = 255
        with io.BytesIO() as buffer:
            Image.fromarray(mask, mode="L").save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {"success": True, "details": {
            "metadata": {"image_size": [image.shape[1], image.shape[0]]},
            "detections": [{"mask": {"base64": encoded}, "score": 0.9}],
        }}


def test_stable_mask_without_semantic_selection_does_not_anchor_runtime_entity():
    binder = _NoMatchBinder()
    observer = ObjectRelativePerceptionObserver(
        _Base(), _Sam(), target_phrase="item", move_vectors={},
        semantic_grounding_binder=binder,
    )
    observer.observe(SimpleNamespace())
    observer.observe(SimpleNamespace())
    assert binder.calls == 1
    assert observer.identity_anchor is None
    assert observer.reference_anchor is None
    assert not observer.last_segmentation.visible


def test_correct_semantic_candidate_remains_invalid_until_runtime_reference_verification():
    evidence = EntityGroundingEvidence(
        "target", "item", "C0", "SELECT", True, "qwen_region_sam_point",
        candidate_count=1, proposal_region_count=1, agent_calls=2,
    )
    entity = RuntimeEntityState(
        "target", "item", "MANIPULAND", identity_anchor=object(), visible=True,
        valid=False, grounding_evidence=evidence,
    )
    assert entity.grounding_evidence is evidence
    assert not entity.valid


def test_grounding_evidence_round_trips_inside_the_existing_belief_state():
    evidence = EntityGroundingEvidence(
        "target", "item", "C0", "SELECT", True, "both", candidate_count=2,
        proposal_region_count=1, agent_calls=2,
    )
    state = BeliefState(runtime_entity_state=RuntimeEntityState(
        "target", "item", "MANIPULAND", grounding_evidence=evidence,
    ))
    assert state.runtime_entity_state.grounding_evidence is evidence
    rebuilt = StateBuilder().update(BeliefState(), SimpleNamespace(
        evidence={"runtime_entity_state": {
            "entity_key": "target", "semantic_phrase": "item", "role": "MANIPULAND",
            "visible": True, "valid": False,
            "grounding_evidence": {
                "entity_key": "target", "semantic_query": "item", "candidate_id": "C0",
                "decision": "SELECT", "valid": True, "source": "both",
                "candidate_count": 2, "proposal_region_count": 1, "agent_calls": 2,
            },
        }},
        proprioception={}, evidence_refs=(), observation_id="obs", frame_id=1,
        fresh=True, done=False,
    ))
    assert rebuilt.runtime_entity_state.grounding_evidence == evidence


def test_semantic_selection_is_one_time_during_entity_initialization():
    binder = _NoMatchBinder()
    observer = ObjectRelativePerceptionObserver(
        _Base(), _Sam(), target_phrase="item", move_vectors={},
        semantic_grounding_binder=binder,
    )
    for _ in range(4):
        observer.observe(SimpleNamespace())
    assert binder.calls == 1


def test_semantic_proposals_wait_until_scene_motion_is_ready():
    binder, sam = _SelectedBinder(), _PointSam()
    observer = ObjectRelativePerceptionObserver(
        _Base(), sam, target_phrase="item", move_vectors={},
        semantic_grounding_binder=binder, scene_ready_required=True,
    )
    for _ in range(3):
        observer.observe(SimpleNamespace())
    assert not observer.scene_motion_ready
    assert binder.calls == 0
    observer.observe(SimpleNamespace())
    assert observer.scene_motion_ready
    assert binder.calls == 1


def test_selected_candidate_waits_for_entity_observation_ready_before_reference():
    binder, sam = _SelectedBinder(), _PointSam()
    observer = ObjectRelativePerceptionObserver(
        _Base(), sam, target_phrase="item", move_vectors={},
        semantic_grounding_binder=binder, scene_ready_required=True,
    )
    model = SimpleNamespace(camera_name2id=lambda _name: 0, cam_fovy=[60.0])
    data = SimpleNamespace(cam_xpos=[np.zeros(3)], cam_xmat=[np.eye(3).reshape(-1)])
    environment = SimpleNamespace(sim=SimpleNamespace(model=model, data=data))
    observations = [observer.observe(environment) for _ in range(5)]
    assert binder.calls == 1
    assert not observer.entity_observation_ready
    assert observer.reference_anchor is None
    assert not observations[-1].evidence["runtime_entity_state"].valid
    ready = observer.observe(environment)
    assert observer.entity_observation_ready
    assert observer.reference_valid
    assert ready.evidence["runtime_entity_state"].valid


def test_runtime_tracks_semantically_selected_entity_with_sam_points_without_qwen_retry():
    binder, sam = _SelectedBinder(), _PointSam()
    observer = ObjectRelativePerceptionObserver(
        _Base(), sam, target_phrase="item", move_vectors={},
        semantic_grounding_binder=binder,
    )
    observer.observe(SimpleNamespace())
    observer.observe(SimpleNamespace())
    assert binder.calls == 1
    assert sam.point_calls == 1
    assert observer.last_segmentation.identity_status == "SAME_TARGET"
    assert observer.last_segmentation.visible


def test_explicit_reground_starts_a_new_bounded_semantic_initialization():
    binder = _NoMatchBinder()
    observer = ObjectRelativePerceptionObserver(
        _Base(), _Sam(), target_phrase="item", move_vectors={},
        semantic_grounding_binder=binder,
    )
    observer.observe(SimpleNamespace())
    observer.request_target_reference_reground()
    observer.observe(SimpleNamespace())
    assert binder.calls == 2


def test_grounding_service_retains_no_physical_runtime_authority():
    binder = SemanticGroundingBinder(
        qwen_client=object(), sam_client=object(), task_instruction="Pick item",
    )
    assert not hasattr(binder, "arbiter")
    assert not hasattr(binder, "executor")
    assert not hasattr(binder, "controller")


def test_grounding_query_normalizer_remains_generic():
    from core.runtime_v3.grounding_query import GroundingQueryNormalizer
    normalize = GroundingQueryNormalizer().normalize
    assert normalize("The alphabet soup") == "alphabet soup"
    assert normalize("A cream cheese") == "cream cheese"
    assert normalize("An object") == "object"


def test_alignment_physical_geometry_implementation_is_unchanged():
    current = (ROOT / "core/runtime_v3/object_relative.py").read_text(encoding="utf-8")
    previous = subprocess.run(
        ["git", "show", "HEAD:core/runtime_v3/object_relative.py"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout
    pattern = re.compile(r"def resolve_object_relative_geometry\([\s\S]*?(?=\ndef make_alignment_option\()")
    assert pattern.search(current).group() == pattern.search(previous).group()


def test_one_physical_semantic_action_still_uses_one_arbiter_approval():
    source = (ROOT / "scripts/runtime_v3_object_relative_alignment.py").read_text()
    assert "class CountingArbiter" in source
    assert "authorization_calls" in source
    assert "pre_action_ready_written" in source

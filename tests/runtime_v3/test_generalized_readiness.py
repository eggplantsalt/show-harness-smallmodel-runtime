from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from core.runtime_v3.grounding_query import GroundingQueryNormalizer
from core.runtime_v3.readiness import EntityObservationReady, SceneMotionReady


ROOT = Path(__file__).resolve().parents[2]


def _mask(size: int, *, x: int = 20, y: int = 20, shape=(128, 128)) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    result[y:y + size, x:x + size] = True
    return result


def _entity_update(gate, *, mask, entity="target", query="package", status="SAME_TARGET", tick=0):
    ys, xs = np.nonzero(mask)
    return gate.update(
        entity_key=entity, grounding_query=query, identity_status=status,
        candidate_id="same-instance", mask=mask,
        centroid_px=(float(xs.mean()), float(ys.mean())),
        bbox_xyxy=(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
        mask_area_px=int(mask.sum()), frame_id=tick,
    )


def test_scene_motion_input_is_canonical_rgb_only_with_no_hidden_signals():
    parameters = inspect.signature(SceneMotionReady.update).parameters
    assert list(parameters) == ["self", "canonical_rgb"]
    assert set(parameters) == {"self", "canonical_rgb"}


def test_same_rgb_sequence_has_the_same_scene_motion_result_independent_of_phrase():
    frames = [np.full((32, 32, 3), value, dtype=np.uint8) for value in (0, 0, 0, 0)]
    outputs = []
    for _phrase in ("alphabet soup", "the alphabet soup", "butter"):
        gate = SceneMotionReady()
        outputs.append([gate.update(frame) for frame in frames])
    assert outputs == [[False, False, False, True]] * 3


def test_scene_motion_threshold_is_shared_and_frozen_from_development_rgb():
    gate = SceneMotionReady()
    record = gate.to_record()
    assert record["max_normalized_rgb_difference"] == 0.00005
    assert record["required_stable_frame_pairs"] == 3
    assert record["roi"] is None
    assert "stable-tail full-frame" in record["threshold_source"]


def test_entity_readiness_uses_normalized_geometry_for_different_object_sizes():
    for size in (8, 48):
        gate = EntityObservationReady()
        assert [_entity_update(gate, mask=_mask(size), tick=tick) for tick in range(3)] == [
            False, False, True,
        ]


def test_centroid_shift_is_normalized_by_bbox_diagonal():
    small = EntityObservationReady()
    _entity_update(small, mask=_mask(20), tick=0)
    moved_small = _mask(20, x=21)
    assert not _entity_update(small, mask=moved_small, tick=1)
    assert small.last_failure == "ENTITY_OBSERVATION_NOT_READY"
    assert small.last_interval["centroid_shift_over_previous_bbox_diagonal"] > 0.02

    large = EntityObservationReady()
    _entity_update(large, mask=_mask(100), tick=0)
    moved_large = _mask(100, x=21)
    assert not _entity_update(large, mask=moved_large, tick=1)
    assert large.last_interval["centroid_shift_over_previous_bbox_diagonal"] < 0.02


def test_small_mask_jitter_does_not_feed_scene_motion_gate():
    motion = SceneMotionReady()
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    for _ in range(4):
        motion.update(frame)
    assert motion.ready
    entity = EntityObservationReady()
    assert not _entity_update(entity, mask=_mask(48), tick=0)
    assert not _entity_update(entity, mask=_mask(48, x=21), tick=1)
    assert entity.last_interval["adjacent_mask_iou"] > 0.5
    assert motion.ready


def test_identity_switch_clears_entity_observation_window_even_after_ready():
    gate = EntityObservationReady()
    mask = _mask(24)
    for tick in range(3):
        _entity_update(gate, mask=mask, tick=tick)
    assert gate.ready
    assert not _entity_update(gate, mask=mask, status="TARGET_IDENTITY_LOST", tick=3)
    assert not gate.ready
    assert gate.last_failure == "IDENTITY_FAILURE"
    assert gate.to_record()["stable_observation_count"] == 0


def test_reference_window_resets_on_entity_or_query_change_and_keeps_one_identity():
    gate = EntityObservationReady()
    mask = _mask(24)
    for tick in range(2):
        _entity_update(gate, mask=mask, tick=tick)
    assert not _entity_update(gate, mask=mask, entity="other", tick=2)
    assert gate.to_record()["entity_key"] == "other"
    assert gate.to_record()["stable_observation_count"] == 1
    assert not _entity_update(gate, mask=mask, entity="other", query="other package", tick=3)
    assert gate.to_record()["grounding_query"] == "other package"
    assert gate.to_record()["stable_observation_count"] == 1


def test_robust_representative_is_a_medoid_and_uses_only_its_entity_window():
    gate = EntityObservationReady()
    for tick, x in enumerate((20, 21, 20)):
        _entity_update(gate, mask=_mask(48, x=x), tick=tick)
    assert gate.ready
    representative = gate.representative_sample()
    assert representative["entity_key"] == "target"
    assert representative["grounding_query"] == "package"
    assert representative["reference_centroid_px"] == [43.5, 43.5]
    assert representative["frame_id"] in (0, 1, 2)


def test_grounding_query_normalizer_is_generic_and_preserves_raw_task_phrase():
    from core.runtime_v3.task_spec import ReferenceTaskCompiler

    normalizer = GroundingQueryNormalizer()
    assert normalizer.normalize("  THE   Alphabet Soup ") == "alphabet soup"
    assert normalizer.normalize("a cream cheese") == "cream cheese"
    assert normalizer.normalize("an object") == "object"
    assert normalizer.normalize("the") == "the"
    assert normalizer.normalize("milk") == "milk"
    task = ReferenceTaskCompiler().compile({
        "instruction": "Pick the cream cheese",
        "entities": [{"key": "target", "semantic_phrase": "The cream cheese",
                      "role": "MANIPULAND"}],
        "focus_entity_key": "target", "goal_kind": "ALIGN",
    })
    assert task.focus_entity.semantic_phrase == "The cream cheese"
    assert normalizer.normalize(task.focus_entity.semantic_phrase) == "cream cheese"


def test_physical_runtime_branches_contain_no_selected_task_names_or_ids():
    object_relative = (ROOT / "core/runtime_v3/object_relative.py").read_text(encoding="utf-8").casefold()
    readiness = (ROOT / "core/runtime_v3/readiness.py").read_text(encoding="utf-8").casefold()
    for value in (object_relative, readiness):
        for name in ("alphabet soup", "salad dressing", "butter", "milk", "cream cheese", "chocolate pudding"):
            assert name not in value
    assert "task_id" not in inspect.getsource(SceneMotionReady)


def test_alignment_generator_and_arbiter_stay_outside_readiness_modules():
    source = (ROOT / "core/runtime_v3/readiness.py").read_text(encoding="utf-8")
    assert "OptionGenerator" not in source
    assert "Arbiter" not in source
    assert "Executor" not in source


def test_single_action_approval_and_qwen_semantic_only_contract_remain_in_force():
    from core.runtime_v3.task_spec import QwenTaskCompiler, ReferenceTaskCompiler
    from scripts.runtime_v3_object_relative_alignment import CountingArbiter

    assert "authorization_calls" in inspect.getsource(CountingArbiter)
    assert "pre_action_ready_written" in inspect.getsource(CountingArbiter.authorize)
    assert "_FORBIDDEN_FIELDS" in inspect.getsource(ReferenceTaskCompiler)
    assert "_ENTITY_FIELDS" in inspect.getsource(QwenTaskCompiler)

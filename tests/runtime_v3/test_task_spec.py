from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from core.runtime_v3.object_relative import MultiScaleAlignmentOptionGenerator
from core.runtime_v3.state import BeliefState, ObjectRelativeState, RuntimeEntityState, StateBuilder
from core.runtime_v3.task_spec import EntitySpec, GoalKind, ReferenceTaskCompiler, TaskSpec
from scripts.runtime_v3_cross_object_alignment import _failure_layer


def _bound_state(key: str, phrase: str, direction: str = "DOWN") -> BeliefState:
    entity = RuntimeEntityState(key, phrase, "MANIPULAND", visible=True, valid=True)
    relative = ObjectRelativeState(
        target_phrase=phrase, target_visible=True, target_centroid_px=(120.0, 90.0),
        eef_projection_px=(140.0, 110.0), target_reference_point_px=(120.0, 90.0),
        target_reference_valid=True, target_identity_status="ANCHORED",
    )
    geometry = {
        "camera_projection_valid": True,
        "workspace_valid": True,
        "multiscale_alignment_valid": True,
        "pixel_error_before_px": 28.2842712475,
        "chosen_lattice_candidate": {
            "direction": direction, "direction_unit": (0.0, 0.0, -1.0),
            "displacement_m": 0.006, "max_ticks": 7,
            "predicted_error_px": 20.0, "predicted_improvement_px": 8.2842712475,
            "valid": True, "scale_contract_valid": True, "workspace_valid": True,
        },
    }
    return BeliefState(
        task_id="benchmark:fixture", frame_id=3, runtime_entity_state=entity,
        object_relative_state=relative, relevant_geometry=geometry,
        evidence_refs=("frame:3",),
    )


def test_task_spec_and_entity_spec_are_immutable_and_validated():
    entity = EntitySpec("target", "milk", "MANIPULAND")
    task = TaskSpec("Pick the milk and place it in the basket", (entity,), "target")
    assert task.goal_kind is GoalKind.ALIGN
    assert task.focus_entity is entity
    with pytest.raises(FrozenInstanceError):
        entity.semantic_phrase = "butter"
    with pytest.raises(ValueError, match="focus_entity_key"):
        TaskSpec("Align", (entity,), "basket")


def test_entity_spec_has_no_task_id_or_physical_fields():
    names = {item.name for item in fields(EntitySpec)}
    assert names == {"key", "semantic_phrase", "role"}
    left = EntitySpec("target", "milk", "MANIPULAND")
    right = EntitySpec("target", "butter", "MANIPULAND")
    assert left.key == right.key


def test_runtime_entity_state_binds_different_semantic_objects_in_one_belief_schema():
    builder = StateBuilder()
    for phrase in ("alphabet soup", "butter", "milk"):
        state = builder.update(
            builder.initialize("benchmark:fixture"),
            type("Observation", (), {"evidence": {"runtime_entity_state": {
                "entity_key": "target", "semantic_phrase": phrase,
                "role": "MANIPULAND", "visible": True, "valid": True,
            }}, "proprioception": {}, "evidence_refs": (), "fresh": True,
              "observation_id": "obs", "frame_id": 1, "done": False})(),
        )
        assert state.runtime_entity_state is not None
        assert state.runtime_entity_state.semantic_phrase == phrase
        assert state.runtime_entity_state.valid


def test_reference_task_compiler_accepts_semantics_and_rejects_physical_fields():
    compiler = ReferenceTaskCompiler()
    spec = compiler.compile({
        "instruction": "Pick the milk and place it in the basket",
        "entities": [{"key": "target", "semantic_phrase": "milk", "role": "MANIPULAND"}],
        "focus_entity_key": "target", "goal_kind": "ALIGN",
    })
    assert spec.focus_entity.semantic_phrase == "milk"
    with pytest.raises(ValueError, match="physical fields"):
        compiler.compile({
            "instruction": "Align milk", "entities": [{
                "key": "target", "semantic_phrase": "milk", "role": "MANIPULAND",
                "direction": "DOWN",
            }], "focus_entity_key": "target",
        })


def test_runtime_alignment_option_uses_entity_key_and_same_contract_for_multiple_entities():
    generator = MultiScaleAlignmentOptionGenerator()
    options = [generator.generate(_bound_state("target", phrase)) for phrase in
               ("alphabet soup", "butter", "milk")]
    assert all(len(items) == 1 for items in options)
    realized = [items[0] for items in options]
    motions = [item.primitive.micro_motion_spec for item in realized]
    assert all(motion.direction == "DOWN" and motion.requested_displacement_m == 0.006
               and motion.max_ticks == 7 for motion in motions)
    assert all(item.expected_effect["entity_key"] == "target" for item in realized)
    assert all("target_phrase" not in item.expected_effect for item in realized)


def test_runtime_physical_modules_have_no_selected_task_or_object_special_cases():
    root = Path(__file__).resolve().parents[2]
    physical_paths = (
        "core/runtime_v3/object_relative.py", "core/runtime_v3/micro_motion.py",
        "core/runtime_v3/options.py", "core/runtime_v3/arbiter.py",
        "core/runtime_v3/executor.py", "core/runtime_v3/effects.py",
    )
    forbidden = (
        "LIBERO_OBJECT:2", "salad dressing", "salad_dressing", "alphabet soup",
        "alphabet_soup", "if task_id ==", "if object_name ==",
    )
    for relative in physical_paths:
        source = (root / relative).read_text(encoding="utf-8").casefold()
        assert not [value for value in forbidden if value.casefold() in source], relative


def test_manifest_is_frozen_and_contains_anchor_plus_three_unseen_objects():
    root = Path(__file__).resolve().parents[2]
    import json

    manifest = json.loads((root / "experiments/runtime_v3/cross_object_align_manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["selection_frozen_before_rollout"] is True
    assert [row["task_id"] for row in manifest["tasks"]] == [2, 0, 6, 7]
    assert len({row["reference_entity_phrase"] for row in manifest["tasks"]}) == 4
    assert all(row["role"] == "MANIPULAND" for row in manifest["tasks"])


def test_cross_object_formal_runner_disables_oracle_and_keeps_it_out_of_runtime_adapter():
    root = Path(__file__).resolve().parents[2]
    driver = (root / "scripts/runtime_v3_cross_object_alignment.py").read_text(
        encoding="utf-8")
    adapter = (root / "core/runtime_v3/adapters/libero_observation.py").read_text(
        encoding="utf-8").casefold()
    assert "diagnostic_oracle=False" in driver
    assert "target_world_position" not in adapter
    assert "check_success" not in adapter
    assert "camera_depths" not in adapter


def test_formal_per_step_trace_keeps_semantic_and_physical_evidence_together():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/runtime_v3_multiscale_alignment.py").read_text(
        encoding="utf-8")
    for field in (
        '"task_id": task_id', '"task_instruction": environment.task_description',
        '"entity_phrase": entity_spec.semantic_phrase', '"target_visible"',
        '"identity_valid"', '"reference_valid"', '"scene_ready"',
        '"selected_direction"', '"selected_scale_m"',
        '"episode_initial_alignment_error_px"', '"predicted_improvement_px"',
        '"actual_improvement_px"', '"effect_verification_success"',
        '"approval_trace_id"', '"semantic_step"', '"termination_reason"',
    ):
        assert field in runner


def test_failure_taxonomy_keeps_scene_and_perception_failures_distinct():
    assert _failure_layer("SceneReady failed after 40 holds") == "SCENE_NOT_READY"
    assert _failure_layer("SAM3 segmentation failed") == "PERCEPTION_FAILURE"
    assert _failure_layer("bounded Executor execution failed") == "PHYSICAL_EXECUTION_FAILURE"
    assert _failure_layer("TRIAL_FAILED") == "UNCLASSIFIED_FAILURE"

from scripts.run_robolab_zeroshot import _robolab_observation_context
from scripts.run_libero_zeroshot import _libero_context_extra
from core.sim.zeroshot_robolab_runner import (
    ZeroshotRobolabRunner,
    _reset_runtime_episode,
    _save_runtime_memory_frame,
    _should_end_after_verified_grasp,
)


def test_grasp_only_mode_exits_only_after_v22_runtime_verification() -> None:
    class Runtime:
        placement_v22_enabled = True
        grasp_verification_only = True

    class LegacyRuntime:
        placement_v22_enabled = False
        grasp_verification_only = True

    assert _should_end_after_verified_grasp(Runtime(), True)
    assert not _should_end_after_verified_grasp(Runtime(), False)
    assert not _should_end_after_verified_grasp(LegacyRuntime(), True)


def test_v22_observation_context_has_no_object_category_rule() -> None:
    context = _robolab_observation_context(
        include_robolab_context=True,
        placement_v22=True,
    ).lower()

    assert "orange" not in context
    assert "fruit" not in context
    assert "chosen target" in context
    assert "agentview" in context


def test_legacy_observation_context_keeps_existing_object_advice() -> None:
    context = _robolab_observation_context(
        include_robolab_context=True,
        placement_v22=False,
    ).lower()

    assert "clearly visible, round orange" in context


def test_disabled_robolab_context_stays_empty() -> None:
    assert not _robolab_observation_context(
        include_robolab_context=False,
        placement_v22=True,
    )


def test_runtime_memory_raw_views_are_saved_before_references_are_created(tmp_path) -> None:
    import numpy as np
    from PIL import Image

    from core.runtime_v2 import VerifiedCapabilityRuntime
    from core.runtime_v2.placement_memory import build_visual_memory_panel

    class Logger:
        def __init__(self):
            self.run_dir = tmp_path
            self.saved = []

        def save_visual_artifacts(
            self, frame_id, *, raw_agentview, raw_wrist, provider_overlay
        ):
            self.saved.append((frame_id, provider_overlay))
            for camera, image in (("agentview", raw_agentview), ("wrist", raw_wrist)):
                path = tmp_path / f"images/raw_{camera}/{frame_id:04d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(path)

    class Runtime:
        visual_memory = object()

    logger = Logger()
    runtime = VerifiedCapabilityRuntime()
    _reset_runtime_episode(runtime, logger)
    assert runtime.visual_memory.episode_id == str(tmp_path.resolve())
    image = np.full((32, 32, 3), 80, dtype=np.uint8)

    # This is called by the runner before observe_frame. The subsequent runtime
    # write can therefore only point to already-persisted raw pixels.
    for frame in (0, 1):
        refs, saved = _save_runtime_memory_frame(
            logger,
            Runtime(),
            frame,
            raw_agentview=image,
            raw_wrist=image,
        )
        assert saved is True
        result = runtime.observe_frame(
            stage="APPROACH",
            evidence={
                "frame_id": frame,
                "camera": "agentview",
                "stage": "APPROACH",
                "target": "object",
                "bbox_xyxy": [8, 8, 20, 24],
                "confidence": 0.9,
                "source": "sam3",
                "visible": True,
                "geometry": {"target_minus_eef_px": [0, 0]},
            },
            previous_action=None,
            agentview=image,
            wrist=image,
            image_refs=refs,
            eef_xyz=(0.0, 0.0, 0.13),
        )
        assert result["event"]["visual_memory_refs"]

    bundle = runtime.visual_memory.placement_bundle(
        instance_id=runtime.belief.target.instance_id,
        grasp_epoch=runtime.belief.grasp_epoch,
    )
    panel, audit = build_visual_memory_panel(
        tmp_path,
        bundle,
        instance_id=runtime.belief.target.instance_id,
        grasp_epoch=runtime.belief.grasp_epoch,
        current_frame=1,
        episode_id=str(tmp_path.resolve()),
    )
    assert audit["frame_ids"] == [0, 1]
    assert panel.shape == (512, 512, 3)
    assert logger.saved == [(0, None), (1, None)]


def test_v21_runner_passes_resolved_memory_panel_into_pregrasp_role(tmp_path) -> None:
    from types import SimpleNamespace

    import numpy as np
    from PIL import Image

    class Resolver:
        def __init__(self):
            self.kwargs = None

        def resolve_pregrasp(self, **kwargs):
            self.kwargs = kwargs
            return {"selected": "UNKNOWN", "reason": "test abstention"}

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    memory = []
    for frame in (1, 2):
        for camera in ("agentview", "wrist"):
            path = tmp_path / f"images/raw_{camera}/{frame:04d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(path)
        memory.append({
            "instance_id": "target-1",
            "grasp_epoch": 0,
            "frame_id": frame,
            "episode_id": str(tmp_path.resolve()),
            "agentview_ref": f"images/raw_agentview/{frame:04d}.png",
            "wrist_ref": f"images/raw_wrist/{frame:04d}.png",
        })

    runner = object.__new__(ZeroshotRobolabRunner)
    resolver = Resolver()
    runner.task_description = "pick up the object"
    runner.controls = SimpleNamespace(controller=resolver)
    runner._images = lambda obs: (image, image)
    runner.verified_runtime = SimpleNamespace(placement_v22_enabled=False)
    runner.logger = SimpleNamespace(run_dir=tmp_path)
    runner.debug = False
    runner._previous_agentview = image
    runner._previous_wrist = image
    runner._previous_frame_id = 1
    runner._previous_executed_action = "MV_DOWN"
    runner._critical_qwen_input = None

    result = runner._resolve_runtime_critical_decision(
        request={
            "kind": "PREGRASP_DECISION",
            "belief": {
                "target": {"instance_id": "target-1"},
                "grasp_epoch": 0,
                "frame_id": 2,
            },
            "allowed_answers": ["REOBSERVE", "UNKNOWN"],
            "allowed_options": ["REOBSERVE"],
            "visual_memory_refs": [
                ref for item in memory for ref in (item["agentview_ref"], item["wrist_ref"])
            ],
            "visual_memory_bundle": memory,
            "evidence_frame_ids": [1, 2],
            "reflection_mode": "double",
            "reflection_trigger": "grasp_entry",
            "visual_alignment": {
                "valid": True,
                "frame_id": 2,
                "camera": "agentview",
                "target_minus_eef_px": [60.0, -25.0],
            },
            "reason": "inspect the frozen same-episode memory panel",
        },
        subgoal=SimpleNamespace(target="object", affordance="body"),
        obs={},
        agentview=image,
    )

    assert result["answer"] == "UNKNOWN"
    assert result["details"]["memory_audit"]["frame_ids"] == [1, 2]
    assert resolver.kwargs["reflection_mode"] == "double"
    assert resolver.kwargs["target_roi_included"] is False
    assert resolver.kwargs["visual_alignment"]["target_minus_eef_px"] == [60.0, -25.0]
    assert resolver.kwargs["prebuilt_memory_panel"].shape == (512, 512, 3)
    assert np.array_equal(runner._critical_qwen_input, resolver.kwargs["prebuilt_memory_panel"])


def test_v22_libero_planner_context_is_evidence_driven_and_object_agnostic() -> None:
    context = _libero_context_extra(placement_v22=True).lower()

    assert "green-capped" not in context
    assert "salad-dressing" not in context
    assert "repeat that same direction" not in context
    assert "measured effect" in context
    assert "feasible semantic options" in context


def test_v21_libero_planner_context_is_calibrated_without_scene_policy() -> None:
    context = _libero_context_extra(placement_v22=False).lower()

    assert "green-capped" not in context
    assert "salad-dressing" not in context
    assert "fixed action order" in context
    assert "same-episode visual memory" in context
    assert "screen-down maps to world +x" in context

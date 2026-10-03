"""Bounded HOLD-only initialization for independent readiness evidence."""

from __future__ import annotations

from typing import Any

from .temporal_calibration import run_v3_tick


def scene_ready_status(*, ready: bool, hold_ticks: int, max_hold_ticks: int) -> str:
    """Name the bounded visual-gate outcome without consulting diagnostics."""
    if int(hold_ticks) < 0 or int(max_hold_ticks) < 0:
        raise ValueError("SceneReady hold counts cannot be negative")
    if ready:
        return "SCENE_READY"
    if int(hold_ticks) >= int(max_hold_ticks):
        return "SCENE_READY_TIMEOUT"
    return "SCENE_READY_PENDING"


def run_scene_ready_holds(
    environment: Any,
    observer: Any,
    controller: Any,
    *,
    task_id: str,
    commanded_step_m: float,
    workspace_z_bounds_m: tuple[float, float],
    max_hold_ticks: int = 40,
) -> dict[str, Any]:
    """Collect scene-motion and entity-observation readiness using HOLD only.

    Formal inputs are canonical RGB and the observer's associated SAM evidence.
    Oracle pose/depth/contact and success state are absent from this interface.
    """
    motion_evidence = getattr(observer, "scene_motion_evidence", None)
    entity_evidence = getattr(observer, "entity_observation_evidence", None)
    base_observer = getattr(observer, "base_observer", None)
    if motion_evidence is None or entity_evidence is None or base_observer is None:
        raise ValueError("observer must enable both readiness gates and provide a base observer")
    if int(max_hold_ticks) < 0:
        raise ValueError("max_hold_ticks cannot be negative")
    observer.begin_readiness()
    samples: list[dict[str, Any]] = []
    initial = observer.observe(environment)
    samples.append(_readiness_sample(observer, initial, environment))
    if observer.scene_ready:
        return _result(observer, ready=True, hold_ticks=0, samples=samples, commands=[])

    commands: list[dict[str, Any]] = []
    for _ in range(int(max_hold_ticks)):
        before_step = int(environment.step_count)
        result = run_v3_tick(
            environment, base_observer, controller, task_id=task_id,
            token=None, direction_unit=None, commanded_step_m=commanded_step_m,
            reset=False, workspace_z_bounds_m=workspace_z_bounds_m,
        )
        if (int(result.get("actions", 0)) != 1
                or result.get("approved_action") != "CALIBRATION_HOLD"
                or not result.get("backend_execution")
                or int(environment.step_count) != before_step + 1):
            raise RuntimeError(f"Readiness initialization must execute exactly one HOLD: {result}")
        observation = observer.observe(environment)
        commands.append({
            "kind": "HOLD", "translation_command_count": 0,
            "approved_option": result["approved_action"],
            "environment_step": int(environment.step_count),
        })
        samples.append(_readiness_sample(observer, observation, environment))
        if observer.scene_ready:
            return _result(observer, ready=True, hold_ticks=len(commands),
                           samples=samples, commands=commands)
    result = _result(observer, ready=False, hold_ticks=len(commands),
                     samples=samples, commands=commands)
    result["reason"] = scene_ready_status(
        ready=False, hold_ticks=len(commands), max_hold_ticks=max_hold_ticks,
    )
    result["termination_reason"] = _failure_reason(observer)
    return result


def _readiness_sample(observer: Any, observation: Any, environment: Any) -> dict[str, Any]:
    motion = observer.scene_motion_evidence
    entity = observer.entity_observation_evidence
    return {
        "environment_step": int(environment.step_count),
        "frame_id": observation.frame_id,
        "scene_ready": bool(observer.scene_ready),
        "robot_ready": True,
        "scene_motion_ready": bool(observer.scene_motion_ready),
        "entity_observation_ready": bool(observer.entity_observation_ready),
        "grounding_success": bool(observer.grounding_success),
        "identity_valid": bool(observer.identity_valid),
        "reference_valid": bool(observer.reference_valid),
        "scene_motion_evidence": motion.to_record(),
        "entity_observation_evidence": entity.to_record(),
    }


def _failure_reason(observer: Any) -> str:
    if not observer.scene_motion_ready:
        return "SCENE_MOTION_NOT_READY"
    if not observer.grounding_success:
        return "SEMANTIC_GROUNDING_FAILURE"
    if not observer.identity_valid:
        return "IDENTITY_FAILURE"
    if not observer.entity_observation_ready:
        return "ENTITY_OBSERVATION_NOT_READY"
    if not observer.reference_valid:
        return "REFERENCE_FAILURE"
    return "READINESS_COMPLETE"


def _result(observer: Any, *, ready: bool, hold_ticks: int,
            samples: list[dict[str, Any]], commands: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ready": bool(ready),
        "robot_ready": True,
        "scene_motion_ready": bool(observer.scene_motion_ready),
        "entity_observation_ready": bool(observer.entity_observation_ready),
        "grounding_success": bool(observer.grounding_success),
        "identity_valid": bool(observer.identity_valid),
        "reference_valid": bool(observer.reference_valid),
        "hold_ticks": int(hold_ticks),
        "ticks_to_ready": int(hold_ticks) if ready else None,
        "termination_reason": "READINESS_COMPLETE" if ready else _failure_reason(observer),
        "samples": samples,
        "commands": commands,
    }

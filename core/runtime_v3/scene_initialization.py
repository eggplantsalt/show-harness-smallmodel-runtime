"""Bounded HOLD-only initialization until visual SceneReady evidence is met."""

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
    """Collect same-target SAM stability evidence while executing only HOLD.

    The observer's scene gate consumes canonical identity/mask geometry. Oracle
    pose and simulator contacts are deliberately absent from this interface.
    """
    evidence = getattr(observer, "scene_ready_evidence", None)
    base_observer = getattr(observer, "base_observer", None)
    if evidence is None or base_observer is None:
        raise ValueError("observer must enable scene_ready_required and provide a base observer")
    if int(max_hold_ticks) < 0:
        raise ValueError("max_hold_ticks cannot be negative")
    samples: list[dict[str, Any]] = []
    initial = observer.observe(environment)
    samples.append({
        "environment_step": int(environment.step_count),
        "frame_id": initial.frame_id,
        "scene_ready": bool(observer.scene_ready),
        "evidence": evidence.to_record(),
    })
    if observer.scene_ready:
        return {"ready": True, "hold_ticks": 0, "samples": samples, "commands": []}

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
            raise RuntimeError(f"SceneReady initialization must execute exactly one HOLD: {result}")
        observation = observer.observe(environment)
        commands.append({
            "kind": "HOLD", "translation_command_count": 0,
            "approved_option": result["approved_action"],
            "environment_step": int(environment.step_count),
        })
        samples.append({
            "environment_step": int(environment.step_count),
            "frame_id": observation.frame_id,
            "scene_ready": bool(observer.scene_ready),
            "evidence": evidence.to_record(),
        })
        if observer.scene_ready:
            return {"ready": True, "hold_ticks": len(commands), "samples": samples,
                    "commands": commands}
    return {"ready": False, "hold_ticks": len(commands), "samples": samples,
            "commands": commands,
            "reason": scene_ready_status(ready=False, hold_ticks=len(commands),
                                         max_hold_ticks=max_hold_ticks)}

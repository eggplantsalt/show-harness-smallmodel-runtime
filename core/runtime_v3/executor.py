"""The only Runtime V3 module that calls an atomic robot controller."""

from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from .arbiter import Arbiter, ApprovedAction


NEGATIVE_PROGRESS_EPSILON_M = 0.0001


class PrimitiveBackend(Protocol):
    def execute_approved_action(self, action: ApprovedAction) -> Any: ...

    def execute_approved_micro_tick(self, action: ApprovedAction) -> Any: ...


@dataclass(frozen=True)
class ExecutionRecord:
    option_id: str
    primitive_kind: str
    started_at: float
    finished_at: float
    result: Any = None


class Executor:
    def __init__(self, backend: PrimitiveBackend, arbiter: Arbiter) -> None:
        self.backend = backend
        self.arbiter = arbiter

    def execute(
        self,
        action: ApprovedAction,
        *,
        tick_observer: Callable[[], Any] | None = None,
    ) -> ExecutionRecord:
        if not self.arbiter.is_approved(action):
            raise TypeError("Executor accepts only an action authorized by its Arbiter")
        if action.primitive.max_steps != 1:
            raise ValueError("Executor accepts exactly one Arbiter-approved option")
        started = time.monotonic()
        if action.primitive.kind == "micro_motion":
            if tick_observer is None:
                raise ValueError("bounded micro-motion requires an observation after every control tick")
            result = self._execute_micro_motion(action, tick_observer)
        else:
            result = self.backend.execute_approved_action(action)
        return ExecutionRecord(
            option_id=action.option_id,
            primitive_kind=action.primitive.kind,
            started_at=started,
            finished_at=time.monotonic(),
            result=result,
        )

    def _execute_micro_motion(
        self, action: ApprovedAction, tick_observer: Callable[[], Any]
    ) -> dict[str, Any]:
        spec = action.primitive.micro_motion_spec
        start = action.start_eef_position_xyz_m
        bounds = action.workspace_z_bounds_m
        if spec is None or start is None or bounds is None:
            raise ValueError("approved micro-motion is missing its sealed spec or safety evidence")
        direction = spec.direction_unit
        previous = start
        cumulative_projection = 0.0
        ticks: list[dict[str, Any]] = []
        termination = "MAX_TICKS_REACHED"
        backend_receipt: Any = None
        started = time.monotonic()

        for tick_index in range(1, spec.max_ticks + 1):
            next_z = previous[2] + direction[2] * spec.control_tick_step_m
            if next_z < bounds[0] - 1e-12 or next_z > bounds[1] + 1e-12:
                termination = "BOUNDARY_STOP"
                break
            backend_receipt = self.backend.execute_approved_micro_tick(action)
            observation = tick_observer()
            position = _eef_position(observation)
            if position is None:
                raise RuntimeError("micro-motion tick observation has no finite EEF position")
            increment = tuple(position[index] - previous[index] for index in range(3))
            incremental_projection = _dot(increment, direction)
            total_delta = tuple(position[index] - start[index] for index in range(3))
            cumulative_projection = _dot(total_delta, direction)
            incremental_off_axis = tuple(
                increment[index] - incremental_projection * direction[index]
                for index in range(3)
            )
            cumulative_off_axis = tuple(
                total_delta[index] - cumulative_projection * direction[index]
                for index in range(3)
            )
            total_norm = _norm(total_delta)
            ticks.append({
                "tick": tick_index,
                "observation_id": getattr(observation, "observation_id", None),
                "frame_id": getattr(observation, "frame_id", None),
                "eef_position_xyz_m": list(position),
                "incremental_delta_xyz_m": list(increment),
                "incremental_projection_m": incremental_projection,
                "incremental_projection_mm": incremental_projection * 1000.0,
                "incremental_off_axis_xyz_m": list(incremental_off_axis),
                "incremental_off_axis_magnitude_m": _norm(incremental_off_axis),
                "total_displacement_xyz_m": list(total_delta),
                "cumulative_projection_m": cumulative_projection,
                "cumulative_projection_mm": cumulative_projection * 1000.0,
                "off_axis_displacement_xyz_m": list(cumulative_off_axis),
                "off_axis_magnitude_m": _norm(cumulative_off_axis),
                "direction_cosine": cumulative_projection / total_norm if total_norm > 1e-12 else None,
                "elapsed_s": time.monotonic() - started,
            })
            previous = position
            if incremental_projection < -NEGATIVE_PROGRESS_EPSILON_M:
                termination = "NEGATIVE_PROGRESS"
                break
            if cumulative_projection >= spec.requested_displacement_m:
                termination = "TARGET_REACHED"
                break

        total_delta = tuple(previous[index] - start[index] for index in range(3))
        cumulative_projection = _dot(total_delta, direction)
        cumulative_off_axis = tuple(
            total_delta[index] - cumulative_projection * direction[index]
            for index in range(3)
        )
        total_norm = _norm(total_delta)
        return {
            "direction": spec.direction,
            "requested_displacement_m": spec.requested_displacement_m,
            "requested_mm": spec.requested_displacement_m * 1000.0,
            "actual_projection_m": cumulative_projection,
            "actual_projection_mm": cumulative_projection * 1000.0,
            "total_displacement_xyz_m": list(total_delta),
            "total_displacement_xyz_mm": [value * 1000.0 for value in total_delta],
            "absolute_displacement_error_m": abs(cumulative_projection - spec.requested_displacement_m),
            "absolute_displacement_error_mm": abs(cumulative_projection - spec.requested_displacement_m) * 1000.0,
            "ticks_executed": len(ticks),
            "max_ticks": spec.max_ticks,
            "termination": termination,
            "off_axis_displacement_xyz_m": list(cumulative_off_axis),
            "off_axis_magnitude_m": _norm(cumulative_off_axis),
            "off_axis_magnitude_mm": _norm(cumulative_off_axis) * 1000.0,
            "direction_cosine": cumulative_projection / total_norm if total_norm > 1e-12 else None,
            "execution_duration_s": time.monotonic() - started,
            "tick_observations": ticks,
            "last_backend_receipt": backend_receipt,
        }


class LiberoPrimitiveBackend:
    """Submit approved V3 primitives and same-direction micro-motion ticks."""

    def __init__(self, environment: Any, controller: Any, arbiter: Arbiter) -> None:
        self.environment = environment
        self.controller = controller
        self.arbiter = arbiter

    def execute_approved_action(self, action: ApprovedAction) -> Any:
        if not self.arbiter.is_approved(action):
            raise TypeError("LIBERO backend accepts only an action authorized by its Arbiter")
        primitive = action.primitive
        if primitive.kind == "micro_motion":
            raise ValueError("micro-motion ticks must be submitted through the bounded Executor loop")
        if primitive.kind == "move":
            if primitive.token is None:
                raise ValueError("move primitive requires an atomic token")
            step_m = primitive.parameters.get("step_m")
            action = self.controller.action_for_atomic(primitive.token, step_m=step_m)
        elif primitive.kind == "grasp":
            action = self.controller.close_gripper()
        elif primitive.kind == "release":
            action = self.controller.open_gripper()
        elif primitive.kind == "hold":
            action = self.controller.hold_action()
        else:
            raise ValueError(f"unsupported primitive kind: {primitive.kind}")
        result = self.environment.step(action)
        if isinstance(result, tuple) and len(result) >= 4:
            info = result[3]
            return {
                "backend_returned": True,
                "done": bool(result[1]),
                "truncated": bool(result[2]),
                "info_keys": sorted(str(key) for key in info) if isinstance(info, dict) else [],
            }
        return {"backend_returned": True}

    def execute_approved_micro_tick(self, action: ApprovedAction) -> Any:
        if not self.arbiter.is_approved(action):
            raise TypeError("LIBERO backend accepts only an action authorized by its Arbiter")
        spec = action.primitive.micro_motion_spec
        if action.primitive.kind != "micro_motion" or spec is None:
            raise ValueError("expected an Arbiter-approved micro-motion")
        token = _DIRECTION_TOKEN[spec.direction]
        configured_vector = getattr(self.controller, "move_vectors", {}).get(token)
        if configured_vector is not None:
            vector = tuple(float(value) for value in configured_vector)
            if len(vector) != 3 or any(
                not math.isclose(vector[index], spec.direction_unit[index], rel_tol=1e-6, abs_tol=1e-6)
                for index in range(3)
            ):
                raise ValueError("approved semantic direction does not match controller mapping")
        controller_action = self.controller.action_for_atomic(
            token, step_m=spec.control_tick_step_m
        )
        result = self.environment.step(controller_action)
        if isinstance(result, tuple) and len(result) >= 4:
            info = result[3]
            return {
                "backend_returned": True,
                "done": bool(result[1]),
                "truncated": bool(result[2]),
                "info_keys": sorted(str(key) for key in info) if isinstance(info, dict) else [],
            }
        return {"backend_returned": True}


_DIRECTION_TOKEN = {
    "FWD": "MV_FWD", "BACK": "MV_BACK", "LEFT": "MV_LEFT",
    "RIGHT": "MV_RIGHT", "UP": "MV_UP", "DOWN": "MV_DOWN",
}


def _eef_position(observation: Any) -> tuple[float, float, float] | None:
    proprioception = getattr(observation, "proprioception", {})
    eef = proprioception.get("end_effector_state") if isinstance(proprioception, Mapping) else None
    raw = eef.get("position_xyz") if isinstance(eef, Mapping) else None
    if not isinstance(raw, (tuple, list)) or len(raw) != 3:
        return None
    try:
        position = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    return position if all(math.isfinite(value) for value in position) else None


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _norm(value: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in value))

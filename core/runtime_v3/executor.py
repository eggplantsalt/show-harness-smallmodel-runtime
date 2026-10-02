"""The only Runtime V3 module that calls an atomic robot controller."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from .arbiter import Arbiter, ApprovedAction


class PrimitiveBackend(Protocol):
    def execute_approved_action(self, action: ApprovedAction) -> Any: ...


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

    def execute(self, action: ApprovedAction) -> ExecutionRecord:
        if not self.arbiter.is_approved(action):
            raise TypeError("Executor accepts only an action authorized by its Arbiter")
        if action.primitive.max_steps != 1:
            raise ValueError("Executor accepts exactly one bounded primitive")
        started = time.monotonic()
        result = self.backend.execute_approved_action(action)
        return ExecutionRecord(
            option_id=action.option_id,
            primitive_kind=action.primitive.kind,
            started_at=started,
            finished_at=time.monotonic(),
            result=result,
        )


class LiberoPrimitiveBackend:
    """One environment step per approved primitive using the existing adapter."""

    def __init__(self, environment: Any, controller: Any, arbiter: Arbiter) -> None:
        self.environment = environment
        self.controller = controller
        self.arbiter = arbiter

    def execute_approved_action(self, action: ApprovedAction) -> Any:
        if not self.arbiter.is_approved(action):
            raise TypeError("LIBERO backend accepts only an action authorized by its Arbiter")
        primitive = action.primitive
        if primitive.kind == "move":
            if primitive.token is None:
                raise ValueError("move primitive requires an atomic token")
            action = self.controller.action_for_atomic(primitive.token)
        elif primitive.kind == "grasp":
            action = self.controller.close_gripper()
        elif primitive.kind == "release":
            action = self.controller.open_gripper()
        elif primitive.kind == "hold":
            action = self.controller.hold_action()
        else:
            raise ValueError(f"unsupported primitive kind: {primitive.kind}")
        return self.environment.step(action)

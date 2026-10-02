"""Thin wrapper around the repository's existing LIBERO task helpers."""

from __future__ import annotations

from typing import Any, Mapping

from core.sim.libero_task import (
    LiberoTaskHandle,
    make_libero_task,
    reset_libero,
    step_libero,
)


class LiberoEnvironmentAdapter:
    """Own LIBERO construction/reset and cache only the latest raw observation."""

    def __init__(self, handle: LiberoTaskHandle) -> None:
        self.handle = handle
        self.env = handle.env
        self.task_id = handle.task_id
        self.task_name = handle.task_name
        self.task_description = handle.task_description
        self.suite_name = handle.suite_name
        self.seed = 0
        self.step_count = 0
        self._observation: Mapping[str, Any] | None = getattr(
            self.env, "_showharness_last_obs", None
        )

    @classmethod
    def create(
        cls,
        *,
        suite_name: str = "LIBERO_OBJECT",
        task_id: int = 0,
        init_state_index: int = 0,
        seed: int = 0,
        camera_height: int = 256,
        camera_width: int = 256,
        horizon: int = 8,
    ) -> "LiberoEnvironmentAdapter":
        handle = make_libero_task(
            suite_name=suite_name,
            task_id=task_id,
            init_state_index=init_state_index,
            seed=seed,
            camera_height=camera_height,
            camera_width=camera_width,
            horizon=horizon,
            settle_steps=0,
        )
        adapter = cls(handle)
        adapter.seed = int(seed)
        return adapter

    def reset(self) -> Mapping[str, Any]:
        observation, _terminated, _truncated = reset_libero(
            self.env,
            self.handle.init_states[self.handle.init_state_index],
            settle_steps=0,
        )
        self.step_count = 0
        self._set_observation(observation)
        return observation

    def get_observation(self) -> Mapping[str, Any]:
        if self._observation is None:
            cached = getattr(self.env, "_showharness_last_obs", None)
            if cached is None:
                raise RuntimeError("LIBERO has no observation; call reset first")
            self._set_observation(cached)
        return self._observation

    def step(self, action: Any):
        """Apply one already-compiled environment action and cache its observation."""
        result = step_libero(self.env, action)
        observation, _terminated, _truncated, _info = result
        self.step_count += 1
        self._set_observation(observation)
        return result

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def _set_observation(self, observation: Mapping[str, Any]) -> None:
        self._observation = observation
        setattr(self.env, "_showharness_last_obs", observation)

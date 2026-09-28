"""Atomic Cartesian controller for LIBERO's robosuite OSC_POSE interface.

LIBERO's default OSC_POSE controller maps normalized XYZ commands to +/-0.05 m
and uses a 7D action ``[dx, dy, dz, drx, dry, drz, gripper]``.  The harness
speaks in the same MV_* atoms used by the real and RoboLab runners, so this
adapter deliberately keeps orientation at zero and exposes the controller
methods expected by the staged zero-shot runner.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from core.action_units import MOVE_ATOMS
from interpreters.sim_state import AtomicControllerState


class LiberoAtomicController:
    """Map one Show-Harness atomic token to a LIBERO OSC_POSE action."""

    def __init__(
        self,
        move_vectors: Mapping[str, Sequence[float]],
        step_m: float = 0.02,
        sim_steps_per_decision: int = 4,
        position_scale_m: float = 0.05,
        open_gripper_action: float = -1.0,
        close_gripper_action: float = 1.0,
    ) -> None:
        missing = [name for name in MOVE_ATOMS if name not in move_vectors]
        if missing:
            raise ValueError(f"move_vectors is missing tokens: {missing}")
        if position_scale_m <= 0:
            raise ValueError("position_scale_m must be positive")
        self.move_vectors = {name: np.asarray(move_vectors[name], dtype=float) for name in MOVE_ATOMS}
        self.step_m = float(step_m)
        self.sim_steps_per_decision = max(1, int(sim_steps_per_decision))
        self.position_scale_m = float(position_scale_m)
        self.open_gripper_action = float(open_gripper_action)
        self.close_gripper_action = float(close_gripper_action)
        self.state = AtomicControllerState(
            gripper_command=self.open_gripper_action,
            gripper_name="OPEN",
        )
        # Optional per-control-step Z correction used by the staged LIBERO runner.
        # OSC_POSE's XY commands can acquire a small vertical drift while carrying
        # an object; the runner sets this only during the horizontal MOVE stage.
        self.vertical_correction_m = 0.0

    def set_vertical_correction(self, correction_m: float = 0.0) -> None:
        """Add a bounded per-control-step Z correction to the next move."""
        self.vertical_correction_m = float(correction_m)

    def _action(self, delta_m: np.ndarray) -> np.ndarray:
        normalized = np.clip(np.asarray(delta_m, dtype=float) / self.position_scale_m, -1.0, 1.0)
        return np.asarray(
            [normalized[0], normalized[1], normalized[2], 0.0, 0.0, 0.0, self.state.gripper_command],
            dtype=np.float32,
        )

    def set_orientation_reference(self, _quat) -> None:
        """Compatibility hook; OSC_POSE already holds orientation at zero delta."""

    def with_orientation_hold(self, action: np.ndarray, _quat_cur) -> np.ndarray:
        return action

    def action_for_atomic(self, token: str, *, step_m: float | None = None) -> np.ndarray:
        if token not in self.move_vectors:
            raise ValueError(f"Unknown move token {token!r}; expected one of {MOVE_ATOMS}")
        self.state.last_atomic = token
        total_m = self.step_m if step_m is None else float(step_m)
        per_control_step = total_m / self.sim_steps_per_decision
        delta = self.move_vectors[token] * per_control_step
        delta = np.asarray(delta, dtype=float).copy()
        delta[2] += self.vertical_correction_m
        return self._action(delta)

    def hold_action(self) -> np.ndarray:
        self.state.last_atomic = None
        return self._action(np.zeros(3, dtype=float))

    def open_gripper(self) -> np.ndarray:
        self.state.gripper_command = self.open_gripper_action
        self.state.gripper_name = "OPEN"
        return self.hold_action()

    def close_gripper(self) -> np.ndarray:
        self.state.gripper_command = self.close_gripper_action
        self.state.gripper_name = "CLOSE"
        return self.hold_action()

"""Convert LIBERO RGB and proprioception into a raw record and V3 evidence."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from core.sim.libero_task import libero_gripper_width, libero_quat, libero_rgb, libero_tcp
from core.runtime_v3.observer import RobotObservation


@dataclass(frozen=True)
class RawObservation:
    observation_index: int
    environment_step: int
    timestamp_monotonic: float
    agentview_rgb: np.ndarray
    wrist_rgb: np.ndarray | None
    eef_position_xyz: tuple[float, float, float]
    eef_quaternion: tuple[float, float, float, float]
    gripper_width_m: float
    raw_keys: tuple[str, ...]


class LiberoObservationAdapter:
    """Read-only RGB/proprio Observer; simulator-only channels are not exposed."""

    def __init__(
        self,
        *,
        max_eef_z_m: float = 0.60,
        min_eef_z_m: float = 0.02,
        safe_lift_step_m: float = 0.005,
        on_raw_observation: Callable[[RawObservation], None] | None = None,
    ) -> None:
        self.max_eef_z_m = float(max_eef_z_m)
        self.min_eef_z_m = float(min_eef_z_m)
        self.safe_lift_step_m = float(safe_lift_step_m)
        self.on_raw_observation = on_raw_observation
        self.observation_index = 0
        self.last_raw: RawObservation | None = None
        self.last_observation: RobotObservation | None = None

    def observe(self, environment: Any) -> RobotObservation:
        raw = environment.get_observation()
        if not isinstance(raw, Mapping):
            raise TypeError("LIBERO observation must be a mapping")
        agentview = libero_rgb(raw, "agentview")
        wrist = None
        try:
            wrist = libero_rgb(raw, "robot0_eye_in_hand")
        except KeyError:
            pass
        position = tuple(float(v) for v in libero_tcp(raw))
        quaternion = tuple(float(v) for v in libero_quat(raw))
        width = float(libero_gripper_width(raw))
        self.observation_index += 1
        step_count = int(getattr(environment, "step_count", 0))
        record = RawObservation(
            observation_index=self.observation_index,
            environment_step=step_count,
            timestamp_monotonic=time.monotonic(),
            agentview_rgb=agentview,
            wrist_rgb=wrist,
            eef_position_xyz=position,
            eef_quaternion=quaternion,
            gripper_width_m=width,
            raw_keys=tuple(sorted(str(key) for key in raw.keys()
                                  if not str(key).casefold().endswith("_depth"))),
        )
        self.last_raw = record
        if self.on_raw_observation is not None:
            self.on_raw_observation(record)

        eef_finite = all(math.isfinite(value) for value in position)
        workspace_valid = (
            eef_finite
            and self.min_eef_z_m <= position[2] <= self.max_eef_z_m
            and position[2] + self.safe_lift_step_m <= self.max_eef_z_m
        )
        geometry = {
            "workspace_valid": bool(workspace_valid),
            "workspace_z_bounds_m": [self.min_eef_z_m, self.max_eef_z_m],
            "safe_lift_step_m": self.safe_lift_step_m,
            "eef_position_xyz": list(position),
        }
        evidence_refs = (
            f"libero:observation:{self.observation_index}:agentview",
            f"libero:observation:{self.observation_index}:wrist",
        ) if wrist is not None else (
            f"libero:observation:{self.observation_index}:agentview",
        )
        observation = RobotObservation(
            observation_id=f"libero-observation-{self.observation_index}",
            frame_id=self.observation_index,
            images={"agentview": agentview, "wrist": wrist},
            proprioception={
                "end_effector_state": {
                    "position_xyz": list(position),
                    "quaternion": list(quaternion),
                },
                # Width is a raw sensor value; this smoke adapter does not
                # infer semantic open/closed/holding state from an absolute threshold.
                "gripper_state": "UNKNOWN",
                "gripper_width_m": width,
            },
            evidence={
                "stage": "SMOKE",
                "relevant_geometry": geometry,
                "uncertainty": {"target": "not_observed_in_infrastructure_smoke"},
            },
            evidence_refs=evidence_refs,
            fresh=True,
            done=False,
        )
        self.last_observation = observation
        return observation

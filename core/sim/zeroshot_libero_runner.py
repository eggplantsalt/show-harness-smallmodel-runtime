"""LIBERO backend for Show-Harness's staged zero-shot planner/controller."""
from __future__ import annotations

from typing import Any

import numpy as np

from core.sim.libero_task import (
    libero_gripper_width,
    libero_quat,
    libero_rgb,
    libero_success,
    libero_tcp,
    reset_libero,
    step_libero,
)
from core.capabilities.camera_geometry import (
    build_robot_geometry_context,
    make_mujoco_calibrations,
)
from core.sim.zeroshot_robolab_runner import ZeroshotRobolabRunner


class ZeroshotLiberoRunner(ZeroshotRobolabRunner):
    """Reuse planner/plugins/logging while replacing only RoboLab I/O hooks."""

    backend_name = "libero_zeroshot_full_harness"

    def _normalize_stage_token(self, token: str, *, subgoal, obs) -> str:
        """Preserve Agent actions unless the legacy normalization is explicitly enabled.

        The clean Qwen profile intentionally lets the Agent own the visual direction.
        Earlier LIBERO experiments rewrote VLM tokens from host geometry; that made a
        raw MV_RIGHT become an executed MV_FWD and obscured whether the model or the
        harness was wrong.  Robot-only low-height protection remains active, while
        host geometry is exposed as evidence and completion validation rather than an
        action substitute.
        """
        self._active_stage = str(getattr(subgoal, "motion", "")).upper()
        if not self.legacy_stage_token_normalization:
            self._update_move_height_hold(token, subgoal=subgoal, obs=obs)
            return self._normalize_low_grasp_token(token, subgoal=subgoal, obs=obs)
        token = super()._normalize_stage_token(token, subgoal=subgoal, obs=obs)
        token = self._normalize_visual_xy_token(token, subgoal=subgoal)
        token = self._normalize_high_approach_token(token, subgoal=subgoal)
        token = self._normalize_lift_token(token, subgoal=subgoal, obs=obs)
        token = self._normalize_place_token(token, subgoal=subgoal)
        self._update_move_height_hold(token, subgoal=subgoal, obs=obs)
        if not self.legacy_stage_token_normalization:
            # The LIBERO eye-in-hand view makes a low, empty wrist image look like
            # an invitation to descend.  Robot-only height evidence is enough to
            # reject that failure mode: once below the empirically calibrated
            # pre-grasp band, recover upward before either DONE or GRASP.
            token = self._normalize_low_grasp_token(token, subgoal=subgoal, obs=obs)
            return token
        if token != "MV_DOWN" or str(getattr(subgoal, "motion", "")).upper() != "APPROACH":
            return self._normalize_low_grasp_token(token, subgoal=subgoal, obs=obs)
        pos = self._tcp(obs)
        table = float(self.table_height_m)
        if float(pos[2]) - table > 0.10:
            return "MV_FWD"
        return self._normalize_low_grasp_token(token, subgoal=subgoal, obs=obs)

    def _physical_step_for(self, token: str, *, target_in_wrist, pre_pose):
        """Use a bounded coarse descent only while placing a held object.

        With a closed gripper, LIBERO's OSC controller moves downward much more
        slowly than during the empty approach.  Keeping the generic fine step made
        PLACE spend hundreds of decisions lowering a bottle by a few millimeters.
        The coarse step is used only above a conservative placement band; the final
        approach returns to the normal fine step.
        """
        if str(token).strip().upper() == "MV_DOWN" and getattr(self, "_active_stage", "") == "PLACE":
            height = float(self._fingertip_position(pre_pose)[2])
            if height > 0.16:
                return min(self.physical_fine_step_m * 3.0, 0.06), "place_coarse"
        return super()._physical_step_for(
            token,
            target_in_wrist=target_in_wrist,
            pre_pose=pre_pose,
        )

    def _update_move_height_hold(self, token: str, *, subgoal, obs) -> None:
        """Keep a carried object at the LIFT height during horizontal MOVE.

        In LIBERO's OSC controller, a sequence of nominally horizontal XY commands
        can drift in Z while the gripper carries a bottle.  That drift changes the
        AgentView projection and makes the MOVE servo chase a moving target.  The
        correction is derived only from current EEF proprioception and the height
        captured at MOVE entry; PLACE remains the only stage allowed to descend.
        """
        controller_setter = getattr(self.controller, "set_vertical_correction", None)
        if not callable(controller_setter):
            return
        stage = str(getattr(subgoal, "motion", "")).upper()
        if stage != "MOVE" or token not in {"MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT"}:
            controller_setter(0.0)
            if stage != "MOVE":
                self._move_height_target_m = None
                self._move_height_identity = None
            return
        identity = (
            str(getattr(subgoal, "sid", ""))
            or str(getattr(subgoal, "target", ""))
            or "MOVE"
        )
        if getattr(self, "_move_height_identity", None) != identity:
            self._move_height_identity = identity
            self._move_height_target_m = float(
                self._fingertip_position(self._tcp(obs))[2]
            )
        current_height = float(self._fingertip_position(self._tcp(obs))[2])
        error = float(self._move_height_target_m) - current_height
        # Convert a bounded per-decision correction to the controller's per-step
        # command.  This is a stabilizer, not a scripted Z trajectory.
        correction = float(np.clip(error, -0.04, 0.04)) / max(
            1, int(getattr(self.controller, "sim_steps_per_decision", 1))
        )
        controller_setter(correction)

    def _normalize_high_approach_token(self, token: str, *, subgoal) -> str:
        """Descend once lateral AgentView alignment is already established.

        AgentView's vertical pixel error mixes world-X depth with camera height.
        At the initial high pose it is therefore unsafe to keep issuing MV_FWD just
        because the bottle is low in the image.  The robot-only height and the
        horizontal pixel error provide the minimal host-side disambiguation: while
        high and laterally aligned, descend; once in the pre-grasp band, let the
        normal visual policy make the small depth corrections.
        """
        if str(getattr(subgoal, "motion", "")).upper() != "APPROACH":
            return token
        harness = self.visual_harness
        maximum = getattr(harness, "approach_completion_max_height_m", None)
        if harness is None or maximum is None:
            return token
        evidence = getattr(harness, "last_evidence", {}) or {}
        geometry = evidence.get("geometry")
        if not isinstance(geometry, dict) or not evidence.get("visible", False):
            return token
        try:
            height = float(geometry.get("eef_height_m"))
            horizontal_error = abs(float(geometry["target_minus_eef_px"][0]))
        except (TypeError, ValueError, KeyError, IndexError):
            return token
        if (
            height > float(maximum)
            and horizontal_error <= 8.0
            and token in {"MV_FWD", "MV_BACK"}
        ):
            return "MV_DOWN"
        return token

    def _normalize_lift_token(self, token: str, *, subgoal, obs) -> str:
        """Make LIFT a monotonic physical phase after a verified grasp.

        A held bottle remains below the hand in AgentView.  The generic visual
        rule therefore tends to reinterpret that image relation as another
        descent and oscillates between MV_UP and MV_DOWN.  During LIFT the only
        safe progress variable is the robot's current fingertip height: keep
        ascending until the calibrated clearance band, then close the stage.
        """
        if str(getattr(subgoal, "motion", "")).upper() != "LIFT":
            return token
        try:
            height = float(self._fingertip_position(self._tcp(obs))[2])
        except (TypeError, ValueError, IndexError):
            return token
        if height < self.lift_clear_height_m:
            return "MV_UP"
        return "DONE"

    def _normalize_visual_xy_token(self, token: str, *, subgoal) -> str:
        """Use visual bbox + camera geometry to close the largest XY error.

        This remains an image-grounded servo correction: the target point comes
        from SAM3/tracking and a configured reference-height ray intersection,
        while the current point comes from robot proprioception.  It is used only
        for APPROACH, before any gripper command, and prevents a VLM from treating
        camera-height parallax as a reason to descend or overshoot world +X.
        """
        stage = str(getattr(subgoal, "motion", "")).upper()
        if stage not in {"APPROACH", "MOVE", "PLACE"}:
            if stage != "MOVE":
                self._move_target_xy_m = None
            return token
        harness = self.visual_harness
        evidence = getattr(harness, "last_evidence", {}) if harness is not None else {}
        geometry = evidence.get("geometry") if isinstance(evidence, dict) else None
        error = geometry.get("target_minus_eef_xy_m") if isinstance(geometry, dict) else None
        if not isinstance(error, (list, tuple)) or len(error) != 2:
            return token
        try:
            x_error, y_error = float(error[0]), float(error[1])
        except (TypeError, ValueError):
            return token

        # In MOVE the destination can become clipped by the fixed AgentView as
        # the EEF travels toward it.  Recomputing the basket center from a clipped
        # bbox makes the target appear to move away, so anchor the destination once
        # from the first complete visual observation of this stage.  The anchor is
        # still a SAM3/tracker-derived camera back-projection, never simulator state
        # or an episode-specific coordinate table.
        if stage == "MOVE":
            if getattr(self, "_move_target_xy_m", None) is None:
                candidate = geometry.get("target_xy_world_estimate")
                bbox = evidence.get("bbox_xyxy") if isinstance(evidence, dict) else None
                complete_bbox = False
                try:
                    complete_bbox = (
                        float(bbox[0]) > 2.0
                        and float(bbox[1]) > 2.0
                        and float(bbox[2]) < 254.0
                        and float(bbox[3]) < 254.0
                    )
                except (TypeError, ValueError, IndexError):
                    complete_bbox = False
                if (
                    isinstance(candidate, (list, tuple))
                    and len(candidate) >= 2
                    and complete_bbox
                    and bool(evidence.get("visible", False))
                ):
                    try:
                        self._move_target_xy_m = (
                            float(candidate[0]),
                            float(candidate[1]),
                        )
                    except (TypeError, ValueError):
                        self._move_target_xy_m = None
            anchor = getattr(self, "_move_target_xy_m", None)
            eef = geometry.get("eef_position_xyz")
            if (
                isinstance(anchor, (list, tuple))
                and len(anchor) == 2
                and isinstance(eef, (list, tuple))
                and len(eef) >= 2
            ):
                try:
                    x_error = float(anchor[0]) - float(eef[0])
                    y_error = float(anchor[1]) - float(eef[1])
                except (TypeError, ValueError):
                    pass
        if max(abs(x_error), abs(y_error)) <= 0.015:
            if stage == "MOVE":
                return "DONE"
            if stage == "PLACE" and token in {
                "MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT"
            }:
                # Once the current visual geometry says the horizontal relation is
                # aligned, do not let a stale language direction keep sliding the
                # held object. PLACE may now descend and let the verifier decide.
                return "MV_DOWN"
            return token
        # Do not replace a valid gripper/termination token; the visual XY servo is
        # only allowed to rewrite motion atoms during APPROACH or MOVE.
        if token not in {"MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN"}:
            return token
        if abs(x_error) >= abs(y_error) and abs(x_error) > 0.015:
            return "MV_FWD" if x_error > 0.0 else "MV_BACK"
        return "MV_LEFT" if y_error > 0.0 else "MV_RIGHT"

    def _normalize_low_grasp_token(self, token: str, *, subgoal, obs) -> str:
        """Keep LIBERO from descending below the physical pre-grasp band.

        This is not an object-state shortcut.  It only uses the robot EEF height
        and a calibration value measured with the same Panda/gripper geometry.  In
        this scene the target is lower than the camera/EEF projection, so treating
        every screen-down pixel as another descent sends the fingers underneath the
        bottle and produces a closed-but-empty gripper.
        """
        harness = self.visual_harness
        minimum = getattr(harness, "grasp_agentview_guard_min_height_m", None)
        if harness is None or minimum is None:
            return token
        stage = str(getattr(subgoal, "motion", "")).upper()
        if stage not in {"APPROACH", "GRASP"}:
            return token
        try:
            height = float(self._fingertip_position(self._tcp(obs))[2])
        except (TypeError, ValueError, IndexError):
            return token
        if height >= float(minimum):
            return token
        if token in {"MV_DOWN", "GRASP", "DONE"}:
            return "MV_UP"
        return token

    def _normalize_place_token(self, token: str, *, subgoal) -> str:
        """Apply the visual Agent's current PLACE correction without fixed geometry.

        The pre-placement reviewer owns the object/receptacle relation. Its action
        may be a lift or horizontal correction. Once it says aligned, the model's
        vertical placement action is preserved. This avoids turning every horizontal
        token into MV_DOWN, which previously hid the recovery action needed for an
        off-center placement.
        """
        if str(getattr(subgoal, "motion", "")).upper() != "PLACE":
            self._place_verification_decision = None
            self._place_alignment_decision = None
            self._place_alignment_action = None
            return token
        # Preserve the token selected from the current observation. The alignment
        # reviewer is advisory; it must not overwrite the fresh world-coordinate
        # correction chosen by _normalize_visual_xy_token with a stale direction
        # from the previous frame.
        return token

    def __init__(
        self,
        *,
        init_state: np.ndarray,
        lift_clear_height_m: float = 0.17,
        legacy_stage_token_normalization: bool = True,
        **kwargs: Any,
    ) -> None:
        self.legacy_stage_token_normalization = bool(legacy_stage_token_normalization)
        self.lift_clear_height_m = float(lift_clear_height_m)
        self._active_stage = ""
        self._move_height_target_m = None
        self._move_height_identity = None
        self._move_target_xy_m = None
        super().__init__(**kwargs)
        self.init_state = np.asarray(init_state)

    def _reset_episode(self):
        obs, terminated, truncated = reset_libero(
            self.env,
            self.init_state,
            settle_steps=self.num_steps_wait,
            hold_action=self.controller.open_gripper(),
        )
        self._last_obs = obs
        return obs, terminated, truncated

    def _step_env(self, action):
        result = step_libero(self.env, action)
        self._last_obs = result[0]
        return result

    def _success(self, obs=None) -> bool:
        return libero_success(self.env)

    def _rgb(self, obs, camera: str) -> np.ndarray:
        return libero_rgb(obs, camera)

    def _tcp(self, obs=None) -> np.ndarray:
        return libero_tcp(obs if obs is not None else self._last_obs)

    def _quat(self, obs=None) -> np.ndarray:
        return libero_quat(obs if obs is not None else self._last_obs)

    def _gripper_width(self, obs=None) -> float:
        return libero_gripper_width(obs if obs is not None else self._last_obs)

    def _fingertip_position(self, flange_position=None) -> np.ndarray:
        # LIBERO's eef site is the Panda grasp center; unlike the RoboLab hand
        # accessor it already corresponds to the point used by the OSC controller.
        return np.asarray(
            self._tcp() if flange_position is None else flange_position,
            dtype=float,
        )

    def _visual_geometry(self, obs, agentview, wrist):
        """Project only the current EEF through LIBERO camera calibration."""
        try:
            calibrations = make_mujoco_calibrations(
                self.env,
                {
                    "agentview": self.agentview_camera,
                    "wrist": self.wrist_camera,
                },
                image_shapes={
                    "agentview": tuple(np.asarray(agentview).shape[:2]),
                    "wrist": tuple(np.asarray(wrist).shape[:2]) if wrist is not None else tuple(np.asarray(agentview).shape[:2]),
                },
                rotations={
                    "agentview": self.agentview_rotation_degrees,
                    "wrist": self.wrist_rotation_degrees,
                },
                flips={
                    "agentview": self.agentview_flip,
                    "wrist": self.wrist_flip,
                },
            )
            fingertip_position = self._fingertip_position(self._tcp(obs))
            context = build_robot_geometry_context(
                calibrations,
                fingertip_position,
            )
            # This is the calibrated LIBERO camera/action convention, not a
            # task rule: the agent still decides whether either correction is
            # appropriate and when to commit a grasp.
            for evidence in context.values():
                if isinstance(evidence, dict):
                    evidence["eef_position_xyz"] = [
                        round(float(value), 5) for value in fingertip_position
                    ]
                    evidence["eef_height_m"] = round(float(fingertip_position[2]), 5)
                    evidence["screen_direction_to_token"] = {
                        "right": "MV_RIGHT",
                        "left": "MV_LEFT",
                        "down": "MV_FWD",
                        "up": "MV_BACK",
                    }
            # Keep static camera calibration available to the visual harness so
            # it can back-project a SAM3 bbox onto a configured reference height.
            # These are camera/robot measurements, not object state.
            for name, calibration in calibrations.items():
                if isinstance(context.get(name), dict):
                    context[name]["camera_calibration"] = {
                        "width": int(calibration.width),
                        "height": int(calibration.height),
                        "fovy_deg": float(calibration.fovy_deg),
                        "position_world": [
                            round(float(value), 6)
                            for value in calibration.position_world
                        ],
                        "camera_to_world": [
                            [round(float(value), 6) for value in row]
                            for row in np.asarray(calibration.camera_to_world).reshape(3, 3)
                        ],
                        "rotation_degrees": int(calibration.rotation_degrees),
                        "flip": str(calibration.flip),
                    }
            return context
        except Exception as exc:  # capability evidence must not break control
            return {"error": f"geometry_unavailable:{type(exc).__name__}:{exc}"}

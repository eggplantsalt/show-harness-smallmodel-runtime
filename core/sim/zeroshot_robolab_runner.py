"""Show-Harness zero-shot staged controller on RoboLab.

The high-level semantics mirror the official real-robot zero-shot runner:
planner -> controller + plugins -> atomic execution -> measured feedback ->
recovery / rollback.

Only the physical execution backend is RoboLab-specific.
"""

from __future__ import annotations

import time
import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from core.action_units import MOVE_ATOMS
from core.record.images import prepare_view
from core.runtime_v2.spatial import triangulate_metric_points_checked
from core.runtime_v2.types import SpatialHealth, SpatialToolResult
from core.sim.mvtoken_robolab_runner import (
    DONE_TOKEN,
    GRASP_TOKEN,
    RELEASE_TOKEN,
    MvTokenRobolabRunner,
    video_fps,
)
from core.sim.robolab_task import (
    reset_robolab,
    rl_ee_quat,
    rl_gripper_width,
    rl_rgb,
    rl_success,
    rl_tcp,
    step_robolab,
)
from core.v0_types import EpisodeResult, SkillContext


NO_DIRECTION = "NONE"
STOP_TOKEN = "STOP"
EMPTY_GRASP_LABEL = "GRASP(empty)"


def _should_end_after_verified_grasp(runtime: Any, verified_grasp: bool) -> bool:
    return bool(
        verified_grasp
        and getattr(runtime, "placement_v22_enabled", False)
        and getattr(runtime, "grasp_verification_only", False)
    )


def _save_runtime_memory_frame(
    logger: Any,
    runtime: Any,
    frame_id: int,
    *,
    raw_agentview: Optional[np.ndarray],
    raw_wrist: Optional[np.ndarray],
) -> tuple[Optional[dict[str, Optional[str]]], bool]:
    """Persist raw views before VCR stores their references or asks Qwen.

    These refs are episode-local paths into the logger's raw image tree. A
    missing view stays missing so the memory panel builder can fail closed.
    """
    if getattr(runtime, "visual_memory", None) is None:
        return None, False
    save = getattr(logger, "save_visual_artifacts", None)
    if not callable(save) or raw_agentview is None:
        return None, False
    save(
        frame_id,
        raw_agentview=raw_agentview,
        raw_wrist=raw_wrist,
        provider_overlay=None,
    )
    refs: dict[str, Optional[str]] = {
        "agentview": f"images/raw_agentview/{int(frame_id):04d}.png",
        "wrist": (
            f"images/raw_wrist/{int(frame_id):04d}.png"
            if raw_wrist is not None else None
        ),
    }
    return refs, True


def _reset_runtime_episode(runtime: Any, logger: Any) -> None:
    """Reset VCR memory into this logger's unique episode namespace."""
    reset = getattr(runtime, "reset", None)
    if not callable(reset):
        return
    if (
        getattr(runtime, "visual_memory", None) is not None
        or bool(getattr(runtime, "placement_v22_enabled", False))
    ):
        reset(episode_id=str(logger.run_dir.resolve()))
    else:
        reset()


@dataclass
class SimAtomicStepResult:
    token: str
    kind: str
    pre_pose: np.ndarray
    post_pose: np.ndarray
    intended_delta_m: np.ndarray
    step_m: float
    step_kind: str
    gripper_closed: bool
    grasp_empty: bool = False
    done: bool = False
    note: str = ""


def _track_v22_target_views(
    *,
    tracker: Any,
    stage: str,
    capability_evidence: dict[str, Any],
    camera_images: dict[str, Any],
    frame_id: int,
    instance_id: Optional[str],
    grasp_epoch: int,
) -> dict[str, dict[str, Any]]:
    """Update independent target streams for visible approach/grasp cameras."""
    if (
        not callable(tracker)
        or not instance_id
        or str(stage).upper() not in {"APPROACH", "GRASP"}
    ):
        return {}
    results: dict[str, dict[str, Any]] = {}
    primary_camera = str(capability_evidence.get("camera", "")).lower()
    primary_image = camera_images.get(primary_camera)
    if primary_camera in {"agentview", "wrist"} and primary_image is not None:
        result = tracker(
            image=primary_image,
            mask=capability_evidence.get("mask"),
            camera=primary_camera,
            frame_id=frame_id,
            instance_id=instance_id,
            grasp_epoch=grasp_epoch,
        )
        if isinstance(result, dict):
            results[primary_camera] = result

    secondary = capability_evidence.get("secondary_view")
    if isinstance(secondary, dict):
        secondary_camera = str(secondary.get("camera", "")).lower()
        secondary_image = camera_images.get(secondary_camera)
        if (
            secondary_camera in {"agentview", "wrist"}
            and secondary_camera != primary_camera
            and secondary_image is not None
        ):
            try:
                secondary_age = int(secondary.get("age_frames", 0) or 0)
            except (TypeError, ValueError):
                secondary_age = 1
            result = tracker(
                image=secondary_image,
                mask=secondary.get("mask") if secondary_age == 0 else None,
                camera=secondary_camera,
                frame_id=frame_id,
                instance_id=instance_id,
                grasp_epoch=grasp_epoch,
            )
            if isinstance(result, dict):
                results[secondary_camera] = result
    return results


def _descend_travel(
    token: str,
    result: Any,
    previous: Optional[tuple[float, float]],
) -> Optional[tuple[float, float]]:
    """Mirror core.runners.real.descend_travel for the simulator."""

    if result is None or getattr(result, "kind", "") not in (
        "move",
        "rotate",
    ):
        return previous

    pre = getattr(result, "pre_pose", None)
    post = getattr(result, "post_pose", None)

    intended = np.asarray(
        getattr(
            result,
            "intended_delta_m",
            [0.0, 0.0, 0.0],
        ),
        dtype=float,
    )

    commanded = float(intended[2])

    if (
        str(token).strip().upper() != "MV_DOWN"
        or pre is None
        or post is None
        or commanded >= 0.0
    ):
        return None

    travelled = float(
        np.asarray(pre, dtype=float)[2]
        - np.asarray(post, dtype=float)[2]
    )

    return max(0.0, travelled), abs(commanded)


class ZeroshotRobolabRunner(MvTokenRobolabRunner):
    """Official staged zero-shot semantics on the RoboLab backend."""

    # Backend hooks.  The planner, controller and plugins are simulator agnostic;
    # subclasses override these small observation/action hooks for another simulator.
    def _reset_episode(self):
        return reset_robolab(
            self.env,
            hold_action=self.controller.open_gripper(),
            settle_steps=self.num_steps_wait,
        )

    def _step_env(self, action):
        return step_robolab(self.env, action)

    def _success(self, obs=None) -> bool:
        return bool(rl_success(self.env))

    def _rgb(self, obs, camera: str) -> np.ndarray:
        return rl_rgb(obs, camera)

    def _tcp(self, obs=None) -> np.ndarray:
        return np.asarray(rl_tcp(self.env), dtype=float)

    def _quat(self, obs=None) -> np.ndarray:
        return np.asarray(rl_ee_quat(self.env), dtype=float)

    def _gripper_width(self, obs=None) -> float:
        return float(rl_gripper_width(self.env))

    def _visual_geometry(self, obs, agentview, wrist):
        """Optional backend hook; subclasses may expose calibration-only evidence."""
        return {}

    @staticmethod
    def _held_target_for(subgoals, current_index: int):
        """Return the semantic object from the most recent GRASP subgoal.

        MOVE/PLACE subgoals quite reasonably name the receptacle, but visual
        placement needs the identity of the object that is supposed to be carried.
        Keeping that identity from the planner's own GRASP description is generic
        across tasks and does not read simulator object state or encode a scene.
        """
        try:
            upper = min(int(current_index), len(subgoals) - 1)
        except (TypeError, ValueError):
            return None, None
        for index in range(upper, -1, -1):
            candidate = subgoals[index]
            if str(getattr(candidate, "motion", "")).upper() == "GRASP":
                return (
                    str(getattr(candidate, "target", "") or "").strip() or None,
                    str(getattr(candidate, "affordance", "") or "").strip() or None,
                )
        return None, None

    def _visual_route_inputs(
        self,
        *,
        agentview: np.ndarray,
        capability_evidence: dict[str, Any],
        stage: str,
        subgoals: Any,
        current_index: int,
        held_target: Optional[str],
        held_affordance: Optional[str],
    ) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        """Build route evidence from the existing visual harness only.

        The route plugin never calls a simulator object-pose API.  At the LIFT
        boundary it asks SAM3 for the carried object and future receptacle once;
        in MOVE/PLACE it reuses the held-object/opening evidence already produced
        by VisualHarness.  This keeps the raw-frame perception path authoritative
        and avoids a second detector or a second GPU model.
        """
        plugin = self.visual_route_plugin
        if plugin is None or self.visual_harness is None:
            return None, None
        stage_name = str(stage or "").upper()
        held: Optional[dict[str, Any]] = None
        destination: Optional[dict[str, Any]] = None
        if stage_name in {"MOVE", "PLACE", "TRANSPORT"}:
            raw_held = capability_evidence.get("held_object")
            if isinstance(raw_held, dict):
                held = dict(raw_held)
            locked_offset = getattr(
                self.verified_runtime, "placement_object_to_gripper_xyz", None
            )
            if held is None:
                held = {}
            if isinstance(locked_offset, (list, tuple)) and len(locked_offset) >= 3:
                held["object_to_gripper_xyz"] = [float(value) for value in locked_offset[:3]]
            locked_points = getattr(
                self.verified_runtime, "placement_object_points_gripper", ()
            )
            if isinstance(locked_points, (list, tuple)) and locked_points:
                held["object_points_gripper"] = [
                    [float(value) for value in point[:3]]
                    for point in locked_points[:2048]
                    if isinstance(point, (list, tuple)) and len(point) >= 3
                ]
            alignment = capability_evidence.get("held_object_alignment")
            if isinstance(alignment, dict):
                if held is None:
                    held = {}
                for key in (
                    "destination_minus_held_center_px",
                    "correction_candidates",
                    "rim_contact_risk",
                    "clearance_progress_ready",
                    "horizontal_motion_stalled",
                    "aligned",
                    "alignment_px",
                    "alignment_threshold_px",
                ):
                    if key in alignment:
                        held[key] = alignment[key]
            opening_mask = capability_evidence.get("opening_mask")
            opening_mask_bbox = (
                opening_mask.get("bbox_xyxy")
                if isinstance(opening_mask, dict)
                else None
            )
            # SAM3 may fall back to a container mask while still returning a
            # valid auxiliary opening mask.  The two masks have different
            # roles: derive the placement bbox from the opening mask and keep
            # the container bbox only as an outer collision/context cue.
            opening_bbox = opening_mask_bbox or capability_evidence.get("bbox_xyxy")
            outer_bbox = capability_evidence.get("outer_destination_bbox_xyxy")
            if opening_bbox is not None or outer_bbox is not None:
                opening_evidence = (
                    opening_mask
                    if isinstance(opening_mask, dict)
                    else (
                        capability_evidence.get("mask")
                        if not bool(getattr(plugin, "placement_v22_enabled", False))
                        else None
                    )
                )
                destination = {
                    "bbox_xyxy": outer_bbox or opening_bbox,
                    "opening_bbox_xyxy": opening_bbox or outer_bbox,
                    "opening_mask": opening_evidence,
                    "outer_mask": capability_evidence.get("mask"),
                    "confidence": capability_evidence.get("confidence", 0.0),
                    "source": capability_evidence.get("source", "visual_harness"),
                }
            return held, destination

        # Keep the normal target tracker as the carried-object visual cue during
        # LIFT.  The first LIFT observation is before the normal MOVE/PLACE
        # held-object relation exists, so also ground the next receptacle in the
        # same raw AgentView when CPU geometry creates the shared route.
        if stage_name == "LIFT" and (
            plugin.route is None or not getattr(plugin.route, "valid", False)
        ):
            raw_target = capability_evidence.get("bbox_xyxy")
            if isinstance(raw_target, (list, tuple)) and len(raw_target) == 4:
                held = {
                    "bbox_xyxy": list(raw_target),
                    "confidence": capability_evidence.get("confidence", 0.0),
                    "source": capability_evidence.get("source", "visual_harness"),
                }
            destination_target, destination_affordance = plugin._next_destination(
                subgoals, current_index
            )
            if held is None and held_target:
                held = self.visual_harness.ground_auxiliary(
                    agentview=agentview,
                    target=held_target,
                    affordance=held_affordance,
                    include_opening=False,
                )
            if destination_target:
                destination = self.visual_harness.ground_auxiliary(
                    agentview=agentview,
                    target=destination_target,
                    affordance=destination_affordance,
                    include_opening=True,
                )
        elif stage_name == "LIFT":
            # After route creation, use the fresh CPU/SAM tracker bbox rather
            # than dropping the held-payload cue from every subsequent frame.
            raw_target = capability_evidence.get("bbox_xyxy")
            if isinstance(raw_target, (list, tuple)) and len(raw_target) == 4:
                held = {
                    "bbox_xyxy": list(raw_target),
                    "confidence": capability_evidence.get("confidence", 0.0),
                    "source": capability_evidence.get("source", "visual_harness"),
                }
            # Keep the destination affordance available for periodic route refreshes
            # during LIFT. The original grounding call is intentionally not repeated
            # on every frame, but reusing the last visual boxes is not simulator pose
            # state and prevents a refresh from degenerating into an invalid route.
            if destination is None and getattr(plugin, "route", None) is not None:
                route = plugin.route
                if (
                    getattr(route, "destination_bbox_xyxy", None) is not None
                    and getattr(route, "opening_bbox_xyxy", None) is not None
                ):
                    destination = {
                        "bbox_xyxy": list(route.destination_bbox_xyxy),
                        "opening_bbox_xyxy": list(route.opening_bbox_xyxy),
                        "confidence": float(getattr(route, "confidence", 0.0)),
                        "source": "visual_route_previous_evidence",
                    }
        return held, destination

    def _apply_orientation_hold(self, action: np.ndarray, obs=None) -> np.ndarray:
        return self.controller.with_orientation_hold(action, self._quat(obs))

    def __init__(
        self,
        *,
        planner: Any,
        controls: Any,
        task_name: str = "",
        seed: int = 0,
        episode_index: int = 0,
        recovery_plugin: Any = None,
        action_chunk_plugin: Any = None,
        variable_step_plugin: Any = None,
        visual_harness: Any = None,
        visual_route_plugin: Any = None,
        verified_runtime: Any = None,
        table_height_m: float,
        fingertip_offset_m: float = 0.1323,
        wrist_grasp_marker: Optional[dict] = None,
        physical_fine_step_m: float = 0.02,
        physical_up_step_m: float = 0.04,
        sim_command_scale: float = 3.6,
        empty_width_m: float = 0.005,
        gripper_close_threshold_m: float = 0.07,
        recent_moves_max: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(
            agent=controls.controller.agent,
            **kwargs,
        )

        self.planner = planner
        self.controls = controls
        self.task_name = str(task_name)
        self.seed = int(seed)
        self.episode_index = int(episode_index)

        self.recovery_plugin = recovery_plugin
        self.action_chunk_plugin = action_chunk_plugin
        self.variable_step_plugin = variable_step_plugin
        self.visual_harness = visual_harness
        self.visual_route_plugin = visual_route_plugin
        self.verified_runtime = verified_runtime

        self.table_height_m = float(table_height_m)
        # Robot geometry only: panda_hand local +Z points toward the fingertips.
        # Object poses and task targets are never read to construct this observation.
        self.fingertip_offset_m = float(fingertip_offset_m)
        self.wrist_grasp_marker = dict(wrist_grasp_marker or {})
        self.wrist_marker_metadata: dict[str, Any] = {"enabled": False}

        # These are physical distances as understood by the Harness/VLM.
        self.physical_fine_step_m = float(
            physical_fine_step_m
        )
        self.physical_up_step_m = float(
            physical_up_step_m
        )

        # Convert physical semantics -> RoboLab calibrated command space.
        self.sim_command_scale = float(sim_command_scale)

        self.empty_width_m = float(empty_width_m)
        self.gripper_close_threshold_m = float(
            gripper_close_threshold_m
        )

        self.recent_moves_max = max(
            1,
            int(recent_moves_max),
        )

        self._descend: Optional[
            tuple[float, float]
        ] = None
        self._place_verification_stage: str | None = None
        self._place_verification_decision: str | None = None
        self._place_verification_last_step = -10_000
        self._place_verification_recovery_action: str | None = None
        # Kept for compatibility with old rollout records. Placement recovery is
        # selected from fresh Agent review; the runner does not force a fixed lift-first
        # sequence or convert it into a recovery state machine.
        self._place_recovery_needs_lift = False
        self._place_alignment_stage: str | None = None
        self._place_alignment_decision: str | None = None
        self._place_alignment_action: str | None = None
        # A rejected/lost grasp invalidates the previous approach completion.  The
        # next approach must contain one fresh Agent-chosen visual correction before
        # DONE can be accepted again; this is a completion contract, not a recovery
        # trajectory or a direction-specific state machine.
        self._reacquire_required = False
        # Relation snapshot for a mechanically empty GRASP.  A later close must
        # be based on a changed visual hypothesis, not the same projected pose.
        self._grasp_retry_anchor: dict[str, tuple[float, float]] | None = None
        self._pending_grasp_pre_action: Optional[dict[str, Any]] = None
        self._active_frame_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Physical adapter
    # ------------------------------------------------------------------

    def _physical_step_for(
        self,
        token: str,
        *,
        target_in_wrist: Optional[bool],
        pre_pose: np.ndarray,
    ) -> tuple[float, str]:
        token = str(token).strip().upper()

        if token == "MV_UP":
            return self.physical_up_step_m, "up"

        plugin = self.variable_step_plugin

        # Transport is a long-horizon route, not the final grasp servo.  Once
        # the EEF is safely above the table, use the configured embodiment
        # coarse translation for horizontal route atoms; retain fine motion for
        # vertical/contact moves.  This keeps the route within its bounded
        # option budget without introducing an object- or scene-specific
        # distance rule.
        if (
            plugin is not None
            and bool(getattr(plugin, "enabled", False))
            and str(getattr(self, "_current_stage", "")).upper() == "TRANSPORT"
            and token in {"MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT"}
        ):
            try:
                height = float(self._fingertip_position(pre_pose)[2])
                if height - float(self.table_height_m) > 0.12:
                    return max(self.physical_fine_step_m, float(plugin.coarse_step_m)), "coarse_transport"
            except (TypeError, ValueError, AttributeError):
                pass

        if (
            plugin is None
            or not getattr(plugin, "enabled", False)
        ):
            return self.physical_fine_step_m, "fine"

        eef_height_m = float(self._fingertip_position(pre_pose)[2])

        step = float(
            plugin.step_m_for(
                token,
                self.physical_fine_step_m,
                eef_height_m=eef_height_m,
                table_height_m=self.table_height_m,
                target_in_wrist=target_in_wrist,
            )
        )

        kind = (
            "fine"
            if abs(
                step - self.physical_fine_step_m
            ) < 1e-12
            else "coarse"
        )

        return step, kind

    def _execute_agentic(
        self,
        token: str,
        obs: dict,
        *,
        target_in_wrist: Optional[bool],
        step_override_m: Optional[float] = None,
    ) -> tuple[
        dict,
        bool,
        bool,
        SimAtomicStepResult,
    ]:
        token = str(token).strip().upper()

        pre_pose = self._tcp(obs)

        terminated = False
        truncated = False

        intended_delta_m = np.zeros(
            3,
            dtype=float,
        )

        step_m = 0.0
        step_kind = ""
        note = ""

        if token == DONE_TOKEN:
            return (
                obs,
                False,
                False,
                SimAtomicStepResult(
                    token=token,
                    kind="done",
                    pre_pose=pre_pose,
                    post_pose=pre_pose.copy(),
                    intended_delta_m=intended_delta_m,
                    step_m=0.0,
                    step_kind="",
                    gripper_closed=(
                        self.controller.state.gripper_name
                        == "CLOSE"
                    ),
                    done=True,
                    note="subgoal done",
                ),
            )

        if token in MOVE_ATOMS:
            physical_step_m, step_kind = (
                self._physical_step_for(
                    token,
                    target_in_wrist=target_in_wrist,
                    pre_pose=pre_pose,
                )
            )
            if step_override_m is not None and token == "MV_UP":
                physical_step_m = max(0.001, float(step_override_m))
                step_kind = "bounded_runtime_lift"

            command_step_m = (
                physical_step_m
                * self.sim_command_scale
            )

            action = (
                self.controller.action_for_atomic(
                    token,
                    step_m=command_step_m,
                )
            )

            intended_delta_m = (
                np.asarray(
                    self.controller.move_vectors[token],
                    dtype=float,
                )
                * physical_step_m
            )

            step_m = physical_step_m

            motion_steps = (
                self.sim_steps_per_decision
            )
            settle_steps = (
                self.settle_steps_per_decision
            )
            kind = "move"

        elif token in (
            GRASP_TOKEN,
            RELEASE_TOKEN,
        ):
            action = (
                self.controller.close_gripper()
                if token == GRASP_TOKEN
                else self.controller.open_gripper()
            )

            motion_steps = (
                self.gripper_hold_steps
                or self.sim_steps_per_decision
            )
            settle_steps = 0
            kind = "gripper"

        else:
            # STOP / unknown non-motion token -> one physical hold.
            action = self.controller.hold_action()
            motion_steps = 1
            settle_steps = 0
            kind = "stop"

        for _ in range(motion_steps):
            obs, terminated, truncated, _ = self._step_env(
                self._apply_orientation_hold(action, obs)
            )

            if terminated or truncated:
                break

        if (
            not terminated
            and not truncated
            and settle_steps
        ):
            hold = self.controller.hold_action()

            for _ in range(settle_steps):
                obs, terminated, truncated, _ = self._step_env(
                    self._apply_orientation_hold(hold, obs)
                )

                if terminated or truncated:
                    break

        post_pose = self._tcp(obs)

        gripper_closed = (
            self.controller.state.gripper_name
            == "CLOSE"
        )

        width_m = self._gripper_width(obs)

        # This is only a low-level contact veto.  ``empty_width_m`` is the
        # conservative nominal empty-close reading, while some LIBERO control
        # steps settle a little above/below that value.  The configured
        # ``gripper_close_threshold_m`` is a robot calibration for that closed,
        # no-object state; it is not an object holding-width rule.  Semantic
        # occupancy still comes from the visual Agent whenever the fingers are
        # visibly separated by an object.
        mechanical_empty_threshold = max(
            self.empty_width_m,
            self.gripper_close_threshold_m,
        )
        v22_enabled = bool(
            getattr(self.verified_runtime, "placement_v22_enabled", False)
        )
        grasp_empty = bool(
            token == GRASP_TOKEN
            and gripper_closed
            # In V2.2 aperture is diagnostic evidence only.  It varies with
            # object shape and contact load, so it cannot veto a semantic grasp
            # candidate by itself.
            and not v22_enabled
            and width_m <= mechanical_empty_threshold
        )

        if grasp_empty:
            note = (
                f"gripper closed empty: "
                f"width={width_m:.4f}m <= "
                f"mechanical threshold={mechanical_empty_threshold:.4f}m"
            )

        result = SimAtomicStepResult(
            token=token,
            kind=kind,
            pre_pose=pre_pose,
            post_pose=post_pose,
            intended_delta_m=intended_delta_m,
            step_m=step_m,
            step_kind=step_kind,
            gripper_closed=gripper_closed,
            grasp_empty=grasp_empty,
            done=False,
            note=note,
        )

        return (
            obs,
            terminated,
            truncated,
            result,
        )

    # ------------------------------------------------------------------
    # State/context adapters
    # ------------------------------------------------------------------

    def _images(self, obs: dict[str, Any]):
        agentview = prepare_view(
            self._rgb(obs, self.agentview_camera),
            rotation_degrees=self.agentview_rotation_degrees,
            flip=self.agentview_flip,
            crop_aspect=self.agentview_crop_aspect,
            square_size=self.agentview_square_size,
        )
        wrist = (
            prepare_view(
                self._rgb(obs, self.wrist_camera),
                rotation_degrees=self.wrist_rotation_degrees,
                flip=self.wrist_flip,
                crop_aspect=self.wrist_crop_aspect,
                square_size=self.wrist_square_size,
            )
            if self.use_wrist_image
            else None
        )
        if wrist is not None and self.wrist_grasp_marker.get("enabled", False):
            from core.sim.robolab_wrist_marker import annotate_wrist_grasp_point

            wrist, self.wrist_marker_metadata = annotate_wrist_grasp_point(
                wrist,
                marker_cfg=self.wrist_grasp_marker,
                raw_shape=self._rgb(obs, self.wrist_camera).shape,
                rotation_degrees=self.wrist_rotation_degrees,
                flip=self.wrist_flip,
                crop_aspect=self.wrist_crop_aspect,
                square_size=self.wrist_square_size,
            )
        # Both logging and VLM calls use this returned frame, including any annotation.
        return agentview, wrist

    def _fingertip_position(self, flange_position=None) -> np.ndarray:
        """Robot proprioception at the grasp point, in the env-local frame.

        ``rl_tcp`` is the controlled panda_hand body, which must remain unchanged for
        motion calibration. Height prompts need the finger grasp point instead. Rotate
        its local +Z offset by the measured hand quaternion; assuming a world vertical
        offset would become inaccurate whenever the hand tilts.
        """
        flange = np.asarray(
            self._tcp() if flange_position is None else flange_position,
            dtype=float,
        )
        quat = self._quat()
        norm = float(np.linalg.norm(quat))
        if not np.isfinite(norm) or norm < 1e-12:
            raise RuntimeError("Invalid hand quaternion in robot proprioception")
        w, x, y, z = quat / norm
        local_z_in_world = np.array(
            [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)]
        )
        return flange + self.fingertip_offset_m * local_z_in_world

    def _normalize_stage_token(self, token: str, *, subgoal, obs) -> str:
        """Allow a backend to apply a stage-level motion safety rule.

        The base runner preserves the VLM token exactly.  Backends with a different
        camera/control contract may override this hook; the override is recorded as
        the executed token by the normal step logger.
        """
        return str(token).strip().upper()

    def _proprio(self) -> dict[str, Any]:
        proprio = {
            "eef_pos": [
                float(x)
                for x in self._fingertip_position()
            ],
            "gripper_width": self._gripper_width(),
            "gripper_command_name": (
                "CLOSED"
                if self.controller.state.gripper_name
                == "CLOSE"
                else "OPEN"
            ),
        }

        if self._descend is not None:
            (
                proprio["descend_moved_m"],
                proprio["descend_commanded_m"],
            ) = self._descend

        return proprio

    def _observed_gripper_state(self) -> str:
        # The binary command is the reliable state exposed to the controller.
        # Width is deliberately not used as an OPEN/CLOSED test here: when an
        # object is held, the fingers remain visibly separated (about 0.037 m
        # for the LIBERO bottle), which is larger than the empty-close
        # threshold. The Agent uses the live views for occupancy; RecoveryPlugin
        # only rejects a clearly empty mechanical close.
        return (
            "CLOSED"
            if self.controller.state.gripper_name == "CLOSE"
            else "OPEN"
        )

    def _recovery_prompt(self, note: str) -> str:
        if self.recovery_plugin is None:
            return ""

        return (
            self.recovery_plugin
            .render_prompt_context(note)
        )

    def _recovery_before(
        self,
        *,
        current_index: int,
        subgoals,
    ):
        if self.recovery_plugin is None:
            return None

        return self.recovery_plugin.before_decision(
            current_index=current_index,
            subgoals=subgoals,
            measured_width_m=self._gripper_width(),
            gripper_closed=(
                self.controller.state.gripper_name
                == "CLOSE"
            ),
        )

    def _recovery_after(
        self,
        *,
        token: str,
        result: SimAtomicStepResult,
        current_index: int,
        subgoals,
        subgoal_done: bool,
    ):
        if self.recovery_plugin is None:
            return None

        return self.recovery_plugin.after_step(
            token=token,
            result=result,
            current_index=current_index,
            subgoals=subgoals,
            measured_width_m=self._gripper_width(),
            subgoal_done=subgoal_done,
            gripper_closed=bool(
                result.gripper_closed
            ),
        )

    def _verify_grasp_visually(self, *, subgoal, obs) -> dict[str, Any] | None:
        """Run one post-close visual confirmation through the high-level Agent."""
        controller = getattr(getattr(self, "controls", None), "controller", None)
        verifier = getattr(controller, "verify_grasp", None)
        if not callable(verifier):
            return None
        agentview, wrist = self._images(obs)
        before = self._pending_grasp_pre_action or {}
        v22 = bool(getattr(self.verified_runtime, "placement_v22_enabled", False))
        verify_kwargs = dict(
            task=self.task_description,
            target=str(getattr(subgoal, "target", "")),
            affordance=str(getattr(subgoal, "affordance", "")),
            agentview_image=agentview,
            wrist_image=wrist,
            debug=self.debug,
        )
        if v22:
            verify_kwargs.update(
                before_agentview_image=before.get("agentview"),
                before_wrist_image=before.get("wrist"),
                before_frame_id=before.get("frame_id"),
                after_frame_id=self._active_frame_id,
                reasoning_enabled=True,
            )
        return verifier(
            **verify_kwargs,
        )

    def _resolve_runtime_critical_decision(
        self,
        *,
        request: dict[str, Any],
        subgoal: Any,
        obs: Any,
        agentview: np.ndarray,
        placement_evidence: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Ask Qwen only for the enumerated semantic choice requested by VCR-v2."""
        kind = str(request.get("kind") or "").upper()
        if kind == "VERIFY_HOLD":
            result = self._verify_grasp_visually(subgoal=subgoal, obs=obs) or {}
            if bool(getattr(self.verified_runtime, "placement_v22_enabled", False)):
                self._pending_grasp_pre_action = None
            return {
                "answer": str(result.get("decision") or "UNKNOWN").upper(),
                "details": result,
            }
        if kind == "VERIFY_SEATED":
            memory_panel = None
            memory_meta = None
            if bool(getattr(self.verified_runtime, "placement_v22_enabled", False)):
                from core.runtime_v2.placement_memory import build_placement_panel

                belief = request.get("belief") or {}
                target_track = belief.get("target") or {}
                try:
                    memory_panel, memory_meta = build_placement_panel(
                        self.logger.run_dir,
                        list(request.get("visual_memory_bundle") or []),
                        instance_id=str(target_track.get("instance_id") or ""),
                        grasp_epoch=int(belief.get("grasp_epoch", -1)),
                        current_frame=int(belief.get("frame_id", -1)),
                        episode_id=str(self.logger.run_dir.resolve()),
                    )
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    return {"answer": "UNKNOWN", "details": {"error": f"placement memory unavailable: {exc}"}}
                if str(getattr(self.verified_runtime, "placement_reflection_mode", "off")) != "off":
                    self._critical_qwen_input = memory_panel
            result = self._verify_place_visually(
                subgoal=subgoal,
                obs=obs,
                placement_evidence=placement_evidence,
                memory_panel=memory_panel,
                memory_bundle=request.get("visual_memory_bundle"),
                reflection_mode=str(getattr(self.verified_runtime, "placement_reflection_mode", "off")),
                reflection_trigger=request.get("reflection_trigger"),
                allowed_options=request.get("allowed_options") or (),
            ) or {}
            if memory_meta is not None:
                result["memory_input"] = memory_meta
            relation = str(result.get("relation") or "").upper()
            remember_decision = getattr(getattr(self.verified_runtime, "visual_memory", None), "record_decision", None)
            belief = request.get("belief") if isinstance(request.get("belief"), dict) else {}
            target_track = belief.get("target") if isinstance(belief.get("target"), dict) else {}
            if callable(remember_decision):
                remember_decision(
                    instance_id=target_track.get("instance_id"),
                    grasp_epoch=int(belief.get("grasp_epoch", -1)),
                    frame_id=int(belief.get("frame_id", -1)),
                    route_epoch=int(belief.get("route_epoch", -1)),
                    relation=relation or "UNKNOWN",
                    reasoning=str(result.get("reasoning") or ""),
                    scene_description={
                        key: result.get(key)
                        for key in (
                            "evidence_for", "evidence_against", "missing_observation",
                            "expected_effect", "failure_condition", "scene_description",
                        )
                        if key in result
                    },
                    evidence_for=list(result.get("evidence_for") or []),
                    evidence_against=list(result.get("evidence_against") or []),
                    missing_observation=str(result.get("missing_observation") or ""),
                    expected_effect=str(result.get("expected_effect") or ""),
                    failure_condition=str(result.get("failure_condition") or ""),
                )
            return {
                "answer": relation or str(result.get("decision") or "UNKNOWN").upper(),
                "details": result,
            }
        if kind == "SELECT_INSTANCE":
            resolver = getattr(self.controls.controller, "resolve_instance", None)
            if not callable(resolver):
                return {"answer": "UNKNOWN", "reason": "resolver unavailable"}
            live_agentview, live_wrist = self._images(obs)
            candidate_camera = str(request.get("camera") or "agentview").lower()
            candidate_image = (
                live_wrist
                if candidate_camera == "wrist" and live_wrist is not None
                else live_agentview
            )
            result = resolver(
                task=self.task_description,
                target=str(getattr(subgoal, "target", "")),
                candidates=request.get("candidates") or [],
                agentview_image=candidate_image,
                other_view_image=(
                    live_agentview
                    if candidate_camera == "wrist"
                    else live_wrist
                ),
                candidate_camera=candidate_camera,
                debug=self.debug,
            ) or {}
            return {
                "answer": str(result.get("selected") or "UNKNOWN").upper(),
                "details": result,
            }
        if kind == "PREGRASP_DECISION":
            resolver = getattr(self.controls.controller, "resolve_pregrasp", None)
            if not callable(resolver):
                return {"answer": "UNKNOWN", "reason": "pregrasp resolver unavailable"}
            live_agentview, live_wrist = self._images(obs)
            memory_panel = None
            memory_meta = None
            crop_meta = None
            placement_v22 = bool(
                getattr(self.verified_runtime, "placement_v22_enabled", False)
            )
            crop_enabled = bool(
                getattr(self.verified_runtime, "pregrasp_target_crop_enabled", False)
            )
            memory_bundle = list(request.get("visual_memory_bundle") or [])
            if memory_bundle:
                from core.runtime_v2.placement_memory import build_visual_memory_panel
                from core.vlm.roles import append_fresh_pregrasp_target_crop

                belief = request.get("belief") or {}
                target_track = belief.get("target") or {}
                try:
                    memory_panel, memory_meta = build_visual_memory_panel(
                        self.logger.run_dir,
                        memory_bundle,
                        instance_id=str(target_track.get("instance_id") or ""),
                        grasp_epoch=int(belief.get("grasp_epoch", -1)),
                        current_frame=int(belief.get("frame_id", -1)),
                        episode_id=str(self.logger.run_dir.resolve()),
                    )
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    return {
                        "answer": "UNKNOWN",
                        "details": {
                            "reason": f"same-episode visual memory unavailable: {exc}",
                        },
                    }
                if tuple(memory_meta.get("frame_ids", ())) != tuple(request.get("evidence_frame_ids") or ()):
                    return {
                        "answer": "UNKNOWN",
                        "details": {
                            "reason": "memory panel frame ids do not match runtime evidence contract",
                            "memory_audit": memory_meta,
                        },
                    }
                if crop_enabled:
                    memory_panel, crop_meta = append_fresh_pregrasp_target_crop(
                        memory_panel,
                        target_track,
                        int(belief.get("frame_id", -1)),
                        agentview_image=live_agentview,
                        wrist_image=live_wrist,
                    )
                    if crop_meta is not None:
                        memory_meta["target_crop"] = crop_meta
                        memory_meta["sent_panel_sha256"] = hashlib.sha256(
                            np.asarray(memory_panel).tobytes()
                        ).hexdigest()
                        memory_meta["sent_panel_shape"] = list(np.asarray(memory_panel).shape)
                self._critical_qwen_input = memory_panel
            else:
                from core.vlm.roles import _make_pregrasp_temporal_panel
                self._critical_qwen_input = _make_pregrasp_temporal_panel(
                    live_agentview, live_wrist,
                    getattr(self, "_previous_agentview", None),
                    getattr(self, "_previous_wrist", None),
                )
                crop_meta = None
            result = resolver(
                task=self.task_description,
                target=str(getattr(subgoal, "target", "")),
                affordance=str(getattr(subgoal, "affordance", "")),
                allowed_actions=request.get("allowed_answers") or [],
                runtime_reason=str(request.get("reason") or ""),
                agentview_image=live_agentview,
                wrist_image=live_wrist,
                previous_agentview_image=getattr(self, "_previous_agentview", None),
                previous_wrist_image=getattr(self, "_previous_wrist", None),
                spatial_belief=request.get("spatial_belief"),
                visual_alignment=request.get("visual_alignment"),
                executed_action=getattr(self, "_previous_executed_action", None),
                visual_memory=request.get("visual_memory_refs") or (),
                visual_memory_bundle=memory_bundle,
                reflection_mode=str(request.get("reflection_mode") or "off"),
                reflection_trigger=(
                    str(request.get("reflection_trigger"))
                    if request.get("reflection_trigger") else None
                ),
                allow_visual_grasp_without_spatial=(
                    placement_v22
                    and not bool(getattr(self.verified_runtime, "require_spatial_ready_for_grasp", True))
                ),
                prebuilt_memory_panel=memory_panel,
                target_roi_included=bool(crop_meta),
                current_frame_id=int((request.get("belief") or {}).get("frame_id", -1)),
                previous_frame_id=(
                    int(self._previous_frame_id)
                    if self._previous_frame_id is not None
                    and int(self._previous_frame_id) in set(request.get("evidence_frame_ids") or ())
                    else None
                ),
                debug=self.debug,
            ) or {}
            if memory_meta is not None:
                result["memory_audit"] = memory_meta
            return {
                "answer": str(result.get("selected") or "UNKNOWN").upper(),
                "details": result,
            }
        return {"answer": "UNKNOWN", "reason": f"unsupported critical kind {kind}"}

    def _verify_place_visually(
        self,
        *,
        subgoal,
        obs,
        placement_evidence: Optional[dict[str, Any]] = None,
        memory_panel: Optional[np.ndarray] = None,
        memory_bundle: Optional[list[dict[str, Any]]] = None,
        reflection_mode: str = "off",
        reflection_trigger: Optional[str] = None,
        allowed_options: Optional[tuple[str, ...] | list[str]] = None,
    ) -> dict[str, Any] | None:
        """Run one low-position placement confirmation through the high-level Agent."""
        controller = getattr(getattr(self, "controls", None), "controller", None)
        verifier = getattr(controller, "verify_place", None)
        if not callable(verifier):
            return None
        agentview, wrist = self._images(obs)
        verify_kwargs = dict(
            task=self.task_description,
            target=str(getattr(subgoal, "target", "")),
            affordance=str(getattr(subgoal, "affordance", "")),
            agentview_image=agentview,
            wrist_image=wrist,
            debug=self.debug,
        )
        if bool(getattr(self.verified_runtime, "placement_v22_enabled", False)):
            verify_kwargs["placement_v22"] = True
            if isinstance(placement_evidence, dict):
                verify_kwargs["placement_evidence"] = placement_evidence
            verify_kwargs.update(
                memory_panel=memory_panel,
                memory_bundle=memory_bundle,
                reflection_mode=reflection_mode,
                reflection_trigger=reflection_trigger,
                allowed_options=allowed_options,
            )
        return verifier(**verify_kwargs)

    def _review_place_alignment(
        self, *, subgoal, obs, recent_moves: str = ""
    ) -> dict[str, Any] | None:
        """Ask the high-level Agent for a visual pre-placement alignment action."""
        controller = getattr(getattr(self, "controls", None), "controller", None)
        reviewer = getattr(controller, "review_place_alignment", None)
        if not callable(reviewer):
            return None
        agentview, wrist = self._images(obs)
        return reviewer(
            task=self.task_description,
            target=str(getattr(subgoal, "target", "")),
            affordance=str(getattr(subgoal, "affordance", "")),
            agentview_image=agentview,
            wrist_image=wrist,
            recent_moves=recent_moves,
            debug=self.debug,
        )

    def _verify_task_visually(self, *, obs) -> dict[str, Any] | None:
        """Ask the Agent whether the whole task is visibly complete."""
        controller = getattr(getattr(self, "controls", None), "controller", None)
        verifier = getattr(controller, "verify_task", None)
        if not callable(verifier):
            return None
        agentview, wrist = self._images(obs)
        return verifier(
            task=self.task_description,
            agentview_image=agentview,
            wrist_image=wrist,
            debug=self.debug,
        )

    @staticmethod
    def _grasp_view_relations(
        evidence: dict[str, Any] | None,
    ) -> dict[str, tuple[float, float]]:
        """Extract comparable AgentView/Wrist image errors for grasp retries.

        This deliberately records only image-space relation, not object size or a
        scene coordinate.  It lets the runner reject a repeated empty-close
        hypothesis while still allowing a new Agent-chosen correction or a fresh
        Wrist observation to change the relation.
        """
        if not isinstance(evidence, dict):
            return {}
        relations: dict[str, tuple[float, float]] = {}
        primary_camera = str(evidence.get("camera", "")).lower()
        geometry = evidence.get("geometry")
        primary_error = geometry.get("target_minus_eef_px") if isinstance(geometry, dict) else None
        if (
            primary_camera in {"agentview", "wrist"}
            and isinstance(primary_error, (list, tuple))
            and len(primary_error) == 2
        ):
            try:
                relations[primary_camera] = (
                    float(primary_error[0]),
                    float(primary_error[1]),
                )
            except (TypeError, ValueError):
                pass
        secondary = evidence.get("secondary_view")
        if isinstance(secondary, dict):
            secondary_camera = str(secondary.get("camera", "")).lower()
            secondary_error = secondary.get("target_minus_eef_px")
            if (
                secondary.get("visible", False)
                and secondary_camera in {"agentview", "wrist"}
                and isinstance(secondary_error, (list, tuple))
                and len(secondary_error) == 2
            ):
                try:
                    relations[secondary_camera] = (
                        float(secondary_error[0]),
                        float(secondary_error[1]),
                    )
                except (TypeError, ValueError):
                    pass
        return relations

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @staticmethod
    def _projection_from_calibration(meta: dict[str, Any]) -> Optional[np.ndarray]:
        """Build a metric world-to-image projection for active parallax."""
        try:
            width = int(meta["width"])
            height = int(meta["height"])
            fovy = float(meta["fovy_deg"])
            position = np.asarray(meta["position_world"], dtype=float).reshape(3)
            camera_to_world = np.asarray(meta["camera_to_world"], dtype=float).reshape(3, 3)
            if width <= 0 or height <= 0 or not np.all(np.isfinite(position)):
                return None
            focal = (height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
            intrinsic = np.asarray(
                [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
                dtype=float,
            )
            # Camera geometry uses MuJoCo's camera-forward -Z convention.
            world_to_forward = np.diag([1.0, -1.0, -1.0]) @ camera_to_world.T
            return intrinsic @ np.column_stack((world_to_forward, -world_to_forward @ position))
        except (TypeError, ValueError, KeyError, IndexError, FloatingPointError):
            return None

    @staticmethod
    def _raw_pixels_from_view(
        pixels: np.ndarray, meta: dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Undo the exact post-render rotation/flip used by ``prepare_view``."""
        try:
            width = int(meta["width"])
            height = int(meta["height"])
            values = np.asarray(pixels, dtype=float).reshape(-1, 2).copy()
            output_width = width if int(meta.get("rotation_degrees", 0)) % 180 == 0 else height
            output_height = height if int(meta.get("rotation_degrees", 0)) % 180 == 0 else width
            mode = str(meta.get("flip", "none")).lower()
            if mode in {"vertical", "both"}:
                values[:, 1] = output_height - 1 - values[:, 1]
            if mode in {"horizontal", "both"}:
                values[:, 0] = output_width - 1 - values[:, 0]
            k = (int(meta.get("rotation_degrees", 0)) % 360) // 90
            x = values[:, 0].copy()
            y = values[:, 1].copy()
            if k == 1:
                values[:, 0], values[:, 1] = width - 1 - y, x
            elif k == 2:
                values[:, 0], values[:, 1] = width - 1 - x, height - 1 - y
            elif k == 3:
                values[:, 0], values[:, 1] = y, height - 1 - x
            return values if np.all(np.isfinite(values)) else None
        except (TypeError, ValueError, KeyError, IndexError):
            return None

    def _active_parallax_result(
        self,
        *,
        capability_evidence: dict[str, Any],
        geometry_context: dict[str, Any],
        frame_id: int,
        instance_id: Optional[str],
        grasp_epoch: int,
        eef_xyz: tuple[float, float, float],
        tracked_points: Optional[dict[str, Any]] = None,
    ) -> Optional[SpatialToolResult]:
        """Triangulate only CoTracker correspondences from a free target mask."""
        if str(capability_evidence.get("camera", "")).lower() != "wrist":
            return None
        runtime_belief = getattr(self.verified_runtime, "belief", None)
        held = getattr(runtime_belief, "held", None)
        if str(getattr(getattr(held, "truth", None), "value", "")).upper() == "TRUE":
            return None
        meta = geometry_context.get("wrist", {}).get("camera_calibration", {})
        if not isinstance(meta, dict):
            return None
        projection = self._projection_from_calibration(meta)
        if projection is None:
            return None
        epoch = int(grasp_epoch)
        if getattr(self, "_parallax_epoch", None) != epoch:
            self._parallax_epoch = epoch
            self._parallax_history = deque(maxlen=32)
            self._parallax_world_history = deque(maxlen=5)
        current = {
            "frame_id": int(frame_id),
            "instance_id": instance_id,
            "projection": projection,
            "eef": np.asarray(eef_xyz, dtype=float),
        }
        history = list(getattr(self, "_parallax_history", ()))
        if not history or int(history[-1].get("frame_id", -1)) != int(frame_id):
            self._parallax_history.append(current)
        tracking = tracked_points if isinstance(tracked_points, dict) else {}
        if str(tracking.get("health", "")).upper() != "VALID":
            return None
        query_frame = tracking.get("query_frame_id")
        prior = next(
            (item for item in history if int(item.get("frame_id", -1)) == int(query_frame or -1)),
            None,
        )
        if prior is None or prior.get("instance_id") != instance_id:
            return None

        def camera_center(value: Any) -> Optional[np.ndarray]:
            try:
                _, _, vt = np.linalg.svd(np.asarray(value, dtype=float).reshape(3, 4))
                homogeneous = vt[-1]
                if abs(float(homogeneous[3])) <= 1e-9:
                    return None
                return homogeneous[:3] / homogeneous[3]
            except (TypeError, ValueError, np.linalg.LinAlgError):
                return None

        center_a = camera_center(prior["projection"])
        center_b = camera_center(current["projection"])
        if center_a is None or center_b is None:
            return None
        baseline = float(np.linalg.norm(center_a - center_b))
        if baseline < 0.006:
            return None
        raw_query = self._raw_pixels_from_view(
            np.asarray(tracking.get("query_points_xy"), dtype=float), meta
        )
        raw_current = self._raw_pixels_from_view(
            np.asarray(tracking.get("current_points_xy"), dtype=float), meta
        )
        if raw_query is None or raw_current is None or raw_query.shape != raw_current.shape:
            return None
        checked = triangulate_metric_points_checked(
            raw_query,
            raw_current,
            prior["projection"],
            current["projection"],
            min_correspondences=4,
            max_reprojection_error_px=3.0,
        )
        if checked is None:
            return None
        center = np.asarray(checked["center_world"], dtype=float)
        surface_spread = float(checked["surface_spread_m"])
        reprojection_error = float(checked["median_reprojection_error_px"])
        table_z = float(self.table_height_m)
        eef = np.asarray(current["eef"], dtype=float)
        if (
            not np.all(np.isfinite(center))
            or not np.isfinite(surface_spread)
            or not (table_z - 0.04 <= float(center[2]) <= float(eef[2]) + 0.08)
        ):
            return None
        self._parallax_world_history.append(center)
        robust_center = np.median(np.stack(list(self._parallax_world_history)), axis=0)
        relative = robust_center - eef
        focal = max(float(np.linalg.norm(np.asarray(projection)[0, :3])), 1.0)
        depth = max(float(np.linalg.norm(robust_center - center_b)), 0.01)
        sigma_m = max(0.003, (depth * depth / (focal * baseline)) * max(reprojection_error, 0.75))
        return SpatialToolResult(
            source="active_parallax",
            instance_id=instance_id,
            frame_id=frame_id,
            target_points_world=(tuple(float(v) for v in robust_center),),
            target_to_gripper_xyz=tuple(float(v) for v in relative),
            uncertainty_std_m=(sigma_m,) * 3,
            confidence=float(np.clip(1.0 - reprojection_error / 3.0, 0.0, 1.0)),
            health=SpatialHealth.VALID,
            diagnostics={
                "query_frame_id": int(query_frame),
                "correspondence_count": int(checked["valid_correspondences"]),
                "baseline_m": baseline,
                "surface_spread_m": surface_spread,
                "median_reprojection_error_px": reprojection_error,
                "estimated_uncertainty_m": sigma_m,
            },
        )

    def run(self) -> EpisodeResult:
        reset_history = getattr(self.agent, "reset", None)
        if callable(reset_history):
            reset_history()
        self.controller.set_orientation_reference(
            None
        )

        obs, terminated, truncated = self._reset_episode()
        self.controller.set_orientation_reference(self._quat(obs))
        success = self._success(obs)

        end_reason = "max_steps_exceeded"
        steps = 0
        raw_plan = ""
        subgoals = []
        replans = 0
        error = None

        video_path: Any = (
            self.logger.run_dir
            / "rollout_failure.mp4"
        )

        current_index = 0
        subgoal_start_step = 0

        recent_moves: list[str] = []
        previous_direction: str | None = None

        recovery_note = ""
        chunk_queue: list[str] = []
        self._reacquire_required = False
        self._grasp_retry_anchor = None
        self._previous_agentview = None
        self._previous_wrist = None
        self._previous_frame_id = None
        self._previous_executed_action = None
        self._pending_grasp_pre_action = None
        self._active_frame_id = None
        self._parallax_epoch = None
        self._parallax_history = deque(maxlen=8)
        self._current_stage = ""
        self._last_anyplace_shadow_frame = -10_000

        if self.visual_route_plugin is not None:
            reset_route = getattr(self.visual_route_plugin, "reset", None)
            if callable(reset_route):
                reset_route()

        self._descend = None
        if self.recovery_plugin is not None:
            reset_recovery = getattr(self.recovery_plugin, "reset", None)
            if callable(reset_recovery):
                reset_recovery()
        if self.visual_harness is not None:
            self.visual_harness.reset()
        if self.verified_runtime is not None:
            # V2.1 and V2.2 both use current-episode visual memory. Bind every
            # VCR memory instance to this unique logger directory so panel
            # resolution verifies the same episode that wrote the raw frames.
            _reset_runtime_episode(self.verified_runtime, self.logger)

        try:
            agentview, wrist = self._images(obs)

            image_roles = [
                (
                    "LIVE AgentView: authoritative global "
                    "scene and object positions."
                )
            ]

            if wrist is not None:
                image_roles.append(
                    (
                        "LIVE Wrist view: local gripper "
                        "detail only; do not infer hidden "
                        "global positions."
                    )
                )

            subgoals, raw_plan = self.planner.plan(
                self.task_description,
                agentview,
                wrist=wrist,
                debug=self.debug,
                image_roles=image_roles,
            )

            diagnostics = self.planner.diagnostics()
            self.logger.write_planner_diagnostics(diagnostics)
            if diagnostics.get("route") == "task_fallback":
                raise RuntimeError(
                    "Planner exhausted model retries; refusing a scripted task fallback. "
                    f"Diagnostics: {diagnostics}"
                )

            if not subgoals:
                raise RuntimeError(
                    "Planner returned no subgoals"
                )

            planner_subgoals = list(subgoals)
            if bool(
                self.visual_route_plugin is not None
                and getattr(self.visual_route_plugin, "enabled", False)
                and getattr(
                    self.visual_route_plugin,
                    "transport_loop_enabled",
                    False,
                )
            ):
                subgoals = self.visual_route_plugin.collapse_transport_subgoals(
                    subgoals
                )

            self.logger.write_plan(
                {
                    "task": self.task_description,
                    "raw_plan": raw_plan,
                    "planner_subgoals": [
                        sg.to_prompt_dict()
                        for sg in planner_subgoals
                    ],
                    "subgoals": [
                        sg.to_prompt_dict()
                        for sg in subgoals
                    ],
                }
            )

            save_planner_prompt = getattr(
                self.logger,
                "save_planner_prompt",
                None,
            )

            planner_prompt = getattr(
                self.planner,
                "last_prompt",
                None,
            )

            if (
                callable(save_planner_prompt)
                and callable(planner_prompt)
            ):
                try:
                    save_planner_prompt(
                        1,
                        str(planner_prompt()),
                    )
                except Exception as exc:
                    print(
                        "[zeroshot-robolab] "
                        "could not save planner prompt:",
                        exc,
                    )

            print("\n[zeroshot-robolab] plan:")

            for i, sg in enumerate(subgoals):
                print(
                    f"  [{i}] {sg.motion}: "
                    f"target={sg.target!r} "
                    f"affordance={sg.affordance!r}",
                    flush=True,
                )

            for step_idx in range(
                self.max_steps
            ):
                steps = step_idx + 1
                loop_started = time.monotonic()

                subgoal = subgoals[
                    current_index
                ]
                self._active_frame_id = int(step_idx)

                subgoal_step = (
                    step_idx
                    - subgoal_start_step
                )

                # An unfinished stage is a failed attempt, never a skipped stage.
                if (
                    subgoal_step
                    >= self.config.max_subgoal_steps
                ):
                    print(
                        "[zeroshot-robolab] "
                        f"subgoal {current_index} "
                        "hit step cap "
                        f"({self.config.max_subgoal_steps})"
                    )

                    steps = step_idx
                    end_reason = "subgoal_step_cap_exceeded"
                    break

                agentview, wrist = (
                    self._images(obs)
                )
                # Keep the detector/runtime input immutable.  Route overlays are
                # useful to the Agent and video, but must never feed back into
                # instance association or appearance descriptors.
                raw_agentview = agentview
                raw_wrist = wrist
                self._critical_qwen_input = None
                image_refs, raw_frames_saved_early = _save_runtime_memory_frame(
                    self.logger,
                    self.verified_runtime,
                    step_idx,
                    raw_agentview=raw_agentview,
                    raw_wrist=raw_wrist,
                )

                capability_evidence = {}
                capability_context = ""
                verified_runtime_decision: dict[str, Any] = {}
                visual_route_output: dict[str, Any] = {}
                visual_route_gate: Any = None
                transport_hold_lost = False
                transport_rollback_index = None
                route_holding_arbiter: dict[str, Any] = {}
                holding_recovery_arbiter: list[dict[str, Any]] = []
                runtime_rollback_index: Optional[int] = None
                route_held_evidence: Optional[dict[str, Any]] = None
                route_destination_evidence: Optional[dict[str, Any]] = None
                held_target, held_affordance = self._held_target_for(
                    subgoals, current_index
                )
                current_gripper_width_m = None
                try:
                    current_gripper_width_m = float(self._gripper_width())
                except (TypeError, ValueError):
                    current_gripper_width_m = None
                geometry_context = self._visual_geometry(obs, agentview, wrist)
                if self.visual_harness is not None:
                    capability_evidence = self.visual_harness.update(
                        agentview=agentview,
                        wrist=wrist,
                        stage=str(subgoal.motion),
                        target=str(subgoal.target),
                        affordance=str(subgoal.affordance),
                        frame_id=step_idx,
                        previous_action=(
                            self.visual_harness.last_action
                        ),
                        geometry=geometry_context,
                        held_target=held_target,
                        held_affordance=held_affordance,
                    )
                    capability_context = self.visual_harness.prompt_context()

                # Ambiguous SAM3 detections still carry per-candidate masks.
                # Resolve those masks only through the existing same-camera
                # target track; never compare AgentView and Wrist pixels.
                transient_candidate_masks = {}
                if isinstance(capability_evidence, dict):
                    transient_candidate_masks = capability_evidence.pop(
                        "_transient_candidate_masks_by_camera", {}
                    )
                secondary_view = (
                    capability_evidence.get("secondary_view")
                    if isinstance(capability_evidence, dict)
                    else None
                )
                secondary_association = None
                secondary_target_mask = None
                preview_secondary = getattr(
                    self.verified_runtime,
                    "associate_secondary_target_view",
                    None,
                )
                if (
                    callable(preview_secondary)
                    and isinstance(secondary_view, dict)
                    and str(getattr(subgoal, "motion", "")).upper() == "GRASP"
                ):
                    secondary_camera = str(secondary_view.get("camera") or "").lower()
                    secondary_image = {
                        "agentview": raw_agentview,
                        "wrist": raw_wrist,
                    }.get(secondary_camera)
                    secondary_association = preview_secondary(
                        evidence={**secondary_view, "stage": "GRASP"},
                        image=secondary_image,
                        camera=secondary_camera,
                        frame_id=step_idx,
                        commit=False,
                    )
                    if secondary_association.get("health") == "VALID":
                        secondary_view["bbox_xyxy"] = secondary_association.get(
                            "bbox_xyxy"
                        )
                        secondary_view["visible"] = True
                        secondary_view["instance_association"] = secondary_association
                        # Preserve a camera-local EEF projection for the
                        # separately associated view.  This lets V2.1 expose a
                        # bounded image-space correction when metric geometry
                        # is unavailable, without mixing Wrist and AgentView
                        # pixels or trusting an unassociated detector proposal.
                        secondary_camera_geometry = (
                            geometry_context.get(secondary_camera, {})
                            if isinstance(geometry_context, dict)
                            else {}
                        )
                        secondary_bbox = secondary_association.get("bbox_xyxy")
                        eef_pixel = secondary_camera_geometry.get("pixel_xy")
                        direction_map = secondary_camera_geometry.get(
                            "screen_direction_to_token"
                        )
                        if (
                            isinstance(secondary_bbox, (list, tuple))
                            and len(secondary_bbox) == 4
                            and isinstance(eef_pixel, (list, tuple))
                            and len(eef_pixel) == 2
                            and isinstance(direction_map, dict)
                            and bool(secondary_camera_geometry.get("valid", False))
                            and bool(secondary_camera_geometry.get("in_frame", False))
                        ):
                            x1, y1, x2, y2 = (
                                float(value) for value in secondary_bbox
                            )
                            dx = (x1 + x2) / 2.0 - float(eef_pixel[0])
                            dy = (y1 + y2) / 2.0 - float(eef_pixel[1])
                            horizontal = str(
                                direction_map.get("right" if dx > 0 else "left", "")
                            ).upper()
                            vertical = str(
                                direction_map.get("down" if dy > 0 else "up", "")
                            ).upper()
                            secondary_view["geometry"] = {
                                "valid": True,
                                "in_frame": True,
                                "frame_id": int(step_idx),
                                "camera": secondary_camera,
                                "pixel_xy": list(eef_pixel),
                                "target_center_xy": [
                                    (x1 + x2) / 2.0,
                                    (y1 + y2) / 2.0,
                                ],
                                "target_minus_eef_px": [dx, dy],
                                "calibrated_correction_candidates": {
                                    "horizontal": horizontal,
                                    "vertical": vertical,
                                },
                                "screen_direction_to_token": dict(direction_map),
                                "camera_calibration": secondary_camera_geometry.get(
                                    "camera_calibration"
                                ),
                            }
                        candidate_rows = (
                            transient_candidate_masks.get(secondary_camera, [])
                            if isinstance(transient_candidate_masks, dict)
                            else []
                        )
                        selected_box = secondary_association.get("bbox_xyxy")
                        for row in candidate_rows:
                            if not isinstance(row, dict) or not isinstance(row.get("mask"), dict):
                                continue
                            try:
                                same_box = np.allclose(
                                    np.asarray(row.get("bbox_xyxy"), dtype=float),
                                    np.asarray(selected_box, dtype=float),
                                    rtol=0.0,
                                    atol=1e-3,
                                )
                            except (TypeError, ValueError):
                                same_box = False
                            if same_box:
                                secondary_target_mask = row["mask"]
                                break
                        if secondary_target_mask is not None:
                            secondary_view["mask"] = secondary_target_mask
                            capability_evidence["secondary_target_mask_audit"] = {
                                "camera": secondary_camera,
                                "frame_id": int(step_idx),
                                "instance_id": secondary_association.get("instance_id"),
                                "bbox_xyxy": selected_box,
                                "area_px": secondary_target_mask.get("area_px"),
                                "source": "sam3_mask_temporally_associated",
                            }

                # Keep one online CoTracker stream per camera for the selected
                # target throughout approach and grasp. The mask is only used
                # to seed that camera's point queries; later frames can continue
                # the same stream without mixing AgentView and Wrist pixels.
                runtime_belief = getattr(self.verified_runtime, "belief", None)
                target_track = getattr(runtime_belief, "target", None)
                target_instance_id = getattr(target_track, "instance_id", None)
                grasp_epoch = int(getattr(runtime_belief, "grasp_epoch", 0) or 0)
                track_visual_points = getattr(
                    self.verified_runtime, "track_visual_points", None
                )
                track_points_by_camera = (
                    _track_v22_target_views(
                        tracker=track_visual_points,
                        stage=str(getattr(subgoal, "motion", "")),
                        capability_evidence=capability_evidence,
                        camera_images={
                            "agentview": raw_agentview,
                            "wrist": raw_wrist,
                        },
                        frame_id=step_idx,
                        instance_id=target_instance_id,
                        grasp_epoch=grasp_epoch,
                    )
                    if isinstance(capability_evidence, dict)
                    else {}
                )
                if track_points_by_camera:
                    capability_evidence["visual_point_tracks_by_camera"] = (
                        track_points_by_camera
                    )
                    primary_camera = str(
                        capability_evidence.get("camera", "")
                    ).lower()
                    if primary_camera in track_points_by_camera:
                        capability_evidence["visual_point_tracks"] = (
                            track_points_by_camera[primary_camera]
                        )

                # Optional V2.1 spatial providers are queried only at the
                # bounded pregrasp decision point.  Their result is evidence,
                # never a direct action or success write.
                spatial_infer = getattr(self.verified_runtime, "infer_spatial", None)
                if (
                    callable(spatial_infer)
                    and str(getattr(subgoal, "motion", "")).upper() == "GRASP"
                    # MoGe/active-parallax evidence is only trusted from the
                    # eye-in-hand view in this profile.  Fixed AgentView has a
                    # different projective scale and its fallback monocular
                    # depth is not a metric grasp residual.  Keep the last
                    # fresh Wrist belief while AgentView remains useful for
                    # semantic visibility and target identity.
                    and str(capability_evidence.get("camera", "")).lower() == "wrist"
                    and isinstance(capability_evidence.get("bbox_xyxy"), (list, tuple))
                    and isinstance(capability_evidence.get("mask"), dict)
                ):
                    spatial_image = raw_wrist
                    track_points = track_points_by_camera.get("wrist")
                    parallax = self._active_parallax_result(
                        capability_evidence=capability_evidence,
                        geometry_context=geometry_context,
                        frame_id=step_idx,
                        instance_id=getattr(target_track, "instance_id", None),
                        grasp_epoch=int(getattr(runtime_belief, "grasp_epoch", 0) or 0),
                        eef_xyz=tuple(
                            float(value)
                            for value in self._fingertip_position(self._tcp(obs))[:3]
                        ),
                        tracked_points=track_points,
                    )
                    spatial_belief = spatial_infer(
                        image=spatial_image,
                        bbox_xyxy=tuple(capability_evidence.get("bbox_xyxy")) if isinstance(capability_evidence.get("bbox_xyxy"), (list, tuple)) and len(capability_evidence.get("bbox_xyxy")) == 4 else None,
                        frame_id=step_idx,
                        instance_id=(
                            getattr(getattr(self.verified_runtime, "belief", None), "target", None).instance_id
                            if getattr(getattr(self.verified_runtime, "belief", None), "target", None) is not None
                            else None
                        ),
                        camera_calibration=(
                            geometry_context.get(str(capability_evidence.get("camera", "agentview")).lower(), {}).get("camera_calibration", {})
                            if isinstance(geometry_context, dict)
                            else {}
                        ),
                        eef_xyz=tuple(float(value) for value in self._fingertip_position(self._tcp(obs))[:3]),
                        extra_results=([parallax] if parallax is not None else None),
                        instance_mask=capability_evidence.get("mask"),
                    )
                    if isinstance(capability_evidence, dict):
                        capability_evidence["spatial_belief"] = spatial_belief

                # V2.1 can optionally use a fresh, identity-associated
                # AgentView mask for MoGe.  The provider still has to pass its
                # calibrated pixel/world reprojection checks in infer_spatial;
                # this is evidence only and never directly authorizes a close.
                secondary_bbox = (
                    secondary_association.get("bbox_xyxy")
                    if isinstance(secondary_association, dict)
                    and secondary_association.get("health") == "VALID"
                    else None
                )
                if (
                    callable(spatial_infer)
                    and bool(
                        getattr(
                            self.verified_runtime,
                            "allow_agentview_identity_fallback",
                            False,
                        )
                    )
                    and str(getattr(subgoal, "motion", "")).upper() == "GRASP"
                    and isinstance(secondary_bbox, (list, tuple))
                    and len(secondary_bbox) == 4
                    and isinstance(secondary_target_mask, dict)
                    and raw_agentview is not None
                ):
                    agentview_calibration = (
                        geometry_context.get("agentview", {}).get(
                            "camera_calibration", {}
                        )
                        if isinstance(geometry_context, dict)
                        else {}
                    )
                    secondary_spatial = spatial_infer(
                        image=raw_agentview,
                        bbox_xyxy=tuple(float(value) for value in secondary_bbox),
                        frame_id=step_idx,
                        instance_id=(
                            secondary_association.get("instance_id")
                            if isinstance(secondary_association, dict)
                            else None
                        ),
                        camera_calibration=agentview_calibration,
                        eef_xyz=tuple(
                            float(value)
                            for value in self._fingertip_position(self._tcp(obs))[:3]
                        ),
                        instance_mask=secondary_target_mask,
                    )
                    capability_evidence["secondary_spatial_belief"] = {
                        **secondary_spatial,
                        "camera": "agentview",
                    }
                    if (
                        str(secondary_spatial.get("health", "")).upper() == "VALID"
                        and int(secondary_spatial.get("frame_id", -1)) == int(step_idx)
                        and str(secondary_spatial.get("instance_id") or "")
                        == str(secondary_association.get("instance_id") or "")
                        and str(
                            (capability_evidence.get("spatial_belief") or {}).get(
                                "health", "UNKNOWN"
                            )
                        ).upper()
                        != "VALID"
                    ):
                        capability_evidence["spatial_belief"] = secondary_spatial

                # Candidate RLE is transient input to CoTracker/MoGe. Keep only
                # its compact provenance audit in serialized step evidence.
                if isinstance(secondary_view, dict):
                    secondary_view.pop("mask", None)

                # V1's Phase-1 runtime keeps its original position in the loop.
                # VCR-v2 implements ``observe_frame`` and is deferred until the
                # route plugin has added its verified transport residuals below.
                runtime_observe_frame = (
                    getattr(self.verified_runtime, "observe_frame", None)
                    if self.verified_runtime is not None
                    else None
                )
                if self.verified_runtime is not None and not callable(
                    runtime_observe_frame
                ):
                    verified_runtime_decision = self.verified_runtime.observe(
                        stage=str(subgoal.motion),
                        evidence=(
                            capability_evidence
                            if isinstance(capability_evidence, dict)
                            else {}
                        ),
                        previous_action=(
                            self.visual_harness.last_action
                            if self.visual_harness is not None
                            else None
                        ),
                    )
                    if isinstance(capability_evidence, dict):
                        capability_evidence["verified_runtime"] = dict(
                            verified_runtime_decision
                        )

                if bool(
                    self.visual_route_plugin is not None
                    and getattr(self.visual_route_plugin, "enabled", False)
                ):
                    route_held, route_destination = self._visual_route_inputs(
                        agentview=agentview,
                        capability_evidence=(
                            capability_evidence
                            if isinstance(capability_evidence, dict)
                            else {}
                        ),
                        stage=str(subgoal.motion),
                        subgoals=subgoals,
                        current_index=current_index,
                        held_target=held_target,
                        held_affordance=held_affordance,
                    )
                    route_held_evidence = route_held
                    route_destination_evidence = route_destination
                    # CoTracker's online window is 16 frames with an 8-frame
                    # update cadence. A short GRASP subgoal often ends before
                    # its first useful correspondence, so continue the held
                    # instance stream through TRANSPORT using its fresh
                    # AgentView mask. Tracks are logged as visual motion
                    # evidence; they do not create metric geometry or actions.
                    runtime_belief = getattr(self.verified_runtime, "belief", None)
                    target_track = getattr(runtime_belief, "target", None)
                    track_visual_points = getattr(
                        self.verified_runtime, "track_visual_points", None
                    )
                    if (
                        str(getattr(subgoal, "motion", "")).upper() == "TRANSPORT"
                        and bool(getattr(self.verified_runtime, "placement_v22_enabled", False))
                        and route_held is not None
                        and target_track is not None
                        and callable(track_visual_points)
                    ):
                        track_result = track_visual_points(
                            image=raw_agentview,
                            mask=route_held.get("mask"),
                            camera="agentview",
                            frame_id=step_idx,
                            instance_id=getattr(target_track, "instance_id", None),
                            grasp_epoch=int(getattr(runtime_belief, "grasp_epoch", 0) or 0),
                        )
                        if isinstance(track_result, dict):
                            track_summary = {
                                key: track_result.get(key)
                                for key in (
                                    "health", "camera", "query_frame_id", "frame_id",
                                    "visible_count", "inference_latency_s",
                                )
                                if key in track_result
                            }
                            motion_summary = track_result.get("motion_summary")
                            if isinstance(motion_summary, dict):
                                track_summary.update(motion_summary)
                            capability_evidence["held_point_track_summary"] = track_summary
                    route_output = self.visual_route_plugin.update(
                        agentview=agentview,
                        wrist=wrist,
                        stage=str(subgoal.motion),
                        subgoals=subgoals,
                        current_index=current_index,
                        frame_id=step_idx,
                        eef_world=self._fingertip_position(self._tcp(obs)),
                        geometry=geometry_context,
                        held_evidence=route_held,
                        destination_evidence=route_destination,
                        gripper_closed=(
                            str(getattr(self.controller.state, "gripper_name", "")).upper()
                            == "CLOSE"
                        ),
                        previous_action=(
                            self.visual_harness.last_action
                            if self.visual_harness is not None
                            else None
                        ),
                        debug=self.debug,
                    )
                    if isinstance(route_output, dict):
                        visual_route_output = route_output
                        route_context = str(route_output.get("context") or "")
                        if route_context:
                            capability_context += route_context
                        route_evidence = route_output.get("evidence")
                        if isinstance(route_evidence, dict):
                            # Placement geometry may be usable while the held
                            # object spatial providers disagree, but that
                            # disagreement must still be visible to VCR. It
                            # converts the shared belief to UNKNOWN and blocks
                            # motion rather than letting a stale route residual
                            # authorize a correction.
                            placement_belief = route_evidence.get("placement_belief")
                            spatial_belief = capability_evidence.get("spatial_belief")
                            if isinstance(placement_belief, dict) and isinstance(spatial_belief, dict):
                                conflicts = list(placement_belief.get("conflicts") or [])
                                provider_conflicts = spatial_belief.get("conflicting_sources")
                                health = str(spatial_belief.get("health") or "").upper()
                                if isinstance(provider_conflicts, (list, tuple)):
                                    conflicts.extend(str(value) for value in provider_conflicts)
                                if health in {"AMBIGUOUS", "STALE", "SENSOR_FAULT"}:
                                    conflicts.append(f"spatial_health:{health}")
                                if conflicts:
                                    placement_belief["conflicts"] = sorted(set(conflicts))
                            capability_evidence["visual_route"] = route_evidence
                            holding_arbiter = route_evidence.get("holding_arbiter")
                            if isinstance(holding_arbiter, dict):
                                route_holding_arbiter = dict(holding_arbiter)
                            transport_hold_lost = bool(
                                str(getattr(subgoal, "motion", "")).upper()
                                == "TRANSPORT"
                                and isinstance(holding_arbiter, dict)
                                and holding_arbiter.get("state") == "LOST"
                            )
                        # Only the annotated copy is passed to Qwen and the logger;
                        # SAM3/tracker above already consumed the raw frame.
                        annotated_agentview = route_output.get("agentview")
                        if annotated_agentview is not None:
                            agentview = annotated_agentview

                        # AnyPlace receives only parent/child point clouds in a
                        # separate GPU1 subprocess.  Its candidates are logged
                        # as shadow evidence and never enter the action path.
                        shadow_infer = getattr(self.verified_runtime, "infer_placement_shadow", None)
                        placement_shadow = route_evidence.get("placement_belief") if isinstance(route_evidence, dict) else None
                        if (
                            callable(shadow_infer)
                            and bool(getattr(self.verified_runtime, "placement_v22_enabled", False))
                            and isinstance(placement_shadow, dict)
                            and step_idx - self._last_anyplace_shadow_frame >= 8
                        ):
                            parent_polygon = placement_shadow.get("opening_free_space_polygon_world")
                            child_points = getattr(self.verified_runtime, "placement_object_points_gripper", ())
                            try:
                                eef_now = np.asarray(self._fingertip_position(self._tcp(obs)), dtype=float).reshape(3)
                                child_world = [
                                    (eef_now + np.asarray(point, dtype=float).reshape(3)).tolist()
                                    for point in child_points
                                    if isinstance(point, (list, tuple)) and len(point) >= 3
                                ]
                                parent_world = [
                                    [float(value[0]), float(value[1]), float(placement_shadow.get("rim_plane_z_m"))]
                                    for value in (parent_polygon or [])
                                    if isinstance(value, (list, tuple)) and len(value) >= 2
                                ]
                            except (TypeError, ValueError, IndexError):
                                child_world, parent_world = [], []
                            if len(parent_world) >= 3 and child_world:
                                capability_evidence["anyplace_shadow"] = shadow_infer(
                                    parent_points=parent_world,
                                    child_points=child_world,
                                    frame_id=step_idx,
                                    instance_id=getattr(getattr(self.verified_runtime, "belief", None).target, "instance_id", None)
                                    if getattr(getattr(self.verified_runtime, "belief", None), "target", None) is not None
                                    else None,
                                )
                                self._last_anyplace_shadow_frame = step_idx

                if callable(runtime_observe_frame):
                    try:
                        runtime_eef = tuple(
                            float(value)
                            for value in self._fingertip_position(self._tcp(obs))[:3]
                        )
                    except (TypeError, ValueError, IndexError):
                        runtime_eef = None
                    verified_runtime_decision = runtime_observe_frame(
                        stage=str(subgoal.motion),
                        evidence=(
                            capability_evidence
                            if isinstance(capability_evidence, dict)
                            else {}
                        ),
                        previous_action=(
                            self.visual_harness.last_action
                            if self.visual_harness is not None
                            else None
                        ),
                        agentview=raw_agentview,
                        wrist=raw_wrist,
                        image_refs=image_refs,
                        eef_xyz=runtime_eef,
                        gripper_closed=(
                            str(
                                getattr(
                                    self.controller.state,
                                    "gripper_name",
                                    "",
                                )
                            ).upper()
                            == "CLOSE"
                        ),
                        gripper_width_m=current_gripper_width_m,
                    )
                    if isinstance(capability_evidence, dict):
                        capability_evidence["verified_runtime"] = dict(
                            verified_runtime_decision
                        )
                    critical_request = verified_runtime_decision.get(
                        "critical_decision"
                    )
                    apply_critical = getattr(
                        self.verified_runtime,
                        "apply_critical_decision",
                        None,
                    )
                    if isinstance(critical_request, dict) and callable(
                        apply_critical
                    ):
                        critical_response = self._resolve_runtime_critical_decision(
                            request=critical_request,
                            subgoal=subgoal,
                            obs=obs,
                            agentview=raw_agentview,
                            placement_evidence=(
                                (capability_evidence.get("visual_route") or {}).get("placement_belief")
                                if isinstance(capability_evidence, dict)
                                and isinstance(capability_evidence.get("visual_route"), dict)
                                else None
                            ),
                        )
                        critical_commit = apply_critical(
                            str(critical_response.get("answer") or "UNKNOWN"),
                            details=critical_response.get("details") if isinstance(critical_response.get("details"), dict) else {},
                        )
                        if critical_commit.get("request_geometry_refresh") and self.visual_route_plugin is not None:
                            request_refresh = getattr(self.visual_route_plugin, "request_geometry_refresh", None)
                            if callable(request_refresh):
                                request_refresh(str(critical_commit.get("semantic_option") or "agent_requested_refresh"))
                        verified_runtime_decision["critical_response"] = (
                            critical_response
                        )
                        verified_runtime_decision["critical_commit"] = critical_commit
                        next_action = critical_commit.get("next_action")
                        if next_action:
                            verified_runtime_decision["action_token"] = str(
                                next_action
                            ).upper()
                        rollback_stage = str(
                            critical_commit.get("rollback_stage") or ""
                        ).upper()
                        if rollback_stage:
                            for candidate_index in range(current_index, -1, -1):
                                if str(
                                    getattr(
                                        subgoals[candidate_index],
                                        "motion",
                                        "",
                                    )
                                ).upper() == rollback_stage:
                                    runtime_rollback_index = candidate_index
                                    break
                        runtime_event = verified_runtime_decision.get("event")
                        if isinstance(runtime_event, dict):
                            runtime_event["critical_response"] = critical_response
                            runtime_event["critical_commit"] = critical_commit
                            runtime_event["qwen_decision"] = critical_response
                    if isinstance(capability_evidence, dict):
                        capability_evidence["verified_runtime"] = dict(
                            verified_runtime_decision
                        )
                    # Wrist and AgentView answer different spatial questions,
                    # but a Wrist candidate must not drive pixel servoing when
                    # the fixed view cannot corroborate the same target.  On
                    # the next frame prefer the fixed camera so temporal
                    # identity and world-XY evidence can recover; this is
                    # especially important after a dropped object is beside a
                    # receptacle, where the Wrist often sees a distractor or
                    # the gripper itself.  The semantic Qwen decision already
                    # made on this frame is retained in the event log.
                    if (
                        self.visual_harness is not None
                        and str(getattr(subgoal, "motion", "")).upper() == "GRASP"
                        and str(capability_evidence.get("camera", "")).lower() == "wrist"
                    ):
                        secondary_view = capability_evidence.get("secondary_view")
                        secondary_unreliable = bool(
                            isinstance(secondary_view, dict)
                            and (
                                not bool(secondary_view.get("visible", False))
                                or str(secondary_view.get("source", "")).lower()
                                in {"sam3_abstain", "unknown", ""}
                            )
                        )
                        runtime_health = str(
                            verified_runtime_decision.get("observation_health", "")
                        ).upper()
                        if secondary_unreliable or runtime_health in {
                            "OCCLUDED",
                            "AMBIGUOUS",
                        }:
                            self.visual_harness.stage_camera_override = "agentview"

                # LIFT has a visual ambiguity that is specific to an eye-in-hand
                # perspective: a carried object remains below the hand in the image
                # even after it is far above the table. Expose the robot-only
                # clearance measurement as evidence, while leaving the semantic
                # held/seated judgment and DONE decision to Qwen.
                route_active = bool(
                    self.visual_route_plugin is not None
                    and getattr(self.visual_route_plugin, "enabled", False)
                    and str(getattr(self.visual_route_plugin, "mode", "shadow")).lower()
                    == "active"
                )
                if str(getattr(subgoal, "motion", "")).upper() == "LIFT" and not route_active:
                    try:
                        lift_height = float(self._fingertip_position(self._tcp(obs))[2])
                    except (TypeError, ValueError, IndexError):
                        lift_height = None
                    if lift_height is not None and lift_height >= self.lift_clear_height_m:
                        capability_context += (
                            "\nLIFT CLEARANCE EVIDENCE: the robot-only EEF/grasp-point "
                            f"height is {lift_height:.3f}m, already above the configured "
                            f"clearance reference {self.lift_clear_height_m:.3f}m. This "
                            "does not prove that the object is held: inspect both live "
                            "views and the closed fingers. If the object is visibly moving "
                            "with the fingers and no longer contacting the table, choose "
                            "DONE now; do not continue MV_UP merely because the carried "
                            "object appears below the hand in the image."
                        )

                # A closed gripper whose measured aperture is approaching the
                # robot's mechanical close value may be under contact load (for
                # example against a receptacle rim).  This is not a semantic
                # held/object classifier and it is not tied to an object size;
                # it is only transit-safety evidence for the Agent.
                if (
                    str(getattr(subgoal, "motion", "")).upper() == "MOVE"
                    and current_gripper_width_m is not None
                    and str(getattr(self.controller.state, "gripper_name", "")).upper()
                    == "CLOSE"
                    and current_gripper_width_m
                    <= 1.5 * max(self.empty_width_m, self.gripper_close_threshold_m)
                ):
                    capability_context += (
                        "\nTRANSIT HOLDING-MARGIN EVIDENCE: the gripper is still "
                        f"commanded CLOSE, but its measured aperture is "
                        f"{current_gripper_width_m:.4f}m, near the robot's mechanical "
                        "close condition. This does not prove the object is absent; it "
                        "can indicate contact or loss of holding margin during transit. "
                        "Inspect both views and the rim/lowest-point relation; if the "
                        "route is contacting or uncertain, choose MV_UP before more "
                        "horizontal motion. The Agent still decides the recovery."
                    )

                v22_enabled = bool(
                    getattr(self.verified_runtime, "placement_v22_enabled", False)
                )
                recovery_decision = (
                    None
                    if v22_enabled
                    else self._recovery_before(
                        current_index=current_index,
                        subgoals=subgoals,
                    )
                )
                if self.recovery_plugin is not None and not v22_enabled:
                    recovery_decision, arbitration = (
                        self.recovery_plugin.arbitrate_transport_holding(
                            recovery_decision,
                            stage=str(getattr(subgoal, "motion", "")),
                            visual_holding_state=str(
                                route_holding_arbiter.get("state", "")
                            ),
                        )
                    )
                    if arbitration is not None:
                        arbitration["timing"] = "before_decision"
                        holding_recovery_arbiter.append(arbitration)

                stage_completion_guard = None
                if self.visual_harness is not None:
                    stage_completion_guard = (
                        self.visual_harness.authorize_stage_completion(
                            str(subgoal.motion)
                        )
                    )
                grasp_agentview_guard = None
                if self.visual_harness is not None:
                    grasp_agentview_guard = self.visual_harness.authorize_grasp(
                        str(subgoal.motion)
                    )

                response = None
                target_in_wrist = None
                open_loop = False
                runtime_v2_blocks_commit = bool(
                    verified_runtime_decision.get("runtime_version")
                    and (
                        verified_runtime_decision.get("observation_health") != "VALID"
                        or verified_runtime_decision.get("critical_decision")
                        or verified_runtime_decision.get("action_token") == STOP_TOKEN
                    )
                )

                # A fresh, host-owned APPROACH completion is a stage transition, not a
                # movement suggestion.  It must win over recovery/chunk heuristics; otherwise
                # a repeated-move recovery token can consume the exact frame on which the
                # target becomes aligned and leave the controller trapped in APPROACH.
                if (
                    stage_completion_guard is not None
                    and stage_completion_guard.get("applied", False)
                    and not verified_runtime_decision.get("runtime_version")
                    and not (
                        str(getattr(subgoal, "motion", "")).upper() == "APPROACH"
                        and self._reacquire_required
                    )
                    and not runtime_v2_blocks_commit
                ):
                    token = DONE_TOKEN
                    chunk_queue = []

                elif (
                    grasp_agentview_guard is not None
                    and grasp_agentview_guard.get("applied", False)
                    and not verified_runtime_decision.get("runtime_version")
                    and not runtime_v2_blocks_commit
                ):
                    token = GRASP_TOKEN
                    chunk_queue = []

                elif (
                    isinstance(verified_runtime_decision, dict)
                    and verified_runtime_decision.get("takeover", False)
                    and verified_runtime_decision.get("action_token")
                ):
                    # The Agent already chose the APPROACH semantic stage.
                    # The runtime owns only this bounded local option.
                    token = str(
                        verified_runtime_decision["action_token"]
                    ).strip().upper()
                    # Reuse the robot adapter's calibrated coarse/fine step
                    # contract, but derive "far" from the runtime option rather
                    # than asking Qwen for a wrist marker.  MOVE_TO_HOVER is the
                    # only coarse horizontal option; precision and contact
                    # options remain fine-grained.
                    runtime_option = str(
                        verified_runtime_decision.get("option") or ""
                    ).upper()
                    if runtime_option == "MOVE_TO_HOVER":
                        target_in_wrist = False
                    elif runtime_option in {
                        "ALIGN_PREGRASP",
                        "DESCEND_TO_GRASP",
                        "ALIGN_OPENING",
                        "DESCEND_TO_SEAT",
                    }:
                        target_in_wrist = True
                    chunk_queue = []

                elif (
                    recovery_decision is not None
                    and recovery_decision.token
                ):
                    token = (
                        recovery_decision.token
                    )
                    chunk_queue = []

                elif chunk_queue:
                    token = chunk_queue.pop(0)
                    target_in_wrist = False
                    open_loop = True

                else:
                    # A repeated visual move is evidence that the last observation did
                    # not visibly converge.  This is only a reflection prompt: the
                    # Agent still decides whether the same-side error remains, whether
                    # another axis is now better, or whether the object/receptacle needs
                    # a fresh visual re-interpretation.
                    if (
                        len(recent_moves) >= 3
                        and len(set(recent_moves[:3])) == 1
                    ):
                        recovery_note = (
                            "Visual convergence check: the last three actions were "
                            f"{recent_moves[0]}. Reassess the latest AgentView and compare "
                            "the current object/receptacle relation with the previous view. "
                            "Repeat the same direction only if the same-side error is still "
                            "visible; change axis only when the image shows overshoot, contact, "
                            "or a clearer correction. Do not descend or declare DONE merely "
                            "because the target is near an image edge."
                        )
                    ctx = SkillContext(
                        task=self.task_description,
                        subgoal=subgoal,
                        subgoal_index=current_index,
                        step_idx=step_idx,
                        subgoal_step_idx=(
                            subgoal_step
                        ),
                        obs=obs,
                        agentview=agentview,
                        wrist=wrist,
                        proprio=self._proprio(),
                        debug=self.debug,
                        capability_context=capability_context,
                    )

                    response = (
                        self.controls.controller.decide(
                            ctx=ctx,
                            recent_moves=(
                                self._recent_moves_text(
                                    recent_moves
                                )
                            ),
                            previous_direction=(
                                previous_direction
                                or NO_DIRECTION
                            ),
                            gripper_state=(
                                self._observed_gripper_state()
                            ),
                            recovery_context=(
                                self._recovery_prompt(
                                    recovery_note
                                )
                            ),
                            capability_context=capability_context,
                        )
                    )

                    token = response.token

                    payload = (
                        getattr(
                            response,
                            "payload",
                            None,
                        )
                        or {}
                    )

                    if payload.get("fallback"):
                        raise RuntimeError(
                            "Controller exhausted model retries; refusing a scripted "
                            f"fallback action. Payload: {payload}"
                        )

                    # Preserve the model's actual reply and any parse-retry provenance.
                    self.logger.log_debug_payload(
                        step_idx,
                        {"token": token, "raw_text": response.raw_text, "payload": payload},
                    )

                    target_in_wrist = (
                        payload.get(
                            "target_in_wrist"
                        )
                    )

                    chunk_queue = []

                    if (
                        self.action_chunk_plugin
                        is not None
                        and target_in_wrist
                        is False
                    ):
                        plan = (
                            payload.get(
                                "chunk_plan"
                            )
                            or []
                        )

                        if plan:
                            token = plan[0]
                            chunk_queue = list(
                                plan[1:]
                            )

                    if (
                        self.prompt_log_every > 0
                        and step_idx
                        % self.prompt_log_every
                        == 0
                    ):
                        prompt_text = (
                            self.controls
                            .controller
                            .last_prompt
                        )

                        if prompt_text:
                            self.logger.save_controller_prompt(
                                step_idx,
                                prompt_text,
                                media=getattr(
                                    self.controls
                                    .controller
                                    .agent,
                                    "last_media",
                                    None,
                                ),
                            )

                if response is not None:
                    reason = self._reasoning(
                        response
                    )
                elif (
                    isinstance(verified_runtime_decision, dict)
                    and verified_runtime_decision.get("takeover", False)
                    and verified_runtime_decision.get("action_token")
                ):
                    reason = (
                        "verified_runtime "
                        + str(verified_runtime_decision.get("option", "OPTION"))
                        + ": "
                        + str(verified_runtime_decision.get("reason", ""))
                    )
                elif recovery_decision is not None:
                    reason = (
                        "recovery: "
                        + str(
                            recovery_decision.reason
                        )
                    )
                elif open_loop:
                    reason = (
                        "action chunk -- "
                        "planned move open-loop"
                    )
                else:
                    reason = ""

                place_alignment = None
                alignment_forced = False
                alignment_forced_action = None
                stage_name = str(getattr(subgoal, "motion", "")).upper()
                if (
                    not bool(getattr(self.verified_runtime, "placement_v22_enabled", False))
                    and stage_name in {"PLACE", "TRANSPORT"}
                ):
                    place_sid = str(getattr(subgoal, "sid", ""))
                    pending_recovery = self._place_verification_recovery_action
                    if pending_recovery in {
                        "MV_UP",
                        "MV_LEFT",
                        "MV_RIGHT",
                        "MV_FWD",
                        "MV_BACK",
                    }:
                        # This is the visual verifier's chosen immediate recovery
                        # action. It is consumed once; the next observation gets a
                        # fresh alignment review rather than replaying a direction.
                        token = pending_recovery
                        self._place_verification_recovery_action = None
                        alignment_forced = True
                        alignment_forced_action = pending_recovery
                    elif self._place_alignment_stage != place_sid:
                        place_alignment = self._review_place_alignment(
                            subgoal=subgoal,
                            obs=obs,
                            recent_moves=", ".join(recent_moves),
                        )
                        self._place_alignment_stage = place_sid
                        self._place_alignment_decision = str(
                            (place_alignment or {}).get("decision", "UNKNOWN")
                        ).upper()
                        self._place_alignment_action = str(
                            (place_alignment or {}).get("recommended_action", "HOLD")
                        ).upper()
                        if (
                            self._place_alignment_decision == "NO"
                            and self._place_alignment_action in {
                            "MV_UP",
                            "MV_LEFT",
                            "MV_RIGHT",
                            "MV_FWD",
                            "MV_BACK",
                            "MV_DOWN",
                            }
                        ):
                            token = self._place_alignment_action
                            alignment_forced = True
                            alignment_forced_action = self._place_alignment_action
                            recovery_note = (
                                "Visual pre-placement review says the held object is "
                                "not aligned with the receptacle: "
                                f"{str((place_alignment or {}).get('reasoning', '')).strip()} "
                                f"Execute {self._place_alignment_action} once, then inspect "
                                "both views again before choosing the next placement action."
                            )
                        elif (
                            self._place_alignment_decision == "YES"
                            and self._place_alignment_action == "MV_DOWN"
                        ):
                            token = "MV_DOWN"

                requested_token = self._normalize_stage_token(
                    token,
                    subgoal=subgoal,
                    obs=obs,
                )
                token = requested_token
                if bool(
                    self.visual_route_plugin is not None
                    and getattr(self.visual_route_plugin, "enabled", False)
                    and not bool(getattr(self.verified_runtime, "placement_v22_enabled", False))
                ):
                    visual_route_gate = self.visual_route_plugin.gate(
                        requested_token,
                        stage=str(subgoal.motion),
                        held_evidence=route_held_evidence,
                        destination_evidence=route_destination_evidence,
                        eef_world=self._fingertip_position(self._tcp(obs)),
                    )
                    if visual_route_gate.executed_token != requested_token:
                        token = visual_route_gate.executed_token
                        recovery_note = (
                            "Visual route gate replaced the requested action: "
                            + str(visual_route_gate.reason)
                        )
                stage_action_guard = None
                stage_name_for_guard = str(getattr(subgoal, "motion", "")).upper()
                gripper_closed_for_guard = (
                    str(getattr(self.controller.state, "gripper_name", "")).upper()
                    == "CLOSE"
                )
                destination_evidence_for_guard = (
                    capability_evidence.get("destination_proximity")
                    if isinstance(capability_evidence, dict)
                    else None
                )
                held_alignment_for_guard = (
                    capability_evidence.get("held_object_alignment")
                    if isinstance(capability_evidence, dict)
                    else None
                )
                held_alignment_known = bool(
                    isinstance(held_alignment_for_guard, dict)
                    and held_alignment_for_guard.get("known")
                )
                held_object_aligned = bool(
                    held_alignment_known
                    and held_alignment_for_guard.get("aligned")
                )
                held_needed_horizontal = (
                    str(
                        (held_alignment_for_guard or {})
                        .get("correction_candidates", {})
                        .get("horizontal", "")
                    )
                    .strip()
                    .upper()
                )
                # If the carried object's own center is still clearly offset and
                # Qwen selected the image-grounded correction toward the opening,
                # an EEF-only overlap must not suppress that correction.  Once the
                # object itself is aligned, the geometric rim guard can intervene.
                object_correction_toward_opening = bool(
                    stage_name_for_guard == "MOVE"
                    and held_alignment_known
                    and not held_object_aligned
                    and held_needed_horizontal
                    and str(token).upper() == held_needed_horizontal
                )
                if token == GRASP_TOKEN and (
                    stage_name_for_guard != "GRASP" or gripper_closed_for_guard
                ):
                    stage_action_guard = {
                        "blocked": True,
                        "requested": GRASP_TOKEN,
                        "reason": (
                            "GRASP is only valid in an open GRASP stage; the Agent must "
                            "choose the current-stage action from fresh views."
                        ),
                    }
                    token = STOP_TOKEN
                    recovery_note = (
                        "The requested GRASP was not valid for the current stage/gripper "
                        "state. Reassess the live views and choose the action for the current "
                        "stage; do not repeat a close while the gripper is already closed."
                    )
                elif (
                    token == GRASP_TOKEN
                    and self._grasp_retry_anchor is not None
                    and not verified_runtime_decision.get("runtime_version")
                ):
                    current_relations = self._grasp_view_relations(capability_evidence)
                    comparable = False
                    changed = False
                    change_tolerance = max(
                        3.0,
                        float(
                            getattr(
                                self.visual_harness,
                                "alignment_ready_px",
                                13.5,
                            )
                        )
                        * 0.4,
                    )
                    for view_name in ("agentview", "wrist"):
                        current = current_relations.get(view_name)
                        previous = self._grasp_retry_anchor.get(view_name)
                        if current is None or previous is None:
                            continue
                        comparable = True
                        if max(
                            abs(float(current[0]) - float(previous[0])),
                            abs(float(current[1]) - float(previous[1])),
                        ) > change_tolerance:
                            changed = True
                            break
                    if comparable and not changed:
                        stage_action_guard = {
                            "blocked": True,
                            "requested": GRASP_TOKEN,
                            "reason": (
                                "the previous GRASP was a mechanical empty close and the "
                                "current comparable view has not changed enough to support "
                                "a new grasp hypothesis"
                            ),
                            "change_tolerance_px": round(change_tolerance, 2),
                        }
                        token = STOP_TOKEN
                        recovery_note = (
                            "The previous close was mechanically empty even though the image "
                            "looked aligned. The current comparable view still has nearly the "
                            "same object/EEF relation, so do not repeat GRASP. Reinspect both "
                            "views and choose a spatial correction that visibly changes the "
                            "relation; if Wrist is occluded, use AgentView plus a new depth "
                            "view before closing. The harness does not choose the direction."
                        )
                elif (
                    not route_active
                    and
                    stage_name_for_guard == "MOVE"
                    and token in {"MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT"}
                    and isinstance(held_alignment_for_guard, dict)
                    and held_alignment_for_guard.get("rim_contact_risk")
                    and not held_alignment_for_guard.get("clearance_progress_ready")
                ):
                    stage_action_guard = {
                        "blocked": True,
                        "requested": token,
                        "replacement": "MV_UP",
                        "reason": (
                            "the carried-object bbox overlaps the destination while its "
                            "center remains offset; inspect possible rim contact before "
                            "another horizontal transit"
                        ),
                        "held_object_alignment": held_alignment_for_guard,
                    }
                    token = "MV_UP"
                    recovery_note = (
                        "The harness performed a visual-risk clearance move because the "
                        "carried object overlaps the destination boundary while its center "
                        "is still offset. Reinspect the fresh AgentView and Wrist; horizontal "
                        "transit remains unsafe only while this visual risk persists, and "
                        "the next non-blocked action is yours. This is not a fixed recovery "
                        "trajectory and does not authorize release."
                    )
                elif (
                    stage_name_for_guard == "MOVE"
                    and token in {"MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT"}
                    and gripper_closed_for_guard
                    and current_gripper_width_m is not None
                    and current_gripper_width_m
                    <= 1.5 * max(self.empty_width_m, self.gripper_close_threshold_m)
                ):
                    stage_action_guard = {
                        "blocked": True,
                        "requested": token,
                        "replacement": "MV_UP",
                        "reason": (
                            "closed-gripper aperture is near the mechanical close condition "
                            "during MOVE; clear possible contact before horizontal transit"
                        ),
                        "gripper_width_m": round(current_gripper_width_m, 5),
                        "mechanical_close_reference_m": round(
                            max(self.empty_width_m, self.gripper_close_threshold_m), 5
                        ),
                    }
                    token = "MV_UP"
                    recovery_note = (
                        "The harness performed one clearance move because the closed "
                        "gripper's aperture is near its mechanical close condition during "
                        "transit. This is not a conclusion that the object is absent. "
                        "Reinspect the fresh AgentView and Wrist; then choose the next "
                        "horizontal correction or other recovery from the image. Do not "
                        "release in MOVE."
                    )
                elif (
                    not route_active
                    and
                    stage_name_for_guard == "MOVE"
                    and token in {"MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT"}
                    and isinstance(capability_evidence, dict)
                    and isinstance(
                        destination_evidence_for_guard, dict
                    )
                    and destination_evidence_for_guard.get("eligible")
                    and not object_correction_toward_opening
                    and (
                        destination_evidence_for_guard.get("inside_destination_bbox")
                        or (
                            str(token).upper()
                            == str(
                                destination_evidence_for_guard.get("same_action", "")
                            ).upper()
                            and int(
                                destination_evidence_for_guard.get("same_action_count", 0)
                                or 0
                            )
                            >= 3
                        )
                    )
                ):
                    stage_action_guard = {
                        "blocked": True,
                        "requested": token,
                        "replacement": "MV_UP",
                        "reason": (
                            "horizontal transit entered or repeatedly approached the "
                            "visible destination bbox; clear the route before the next "
                            "visual review"
                        ),
                        "destination_proximity": destination_evidence_for_guard,
                        "held_object_alignment": held_alignment_for_guard,
                    }
                    # This is a one-step geometric clearance primitive, not a
                    # placement/recovery policy: it removes the immediate rim
                    # collision risk, then the Agent owns the next decision.
                    token = "MV_UP"
                    recovery_note = (
                        "The harness performed one geometric clearance move because the "
                        "repeated horizontal transit entered the visible destination "
                        "boundary. Reinspect the fresh AgentView for the held object's "
                        "lowest point, the opening, and any rim contact. The next action "
                        "is yours: continue the visually justified correction, lift again, "
                        "or reassess placement. Do not release in MOVE."
                    )
                elif (
                    token == RELEASE_TOKEN
                    and stage_name_for_guard != "RELEASE"
                    and not bool(getattr(self.verified_runtime, "placement_v22_enabled", False))
                    and not (
                        stage_name_for_guard == "TRANSPORT"
                        and transport_hold_lost
                    )
                    and not (
                        recovery_decision is not None
                        and str(getattr(recovery_decision, "token", "")).upper()
                        == RELEASE_TOKEN
                    )
                ):
                    stage_action_guard = {
                        "blocked": True,
                        "requested": RELEASE_TOKEN,
                        "reason": "RELEASE is only valid after the dedicated RELEASE stage.",
                    }
                    token = STOP_TOKEN
                    recovery_note = (
                        "The requested RELEASE was early. Reinspect the held object and the "
                        "destination; do not release until the RELEASE stage and visual "
                        "placement evidence both support it."
                    )
                elif stage_name_for_guard == "LIFT" and token == "MV_DOWN":
                    stage_action_guard = {
                        "blocked": True,
                        "requested": token,
                        "reason": (
                            "a held-object LIFT stage cannot use descent; the Agent must "
                            "reassess and choose MV_UP or DONE"
                        ),
                    }
                    token = STOP_TOKEN
                    recovery_note = (
                        "LIFT descent was blocked because the current stage is to clear the "
                        "held object from the table. Reassess the two views and choose MV_UP "
                        "while the object is held, or reassess whether the object was lost."
                    )
                elif (
                    stage_name_for_guard == "APPROACH"
                    and token == DONE_TOKEN
                    and self._reacquire_required
                    and not verified_runtime_decision.get("runtime_version")
                ):
                    stage_action_guard = {
                        "blocked": True,
                        "requested": DONE_TOKEN,
                        "reason": (
                            "a previous grasp was rejected or lost; fresh visual reacquisition "
                            "is required before APPROACH DONE"
                        ),
                    }
                    token = STOP_TOKEN
                    recovery_note = (
                        "The previous grasp was rejected or lost, so the old alignment cannot "
                        "be reused. Reinspect the current AgentView and Wrist; choose one "
                        "visually justified corrective motion that changes the object/EEF "
                        "relation, then reassess before DONE or GRASP. The harness does not "
                        "choose the direction."
                    )
                elif (
                    stage_name_for_guard == "APPROACH"
                    and token == "MV_DOWN"
                    and self._reacquire_required
                    # VCR-v2 has already validated the current identity and
                    # robot-only height band.  Its recovery option needs to
                    # descend into HOVER; the legacy screen-parallax guard
                    # would rewrite that descent into endless FWD/BACK moves.
                    and not verified_runtime_decision.get("runtime_version")
                    and self.visual_harness is not None
                    and not bool(
                        getattr(self.visual_harness, "last_evidence", {})
                        .get("geometry", {})
                        .get("alignment_ready", False)
                    )
                ):
                    # A fallen/moved object can look merely "lower" because of
                    # parallax. Reacquire its current spatial relation before
                    # descending. Use one fresh calibrated correction only when
                    # the Agent keeps repeating the unsafe descent: this prevents
                    # a STOP loop while keeping the action grounded in the current
                    # target/EEF relation.
                    geometry = getattr(self.visual_harness, "last_evidence", {}).get(
                        "geometry", {}
                    )
                    alignment = geometry.get("target_minus_eef_px")
                    candidates = geometry.get("calibrated_correction_candidates", {})
                    replacement = None
                    if isinstance(alignment, (list, tuple)) and len(alignment) == 2:
                        try:
                            axis = 0 if abs(float(alignment[0])) >= abs(float(alignment[1])) else 1
                            key = "horizontal" if axis == 0 else "vertical"
                            replacement = str(candidates.get(key, "")).strip().upper()
                        except (TypeError, ValueError):
                            replacement = None
                    if replacement not in {"MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT"}:
                        replacement = "STOP"
                    stage_action_guard = {
                        "blocked": True,
                        "requested": token,
                        "replacement": replacement,
                        "reason": "reacquisition requires fresh spatial alignment before descent",
                    }
                    token = replacement
                    recovery_note = (
                        "Recovery descent was blocked: the CURRENT target is not yet "
                        "spatially aligned with the open fingers. Reinspect fresh "
                        "AgentView/Wrist. The harness selected one current-frame "
                        f"correction ({replacement}) from the largest calibrated error; "
                        "reassess immediately afterward and do not use MV_DOWN just "
                        "because the object appears lower in the image."
                    )
                guard_decision = None
                if self.visual_harness is not None:
                    if (
                        stage_name_for_guard == "TRANSPORT"
                        and token == DONE_TOKEN
                    ):
                        # In TRANSPORT, DONE means "refresh my short-horizon
                        # intent". It is not a semantic stage completion and
                        # therefore must not pass through the commit guard.
                        guard_decision = {
                            "token": token,
                            "authorized": True,
                            "blocked": False,
                            "mode": "transport_intent_refresh",
                            "reason": "DONE requests Qwen intent refresh",
                        }
                    else:
                        guard_decision = self.visual_harness.authorize(token)
                    if guard_decision.get("blocked", False):
                        token = STOP_TOKEN

                route_trace = (
                    visual_route_output.get("evidence", {})
                    if isinstance(visual_route_output, dict)
                    else {}
                )
                route_trace = route_trace if isinstance(route_trace, dict) else {}
                intent_trace = route_trace.get("intent") or {}
                route_plan_trace = route_trace.get("route") or {}
                progress_trace = route_trace.get("progress") or {}
                runtime_event = verified_runtime_decision.get("event")
                if isinstance(runtime_event, dict):
                    runtime_event["requested_action"] = requested_token
                    runtime_event["executed_action"] = token
                print(
                    "[zeroshot-robolab] "
                    f"step={step_idx:02d} "
                    f"sg={current_index}/"
                    f"{len(subgoals)-1} "
                    f"stage={subgoal.motion} "
                    f"mode={'transport' if stage_name_for_guard == 'TRANSPORT' else 'staged'} "
                    f"intent={intent_trace.get('intent', '-')} "
                    f"route={route_plan_trace.get('route_id', '-')} "
                    f"active_leg={route_plan_trace.get('active_leg', '-')} "
                    f"requested={requested_token} "
                    f"executed={token} "
                    f"progress={progress_trace.get('along_route_progress', '-')}"
                    + (
                        f" | {reason}"
                        if reason
                        else ""
                    ),
                    flush=True,
                )

                self._current_stage = stage_name_for_guard
                if token == GRASP_TOKEN:
                    self._pending_grasp_pre_action = {
                        "frame_id": int(step_idx),
                        "agentview": np.asarray(raw_agentview).copy(),
                        "wrist": None if raw_wrist is None else np.asarray(raw_wrist).copy(),
                    }
                (
                    obs,
                    terminated,
                    truncated,
                    result,
                ) = self._execute_agentic(
                    token,
                    obs,
                    target_in_wrist=(
                        target_in_wrist
                    ),
                    step_override_m=(
                        float(
                            verified_runtime_decision.get(
                                "grasp_diagnostic_lift_step_m"
                                if bool(verified_runtime_decision.get("grasp_diagnostic_lift"))
                                else "reobserve_lift_step_m"
                            )
                        )
                        if token == "MV_UP"
                        and (
                            bool(verified_runtime_decision.get("grasp_diagnostic_lift"))
                            or bool(verified_runtime_decision.get("reobserve_for_view"))
                        )
                        else None
                    ),
                )

                # The controller/route guards may rewrite the requested token.
                # VCR-v2 learns only this adapter receipt, never the stale request.
                runtime_commit = None
                runtime_commit_fn = getattr(self.verified_runtime, "commit_executed_action", None)
                if callable(runtime_commit_fn):
                    runtime_commit = runtime_commit_fn(
                        executed_action=str(getattr(result, "token", token)).upper(),
                        authorized_action=str(token).upper(),
                    )
                    if isinstance(verified_runtime_decision.get("event"), dict):
                        verified_runtime_decision["event"]["action_receipt"] = runtime_commit
                    verified_runtime_decision["action_receipt"] = runtime_commit
                self._previous_agentview = raw_agentview
                self._previous_wrist = raw_wrist
                self._previous_frame_id = int(step_idx)
                self._previous_executed_action = str(getattr(result, "token", token)).upper()

                # PRE_DESCENT is a semantic hand-off, not an open-loop
                # vertical trajectory.  After one bounded downward atom, force
                # a fresh Qwen route review so contact, seating, or a new rim
                # relation can change the next intent.
                if (
                    self.visual_route_plugin is not None
                    and stage_name_for_guard == "TRANSPORT"
                    and str(
                        getattr(self.visual_route_plugin, "last_progress", None)
                        and getattr(
                            self.visual_route_plugin.last_progress,
                            "active_leg",
                            "",
                        )
                        or ""
                    ).upper()
                    in {"PRE_DESCENT", "DESCENT"}
                    and str(getattr(result, "token", token)).upper() == "MV_DOWN"
                ):
                    self.visual_route_plugin.request_intent_refresh(
                        "bounded_pre_descent_step",
                        critical=True,
                    )

                if self.visual_harness is not None:
                    self.visual_harness.mark_action(token)

                if (
                    stage_name_for_guard == "TRANSPORT"
                    and transport_hold_lost
                    and token == RELEASE_TOKEN
                ):
                    # A temporal visual loss confirmed by Qwen returns to the
                    # existing semantic reacquisition chain.  No grasp pose,
                    # direction, or object-size rule is introduced here.
                    for rollback in range(current_index - 1, -1, -1):
                        if str(getattr(subgoals[rollback], "motion", "")).upper() == "APPROACH":
                            transport_rollback_index = rollback
                            break

                success = self._success(obs)

                grasp_verification = None
                place_verification = None
                agent_grasp_decision = None
                if (
                    token == GRASP_TOKEN
                    and result.gripper_closed
                    and not bool(getattr(self.verified_runtime, "placement_v22_enabled", False))
                ):
                    grasp_verification = self._verify_grasp_visually(
                        subgoal=subgoal,
                        obs=obs,
                    )
                    if isinstance(grasp_verification, dict) and not bool(
                        getattr(self.verified_runtime, "placement_v22_enabled", False)
                    ):
                        verify_event = getattr(
                            self.recovery_plugin,
                            "agent_grasp_verification",
                            None,
                        )
                        if callable(verify_event):
                            agent_grasp_decision = verify_event(
                                verdict=grasp_verification.get("decision", "UNKNOWN"),
                                current_index=current_index,
                                subgoals=subgoals,
                                reasoning=grasp_verification.get("reasoning", ""),
                            )

                # Validate PLACE when the controller declares DONE, or when a
                # downward command physically stalls. The latter is a low-level
                # contact/settling event, not a fixed EEF-height semantic threshold:
                # different receptacles and object geometries can have different
                # valid approach heights. The Agent still decides from both live
                # views whether the object is placed and how to recover.
                downward_stalled = False
                if token == "MV_DOWN" and result.gripper_closed:
                    try:
                        downward_stalled = bool(
                            float(result.pre_pose[2]) - float(result.post_pose[2])
                            <= 1.0e-4
                        )
                    except (TypeError, ValueError, IndexError):
                        downward_stalled = False
                transport_stage = (
                    str(getattr(subgoal, "motion", "")).upper()
                    == "TRANSPORT"
                )
                transport_ready = bool(
                    transport_stage
                    and self.visual_route_plugin is not None
                    and self.visual_route_plugin.ready_to_release()
                )
                if (
                    transport_stage
                    and downward_stalled
                    and self.visual_route_plugin is not None
                ):
                    self.visual_route_plugin.request_intent_refresh(
                        "descent_stall",
                        critical=True,
                    )
                if (
                    not bool(getattr(self.verified_runtime, "placement_v22_enabled", False))
                    and
                    (
                        str(getattr(subgoal, "motion", "")).upper()
                        == "PLACE"
                        or transport_ready
                    )
                    and result.gripper_closed
                    and (
                        token == DONE_TOKEN
                        or (
                            downward_stalled
                            and self._place_verification_stage
                            != str(getattr(subgoal, "sid", ""))
                        )
                    )
                ):
                    place_verification = self._verify_place_visually(
                        subgoal=subgoal,
                        obs=obs,
                    )
                    place_sid = str(getattr(subgoal, "sid", ""))
                    self._place_verification_stage = place_sid
                    self._place_verification_decision = str(
                        (place_verification or {}).get("decision", "UNKNOWN")
                    ).upper()
                    self._place_verification_last_step = step_idx
                    if self._place_verification_decision != "YES":
                        # A failed/uncertain DONE invalidates the old alignment
                        # review. The next action must inspect the new geometry.
                        self._place_alignment_stage = None
                    if self._place_verification_decision == "NO":
                        self._place_verification_stage = None
                        if transport_stage:
                            # The placement verifier is evidence for the next Qwen
                            # intent. Its bounded recovery choice is exposed as
                            # exactly one executable atom; after that fresh
                            # observation the route intent is replanned.
                            recovery_action = str(
                                (place_verification or {}).get("recovery_action", "HOLD")
                            ).upper()
                            if recovery_action in {
                                "MV_UP", "MV_LEFT", "MV_RIGHT", "MV_FWD", "MV_BACK",
                            }:
                                self._place_verification_recovery_action = recovery_action
                            set_feedback = getattr(
                                self.visual_route_plugin,
                                "set_placement_feedback",
                                None,
                            )
                            if callable(set_feedback):
                                set_feedback(
                                    decision="NO",
                                    reasoning=str(
                                        (place_verification or {}).get("reasoning", "")
                                    ),
                                    recovery_action=recovery_action,
                                )
                            self.visual_route_plugin.request_intent_refresh(
                                "placement_verification_failed",
                                critical=True,
                            )
                            recovery_note = (
                                "Fresh visual placement verification rejected READY_TO_RELEASE: "
                                f"{str((place_verification or {}).get('reasoning', '')).strip()} "
                                "Replan the short-horizon transport intent from the new frame."
                            )
                        else:
                            # Legacy PLACE behavior remains unchanged when the
                            # visual-route transport loop is disabled.
                            recovery_action = str(
                                (place_verification or {}).get("recovery_action", "HOLD")
                            ).upper()
                            if recovery_action in {
                                "MV_UP", "MV_LEFT", "MV_RIGHT", "MV_FWD", "MV_BACK",
                            }:
                                self._place_verification_recovery_action = recovery_action
                            recovery_note = (
                                "Visual placement verifier says the object is not placed: "
                                f"{str((place_verification or {}).get('reasoning', '')).strip()} "
                                "keep holding it and follow the fresh visual recovery action "
                                f"chosen by the Agent ({recovery_action or 'HOLD'}); do not release."
                            )
                    elif (
                        transport_stage
                        and self._place_verification_decision == "UNKNOWN"
                    ):
                        set_feedback = getattr(
                            self.visual_route_plugin,
                            "set_placement_feedback",
                            None,
                        )
                        if callable(set_feedback):
                            set_feedback(
                                decision="UNKNOWN",
                                reasoning=str(
                                    (place_verification or {}).get("reasoning", "")
                                ),
                                recovery_action="HOLD",
                            )

                subgoal_done = bool(
                    getattr(
                        result,
                        "done",
                        False,
                    )
                    or token == DONE_TOKEN
                )
                if transport_stage:
                    # DONE refreshes Qwen's receding-horizon plan.  It cannot
                    # skip the whole transport goal.
                    subgoal_done = False
                    if token == DONE_TOKEN and self.visual_route_plugin is not None:
                        self.visual_route_plugin.request_intent_refresh(
                            "controller_done",
                            critical=True,
                        )

                visual_grasp_decision = (
                    str((grasp_verification or {}).get("decision", ""))
                    .strip()
                    .upper()
                    if isinstance(grasp_verification, dict)
                    else ""
                )

                # The Agent owns the semantic GRASP decision. A visual YES is
                # authoritative for object thickness and Wrist occlusion, but a
                # fully closed no-contact signal is a separate low-level safety
                # veto: it is evidence of an obvious empty close, not a holding
                # width band. This keeps the verifier general without accepting a
                # visually plausible but physically empty re-grasp.
                verified_grasp = bool(
                    token == GRASP_TOKEN
                    and result.gripper_closed
                    and (
                        visual_grasp_decision == "YES"
                        and not result.grasp_empty
                        or (
                            not isinstance(grasp_verification, dict)
                            and not result.grasp_empty
                        )
                    )
                )
                if bool(getattr(self.verified_runtime, "placement_v22_enabled", False)):
                    # The post-close answer is a candidate. V2.2 advances only
                    # after the runtime observes one bounded lift and a fresh
                    # stable object/EEF relation.
                    verified_grasp = False
                if agent_grasp_decision is not None:
                    # A NO or UNKNOWN visual verdict keeps the semantic GRASP stage
                    # open. NO will also request the normal release/rollback path;
                    # UNKNOWN lets the next controller decision inspect live views.
                    verified_grasp = False
                report_grasp = getattr(
                    self.verified_runtime,
                    "report_grasp_verdict",
                    None,
                )
                if (
                    token == GRASP_TOKEN
                    and callable(report_grasp)
                    and not bool(getattr(self.verified_runtime, "placement_v22_enabled", False))
                ):
                    runtime_grasp_verdict = report_grasp(
                        verdict=visual_grasp_decision or "UNKNOWN",
                        frame_id=step_idx,
                        mechanically_empty=bool(result.grasp_empty),
                        diagnostic_lift_clear=bool(
                            (grasp_verification or {}).get("diagnostic_lift_clear", False)
                        ),
                        evidence_for=(grasp_verification or {}).get("evidence_for", ()),
                        reasoning=(
                            str((grasp_verification or {}).get("reasoning", ""))
                            if isinstance(grasp_verification, dict)
                            else ""
                        ),
                    )
                    verified_runtime_decision["post_action_verification"] = (
                        runtime_grasp_verdict
                    )
                    verified_runtime_decision["belief"] = runtime_grasp_verdict.get(
                        "belief"
                    )
                if (
                    bool(getattr(self.verified_runtime, "placement_v22_enabled", False))
                    and token == DONE_TOKEN
                    and str(getattr(subgoal, "motion", "")).upper() == "GRASP"
                ):
                    runtime_belief = getattr(self.verified_runtime, "belief", None)
                    held = getattr(runtime_belief, "held", None)
                    held_truth = str(getattr(getattr(held, "truth", None), "value", "")).upper()
                    verified_grasp = held_truth == "TRUE"
                if verified_grasp:
                    subgoal_done = True
                elif (
                    str(getattr(subgoal, "motion", "")).upper() == "GRASP"
                    and token == DONE_TOKEN
                ):
                    # A GRASP stage cannot be completed by a bare DONE. The semantic
                    # close must have been executed and visually verified; otherwise
                    # keep the stage open for the Agent's next fresh decision.
                    subgoal_done = False
                    recovery_note = (
                        "GRASP is not complete: a close has not been visually verified. "
                        "Inspect both views and the gripper state before choosing GRASP or "
                        "a corrective approach action."
                    )

                if (
                    isinstance(place_verification, dict)
                    and str(place_verification.get("decision", "")).upper() == "YES"
                ):
                    subgoal_done = True
                elif (
                    str(getattr(subgoal, "motion", "")).upper() == "PLACE"
                    and token == DONE_TOKEN
                    and place_verification is not None
                ):
                    # A DONE token is not enough to release a visibly misaligned
                    # object. The verifier's next action is selected from the
                    # current images, not from a hard-coded lift/left sequence.
                    subgoal_done = False

                # Qwen still chooses DONE, but a fresh host relation may reject a
                # visibly premature MOVE completion. This is a completion contract,
                # not a scripted replacement action: the next Qwen call must inspect
                # the live views and choose its own correction.
                move_completion_guard = None
                if (
                    str(getattr(subgoal, "motion", "")).upper() == "MOVE"
                    and token == DONE_TOKEN
                    and isinstance(capability_evidence, dict)
                ):
                    evidence_geometry = capability_evidence.get("geometry")
                    held_alignment = capability_evidence.get("held_object_alignment")
                    held_alignment_known = bool(
                        isinstance(held_alignment, dict)
                        and held_alignment.get("known")
                    )
                    # When the carried object is grounded, its body-to-opening
                    # relation is the semantic completion signal.  EEF-to-opening
                    # geometry remains the fallback when SAM3/tracking abstains.
                    evidence_alignment = (
                        held_alignment.get("destination_minus_held_center_px")
                        if held_alignment_known
                        else (
                            evidence_geometry.get("target_minus_eef_px")
                            if isinstance(evidence_geometry, dict)
                            else None
                        )
                    )
                    alignment_known = (
                        isinstance(evidence_alignment, (list, tuple))
                        and len(evidence_alignment) == 2
                    )
                    evidence_visible = (
                        bool(held_alignment_known)
                        if held_alignment_known
                        else bool(capability_evidence.get("visible", False))
                    )
                    evidence_confidence = (
                        float(held_alignment.get("confidence", 0.0) or 0.0)
                        if held_alignment_known
                        else float(capability_evidence.get("confidence", 0.0) or 0.0)
                    )
                    not_aligned = bool(
                        alignment_known
                        and (
                            not bool(held_alignment.get("aligned"))
                            if held_alignment_known
                            else (
                                isinstance(evidence_geometry, dict)
                                and not evidence_geometry.get("alignment_ready", False)
                            )
                        )
                    )
                    if evidence_visible and evidence_confidence >= 0.4 and not_aligned:
                        subgoal_done = False
                        move_completion_guard = {
                            "blocked": True,
                            "reason": (
                                "fresh held-object geometry contradicts MOVE DONE"
                                if held_alignment_known
                                else "fresh host geometry contradicts MOVE DONE"
                            ),
                            "alignment_px": list(evidence_alignment),
                            "threshold_px": (
                                held_alignment.get("alignment_threshold_px")
                                if held_alignment_known
                                else (
                                    evidence_geometry.get("alignment_ready_threshold_px")
                                    if isinstance(evidence_geometry, dict)
                                    else None
                                )
                            ),
                        }
                        recovery_note = (
                            "The latest host visual evidence does not support ending MOVE: "
                            f"the receptacle is still {list(evidence_alignment)} pixels from "
                            "the projected EEF. Re-read both views and choose the action that "
                            "reduces the actual object/opening error; do not emit DONE yet."
                        )

                final_check = None
                replan_requested = False
                if (
                    subgoal_done
                    and not success
                    and current_index + 1 >= len(subgoals)
                ):
                    final_check = self._verify_task_visually(obs=obs)
                    complete = bool(
                        isinstance(final_check, dict)
                        and final_check.get("complete", False)
                    )
                    if not complete and self.planner is not None:
                        replan_requested = replans < int(
                            getattr(self.config, "max_replans", 0)
                        )
                    if not complete and not replan_requested:
                        subgoal_done = False

                if alignment_forced and stage_name == "PLACE":
                    # Consume exactly one visual recommendation. The next decision
                    # must inspect a new frame again: a single atomic move changes
                    # the object/receptacle relation, so retaining the old review
                    # lets a wrong direction repeat for an entire PLACE stage.
                    self._place_alignment_decision = "UNKNOWN"
                    self._place_alignment_action = None
                    self._place_alignment_stage = None

                if stage_name == "PLACE" and not terminated and not truncated:
                    # PLACE is a closed visual servo loop, not an open-loop move
                    # chunk. Re-review after every executed atom so horizontal
                    # corrections, camera parallax, and rim contact are judged from
                    # the current observation rather than a stale alignment result.
                    self._place_alignment_stage = None

                self._descend = (
                    _descend_travel(
                        token,
                        result,
                        self._descend,
                    )
                )

                # A LIBERO step can terminate the environment at its horizon.  Do
                # not ask recovery to issue a compensating RELEASE after that step:
                # robosuite rejects every action once an episode is terminated.  The
                # old ordering turned an ordinary horizon termination into a runner
                # traceback and, more importantly, obscured the last physical result.
                post_recovery = None
                if not terminated and not truncated:
                    v22_enabled = bool(
                        getattr(self.verified_runtime, "placement_v22_enabled", False)
                    )
                    if v22_enabled:
                        # V2.2 has one action authority.  The legacy recovery
                        # plugin may still log evidence, but it cannot release,
                        # roll back, or move the robot beside the VCR runtime.
                        post_recovery = None
                    elif agent_grasp_decision is not None:
                        post_recovery = agent_grasp_decision
                    elif not (
                        token == GRASP_TOKEN
                        and visual_grasp_decision == "YES"
                        and not result.grasp_empty
                    ):
                        post_recovery = self._recovery_after(
                            token=token,
                            result=result,
                            current_index=current_index,
                            subgoals=subgoals,
                            subgoal_done=subgoal_done,
                        )

                if self.recovery_plugin is not None and not bool(
                    getattr(self.verified_runtime, "placement_v22_enabled", False)
                ):
                    post_recovery, arbitration = (
                        self.recovery_plugin.arbitrate_transport_holding(
                            post_recovery,
                            stage=stage_name_for_guard,
                            visual_holding_state=str(
                                route_holding_arbiter.get("state", "")
                            ),
                        )
                    )
                    if arbitration is not None:
                        arbitration["timing"] = "after_step"
                        holding_recovery_arbiter.append(arbitration)

                if post_recovery is not None:
                    recovery_decision = (
                        post_recovery
                    )
                    rollback_index = getattr(post_recovery, "rollback_index", None)
                    if rollback_index is not None:
                        try:
                            rollback_stage = str(
                                getattr(subgoals[int(rollback_index)], "motion", "")
                            ).upper()
                        except (IndexError, TypeError, ValueError):
                            rollback_stage = ""
                        if rollback_stage == "APPROACH":
                            self._reacquire_required = True

                if (
                    not terminated
                    and not truncated
                    and
                    recovery_decision is not None
                    and recovery_decision.release
                ):
                    (
                        obs,
                        term2,
                        trunc2,
                        _release_result,
                    ) = self._execute_agentic(
                        RELEASE_TOKEN,
                        obs,
                        target_in_wrist=None,
                    )

                    terminated = (
                        terminated or term2
                    )
                    truncated = (
                        truncated or trunc2
                    )
                    success = self._success(obs)

                if (
                    recovery_decision is not None
                    and recovery_decision.block_done
                ):
                    subgoal_done = False

                # Verified close expires a previous
                # empty/lost-grasp note.
                if token == GRASP_TOKEN and result.gripper_closed and verified_grasp:
                    recovery_note = ""
                    self._grasp_retry_anchor = None

                if token == GRASP_TOKEN and result.grasp_empty:
                    self._grasp_retry_anchor = self._grasp_view_relations(
                        capability_evidence
                    )

                if (
                    recovery_decision is not None
                    and recovery_decision.prompt_note
                ):
                    recovery_note = (
                        recovery_decision.prompt_note
                    )

                if (
                    recovery_decision is not None
                    and recovery_decision.reset_history
                ):
                    previous_direction = None
                    recent_moves.clear()

                elif token in MOVE_ATOMS:
                    recent_moves.insert(
                        0,
                        token,
                    )
                    del recent_moves[
                        self.recent_moves_max :
                    ]
                    previous_direction = token
                    if stage_name == "APPROACH" and self._reacquire_required:
                        # Keep the barrier through reacquisition. A single
                        # arbitrary move must not authorize stale descent/DONE.
                        pass

                elif token == GRASP_TOKEN:
                    recent_moves.insert(
                        0,
                        GRASP_TOKEN,
                    )
                    del recent_moves[
                        self.recent_moves_max :
                    ]
                    previous_direction = None
                    self._reacquire_required = False

                elif token == RELEASE_TOKEN:
                    previous_direction = None

                step_record = self._record_stage(
                    step_idx=step_idx,
                    subgoal=subgoal,
                    subgoal_index=(
                        current_index
                    ),
                    subgoal_step=(
                        subgoal_step
                    ),
                    token=token,
                    response=response,
                    result=result,
                    subgoal_done=(
                        subgoal_done
                    ),
                    success=success,
                    env_done=bool(
                        terminated
                        or truncated
                    ),
                    recovery_decision=(
                        recovery_decision
                    ),
                    open_loop=open_loop,
                )
                if guard_decision is not None:
                    step_record["action_guard"] = guard_decision
                    if token != requested_token:
                        step_record["requested_act"] = requested_token
                if stage_action_guard is not None:
                    step_record["stage_action_guard"] = stage_action_guard
                    step_record["requested_act"] = requested_token
                if verified_grasp:
                    step_record["verified_grasp"] = True
                if stage_completion_guard is not None:
                    step_record["stage_completion_guard"] = stage_completion_guard
                if grasp_agentview_guard is not None:
                    step_record["grasp_agentview_guard"] = grasp_agentview_guard
                if grasp_verification is not None:
                    step_record["grasp_verification"] = grasp_verification
                if place_verification is not None:
                    step_record["place_verification"] = place_verification
                if place_alignment is not None:
                    step_record["place_alignment"] = place_alignment
                if move_completion_guard is not None:
                    step_record["move_completion_guard"] = move_completion_guard
                if visual_route_gate is not None:
                    step_record["visual_route_gate"] = visual_route_gate.to_dict()
                if visual_route_output:
                    route_evidence = visual_route_output.get("evidence")
                    if isinstance(route_evidence, dict):
                        step_record["visual_route"] = route_evidence
                if holding_recovery_arbiter:
                    step_record["holding_recovery_arbiter"] = (
                        holding_recovery_arbiter
                    )
                if final_check is not None:
                    step_record["final_task_check"] = final_check
                if capability_evidence:
                    step_record["capability"] = capability_evidence
                if runtime_commit is not None:
                    step_record["runtime_action_receipt"] = runtime_commit

                save_visual_artifacts = getattr(self.logger, "save_visual_artifacts", None)
                if callable(save_visual_artifacts):
                    save_visual_artifacts(
                        step_idx,
                        raw_agentview=None if raw_frames_saved_early else raw_agentview,
                        raw_wrist=None if raw_frames_saved_early else raw_wrist,
                        provider_overlay=agentview,
                        qwen_input=self._critical_qwen_input,
                    )

                self.logger.log_step(
                    step_idx=step_idx,
                    agentview=agentview,
                    wrist=wrist,
                    record=step_record,
                )

                if _should_end_after_verified_grasp(
                    self.verified_runtime, verified_grasp
                ):
                    end_reason = "grasp_verification_only_complete"
                    break

                if (
                    verified_runtime_decision.get("runtime_version")
                    and verified_runtime_decision.get("status") == "FAILED"
                ):
                    failure = verified_runtime_decision.get("failure")
                    code = (
                        str(failure.get("code") or "failed").lower()
                        if isinstance(failure, dict)
                        else "failed"
                    )
                    end_reason = f"runtime_v2_{code}"
                    break

                # Simulator success is authoritative.
                if success:
                    end_reason = "success"
                    break

                if terminated:
                    end_reason = "env_terminated_without_success"
                    break

                if truncated:
                    print(
                        "[zeroshot-robolab] "
                        f"step {step_idx}: "
                        "environment truncated"
                    )
                    end_reason = "env_truncated"
                    break

                if replan_requested:
                    reason_text = str(
                        (final_check or {}).get("reason", "the previous plan did not complete")
                    ).strip()
                    agentview_now, wrist_now = self._images(obs)
                    recovery_task = (
                        f"{self.task_description}\n\n"
                        "RECOVERY REFLECTION: The previous plan ended, but the fresh visual "
                        f"outcome check says the task is not complete: {reason_text} "
                        "Replan the remaining task from the CURRENT AgentView and Wrist images. "
                        "Do not assume the object is still held or that the old grasp/place pose "
                        "is valid. Locate the object's current silhouette, determine whether it is "
                        "tilted, outside, or still recoverable, and choose whatever grounded "
                        "approach/grasp/lift/align/release sequence the current scene requires."
                    )
                    try:
                        replanned_subgoals, replanned_raw = self.planner.plan(
                            recovery_task,
                            agentview_now,
                            wrist=wrist_now,
                            debug=self.debug,
                            image_roles=image_roles,
                        )
                    except Exception as exc:
                        end_reason = f"visual_replan_failed: {type(exc).__name__}: {exc}"
                        break
                    if not replanned_subgoals:
                        end_reason = "visual_replan_returned_no_subgoals"
                        break
                    replans += 1
                    planner_replanned_subgoals = list(replanned_subgoals)
                    if bool(
                        self.visual_route_plugin is not None
                        and getattr(self.visual_route_plugin, "enabled", False)
                        and getattr(
                            self.visual_route_plugin,
                            "transport_loop_enabled",
                            False,
                        )
                    ):
                        replanned_subgoals = (
                            self.visual_route_plugin.collapse_transport_subgoals(
                                replanned_subgoals
                            )
                        )
                    subgoals = replanned_subgoals
                    raw_plan = replanned_raw
                    diagnostics = self.planner.diagnostics()
                    self.logger.write_planner_diagnostics(diagnostics)
                    self.logger.write_plan(
                        {
                            "task": self.task_description,
                            "replan": replans,
                            "replan_reason": reason_text,
                            "raw_plan": raw_plan,
                            "planner_subgoals": [
                                sg.to_prompt_dict()
                                for sg in planner_replanned_subgoals
                            ],
                            "subgoals": [sg.to_prompt_dict() for sg in subgoals],
                        }
                    )
                    save_planner_prompt = getattr(self.logger, "save_planner_prompt", None)
                    planner_prompt = getattr(self.planner, "last_prompt", None)
                    if callable(save_planner_prompt) and callable(planner_prompt):
                        save_planner_prompt(replans + 1, str(planner_prompt()))
                    reset_agent = getattr(self.agent, "reset", None)
                    if callable(reset_agent):
                        reset_agent()
                    if self.visual_harness is not None:
                        self.visual_harness.reset()
                    if self.visual_route_plugin is not None:
                        reset_route = getattr(
                            self.visual_route_plugin,
                            "reset",
                            None,
                        )
                        if callable(reset_route):
                            reset_route()
                    self._place_verification_stage = None
                    self._place_verification_decision = None
                    self._place_verification_recovery_action = None
                    self._place_recovery_needs_lift = False
                    self._place_alignment_stage = None
                    self._place_alignment_decision = None
                    self._place_alignment_action = None
                    self._reacquire_required = False
                    self._grasp_retry_anchor = None
                    current_index = 0
                    subgoal_start_step = step_idx + 1
                    previous_direction = None
                    recent_moves.clear()
                    recovery_note = (
                        "A fresh visual outcome check found the previous plan incomplete. "
                        "Use the new plan and current images; independently reassess whether "
                        "the object is held, fallen, tilted, or already recoverable."
                    )
                    chunk_queue = []
                    continue

                if final_check is not None and not final_check.get("complete", False):
                    end_reason = "plan_complete_visual_check_failed"
                    break

                if transport_rollback_index is not None:
                    current_index = int(transport_rollback_index)
                    subgoal_start_step = step_idx + 1
                    previous_direction = None
                    recent_moves.clear()
                    chunk_queue = []
                    self._reacquire_required = True
                    if self.visual_harness is not None:
                        self.visual_harness.reset()
                    if self.visual_route_plugin is not None:
                        self.visual_route_plugin.reset()
                    recovery_note = (
                        "Temporal visual evidence and Qwen agreed that the held object "
                        "was lost. The gripper has been opened; reacquire the object "
                        "from the current live scene before starting a new transport epoch."
                    )
                    continue

                if runtime_rollback_index is not None:
                    current_index = int(runtime_rollback_index)
                    subgoal_start_step = step_idx + 1
                    previous_direction = None
                    recent_moves.clear()
                    chunk_queue = []
                    self._reacquire_required = bool(
                        str(
                            getattr(subgoals[current_index], "motion", "")
                        ).upper()
                        == "APPROACH"
                    )
                    recovery_note = (
                        "VCR-v2 verifier rejected the previous physical transition; "
                        "resume from the typed recovery option using fresh observations."
                    )
                    continue

                if (
                    recovery_decision is not None
                    and recovery_decision.rollback_index
                    is not None
                ):
                    current_index = int(
                        recovery_decision
                        .rollback_index
                    )

                    subgoal_start_step = (
                        step_idx + 1
                    )

                    previous_direction = None
                    recent_moves.clear()

                    if getattr(
                        recovery_decision,
                        "grasp_empty",
                        False,
                    ):
                        recent_moves.insert(
                            0,
                            EMPTY_GRASP_LABEL,
                        )

                    chunk_queue = []
                    continue

                if subgoal_done:
                    if (
                        current_index + 1
                        >= len(subgoals)
                    ):
                        end_reason = (
                            "plan_complete_without_env_success"
                        )
                        break

                    current_index += 1
                    subgoal_start_step = (
                        step_idx + 1
                    )

                    previous_direction = None
                    recent_moves.clear()
                    chunk_queue = []

                    if (
                        str(subgoal.motion)
                        .upper()
                        == "GRASP"
                    ):
                        recovery_note = ""

                elapsed = (
                    time.monotonic()
                    - loop_started
                )

                if (
                    self.loop_period_s
                    > elapsed
                ):
                    time.sleep(
                        self.loop_period_s
                        - elapsed
                    )

        except KeyboardInterrupt:
            end_reason = "interrupted"
            print(
                "\n[zeroshot-robolab] interrupted"
            )

        except Exception as exc:
            end_reason = "error"
            error = f"{type(exc).__name__}: {exc}"
            raise

        finally:
            capability_snapshot = (
                self.visual_harness.snapshot()
                if self.visual_harness is not None
                else None
            )
            if self.visual_harness is not None:
                self.visual_harness.close()
            video_path = self.logger.close(
                success=success,
                fps=video_fps(
                    self.config.video_fps
                ),
            )

            self.logger.write_summary(
                {
                    "success": success,
                    "steps": steps,
                    "max_steps": self.max_steps,
                    "end_reason": end_reason,
                    "error": error,
                    "video_path": str(
                        video_path
                    ),
                    "run_dir": str(
                        self.logger.run_dir
                    ),
                    "control_mode": (
                        "robolab_zeroshot_full_harness"
                    ),
                    "task": (
                        self.task_description
                    ),
                    "task_name": self.task_name,
                    "task_description": self.task_description,
                    "seed": self.seed,
                    "episode_index": self.episode_index,
                    "fingertip_offset_m": self.fingertip_offset_m,
                    "wrist_grasp_marker": self.wrist_marker_metadata,
                    "raw_plan": raw_plan,
                    "subgoals": [
                        sg.to_prompt_dict()
                        for sg in subgoals
                    ],
                    "capabilities": capability_snapshot,
                    "model_metrics": self._model_metrics(),
                }
            )

            if self.close_env:
                try:
                    self.env.close()
                except Exception:
                    pass

        return EpisodeResult(
            success=success,
            steps=steps,
            end_reason=end_reason,
            video_path=str(video_path),
            run_dir=str(
                self.logger.run_dir
            ),
        )

    def _model_metrics(self) -> dict[str, Any] | None:
        """Return host-side model accounting without exposing credentials or prompts."""
        agent = getattr(getattr(self.controls, "controller", None), "agent", None)
        client = getattr(agent, "client", None)
        metrics = getattr(client, "metrics", None)
        if not isinstance(metrics, dict):
            return None
        return {
            "provider": getattr(client, "provider", None),
            "model": getattr(client, "model", None),
            "model_calls": int(metrics.get("model_calls", 0)),
            "prompt_tokens": int(metrics.get("prompt_tokens", 0)),
            "completion_tokens": int(metrics.get("completion_tokens", 0)),
            "total_tokens": int(metrics.get("total_tokens", 0)),
            "latency_s": round(float(metrics.get("latency_s", 0.0)), 3),
            "api_cost_usd": round(float(metrics.get("api_cost_usd", 0.0)), 6),
        }

    def _record_stage(
        self,
        *,
        step_idx: int,
        subgoal,
        subgoal_index: int,
        subgoal_step: int,
        token: str,
        response: Any,
        result: SimAtomicStepResult,
        subgoal_done: bool,
        success: bool,
        env_done: bool,
        recovery_decision: Any = None,
        open_loop: bool = False,
    ) -> dict[str, Any]:

        payload = (
            getattr(
                response,
                "payload",
                None,
            )
            or {}
        )

        latency_s = payload.get(
            "latency_s"
        )

        json_payload = payload.get(
            "json"
        )

        reasoning = (
            str(
                json_payload.get(
                    "reasoning"
                )
                or ""
            )
            if isinstance(
                json_payload,
                dict,
            )
            else ""
        )

        record: dict[str, Any] = {
            "i": int(step_idx),
            "sg": int(subgoal_index),
            "sg_step": int(
                subgoal_step
            ),
            "stage": subgoal.motion,
            "sid": subgoal.id,
            "act": token,
            "eef": [
                round(float(x), 3)
                for x in self._tcp()
            ],
            "fingertip": [round(float(x), 5) for x in self._fingertip_position()],
            "w": round(self._gripper_width(), 5),
            "grip": (
                self.controller
                .state
                .gripper_name
            ),
        }

        if result.step_kind:
            record["step_kind"] = (
                result.step_kind
            )
            record["step_cm"] = round(
                float(result.step_m)
                * 100.0,
                1,
            )

        if open_loop:
            record["open_loop"] = True

        if latency_s is not None or reasoning:
            c: dict[str, Any] = {}

            if latency_s is not None:
                c["ms"] = int(
                    round(
                        float(latency_s)
                        * 1000.0
                    )
                )

            if reasoning:
                c["why"] = reasoning

            record["vlm"] = {
                "c": c
            }

        if result.grasp_empty:
            # Keep the low-level signal inspectable without calling it a semantic
            # failure: a visual verifier may correctly accept a thin/occluded hold.
            record["mechanical_grasp_empty"] = result.note or True
            record["grasp_fail"] = result.note or True

        if recovery_decision is not None:
            record["recover"] = True

            recovery_record = {}

            for key in (
                "event",
                "reason",
                "rollback_index",
                "token",
                "release",
            ):
                value = getattr(
                    recovery_decision,
                    key,
                    None,
                )

                if value not in (
                    None,
                    "",
                    False,
                ):
                    recovery_record[
                        key
                    ] = value

            if recovery_record:
                record[
                    "recovery"
                ] = recovery_record

        if subgoal_done:
            record["sg_done"] = True

        if success:
            record["ok"] = True

        if env_done:
            record["env_done"] = True

        return record

    @staticmethod
    def _reasoning(
        response: Any,
    ) -> str:
        payload = (
            getattr(
                response,
                "payload",
                None,
            )
            or {}
        )

        obj = payload.get("json")

        if isinstance(obj, dict):
            return str(
                obj.get("reasoning")
                or ""
            ).strip()

        return ""

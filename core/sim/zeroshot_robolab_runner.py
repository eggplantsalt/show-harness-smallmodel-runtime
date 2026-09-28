"""Show-Harness zero-shot staged controller on RoboLab.

The high-level semantics mirror the official real-robot zero-shot runner:
planner -> controller + plugins -> atomic execution -> measured feedback ->
recovery / rollback.

Only the physical execution backend is RoboLab-specific.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from core.action_units import MOVE_ATOMS
from core.record.images import prepare_view
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
            opening_bbox = capability_evidence.get("bbox_xyxy")
            outer_bbox = capability_evidence.get("outer_destination_bbox_xyxy")
            if opening_bbox is not None or outer_bbox is not None:
                destination = {
                    "bbox_xyxy": outer_bbox or opening_bbox,
                    "opening_bbox_xyxy": opening_bbox or outer_bbox,
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
        grasp_empty = bool(
            token == GRASP_TOKEN
            and gripper_closed
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
        return verifier(
            task=self.task_description,
            target=str(getattr(subgoal, "target", "")),
            affordance=str(getattr(subgoal, "affordance", "")),
            agentview_image=agentview,
            wrist_image=wrist,
            debug=self.debug,
        )

    def _verify_place_visually(self, *, subgoal, obs) -> dict[str, Any] | None:
        """Run one low-position placement confirmation through the high-level Agent."""
        controller = getattr(getattr(self, "controls", None), "controller", None)
        verifier = getattr(controller, "verify_place", None)
        if not callable(verifier):
            return None
        agentview, wrist = self._images(obs)
        return verifier(
            task=self.task_description,
            target=str(getattr(subgoal, "target", "")),
            affordance=str(getattr(subgoal, "affordance", "")),
            agentview_image=agentview,
            wrist_image=wrist,
            debug=self.debug,
        )

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

                capability_evidence = {}
                capability_context = ""
                visual_route_output: dict[str, Any] = {}
                visual_route_gate: Any = None
                transport_hold_lost = False
                transport_rollback_index = None
                route_holding_arbiter: dict[str, Any] = {}
                holding_recovery_arbiter: list[dict[str, Any]] = []
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

                recovery_decision = (
                    self._recovery_before(
                        current_index=current_index,
                        subgoals=subgoals,
                    )
                )
                if self.recovery_plugin is not None:
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

                # A fresh, host-owned APPROACH completion is a stage transition, not a
                # movement suggestion.  It must win over recovery/chunk heuristics; otherwise
                # a repeated-move recovery token can consume the exact frame on which the
                # target becomes aligned and leave the controller trapped in APPROACH.
                if (
                    stage_completion_guard is not None
                    and stage_completion_guard.get("applied", False)
                    and not (
                        str(getattr(subgoal, "motion", "")).upper() == "APPROACH"
                        and self._reacquire_required
                    )
                ):
                    token = DONE_TOKEN
                    chunk_queue = []

                elif (
                    grasp_agentview_guard is not None
                    and grasp_agentview_guard.get("applied", False)
                ):
                    token = GRASP_TOKEN
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
                if stage_name == "PLACE":
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
                elif token == GRASP_TOKEN and self._grasp_retry_anchor is not None:
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
                ):
                    grasp_verification = self._verify_grasp_visually(
                        subgoal=subgoal,
                        obs=obs,
                    )
                    if isinstance(grasp_verification, dict):
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
                            # intent, never a one-shot host-selected recovery atom.
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
                if agent_grasp_decision is not None:
                    # A NO or UNKNOWN visual verdict keeps the semantic GRASP stage
                    # open. NO will also request the normal release/rollback path;
                    # UNKNOWN lets the next controller decision inspect live views.
                    verified_grasp = False
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
                    if agent_grasp_decision is not None:
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

                if self.recovery_plugin is not None:
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

                self.logger.log_step(
                    step_idx=step_idx,
                    agentview=agentview,
                    wrist=wrist,
                    record=step_record,
                )

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

"""Verified Capability Runtime v2.

VCR-v2 is intentionally a runtime, not another policy model.  It consumes only
the image/proprioceptive evidence already available to the agent, maintains a
transactional physical belief, and owns bounded atomic control inside a semantic
option selected by the planner.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from core.capabilities.camera_geometry import (
    CameraCalibration,
    backproject_pixel_to_plane,
    camera_point_roundtrip_error,
    opencv_camera_points_to_world,
)

from .control import (
    DONE, MOVE_BACK, MOVE_DOWN, MOVE_FWD, MOVE_LEFT, MOVE_RIGHT, MOVE_UP,
    OPPOSITE, STOP, ControlDecision, ResidualController,
)
from .tracking import AssociationResult, EntityTracker, bbox_tuple
from .types import (
    CriticalDecisionRequest,
    EvidenceValue,
    FailureCode,
    FailureEvent,
    ObservationHealth,
    OptionName,
    OptionStatus,
    PhysicalBeliefState,
    RecoveryContext,
    TruthValue,
    Verdict,
    SemanticPregraspAction,
    SpatialBelief,
    SpatialHealth,
    SpatialRelation,
    jsonable,
)
from .verifiers import verify_hold, verify_seated
from .spatial import SpatialToolBus, classify_relation, fuse_spatial_results
from .types import SpatialToolResult
from .providers import CoTrackerOnlineProvider, MogeDepthProvider
from .placement import PlacementSpatialHarness
from .types import PlacementBelief, PlacementCandidate, PlacementRelation
from .anyplace_shadow import AnyPlaceShadowProvider


@dataclass(frozen=True)
class RuntimeLimits:
    locate: int = 10
    hover: int = 80
    align: int = 40
    transfer: int = 80
    commit: int = 15
    recovery: int = 60
    max_recovery_attempts: int = 2
    episode: int = 300


STAGE_OPTION = {
    "APPROACH": OptionName.ALIGN_PREGRASP,
    "GRASP": OptionName.DESCEND_TO_GRASP,
    "LIFT": OptionName.LIFT_CLEAR,
    "MOVE": OptionName.TRANSFER,
    "TRANSPORT": OptionName.TRANSFER,
    "PLACE": OptionName.ALIGN_OPENING,
    "RELEASE": OptionName.OPEN_GRIPPER,
}

PREGRASP_ACTIONS = (
    "MV_LEFT",
    "MV_RIGHT",
    "MV_FWD",
    "MV_BACK",
    "MV_UP",
    "MV_DOWN",
    "GRASP",
    "UNKNOWN",
)

SEMANTIC_PREGRASP_ACTIONS = tuple(item.value for item in SemanticPregraspAction)


def _candidate_spatial_key(candidate: Any) -> tuple[float, float, float, float]:
    box = bbox_tuple(candidate.get("bbox_xyxy")) if isinstance(candidate, dict) else None
    if box is None:
        return (math.inf, math.inf, math.inf, math.inf)
    x1, y1, x2, y2 = box
    return (y1, x1, y2, x2)


class VerifiedCapabilityRuntime:
    """State/option/verifier runtime with a V1-compatible decision contract."""

    runtime_version = "vcr-v2.1"

    def __init__(
        self,
        *,
        enabled: bool = True,
        mode: str = "active",
        alignment_px: float = 13.5,
        confidence_threshold: float = 0.4,
        empty_width_m: float = 0.004,
        lift_clear_height_m: float = 0.17,
        approach_min_height_m: Optional[float] = None,
        approach_max_height_m: Optional[float] = None,
        height_tolerance_m: float = 0.003,
        support_plane_z_m: float = 0.015,
        world_alignment_tolerance_m: float = 0.008,
        wrist_grasp_anchor_px: tuple[float, float] = (128.0, 70.0),
        wrist_alignment_fraction: float = 0.20,
        wrist_alignment_min_px: float = 4.0,
        wrist_final_descent_steps: int = 1,
        sensor_fault_limit: int = 3,
        min_progress_px: float = 1.0,
        wrong_direction_px: float = 1.5,
        no_progress_limit: int = 3,
        wrong_direction_limit: int = 2,
        cycle_window: int = 8,
        effect_alpha: float = 0.5,
        axis_hold_steps: int = 3,
        semantic_pregrasp_enabled: bool = False,
        require_spatial_ready_for_grasp: bool = False,
        pregrasp_target_crop_enabled: bool = False,
        allow_agentview_identity_fallback: bool = False,
        depth_positive_action: str = "MV_FWD",
        depth_negative_action: str = "MV_BACK",
        lateral_positive_action: str = "MV_LEFT",
        lateral_negative_action: str = "MV_RIGHT",
        height_positive_action: str = "MV_UP",
        height_negative_action: str = "MV_DOWN",
        probe_depth_action: str = "MV_BACK",
        temporal_identity_lock_on_detector_ties: bool = False,
        metric_approach_uses_hover_budget: bool = False,
        semantic_evidence_max_age_frames: int = 1,
        grasp_reobserve_step_m: float = 0.02,
        visual_memory_max_entries: int = 8,
        spatial_max_age_frames: int = 2,
        spatial_propagation_max_age_frames: int = 8,
        recovery_descent_stall_tolerance_m: float = 0.0015,
        placement_v22_enabled: bool = False,
        placement_action_step_m: float = 0.01,
        placement_enter_margin_m: float = 0.02,
        placement_exit_margin_m: float = 0.01,
        placement_confirm_frames: int = 2,
        placement_reflection_mode: str = "off",
        pregrasp_reflection_mode: str = "off",
        grasp_diagnostic_min_lift_m: float = 0.005,
        grasp_verification_only: bool = False,
        anyplace_shadow_provider: Optional[AnyPlaceShadowProvider] = None,
        limits: RuntimeLimits = RuntimeLimits(),
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = str(mode or "active").lower()
        if self.mode not in {"shadow", "active"}:
            raise ValueError("runtime_v2.mode must be shadow or active")
        self.alignment_px = max(1.0, float(alignment_px))
        self.confidence_threshold = float(confidence_threshold)
        self.empty_width_m = max(0.0, float(empty_width_m))
        self.lift_clear_height_m = float(lift_clear_height_m)
        self.approach_min_height_m = (
            None if approach_min_height_m is None else float(approach_min_height_m)
        )
        self.approach_max_height_m = (
            None if approach_max_height_m is None else float(approach_max_height_m)
        )
        self.height_tolerance_m = max(0.0, float(height_tolerance_m))
        self.support_plane_z_m = float(support_plane_z_m)
        self.world_alignment_tolerance_m = max(
            0.001, float(world_alignment_tolerance_m)
        )
        self.wrist_grasp_anchor_px = tuple(float(v) for v in wrist_grasp_anchor_px)
        self.wrist_alignment_fraction = max(0.05, float(wrist_alignment_fraction))
        self.wrist_alignment_min_px = max(1.0, float(wrist_alignment_min_px))
        self.wrist_final_descent_steps = max(0, int(wrist_final_descent_steps))
        self.semantic_pregrasp_enabled = bool(semantic_pregrasp_enabled)
        self.require_spatial_ready_for_grasp = bool(require_spatial_ready_for_grasp)
        self.pregrasp_target_crop_enabled = bool(pregrasp_target_crop_enabled)
        self.allow_agentview_identity_fallback = bool(
            allow_agentview_identity_fallback
        )
        self.temporal_identity_lock_on_detector_ties = bool(
            temporal_identity_lock_on_detector_ties
        )
        self.metric_approach_uses_hover_budget = bool(
            metric_approach_uses_hover_budget
        )
        self.semantic_evidence_max_age_frames = max(
            0, int(semantic_evidence_max_age_frames)
        )
        self.grasp_reobserve_step_m = max(0.001, float(grasp_reobserve_step_m))
        self.axis_actions = {
            "depth_positive": str(depth_positive_action).upper(),
            "depth_negative": str(depth_negative_action).upper(),
            "lateral_positive": str(lateral_positive_action).upper(),
            "lateral_negative": str(lateral_negative_action).upper(),
            "height_positive": str(height_positive_action).upper(),
            "height_negative": str(height_negative_action).upper(),
            "probe_depth": str(probe_depth_action).upper(),
        }
        from .memory import VisualMemory
        self.visual_memory_max_entries = max(1, int(visual_memory_max_entries))
        self.visual_memory = VisualMemory(max_entries_per_epoch=self.visual_memory_max_entries)
        # A metric residual is tied to the EEF pose at the frame that produced
        # it.  If no new Wrist/tool observation arrives, keeping that residual
        # indefinitely turns a valid old estimate into a false command after
        # the robot has moved.  This is a temporal-freshness contract, not a
        # task- or object-specific threshold.
        self.spatial_max_age_frames = max(0, int(spatial_max_age_frames))
        self.spatial_propagation_max_age_frames = max(
            self.spatial_max_age_frames,
            int(spatial_propagation_max_age_frames),
        )
        # A recovery descent can be stopped by an embodiment safety/contact
        # boundary before the nominal pregrasp height band is reached.  This
        # tolerance is applied only to the measured EEF displacement following
        # an actually executed downward command; it is not a target/object
        # height and cannot authorize a grasp on its own.
        self.recovery_descent_stall_tolerance_m = max(
            0.0, float(recovery_descent_stall_tolerance_m)
        )
        self.placement_v22_enabled = bool(placement_v22_enabled)
        self.grasp_verification_only = bool(
            grasp_verification_only and self.placement_v22_enabled
        )
        self.grasp_diagnostic_min_lift_m = max(0.001, float(grasp_diagnostic_min_lift_m))
        self.placement_reflection_mode = str(placement_reflection_mode).lower()
        if self.placement_reflection_mode not in {"off", "single", "double"}:
            raise ValueError("placement_reflection_mode must be off, single, or double")
        self.pregrasp_reflection_mode = str(pregrasp_reflection_mode).lower()
        if self.pregrasp_reflection_mode not in {"off", "single", "double"}:
            raise ValueError("pregrasp_reflection_mode must be off, single, or double")
        self.placement_harness = PlacementSpatialHarness(
            action_step_m=placement_action_step_m,
            enter_margin_m=placement_enter_margin_m,
            exit_margin_m=placement_exit_margin_m,
            confirm_frames=placement_confirm_frames,
        )
        self.grasp_diagnostic_lift_step_m = max(0.001, float(placement_action_step_m))
        self.spatial_bus = SpatialToolBus()
        self.visual_point_tracker: Optional[CoTrackerOnlineProvider] = None
        self.sensor_fault_limit = max(1, int(sensor_fault_limit))
        self.limits = limits
        self.target_trackers = {
            "agentview": EntityTracker(role="target", camera="agentview"),
            "wrist": EntityTracker(role="target", camera="wrist"),
        }
        self.destination_trackers = {
            "agentview": EntityTracker(role="destination", camera="agentview"),
            "wrist": EntityTracker(role="destination", camera="wrist"),
        }
        # Compatibility aliases used by offline probes/tests.
        self.target_tracker = self.target_trackers["agentview"]
        self.destination_tracker = self.destination_trackers["agentview"]
        self.controller = ResidualController(
            min_progress_px=min_progress_px,
            wrong_direction_px=wrong_direction_px,
            no_progress_limit=no_progress_limit,
            wrong_direction_limit=wrong_direction_limit,
            cycle_window=cycle_window,
            effect_alpha=effect_alpha,
            axis_hold_steps=axis_hold_steps,
        )
        self.anyplace_shadow = anyplace_shadow_provider
        self.reset()

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Optional["VerifiedCapabilityRuntime"]:
        section = cfg.get("runtime_v2")
        if not isinstance(section, dict) or not bool(section.get("enabled", False)):
            return None
        capabilities = cfg.get("capabilities")
        capabilities = capabilities if isinstance(capabilities, dict) else {}
        budgets = section.get("budgets")
        budgets = budgets if isinstance(budgets, dict) else {}
        runtime = cls(
            enabled=True,
            mode=str(section.get("mode", "active")),
            alignment_px=float(
                section.get("alignment_px", capabilities.get("alignment_ready_px", 13.5))
            ),
            confidence_threshold=float(
                section.get("confidence_threshold", capabilities.get("confidence_threshold", 0.4))
            ),
            empty_width_m=float(section.get("empty_width_m", cfg.get("empty_width_m", 0.004))),
            lift_clear_height_m=float(
                section.get("lift_clear_height_m", cfg.get("lift_clear_height_m", 0.17))
            ),
            approach_min_height_m=capabilities.get(
                "approach_completion_min_height_m"
            ),
            approach_max_height_m=capabilities.get(
                "approach_completion_max_height_m"
            ),
            height_tolerance_m=float(section.get("height_tolerance_m", 0.003)),
            support_plane_z_m=float(
                section.get("support_plane_z_m", cfg.get("table_height_m", 0.015))
            ),
            world_alignment_tolerance_m=float(
                section.get("world_alignment_tolerance_m", 0.008)
            ),
            wrist_grasp_anchor_px=tuple(
                section.get("wrist_grasp_anchor_px", (128.0, 70.0))
            ),
            wrist_alignment_fraction=float(
                section.get("wrist_alignment_fraction", 0.20)
            ),
            wrist_alignment_min_px=float(
                section.get("wrist_alignment_min_px", 4.0)
            ),
            wrist_final_descent_steps=int(
                section.get("wrist_final_descent_steps", 1)
            ),
            sensor_fault_limit=int(section.get("sensor_fault_limit", 3)),
            min_progress_px=float(section.get("min_progress_px", 1.0)),
            wrong_direction_px=float(section.get("wrong_direction_px", 1.5)),
            no_progress_limit=int(section.get("no_progress_limit", 3)),
            wrong_direction_limit=int(section.get("wrong_direction_limit", 2)),
            cycle_window=int(section.get("cycle_window", 8)),
            effect_alpha=float(section.get("effect_alpha", 0.5)),
            axis_hold_steps=int(section.get("axis_hold_steps", 3)),
            semantic_pregrasp_enabled=bool(section.get("semantic_pregrasp_enabled", False)),
            require_spatial_ready_for_grasp=bool(section.get("require_spatial_ready_for_grasp", False)),
            pregrasp_target_crop_enabled=bool(
                section.get("pregrasp_target_crop_enabled", False)
                or (
                    isinstance(section.get("visual_memory"), dict)
                    and section["visual_memory"].get(
                        "pregrasp_target_crop_enabled", False
                    )
                )
            ),
            allow_agentview_identity_fallback=bool(
                section.get("spatial_tools", {}).get(
                    "allow_agentview_identity_fallback", False
                )
            ) if isinstance(section.get("spatial_tools"), dict) else False,
            depth_positive_action=str(section.get("depth_positive_action", "MV_FWD")),
            depth_negative_action=str(section.get("depth_negative_action", "MV_BACK")),
            lateral_positive_action=str(section.get("lateral_positive_action", "MV_LEFT")),
            lateral_negative_action=str(section.get("lateral_negative_action", "MV_RIGHT")),
            height_positive_action=str(section.get("height_positive_action", "MV_UP")),
            height_negative_action=str(section.get("height_negative_action", "MV_DOWN")),
            probe_depth_action=str(section.get("probe_depth_action", "MV_BACK")),
            temporal_identity_lock_on_detector_ties=bool(
                section.get("temporal_identity_lock_on_detector_ties", False)
            ),
            metric_approach_uses_hover_budget=bool(
                section.get("metric_approach_uses_hover_budget", False)
            ),
            semantic_evidence_max_age_frames=int(
                section.get("visual_memory", {}).get("semantic_evidence_max_age_frames", 1)
            ) if isinstance(section.get("visual_memory"), dict) else 1,
            grasp_reobserve_step_m=float(section.get("grasp_reobserve_step_m", 0.02)),
            visual_memory_max_entries=int(section.get("visual_memory", {}).get("max_entries_per_epoch", 8)) if isinstance(section.get("visual_memory"), dict) else 8,
            spatial_max_age_frames=int(section.get("spatial_tools", {}).get("max_age_frames", 2)) if isinstance(section.get("spatial_tools"), dict) else 2,
            spatial_propagation_max_age_frames=int(section.get("spatial_tools", {}).get("propagation_max_age_frames", 8)) if isinstance(section.get("spatial_tools"), dict) else 8,
            recovery_descent_stall_tolerance_m=float(
                section.get("recovery_descent_stall_tolerance_m", 0.0015)
            ),
            placement_v22_enabled=bool(section.get("placement_v22_enabled", False)),
            placement_action_step_m=float(section.get("placement_action_step_m", 0.01)),
            placement_enter_margin_m=float(section.get("placement_enter_margin_m", 0.02)),
            placement_exit_margin_m=float(section.get("placement_exit_margin_m", 0.01)),
            placement_confirm_frames=int(section.get("placement_confirm_frames", 2)),
            placement_reflection_mode=str(section.get("placement_reflection_mode", "off")),
            pregrasp_reflection_mode=str(section.get("pregrasp_reflection_mode", "off")),
            grasp_diagnostic_min_lift_m=float(section.get("grasp_diagnostic_min_lift_m", 0.005)),
            grasp_verification_only=bool(section.get("grasp_verification_only", False)),
            limits=RuntimeLimits(
                locate=int(budgets.get("locate", 10)),
                hover=int(budgets.get("hover", 80)),
                align=int(budgets.get("align", 40)),
                transfer=int(budgets.get("transfer", 80)),
                commit=int(budgets.get("commit", 15)),
                recovery=int(budgets.get("recovery", 60)),
                max_recovery_attempts=int(budgets.get("max_recovery_attempts", 2)),
                episode=int(budgets.get("episode", 300)),
            ),
        )
        spatial_cfg = section.get("spatial_tools")
        if isinstance(spatial_cfg, dict) and bool(spatial_cfg.get("enabled", False)):
            provider = str(spatial_cfg.get("provider", "")).lower()
            if "moge" in provider:
                moge_provider = MogeDepthProvider(
                    repo_dir=str(spatial_cfg.get("repo_dir", "/root/autodl-tmp/openeta-services/MoGe")),
                    checkpoint=str(spatial_cfg.get("checkpoint", "Ruicheng/moge-3-vitl")),
                    device=str(spatial_cfg.get("device", "cuda:1")),
                    max_latency_s=float(spatial_cfg.get("max_latency_s", 0.5)),
                    model_version=str(spatial_cfg.get("model_version", "v3")),
                    service_python=str(spatial_cfg.get("service_python")) if spatial_cfg.get("service_python") else None,
                )
                runtime.spatial_bus.register(moge_provider.source, moge_provider)
        tracking_cfg = section.get("visual_tracking")
        if isinstance(tracking_cfg, dict) and bool(tracking_cfg.get("enabled", False)):
            runtime.visual_point_tracker = CoTrackerOnlineProvider(
                repo_dir=str(tracking_cfg.get("repo_dir", "/root/autodl-tmp/openeta-services/CoTracker")),
                checkpoint=str(tracking_cfg.get("checkpoint", "/root/autodl-tmp/openeta-services/checkpoints/cotracker3/scaled_online.pth")),
                device=str(tracking_cfg.get("device", "cuda:1")),
                window_len=int(tracking_cfg.get("window_len", 16)),
                max_points=int(tracking_cfg.get("max_points", 64)),
                min_visible_points=int(tracking_cfg.get("min_visible_points", 4)),
            )
        runtime.anyplace_shadow = AnyPlaceShadowProvider.from_config(
            section.get("anyplace_shadow")
        )
        return runtime

    def reset(self, *, episode_id: Optional[str] = None) -> None:
        self.belief = PhysicalBeliefState()
        for tracker in self.target_trackers.values():
            tracker.reset()
        for tracker in self.destination_trackers.values():
            tracker.reset()
        self.controller.reset()
        self.visual_memory.reset(episode_id=episode_id)
        if hasattr(self, "_pregrasp_reflection_seen"):
            self._pregrasp_reflection_seen.clear()
        if hasattr(self, "_grasp_entry_review_seen"):
            self._grasp_entry_review_seen.clear()
        if hasattr(self, "_grasp_entry_reobserve_used"):
            self._grasp_entry_reobserve_used.clear()
        if hasattr(self, "_grasp_entry_reobserve_expected_z_m"):
            self._grasp_entry_reobserve_expected_z_m.clear()
        if hasattr(self, "_grasp_entry_followup_seen"):
            self._grasp_entry_followup_seen.clear()
        if hasattr(self, "_placement_reflection_seen"):
            self._placement_reflection_seen.clear()
        self.sensor_fault_streak = 0
        self.episode_steps = 0
        self.option_steps = 0
        self.recovery_steps = 0
        self.last_stage = ""
        self._wrist_final_descent_count = 0
        self.last_event: dict[str, Any] = {}
        self._pending_candidates: tuple[dict[str, Any], ...] = ()
        self._pending_image: Optional[np.ndarray] = None
        self._pending_frame_id: Optional[int] = None
        self._pending_role: str = "target"
        self._pending_tracker: Optional[EntityTracker] = None
        self._pending_critical_kind: str = ""
        self._pending_allowed_answers: tuple[str, ...] = ()
        self._pending_evidence_frame_ids: tuple[int, ...] = ()
        self._pending_control_error: Optional[tuple[float, float]] = None
        # A placement verifier describes the physical relation; it does not
        # own the next motion.  Keep one bounded resume relation so the next
        # route observation can compile the motion from metric residuals
        # without immediately asking the same semantic question again.
        self._placement_verifier_relation: Optional[PlacementRelation] = None
        self._placement_verifier_action_used = False
        self._placement_reflection_seen: set[tuple[int, int, str]] = set()
        self._pregrasp_reflection_seen: set[tuple[int, int, str]] = set()
        self._grasp_entry_review_seen: set[tuple[int, str]] = set()
        self._grasp_entry_reobserve_used: set[tuple[int, str]] = set()
        self._grasp_entry_reobserve_expected_z_m: dict[tuple[int, str], float] = {}
        self._grasp_entry_followup_seen: set[tuple[int, str]] = set()
        self._last_placement_phase = ""
        self._placement_contact_verified = False
        self._pending_contact_supported = False
        self._placement_support_candidate: Optional[dict[str, Any]] = None
        self._placement_support_stability_count = 0
        self._placement_support_confirm_frame: Optional[int] = None
        self._placement_seated_hypothesis: Optional[dict[str, Any]] = None
        self._placement_effect_no_progress = 0
        # A stale/unknown placement provider may be given one reversible
        # high-clearance observation probe.  The key prevents an unchanged
        # provider failure from becoming an implicit motion loop.
        self._placement_unknown_probe_key: Optional[tuple[Any, ...]] = None
        self._placement_unknown_probe_used = False
        self._grasp_candidate: Optional[dict[str, Any]] = None
        self._grasp_diagnostic_lift_authorized = False
        self._grasp_diagnostic_lift_attempted = False
        self._grasp_lift_observation: Optional[dict[str, Any]] = None
        self._grasp_hold_confirmed = False
        self.recovery_attempt_counts: dict[str, int] = {}
        self.placement_harness.reset()
        if self.visual_point_tracker is not None:
            self.visual_point_tracker.reset()
        self.placement_belief = PlacementBelief()
        self.placement_object_to_gripper_xyz: Optional[tuple[float, float, float]] = None
        self.placement_object_points_gripper: tuple[tuple[float, float, float], ...] = ()
        if not hasattr(self, "anyplace_shadow"):
            self.anyplace_shadow = None

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "runtime_version": self.runtime_version,
            "authority": "vcr_runtime",
            "online_models": ["configured_qwen8b", "sam3", "optional_spatial_capability_providers"],
            "privileged_state_to_policy": False,
            "alignment_px": self.alignment_px,
            "semantic_pregrasp_enabled": self.semantic_pregrasp_enabled,
            "require_spatial_ready_for_grasp": self.require_spatial_ready_for_grasp,
            "temporal_identity_lock_on_detector_ties": self.temporal_identity_lock_on_detector_ties,
            "metric_approach_uses_hover_budget": self.metric_approach_uses_hover_budget,
            "semantic_evidence_max_age_frames": self.semantic_evidence_max_age_frames,
            "axis_actions": dict(self.axis_actions),
            "visual_memory": {"max_entries_per_epoch": self.visual_memory_max_entries},
            "spatial_max_age_frames": self.spatial_max_age_frames,
            "spatial_propagation_max_age_frames": self.spatial_propagation_max_age_frames,
            "recovery_descent_stall_tolerance_m": self.recovery_descent_stall_tolerance_m,
            "placement_v22_enabled": self.placement_v22_enabled,
            "grasp_verification_only": self.grasp_verification_only,
            "placement_reflection_mode": self.placement_reflection_mode,
            "pregrasp_reflection_mode": self.pregrasp_reflection_mode,
            "placement_authority": "vcr_runtime" if self.placement_v22_enabled else "option_runtime",
            "placement_object_to_gripper_xyz": self.placement_object_to_gripper_xyz,
            "placement_object_point_count": len(self.placement_object_points_gripper),
            "anyplace_shadow": self.anyplace_shadow.metadata()
            if self.anyplace_shadow is not None
            else {"enabled": False},
            "sensor_fault_limit": self.sensor_fault_limit,
            "approach_height_band_m": [
                self.approach_min_height_m,
                self.approach_max_height_m,
            ],
            "budgets": jsonable(self.limits),
        }

    def infer_spatial(
        self,
        *,
        image: Optional[np.ndarray],
        bbox_xyxy: Optional[tuple[float, float, float, float]],
        frame_id: Optional[int],
        instance_id: Optional[str],
        camera_calibration: Optional[dict[str, Any]] = None,
        eef_xyz: Optional[tuple[float, float, float]] = None,
        extra_results: Optional[Sequence[SpatialToolResult]] = None,
        instance_mask: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Run registered spatial providers on a detector crop in shadow-safe form."""
        extra_results = [item for item in (extra_results or ()) if isinstance(item, SpatialToolResult)]
        if image is None and not extra_results:
            return {"health": SpatialHealth.UNKNOWN.value, "relations": [SpatialRelation.UNKNOWN.value], "frame_id": frame_id, "instance_id": instance_id, "diagnostics": {"reason": "no_provider_or_image"}}
        mask = None
        if instance_mask is not None and image is not None:
            selected = CoTrackerOnlineProvider._decode_mask(
                instance_mask, np.asarray(image).shape[:2]
            )
            if selected is not None:
                mask = selected
        calibration = camera_calibration if isinstance(camera_calibration, dict) else {}
        if image is not None and self.spatial_bus.providers:
            self.spatial_bus.infer(
                image=image, mask=mask, bbox_xyxy=bbox_xyxy,
                frame_id=frame_id, instance_id=instance_id,
                camera_calibration=calibration,
            )
        else:
            self.spatial_bus.last_results = []
        # Providers return points in their camera frame.  Convert the robust
        # median to world/gripper-relative coordinates using ordinary calibration
        # and proprioception, never simulator object state.
        results: list[SpatialToolResult] = []
        rotation = np.asarray(calibration.get("camera_to_world"), dtype=float) if calibration.get("camera_to_world") is not None else None
        camera_position = np.asarray(calibration.get("position_world"), dtype=float) if calibration.get("position_world") is not None else None
        eef = np.asarray(eef_xyz, dtype=float) if eef_xyz is not None else None
        raw_results = list(self.spatial_bus.last_results) + extra_results
        # Once calibrated active parallax is available, the monocular fallback
        # is retained in diagnostics but cannot numerically override it.  This
        # avoids fusing an uncalibrated monocular scale with metric triangulation.
        if any(item.source == "active_parallax" and item.health == SpatialHealth.VALID for item in extra_results):
            raw_results = [item for item in raw_results if item.source == "active_parallax"]
        for result in raw_results:
            points = np.asarray(result.target_points_camera, dtype=float).reshape(-1, 3) if result.target_points_camera else np.empty((0, 3))
            relative = result.target_to_gripper_xyz
            world_points: tuple[tuple[float, float, float], ...] = result.target_points_world
            if relative is None and len(points) and rotation is not None and rotation.shape == (3, 3) and camera_position is not None and camera_position.shape == (3,) and eef is not None:
                diagnostics = dict(result.diagnostics)
                if str(result.source).lower().startswith("moge"):
                    try:
                        from core.capabilities.camera_geometry import opencv_camera_points_to_world

                        world = opencv_camera_points_to_world(
                            points,
                            camera_to_world=rotation,
                            position_world=camera_position,
                            rotation_degrees=int(calibration.get("rotation_degrees", 0)),
                            flip=str(calibration.get("flip", "none")),
                        )
                        pixels = np.asarray(result.target_points_pixels, dtype=float).reshape(-1, 2)
                        if len(pixels) != len(world):
                            raise ValueError("MoGe point-to-pixel correspondences are missing")
                        calibration_obj = CameraCalibration(
                            name=str(calibration.get("name", "camera")),
                            width=int(calibration["width"]),
                            height=int(calibration["height"]),
                            fovy_deg=float(calibration["fovy_deg"]),
                            position_world=camera_position,
                            camera_to_world=rotation,
                            rotation_degrees=int(calibration.get("rotation_degrees", 0)),
                            flip=str(calibration.get("flip", "none")),
                        )
                        residuals = camera_point_roundtrip_error(calibration_obj, world, pixels)
                        finite = residuals[np.isfinite(residuals)]
                        diagnostics["coordinate_contract"] = "opencv_raw_sensor_to_mujoco_camera_to_world"
                        diagnostics["roundtrip_projectable_count"] = int(len(finite))
                        diagnostics["roundtrip_sample_count"] = int(len(residuals))
                        diagnostics["roundtrip_median_px"] = float(np.median(finite)) if len(finite) else None
                        diagnostics["roundtrip_max_px"] = float(np.max(finite)) if len(finite) else None
                        if len(finite) < max(4, int(np.ceil(0.5 * len(residuals)))):
                            raise ValueError("too few MoGe points reproject into the calibrated raw image")
                        if float(np.median(finite)) > 8.0:
                            raise ValueError("MoGe metric points fail camera reprojection check")
                        result_health = result.health
                    except (KeyError, TypeError, ValueError, IndexError) as exc:
                        world = np.empty((0, 3), dtype=float)
                        diagnostics["coordinate_error"] = str(exc)
                        result_health = SpatialHealth.UNKNOWN
                else:
                    world = (rotation @ points.T).T + camera_position
                    result_health = result.health
                if len(world):
                    center = np.median(world, axis=0)
                    relative = tuple(float(x) for x in (center - eef))
                    world_points = tuple(tuple(float(x) for x in row) for row in world[::max(1, len(world) // 32)])
                    if str(result.source).lower().startswith("moge"):
                        # Pixel reprojection validates camera orientation but
                        # cannot validate metric depth scale: any point along
                        # a camera ray projects to the same pixel.  Reject a
                        # MoGe cloud whose robust center falls below the
                        # configured physical support plane, even when its
                        # image reprojection is excellent.
                        median_world_z = float(np.median(world[:, 2]))
                        plane_tolerance = max(
                            0.005, float(self.world_alignment_tolerance_m)
                        )
                        diagnostics["support_plane_check"] = {
                            "support_plane_z_m": float(self.support_plane_z_m),
                            "median_target_z_m": median_world_z,
                            "tolerance_m": plane_tolerance,
                            "passed": bool(
                                median_world_z
                                >= float(self.support_plane_z_m) - plane_tolerance
                            ),
                        }
                        if not diagnostics["support_plane_check"]["passed"]:
                            result_health = SpatialHealth.UNKNOWN
                else:
                    result_health = SpatialHealth.UNKNOWN
            else:
                diagnostics = dict(result.diagnostics)
                result_health = result.health if relative is not None else SpatialHealth.UNKNOWN
            results.append(SpatialToolResult(
                source=result.source,
                instance_id=result.instance_id,
                frame_id=result.frame_id,
                target_points_camera=result.target_points_camera,
                target_points_pixels=result.target_points_pixels,
                target_points_world=world_points,
                target_to_gripper_xyz=relative,
                covariance=result.covariance,
                uncertainty_std_m=result.uncertainty_std_m,
                confidence=result.confidence,
                health=result_health if relative is not None else SpatialHealth.UNKNOWN,
                stale=result.stale,
                diagnostics=diagnostics,
            ))
        rejected_provider_audits = [
            {
                "source": item.source,
                "instance_id": item.instance_id,
                "frame_id": item.frame_id,
                "health": item.health.value,
                "diagnostics": dict(item.diagnostics),
            }
            for item in results
            if item.health != SpatialHealth.VALID
        ]
        belief = fuse_spatial_results(
            results,
            frame_id=frame_id,
            instance_id=instance_id,
            require_uncertainty=self.placement_v22_enabled,
        )
        payload = jsonable(belief)
        if rejected_provider_audits:
            payload.setdefault("diagnostics", {})[
                "rejected_providers"
            ] = rejected_provider_audits
        return payload

    def track_visual_points(
        self, *, image: Optional[np.ndarray], mask: Any, camera: str,
        frame_id: int, instance_id: Optional[str], grasp_epoch: int,
    ) -> Optional[dict[str, Any]]:
        """Return grounded within-camera CoTracker correspondences when enabled."""
        if self.visual_point_tracker is None or image is None:
            return None
        try:
            result = self.visual_point_tracker.infer(
                image=image, mask=mask, camera=camera, frame_id=frame_id,
                instance_id=instance_id, grasp_epoch=grasp_epoch,
            )
        except Exception as exc:
            return {
                "health": "SENSOR_FAULT",
                "reason": f"{type(exc).__name__}: {exc}",
                "camera": str(camera).lower(),
                "frame_id": int(frame_id),
            }
        if not isinstance(result, dict):
            return {"health": "UNAVAILABLE", "reason": "tracker_returned_non_object"}
        if str(result.get("health", "")).upper() == "VALID":
            try:
                before = np.asarray(result["query_points_xy"], dtype=float).reshape(-1, 2)
                after = np.asarray(result["current_points_xy"], dtype=float).reshape(-1, 2)
                if before.shape != after.shape or not len(before):
                    raise ValueError("track point shape mismatch")
                delta = after - before
                finite = np.isfinite(delta).all(axis=1)
                if not finite.any():
                    raise ValueError("no finite point displacements")
                median = np.median(delta[finite], axis=0)
                result["motion_summary"] = {
                    "median_displacement_px": [round(float(value), 3) for value in median],
                    "median_track_motion_px": round(
                        float(np.median(np.linalg.norm(delta[finite], axis=1))), 3
                    ),
                    "visible_tracks": int(finite.sum()),
                }
            except (KeyError, TypeError, ValueError):
                result["health"] = "AMBIGUOUS"
                result["reason"] = "invalid_track_correspondences"
        return result

    def infer_placement_shadow(
        self,
        *,
        parent_points: Any,
        child_points: Any,
        frame_id: Optional[int] = None,
        instance_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run AnyPlace as evidence only; candidates never authorize motion."""
        if self.anyplace_shadow is None:
            return {"health": "UNKNOWN", "reason": "anyplace_shadow_disabled", "candidates": []}
        return jsonable(
            self.anyplace_shadow.infer(
                parent_points=parent_points,
                child_points=child_points,
                frame_id=frame_id,
                instance_id=instance_id,
            )
        )

    @staticmethod
    def _tool_error(value: Any) -> Optional[str]:
        if isinstance(value, dict):
            error = value.get("error")
            if error:
                return str(error)
            backend = value.get("backend_metadata")
            nested = VerifiedCapabilityRuntime._tool_error(backend)
            if nested:
                return nested
            for key in ("primary", "fallback", "probe", "secondary", "opening"):
                nested = VerifiedCapabilityRuntime._tool_error(value.get(key))
                if nested:
                    return nested
        return None

    def _health(self, evidence: dict[str, Any]) -> tuple[ObservationHealth, str]:
        source = str(evidence.get("source") or "unknown")
        tool = evidence.get("tool")
        error = self._tool_error(tool)
        if error or source == "sam3_disabled":
            self.sensor_fault_streak += 1
            return ObservationHealth.SENSOR_FAULT, error or source
        self.sensor_fault_streak = 0
        abstain = ""
        if isinstance(tool, dict):
            abstain = str(tool.get("abstain_reason") or "")
        if abstain == "ambiguous_top_detections" or source == "instance_association_rejected":
            return ObservationHealth.AMBIGUOUS, abstain or source
        if source == "occlusion_memory":
            return ObservationHealth.OCCLUDED, "recent target occlusion"
        if not bool(evidence.get("visible", False)):
            return ObservationHealth.OCCLUDED, abstain or source
        return ObservationHealth.VALID, source

    @staticmethod
    def _geometry_error(evidence: dict[str, Any]) -> Optional[tuple[float, float]]:
        geometry = evidence.get("geometry")
        geometry = geometry if isinstance(geometry, dict) else {}
        value = geometry.get("target_minus_eef_px")
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None

    def _current_visual_alignment(
        self, evidence: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Return a fresh AgentView pixel residual for one bounded alignment.

        The target identity must already be locked in the current frame.  When
        Wrist is the primary detector, AgentView can supply this residual only
        through its separately associated same-instance view; pixel coordinates
        are never mixed across cameras.
        """
        target = self.belief.target
        frame_id = self.belief.frame_id
        if (
            target is None
            or frame_id is None
            or self.belief.observation_health != ObservationHealth.VALID
            or str(target.camera).lower() != "agentview"
            or target.last_confirmed_frame != int(frame_id)
            or not target.instance_id
        ):
            return None

        camera = str(evidence.get("camera") or "").lower()
        geometry = None
        if camera == "agentview":
            geometry = evidence.get("geometry")
        else:
            secondary = evidence.get("secondary_view")
            secondary = secondary if isinstance(secondary, dict) else {}
            association = secondary.get("instance_association")
            association = association if isinstance(association, dict) else {}
            try:
                association_frame = int(association.get("frame_id", -1))
            except (TypeError, ValueError, OverflowError):
                return None
            if (
                str(secondary.get("camera") or "").lower() != "agentview"
                or str(association.get("health") or "").upper() != "VALID"
                or str(association.get("instance_id") or "")
                != str(target.instance_id)
                or association_frame != int(frame_id)
            ):
                return None
            geometry = secondary.get("geometry")
        if not isinstance(geometry, dict):
            return None
        try:
            if geometry.get("valid") is False or geometry.get("in_frame") is False:
                return None
            geometry_frame = geometry.get("frame_id")
            if geometry_frame is not None and int(geometry_frame) != int(frame_id):
                return None
            error = tuple(float(value) for value in geometry["target_minus_eef_px"])
            candidates = geometry.get("calibrated_correction_candidates")
            if (
                len(error) != 2
                or not all(math.isfinite(value) for value in error)
                or not isinstance(candidates, dict)
                or not isinstance(geometry.get("camera_calibration"), dict)
            ):
                return None
            candidates = {
                "horizontal": str(candidates.get("horizontal") or "").upper(),
                "vertical": str(candidates.get("vertical") or "").upper(),
            }
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if any(
            action not in {MOVE_LEFT, MOVE_RIGHT, MOVE_FWD, MOVE_BACK}
            for action in candidates.values()
        ):
            return None
        return {
            "valid": True,
            "frame_id": int(frame_id),
            "instance_id": str(target.instance_id),
            "camera": "agentview",
            "target_minus_eef_px": [float(error[0]), float(error[1])],
            "calibrated_correction_candidates": candidates,
            "alignment_tolerance_px": float(self.alignment_px),
            "source": "fresh_same_instance_agentview_projection",
        }

    @staticmethod
    def _geometry_priors(evidence: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        geometry = evidence.get("geometry")
        geometry = geometry if isinstance(geometry, dict) else {}
        candidates = geometry.get("calibrated_correction_candidates")
        candidates = candidates if isinstance(candidates, dict) else {}
        return candidates.get("horizontal"), candidates.get("vertical")

    @staticmethod
    def _tracked_geometry_error(
        evidence: dict[str, Any], track: Optional[Any]
    ) -> Optional[tuple[float, float]]:
        """Project a locked instance against the current EEF without a new bbox."""
        if track is None:
            return None
        geometry = evidence.get("geometry")
        geometry = geometry if isinstance(geometry, dict) else {}
        eef_pixel = geometry.get("pixel_xy")
        if not isinstance(eef_pixel, (list, tuple)) or len(eef_pixel) != 2:
            return None
        box = bbox_tuple(getattr(track, "bbox_xyxy", None))
        if box is None:
            return None
        return (
            (box[0] + box[2]) / 2.0 - float(eef_pixel[0]),
            (box[1] + box[3]) / 2.0 - float(eef_pixel[1]),
        )

    def _support_plane_error(
        self,
        evidence: dict[str, Any],
        track: Optional[Any],
        eef_xyz: Optional[tuple[float, float, float]],
    ) -> Optional[tuple[float, float]]:
        """Estimate object XY from its visible support point on the table plane."""
        if track is None or getattr(track, "camera", "") != "agentview" or eef_xyz is None:
            return None
        box = bbox_tuple(getattr(track, "bbox_xyxy", None))
        geometry = evidence.get("geometry")
        geometry = geometry if isinstance(geometry, dict) else {}
        meta = geometry.get("camera_calibration")
        if box is None or not isinstance(meta, dict):
            return None
        try:
            calibration = CameraCalibration(
                name="agentview",
                width=int(meta["width"]),
                height=int(meta["height"]),
                fovy_deg=float(meta["fovy_deg"]),
                position_world=np.asarray(meta["position_world"], dtype=float),
                camera_to_world=np.asarray(meta["camera_to_world"], dtype=float),
                rotation_degrees=int(meta.get("rotation_degrees", 0)),
                flip=str(meta.get("flip", "none")),
            )
            support_world = backproject_pixel_to_plane(
                calibration,
                [(box[0] + box[2]) / 2.0, box[3]],
                self.support_plane_z_m,
            )
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        if support_world is None:
            return None
        return (
            float(support_world[0]) - float(eef_xyz[0]),
            float(support_world[1]) - float(eef_xyz[1]),
        )

    def _wrist_grasp_error(
        self, track: Optional[Any]
    ) -> tuple[Optional[tuple[float, float]], float]:
        if track is None or getattr(track, "camera", "") != "wrist":
            return None, self.wrist_alignment_min_px
        box = bbox_tuple(getattr(track, "bbox_xyxy", None))
        if box is None:
            return None, self.wrist_alignment_min_px
        center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        tolerance = max(
            self.wrist_alignment_min_px,
            (box[2] - box[0]) * self.wrist_alignment_fraction,
        )
        # Only the lateral coordinate is a calibrated gripper-corridor error.
        # The Wrist image's vertical coordinate changes with object height,
        # perspective and the final descent; treating it as planar forward/back
        # motion destroyed an already verified AgentView world-XY alignment.
        # Depth remains owned by support-plane geometry and bounded descent.
        return (
            center[0] - self.wrist_grasp_anchor_px[0],
            0.0,
        ), tolerance

    @staticmethod
    def _route_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
        route = evidence.get("visual_route")
        return route if isinstance(route, dict) else {}

    def _advance_grasp_candidate(
        self,
        *,
        stage: str,
        frame_id: Optional[int],
        health: ObservationHealth,
        evidence: dict[str, Any],
        camera: str,
        eef_xyz: Optional[tuple[float, float, float]],
        previous_eef_xyz: Optional[tuple[float, float, float]],
        previous_residual: Optional[tuple[float, float]],
        observed_residual: Optional[tuple[float, float]],
        gripper_closed: bool,
    ) -> ControlDecision:
        """Run the one-action, multi-frame V2.2 held-object check."""
        candidate = self._grasp_candidate or {}
        frame = int(frame_id if frame_id is not None else -1)
        receipt = self.last_event.get("action_receipt", {}) if isinstance(self.last_event, dict) else {}
        receipt = receipt if isinstance(receipt, dict) else {}
        executed = str(receipt.get("executed_action") or "").upper()
        authorized = str(receipt.get("authorized_action") or "").upper()
        valid_observation = bool(
            stage == "GRASP"
            and health == ObservationHealth.VALID
            and frame > int(candidate.get("frame_id", -1))
            and gripper_closed
            and observed_residual is not None
            and self.belief.target is not None
            and self.belief.target.instance_id == candidate.get("instance_id")
            and self.belief.target.camera == camera
        )

        if self._grasp_diagnostic_lift_authorized and not self._grasp_diagnostic_lift_attempted:
            if valid_observation:
                return ControlDecision(
                    MOVE_UP,
                    "one Qwen-cleared bounded lift to verify visual object/EEF co-motion",
                    None,
                )
            self._grasp_diagnostic_lift_authorized = False
            self._grasp_candidate = None
            return ControlDecision(STOP, "diagnostic lift preconditions became stale", None)

        if self._grasp_diagnostic_lift_attempted and self._grasp_lift_observation is None:
            # The action receipt must confirm the exact bounded diagnostic move.
            if executed != MOVE_UP or authorized != MOVE_UP:
                self._grasp_candidate = None
                self._grasp_diagnostic_lift_attempted = False
                return ControlDecision(STOP, "diagnostic lift was not executed; hold remains unknown", None)
            try:
                dz = float(eef_xyz[2]) - float(previous_eef_xyz[2])
            except (TypeError, ValueError, IndexError):
                dz = 0.0
            baseline = previous_residual
            track = self.belief.target
            if track is None:
                baseline = None
            dims = (
                np.asarray(track.bbox_xyxy, dtype=float)
                if track is not None
                else np.zeros(4, dtype=float)
            )
            diagonal = float(np.hypot(dims[2] - dims[0], dims[3] - dims[1])) if track is not None else 0.0
            tolerance_px = max(3.0, 0.12 * diagonal)
            residual_change = (
                float(np.linalg.norm(np.asarray(observed_residual, dtype=float) - np.asarray(baseline, dtype=float)))
                if observed_residual is not None and baseline is not None
                else math.inf
            )
            if not valid_observation or dz < self.grasp_diagnostic_min_lift_m or residual_change > tolerance_px:
                self._grasp_candidate = None
                self._grasp_diagnostic_lift_attempted = False
                return ControlDecision(STOP, "lift effect did not verify a stable object/EEF relation", None)
            self._grasp_lift_observation = {
                "frame_id": frame,
                "eef_xyz": tuple(float(value) for value in eef_xyz[:3]),
                "residual": tuple(float(value) for value in observed_residual),
                "tolerance_px": tolerance_px,
                "lift_delta_z_m": dz,
            }
            return ControlDecision(STOP, "hold candidate needs one fresh post-lift stability observation", None)

        if self._grasp_lift_observation is not None:
            prior = self._grasp_lift_observation
            eef_delta = math.inf
            if eef_xyz is not None:
                eef_delta = float(np.linalg.norm(
                    np.asarray(eef_xyz, dtype=float)[:3] - np.asarray(prior["eef_xyz"], dtype=float)
                ))
            residual_change = (
                float(np.linalg.norm(np.asarray(observed_residual, dtype=float) - np.asarray(prior["residual"], dtype=float)))
                if observed_residual is not None
                else math.inf
            )
            stable = bool(
                valid_observation
                and executed == STOP
                and frame > int(prior["frame_id"])
                and eef_delta <= self.grasp_diagnostic_min_lift_m
                and residual_change <= float(prior["tolerance_px"])
            )
            if stable:
                self.belief.held = EvidenceValue(
                    True,
                    TruthValue.TRUE,
                    "fresh_post_lift_object_eef_comotion_and_stability",
                    0.9,
                    frame,
                )
                self._grasp_hold_confirmed = True
                self.belief.grasp_epoch += 1
                self.belief.recovery = None
                self.recovery_steps = 0
                self._grasp_candidate = None
                self._grasp_diagnostic_lift_attempted = False
                self._grasp_lift_observation = None
                return ControlDecision(STOP, "held state verified by post-lift temporal evidence", None)
            self.belief.held = EvidenceValue(
                None, TruthValue.UNKNOWN, "post_lift_stability_unverified", 0.0, frame_id
            )
            self._grasp_candidate = None
            self._grasp_diagnostic_lift_attempted = False
            self._grasp_lift_observation = None
            return ControlDecision(STOP, "post-lift stability did not confirm holding", None)

        return ControlDecision(STOP, "grasp candidate awaits its single diagnostic lift", None)

    @staticmethod
    def _frame_id(evidence: dict[str, Any]) -> Optional[int]:
        try:
            value = evidence.get("frame_id")
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _option_for_stage(
        self, stage: str, evidence: Optional[dict[str, Any]] = None
    ) -> OptionName:
        stage = str(stage or "").upper()
        if self.placement_v22_enabled and stage == "TRANSPORT":
            route = self._route_evidence(evidence or {})
            placement = route.get("placement_belief")
            relation = str(placement.get("relation", "UNKNOWN")).upper() if isinstance(placement, dict) else "UNKNOWN"
            if self.belief.seated.truth == TruthValue.TRUE:
                return OptionName.OPEN_GRIPPER
            if relation in {PlacementRelation.ABOVE_ALIGNED.value, PlacementRelation.DESCENDING_CLEAR.value}:
                return OptionName.DESCEND_TO_SEAT
            if relation == PlacementRelation.SEATED_HELD.value:
                return OptionName.VERIFY_SEATED
            return OptionName.ALIGN_OPENING
        if self.belief.recovery is not None and stage in {"APPROACH", "GRASP"}:
            if (
                self.belief.recovery.failure.code
                in {FailureCode.NO_PROGRESS, FailureCode.OSCILLATION}
                and self.recovery_steps >= 1
            ):
                # NO_PROGRESS recovery is one stopped re-observation followed
                # by one bounded alternative probe in the original option.  It
                # must not turn into a 60-step duplicate approach controller.
                return self.belief.recovery.resume_option
            return OptionName.RELOCALIZE
        if stage == "APPROACH":
            error = self._geometry_error(evidence or {})
            if error is not None and max(abs(error[0]), abs(error[1])) > 2.0 * self.alignment_px:
                return OptionName.MOVE_TO_HOVER
            return OptionName.ALIGN_PREGRASP
        return STAGE_OPTION.get(stage, OptionName.LOCATE_TARGET)

    def _budget_for(self, option: OptionName) -> int:
        # In V2.2 ALIGN_OPENING/DESCEND_TO_SEAT/VERIFY_SEATED are one
        # transactional placement option.  Check this before the legacy
        # pixel-align branch; otherwise ALIGN_OPENING is silently capped at
        # the old 40-step pregrasp budget during a long metric transfer.
        if self.placement_v22_enabled and option in {
            OptionName.ALIGN_OPENING,
            OptionName.DESCEND_TO_SEAT,
            OptionName.VERIFY_SEATED,
        }:
            return self.limits.transfer
        if option in {OptionName.LOCATE_TARGET, OptionName.LOCATE_DESTINATION}:
            return self.limits.locate
        if option in {OptionName.MOVE_TO_HOVER, OptionName.LIFT_CLEAR}:
            return self.limits.hover
        if option in {OptionName.ALIGN_PREGRASP, OptionName.ALIGN_OPENING}:
            return self.limits.align
        if option == OptionName.TRANSFER:
            return self.limits.transfer
        if option == OptionName.RELOCALIZE:
            return self.limits.recovery
        return self.limits.commit

    def _failure(
        self, code: FailureCode, option: OptionName, frame_id: Optional[int], reason: str
    ) -> FailureEvent:
        return FailureEvent(
            code=code,
            option=option,
            frame_id=frame_id,
            reason=reason,
            action_history=tuple(item["action"] for item in self.controller.snapshot()["history"]),
        )

    def _begin_recovery(self, failure: FailureEvent, resume: OptionName) -> None:
        key = f"{failure.option.value}:{failure.code.value}"
        attempt = self.recovery_attempt_counts.get(key, 0) + 1
        self.recovery_attempt_counts[key] = attempt
        target_id = self.belief.target.instance_id if self.belief.target else None
        self.belief.recovery = RecoveryContext(
            failure=failure,
            target_instance_id=target_id,
            resume_option=resume,
            attempt=attempt,
            started_frame=failure.frame_id,
        )
        self.belief.route_epoch += 1
        self.recovery_steps = 0
        self.controller.start_recovery(resume.value)

    def _track_for_stage(
        self,
        *,
        stage: str,
        evidence: dict[str, Any],
        observation_image: Optional[np.ndarray],
        camera: str,
        frame_id: int,
    ) -> AssociationResult:
        stage = stage.upper()
        camera_key = "wrist" if str(camera).lower() == "wrist" else "agentview"
        if stage in {"MOVE", "PLACE", "TRANSPORT"}:
            tracker = self.destination_trackers[camera_key]
            destination = tracker.update(
                evidence=evidence,
                image=observation_image,
                frame_id=frame_id,
                semantic_label=str(evidence.get("target") or "destination"),
            )
            if destination.track is not None and self.belief.destination is not None:
                destination.track.instance_id = self.belief.destination.instance_id
            if destination.health == ObservationHealth.VALID:
                self.belief.destination = destination.track
            held = evidence.get("held_object")
            if isinstance(held, dict) and held.get("bbox_xyxy") is not None:
                held_evidence = dict(held)
                held_evidence.setdefault("visible", True)
                held_tracker = self.target_trackers[camera_key]
                held_tracker.update(
                    evidence=held_evidence,
                    image=observation_image,
                    frame_id=frame_id,
                    semantic_label=(
                        self.belief.target.semantic_label
                        if self.belief.target
                        else "held target"
                    ),
                )
                self.belief.target = held_tracker.track
            return destination
        tracker = self.target_trackers[camera_key]
        target = tracker.update(
            evidence=evidence,
            image=observation_image,
            frame_id=frame_id,
            semantic_label=str(evidence.get("target") or "target"),
        )
        if target.track is not None and self.belief.target is not None:
            target.track.instance_id = self.belief.target.instance_id
        if target.health == ObservationHealth.VALID:
            self.belief.target = target.track
        return target

    def associate_secondary_target_view(
        self,
        *,
        evidence: dict[str, Any],
        image: Optional[np.ndarray],
        camera: str,
        frame_id: int,
        commit: bool = False,
    ) -> dict[str, Any]:
        """Associate a secondary GRASP view against its own existing pixel track.

        A secondary camera may refresh the same target's identity and bbox, but
        it cannot make the primary observation healthy or compare coordinates
        across cameras.  A fresh pre-existing same-camera track is mandatory;
        this method never seeds identity from a category-only candidate list.
        """
        camera_key = str(camera or "").lower()
        if camera_key not in {"agentview", "wrist"} or image is None:
            return {"health": "UNAVAILABLE", "reason": "invalid_secondary_camera_or_image"}
        if str(evidence.get("stage") or "GRASP").upper() != "GRASP":
            return {"health": "UNAVAILABLE", "reason": "secondary_identity_refresh_is_grasp_only"}
        target = self.belief.target
        tracker = self.target_trackers[camera_key]
        existing = tracker.track
        if target is None or existing is None:
            return {"health": "UNKNOWN", "reason": "no_existing_target_identity"}
        if (
            str(existing.instance_id) != str(target.instance_id)
            or str(existing.camera).lower() != camera_key
        ):
            return {"health": "UNKNOWN", "reason": "secondary_track_identity_mismatch"}

        probe = EntityTracker(
            role=tracker.role,
            camera=tracker.camera,
            motion_gate_scale=tracker.motion_gate_scale,
            motion_gate_min_px=tracker.motion_gate_min_px,
            ambiguity_margin=tracker.ambiguity_margin,
            max_missed_frames=tracker.max_missed_frames,
        )
        probe.track = copy.deepcopy(existing)
        probe._decision_lock_frames = int(tracker._decision_lock_frames)
        result = probe.update(
            evidence=evidence,
            image=image,
            frame_id=int(frame_id),
            semantic_label=target.semantic_label,
        )
        associated = result.track
        eligible = bool(
            result.health == ObservationHealth.VALID
            and associated is not None
            and str(associated.instance_id) == str(target.instance_id)
            and str(associated.camera).lower() == camera_key
            and int(associated.last_confirmed_frame) == int(frame_id)
            and int(associated.missed_frames) == 0
            and (
                float(getattr(associated, "association_confidence", 0.0)) >= 0.90
                or bool(result.decision_lock_active)
            )
        )
        summary = {
            "health": "VALID" if eligible else result.health.value,
            "reason": result.reason if eligible else f"secondary_association_rejected:{result.reason}",
            "camera": camera_key,
            "frame_id": int(frame_id),
            "instance_id": str(associated.instance_id) if associated is not None else None,
            "bbox_xyxy": list(associated.bbox_xyxy) if eligible and associated is not None else None,
            "association_confidence": (
                float(associated.association_confidence)
                if eligible and associated is not None
                else None
            ),
            "candidate_count": len(result.candidates),
            "decision_lock_active": bool(result.decision_lock_active),
            "committed": bool(eligible and commit),
        }
        if eligible and commit and associated is not None:
            tracker.track = associated
            tracker._decision_lock_frames = probe._decision_lock_frames
            self.belief.target = associated
        return summary

    def _placement_from_route(self, route_evidence: dict[str, Any]) -> PlacementBelief:
        """Convert route evidence into the shared placement contract."""
        raw = route_evidence.get("placement_belief")
        raw = raw if isinstance(raw, dict) else {}
        try:
            relation = PlacementRelation(str(raw.get("relation", "UNKNOWN")).upper())
        except ValueError:
            relation = PlacementRelation.UNKNOWN
        if raw.get("conflicts") or raw.get("fresh") is False:
            relation = PlacementRelation.UNKNOWN
        residual = raw.get("eef_residual_world")
        residual_tuple = (
            tuple(float(value) for value in residual[:3])
            if isinstance(residual, (list, tuple)) and len(residual) >= 3
            else None
        )
        uncertainty = raw.get("uncertainty_m")
        try:
            uncertainty_tuple = (
                tuple(float(value) for value in uncertainty[:3])
                if isinstance(uncertainty, (list, tuple)) and len(uncertainty) >= 3
                and all(value is not None and math.isfinite(float(value)) for value in uncertainty[:3])
                else None
            )
        except (TypeError, ValueError):
            uncertainty_tuple = None
        candidate = PlacementCandidate(
            candidate_id=f"placement-{raw.get('grasp_epoch', 'unknown')}",
            eef_pose_world=None,
            source="visual_route_rim_plane",
            reachable=True,
            containment_margin_m=(
                float(raw["containment_margin_m"])
                if raw.get("containment_margin_m") is not None
                else None
            ),
            uncertainty_m=uncertainty_tuple,
        )
        belief = PlacementBelief(
            relation=relation,
            frame_id=raw.get("frame_id"),
            grasp_epoch=raw.get("grasp_epoch"),
            selected_candidate=candidate,
            opening_plane_origin_world=(
                tuple(float(value) for value in raw["opening_xy_world"][:2])
                + (float(raw.get("rim_plane_z_m")),)
                if isinstance(raw.get("opening_xy_world"), (list, tuple))
                and len(raw.get("opening_xy_world")) >= 2
                and raw.get("rim_plane_z_m") is not None
                else None
            ),
            opening_free_space_polygon_world=tuple(
                tuple(float(value) for value in point[:2])
                for point in raw.get("opening_free_space_polygon_world", ())
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ),
            object_footprint_world=tuple(
                tuple(float(value) for value in point[:2])
                for point in raw.get("object_footprint_world", ())
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ),
            eef_residual_world=residual_tuple,
            rim_clearance_m=(
                float(raw["rim_clearance_m"])
                if raw.get("rim_clearance_m") is not None
                else None
            ),
            containment_margin_m=candidate.containment_margin_m,
            uncertainty_m=uncertainty_tuple,
            evidence_sources=tuple(str(value) for value in raw.get("evidence_sources", ())),
            conflicts=tuple(str(value) for value in raw.get("conflicts", ())),
            fresh=bool(raw.get("fresh", False)),
            diagnostics={
                "route": route_evidence.get("route"),
                "route_id": (route_evidence.get("route") or {}).get("route_id") if isinstance(route_evidence.get("route"), dict) else None,
            },
        )
        offset = raw.get("object_to_gripper_xyz")
        if isinstance(offset, (list, tuple)) and len(offset) >= 3:
            belief = PlacementBelief(
                **{
                    **belief.__dict__,
                    "object_to_gripper_xyz": tuple(float(value) for value in offset[:3]),
                }
            )
        return self.placement_harness.update(belief)

    def _lock_placement_grasp_geometry(
        self, *, eef_xyz: Optional[tuple[float, float, float]]
    ) -> None:
        """Lock object geometry once a held identity is already verified.

        The grasp verifier can commit on the action frame immediately before
        TRANSPORT, so ``previous_held`` may already be TRUE when this method
        first sees a valid spatial belief.  Locking only on a FALSE->TRUE
        transition silently loses the grasp offset and makes the route assume
        a centered grasp.  This method is idempotent and never replaces an
        existing grasp-epoch lock with later noisy observations.
        """

        spatial = self.belief.spatial
        if spatial is None or spatial.health != SpatialHealth.VALID:
            return
        if spatial.conflicting_sources:
            return
        if (
            self.placement_object_to_gripper_xyz is None
            and spatial.fused_relative_xyz is not None
        ):
            self.placement_object_to_gripper_xyz = tuple(
                float(value) for value in spatial.fused_relative_xyz[:3]
            )
        if self.placement_object_points_gripper or eef_xyz is None:
            return
        providers = spatial.diagnostics.get("providers", [])
        if not isinstance(providers, list):
            return
        eef = np.asarray(eef_xyz, dtype=float).reshape(-1)
        if eef.size < 3 or not np.all(np.isfinite(eef[:3])):
            return
        points: list[tuple[float, float, float]] = []
        for provider in providers:
            if not isinstance(provider, dict):
                continue
            raw_points = provider.get("target_points_world")
            if not isinstance(raw_points, (list, tuple)):
                continue
            for raw_point in raw_points[:2048]:
                try:
                    point = np.asarray(raw_point, dtype=float).reshape(-1)
                except (TypeError, ValueError):
                    continue
                if point.size >= 3 and np.all(np.isfinite(point[:3])):
                    points.append(tuple(float(value) for value in (point[:3] - eef[:3])))
        self.placement_object_points_gripper = tuple(points)

    def _control_from_route(self, route_evidence: dict[str, Any]) -> Optional[ControlDecision]:
        progress = route_evidence.get("progress")
        progress = progress if isinstance(progress, dict) else {}
        if self.placement_v22_enabled:
            raw_placement = route_evidence.get("placement_belief")
            # Route progress changes every observation even when the geometry
            # provider has not refreshed its mask.  Recompile from the current
            # shared evidence on every frame; caching only by provider frame
            # would freeze a CLEARANCE residual and can suppress a valid
            # signed atom after the robot has moved.
            if isinstance(raw_placement, dict):
                self.placement_belief = self._placement_from_route(route_evidence)
            placement = self.placement_belief
            if self.belief.seated.truth == TruthValue.TRUE:
                if self._placement_release_ready():
                    return ControlDecision("RELEASE", "Qwen hypothesis plus fresh stable support evidence", None)
                self.belief.seated = EvidenceValue(None, TruthValue.UNKNOWN, "release_gate_lost_fresh_support", 0.0, self.belief.frame_id)
                return ControlDecision(STOP, "fresh stable support gate no longer holds", None)
            if (
                placement.relation == PlacementRelation.RIM_CONTACT
                and not self._pending_contact_supported
            ):
                return ControlDecision(
                    STOP,
                    "stall or overlap needs semantic review before any rim-clear motion",
                    None,
                )
            route_plan = route_evidence.get("route")
            route_plan = route_plan if isinstance(route_plan, dict) else {}
            if placement.fresh:
                self._placement_unknown_probe_key = None
                self._placement_unknown_probe_used = False
            else:
                probe_key = (
                    route_plan.get("route_id"),
                    placement.grasp_epoch,
                )
                if probe_key != self._placement_unknown_probe_key:
                    self._placement_unknown_probe_key = probe_key
                    self._placement_unknown_probe_used = False
                # The route is stale, so its signed residual must not move the
                # robot.  One upward atom is the embodiment-generic probe that
                # increases payload/rim clearance and requests a fresh mask;
                # after that, an unchanged provider failure remains STOP.
                if (
                    not self._placement_unknown_probe_used
                    and self.belief.held.truth == TruthValue.TRUE
                    and self.belief.eef_xyz.value is not None
                    and len(self.belief.eef_xyz.value) >= 3
                    and str(route_plan.get("active_leg", "")).upper()
                    in {"CLEARANCE", "TRANSFER", "PRE_DESCENT", "DESCENT"}
                ):
                    self._placement_unknown_probe_used = True
                    return ControlDecision(
                        MOVE_UP,
                        "stale placement evidence; one reversible high-clearance probe requests fresh geometry",
                        None,
                    )
            # A semantic verifier may report ABOVE_* while the route harness
            # is at its contact boundary.  Consume that report once, but let
            # the metric route choose the signed action.  With no horizontal
            # residual, the only remaining geometric correction is a bounded
            # descent; on the following fresh frame normal RIM_CONTACT logic
            # can clear the rim if the descent had no effect.
            resume = self._placement_verifier_relation
            if resume in {
                PlacementRelation.ABOVE_UNALIGNED,
                PlacementRelation.ABOVE_ALIGNED,
            }:
                if not self._placement_verifier_action_used:
                    self._placement_verifier_action_used = True
                    if resume == PlacementRelation.ABOVE_ALIGNED:
                        return ControlDecision(
                            MOVE_DOWN,
                            "verifier reports above/aligned; resume bounded descent from metric route",
                            None,
                        )
                    candidates = progress.get("route_direction_candidates")
                    if isinstance(candidates, list) and candidates:
                        decision = self.placement_harness.compile_action(
                            relation=PlacementRelation.ABOVE_UNALIGNED,
                            residual_world=placement.eef_residual_world,
                            route_candidates=candidates,
                        )
                        if decision.action_token != STOP:
                            return decision
                    residual = placement.eef_residual_world
                    try:
                        horizontal_residual = float(np.linalg.norm(np.asarray(residual, dtype=float)[:2]))
                    except (TypeError, ValueError, IndexError):
                        horizontal_residual = float("inf")
                    if horizontal_residual <= self.placement_harness.action_step_m:
                        return ControlDecision(
                            MOVE_DOWN,
                            "verifier reports above; metric horizontal residual is within one action",
                            None,
                        )
                    return ControlDecision(
                        STOP,
                        "verifier reports unaligned but no signed metric correction is available",
                        None,
                    )
                # The one verifier-directed atom has already been tested.  A
                # fresh route observation now owns the decision again.
                self._placement_verifier_relation = None
                self._placement_verifier_action_used = False
            return self.placement_harness.compile_action(
                relation=placement.relation,
                residual_world=placement.eef_residual_world,
                route_candidates=progress.get("route_direction_candidates"),
            )
        # Typed transport recovery: the route plugin intentionally withholds
        # lateral candidates while the payload is in a contact/stall phase.
        # In that phase the only reversible, embodiment-generic action is to
        # lift clear while maintaining the verified hold.  Returning STOP here
        # caused a healthy held payload to remain frozen forever with
        # ``active_leg=RECOVER_CLEAR``.
        if str(progress.get("active_leg", "")).upper() == "RECOVER_CLEAR":
            return ControlDecision(MOVE_UP, "typed route recovery: lift clear before replanning", None)
        active_leg = str(progress.get("active_leg", "")).upper()
        intent = route_evidence.get("intent")
        intent_name = str(intent.get("intent", "")).upper() if isinstance(intent, dict) else ""
        if active_leg in {"PRE_DESCENT", "DESCENT"} and intent_name == "READY_TO_RELEASE":
            # Qwen's release intent is still only a request for the independent
            # placement verifier.  DONE lets the runner obtain a fresh visual
            # seating verdict; it never opens the gripper by itself.
            return ControlDecision(DONE, "semantic route intent requests placement verification", None)
        if active_leg in {"PRE_DESCENT", "DESCENT"} and intent_name in {
            "ALIGN",
            "DESCEND",
        }:
            # Metric XY convergence has already ended lateral transfer.  The
            # semantic route intent now authorizes one vertical atom; the
            # placement verifier remains responsible for deciding whether the
            # payload is seated before any release.
            return ControlDecision(MOVE_DOWN, "semantic route intent requests bounded descent", None)
        candidates = progress.get("route_direction_candidates")
        if not isinstance(candidates, list) or not candidates:
            return None
        valid = [item for item in candidates if isinstance(item, dict) and item.get("token")]
        if not valid:
            return None
        valid.sort(key=lambda item: float(item.get("cosine", 0.0) or 0.0), reverse=True)
        best = valid[0]
        if float(best.get("cosine", 0.0) or 0.0) <= 0.0:
            return None
        return ControlDecision(
            str(best["token"]).upper(),
            "minimum calibrated 3D route residual",
            None,
        )

    @staticmethod
    def _spatial_from_evidence(evidence: dict[str, Any]) -> Optional[dict[str, Any]]:
        value = evidence.get("spatial_belief")
        return value if isinstance(value, dict) else None

    @staticmethod
    def _spatial_prompt_payload(value: Optional[SpatialBelief]) -> Optional[dict[str, Any]]:
        """Return the bounded geometry contract sent to Qwen.

        Provider diagnostics can contain masked point clouds and calibration
        details.  Those are useful in the event log, but putting them in every
        multimodal prompt quickly exhausts a small VLM's context window.  The
        semantic agent needs the fused relation, uncertainty, source agreement,
        and freshness—not the raw provider payload.
        """
        if value is None:
            return None
        return {
            "health": value.health.value,
            "instance_id": value.instance_id,
            "frame_id": value.frame_id,
            "fused_relative_xyz": list(value.fused_relative_xyz)
            if value.fused_relative_xyz is not None
            else None,
            "relations": [item.value for item in value.relations],
            "uncertainty": list(value.uncertainty) if value.uncertainty is not None else None,
            "agreeing_sources": list(value.agreeing_sources),
            "conflicting_sources": list(value.conflicting_sources),
        }

    @staticmethod
    def _spatial_object(value: Optional[dict[str, Any]]) -> Optional[SpatialBelief]:
        if not isinstance(value, dict):
            return None
        try:
            relations = tuple(SpatialRelation(str(item).upper()) for item in value.get("relations", (SpatialRelation.UNKNOWN.value,)))
            health = SpatialHealth(str(value.get("health", SpatialHealth.UNKNOWN.value)).upper())
        except ValueError:
            return None
        xyz = value.get("fused_relative_xyz")
        unc = value.get("uncertainty")
        return SpatialBelief(
            fused_relative_xyz=tuple(float(x) for x in xyz[:3]) if isinstance(xyz, (list, tuple)) and len(xyz) >= 3 else None,
            relations=relations,
            uncertainty=tuple(float(x) for x in unc[:3]) if isinstance(unc, (list, tuple)) and len(unc) >= 3 else None,
            agreeing_sources=tuple(str(x) for x in value.get("agreeing_sources", ())),
            conflicting_sources=tuple(str(x) for x in value.get("conflicting_sources", ())),
            health=health,
            frame_id=value.get("frame_id"),
            instance_id=value.get("instance_id"),
            diagnostics=dict(value.get("diagnostics", {})) if isinstance(value.get("diagnostics"), dict) else {},
        )

    def _compile_semantic_pregrasp_raw(
        self, semantic: str, *, evidence: dict[str, Any], eef_xyz: Optional[tuple[float, float, float]]
    ) -> ControlDecision:
        """Compile a bounded semantic choice into one embodiment action.

        Qwen never supplies a direction token in this mode.  Direction comes
        from the fused spatial belief or, for a reversible probe, the explicit
        embodiment calibration in the runtime config.
        """
        semantic = str(semantic or "UNKNOWN").strip().upper()
        spatial = self._spatial_from_evidence(evidence) or {}
        relations = {str(item).upper() for item in spatial.get("relations", ())}
        health = str(spatial.get("health", SpatialHealth.UNKNOWN.value)).upper()

        def correction_from_belief() -> ControlDecision:
            """Compile the highest-confidence signed residual into one atom."""
            if SpatialRelation.FRONT.value in relations:
                return ControlDecision(
                    self.axis_actions["depth_positive"],
                    "grasp choice contradicted by target FRONT of gripper; correct depth first",
                    None,
                )
            if SpatialRelation.BACK.value in relations:
                return ControlDecision(
                    self.axis_actions["depth_negative"],
                    "grasp choice contradicted by target BACK of gripper; correct depth first",
                    None,
                )
            if SpatialRelation.LEFT.value in relations:
                return ControlDecision(
                    self.axis_actions["lateral_positive"],
                    "grasp choice contradicted by lateral residual; correct lateral alignment first",
                    None,
                )
            if SpatialRelation.RIGHT.value in relations:
                return ControlDecision(
                    self.axis_actions["lateral_negative"],
                    "grasp choice contradicted by lateral residual; correct lateral alignment first",
                    None,
                )
            if SpatialRelation.ABOVE.value in relations:
                return ControlDecision(
                    self.axis_actions["height_positive"],
                    "grasp choice contradicted by target ABOVE gripper; correct height first",
                    None,
                )
            if SpatialRelation.BELOW.value in relations:
                return ControlDecision(
                    self.axis_actions["height_negative"],
                    "grasp choice contradicted by target BELOW gripper; correct height first",
                    None,
                )
            return ControlDecision(STOP, "spatial evidence has no signed correction", None)

        if semantic == SemanticPregraspAction.UNKNOWN.value:
            return ControlDecision(STOP, "Qwen abstained or spatial evidence is unknown", None)
        if semantic == SemanticPregraspAction.VISUAL_ALIGN.value:
            alignment = evidence.get("visual_alignment")
            if not isinstance(alignment, dict) or not bool(alignment.get("valid", False)):
                return ControlDecision(
                    STOP,
                    "VISUAL_ALIGN requires a fresh same-instance AgentView projection",
                    None,
                )
            try:
                error = tuple(
                    float(value)
                    for value in alignment.get("target_minus_eef_px", ())
                )
                candidates = alignment.get("calibrated_correction_candidates")
                current_frame = self.belief.frame_id
                same_target = (
                    self.belief.target is not None
                    and str(alignment.get("instance_id") or "")
                    == str(self.belief.target.instance_id)
                )
                fresh = (
                    current_frame is not None
                    and int(alignment.get("frame_id", -1)) == int(current_frame)
                    and self.belief.target is not None
                    and self.belief.target.last_confirmed_frame == int(current_frame)
                )
                tolerance = float(
                    alignment.get("alignment_tolerance_px", self.alignment_px)
                )
            except (TypeError, ValueError, OverflowError):
                return ControlDecision(STOP, "invalid visual alignment evidence", None)
            if (
                len(error) != 2
                or not all(math.isfinite(value) for value in error)
                or not same_target
                or not fresh
                or str(alignment.get("camera") or "").lower() != "agentview"
                or not isinstance(candidates, dict)
                or max(abs(error[0]), abs(error[1])) <= tolerance
            ):
                return ControlDecision(
                    STOP,
                    "VISUAL_ALIGN evidence is stale, aligned, or not grounded to the current target",
                    None,
                )
            axis = 0 if abs(error[0]) >= abs(error[1]) else 1
            action_key = "horizontal" if axis == 0 else "vertical"
            action = str(candidates.get(action_key) or "").upper()
            if action not in {MOVE_LEFT, MOVE_RIGHT, MOVE_FWD, MOVE_BACK}:
                return ControlDecision(
                    STOP,
                    "VISUAL_ALIGN has no calibrated action for the largest residual axis",
                    None,
                )
            return ControlDecision(
                action,
                "one bounded calibrated AgentView correction; obtain a new frame before deciding again",
                None,
            )
        if semantic == SemanticPregraspAction.GRASP.value:
            inside = SpatialRelation.INSIDE_ENVELOPE.value in relations
            ready = health == SpatialHealth.VALID.value and inside
            if self.require_spatial_ready_for_grasp and not ready:
                if health != SpatialHealth.VALID.value:
                    return ControlDecision(STOP, "GRASP blocked until fresh spatial envelope evidence", None)
                return correction_from_belief()
            if health == SpatialHealth.AMBIGUOUS.value:
                return ControlDecision(STOP, "GRASP blocked by conflicting spatial providers", None)
            return ControlDecision("GRASP", "Qwen semantic grasp choice passed runtime gate", None)
        if semantic == SemanticPregraspAction.SELECT_GRASP.value:
            # The first V2.1 provider is deliberately shadow-only and does not
            # yet expose numbered 6-DoF candidates.  Qwen's SELECT_GRASP is
            # therefore the semantic selection of the single envelope-aware
            # candidate compiled by this runtime.  It still passes through the
            # same fresh-spatial gate as an explicit GRASP token; it is never a
            # free-form action or a success assertion.
            if health != SpatialHealth.VALID.value:
                return ControlDecision(STOP, "candidate selection requires fresh spatial evidence", None)
            if SpatialRelation.INSIDE_ENVELOPE.value not in relations:
                return correction_from_belief()
            return ControlDecision("GRASP", "Qwen selected the runtime's validated grasp candidate", None)
        if semantic == SemanticPregraspAction.PROBE_DEPTH.value:
            return ControlDecision(self.axis_actions["probe_depth"], "reversible depth probe requested", None)
        if semantic == SemanticPregraspAction.CORRECT_DEPTH.value:
            if SpatialRelation.FRONT.value in relations:
                action = self.axis_actions["depth_positive"]
            elif SpatialRelation.BACK.value in relations:
                action = self.axis_actions["depth_negative"]
            else:
                # ``CORRECT_DEPTH`` is not permission to guess. A reversible
                # probe has its own semantic token so one abstention cannot
                # silently turn into an unbounded sequence of probes.
                return ControlDecision(
                    STOP,
                    "depth correction requires signed spatial evidence; request PROBE_DEPTH",
                    None,
                )
            return ControlDecision(action, "compiled signed depth relation", None)
        if semantic == SemanticPregraspAction.CORRECT_LATERAL.value:
            if SpatialRelation.LEFT.value in relations:
                action = self.axis_actions["lateral_positive"]
            elif SpatialRelation.RIGHT.value in relations:
                action = self.axis_actions["lateral_negative"]
            else:
                return ControlDecision(STOP, "lateral correction requested without a signed spatial belief", None)
            return ControlDecision(action, "compiled signed lateral relation", None)
        if semantic == SemanticPregraspAction.CORRECT_HEIGHT.value:
            if eef_xyz is not None and self.approach_min_height_m is not None and eef_xyz[2] <= self.approach_min_height_m:
                return ControlDecision(STOP, "height safety floor reached; remaining error is planar", None)
            if SpatialRelation.ABOVE.value in relations:
                return ControlDecision(self.axis_actions["height_negative"], "compiled signed height relation", None)
            if SpatialRelation.BELOW.value in relations:
                return ControlDecision(self.axis_actions["height_positive"], "compiled signed height relation", None)
            return ControlDecision(STOP, "height correction requested without a signed spatial belief", None)
        return ControlDecision(STOP, f"unsupported semantic pregrasp action {semantic}", None)

    def _compile_semantic_pregrasp(
        self, semantic: str, *, evidence: dict[str, Any], eef_xyz: Optional[tuple[float, float, float]]
    ) -> ControlDecision:
        """Compile a semantic choice without reopening a measured failed action.

        Semantic options used to bypass the primitive-action watchdog: Qwen
        could keep asking for ``PROBE_DEPTH`` after the configured probe
        direction had repeatedly moved the target the wrong way.  Keep the
        semantic interface, but adapt a reversible probe to the opposite
        direction after a measured wrong-way effect.  Other contradicted
        actions fail closed until fresh geometry supports a different choice.
        """
        semantic = str(semantic or "UNKNOWN").strip().upper()
        if semantic == SemanticPregraspAction.GRASP.value and not self.require_spatial_ready_for_grasp:
            target = self.belief.target
            frame_id = self.belief.frame_id
            eef = self.belief.eef_xyz
            if (
                self.belief.observation_health != ObservationHealth.VALID
                or target is None
                or target.camera.lower() != "wrist"
                or target.missed_frames != 0
                or frame_id is None
                or target.last_confirmed_frame != int(frame_id)
            ):
                return ControlDecision(
                    STOP,
                    "visual GRASP requires a fresh, unambiguous Wrist target identity",
                    None,
                )
            if eef_xyz is None or eef.truth != TruthValue.TRUE or eef.frame_id != frame_id:
                return ControlDecision(
                    STOP,
                    "visual GRASP requires fresh robot pose evidence",
                    None,
                )
            if self.approach_min_height_m is not None and eef_xyz[2] < self.approach_min_height_m:
                return ControlDecision(STOP, "visual GRASP is below the configured approach safety band", None)
            if self.approach_max_height_m is not None and eef_xyz[2] > self.approach_max_height_m:
                return ControlDecision(STOP, "visual GRASP is above the configured approach safety band", None)
        decision = self._compile_semantic_pregrasp_raw(
            semantic, evidence=evidence, eef_xyz=eef_xyz
        )
        contradicted = set(self.controller.contradicted_actions("QWEN_PREGRASP"))

        if (
            semantic == SemanticPregraspAction.VISUAL_ALIGN.value
            and decision.action_token in contradicted
        ):
            alignment = evidence.get("visual_alignment")
            alignment = alignment if isinstance(alignment, dict) else {}
            try:
                error = tuple(
                    float(value)
                    for value in alignment.get("target_minus_eef_px", ())
                )
                candidates = alignment.get("calibrated_correction_candidates")
                tolerance = float(
                    alignment.get("alignment_tolerance_px", self.alignment_px)
                )
            except (TypeError, ValueError, OverflowError):
                error, candidates, tolerance = (), None, self.alignment_px
            if len(error) == 2 and isinstance(candidates, dict):
                axes = sorted(
                    ((abs(error[0]), "horizontal"), (abs(error[1]), "vertical")),
                    reverse=True,
                )
                alternate = next(
                    (
                        str(candidates.get(axis) or "").upper()
                        for magnitude, axis in axes
                        if magnitude > tolerance
                        and str(candidates.get(axis) or "").upper()
                        in {MOVE_LEFT, MOVE_RIGHT, MOVE_FWD, MOVE_BACK}
                        and str(candidates.get(axis) or "").upper()
                        not in contradicted
                    ),
                    None,
                )
                if alternate is not None:
                    decision = ControlDecision(
                        alternate,
                        "largest calibrated visual axis was contradicted; use the next fresh uncontradicted axis once",
                        None,
                    )

        if semantic == SemanticPregraspAction.PROBE_DEPTH.value:
            nominal = self.axis_actions["probe_depth"]
            fallback = OPPOSITE.get(nominal)
            transition = self.controller.last_transition
            prior_action = str(transition.get("action") or "").upper()
            prior_status = str(transition.get("status") or "").upper()

            # One observed counterproductive probe is enough to reverse this
            # explicitly reversible, one-step option.  If the reverse probe
            # also fails to improve the observation, stop and ask for new
            # evidence instead of alternating directions indefinitely.
            if prior_status in {"WRONG_DIRECTION", "NO_PROGRESS"}:
                if prior_action == nominal and fallback is not None:
                    if fallback in contradicted:
                        return ControlDecision(
                            STOP,
                            "both configured and reverse depth probes are contradicted; refresh evidence or replan",
                            None,
                        )
                    return ControlDecision(
                        fallback,
                        "previous depth probe did not improve the observed residual; testing the opposite direction once",
                        None,
                    )
                if prior_action == fallback:
                    return ControlDecision(
                        STOP,
                        "the opposite depth probe also failed to improve the observation; request fresh evidence or replan",
                        None,
                    )

            nominal_direction, nominal_count = self.controller.effect_direction_streaks.get(
                ("QWEN_PREGRASP", nominal), (0, 0)
            )
            if nominal_direction < 0 and nominal_count >= 1:
                if fallback is None or fallback in contradicted:
                    return ControlDecision(
                        STOP,
                        "configured depth probe was observed moving the wrong way and no uncontradicted reverse remains",
                        None,
                    )
                return ControlDecision(
                    fallback,
                    "configured depth probe has a measured wrong-way effect; keep using the observed reverse direction",
                    None,
                )

            if decision.action_token in contradicted:
                if fallback is not None and fallback not in contradicted:
                    return ControlDecision(
                        fallback,
                        "configured depth direction was empirically contradicted; using the uncontradicted reverse probe",
                        None,
                    )
                return ControlDecision(
                    STOP,
                    "depth probe directions are empirically contradicted; refresh evidence or replan",
                    None,
                )

        if decision.action_token in contradicted:
            return ControlDecision(
                STOP,
                f"{decision.action_token} was empirically contradicted; refresh evidence or choose another feasible option",
                None,
            )
        return decision

    def _available_semantic_pregrasp_actions(
        self, *, evidence: dict[str, Any], eef_xyz: Optional[tuple[float, float, float]]
    ) -> tuple[str, ...]:
        """Expose only semantic choices that currently compile to a safe action."""
        available: list[str] = []
        for semantic in SEMANTIC_PREGRASP_ACTIONS:
            if semantic == SemanticPregraspAction.UNKNOWN.value:
                available.append(semantic)
                continue
            compiled = self._compile_semantic_pregrasp(
                semantic, evidence=evidence, eef_xyz=eef_xyz
            )
            if compiled.action_token != STOP:
                available.append(semantic)
        return tuple(available)

    def _choose_action(
        self,
        *,
        option: OptionName,
        stage: str,
        evidence: dict[str, Any],
        previous_action: Optional[str],
        eef_xyz: Optional[tuple[float, float, float]],
        residual_override: Optional[tuple[float, float]] = None,
        world_xy_error: Optional[tuple[float, float]] = None,
        observation_camera: str = "agentview",
        eef_z_stalled: bool = False,
    ) -> ControlDecision:
        route = self._route_evidence(evidence)
        if self.placement_v22_enabled and stage == "TRANSPORT":
            route_decision = self._control_from_route(route)
            if route_decision is not None:
                return route_decision
            return ControlDecision(STOP, "placement route evidence unavailable", None)
        if option == OptionName.TRANSFER:
            route_decision = self._control_from_route(route)
            if route_decision is not None:
                return route_decision
            return ControlDecision(STOP, "route residual unavailable", None, "NO_PROGRESS")

        if option == OptionName.OPEN_GRIPPER:
            if self.belief.seated.truth == TruthValue.TRUE:
                return ControlDecision("RELEASE", "placement verified before release", None)
            return ControlDecision(STOP, "release blocked until placement is verified", None)

        if option == OptionName.LIFT_CLEAR:
            if self.belief.held.truth != TruthValue.TRUE:
                return ControlDecision(STOP, "hold is not verified", None)
            if eef_xyz is None:
                return ControlDecision(STOP, "EEF height unavailable", None)
            return (
                ControlDecision(MOVE_UP, "lift toward generic clearance height", None)
                if float(eef_xyz[2]) < self.lift_clear_height_m
                else ControlDecision(DONE, "clearance height reached", None)
            )

        alignment = evidence.get("held_object_alignment")
        if option == OptionName.ALIGN_OPENING and isinstance(alignment, dict):
            if self.belief.seated.truth == TruthValue.TRUE:
                return ControlDecision(DONE, "placement support already verified", None)
            if bool(alignment.get("rim_contact_risk", False)):
                return ControlDecision(MOVE_UP, "clear receptacle rim before realignment", None)
            value = alignment.get("destination_minus_held_center_px")
            if isinstance(value, (list, tuple)) and len(value) == 2:
                error = (float(value[0]), float(value[1]))
                candidates = alignment.get("correction_candidates")
                candidates = candidates if isinstance(candidates, dict) else {}
                return self.controller.decide(
                    context=option.value,
                    error=error,
                    horizontal_prior=candidates.get("horizontal"),
                    depth_prior=candidates.get("vertical"),
                    previous_action=previous_action,
                    tolerance_px=float(alignment.get("alignment_threshold_px", self.alignment_px)),
                )

        wrist_error, wrist_tolerance = self._wrist_grasp_error(self.belief.target)
        if stage == "GRASP" and observation_camera == "wrist" and wrist_error is not None:
            decision = self.controller.decide(
                context=f"{option.value}:wrist_px",
                error=wrist_error,
                # Eye-in-hand motion moves the image in the opposite horizontal
                # direction; these signs are an embodiment/camera convention,
                # not an object-specific rule.
                horizontal_prior=(
                    "MV_LEFT" if wrist_error[0] < 0.0 else "MV_RIGHT"
                ),
                depth_prior=None,
                previous_action=previous_action,
                tolerance_px=wrist_tolerance,
            )
        elif stage in {"APPROACH", "GRASP"} and world_xy_error is not None:
            # Controller axis 0 is left/right and axis 1 is forward/back.
            # World residual is stored as (x, y), so reorder to (y, x).
            world_error_mm = (
                float(world_xy_error[1]) * 1000.0,
                float(world_xy_error[0]) * 1000.0,
            )
            decision = self.controller.decide(
                context=f"{option.value}:world_xy_mm",
                error=world_error_mm,
                horizontal_prior=(
                    "MV_LEFT" if world_xy_error[1] > 0.0 else "MV_RIGHT"
                ),
                depth_prior=(
                    "MV_FWD" if world_xy_error[0] > 0.0 else "MV_BACK"
                ),
                previous_action=previous_action,
                tolerance_px=self.world_alignment_tolerance_m * 1000.0,
            )
        else:
            error = residual_override or self._geometry_error(evidence)
            if error is None:
                return ControlDecision(STOP, "fresh spatial residual unavailable", None)
            horizontal, depth = self._geometry_priors(evidence)
            decision = self.controller.decide(
                context=option.value,
                error=error,
                horizontal_prior=horizontal,
                depth_prior=depth,
                previous_action=previous_action,
                tolerance_px=self.alignment_px,
            )
        if stage in {"APPROACH", "GRASP"} and decision.action_token == DONE:
            if eef_xyz is None:
                return ControlDecision(STOP, "aligned but EEF height is unavailable", None)
            height = float(eef_xyz[2])
            if stage == "APPROACH":
                if (
                    option == OptionName.RELOCALIZE
                    and eef_z_stalled
                    and previous_action == MOVE_DOWN
                ):
                    # A typed recovery may be approaching an object beside a
                    # rim/obstacle.  If the executed downward command produced
                    # no measurable EEF descent, continuing to request
                    # MV_DOWN is not recovery; it is an unbounded collision
                    # retry.  Finish relocalization and let the next GRASP
                    # option ask Qwen with fresh Wrist/spatial evidence.
                    return ControlDecision(
                        DONE,
                        "relocalization reached an embodiment descent/contact boundary",
                        None,
                    )
                if (
                    self.approach_max_height_m is not None
                    and height > self.approach_max_height_m
                ):
                    return ControlDecision(
                        MOVE_DOWN,
                        "horizontal residual aligned; enter configured robot pregrasp band",
                        None,
                    )
                if (
                    self.approach_min_height_m is not None
                    and height < self.approach_min_height_m
                ):
                    return ControlDecision(
                        MOVE_UP,
                        "EEF below configured robot pregrasp band",
                        None,
                    )
            elif stage == "GRASP":
                if observation_camera == "wrist" and wrist_error is not None:
                    if (
                        self._wrist_final_descent_count
                        < self.wrist_final_descent_steps
                    ):
                        self._wrist_final_descent_count += 1
                        return ControlDecision(
                            MOVE_DOWN,
                            "wrist lateral corridor aligned; execute bounded final grasp approach",
                            None,
                        )
                    return ControlDecision(
                        "GRASP",
                        "wrist lateral corridor remained aligned after final grasp approach",
                        None,
                    )
                if self.approach_min_height_m is None or (
                    height > self.approach_min_height_m + self.height_tolerance_m
                ):
                    return ControlDecision(
                        MOVE_DOWN,
                        "world XY aligned; descend until wrist target acquisition",
                        None,
                    )
                return ControlDecision(
                    STOP,
                    "wrist target unavailable at configured robot safety floor",
                    None,
                    "NO_PROGRESS",
                )
        if stage == "PLACE" and decision.action_token == DONE:
            return ControlDecision(MOVE_DOWN, "opening aligned; descend one verified step", None)
        return decision

    def _placement_release_ready(self) -> bool:
        """Independent physical gate for a V2.2 Qwen seating hypothesis."""
        placement = self.placement_belief
        if not placement.fresh or placement.conflicts or self.belief.held.truth != TruthValue.TRUE:
            return False
        if placement.grasp_epoch != self.belief.grasp_epoch or placement.frame_id != self.belief.frame_id:
            return False
        if placement.containment_margin_m is None or placement.rim_clearance_m is None or placement.uncertainty_m is None:
            return False
        if self.belief.held.frame_id != self.belief.frame_id:
            return False
        sigma = float(np.linalg.norm(np.asarray(placement.uncertainty_m)))
        if placement.containment_margin_m < 2.0 * sigma + 0.5 * self.placement_harness.action_step_m:
            return False
        if placement.rim_clearance_m < -sigma:
            return False
        return bool(
            self._placement_contact_verified
            and self._placement_support_stability_count >= 2
            and self._placement_support_confirm_frame == self.belief.frame_id
            and self._placement_seated_hypothesis is not None
        )

    def _update_placement_support_evidence(
        self, *, frame_id: int, previous_action: Optional[str], eef_z_stalled: bool,
        progress: dict[str, Any], route_evidence: dict[str, Any],
    ) -> None:
        """Collect a contact candidate plus a separate stable, fresh frame."""
        placement_raw = route_evidence.get("placement_belief")
        if not isinstance(placement_raw, dict):
            self._placement_support_candidate = None
            self._placement_support_stability_count = 0
            self._placement_support_confirm_frame = None
            self._placement_seated_hypothesis = None
            self._placement_contact_verified = False
            return
        placement = self.placement_belief
        target_id = self.belief.target.instance_id if self.belief.target is not None else None
        route_plan = route_evidence.get("route") if isinstance(route_evidence.get("route"), dict) else {}
        route_id = route_plan.get("route_id")
        sigma = (
            float(np.linalg.norm(np.asarray(placement.uncertainty_m, dtype=float)))
            if placement.uncertainty_m is not None else math.inf
        )
        margin = placement.containment_margin_m
        clearance = placement.rim_clearance_m
        geometrically_supported = bool(
            placement.fresh and not placement.conflicts
            and self.belief.held.truth == TruthValue.TRUE
            and self.belief.held.frame_id == self.belief.frame_id
            and placement.grasp_epoch == self.belief.grasp_epoch
            and margin is not None and clearance is not None and math.isfinite(sigma)
            and margin >= 2.0 * sigma + 0.5 * self.placement_harness.action_step_m
            and clearance >= -sigma
        )
        receipt = self.last_event.get("action_receipt", {}) if isinstance(self.last_event, dict) else {}
        receipt = receipt if isinstance(receipt, dict) else {}
        actually_executed_descent = bool(
            str(receipt.get("executed_action") or "").upper() == MOVE_DOWN
            and str(receipt.get("authorized_action") or "").upper() == MOVE_DOWN
        )
        observed_contact = bool(
            progress.get("contact_or_stall") and eef_z_stalled
            and actually_executed_descent
        )
        candidate = self._placement_support_candidate
        if observed_contact and geometrically_supported:
            self._placement_contact_verified = True
            self._placement_support_candidate = {
                "frame_id": int(frame_id), "route_epoch": int(self.belief.route_epoch),
                "grasp_epoch": int(self.belief.grasp_epoch), "instance_id": target_id,
                "route_id": route_id, "eef_xyz": self.belief.eef_xyz.value,
                "containment_margin_m": float(margin), "rim_clearance_m": float(clearance),
            }
            self._placement_support_stability_count = 1
            self._placement_support_confirm_frame = int(frame_id)
            candidate = self._placement_support_candidate
        elif candidate is not None:
            same_transaction = bool(
                int(candidate.get("route_epoch", -1)) == int(self.belief.route_epoch)
                and int(candidate.get("grasp_epoch", -1)) == int(self.belief.grasp_epoch)
                and candidate.get("instance_id") == target_id
                and candidate.get("route_id") == route_id
            )
            after_no_action = bool(
                previous_action == STOP and int(frame_id) > int(candidate.get("frame_id", -1))
            )
            eef_delta = None
            if candidate.get("eef_xyz") is not None and self.belief.eef_xyz.value is not None:
                eef_delta = float(np.linalg.norm(
                    np.asarray(self.belief.eef_xyz.value, dtype=float)
                    - np.asarray(candidate["eef_xyz"], dtype=float)
                ))
            stable_geometry = bool(
                geometrically_supported and same_transaction and after_no_action
                and eef_delta is not None
                and eef_delta <= max(0.25 * self.placement_harness.action_step_m, 0.002)
                and abs(float(margin) - float(candidate["containment_margin_m"])) <= max(2.0 * sigma, 0.002)
                and abs(float(clearance) - float(candidate["rim_clearance_m"])) <= max(2.0 * sigma, 0.002)
            )
            if stable_geometry:
                self._placement_support_stability_count = 2
                self._placement_support_confirm_frame = int(frame_id)
            elif not same_transaction or not geometrically_supported:
                self._placement_support_candidate = None
                self._placement_support_stability_count = 0
                self._placement_support_confirm_frame = None
                self._placement_seated_hypothesis = None
                self._placement_contact_verified = False
        if self._placement_seated_hypothesis is not None:
            hypothesis = self._placement_seated_hypothesis
            if (
                self._placement_support_stability_count >= 2
                and self._placement_support_confirm_frame == int(frame_id)
                and int(hypothesis.get("route_epoch", -1)) == int(self.belief.route_epoch)
                and int(hypothesis.get("grasp_epoch", -1)) == int(self.belief.grasp_epoch)
                and hypothesis.get("instance_id") == target_id
            ):
                self.belief.seated = EvidenceValue(
                    True, TruthValue.TRUE, "qwen_hypothesis_plus_fresh_stable_support",
                    0.8, self.belief.frame_id,
                )
                if isinstance(self.last_event, dict):
                    self.last_event["placement_support_stability"] = {
                        "count": self._placement_support_stability_count,
                        "candidate_frame": self._placement_support_candidate.get("frame_id") if self._placement_support_candidate else None,
                        "confirmed_frame": int(frame_id), "route_id": route_id,
                    }

    @staticmethod
    def _placement_action_effect(
        *, action: Optional[str], motion_delta: Optional[tuple[float, float, float]],
        old_placement: Any, new_placement: Any,
    ) -> dict[str, Any]:
        action_name = str(action or "").upper()
        if action_name not in {MOVE_LEFT, MOVE_RIGHT, MOVE_FWD, MOVE_BACK, MOVE_UP, MOVE_DOWN}:
            return {"status": "NO_MOTION_ACTION", "action": action_name or None}
        if motion_delta is None:
            return {"status": "UNKNOWN", "action": action_name, "reason": "EEF delta unavailable"}
        delta = np.asarray(motion_delta, dtype=float)
        moved = float(np.linalg.norm(delta))
        if moved < 0.001:
            return {"status": "NO_EEF_MOTION", "action": action_name, "eef_delta_m": delta.tolist()}
        def residual(value: Any) -> Optional[np.ndarray]:
            if not isinstance(value, dict):
                return None
            raw = value.get("eef_residual_world")
            try:
                vector = np.asarray(raw, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                return None
            return vector[:3] if vector.size >= 3 and np.all(np.isfinite(vector[:3])) else None
        old_residual, new_residual = residual(old_placement), residual(new_placement)
        if old_residual is not None and new_residual is not None:
            before = float(np.linalg.norm(old_residual))
            after = float(np.linalg.norm(new_residual))
            progress_m = before - after
            status = "IMPROVING" if progress_m > 0.001 else ("WRONG_DIRECTION" if progress_m < -0.001 else "NO_ROUTE_PROGRESS")
            return {"status": status, "action": action_name, "eef_delta_m": delta.tolist(), "route_residual_before_m": before, "route_residual_after_m": after, "progress_m": progress_m}
        expected_sign = -1.0 if action_name in {MOVE_DOWN, MOVE_LEFT, MOVE_FWD} else 1.0
        axis = 2 if action_name in {MOVE_UP, MOVE_DOWN} else None
        if axis is not None:
            projected = expected_sign * float(delta[axis])
            status = "MOTION_OBSERVED" if projected > 0.001 else "WRONG_DIRECTION"
        else:
            status = "MOTION_OBSERVED"
        return {"status": status, "action": action_name, "eef_delta_m": delta.tolist(), "expected_effect": "reduce current route residual"}

    def observe_frame(
        self,
        *,
        stage: str,
        evidence: dict[str, Any] | None,
        previous_action: Optional[str],
        agentview: Optional[np.ndarray] = None,
        wrist: Optional[np.ndarray] = None,
        image_refs: Optional[dict[str, str]] = None,
        eef_xyz: Optional[tuple[float, float, float]] = None,
        gripper_closed: bool = False,
        gripper_width_m: Optional[float] = None,
    ) -> dict[str, Any]:
        evidence = evidence if isinstance(evidence, dict) else {}
        self._pending_evidence_frame_ids = ()
        stage_name = str(stage or "").upper()
        stage_changed = stage_name != self.last_stage
        if stage_changed:
            if stage_name == "GRASP" or self.last_stage == "GRASP":
                self._wrist_final_descent_count = 0
            self.last_stage = stage_name
        frame_id = self._frame_id(evidence)
        frame_number = int(frame_id if frame_id is not None else self.episode_steps)
        belief_before = copy.deepcopy(self.belief)
        prior_receipt = self.last_event.get("action_receipt", {}) if isinstance(self.last_event, dict) else {}
        prior_receipt = prior_receipt if isinstance(prior_receipt, dict) else {}
        just_closed = bool(
            str(prior_receipt.get("executed_action") or "").upper() == "GRASP"
            and str(prior_receipt.get("authorized_action") or "").upper() == "GRASP"
        )
        self.episode_steps += 1

        previous_eef_xyz = belief_before.eef_xyz.value
        eef_z_stalled = bool(
            previous_action == MOVE_DOWN
            and previous_eef_xyz is not None
            and eef_xyz is not None
            and float(previous_eef_xyz[2]) - float(eef_xyz[2])
            <= self.recovery_descent_stall_tolerance_m
        )

        option = self._option_for_stage(stage_name, evidence)
        if option != self.belief.current_option:
            self.option_steps = 0
            self.controller.reset_transient()
        self.option_steps += 1
        if self.belief.recovery is not None:
            self.recovery_steps += 1
        self.belief.current_option = option
        self.belief.frame_id = frame_id
        if eef_xyz is not None:
            self.belief.eef_xyz = EvidenceValue(
                tuple(float(value) for value in eef_xyz),
                TruthValue.TRUE,
                "robot_proprioception",
                1.0,
                frame_id,
            )

        base_health, health_reason = self._health(evidence)
        observation_camera = (
            "wrist" if str(evidence.get("camera") or "").lower() == "wrist" else "agentview"
        )
        observation_image = wrist if observation_camera == "wrist" else agentview
        association = self._track_for_stage(
            stage=stage_name,
            evidence=evidence,
            observation_image=observation_image,
            camera=observation_camera,
            frame_id=frame_number,
        )
        secondary_association: dict[str, Any] | None = None
        secondary_view = evidence.get("secondary_view")
        if stage_name == "GRASP" and isinstance(secondary_view, dict):
            secondary_camera = str(secondary_view.get("camera") or "").lower()
            secondary_image = wrist if secondary_camera == "wrist" else agentview
            if secondary_camera in {"agentview", "wrist"} and secondary_camera != observation_camera:
                secondary_payload = dict(secondary_view)
                secondary_payload["stage"] = stage_name
                secondary_association = self.associate_secondary_target_view(
                    evidence=secondary_payload,
                    image=secondary_image,
                    camera=secondary_camera,
                    frame_id=frame_number,
                    commit=True,
                )
                secondary_view["instance_association"] = secondary_association
                if secondary_association.get("health") == "VALID":
                    # Keep this as camera-local identity evidence.  It must not
                    # upgrade health for the (possibly occluded) primary view.
                    secondary_view["bbox_xyxy"] = secondary_association.get("bbox_xyxy")
                    secondary_view["visible"] = True
        health = base_health
        if base_health == ObservationHealth.VALID:
            health = association.health
            health_reason = association.reason
        if (
            self.temporal_identity_lock_on_detector_ties
            and base_health == ObservationHealth.AMBIGUOUS
            and association.health == ObservationHealth.VALID
            and association.reason == "instance_associated"
            and association.track is not None
            and self.belief.target is not None
            and association.track.instance_id == self.belief.target.instance_id
            and str(association.track.camera).lower() == observation_camera
            and int(association.track.last_confirmed_frame) == frame_number
            and int(association.track.missed_frames) == 0
            and (
                float(getattr(association.track, "association_confidence", 0.0)) >= 0.90
                or bool(getattr(association, "decision_lock_active", False))
            )
        ):
            # A fresh, high-confidence temporal match can resolve a detector's
            # per-frame score tie after the semantic resolver has already
            # established the target identity.  The detector's other candidate
            # remains in the event log; no match, stale match, camera handoff,
            # or tracker-level ambiguity still fails closed and asks Qwen. For
            # the tracker's short post-selection lock, honor its own motion-gated
            # continuation even if the normalized appearance score is slightly
            # below the general confidence cutoff; otherwise Qwen is asked to
            # reselect immediately and can switch to a different instance.
            health = ObservationHealth.VALID
            health_reason = "temporal_identity_disambiguated_detector_tie"
        if (
            option == OptionName.DESCEND_TO_GRASP
            and observation_camera == "agentview"
            and association.track is not None
            and association.reason == "no_candidate_in_motion_gate"
            and base_health != ObservationHealth.SENSOR_FAULT
        ):
            # The hand is expected to occlude the object during the bounded
            # descent.  Preserve the identity lock instead of escalating a
            # rejected distractor to semantic ambiguity.
            health = ObservationHealth.OCCLUDED
            health_reason = "expected_close_range_occlusion"
        if (
            health == ObservationHealth.AMBIGUOUS
            and self.belief.recovery is not None
            and association.track is not None
            and self.belief.target is not None
            and association.track.instance_id == self.belief.target.instance_id
            # The detector can also mark its top scores ambiguous even when
            # the temporal tracker has a fresh, high-confidence continuation
            # of the original instance.  During typed recovery the temporal
            # identity is the stronger evidence: accepting that continuation
            # avoids asking Qwen to choose between two duplicate masks of the
            # same fallen object.  We still require a non-missed track and a
            # strong association; a missing or out-of-gate target remains an
            # UNKNOWN/STOP condition.
            and association.reason in {
                "association_margin_too_small",
                "instance_associated",
            }
            and association.track.missed_frames == 0
            and float(getattr(association.track, "association_confidence", 0.0)) >= 0.90
        ):
            # Recovery is allowed to retain the original instance lock when
            # the tracker still has a fresh, motion-gated observation of that
            # same track but two nearby candidates have nearly identical
            # association costs.  This is not a detector-score override: a
            # missing/out-of-gate target remains AMBIGUOUS and stops.  The
            # lock prevents a fallen object beside a basket from becoming
            # permanently unrecoverable merely because a distractor overlaps
            # the crop, while preserving the original instance_id/epoch.
            health = ObservationHealth.VALID
            health_reason = "recovery_temporal_identity_lock"
        self.belief.observation_health = health
        spatial_dict = self._spatial_from_evidence(evidence)
        if spatial_dict is not None:
            self.belief.spatial = self._spatial_object(spatial_dict)
        elif self.belief.spatial is not None:
            spatial_frame = self.belief.spatial.frame_id
            spatial_age = (
                None
                if spatial_frame is None
                else max(0, frame_number - int(spatial_frame))
            )
            propagated = False
            if spatial_age is not None and spatial_age <= self.spatial_propagation_max_age_frames and eef_xyz is not None:
                # A static target's world point remains fixed while the EEF
                # moves.  Re-expressing that point against fresh proprioception
                # is safer than reusing the old relative vector, and lets the
                # runtime finish a bounded correction when Wrist detection
                # briefly drops out.  The uncertainty grows with age and the
                # propagated belief cannot outlive the explicit bound above.
                providers = self.belief.spatial.diagnostics.get("providers", {})
                if isinstance(providers, list):
                    world_points: list[np.ndarray] = []
                    for provider in providers:
                        if not isinstance(provider, dict):
                            continue
                        points = provider.get("target_points_world")
                        if not isinstance(points, (list, tuple)):
                            continue
                        for point in points:
                            try:
                                value = np.asarray(point, dtype=float).reshape(3)
                            except (TypeError, ValueError):
                                continue
                            if np.all(np.isfinite(value)):
                                world_points.append(value)
                    if world_points:
                        world = np.median(np.stack(world_points), axis=0)
                        relative = world - np.asarray(eef_xyz, dtype=float)
                        old_unc = np.asarray(
                            self.belief.spatial.uncertainty or (0.01, 0.01, 0.01),
                            dtype=float,
                        ).reshape(3)
                        uncertainty = np.maximum(
                            np.abs(old_unc),
                            0.003 + 0.003 * float(spatial_age),
                        )
                        diagnostics = dict(self.belief.spatial.diagnostics)
                        diagnostics["propagated"] = True
                        diagnostics["propagation_age_frames"] = int(spatial_age)
                        diagnostics["target_world"] = tuple(float(v) for v in world)
                        self.belief.spatial = SpatialBelief(
                            fused_relative_xyz=tuple(float(v) for v in relative),
                            relations=classify_relation(relative, uncertainty),
                            uncertainty=tuple(float(v) for v in uncertainty),
                            agreeing_sources=self.belief.spatial.agreeing_sources,
                            conflicting_sources=self.belief.spatial.conflicting_sources,
                            health=SpatialHealth.VALID,
                            frame_id=frame_number,
                            instance_id=self.belief.spatial.instance_id,
                            diagnostics=diagnostics,
                        )
                        propagated = True
            if not propagated and (
                spatial_age is None
                or spatial_age > self.spatial_propagation_max_age_frames
            ):
                # Do not let AgentView frames silently reuse a Wrist/MoGe
                # residual after its EEF reference has gone stale.  The next
                # action must come from fresh visual geometry or a new Wrist
                # spatial observation.
                self.belief.spatial = None
        if self.belief.target is not None:
            from .types import VisualMemoryEntry
            receipt = self.last_event.get("action_receipt", {}) if isinstance(self.last_event, dict) else {}
            receipt = receipt if isinstance(receipt, dict) else {}
            route_for_memory = self._route_evidence(evidence)
            route_plan = route_for_memory.get("route") if isinstance(route_for_memory, dict) else {}
            route_plan = route_plan if isinstance(route_plan, dict) else {}
            refs = image_refs if isinstance(image_refs, dict) else {}
            motion_delta = None
            if previous_eef_xyz is not None and eef_xyz is not None:
                motion_delta = tuple(float(now) - float(before) for before, now in zip(previous_eef_xyz[:3], eef_xyz[:3]))
            previous_event = self.last_event if isinstance(self.last_event, dict) else {}
            previous_event_evidence = previous_event.get("evidence") if isinstance(previous_event.get("evidence"), dict) else {}
            previous_placement = previous_event_evidence.get("placement_belief")
            current_placement = route_for_memory.get("placement_belief") if isinstance(route_for_memory, dict) else None
            effect = self._placement_action_effect(
                action=receipt.get("executed_action", previous_action),
                motion_delta=motion_delta,
                old_placement=previous_placement,
                new_placement=current_placement,
            )
            track_summary = evidence.get("held_point_track_summary")
            if isinstance(track_summary, dict):
                effect["visual_point_track"] = {
                    key: track_summary.get(key)
                    for key in (
                        "health", "camera", "query_frame_id", "frame_id",
                        "visible_tracks", "median_displacement_px",
                        "median_track_motion_px", "inference_latency_s",
                    )
                    if key in track_summary
                }
            if effect.get("status") in {"NO_EEF_MOTION", "NO_ROUTE_PROGRESS", "WRONG_DIRECTION"}:
                self._placement_effect_no_progress += 1
            elif effect.get("status") in {"IMPROVING", "MOTION_OBSERVED"}:
                self._placement_effect_no_progress = 0
            expected_effect = (
                "reduce route residual" if receipt.get("executed_action") in {MOVE_LEFT, MOVE_RIGHT, MOVE_FWD, MOVE_BACK}
                else ("observe downward or support response" if receipt.get("executed_action") == MOVE_DOWN else "increase clearance" if receipt.get("executed_action") == MOVE_UP else None)
            )
            prior_memory = self.visual_memory.recent(
                instance_id=self.belief.target.instance_id,
                grasp_epoch=self.belief.grasp_epoch,
                limit=1,
            )
            before_entry = prior_memory[0] if prior_memory else None
            appended_memory = self.visual_memory.append(
                VisualMemoryEntry(
                    instance_id=self.belief.target.instance_id,
                    grasp_epoch=self.belief.grasp_epoch,
                    frame_id=frame_number,
                    episode_id=self.visual_memory.episode_id,
                    before_frame_id=(before_entry.frame_id if before_entry is not None else None),
                    before_agentview_ref=(before_entry.agentview_ref if before_entry is not None else None),
                    before_wrist_ref=(before_entry.wrist_ref if before_entry is not None else None),
                    agentview_ref=refs.get("agentview"),
                    wrist_ref=refs.get("wrist"),
                    requested_action=receipt.get("requested_action"),
                    authorized_action=receipt.get("authorized_action"),
                    executed_action=receipt.get("executed_action", previous_action),
                    motion_delta=motion_delta,
                    depth_summary=(spatial_dict or {}),
                    route_phase=str(route_plan.get("active_leg") or ""),
                    route_epoch=int(self.belief.route_epoch),
                    predicted_effect=tuple(float(value) for value in previous_event.get("predicted_effect", ())) if isinstance(previous_event.get("predicted_effect"), (list, tuple)) else None,
                    observed_effect={**effect, "previous_frame_id": previous_event.get("frame_id"), "current_frame_id": frame_number},
                    effect_status=effect.get("status"),
                    placement_summary={
                        "expected_effect": expected_effect,
                        "route_uncertainty_m": current_placement.get("uncertainty_m") if isinstance(current_placement, dict) else None,
                        "evidence_sources": current_placement.get("evidence_sources") if isinstance(current_placement, dict) else [],
                    },
                    tags=("AFTER_ACTION",) if receipt.get("executed_action") else ("OBSERVATION",),
                )
            )
            effect_status = str(effect.get("status", "UNKNOWN")).upper()
            phase_changed = bool(
                before_entry is not None
                and before_entry.route_phase != appended_memory.route_phase
                and appended_memory.route_phase
            )
            noteworthy_effect = effect_status in {
                "NO_EEF_MOTION", "NO_ROUTE_PROGRESS", "WRONG_DIRECTION", "IMPROVING"
            }
            tags = []
            if stage_changed and stage_name in {"APPROACH", "GRASP", "LIFT", "TRANSPORT", "MOVE", "PLACE", "RELEASE"}:
                tags.append(f"STAGE_ENTRY_{stage_name}")
            if just_closed:
                tags.append("POST_GRASP_CLOSE")
            if phase_changed:
                tags.append("ROUTE_PHASE_CHANGE")
            if noteworthy_effect:
                tags.append(f"ACTION_EFFECT_{effect_status}")
            if health != belief_before.observation_health:
                tags.append(f"OBSERVATION_HEALTH_{health.value}")
            if tags:
                action_key = str(receipt.get("executed_action") or "NONE").upper()
                event_key = (
                    f"{self.belief.grasp_epoch}:{self.belief.route_epoch}:"
                    f"{appended_memory.route_phase}:{action_key}:{'+'.join(tags)}"
                )
                self.visual_memory.mark(
                    instance_id=self.belief.target.instance_id,
                    grasp_epoch=self.belief.grasp_epoch,
                    frame_id=frame_number,
                    tag="+".join(tags),
                    event_key=event_key,
                )

        observed_residual = self._geometry_error(evidence)
        support_plane_error = (
            self._support_plane_error(evidence, association.track, eef_xyz)
            if observation_camera == "agentview"
            else None
        )
        if (
            self.metric_approach_uses_hover_budget
            and stage_name == "APPROACH"
            and health == ObservationHealth.VALID
            and support_plane_error is not None
        ):
            # A detector can abstain on per-frame score ties while the temporal
            # tracker still provides a fresh target bbox. In that case the
            # earlier pixel-only option selection sees no bbox and picks the
            # short fine-alignment budget. Reclassify from the already-validated
            # metric support-plane residual: far approach uses MOVE_TO_HOVER's
            # bounded budget; local alignment retains the shorter ALIGN budget.
            far_axis_error_m = max(abs(float(value)) for value in support_plane_error)
            desired_option = (
                OptionName.MOVE_TO_HOVER
                if far_axis_error_m > 2.0 * self.world_alignment_tolerance_m
                else OptionName.ALIGN_PREGRASP
            )
            if desired_option != option:
                option = desired_option
                self.option_steps = 1
                self.belief.current_option = option
                self.controller.reset_transient()
        if health == ObservationHealth.VALID and observed_residual is not None:
            # Geometry becomes canonical only after the corresponding entity
            # association passes.  Rejected/occluded detector boxes must never
            # overwrite the last verified spatial relation.
            self.belief.alignment_residual = EvidenceValue(
                observed_residual,
                TruthValue.TRUE,
                "verified_entity_geometry",
                float(association.track.association_confidence)
                if association.track is not None
                else 0.0,
                frame_id,
            )
        if health == ObservationHealth.VALID and support_plane_error is not None:
            self.belief.world_xy_residual_m = EvidenceValue(
                support_plane_error,
                TruthValue.TRUE,
                "target_support_point_table_plane",
                float(association.track.association_confidence)
                if association.track is not None
                else 0.0,
                frame_id,
            )

        local_recovery_completed = bool(
            health == ObservationHealth.VALID
            and self.belief.recovery is not None
            and self.belief.recovery.failure.code
            in {FailureCode.NO_PROGRESS, FailureCode.OSCILLATION}
            and option == self.belief.recovery.resume_option
        )
        if local_recovery_completed:
            self.belief.recovery = None
            self.recovery_steps = 0

        route = self._route_evidence(evidence)
        holding = route.get("holding_arbiter")
        holding = holding if isinstance(holding, dict) else {}
        width_empty = bool(
            not self.placement_v22_enabled
            and gripper_width_m is not None
            and float(gripper_width_m) <= self.empty_width_m
        )
        hold_verdict = verify_hold(
            frame_id=frame_id,
            gripper_closed=gripper_closed,
            holding_evidence=holding,
            target_visible=self.belief.target is not None,
            width_clearly_empty=width_empty,
        )
        if (
            self.placement_v22_enabled
            and not self._grasp_hold_confirmed
            and hold_verdict.verdict == Verdict.PASS
        ):
            hold_verdict = verify_hold(
                frame_id=frame_id,
                gripper_closed=gripper_closed,
                holding_evidence=None,
                target_visible=self.belief.target is not None,
                width_clearly_empty=False,
            )
        previous_held = self.belief.held.truth
        if hold_verdict.verdict == Verdict.PASS:
            self.belief.held = EvidenceValue(True, TruthValue.TRUE, hold_verdict.reason, 0.9, frame_id)
            if previous_held != TruthValue.TRUE:
                self.belief.grasp_epoch += 1
                # Lock the metric object-to-EEF transform at the grasp epoch.
                # It is reused for placement target compilation and is never
                # silently replaced by a later noisy frame.
        if self.belief.held.truth == TruthValue.TRUE:
            self._lock_placement_grasp_geometry(eef_xyz=eef_xyz)
        elif hold_verdict.verdict == Verdict.FAIL:
            self.belief.held = EvidenceValue(False, TruthValue.FALSE, hold_verdict.reason, 0.9, frame_id)
        elif gripper_closed and previous_held != TruthValue.TRUE:
            self.belief.held = EvidenceValue(None, TruthValue.UNKNOWN, hold_verdict.reason, 0.0, frame_id)

        placement_raw = route.get("placement_belief")
        if self.placement_v22_enabled and isinstance(placement_raw, dict):
            self.placement_belief = self._placement_from_route(route)

        alignment = evidence.get("held_object_alignment")
        progress = route.get("progress")
        progress = progress if isinstance(progress, dict) else {}
        placement_trigger = None
        if self.placement_v22_enabled and stage_name == "TRANSPORT" and isinstance(placement_raw, dict):
            route_plan = route.get("route") if isinstance(route.get("route"), dict) else {}
            phase = str(route_plan.get("active_leg") or progress.get("active_leg") or "").upper()
            if phase not in {"PRE_DESCENT", "DESCENT"} or previous_action == MOVE_UP:
                self._placement_contact_verified = False
                if previous_action == MOVE_UP or phase not in {"PRE_DESCENT", "DESCENT"}:
                    self._placement_support_candidate = None
                    self._placement_support_stability_count = 0
                    self._placement_support_confirm_frame = None
                    self._placement_seated_hypothesis = None
            if self.placement_belief.grasp_epoch != self.belief.grasp_epoch:
                self._placement_contact_verified = False
                self._placement_support_candidate = None
                self._placement_support_stability_count = 0
                self._placement_support_confirm_frame = None
                self._placement_seated_hypothesis = None
            self._update_placement_support_evidence(
                frame_id=frame_number,
                previous_action=previous_action,
                eef_z_stalled=eef_z_stalled,
                progress=progress,
                route_evidence=route,
            )
            if phase in {"PRE_DESCENT", "DESCENT"} and self._last_placement_phase not in {"PRE_DESCENT", "DESCENT"}:
                placement_trigger = "seating_entry"
            elif bool(progress.get("contact_or_stall")):
                placement_trigger = "contact_confirmed"
            elif bool(progress.get("contact_candidate")):
                placement_trigger = "contact_risk"
            elif self._last_placement_phase == "RECOVER_CLEAR" and phase != "RECOVER_CLEAR":
                placement_trigger = "clear_recovery"
            elif self._placement_support_candidate is not None and previous_action == STOP and self._placement_seated_hypothesis is None:
                placement_trigger = "post_stall_reobserve"
            elif self._placement_effect_no_progress >= 2:
                placement_trigger = "effect_anomaly"
            self._last_placement_phase = phase
            if self.belief.target is not None:
                self.visual_memory.update_latest(
                    instance_id=self.belief.target.instance_id,
                    grasp_epoch=self.belief.grasp_epoch,
                    route_phase=phase,
                    placement_summary={key: placement_raw.get(key) for key in (
                        "frame_id", "rim_clearance_m", "containment_margin_m",
                        "uncertainty_m", "fresh", "conflicts",
                    )},
                )
                if placement_trigger:
                    self.visual_memory.mark(
                        instance_id=self.belief.target.instance_id,
                        grasp_epoch=self.belief.grasp_epoch,
                        frame_id=frame_number,
                        tag=placement_trigger.upper(),
                        event_key=f"{self.belief.route_epoch}:{placement_trigger}",
                    )
        placement_alignment = dict(alignment) if isinstance(alignment, dict) else {}
        if self.placement_v22_enabled and isinstance(placement_raw, dict):
            placement_alignment["relation"] = placement_raw.get("relation", "UNKNOWN")
        seated_verdict = verify_seated(
            frame_id=frame_id,
            alignment=placement_alignment or None,
            holding_verdict=hold_verdict.verdict,
            contact_or_stall=bool(progress.get("contact_or_stall", False)),
        )
        if stage_name in {"PLACE", "RELEASE", "TRANSPORT"} and not self.placement_v22_enabled:
            if seated_verdict.verdict == Verdict.PASS:
                self.belief.seated = EvidenceValue(True, TruthValue.TRUE, seated_verdict.reason, 0.9, frame_id)
            elif seated_verdict.verdict == Verdict.FAIL:
                self.belief.seated = EvidenceValue(False, TruthValue.FALSE, seated_verdict.reason, 0.9, frame_id)

        failure: Optional[FailureEvent] = None
        critical: Optional[CriticalDecisionRequest] = None
        action = ControlDecision(STOP, "runtime waiting for valid evidence", None)
        status = OptionStatus.RUNNING

        if self.episode_steps > self.limits.episode:
            failure = self._failure(
                FailureCode.OPTION_BUDGET_EXCEEDED, option, frame_id, "episode budget exceeded"
            )
        elif health == ObservationHealth.SENSOR_FAULT:
            action = ControlDecision(STOP, "sensor fault: fail closed", None)
            if self.sensor_fault_streak >= self.sensor_fault_limit:
                failure = self._failure(
                    FailureCode.SENSOR_FAULT, option, frame_id, health_reason
                )
        elif self.option_steps > self._budget_for(option):
            # Budgets govern valid, ambiguous, and occluded observations alike;
            # otherwise a stale/occluded stream can hold the robot forever.
            failure = self._failure(
                FailureCode.OPTION_BUDGET_EXCEEDED,
                option,
                frame_id,
                f"{option.value} exceeded {self._budget_for(option)} steps",
            )
            action = ControlDecision(STOP, failure.reason, None)
        elif self.belief.recovery is not None and (
            self.recovery_steps > self.limits.recovery
            or self.belief.recovery.attempt > self.limits.max_recovery_attempts
        ):
            failure = self._failure(
                FailureCode.RECOVERY_BUDGET_EXCEEDED,
                option,
                frame_id,
                "typed recovery budget exceeded",
            )
            action = ControlDecision(STOP, failure.reason, None)
        elif health == ObservationHealth.AMBIGUOUS:
            action = ControlDecision(STOP, "target identity is ambiguous", None)
            failure = self._failure(
                FailureCode.TARGET_AMBIGUOUS, option, frame_id, health_reason
            )
            # Candidate IDs must not reorder just because SAM scores fluctuate.
            # Spatial sorting is local to this one frozen frame; the committed
            # bbox is still re-associated temporally on the next observation.
            candidates = tuple(sorted(association.candidates, key=_candidate_spatial_key))
            critical = CriticalDecisionRequest(
                kind="SELECT_INSTANCE",
                candidate_ids=tuple(f"candidate-{i}" for i, _ in enumerate(candidates)),
                allowed_answers=tuple(f"candidate-{i}" for i, _ in enumerate(candidates)) + ("UNKNOWN",),
                belief=jsonable(self.belief),
                reason=health_reason,
                candidates=candidates,
                camera=observation_camera,
            )
            self._pending_candidates = list(candidates)
            self._pending_image = observation_image
            self._pending_frame_id = frame_id
            self._pending_role = (
                "destination" if stage_name in {"MOVE", "PLACE", "TRANSPORT"} else "target"
            )
            self._pending_tracker = (
                self.destination_trackers[observation_camera]
                if self._pending_role == "destination"
                else self.target_trackers[observation_camera]
            )
            self._pending_critical_kind = "SELECT_INSTANCE"
            self._pending_allowed_answers = critical.allowed_answers
            self._pending_control_error = None
        elif (
            self.placement_v22_enabled
            and stage_name == "GRASP"
            and just_closed
            and not self._grasp_hold_confirmed
            and self._grasp_candidate is None
        ):
            action = ControlDecision(STOP, "fresh temporal grasp verification required", None)
            critical = CriticalDecisionRequest(
                kind="VERIFY_HOLD",
                candidate_ids=(),
                allowed_answers=("YES", "NO", "UNKNOWN"),
                belief=jsonable(self.belief),
                reason=(
                    "compare the current dual view with the immediately pre-close frame; "
                    "a YES is only a candidate and must cite current evidence plus visible clearance"
                ),
                camera=observation_camera,
                evidence_frame_ids=(frame_number - 1, frame_number),
            )
            self._pending_critical_kind = "VERIFY_HOLD"
            self._pending_allowed_answers = critical.allowed_answers
        elif self.placement_v22_enabled and stage_name == "GRASP" and self._grasp_candidate is not None:
            action = self._advance_grasp_candidate(
                stage=stage_name,
                frame_id=frame_id,
                health=health,
                evidence=evidence,
                camera=observation_camera,
                eef_xyz=eef_xyz,
                previous_eef_xyz=previous_eef_xyz,
                previous_residual=belief_before.alignment_residual.value,
                observed_residual=observed_residual,
                gripper_closed=gripper_closed,
            )
        elif self.placement_v22_enabled and stage_name == "GRASP" and self._grasp_hold_confirmed:
            action = ControlDecision(DONE, "V2.2 multi-frame hold verification passed", None)
        elif (
            self.placement_v22_enabled
            and stage_name == "TRANSPORT"
            and isinstance(placement_raw, dict)
            and placement_trigger is not None
            and (self.belief.grasp_epoch, self.belief.route_epoch, placement_trigger)
            not in self._placement_reflection_seen
            and (
                str(placement_raw.get("relation", "UNKNOWN")).upper() == PlacementRelation.RIM_CONTACT.value
                or self.placement_reflection_mode != "off"
            )
            and self.belief.seated.truth != TruthValue.TRUE
            and not (
                self.belief.recovery is not None
                and self.belief.recovery.failure.code == FailureCode.RIM_CONTACT
            )
            and not (
                self._placement_verifier_relation
                in {
                    PlacementRelation.ABOVE_UNALIGNED,
                    PlacementRelation.ABOVE_ALIGNED,
                }
                and not self._placement_verifier_action_used
            )
        ):
            # Freeze one observation while the semantic verifier inspects its
            # bounded temporal context. Provider risk alone cannot authorize lift.
            action = ControlDecision(STOP, "placement keyframe requires semantic verification", None)
            if placement_trigger:
                self._placement_reflection_seen.add((self.belief.grasp_epoch, self.belief.route_epoch, placement_trigger))
            memory_bundle = self.visual_memory.placement_bundle(
                instance_id=self.belief.target.instance_id if self.belief.target else None,
                grasp_epoch=self.belief.grasp_epoch,
                route_epoch=self.belief.route_epoch,
            )
            receipt = self.last_event.get("action_receipt", {}) if isinstance(self.last_event, dict) else {}
            receipt = receipt if isinstance(receipt, dict) else {}
            down_attempt_executed = bool(
                str(receipt.get("executed_action") or "").upper() == MOVE_DOWN
                and str(receipt.get("authorized_action") or "").upper() == MOVE_DOWN
            )
            self._pending_contact_supported = bool(
                down_attempt_executed
                and progress.get("contact_or_stall") and eef_z_stalled
                and self.placement_belief.fresh and not self.placement_belief.conflicts
                and self.belief.held.truth == TruthValue.TRUE
                and self.belief.held.frame_id == self.belief.frame_id
                and self.placement_belief.uncertainty_m is not None
                and self.placement_belief.containment_margin_m is not None
                and self.placement_belief.rim_clearance_m is not None
                and (
                    self.placement_belief.containment_margin_m
                    < 2.0 * float(np.linalg.norm(self.placement_belief.uncertainty_m))
                    + 0.5 * self.placement_harness.action_step_m
                    or self.placement_belief.rim_clearance_m
                    < -float(np.linalg.norm(self.placement_belief.uncertainty_m))
                )
            )
            relation_now = self.placement_belief.relation
            allowed_options = ["REOBSERVE", "CHANGE_VIEW", "HOLD"]
            if (
                not self.placement_belief.fresh
                or self.placement_belief.conflicts
                or self.placement_belief.uncertainty_m is None
            ):
                allowed_options.append("REFRESH_GEOMETRY")
            if self._placement_effect_no_progress >= 2:
                allowed_options.append("REPLAN_PLACEMENT")
            if (
                relation_now in {PlacementRelation.ABOVE_ALIGNED, PlacementRelation.DESCENDING_CLEAR}
                and self.placement_belief.fresh and not self.placement_belief.conflicts
                and not progress.get("contact_or_stall")
            ):
                allowed_options.append("CONTINUE_DESCENT")
            if self._pending_contact_supported:
                allowed_options.append("CLEAR_RIM")
            critical = CriticalDecisionRequest(
                kind="VERIFY_SEATED",
                candidate_ids=(),
                allowed_answers=(
                    PlacementRelation.ABOVE_UNALIGNED.value,
                    PlacementRelation.ABOVE_ALIGNED.value,
                    PlacementRelation.DESCENDING_CLEAR.value,
                    PlacementRelation.SEATED_HELD.value,
                    PlacementRelation.RIM_CONTACT.value,
                    PlacementRelation.LOST.value,
                    "UNKNOWN",
                ),
                belief=jsonable(self.belief),
                reason="classify frozen placement evidence; runtime independently gates contact and release",
                visual_memory_refs=tuple(
                    ref for entry in memory_bundle for ref in
                    (entry.get("agentview_ref"), entry.get("wrist_ref")) if ref
                ),
                visual_memory_bundle=tuple(memory_bundle),
                reflection_trigger=placement_trigger or "contact_confirmed",
                allowed_options=tuple(dict.fromkeys(allowed_options)),
            )
            self._pending_critical_kind = "VERIFY_SEATED"
            self._pending_allowed_answers = critical.allowed_answers
        elif (
            self.semantic_pregrasp_enabled
            and stage_name == "GRASP"
            and health == ObservationHealth.OCCLUDED
            and self.belief.target is not None
            and str(self.belief.target.camera).lower() == "agentview"
            and (
                self.belief.grasp_epoch,
                str(self.belief.target.instance_id),
            ) not in self._grasp_entry_review_seen
            and frame_number - int(self.belief.target.last_confirmed_frame)
            <= self.semantic_evidence_max_age_frames
            and self.belief.eef_xyz.value is not None
            and self.approach_min_height_m is not None
            and self.approach_max_height_m is not None
            and self.approach_min_height_m
            <= float(self.belief.eef_xyz.value[2])
            <= self.approach_max_height_m
        ):
            # At grasp entry the target can disappear from Wrist precisely
            # because the gripper occludes it. Give Thinking one frozen,
            # same-episode memory review, but restrict its choice to a bounded
            # clearance lift or abstention. A fresh visible observation is
            # still required before any close/grasp decision.
            memory_bundle = self.visual_memory.placement_bundle(
                instance_id=self.belief.target.instance_id,
                grasp_epoch=self.belief.grasp_epoch,
                route_epoch=None,
                limit=3,
            )
            self._grasp_entry_review_seen.add(
                (self.belief.grasp_epoch, str(self.belief.target.instance_id))
            )
            has_current_dual_view = any(
                int(item.get("frame_id", -1)) == frame_number
                and bool(item.get("agentview_ref"))
                and bool(item.get("wrist_ref"))
                for item in memory_bundle
                if isinstance(item, dict)
            )
            if has_current_dual_view:
                action = ControlDecision(
                    STOP,
                    "grasp entry is occluded; request a bounded visual-memory review",
                    None,
                )
                reflection_trigger = "grasp_entry"
                reflection_mode = "off"
                reflection_key = (
                    self.belief.grasp_epoch,
                    self.belief.route_epoch,
                    reflection_trigger,
                )
                if (
                    self.pregrasp_reflection_mode == "double"
                    and reflection_key not in self._pregrasp_reflection_seen
                ):
                    reflection_mode = "double"
                    self._pregrasp_reflection_seen.add(reflection_key)
                self._pending_evidence_frame_ids = tuple(
                    int(item["frame_id"])
                    for item in memory_bundle
                    if isinstance(item, dict) and item.get("frame_id") is not None
                )
                critical = CriticalDecisionRequest(
                    kind="PREGRASP_DECISION",
                    candidate_ids=(),
                    allowed_answers=("REOBSERVE", "UNKNOWN"),
                    belief=jsonable(self.belief),
                    spatial_belief=self._spatial_prompt_payload(self.belief.spatial),
                    visual_memory_refs=tuple(
                        ref for item in memory_bundle
                        for ref in (item.get("agentview_ref"), item.get("wrist_ref"))
                        if ref
                    ),
                    visual_memory_bundle=tuple(memory_bundle),
                    evidence_frame_ids=self._pending_evidence_frame_ids,
                    allowed_options=("REOBSERVE",),
                    reflection_trigger=reflection_trigger,
                    reflection_mode=reflection_mode,
                    reason=(
                        "the Wrist target is occluded at grasp entry; inspect the current dual view "
                        "and only current-episode visual memory. Choose REOBSERVE if one safe, "
                        "bounded upward view-clearance action could reveal the target; runtime "
                        "will compile it to a single lift capped by configured clearance height "
                        "and require a fresh dual-view observation. Choose UNKNOWN if no useful "
                        "safe observation action is supported. Do not infer contact or authorize "
                        "GRASP from occlusion."
                    ),
                )
                self._pending_critical_kind = "PREGRASP_DECISION"
                self._pending_allowed_answers = critical.allowed_answers
            else:
                action = ControlDecision(
                    STOP,
                    "grasp entry is occluded but current raw dual-view memory is unavailable",
                    None,
                )
        elif (
            self.semantic_pregrasp_enabled
            and stage_name == "GRASP"
            and health == ObservationHealth.OCCLUDED
            and self.belief.target is not None
            and str(self.belief.target.camera).lower() == "agentview"
            and self.belief.eef_xyz.value is not None
            and self.belief.eef_xyz.truth == TruthValue.TRUE
            and self.belief.eef_xyz.frame_id == frame_id
            and self.approach_min_height_m is not None
            and self.approach_max_height_m is not None
            and self.approach_min_height_m - self.height_tolerance_m
            <= float(self.belief.eef_xyz.value[2])
            <= self._grasp_entry_reobserve_expected_z_m.get(
                (self.belief.grasp_epoch, str(self.belief.target.instance_id)),
                float("-inf"),
            ) + self.height_tolerance_m
            and self.belief.target.last_confirmed_frame is not None
            and frame_number - int(self.belief.target.last_confirmed_frame)
            <= self.semantic_evidence_max_age_frames
            and (
                self.belief.grasp_epoch,
                str(self.belief.target.instance_id),
            ) in self._grasp_entry_reobserve_used
            and (
                self.belief.grasp_epoch,
                str(self.belief.target.instance_id),
            ) not in self._grasp_entry_followup_seen
            and str(prior_receipt.get("authorized_action") or "").upper() == MOVE_UP
            and str(prior_receipt.get("executed_action") or "").upper() == MOVE_UP
        ):
            # The first bounded upward observation ran, but the fresh Wrist
            # view still does not resolve the same live target. Give the model
            # one new-frame ReAct decision so it can use the measured failure
            # and choose one bounded depth probe or abstain. It cannot repeat
            # the lift or issue GRASP from this occluded branch.
            target_key = (
                self.belief.grasp_epoch,
                str(self.belief.target.instance_id),
            )
            self._grasp_entry_followup_seen.add(target_key)
            memory_bundle = self.visual_memory.placement_bundle(
                instance_id=self.belief.target.instance_id,
                grasp_epoch=self.belief.grasp_epoch,
                route_epoch=None,
                limit=3,
            )
            has_current_dual_view = any(
                int(item.get("frame_id", -1)) == frame_number
                and bool(item.get("agentview_ref"))
                and bool(item.get("wrist_ref"))
                for item in memory_bundle
                if isinstance(item, dict)
            )
            if has_current_dual_view:
                self._pending_evidence_frame_ids = tuple(
                    int(item["frame_id"])
                    for item in memory_bundle
                    if isinstance(item, dict) and item.get("frame_id") is not None
                )
                reflection_trigger = "grasp_reobserve_outcome"
                reflection_key = (
                    self.belief.grasp_epoch,
                    self.belief.route_epoch,
                    reflection_trigger,
                )
                reflection_mode = "off"
                if (
                    self.pregrasp_reflection_mode == "double"
                    and reflection_key not in self._pregrasp_reflection_seen
                ):
                    reflection_mode = "double"
                    self._pregrasp_reflection_seen.add(reflection_key)
                critical = CriticalDecisionRequest(
                    kind="PREGRASP_DECISION",
                    candidate_ids=(),
                    allowed_answers=("PROBE_DEPTH", "UNKNOWN"),
                    belief=jsonable(self.belief),
                    spatial_belief=self._spatial_prompt_payload(self.belief.spatial),
                    visual_memory_refs=tuple(
                        ref for item in memory_bundle
                        for ref in (item.get("agentview_ref"), item.get("wrist_ref"))
                        if ref
                    ),
                    visual_memory_bundle=tuple(memory_bundle),
                    evidence_frame_ids=self._pending_evidence_frame_ids,
                    allowed_options=("PROBE_DEPTH",),
                    reflection_trigger=reflection_trigger,
                    reflection_mode=reflection_mode,
                    reason=(
                        "the previously authorized upward REOBSERVE action executed, but its fresh "
                        "Wrist view still has no current target pixels. Compare this frame with the "
                        "recorded action effect and same-episode memory. Choose one bounded PROBE_DEPTH "
                        "only if it is likely to reveal the target-to-gripper relation; otherwise "
                        "choose UNKNOWN. The prior upward action is unavailable, and this event cannot "
                        "authorize GRASP. If the probe does not produce a fresh visible target, stop."
                    ),
                )
                action = ControlDecision(
                    STOP,
                    "bounded REOBSERVE did not resolve occlusion; request one outcome review",
                    None,
                )
                self._pending_critical_kind = "PREGRASP_DECISION"
                self._pending_allowed_answers = critical.allowed_answers
            else:
                action = ControlDecision(
                    STOP,
                    "REOBSERVE outcome has no current dual-view memory; movement withheld",
                    None,
                )
        elif (
            self.semantic_pregrasp_enabled
            and stage_name == "GRASP"
            and health == ObservationHealth.OCCLUDED
        ):
            # Do not keep servoing on an old target residual after the view is
            # lost. The grasp-entry reflection above gets one bounded chance;
            # subsequent occlusion requires a newly visible, fresh observation.
            action = ControlDecision(
                STOP,
                "grasp target remains occluded; stale geometry cannot authorize motion",
                None,
            )
        elif (
            health == ObservationHealth.VALID
            and stage_name == "GRASP"
            and (
                observation_camera == "wrist"
                or (
                    self.semantic_pregrasp_enabled
                    and self.belief.spatial is not None
                    and self.belief.spatial.health == SpatialHealth.VALID
                )
            )
        ):
            # Pixel geometry can establish candidate identity and whether an
            # action had an observable effect, but it cannot universally infer
            # a grasp pose across object shapes and gripper embodiments.  Give
            # Qwen the raw dual views at this bounded semantic decision point.
            visual_alignment = self._current_visual_alignment(evidence)
            if visual_alignment is not None:
                spatial_error = tuple(
                    float(value)
                    for value in visual_alignment["target_minus_eef_px"]
                )
            elif (
                self.controller.last_context == "QWEN_PREGRASP"
                and self.controller.last_error is not None
            ):
                # No fresh, same-camera target projection is available.  Do
                # not compare a Wrist residual with an earlier AgentView
                # residual; preserve the baseline so the action watchdog sees
                # no demonstrated progress instead of inventing one.
                spatial_error = self.controller.last_error
            else:
                spatial_error = observed_residual or (0.0, 0.0)
            transition = self.controller.observe_transition(
                context="QWEN_PREGRASP",
                error=spatial_error,
                previous_action=previous_action,
            )
            prior_failure = (
                self.belief.recovery.failure.reason
                if self.belief.recovery is not None
                else ""
            )
            contradicted = set(
                self.controller.contradicted_actions("QWEN_PREGRASP")
            )
            semantic_evidence = {
                "spatial_belief": self._spatial_prompt_payload(self.belief.spatial),
                "visual_alignment": visual_alignment,
            }
            allowed_pregrasp = (
                self._available_semantic_pregrasp_actions(
                    evidence=semantic_evidence,
                    eef_xyz=self.belief.eef_xyz.value,
                )
                if self.semantic_pregrasp_enabled
                else tuple(action for action in PREGRASP_ACTIONS if action not in contradicted)
            )
            effect_summary = (
                f"last action {transition.get('action')} changed target-minus-EEF by "
                f"{transition.get('observed_delta_px')} px and was "
                f"{transition.get('status', 'UNKNOWN')}"
                if transition.get("action")
                else f"last effect: {transition.get('status', 'UNKNOWN')}"
            )
            pregrasp_memory_bundle = self.visual_memory.placement_bundle(
                instance_id=self.belief.target.instance_id if self.belief.target else None,
                grasp_epoch=self.belief.grasp_epoch,
                route_epoch=(self.belief.route_epoch if self.placement_v22_enabled else None),
                limit=3,
            )
            reflection_trigger = None
            if stage_changed:
                reflection_trigger = "grasp_entry"
            elif transition.get("status") in {
                "NO_EEF_MOTION", "NO_ROUTE_PROGRESS", "WRONG_DIRECTION"
            }:
                reflection_trigger = "action_effect_anomaly"
            elif prior_failure:
                reflection_trigger = "recovery_after_failure"
            reflection_mode = "off"
            if reflection_trigger and self.pregrasp_reflection_mode == "double":
                reflection_key = (
                    self.belief.grasp_epoch,
                    self.belief.route_epoch,
                    reflection_trigger,
                )
                if reflection_key not in self._pregrasp_reflection_seen:
                    reflection_mode = "double"
                    self._pregrasp_reflection_seen.add(reflection_key)
            self._pending_evidence_frame_ids = tuple(
                int(item["frame_id"])
                for item in pregrasp_memory_bundle
                if isinstance(item, dict) and item.get("frame_id") is not None
            )
            critical = CriticalDecisionRequest(
                kind="PREGRASP_DECISION",
                candidate_ids=(),
                allowed_answers=allowed_pregrasp,
                belief=jsonable(self.belief),
                spatial_belief=self._spatial_prompt_payload(self.belief.spatial),
                visual_alignment=visual_alignment,
                visual_memory_refs=tuple(
                    ref for item in pregrasp_memory_bundle
                    for ref in (item.get("agentview_ref"), item.get("wrist_ref")) if ref
                ),
                visual_memory_bundle=tuple(pregrasp_memory_bundle),
                evidence_frame_ids=self._pending_evidence_frame_ids,
                reflection_trigger=reflection_trigger,
                reflection_mode=reflection_mode,
                reason=(
                    "inspect both live views and choose one bounded pregrasp action; "
                    "runtime will re-observe and verify its effect"
                    + (f"; prior failure: {prior_failure}" if prior_failure else "")
                    + f"; {effect_summary}"
                    + (
                        "; empirically contradicted actions removed from ALLOWED: "
                        + ", ".join(sorted(contradicted))
                        if contradicted
                        else ""
                    )
                    + (
                        "; semantic choices are filtered by current evidence and measured action effects; "
                        "a depth probe reverses after a counterproductive result and stops if both directions fail"
                        if self.semantic_pregrasp_enabled
                        else ""
                    )
                    + (
                        "; spatial UNKNOWN does not block a visually supported GRASP, but runtime requires current Wrist identity and in-band robot pose"
                        if self.semantic_pregrasp_enabled and not self.require_spatial_ready_for_grasp
                        else ""
                    )
                ),
                camera="wrist",
            )
            action = ControlDecision(
                STOP,
                "Qwen pregrasp spatial decision required from raw dual views",
                None,
            )
            self._pending_critical_kind = "PREGRASP_DECISION"
            self._pending_allowed_answers = critical.allowed_answers
            self._pending_control_error = spatial_error
        elif (
            health == ObservationHealth.OCCLUDED
            and option != OptionName.DESCEND_TO_GRASP
        ):
            action = ControlDecision(STOP, "target is occluded; movement withheld", None)
        elif previous_held == TruthValue.TRUE and hold_verdict.verdict == Verdict.FAIL:
            failure = self._failure(
                FailureCode.LOST_HOLD, option, frame_id, hold_verdict.reason
            )
            self._begin_recovery(failure, OptionName.LIFT_CLEAR)
            action = ControlDecision("RELEASE", "lost hold; open before visual reacquisition", None)
        elif option == OptionName.LIFT_CLEAR and self.belief.held.truth == TruthValue.UNKNOWN:
            action = ControlDecision(STOP, "semantic hold verification required", None)
            critical = CriticalDecisionRequest(
                kind="VERIFY_HOLD",
                candidate_ids=(),
                allowed_answers=("YES", "NO", "UNKNOWN"),
                belief=jsonable(self.belief),
                reason="geometry and gripper evidence do not prove object occupancy",
            )
            self._pending_critical_kind = "VERIFY_HOLD"
        elif option == OptionName.OPEN_GRIPPER and self.belief.seated.truth != TruthValue.TRUE:
            action = ControlDecision(STOP, "semantic placement verification required", None)
            memory_bundle = self.visual_memory.placement_bundle(
                instance_id=self.belief.target.instance_id if self.belief.target else None,
                grasp_epoch=self.belief.grasp_epoch,
            ) if self.placement_v22_enabled else []
            critical = CriticalDecisionRequest(
                kind="VERIFY_SEATED",
                candidate_ids=(),
                allowed_answers=("YES", "NO", "UNKNOWN"),
                belief=jsonable(self.belief),
                reason="release requires independent visible placement evidence",
                visual_memory_refs=tuple(
                    ref for entry in memory_bundle for ref in
                    (entry.get("agentview_ref"), entry.get("wrist_ref")) if ref
                ),
                visual_memory_bundle=tuple(memory_bundle),
                reflection_trigger="release_check" if self.placement_v22_enabled else None,
            )
            self._pending_critical_kind = "VERIFY_SEATED"
        else:
            committed_residual: Optional[tuple[float, float]] = None
            if (
                health == ObservationHealth.OCCLUDED
                and option == OptionName.DESCEND_TO_GRASP
            ):
                committed_residual = self._tracked_geometry_error(
                    evidence, self.belief.target
                )
                if committed_residual is None and (
                    self.belief.alignment_residual.truth == TruthValue.TRUE
                    and self.belief.alignment_residual.value is not None
                    and frame_id is not None
                    and self.belief.alignment_residual.frame_id is not None
                    and frame_id - self.belief.alignment_residual.frame_id <= 1
                ):
                    committed_residual = tuple(self.belief.alignment_residual.value)
            action = self._choose_action(
                option=option,
                stage=stage_name,
                evidence=evidence,
                previous_action=previous_action,
                eef_xyz=eef_xyz,
                residual_override=committed_residual,
                world_xy_error=support_plane_error,
                observation_camera=observation_camera,
                eef_z_stalled=eef_z_stalled,
            )
            if action.failure == "NO_PROGRESS":
                failure = self._failure(FailureCode.NO_PROGRESS, option, frame_id, action.reason)
            elif action.failure == "OSCILLATION":
                failure = self._failure(FailureCode.OSCILLATION, option, frame_id, action.reason)
            if failure is not None and (
                self.belief.recovery is None
                or failure.code in {FailureCode.NO_PROGRESS, FailureCode.OSCILLATION}
            ):
                self._begin_recovery(failure, option)
            if action.action_token == DONE:
                status = OptionStatus.SUCCEEDED
                if option == OptionName.RELOCALIZE:
                    self.belief.recovery = None
                    self.recovery_steps = 0

        if critical is not None:
            status = OptionStatus.NEED_DECISION
        elif failure is not None and failure.code in {
            FailureCode.SENSOR_FAULT,
            FailureCode.OPTION_BUDGET_EXCEEDED,
            FailureCode.RECOVERY_BUDGET_EXCEEDED,
        }:
            status = OptionStatus.FAILED

        takeover = bool(self.enabled and self.mode == "active")
        before_residual = belief_before.alignment_residual.value
        after_residual = self.belief.alignment_residual.value
        event = {
            "runtime_version": self.runtime_version,
            "frame_id": frame_id,
            "stage": stage_name,
            "observation_health": health.value,
            "health_reason": health_reason,
            "belief_before": jsonable(belief_before),
            "entity_tracks": {
                "target": jsonable(self.belief.target),
                "destination": jsonable(self.belief.destination),
            },
            "evidence": {
                "camera": evidence.get("camera"),
                "source": evidence.get("source"),
                "bbox_xyxy": evidence.get("bbox_xyxy"),
                "confidence": evidence.get("confidence"),
                "secondary_view": (
                    {
                        key: secondary_view.get(key)
                        for key in (
                            "camera", "bbox_xyxy", "visible", "source",
                            "instance_association", "geometry",
                        )
                        if key in secondary_view
                    }
                    if isinstance(secondary_view, dict)
                    else None
                ),
                "secondary_target_mask_audit": evidence.get(
                    "secondary_target_mask_audit"
                ),
                "geometry": evidence.get("geometry"),
                "visual_route": evidence.get("visual_route"),
                "spatial_belief": evidence.get("spatial_belief"),
                "secondary_spatial_belief": evidence.get(
                    "secondary_spatial_belief"
                ),
                "depth_sources": evidence.get("depth_sources"),
                "gripper_envelope_relation": evidence.get("gripper_envelope_relation"),
                "placement_belief": route.get("placement_belief"),
                "anyplace_shadow": evidence.get("anyplace_shadow"),
            },
            "active_option": option.value,
            "residual_before": list(before_residual) if before_residual else None,
            "residual_after": list(after_residual) if after_residual else None,
            "requested_action": action.action_token,
            "executed_action": action.action_token if takeover else None,
            "authorized_action": action.action_token if takeover else None,
            "predicted_effect": list(action.predicted_effect) if action.predicted_effect else None,
            "observed_effect": self.controller.snapshot().get("transition"),
            "transition_verdict": {
                "status": status.value,
                "reason": action.reason,
            },
            "belief_after": jsonable(self.belief),
            "failure_event": jsonable(failure) if failure else None,
            "recovery_context": jsonable(self.belief.recovery) if self.belief.recovery else None,
            "critical_decision": jsonable(critical) if critical else None,
            "grasp_diagnostic_lift_offer": bool(
                self.placement_v22_enabled
                and action.action_token == MOVE_UP
                and self._grasp_diagnostic_lift_authorized
                and not self._grasp_diagnostic_lift_attempted
            ),
            "qwen_decision": None,
            "oracle_metadata": None,
            "visual_memory_refs": list(self.visual_memory.refs(
                instance_id=self.belief.target.instance_id if self.belief.target else None,
                grasp_epoch=self.belief.grasp_epoch,
                limit=4,
            )),
        }
        self.last_event = event
        return {
            "runtime_version": self.runtime_version,
            "takeover": takeover,
            "action_token": action.action_token,
            "status": status.value,
            "reason": action.reason,
            "option": option.value,
            "observation_health": health.value,
            "belief": jsonable(self.belief),
            "failure": jsonable(failure) if failure else None,
            "critical_decision": jsonable(critical) if critical else None,
            "grasp_diagnostic_lift": bool(
                self.placement_v22_enabled
                and action.action_token == MOVE_UP
                and self._grasp_diagnostic_lift_authorized
                and not self._grasp_diagnostic_lift_attempted
            ),
            "grasp_diagnostic_lift": bool(
                self.placement_v22_enabled
                and action.action_token == MOVE_UP
                and self._grasp_diagnostic_lift_authorized
                and not self._grasp_diagnostic_lift_attempted
            ),
            "grasp_diagnostic_lift_step_m": self.grasp_diagnostic_lift_step_m,
            "controller": self.controller.snapshot(),
            "event": event,
        }

    def observe(
        self,
        *,
        stage: str,
        evidence: dict[str, Any] | None,
        previous_action: Optional[str],
    ) -> dict[str, Any]:
        """Compatibility fallback for tools that do not expose raw frame inputs."""
        return self.observe_frame(
            stage=stage,
            evidence=evidence,
            previous_action=previous_action,
        )

    def commit_executed_action(
        self, *, executed_action: str, authorized_action: Optional[str] = None
    ) -> dict[str, Any]:
        """Tell the runtime what actually reached the robot adapter."""
        executed = str(executed_action or "").strip().upper()
        authorized = str(authorized_action or "").strip().upper()
        if (
            self.placement_v22_enabled
            and self._grasp_diagnostic_lift_authorized
            and not self._grasp_diagnostic_lift_attempted
            and bool(self.last_event.get("grasp_diagnostic_lift_offer", False))
        ):
            self._grasp_diagnostic_lift_authorized = False
            if executed == MOVE_UP and authorized == MOVE_UP:
                self._grasp_diagnostic_lift_attempted = True
            else:
                self._grasp_candidate = None
                self.belief.held = EvidenceValue(
                    None, TruthValue.UNKNOWN, "diagnostic_lift_not_executed", 0.0, self.belief.frame_id
                )
        result = self.controller.commit_executed_action(
            executed_action=executed,
            authorized_action=authorized_action,
        )
        result.setdefault("authorized_action", authorized)
        result.setdefault("requested_action", authorized)
        if isinstance(self.last_event, dict):
            self.last_event["requested_action"] = result.get("requested_action", self.last_event.get("requested_action"))
            self.last_event["authorized_action"] = result.get("authorized_action", authorized_action)
            self.last_event["executed_action"] = result.get("executed_action", executed_action)
            self.last_event["action_receipt"] = result
        return result

    def apply_critical_decision(self, answer: str, *, details: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Commit a constrained Qwen answer; free-form actions are never accepted."""
        normalized = str(answer or "UNKNOWN").strip().upper()
        details = details if isinstance(details, dict) else {}
        agent_decision = details.get("agent_decision")
        if (
            self._pending_critical_kind == "PREGRASP_DECISION"
            and (self.placement_v22_enabled or agent_decision is not None)
        ):
            valid = isinstance(agent_decision, dict)
            allowed_frames = set(self._pending_evidence_frame_ids)
            citations: list[dict[str, Any]] = []
            if valid and str(agent_decision.get("selected", "")).upper() != normalized:
                valid = False
            if valid and not bool(agent_decision.get("validated", False)):
                valid = False
            for field in ("evidence_for", "evidence_against"):
                values = agent_decision.get(field, []) if valid else []
                if not isinstance(values, list):
                    valid = False
                    break
                for item in values:
                    if not isinstance(item, dict):
                        valid = False
                        break
                    try:
                        frame = int(item.get("frame_id", -1))
                    except (TypeError, ValueError):
                        valid = False
                        break
                    camera = str(item.get("camera", "")).lower()
                    observation = str(item.get("observation", "")).strip()
                    if frame not in allowed_frames or camera not in {"agentview", "wrist"} or not observation:
                        valid = False
                        break
                    citations.append({"frame_id": frame, "camera": camera, "observation": observation})
            decision_evidence_for = (
                agent_decision.get("evidence_for", [])
                if isinstance(agent_decision, dict)
                else []
            )
            if normalized != "UNKNOWN" and not any(
                item["frame_id"] == self.belief.frame_id for item in decision_evidence_for
                if isinstance(item, dict)
            ):
                valid = False
            if not valid:
                normalized = "UNKNOWN"
            if valid and isinstance(agent_decision, dict) and self.belief.target is not None and self.belief.frame_id is not None:
                scene_description = dict(agent_decision)
                scene_description["selected_option"] = normalized
                self.visual_memory.record_decision(
                    instance_id=self.belief.target.instance_id,
                    grasp_epoch=self.belief.grasp_epoch,
                    frame_id=int(self.belief.frame_id),
                    route_epoch=int(self.belief.route_epoch),
                    relation=str(agent_decision.get("state_hypothesis") or "UNKNOWN"),
                    reasoning=str(agent_decision.get("summary") or ""),
                    scene_description=scene_description,
                    evidence_for=[item for item in citations if item["frame_id"] == self.belief.frame_id],
                    evidence_against=[item for item in citations if item["frame_id"] != self.belief.frame_id],
                    missing_observation=str(agent_decision.get("missing_observation") or ""),
                    expected_effect=str(agent_decision.get("expected_effect") or ""),
                    failure_condition=str(agent_decision.get("failure_condition") or ""),
                )
        requested_next_step = str(details.get("next_step") or "HOLD").strip().upper()
        allowed_semantic_options = {
            "CONTINUE_DESCENT", "REOBSERVE", "CHANGE_VIEW",
            "REFRESH_GEOMETRY", "REPLAN_PLACEMENT", "CLEAR_RIM", "HOLD",
        }
        if requested_next_step not in allowed_semantic_options:
            requested_next_step = "HOLD"
        frame_id = self.belief.frame_id
        if self.placement_v22_enabled and self._pending_critical_kind == "VERIFY_SEATED":
            if normalized in {"YES", PlacementRelation.SEATED_HELD.value, PlacementRelation.RELEASED_STABLE.value} and not self._placement_release_ready():
                self._pending_critical_kind = ""
                candidate = self._placement_support_candidate
                if candidate is not None and self._placement_support_stability_count == 1 and self._placement_contact_verified:
                    self._placement_seated_hypothesis = {
                        "frame_id": int(frame_id if frame_id is not None else -1),
                        "route_epoch": int(self.belief.route_epoch),
                        "grasp_epoch": int(self.belief.grasp_epoch),
                        "instance_id": self.belief.target.instance_id if self.belief.target is not None else None,
                        "relation": normalized,
                    }
                    self.belief.seated = EvidenceValue(None, TruthValue.UNKNOWN, "awaiting_new_stable_support_observation", 0.0, frame_id)
                    return {"accepted": True, "answer": normalized, "state": "PENDING_STABILITY", "next_action": STOP, "reason": "one fresh observation after the contact hypothesis is required"}
                self.belief.seated = EvidenceValue(None, TruthValue.UNKNOWN, "independent_seating_gate_unproven", 0.0, frame_id)
                return {"accepted": True, "answer": normalized, "state": "UNVERIFIED", "next_action": STOP, "reason": "fresh containment, descent support, or held identity is unproven"}
            if normalized in {"NO", PlacementRelation.RIM_CONTACT.value} and not self._pending_contact_supported:
                normalized = PlacementRelation.UNKNOWN.value
        if self._pending_critical_kind == "PREGRASP_DECISION":
            if normalized not in self._pending_allowed_answers:
                return {
                    "accepted": False,
                    "answer": normalized,
                    "reason": "answer is outside the bounded pregrasp action set",
                }
            if normalized == "UNKNOWN":
                return {
                    "accepted": True,
                    "answer": normalized,
                    "next_action": STOP,
                    "reason": "Qwen abstained; runtime fails closed for this observation",
                }
            if normalized == "REOBSERVE":
                target = self.belief.target
                eef = self.belief.eef_xyz
                current_frame = self.belief.frame_id
                key = (
                    self.belief.grasp_epoch,
                    str(target.instance_id) if target is not None else "",
                )
                request = (
                    self.last_event.get("critical_decision")
                    if isinstance(self.last_event, dict)
                    else None
                )
                is_grasp_entry_request = (
                    isinstance(request, dict)
                    and request.get("reflection_trigger") == "grasp_entry"
                )
                preconditions_hold = (
                    self.semantic_pregrasp_enabled
                    and self.belief.observation_health == ObservationHealth.OCCLUDED
                    and target is not None
                    and bool(target.instance_id)
                    and target.camera.lower() == "agentview"
                    and current_frame is not None
                    and 0
                    <= int(current_frame) - int(target.last_confirmed_frame)
                    <= self.semantic_evidence_max_age_frames
                    and eef is not None
                    and eef.truth == TruthValue.TRUE
                    and eef.frame_id == current_frame
                    and eef.value is not None
                    and self.approach_min_height_m is not None
                    and self.approach_max_height_m is not None
                    and self.approach_min_height_m
                    <= float(eef.value[2])
                    <= self.approach_max_height_m
                    and float(eef.value[2]) < self.lift_clear_height_m
                    and is_grasp_entry_request
                    and key in self._grasp_entry_review_seen
                    and key not in self._grasp_entry_reobserve_used
                )
                if not preconditions_hold:
                    return {
                        "accepted": True,
                        "answer": normalized,
                        "next_action": STOP,
                        "reason": (
                            "REOBSERVE denied: fresh identity, pose, event, or clearance "
                            "precondition failed"
                        ),
                    }
                assert eef is not None and eef.value is not None
                clearance_remaining = self.lift_clear_height_m - float(eef.value[2])
                lift_step_m = min(self.grasp_reobserve_step_m, clearance_remaining)
                if lift_step_m <= max(0.001, self.height_tolerance_m):
                    return {
                        "accepted": True,
                        "answer": normalized,
                        "next_action": STOP,
                        "reason": (
                            "REOBSERVE denied: insufficient clearance for a measurable "
                            "bounded lift"
                        ),
                    }
                self._grasp_entry_reobserve_used.add(key)
                # Remember the envelope of the action we actually authorized.
                # The next observation is expected to be above the approach
                # completion band; using approach_max_height_m here would
                # reject the very REOBSERVE motion the runtime just compiled.
                self._grasp_entry_reobserve_expected_z_m[key] = (
                    float(eef.value[2]) + lift_step_m
                )
                self.controller.prime_external_action(
                    context="QWEN_PREGRASP",
                    error=self._pending_control_error or (0.0, 0.0),
                    action=MOVE_UP,
                    defer=True,
                )
                return {
                    "accepted": True,
                    "answer": normalized,
                    "semantic_choice": normalized,
                    "next_action": MOVE_UP,
                    "requested_action": MOVE_UP,
                    "authorized_action": MOVE_UP,
                    "reobserve_for_view": True,
                    "reobserve_lift_step_m": lift_step_m,
                    "reason": (
                        "runtime compiled REOBSERVE to one bounded lift; obtain a fresh "
                        "dual-view frame before another decision"
                    ),
                }
            if self.semantic_pregrasp_enabled and normalized in SEMANTIC_PREGRASP_ACTIONS:
                decision = self._compile_semantic_pregrasp(
                    normalized,
                    evidence={
                        "spatial_belief": self._spatial_prompt_payload(self.belief.spatial),
                        "visual_alignment": self._current_visual_alignment(
                            (self.last_event or {}).get("evidence", {})
                            if isinstance(self.last_event, dict)
                            else {}
                        ),
                    },
                    eef_xyz=self.belief.eef_xyz.value,
                )
                if decision.action_token == STOP:
                    return {
                        "accepted": True,
                        "answer": normalized,
                        "next_action": STOP,
                        "reason": decision.reason,
                    }
                self.controller.prime_external_action(
                    context="QWEN_PREGRASP",
                    error=(
                        tuple(
                            float(value)
                            for value in (
                                self._current_visual_alignment(
                                    (self.last_event or {}).get("evidence", {})
                                    if isinstance(self.last_event, dict)
                                    else {}
                                )
                                or {}
                            ).get("target_minus_eef_px", ())
                        )
                        if normalized == SemanticPregraspAction.VISUAL_ALIGN.value
                        else (self._pending_control_error or (0.0, 0.0))
                    ),
                    action=decision.action_token,
                    defer=True,
                )
                return {
                    "accepted": True,
                    "answer": normalized,
                    "semantic_choice": normalized,
                    "next_action": decision.action_token,
                    "reason": decision.reason,
                    "requested_action": decision.action_token,
                    "authorized_action": decision.action_token,
                }
            if normalized in {
                "MV_LEFT",
                "MV_RIGHT",
                "MV_FWD",
                "MV_BACK",
                "MV_UP",
                "MV_DOWN",
            }:
                self.controller.prime_external_action(
                    context="QWEN_PREGRASP",
                    error=self._pending_control_error or (0.0, 0.0),
                    action=normalized,
                    defer=True,
                )
            return {
                "accepted": True,
                "answer": normalized,
                "next_action": normalized,
                "reason": "bounded Qwen pregrasp choice; re-observe after one action",
            }
        if normalized.startswith("CANDIDATE-"):
            try:
                index = int(normalized.split("-", 1)[1])
                candidate = self._pending_candidates[index]
            except (ValueError, IndexError):
                return {"accepted": False, "answer": normalized, "reason": "unknown candidate id"}
            tracker = self._pending_tracker or (
                self.destination_tracker
                if self._pending_role == "destination"
                else self.target_tracker
            )
            canonical_id = (
                self.belief.destination.instance_id
                if self._pending_role == "destination" and self.belief.destination is not None
                else (
                    self.belief.target.instance_id
                    if self._pending_role == "target" and self.belief.target is not None
                    else None
                )
            )
            current = tracker.track
            label = current.semantic_label if current is not None else self._pending_role
            track = tracker.commit_candidate(
                candidate=candidate,
                image=self._pending_image,
                frame_id=int(self._pending_frame_id or 0),
                semantic_label=label,
            )
            if track is None:
                return {"accepted": False, "answer": normalized, "reason": "invalid candidate bbox"}
            if canonical_id is not None:
                track.instance_id = canonical_id
            if self._pending_role == "destination":
                self.belief.destination = track
            else:
                self.belief.target = track
            self.belief.observation_health = ObservationHealth.VALID
            return {
                "accepted": True,
                "answer": normalized.lower(),
                "instance_id": track.instance_id,
            }
        if (
            self._pending_critical_kind == "VERIFY_SEATED"
            and normalized in {
                "YES",
                PlacementRelation.SEATED_HELD.value,
                PlacementRelation.RELEASED_STABLE.value,
            }
        ):
            self._pending_critical_kind = ""
            self._placement_verifier_relation = None
            self._placement_verifier_action_used = False
            self._placement_seated_hypothesis = {
                "frame_id": int(frame_id if frame_id is not None else -1),
                "route_epoch": int(self.belief.route_epoch),
                "grasp_epoch": int(self.belief.grasp_epoch),
                "instance_id": self.belief.target.instance_id if self.belief.target is not None else None,
                "relation": normalized,
            }
            self.belief.seated = EvidenceValue(None, TruthValue.UNKNOWN, "awaiting_fresh_stable_support_observation", 0.0, frame_id)
            return {
                "accepted": True,
                "answer": normalized,
                "state": "PENDING_STABILITY",
                "next_action": STOP if self.placement_v22_enabled else None,
                "reason": "one new stable observation is required before release",
            }
        if (
            self._pending_critical_kind == "VERIFY_SEATED"
            and normalized in {
                "NO",
                PlacementRelation.RIM_CONTACT.value,
                PlacementRelation.ABOVE_UNALIGNED.value,
                PlacementRelation.ABOVE_ALIGNED.value,
                PlacementRelation.DESCENDING_CLEAR.value,
                PlacementRelation.UNKNOWN.value,
                PlacementRelation.LOST.value,
            }
        ):
            self._pending_critical_kind = ""
            if normalized in {
                PlacementRelation.ABOVE_UNALIGNED.value,
                PlacementRelation.ABOVE_ALIGNED.value,
                PlacementRelation.DESCENDING_CLEAR.value,
            }:
                self._placement_verifier_relation = (
                    PlacementRelation.ABOVE_ALIGNED
                    if normalized == PlacementRelation.DESCENDING_CLEAR.value
                    else PlacementRelation(normalized)
                )
                self._placement_verifier_action_used = False
            else:
                self._placement_verifier_relation = None
                self._placement_verifier_action_used = False
            if normalized == PlacementRelation.UNKNOWN.value:
                self.belief.seated = EvidenceValue(None, TruthValue.UNKNOWN, "qwen_critical_placement_verifier", 0.0, frame_id)
            else:
                self.belief.seated = EvidenceValue(False, TruthValue.FALSE, "qwen_critical_placement_verifier", 0.7, frame_id)
            if normalized in {"NO", PlacementRelation.RIM_CONTACT.value} and requested_next_step == "CLEAR_RIM":
                failure = self._failure(
                    FailureCode.RIM_CONTACT,
                    self.belief.current_option,
                    frame_id,
                    "critical placement verifier rejected seating",
                )
                self._begin_recovery(failure, OptionName.ALIGN_OPENING)
            else:
                failure = None
            next_action = {
                PlacementRelation.RIM_CONTACT.value: MOVE_UP,
                "NO": MOVE_UP,
                PlacementRelation.ABOVE_ALIGNED.value: MOVE_DOWN,
                PlacementRelation.DESCENDING_CLEAR.value: MOVE_DOWN,
                # If the current metric route is already horizontally aligned,
                # this is the generic vertical continuation.  The next route
                # observation still owns the signed action and may abstain.
                PlacementRelation.ABOVE_UNALIGNED.value: (
                    MOVE_DOWN
                    if self.placement_belief.eef_residual_world is not None
                    and float(np.linalg.norm(np.asarray(self.placement_belief.eef_residual_world, dtype=float)[:2]))
                    <= self.placement_harness.action_step_m
                    else STOP
                ),
                PlacementRelation.UNKNOWN.value: STOP,
                PlacementRelation.LOST.value: STOP,
            }.get(normalized, STOP)
            request_geometry_refresh = requested_next_step in {"REFRESH_GEOMETRY", "REPLAN_PLACEMENT"}
            if requested_next_step in {"REOBSERVE", "CHANGE_VIEW", "REFRESH_GEOMETRY", "REPLAN_PLACEMENT", "HOLD"}:
                next_action = STOP
            elif requested_next_step == "CLEAR_RIM":
                next_action = MOVE_UP if self._pending_contact_supported and self.belief.held.truth == TruthValue.TRUE else STOP
            elif requested_next_step == "CONTINUE_DESCENT":
                can_continue = bool(
                    normalized in {PlacementRelation.ABOVE_ALIGNED.value, PlacementRelation.DESCENDING_CLEAR.value}
                    and self.placement_belief.fresh and not self.placement_belief.conflicts
                    and not self._pending_contact_supported
                )
                next_action = MOVE_DOWN if can_continue else STOP
            return {
                "accepted": True,
                "answer": normalized,
                "state": "NOT_SEATED",
                "next_action": next_action,
                "semantic_option": requested_next_step,
                "request_geometry_refresh": request_geometry_refresh,
                "request_alternate_view": requested_next_step == "CHANGE_VIEW",
                "rollback_stage": "PLACE",
            }
        if self.placement_v22_enabled and self._pending_critical_kind == "VERIFY_HOLD":
            self._pending_critical_kind = ""
            return self.report_grasp_verdict(
                verdict=normalized,
                frame_id=frame_id,
                mechanically_empty=False,
                reasoning=str(details.get("reasoning") or ""),
                diagnostic_lift_clear=bool(details.get("diagnostic_lift_clear", False)),
                evidence_for=details.get("evidence_for") or (),
            )
        if normalized == "YES":
            previous = self.belief.held.truth
            self.belief.held = EvidenceValue(
                True, TruthValue.TRUE, "qwen_critical_hold_verifier", 0.7, frame_id
            )
            if previous != TruthValue.TRUE:
                self.belief.grasp_epoch += 1
            # A positively verified new grasp resolves any relocalization or
            # no-progress recovery that led back to the grasp option.
            self.belief.recovery = None
            self.recovery_steps = 0
            return {"accepted": True, "answer": normalized, "state": "HELD"}
        if normalized == "NO":
            self.belief.held = EvidenceValue(
                False, TruthValue.FALSE, "qwen_critical_hold_verifier", 0.7, frame_id
            )
            failure = self._failure(
                FailureCode.EMPTY_GRASP,
                self.belief.current_option,
                frame_id,
                "critical hold verifier rejected grasp",
            )
            self._begin_recovery(failure, OptionName.ALIGN_PREGRASP)
            return {
                "accepted": True,
                "answer": normalized,
                "state": "NOT_HELD",
                "next_action": "RELEASE",
                "rollback_stage": "APPROACH",
            }
        return {"accepted": False, "answer": "UNKNOWN", "reason": "decision remained unknown"}

    def report_grasp_verdict(
        self,
        *,
        verdict: str,
        frame_id: Optional[int],
        mechanically_empty: bool,
        reasoning: str = "",
        diagnostic_lift_clear: bool = False,
        evidence_for: Sequence[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        """Transactionally commit the immediate post-close verifier result."""
        normalized = str(verdict or "UNKNOWN").strip().upper()
        if (mechanically_empty and not self.placement_v22_enabled) or normalized == "NO":
            self._wrist_final_descent_count = 0
            if self.placement_v22_enabled:
                self._grasp_candidate = None
                self._grasp_diagnostic_lift_authorized = False
                self._grasp_diagnostic_lift_attempted = False
                self._grasp_lift_observation = None
                self._grasp_hold_confirmed = False
            reason = (
                "mechanically empty close"
                if mechanically_empty and not self.placement_v22_enabled
                else (reasoning or "visual verifier rejected hold")
            )
            self.belief.held = EvidenceValue(
                False, TruthValue.FALSE, "post_close_hold_verifier", 0.9, frame_id
            )
            failure = self._failure(
                FailureCode.EMPTY_GRASP,
                OptionName.VERIFY_HOLD,
                frame_id,
                reason,
            )
            self._begin_recovery(failure, OptionName.MOVE_TO_HOVER)
            result: dict[str, Any] = {
                "verdict": Verdict.FAIL.value,
                "failure": jsonable(failure),
                "recovery_context": jsonable(self.belief.recovery),
            }
        elif normalized == "YES":
            if self.placement_v22_enabled:
                track = self.belief.target
                evidence_citations_valid = isinstance(evidence_for, (list, tuple))
                current_cited = False
                if evidence_citations_valid:
                    try:
                        for item in evidence_for:
                            if not isinstance(item, dict):
                                evidence_citations_valid = False
                                break
                            cited_frame = int(item.get("frame_id", -1))
                            if (
                                frame_id is None
                                or cited_frame not in {int(frame_id), int(frame_id) - 1}
                                or str(item.get("camera", "")).lower() not in {"agentview", "wrist"}
                                or not str(item.get("observation", "")).strip()
                            ):
                                evidence_citations_valid = False
                                break
                            current_cited = current_cited or cited_frame == int(frame_id)
                    except (TypeError, ValueError):
                        evidence_citations_valid = False
                fresh = bool(
                    frame_id is not None
                    and frame_id == self.belief.frame_id
                    and track is not None
                    and self.belief.observation_health == ObservationHealth.VALID
                    and self.belief.eef_xyz.value is not None
                    and self.belief.alignment_residual.value is not None
                    and evidence_citations_valid
                    and current_cited
                )
                candidate_ready = bool(fresh and diagnostic_lift_clear)
                self._grasp_candidate = (
                    {
                        "frame_id": int(frame_id),
                        "instance_id": track.instance_id,
                        "camera": track.camera,
                        "eef_xyz": tuple(float(value) for value in self.belief.eef_xyz.value[:3]),
                        "residual": tuple(float(value) for value in self.belief.alignment_residual.value[:2]),
                    }
                    if candidate_ready
                    else None
                )
                self._grasp_diagnostic_lift_authorized = candidate_ready
                self._grasp_diagnostic_lift_attempted = False
                self._grasp_lift_observation = None
                self._grasp_hold_confirmed = False
                self.belief.held = EvidenceValue(
                    None,
                    TruthValue.UNKNOWN,
                    "Qwen grasp candidate awaits bounded action-effect verification",
                    0.0,
                    frame_id,
                )
                result = {
                    "verdict": "CANDIDATE" if candidate_ready else Verdict.UNKNOWN.value,
                    "failure": None,
                    "diagnostic_lift_authorized": self._grasp_diagnostic_lift_authorized,
                    "reason": (
                        "one bounded diagnostic lift is authorized after fresh visual clearance"
                        if self._grasp_diagnostic_lift_authorized
                        else "fresh identity, geometry, or visible lift path is unproven"
                    ),
                }
            else:
                previous = self.belief.held.truth
                self.belief.held = EvidenceValue(
                    True,
                    TruthValue.TRUE,
                    "post_close_multiview_verifier",
                    0.7,
                    frame_id,
                )
                if previous != TruthValue.TRUE:
                    self.belief.grasp_epoch += 1
                self.belief.recovery = None
                self.recovery_steps = 0
                result = {"verdict": Verdict.PASS.value, "failure": None}
        else:
            if self.placement_v22_enabled:
                self._grasp_candidate = None
                self._grasp_diagnostic_lift_authorized = False
                self._grasp_diagnostic_lift_attempted = False
                self._grasp_lift_observation = None
                self._grasp_hold_confirmed = False
            self.belief.held = EvidenceValue(
                None,
                TruthValue.UNKNOWN,
                "post_close_verifier_abstained",
                0.0,
                frame_id,
            )
            result = {"verdict": Verdict.UNKNOWN.value, "failure": None}

        result["belief"] = jsonable(self.belief)
        if isinstance(self.last_event, dict):
            self.last_event["post_action_verification"] = result
            self.last_event["belief_after"] = jsonable(self.belief)
            if result.get("failure") is not None:
                self.last_event["failure_event"] = result["failure"]
                self.last_event["recovery_context"] = result.get("recovery_context")
        return result

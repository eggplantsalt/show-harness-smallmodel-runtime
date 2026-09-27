"""Adaptive visual-route transport support for Qwen.

CPU calibration owns coordinates, Qwen owns a revisable short-horizon intent, and
the normal controller owns every atomic action.  The plugin never reads simulator
object poses and never replaces a directional action.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw

from core.capabilities.camera_geometry import (
    CameraCalibration,
    backproject_pixel_to_plane,
    estimate_vertical_line_height,
    project_point,
)

ROUTE_PROMPT_PATH = Path(__file__).with_name("visual_route.txt")
MOVE_TOKENS = {"MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT"}
INTENTS = {"CLEAR", "TRANSFER", "ALIGN", "DESCEND", "RECOVER_CLEAR", "REACQUIRE", "READY_TO_RELEASE"}
EXPECTED_CHANGES = {"MORE_CLEARANCE", "MORE_ROUTE_PROGRESS", "LESS_OPENING_ERROR", "LOWER_STABLE", "LESS_CONTACT", "REACQUIRE_HOLD", "SEATED"}


class RoutePhase(str, Enum):
    """Measured route leg.  This is evidence, not a host action policy."""
    CLEARANCE = "CLEARANCE"
    TRANSFER = "TRANSFER"
    PRE_DESCENT = "PRE_DESCENT"
    DESCENT = "DESCENT"
    RECOVER_CLEAR = "RECOVER_CLEAR"


@dataclass(frozen=True)
class TransportGoal:
    held_target: str
    held_affordance: str
    destination_target: str
    destination_affordance: str


@dataclass
class RoutePlan:
    route_id: str
    phase: RoutePhase
    waypoints_world: list[list[float]] = field(default_factory=list)
    waypoints_px: list[list[float]] = field(default_factory=list)
    safe_transport_z_m: Optional[float] = None
    destination_xy_world: Optional[list[float]] = None
    confidence: float = 0.0
    evidence_frame_id: Optional[int] = None
    valid: bool = False
    held_bbox_xyxy: Optional[list[int]] = None
    destination_bbox_xyxy: Optional[list[int]] = None
    opening_bbox_xyxy: Optional[list[int]] = None
    estimated_rim_height_m: Optional[float] = None
    payload_below_eef_m: Optional[float] = None
    payload_lowest_z_m: Optional[float] = None
    completed_waypoints: int = 0
    last_reason: str = ""
    eef_px_xy: Optional[list[float]] = None
    grasp_epoch: Optional[int] = None
    anchor_eef_world: Optional[list[float]] = None
    active_leg: str = "CLEARANCE"
    initial_transfer_distance_m: Optional[float] = None
    geometry_source: str = "calibrated_pixel_rays"
    freshness_frames: int = 0

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["phase"] = self.phase.value
        result["confidence"] = round(float(self.confidence), 4)
        return result


@dataclass(frozen=True)
class TransportIntent:
    intent_id: str
    intent: str
    route_assessment: str
    held_assessment: str
    expected_change: str
    confidence: str
    reasoning: str
    based_on_route_id: str
    evidence_frame_id: int
    trigger: str = ""
    raw_text: str = ""
    latency_s: Optional[float] = None
    held_assessment_accepted: bool = True
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class TransportProgress:
    clearance_residual_m: Optional[float] = None
    active_leg: str = "UNKNOWN"
    along_route_progress: Optional[float] = None
    cross_track_error_m: Optional[float] = None
    opening_error_normalized: Optional[list[float]] = None
    descent_delta_m: Optional[float] = None
    hold_comotion_score: Optional[float] = None
    contact_or_stall: bool = False
    expected_change_satisfied: Optional[bool] = None
    expected_change: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# Compatibility for old telemetry readers/tests.  It is no longer used by the loop.
@dataclass(frozen=True)
class RouteReview:
    verdict: str = "UNKNOWN"
    issue: str = "NONE"
    recommended_phase: str = ""
    reasoning: str = ""
    raw_text: str = ""
    latency_s: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class RouteGateDecision:
    requested_token: str
    executed_token: str
    allowed: bool
    reason: str = ""
    trigger_reflection: bool = False
    phase: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _box(value: Any) -> Optional[tuple[float, float, float, float]]:
    try:
        if value is None or len(value) != 4:
            return None
        box = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def _complete(box: Optional[tuple[float, ...]], width: int, height: int) -> bool:
    return bool(box and box[0] > 2 and box[1] > 2 and box[2] < width - 2 and box[3] < height - 2)


def _calibration(meta: Any) -> Optional[CameraCalibration]:
    if not isinstance(meta, dict):
        return None
    try:
        return CameraCalibration(
            name="agentview", width=int(meta["width"]), height=int(meta["height"]),
            fovy_deg=float(meta["fovy_deg"]), position_world=np.asarray(meta["position_world"], dtype=float),
            camera_to_world=np.asarray(meta["camera_to_world"], dtype=float),
            rotation_degrees=int(meta.get("rotation_degrees", 0)), flip=str(meta.get("flip", "none")),
        )
    except (TypeError, ValueError, KeyError):
        return None


def _rounded(value: Any, digits: int = 5) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


class VisualRoutePlugin:
    def __init__(
        self, *, enabled: bool = False, mode: str = "shadow", client: Any = None,
        table_height_m: float = 0.015, clearance_margin_m: float = 0.02,
        geometry_window: int = 3, stall_steps: int = 2, replan_interval: int = 8,
        review_max_tokens: int = 192, intent_interval: int = 6, intent_ttl: int = 6,
        intent_cooldown: int = 2, alignment_px: float = 13.5,
        alignment_fraction: float = 0.30, alignment_min_px: float = 4.0,
        ray_residual_max_m: float = 0.08, transport_loop_enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = str(mode).lower()
        if self.mode not in {"shadow", "active"}:
            raise ValueError("visual_route mode must be shadow or active")
        self.client = client
        self.table_height_m = float(table_height_m)
        self.clearance_margin_m = max(0.0, float(clearance_margin_m))
        self.geometry_window = max(1, int(geometry_window))
        self.stall_steps = max(1, int(stall_steps))
        self.replan_interval = max(0, int(replan_interval))
        self.review_max_tokens = min(192, max(64, int(review_max_tokens)))
        self.intent_interval = max(1, int(intent_interval))
        self.intent_ttl = max(1, int(intent_ttl))
        self.intent_cooldown = max(0, int(intent_cooldown))
        self.alignment_px = max(0.0, float(alignment_px))
        self.alignment_fraction = min(0.49, max(0.05, float(alignment_fraction)))
        self.alignment_min_px = max(1.0, float(alignment_min_px))
        self.ray_residual_max_m = max(0.001, float(ray_residual_max_m))
        self.transport_loop_enabled = bool(transport_loop_enabled)
        self.reset()

    @classmethod
    def from_config(cls, cfg: dict[str, Any], *, client: Any = None, table_height_m: Optional[float] = None):
        raw = cfg.get("visual_route") or {}
        plugins = cfg.get("plugins") or {}
        enabled = bool(raw.get("enabled", plugins.get("visual_route", False))) if isinstance(raw, dict) else bool(plugins.get("visual_route", False))
        if isinstance(plugins.get("visual_route"), dict):
            enabled = bool(plugins["visual_route"].get("enabled", True))
        get = raw.get if isinstance(raw, dict) else lambda _key, default=None: default
        return cls(
            enabled=enabled, mode=str(get("mode", cfg.get("visual_route_mode", "shadow"))), client=client,
            table_height_m=float(table_height_m if table_height_m is not None else cfg.get("table_height_m", 0.015)),
            clearance_margin_m=float(get("clearance_margin_m", cfg.get("route_clearance_margin_m", 0.02))),
            geometry_window=int(get("geometry_window", cfg.get("route_geometry_window", 3))),
            stall_steps=int(get("stall_steps", cfg.get("route_stall_steps", 2))),
            replan_interval=int(get("replan_interval", cfg.get("route_replan_interval", 8))),
            review_max_tokens=int(get("intent_max_tokens", cfg.get("route_intent_max_tokens", 192))),
            intent_interval=int(get("intent_interval", cfg.get("route_intent_interval", 6))),
            intent_ttl=int(get("intent_ttl", cfg.get("route_intent_ttl", 6))),
            intent_cooldown=int(get("intent_cooldown", cfg.get("route_intent_cooldown", 2))),
            alignment_px=float(get("alignment_px", 13.5)),
            alignment_fraction=float(get("alignment_fraction", cfg.get("route_alignment_fraction", 0.30))),
            alignment_min_px=float(get("alignment_min_px", cfg.get("route_alignment_min_px", 4.0))),
            ray_residual_max_m=float(get("ray_residual_max_m", 0.08)),
            transport_loop_enabled=bool(get("transport_loop_enabled", True)),
        )

    def reset(self) -> None:
        self.route: Optional[RoutePlan] = None
        self.last_intent: Optional[TransportIntent] = None
        self.last_progress: Optional[TransportProgress] = None
        self.last_review: Optional[RouteReview] = None
        self.last_rendered: Optional[np.ndarray] = None
        self.last_evidence: dict[str, Any] = {}
        self._route_counter = self._intent_counter = self._grasp_epoch_counter = 0
        self._active_grasp_epoch: Optional[int] = None
        self._last_stage = ""
        self._geometry_samples: list[dict[str, Any]] = []
        self._frames_since_geometry_refresh = self._geometry_refresh_count = 0
        self._last_geometry_refresh_frame = -1
        self._last_geometry_refresh_reason = self._geometry_replan_requested = ""
        self._frames_since_intent = self._intent_refresh_count = 0
        self._last_intent_frame = -10_000
        self._last_intent_trigger = self._forced_intent_event = ""
        self._forced_intent_critical = False
        self._prediction_miss_count = 0
        self._last_route_phase = ""
        self._previous_metrics: dict[str, Any] = {}
        self._previous_eef = self._previous_eef_px = self._previous_held_center = None
        self._current_eef_px: Optional[list[float]] = None
        self._descent_stall_count = self._hold_missing_count = 0
        self._held_latched = False
        self._holding_arbiter: dict[str, Any] = {"state": "UNKNOWN", "reason": "transport_not_started", "rejected_qwen_assessment": None}
        self._goal_context: dict[str, str] = {}

    def _next_destination(self, subgoals: Any, current_index: int) -> tuple[str, str]:
        try:
            items = list(subgoals)[int(current_index) + 1:]
        except (TypeError, ValueError):
            items = []
        for item in items:
            if str(getattr(item, "motion", "")).upper() in {"MOVE", "PLACE", "TRANSPORT"}:
                return str(getattr(item, "target", "") or ""), str(getattr(item, "affordance", "") or "")
        return "", ""

    def request_intent_refresh(self, reason: str, *, critical: bool = False) -> None:
        self._forced_intent_event, self._forced_intent_critical = str(reason or "external_refresh"), bool(critical)

    def ready_to_release(self) -> bool:
        return bool(self.last_intent and self.last_intent.intent == "READY_TO_RELEASE" and self.last_intent.held_assessment != "LOST")

    @staticmethod
    def collapse_transport_subgoals(subgoals: Any) -> list[Any]:
        """Collapse each consecutive LIFT/MOVE/PLACE group, preserving object boundaries."""
        from core.v0_types import Subgoal
        items = list(subgoals)
        result: list[Any] = []
        index = 0
        while index < len(items):
            if str(getattr(items[index], "motion", "")).upper() not in {"LIFT", "MOVE", "PLACE"}:
                result.append(items[index]); index += 1; continue
            group = []
            while index < len(items) and str(getattr(items[index], "motion", "")).upper() in {"LIFT", "MOVE", "PLACE"}:
                group.append(items[index]); index += 1
            destination = next((item for item in reversed(group) if str(getattr(item, "motion", "")).upper() in {"MOVE", "PLACE"}), group[-1])
            result.append(Subgoal(
                id=f"transport_{getattr(group[0], 'id', len(result))}",
                target=str(getattr(destination, "target", "")),
                affordance=str(getattr(destination, "affordance", "")),
                motion="TRANSPORT",
                description="Continuously transport the verified held object to the destination using live route, progress, and periodically refreshed Qwen intent evidence.",
                completion="The held object is visibly seated in the destination and Qwen marks it ready for visual placement verification before release.",
            ))
        return result

    def _agent_calibration(self, geometry: dict[str, Any]) -> Optional[CameraCalibration]:
        agent = geometry.get("agentview") if isinstance(geometry, dict) else None
        return _calibration(agent.get("camera_calibration") if isinstance(agent, dict) else None)

    def _update_eef_px(self, geometry: dict[str, Any], eef_world: Any) -> None:
        calibration = self._agent_calibration(geometry)
        projected = project_point(calibration, np.asarray(eef_world, dtype=float).reshape(-1)) if calibration is not None and eef_world is not None else None
        self._current_eef_px = [round(float(v), 2) for v in projected["pixel_xy"]] if projected else None

    def _median_box(self, key: str, fallback: Any) -> Optional[tuple[float, ...]]:
        values = [_box(sample.get(key)) for sample in self._geometry_samples]
        valid = [value for value in values if value is not None]
        return tuple(float(v) for v in np.median(np.asarray(valid), axis=0)) if valid else _box(fallback)

    def _invalid(self, route_id: str, frame_id: int, reason: str, held: Any = None, destination: Any = None, opening: Any = None) -> RoutePlan:
        as_ints = lambda value: [int(round(v)) for v in value] if value else None
        return RoutePlan(route_id, RoutePhase.CLEARANCE, confidence=0.0, evidence_frame_id=int(frame_id), valid=False,
                         held_bbox_xyxy=as_ints(held), destination_bbox_xyxy=as_ints(destination), opening_bbox_xyxy=as_ints(opening),
                         last_reason=reason, grasp_epoch=self._active_grasp_epoch, active_leg="UNKNOWN")

    def _estimate_route(self, *, frame_id: int, eef_world: Any, geometry: dict[str, Any], held_evidence: Optional[dict[str, Any]], destination_evidence: Optional[dict[str, Any]], preserve_grasp_geometry: bool = False, reset_anchor: bool = False) -> RoutePlan:
        self._route_counter += 1
        route_id = f"route-{self._route_counter:04d}"
        calibration = self._agent_calibration(geometry)
        eef_values = [sample.get("eef_world") for sample in self._geometry_samples if sample.get("eef_world") is not None]
        try:
            eef = np.median(np.asarray(eef_values, dtype=float), axis=0).reshape(-1) if eef_values else np.asarray(eef_world, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            eef = np.asarray([])
        held = self._median_box("held_bbox_xyxy", (held_evidence or {}).get("bbox_xyxy"))
        destination = self._median_box("destination_bbox_xyxy", (destination_evidence or {}).get("bbox_xyxy"))
        opening = self._median_box("opening_bbox_xyxy", (destination_evidence or {}).get("opening_bbox_xyxy"))
        size = (calibration.width, calibration.height) if calibration else (256, 256)
        if calibration is None or eef.size < 3 or not all(_complete(box, *size) for box in (held, destination, opening)):
            return self._invalid(route_id, frame_id, "incomplete_visual_route_evidence", held, destination, opening)
        opening_center, destination_center = _center(opening), _center(destination)  # type: ignore[arg-type]
        opening_world = backproject_pixel_to_plane(calibration, opening_center, self.table_height_m)
        destination_world = backproject_pixel_to_plane(calibration, destination_center, self.table_height_m)
        if opening_world is None:
            return self._invalid(route_id, frame_id, "destination_backprojection_failed", held, destination, opening)
        destination_world = destination_world if destination_world is not None else opening_world.copy()

        payload_probe = estimate_vertical_line_height(calibration, (_center(held)[0], held[3]), eef[:2])  # type: ignore[index]
        payload_below = payload_lowest = None
        payload_residual = math.inf
        if payload_probe:
            payload_residual = float(payload_probe.get("residual_m", math.inf))
            candidate_lowest = float(payload_probe.get("height_m", math.nan))
            candidate_below = float(eef[2]) - candidate_lowest
            if payload_residual <= self.ray_residual_max_m and math.isfinite(candidate_lowest) and 0.005 <= candidate_below <= 0.50:
                payload_lowest, payload_below = candidate_lowest, candidate_below
        rim_values, rim_residuals = [], []
        for px in ((opening[0], opening[1]), (_center(opening)[0], opening[1]), (opening[2], opening[1])):  # type: ignore[index]
            estimate = estimate_vertical_line_height(calibration, px, opening_world[:2])
            if estimate and float(estimate.get("residual_m", math.inf)) <= self.ray_residual_max_m:
                value = float(estimate.get("height_m", math.nan))
                if math.isfinite(value):
                    rim_values.append(value); rim_residuals.append(float(estimate["residual_m"]))
        rim_height = float(np.median(rim_values)) if rim_values else None
        old = self.route if preserve_grasp_geometry and self.route and self.route.valid else None
        if old:
            # Core invariant: periodic refresh cannot make safe_z follow current EEF.
            payload_below, rim_height, safe_z = old.payload_below_eef_m, old.estimated_rim_height_m, old.safe_transport_z_m
            payload_lowest = float(eef[2]) - float(payload_below) if payload_below is not None else old.payload_lowest_z_m
        elif payload_below is not None and rim_height is not None:
            safe_z = float(rim_height) + float(payload_below) + self.clearance_margin_m
        else:
            return self._invalid(route_id, frame_id, "payload_vertical_ray_invalid" if payload_below is None else "rim_vertical_ray_invalid", held, destination, opening)
        assert payload_below is not None and rim_height is not None and safe_z is not None
        anchor = np.asarray(old.anchor_eef_world, dtype=float) if old and old.anchor_eef_world and not reset_anchor else np.asarray(eef[:3], dtype=float)
        world = [np.asarray([anchor[0], anchor[1], safe_z]), np.asarray([destination_world[0], destination_world[1], safe_z]), np.asarray([opening_world[0], opening_world[1], safe_z])]
        pixels = []
        for point in world:
            payload_point = point.copy(); payload_point[2] -= 0.5 * float(payload_below)
            projected = project_point(calibration, payload_point)
            if not projected:
                return self._invalid(route_id, frame_id, "route_projection_failed", held, destination, opening)
            pixels.append([round(float(v), 2) for v in projected["pixel_xy"]])
        eef_projection = project_point(calibration, eef[:3])
        initial_distance = old.initial_transfer_distance_m if old and old.initial_transfer_distance_m is not None else float(np.linalg.norm(eef[:2] - opening_world[:2]))
        confidence = max(0.0, 0.95 - min(0.35, payload_residual * 2) - min(0.2, float(np.median(rim_residuals)) if rim_residuals else 0.1))
        return RoutePlan(
            route_id, old.phase if old else RoutePhase.CLEARANCE,
            [[round(float(v), 5) for v in point] for point in world], pixels,
            round(float(safe_z), 5), [round(float(opening_world[0]), 5), round(float(opening_world[1]), 5)], round(confidence, 4), int(frame_id), True,
            [int(round(v)) for v in held], [int(round(v)) for v in destination], [int(round(v)) for v in opening],
            round(float(rim_height), 5), round(float(payload_below), 5), _rounded(payload_lowest),
            old.completed_waypoints if old else 0, "", [round(float(v), 2) for v in eef_projection["pixel_xy"]] if eef_projection else None,
            self._active_grasp_epoch, [round(float(v), 5) for v in anchor], old.active_leg if old else "CLEARANCE", round(float(initial_distance), 5),
        )

    def _replace_route(self, *, frame_id: int, eef_world: Any, geometry: dict[str, Any], held_evidence: Optional[dict[str, Any]], destination_evidence: Optional[dict[str, Any]], reason: str, reset_anchor: bool = False) -> None:
        candidate = self._estimate_route(frame_id=frame_id, eef_world=eef_world, geometry=geometry, held_evidence=held_evidence, destination_evidence=destination_evidence, preserve_grasp_geometry=bool(self.route and self.route.valid), reset_anchor=reset_anchor)
        if not candidate.valid and self.route and self.route.valid:
            self.route.freshness_frames += 1; self.route.confidence = max(0.0, self.route.confidence - 0.05)
            self.route.last_reason = f"refresh_rejected:{reason}:{candidate.last_reason}"
        else:
            if candidate.valid:
                candidate.last_reason = reason
            self.route = candidate
        self._frames_since_geometry_refresh = 0
        self._geometry_refresh_count += 1
        self._last_geometry_refresh_frame, self._last_geometry_refresh_reason = int(frame_id), str(reason)
        self._geometry_replan_requested = ""

    def _alignment_signal(self, held_evidence: Optional[dict[str, Any]], destination_evidence: Optional[dict[str, Any]]) -> dict[str, Any]:
        held, destination = held_evidence or {}, destination_evidence or {}
        held_box, opening = _box(held.get("bbox_xyxy") or held.get("held_bbox_xyxy")), _box(destination.get("opening_bbox_xyxy"))
        if held_box and opening:
            hc, oc = _center(held_box), _center(opening)
            dx, dy = oc[0] - hc[0], oc[1] - hc[1]
            ow, oh, hw = max(1.0, opening[2] - opening[0]), max(1.0, opening[3] - opening[1]), max(1.0, held_box[2] - held_box[0])
            limits = [max(self.alignment_min_px, self.alignment_fraction * ow), max(self.alignment_min_px, self.alignment_fraction * oh)]
            fits = hw <= ow * 1.1
            return {"known": True, "aligned": bool(fits and abs(dx) <= limits[0] and abs(dy) <= limits[1]), "source": "opening_relative_geometry", "error_px": [round(dx, 2), round(dy, 2)], "normalized_error": [round(dx / ow, 4), round(dy / oh, 4)], "limits_px": [round(v, 2) for v in limits], "opening_size_px": [round(ow, 2), round(oh, 2)], "held_width_px": round(hw, 2), "width_fits": fits}
        if "known" in held and "aligned" in held:
            return {"known": bool(held.get("known")), "aligned": bool(held.get("aligned")), "source": "visual_harness_relation", "error_px": held.get("destination_minus_held_center_px"), "limits_px": [self.alignment_px] * 2}
        return {"known": False, "aligned": False, "source": "insufficient_opening_geometry"}

    def _compute_progress(self, eef_world: Any, held: Optional[dict[str, Any]], destination: Optional[dict[str, Any]], previous_action: Optional[str]) -> TransportProgress:
        if not self.route or not self.route.valid:
            return TransportProgress()
        try:
            eef = np.asarray(eef_world, dtype=float).reshape(-1)
            safe = float(self.route.safe_transport_z_m)
            goal = np.asarray(self.route.destination_xy_world, dtype=float)
        except (TypeError, ValueError):
            return TransportProgress(active_leg=self.route.active_leg)
        residual = max(0.0, safe - float(eef[2]))
        remaining = float(np.linalg.norm(goal - eef[:2])); initial = max(1e-6, float(self.route.initial_transfer_distance_m or remaining or 1))
        along = min(1.0, max(0.0, 1.0 - remaining / initial))
        cross = None
        if self.route.anchor_eef_world:
            start = np.asarray(self.route.anchor_eef_world[:2]); line = goal - start; denom = float(np.dot(line, line))
            if denom > 1e-9:
                t = min(1.0, max(0.0, float(np.dot(eef[:2] - start, line) / denom))); cross = float(np.linalg.norm(eef[:2] - (start + t * line)))
        alignment = self._alignment_signal(held, destination); opening_error = alignment.get("normalized_error")
        dz = float(self._previous_eef[2] - eef[2]) if self._previous_eef is not None else None
        held_box = _box((held or {}).get("bbox_xyxy")); held_center = np.asarray(_center(held_box)) if held_box else None
        eef_px = np.asarray(self._current_eef_px) if self._current_eef_px else None
        comotion = None
        if held_center is not None and eef_px is not None and self._previous_held_center is not None and self._previous_eef_px is not None:
            comotion = math.exp(-float(np.linalg.norm((held_center - self._previous_held_center) - (eef_px - self._previous_eef_px))) / 20.0)
        down_stall = str(previous_action or "").upper() == "MV_DOWN" and dz is not None and dz <= 1e-4
        self._descent_stall_count = self._descent_stall_count + 1 if down_stall else 0
        contact = bool((held or {}).get("rim_contact_risk") or self._descent_stall_count >= self.stall_steps)
        expected = self.last_intent.expected_change if self.last_intent else ""
        old = self._previous_metrics; satisfied = None
        old_error = old.get("opening_error"); new_error = max(abs(float(v)) for v in opening_error) if opening_error is not None else None
        if expected and old:
            if expected == "MORE_CLEARANCE": satisfied = residual < float(old.get("clearance", math.inf)) - 1e-4
            elif expected == "MORE_ROUTE_PROGRESS": satisfied = along > float(old.get("along", -math.inf)) + 1e-4
            elif expected == "LESS_OPENING_ERROR" and new_error is not None: satisfied = old_error is not None and new_error < float(old_error) - .005
            elif expected == "LOWER_STABLE": satisfied = bool(dz is not None and dz > 1e-4 and (old_error is None or new_error is None or new_error <= float(old_error) + .05))
            elif expected == "LESS_CONTACT": satisfied = not contact
            elif expected == "REACQUIRE_HOLD": satisfied = self._holding_arbiter.get("state") == "HELD"
            elif expected == "SEATED": satisfied = bool(alignment.get("aligned") and dz is not None and dz >= 0)
        self._previous_metrics = {"clearance": residual, "along": along, "opening_error": new_error}
        self._previous_eef, self._previous_eef_px, self._previous_held_center = eef.copy(), eef_px.copy() if eef_px is not None else None, held_center.copy() if held_center is not None else None
        return TransportProgress(round(residual, 5), self.route.active_leg, round(along, 4), _rounded(cross), [round(float(v), 4) for v in opening_error] if opening_error is not None else None, _rounded(dz), _rounded(comotion, 4), contact, satisfied, expected)

    def _update_holding(self, held: Optional[dict[str, Any]], gripper_closed: bool, comotion: Optional[float]) -> None:
        box = _box((held or {}).get("bbox_xyxy"))
        try: confidence = float((held or {}).get("confidence", 0) or 0)
        except (TypeError, ValueError): confidence = 0.0
        present = box is not None and confidence >= .35
        if self._held_latched and gripper_closed and present:
            self._hold_missing_count = 0; self._holding_arbiter = {"state": "HELD", "reason": "latched_closed_with_visual_track", "visual_track_confidence": round(confidence, 4), "hold_comotion_score": comotion, "rejected_qwen_assessment": None}
        elif self._held_latched and gripper_closed:
            self._hold_missing_count += 1; self._holding_arbiter = {"state": "SUSPECTED_LOST" if self._hold_missing_count >= 2 else "UNKNOWN", "reason": "visual_track_missing_consecutive_frames", "missing_frames": self._hold_missing_count, "hold_comotion_score": comotion, "rejected_qwen_assessment": None}
        else:
            self._holding_arbiter = {"state": "UNKNOWN", "reason": "gripper_not_closed_or_hold_not_latched", "rejected_qwen_assessment": None}

    def _update_phase(self, stage: str, eef_world: Any, held: Optional[dict[str, Any]], destination: Optional[dict[str, Any]], *, previous_action: Optional[str] = None) -> None:
        if not self.route or not self.route.valid: return
        try: z = float(np.asarray(eef_world).reshape(-1)[2]); safe = float(self.route.safe_transport_z_m)
        except (TypeError, ValueError, IndexError): return
        aligned = self._alignment_signal(held, destination); risk = bool((held or {}).get("rim_contact_risk"))
        stalled = bool(self.last_progress and self.last_progress.contact_or_stall)
        if risk or stalled: phase = RoutePhase.RECOVER_CLEAR
        elif z + .003 < safe: phase = RoutePhase.CLEARANCE
        elif not (aligned.get("known") and aligned.get("aligned")): phase = RoutePhase.TRANSFER
        elif self.last_intent and self.last_intent.intent in {"DESCEND", "READY_TO_RELEASE"}: phase = RoutePhase.DESCENT
        else: phase = RoutePhase.PRE_DESCENT
        self.route.phase = phase; self.route.active_leg = phase.value
        self.route.completed_waypoints = {RoutePhase.CLEARANCE: 0, RoutePhase.TRANSFER: 1, RoutePhase.PRE_DESCENT: 2, RoutePhase.DESCENT: 2, RoutePhase.RECOVER_CLEAR: 0}[phase]

    def _intent_trigger(self, frame_id: int) -> tuple[str, bool]:
        if self._forced_intent_event: return self._forced_intent_event, self._forced_intent_critical
        if self.last_intent is None: return "transport_entry", True
        if self.last_progress and self.last_progress.contact_or_stall: return "contact_or_stall", True
        if self._holding_arbiter.get("state") in {"SUSPECTED_LOST", "LOST"}: return "possible_hold_loss", True
        if not self.route or not self.route.valid: return "route_invalid", True
        if self.route.phase.value != self._last_route_phase: return "geometry_relation_changed", False
        if self._prediction_miss_count >= 2: return "prediction_miss_two_steps", False
        if self._frames_since_intent >= self.intent_ttl: return "intent_ttl_expired", False
        if self._frames_since_intent >= self.intent_interval: return f"periodic_{self.intent_interval}_actions", False
        return "", False

    def _plan_intent(self, *, image: np.ndarray, wrist: Any, frame_id: int, trigger: str, held: Optional[dict[str, Any]], destination: Optional[dict[str, Any]], previous_action: Optional[str], debug: bool) -> TransportIntent:
        self._intent_counter += 1; intent_id = f"intent-{self._intent_counter:04d}"; route_id = self.route.route_id if self.route else ""
        if self.client is None:
            return TransportIntent(intent_id, "REACQUIRE", "UNKNOWN", "UNKNOWN", "REACQUIRE_HOLD", "LOW", "intent planner unavailable", route_id, int(frame_id), trigger)
        evidence = {"goal": self._goal_context, "trigger": trigger, "previous_intent": self.last_intent.to_dict() if self.last_intent else None, "previous_action": previous_action, "observed_progress": self.last_progress.to_dict() if self.last_progress else None, "route": {key: self.route.to_dict().get(key) for key in ("route_id", "valid", "active_leg", "confidence", "safe_transport_z_m", "last_reason")} if self.route else None, "holding_arbiter": self._holding_arbiter, "opening_alignment": self._alignment_signal(held, destination)}
        prompt = ROUTE_PROMPT_PATH.read_text(encoding="utf-8").strip() + "\n\nCURRENT CLOSED-LOOP EVIDENCE:\n" + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
        schema = {"type": "object", "properties": {"intent": {"type": "string", "enum": sorted(INTENTS)}, "route_assessment": {"type": "string", "enum": ["VALID", "REPLAN", "UNKNOWN"]}, "held_assessment": {"type": "string", "enum": ["HELD", "LOST", "UNKNOWN"]}, "expected_change": {"type": "string", "enum": sorted(EXPECTED_CHANGES)}, "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}, "reasoning": {"type": "string"}}, "required": ["intent", "route_assessment", "held_assessment", "expected_change", "confidence", "reasoning"], "additionalProperties": False}
        started = time.monotonic()
        before_metrics = dict(getattr(self.client, "metrics", {}) or {})
        try:
            response = self.client.complete_json(prompt, image, wrist_image=wrist, schema=schema, max_tokens=self.review_max_tokens, temperature=0.0, chat_template_kwargs={"enable_thinking": False, "thinking": False}, debug=debug, agentview_label="AgentView with thin CPU route", wrist_label="Wrist live holding evidence")
            payload = response.payload.get("json")
            if not isinstance(payload, dict): raise RuntimeError("transport intent planner returned no JSON")
            held_assessment = str(payload.get("held_assessment", "UNKNOWN")).upper(); accepted = not (held_assessment == "LOST" and self._holding_arbiter.get("state") == "HELD")
            if not accepted: self._holding_arbiter["rejected_qwen_assessment"] = "LOST"
            after_metrics = dict(getattr(self.client, "metrics", {}) or {})
            prompt_tokens = int(after_metrics.get("prompt_tokens", 0)) - int(before_metrics.get("prompt_tokens", 0))
            completion_tokens = int(after_metrics.get("completion_tokens", 0)) - int(before_metrics.get("completion_tokens", 0))
            result = TransportIntent(intent_id, str(payload["intent"]).upper(), str(payload["route_assessment"]).upper(), held_assessment, str(payload["expected_change"]).upper(), str(payload["confidence"]).upper(), str(payload.get("reasoning", "")), route_id, int(frame_id), trigger, response.raw_text, time.monotonic() - started, accepted, prompt_tokens, completion_tokens)
            if result.route_assessment == "REPLAN": self._geometry_replan_requested = "qwen_route_assessment_replan"
            return result
        except Exception as exc:
            return TransportIntent(intent_id, self.last_intent.intent if self.last_intent else "REACQUIRE", "UNKNOWN", "UNKNOWN", self.last_intent.expected_change if self.last_intent else "REACQUIRE_HOLD", "LOW", f"intent planner unavailable: {type(exc).__name__}: {exc}", route_id, int(frame_id), trigger, latency_s=time.monotonic() - started)

    def update(self, *, agentview: np.ndarray, wrist: Any, stage: str, subgoals: Any, current_index: int, frame_id: int, eef_world: Any, geometry: dict[str, Any], held_evidence: Optional[dict[str, Any]], destination_evidence: Optional[dict[str, Any]], gripper_closed: bool, previous_action: Optional[str] = None, debug: bool = False) -> dict[str, Any]:
        if not self.enabled: return {"agentview": agentview, "context": "", "evidence": {}, "route_gate": None}
        stage_name = str(stage).upper(); supported = {"TRANSPORT"} if self.transport_loop_enabled else {"LIFT", "MOVE", "PLACE"}
        if stage_name not in supported:
            self.route = None; self.last_intent = None; self.last_progress = None; self._geometry_samples = []; self._active_grasp_epoch = None; self._last_stage = stage_name
            return {"agentview": agentview, "context": "", "evidence": {"enabled": True, "mode": self.mode, "route": None, "stage": stage_name, "frame_id": int(frame_id), "route_discarded": "stage_outside_transport"}, "route_gate": None}
        entering = stage_name != self._last_stage or self._active_grasp_epoch is None
        try:
            current_goal = list(subgoals)[int(current_index)]
        except (TypeError, ValueError, IndexError):
            current_goal = None
        held_target = held_affordance = ""
        try:
            for item in reversed(list(subgoals)[: int(current_index)]):
                if str(getattr(item, "motion", "")).upper() == "GRASP":
                    held_target = str(getattr(item, "target", "") or "")
                    held_affordance = str(getattr(item, "affordance", "") or "")
                    break
        except (TypeError, ValueError):
            pass
        self._goal_context = {
            "held_target": held_target,
            "held_affordance": held_affordance,
            "destination_target": str(getattr(current_goal, "target", "") or ""),
            "destination_affordance": str(getattr(current_goal, "affordance", "") or ""),
        }
        if entering:
            self._grasp_epoch_counter += 1; self._active_grasp_epoch = self._grasp_epoch_counter; self._held_latched = bool(gripper_closed)
            self._geometry_samples = []; self._previous_metrics = {}; self._previous_eef = self._previous_eef_px = self._previous_held_center = None
            self._frames_since_intent = self._frames_since_geometry_refresh = 0; self._forced_intent_event = "transport_entry"; self._forced_intent_critical = True
        self._last_stage = stage_name; self._update_eef_px(geometry, eef_world)
        self._geometry_samples.append({"eef_world": np.asarray(eef_world, dtype=float).reshape(-1).tolist() if eef_world is not None else None, "held_bbox_xyxy": (held_evidence or {}).get("bbox_xyxy"), "destination_bbox_xyxy": (destination_evidence or {}).get("bbox_xyxy"), "opening_bbox_xyxy": (destination_evidence or {}).get("opening_bbox_xyxy")})
        self._geometry_samples = self._geometry_samples[-self.geometry_window:]
        if self.route is None:
            self._replace_route(frame_id=frame_id, eef_world=eef_world, geometry=geometry, held_evidence=held_evidence, destination_evidence=destination_evidence, reason="initial_verified_grasp")
        else:
            self._frames_since_geometry_refresh += 1; refresh = self._geometry_replan_requested
            if not refresh and self.replan_interval and self._frames_since_geometry_refresh >= self.replan_interval: refresh = f"periodic_{self.replan_interval}_frames"
            if refresh: self._replace_route(frame_id=frame_id, eef_world=eef_world, geometry=geometry, held_evidence=held_evidence, destination_evidence=destination_evidence, reason=refresh, reset_anchor=refresh == "contact_recovery_complete")
        self.last_progress = self._compute_progress(eef_world, held_evidence, destination_evidence, previous_action)
        self._update_holding(held_evidence, gripper_closed, self.last_progress.hold_comotion_score)
        if self.last_progress.expected_change_satisfied is False: self._prediction_miss_count += 1
        elif self.last_progress.expected_change_satisfied is True: self._prediction_miss_count = 0
        phase_before_update = self.route.phase if self.route and self.route.valid else None
        self._update_phase(stage_name, eef_world, held_evidence, destination_evidence, previous_action=previous_action)
        if (
            phase_before_update == RoutePhase.RECOVER_CLEAR
            and self.route is not None
            and self.route.phase != RoutePhase.RECOVER_CLEAR
        ):
            self._geometry_replan_requested = "contact_recovery_complete"
            self.request_intent_refresh("contact_recovery_complete")
        if self.route: self.last_progress = replace(self.last_progress, active_leg=self.route.active_leg)
        rendered = self.render(agentview, held_evidence=held_evidence, destination_evidence=destination_evidence)
        self._frames_since_intent += 1; trigger, critical = self._intent_trigger(int(frame_id)); cooldown = int(frame_id) - self._last_intent_frame >= self.intent_cooldown
        if trigger and (critical or cooldown):
            self.last_intent = self._plan_intent(image=rendered, wrist=wrist, frame_id=frame_id, trigger=trigger, held=held_evidence, destination=destination_evidence, previous_action=previous_action, debug=debug)
            if (
                self.last_intent.held_assessment == "LOST"
                and self.last_intent.held_assessment_accepted
                and self._hold_missing_count >= 2
            ):
                self._holding_arbiter["state"] = "LOST"
                self._holding_arbiter["reason"] = (
                    "two_frame_visual_loss_confirmed_by_qwen_temporal_review"
                )
            self._intent_refresh_count += 1; self._last_intent_frame = int(frame_id); self._last_intent_trigger = trigger; self._frames_since_intent = self._prediction_miss_count = 0; self._forced_intent_event = ""; self._forced_intent_critical = False
            if self.last_intent.intent == "CLEAR" and self.last_progress.clearance_residual_m == 0.0 and not self.last_progress.contact_or_stall: self.request_intent_refresh("intent_disagreement_clearance_complete")
        self._last_route_phase = self.route.phase.value if self.route else ""
        context = self.prompt_context(held_evidence=held_evidence, destination_evidence=destination_evidence, gripper_closed=gripper_closed, eef_world=eef_world, stage=stage_name)
        evidence = {"enabled": True, "mode": self.mode, "stage": stage_name, "frame_id": int(frame_id), "grasp_epoch": self._active_grasp_epoch, "route": self.route.to_dict() if self.route else None, "intent": self.last_intent.to_dict() if self.last_intent else None, "progress": self.last_progress.to_dict(), "holding_arbiter": dict(self._holding_arbiter), "geometry_refresh_count": self._geometry_refresh_count, "last_geometry_refresh_frame": self._last_geometry_refresh_frame, "last_geometry_refresh_reason": self._last_geometry_refresh_reason, "frames_since_geometry_refresh": self._frames_since_geometry_refresh, "intent_refresh_count": self._intent_refresh_count, "last_intent_frame": self._last_intent_frame, "last_intent_trigger": self._last_intent_trigger, "intent_age": self._frames_since_intent, "prediction_miss_count": self._prediction_miss_count, "previous_action": previous_action}
        self.last_rendered, self.last_evidence = rendered, evidence
        return {"agentview": rendered, "context": context, "evidence": evidence, "route_gate": None}

    def gate(self, token: str, *, stage: str, held_evidence: Optional[dict[str, Any]] = None, destination_evidence: Optional[dict[str, Any]] = None, eef_world: Any = None) -> RouteGateDecision:
        requested = str(token or "").upper(); phase = self.route.phase.value if self.route else ""
        if not self.enabled or self.mode != "active" or not self.route: return RouteGateDecision(requested, requested, True, phase=phase)
        reasons = []; progress = self.last_progress; intent = self.last_intent.intent if self.last_intent else ""
        if progress and progress.contact_or_stall and requested == "MV_DOWN": reasons.append("down_during_contact_or_stall")
        if progress and progress.clearance_residual_m and requested in MOVE_TOKENS: reasons.append("lateral_before_measured_clearance")
        if intent == "TRANSFER" and requested == "MV_UP" and progress and progress.clearance_residual_m == 0.0: reasons.append("continued_lift_after_clearance")
        alignment = self._alignment_signal(held_evidence, destination_evidence)
        if requested == "MV_DOWN" and alignment.get("known") and not alignment.get("aligned"):
            reasons.append("down_while_not_aligned")
        if requested == "MV_DOWN" and bool((held_evidence or {}).get("rim_contact_risk")):
            reasons.append("rim_contact_risk")
        if reasons: self.request_intent_refresh("action_route_conflict")
        return RouteGateDecision(requested, requested, True, ";".join(reasons), bool(reasons), phase)

    @staticmethod
    def _arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], color: tuple[int, int, int, int], scale: float) -> None:
        dx, dy = end[0] - start[0], end[1] - start[1]; length = math.hypot(dx, dy)
        if length < 7 * scale: return
        ux, uy = dx / length, dy / length; tip = (start[0] + .62 * dx, start[1] + .62 * dy); size, wing = 4.5 * scale, 2.8 * scale
        left = (tip[0] - size * ux + wing * uy, tip[1] - size * uy - wing * ux); right = (tip[0] - size * ux - wing * uy, tip[1] - size * uy + wing * ux)
        draw.line([left, tip, right], fill=(0, 0, 0, 180), width=max(2, int(scale * 1.75))); draw.line([left, tip, right], fill=color, width=max(1, int(scale)))

    def render(self, image: np.ndarray, *, held_evidence: Optional[dict[str, Any]], destination_evidence: Optional[dict[str, Any]]) -> np.ndarray:
        raw = np.asarray(image)
        if not self.route or not self.route.valid: return raw.copy()
        base = Image.fromarray(raw.astype(np.uint8), mode="RGB"); factor = 4
        canvas = base.resize((base.width * factor, base.height * factor), Image.Resampling.LANCZOS); draw = ImageDraw.Draw(canvas, "RGBA")
        held = _box((held_evidence or {}).get("bbox_xyxy")) or _box(self.route.held_bbox_xyxy)
        start = _center(held) if held else tuple(self.route.eef_px_xy or self.route.waypoints_px[0]); points = [start] + [tuple(p) for p in self.route.waypoints_px]; scaled = [(x * factor, y * factor) for x, y in points]
        complete = min(len(scaled) - 1, int(self.route.completed_waypoints))
        for index in range(len(scaled) - 1):
            color = (45, 196, 92, 235) if index < complete else (35, 145, 245, 235)
            draw.line([scaled[index], scaled[index + 1]], fill=(0, 0, 0, 155), width=7); draw.line([scaled[index], scaled[index + 1]], fill=color, width=5); self._arrow(draw, scaled[index], scaled[index + 1], color, factor)
        for point, radius, color in ((scaled[0], 3 * factor, (255, 211, 45, 245)), (scaled[min(len(scaled) - 1, max(1, complete + 1))], 3.5 * factor, (255, 211, 45, 245))):
            draw.ellipse([point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius], fill=color, outline=(0, 0, 0, 160), width=2)
        opening = _box((destination_evidence or {}).get("opening_bbox_xyxy")) or _box(self.route.opening_bbox_xyxy); goal = _center(opening) if opening else tuple(self.route.waypoints_px[-1]); goal = (goal[0] * factor, goal[1] * factor); radius = 5 * factor
        draw.ellipse([goal[0] - radius, goal[1] - radius, goal[0] + radius, goal[1] + radius], outline=(35, 145, 245, 235), width=5)
        if held: draw.rectangle([int(round(v * factor)) for v in held], outline=(255, 211, 45, 225), width=5)
        return np.asarray(canvas.resize(base.size, Image.Resampling.LANCZOS))

    def prompt_context(self, *, held_evidence: Optional[dict[str, Any]], destination_evidence: Optional[dict[str, Any]], gripper_closed: bool, eef_world: Any = None, stage: str = "") -> str:
        if not self.enabled or str(stage).upper() != "TRANSPORT": return ""
        route = f"route={self.route.route_id}, active_leg={self.route.active_leg}, confidence={self.route.confidence:.2f}" if self.route and self.route.valid else "route=INVALID/UNKNOWN"
        intent = f"intent={self.last_intent.intent}, expected_change={self.last_intent.expected_change}, confidence={self.last_intent.confidence}, age={self._frames_since_intent}" if self.last_intent else "intent=UNKNOWN"
        progress = json.dumps(self.last_progress.to_dict(), ensure_ascii=False, separators=(",", ":")) if self.last_progress else "unknown"
        lost_note = ""
        if self._holding_arbiter.get("state") == "LOST":
            lost_note = (
                " The two-frame holding arbiter and Qwen temporal review agree that "
                "the object was lost. Choose RELEASE to reopen before the runner "
                "returns to visual reacquisition; do not continue the stale route."
            )
        return "\nTRANSPORT CLOSED LOOP: This is one continuous transport goal, not a LIFT/MOVE/PLACE stage sequence. The short-term intent below was produced by Qwen and is revisable, not a host action. CPU route geometry supplies evidence only and never replaces your token. The thin overlay has no text: blue is remaining payload path, green is completed path, yellow marks the carried object/current waypoint, and the blue ring is the opening endpoint. " + f"{intent}; {route}. Observed progress={progress}. Choose the next atomic action from the live images and residuals. If current evidence contradicts the old intent or route, output DONE to request an immediate intent refresh. Do not infer that more lifting is needed merely because the payload appears below the hand." + lost_note

    def metadata(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "mode": self.mode, "transport_loop_enabled": self.transport_loop_enabled, "clearance_margin_m": self.clearance_margin_m, "geometry_window": self.geometry_window, "stall_steps": self.stall_steps, "geometry_refresh_interval": self.replan_interval, "intent_interval": self.intent_interval, "intent_ttl": self.intent_ttl, "intent_cooldown": self.intent_cooldown, "geometry_refresh_count": self._geometry_refresh_count, "intent_refresh_count": self._intent_refresh_count, "route": self.route.to_dict() if self.route else None, "intent": self.last_intent.to_dict() if self.last_intent else None, "progress": self.last_progress.to_dict() if self.last_progress else None, "holding_arbiter": dict(self._holding_arbiter)}

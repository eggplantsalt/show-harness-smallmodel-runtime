"""The sole canonical state schema for Runtime V3."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ObjectRelativeState:
    """Target identity evidence and the fixed visual control reference."""

    target_phrase: str
    target_visible: bool
    target_quality_score: Optional[float] = None
    target_centroid_px: Optional[tuple[float, float]] = None
    target_bbox_xyxy: Optional[tuple[float, float, float, float]] = None
    target_mask_area_px: Optional[int] = None
    eef_projection_px: Optional[tuple[float, float]] = None
    image_error_px: Optional[tuple[float, float]] = None
    image_error_norm_px: Optional[float] = None
    camera: str = "agentview"
    evidence_timestamp: Optional[float] = None
    source_width: Optional[int] = None
    source_height: Optional[int] = None
    target_identity_status: str = "UNANCHORED"
    target_candidate_id: Optional[str] = None
    target_reference_point_px: Optional[tuple[float, float]] = None
    target_reference_valid: bool = False
    target_reference_camera: Optional[str] = None
    target_reference_invalidation_reason: Optional[str] = None


@dataclass(frozen=True)
class BeliefState:
    task_id: Optional[str] = None
    step_id: int = 0
    stage: Optional[str] = None
    target_identity: Optional[str] = None
    target_confidence: Optional[float] = None
    target_pose: Optional[tuple[float, ...]] = None
    target_image_position: Optional[tuple[float, float]] = None
    object_relative_state: Optional[ObjectRelativeState] = None
    end_effector_state: Optional[Mapping[str, Any]] = None
    gripper_state: Optional[str] = None
    gripper_width_m: Optional[float] = None
    holding_state: Optional[str] = None
    contact_state: Optional[str] = None
    relevant_geometry: Mapping[str, Any] = field(default_factory=dict)
    last_action: Optional[str] = None
    last_expected_effect: Optional[Mapping[str, Any]] = None
    last_observed_effect: Optional[Mapping[str, Any]] = None
    uncertainty: Optional[Mapping[str, Any]] = None
    evidence_refs: tuple[str, ...] = ()
    observation_id: Optional[str] = None
    frame_id: Optional[int] = None
    observation_fresh: bool = False
    done: bool = False


class StateBuilder:
    """Translate observations into a single immutable canonical state value."""

    def initialize(self, task_id: Optional[str] = None) -> BeliefState:
        return BeliefState(task_id=task_id)

    def update(
        self,
        previous: BeliefState,
        observation: Any,
        *,
        action: Optional[str] = None,
        expected_effect: Optional[Mapping[str, Any]] = None,
        observed_effect: Optional[Mapping[str, Any]] = None,
    ) -> BeliefState:
        evidence = dict(getattr(observation, "evidence", {}) or {})
        proprioception = dict(getattr(observation, "proprioception", {}) or {})
        geometry = dict(evidence.get("relevant_geometry", {}) or {})
        object_relative = _object_relative_state(evidence.get("object_relative_state"))
        refs = tuple(str(ref) for ref in (getattr(observation, "evidence_refs", ()) or ()))
        return BeliefState(
            task_id=previous.task_id,
            step_id=previous.step_id + 1,
            stage=evidence.get("stage", previous.stage),
            target_identity=evidence.get("target_identity", previous.target_identity),
            target_confidence=evidence.get("target_confidence", previous.target_confidence),
            target_pose=evidence.get("target_pose", previous.target_pose),
            target_image_position=evidence.get("target_image_position", previous.target_image_position),
            object_relative_state=(object_relative if object_relative is not None
                                   else previous.object_relative_state),
            end_effector_state=proprioception.get("end_effector_state"),
            gripper_state=proprioception.get("gripper_state"),
            gripper_width_m=proprioception.get("gripper_width_m"),
            holding_state=evidence.get("holding_state", previous.holding_state),
            contact_state=evidence.get("contact_state", previous.contact_state),
            relevant_geometry=geometry,
            last_action=action if action is not None else previous.last_action,
            last_expected_effect=(dict(expected_effect) if expected_effect is not None
                                  else previous.last_expected_effect),
            last_observed_effect=(dict(observed_effect) if observed_effect is not None
                                  else previous.last_observed_effect),
            uncertainty=evidence.get("uncertainty"),
            evidence_refs=refs,
            observation_id=getattr(observation, "observation_id", None),
            frame_id=getattr(observation, "frame_id", None),
            observation_fresh=bool(getattr(observation, "fresh", False)),
            done=bool(getattr(observation, "done", False)),
        )


def _object_relative_state(value: Any) -> Optional[ObjectRelativeState]:
    if isinstance(value, ObjectRelativeState):
        return value
    if not isinstance(value, Mapping):
        return None

    def pair(name: str) -> Optional[tuple[float, float]]:
        raw = value.get(name)
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            return None
        try:
            result = tuple(float(item) for item in raw)
        except (TypeError, ValueError):
            return None
        return result if all(math.isfinite(item) for item in result) else None

    bbox_raw = value.get("target_bbox_xyxy")
    bbox = None
    if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
        try:
            candidate = tuple(float(item) for item in bbox_raw)
            if all(math.isfinite(item) for item in candidate):
                bbox = candidate
        except (TypeError, ValueError):
            pass
    try:
        score = float(value["target_quality_score"]) if value.get("target_quality_score") is not None else None
    except (TypeError, ValueError):
        score = None
    try:
        area = int(value["target_mask_area_px"]) if value.get("target_mask_area_px") is not None else None
    except (TypeError, ValueError):
        area = None
    try:
        error_norm = float(value["image_error_norm_px"]) if value.get("image_error_norm_px") is not None else None
    except (TypeError, ValueError):
        error_norm = None
    try:
        timestamp = float(value["evidence_timestamp"]) if value.get("evidence_timestamp") is not None else None
    except (TypeError, ValueError):
        timestamp = None
    return ObjectRelativeState(
        target_phrase=str(value.get("target_phrase", "")),
        target_visible=bool(value.get("target_visible", False)),
        target_quality_score=score,
        target_centroid_px=pair("target_centroid_px"),
        target_bbox_xyxy=bbox,
        target_mask_area_px=area,
        eef_projection_px=pair("eef_projection_px"),
        image_error_px=pair("image_error_px"),
        image_error_norm_px=error_norm,
        camera=str(value.get("camera", "agentview")),
        evidence_timestamp=timestamp,
        source_width=(int(value["source_width"]) if value.get("source_width") is not None else None),
        source_height=(int(value["source_height"]) if value.get("source_height") is not None else None),
        target_identity_status=str(value.get("target_identity_status", "UNANCHORED")),
        target_candidate_id=(str(value["target_candidate_id"])
                             if value.get("target_candidate_id") is not None else None),
        target_reference_point_px=pair("target_reference_point_px"),
        target_reference_valid=bool(value.get("target_reference_valid", False)),
        target_reference_camera=(str(value["target_reference_camera"])
                                 if value.get("target_reference_camera") is not None else None),
        target_reference_invalidation_reason=(
            str(value["target_reference_invalidation_reason"])
            if value.get("target_reference_invalidation_reason") is not None else None
        ),
    )

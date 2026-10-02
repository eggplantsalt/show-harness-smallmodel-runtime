"""The sole canonical state schema for Runtime V3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class BeliefState:
    task_id: Optional[str] = None
    step_id: int = 0
    stage: Optional[str] = None
    target_identity: Optional[str] = None
    target_confidence: Optional[float] = None
    target_pose: Optional[tuple[float, ...]] = None
    target_image_position: Optional[tuple[float, float]] = None
    end_effector_state: Optional[Mapping[str, Any]] = None
    gripper_state: Optional[str] = None
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
        refs = tuple(str(ref) for ref in (getattr(observation, "evidence_refs", ()) or ()))
        return BeliefState(
            task_id=previous.task_id,
            step_id=previous.step_id + 1,
            stage=evidence.get("stage", previous.stage),
            target_identity=evidence.get("target_identity", previous.target_identity),
            target_confidence=evidence.get("target_confidence", previous.target_confidence),
            target_pose=evidence.get("target_pose", previous.target_pose),
            target_image_position=evidence.get("target_image_position", previous.target_image_position),
            end_effector_state=proprioception.get("end_effector_state"),
            gripper_state=proprioception.get("gripper_state"),
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

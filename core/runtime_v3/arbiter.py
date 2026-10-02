"""The only Runtime V3 authority that can approve a physical action."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .options import PrimitiveCommand, RuntimeOption
from .selector import Selection
from .state import BeliefState


class DecisionKind(str, Enum):
    APPROVED = "APPROVED"
    REOBSERVE = "REOBSERVE"
    ABORT = "ABORT"
    INVALID_SELECTION = "INVALID_SELECTION"


@dataclass(frozen=True)
class ApprovedAction:
    option_id: str
    primitive: PrimitiveCommand
    expected_effect: Mapping[str, Any]
    state_step_id: int
    evidence_frame_id: int
    start_eef_position_xyz_m: tuple[float, float, float] | None = None
    workspace_z_bounds_m: tuple[float, float] | None = None
    _approval: object | None = None


@dataclass(frozen=True)
class ArbiterDecision:
    kind: DecisionKind
    action: ApprovedAction | None = None
    reason: str = ""


def _read_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
    return current


class Arbiter:
    def __init__(self) -> None:
        self.__seal = object()

    def is_approved(self, action: ApprovedAction) -> bool:
        return isinstance(action, ApprovedAction) and action._approval is self.__seal

    def authorize(
        self,
        state: BeliefState,
        options: Sequence[RuntimeOption],
        selection: Selection,
    ) -> ArbiterDecision:
        if selection.status == "INVALID_SELECTION" or selection.option_id == "INVALID_SELECTION":
            return ArbiterDecision(DecisionKind.INVALID_SELECTION, reason="selector output failed schema validation")
        if selection.option_id == "REOBSERVE":
            return ArbiterDecision(DecisionKind.REOBSERVE, reason="selector requested a fresh observation")
        if selection.option_id == "ABORT":
            return ArbiterDecision(DecisionKind.ABORT, reason="selector requested abort")
        matches = [option for option in options if option.option_id == selection.option_id]
        if len(matches) != 1:
            return ArbiterDecision(DecisionKind.INVALID_SELECTION, reason="selected option is absent or ambiguous")
        option = matches[0]
        if not state.observation_fresh or state.frame_id is None:
            return ArbiterDecision(DecisionKind.REOBSERVE, reason="current observation is stale")
        if option.evidence_frame_id is not None and option.evidence_frame_id != state.frame_id:
            return ArbiterDecision(DecisionKind.REOBSERVE, reason="option evidence is stale")
        if option.evidence and not set(option.evidence).issubset(state.evidence_refs):
            return ArbiterDecision(DecisionKind.REOBSERVE, reason="option evidence references are unavailable")
        if not self._preconditions_met(state, option.preconditions):
            return ArbiterDecision(DecisionKind.REOBSERVE, reason="option preconditions are not satisfied")
        primitive = option.primitive
        if primitive.max_steps != 1 or not 0 < primitive.max_duration_s <= 5.0:
            return ArbiterDecision(DecisionKind.ABORT, reason="primitive exceeds the V3 bounded execution contract")
        if primitive.kind not in {"move", "grasp", "release", "hold", "micro_motion"}:
            return ArbiterDecision(DecisionKind.ABORT, reason="unknown primitive kind")
        if primitive.kind == "move" and primitive.token not in {
            "MV_LEFT", "MV_RIGHT", "MV_FWD", "MV_BACK", "MV_UP", "MV_DOWN",
        }:
            return ArbiterDecision(DecisionKind.ABORT, reason="move primitive token is outside the atomic vocabulary")
        if primitive.kind != "move" and primitive.token is not None:
            return ArbiterDecision(DecisionKind.ABORT, reason="gripper/hold primitive cannot carry a motion token")
        start_position = None
        workspace_z_bounds = None
        if primitive.kind == "micro_motion":
            spec = primitive.micro_motion_spec
            if spec is None:
                return ArbiterDecision(DecisionKind.ABORT, reason="micro-motion requires an approved execution spec")
            if primitive.token is not None:
                return ArbiterDecision(DecisionKind.ABORT, reason="micro-motion direction is semantic, not a controller token")
            if not bool(state.relevant_geometry.get("workspace_valid")):
                return ArbiterDecision(DecisionKind.REOBSERVE, reason="workspace observation is not valid")
            raw_position = None
            if isinstance(state.end_effector_state, Mapping):
                raw_position = state.end_effector_state.get("position_xyz")
            try:
                position = tuple(float(value) for value in raw_position)
            except (TypeError, ValueError):
                position = ()
            raw_bounds = state.relevant_geometry.get("workspace_z_bounds_m")
            try:
                bounds = tuple(float(value) for value in raw_bounds)
            except (TypeError, ValueError):
                bounds = ()
            if (len(position) != 3 or len(bounds) != 2
                    or not all(math.isfinite(value) for value in (*position, *bounds))
                    or bounds[0] >= bounds[1]
                    or not bounds[0] <= position[2] <= bounds[1]):
                return ArbiterDecision(DecisionKind.REOBSERVE, reason="micro-motion needs finite EEF position and workspace bounds")
            start_position = position
            workspace_z_bounds = bounds
        elif primitive.micro_motion_spec is not None:
            return ArbiterDecision(DecisionKind.ABORT, reason="non-micro-motion primitive cannot carry a micro-motion spec")
        action = ApprovedAction(
            option_id=option.option_id,
            primitive=primitive,
            expected_effect=dict(option.expected_effect),
            state_step_id=state.step_id,
            evidence_frame_id=state.frame_id,
            start_eef_position_xyz_m=start_position,
            workspace_z_bounds_m=workspace_z_bounds,
            _approval=self.__seal,
        )
        return ArbiterDecision(DecisionKind.APPROVED, action=action)

    @staticmethod
    def _preconditions_met(state: BeliefState, conditions: Mapping[str, Any]) -> bool:
        for path, expected in conditions.items():
            if _read_path(state, path) != expected:
                return False
        return True

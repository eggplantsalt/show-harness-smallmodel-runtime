"""Small, task-agnostic placement spatial harness.

This module is deliberately provider-agnostic.  It turns metric evidence into
one stable placement relation and compiles that relation into one bounded
action.  It does not know object names, receptacle names, simulator poses, or
fixed image coordinates.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Optional

import numpy as np

from core.capabilities.camera_geometry import CameraCalibration, backproject_pixel_to_plane

from .control import (
    DONE,
    MOVE_BACK,
    MOVE_DOWN,
    MOVE_FWD,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_UP,
    STOP,
    ControlDecision,
)
from .types import (
    PlacementBelief,
    PlacementCandidate,
    PlacementRelation,
    SemanticPlacementAction,
)


def project_opening_at_plane(
    calibration: CameraCalibration,
    pixel_xy: Iterable[float],
    plane_z_m: float,
) -> Optional[np.ndarray]:
    """Back-project an opening point onto the measured rim plane.

    Keeping this helper explicit prevents callers from accidentally using the
    support/table plane for an elevated receptacle opening.
    """

    return backproject_pixel_to_plane(calibration, pixel_xy, float(plane_z_m))


def _finite_vector(value: Any, size: int) -> Optional[np.ndarray]:
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size < size or not np.all(np.isfinite(vector[:size])):
        return None
    return vector[:size]


class PlacementHysteresis:
    """Schmitt-style relation latch with consecutive-frame confirmation."""

    def __init__(self, *, enter_margin_m: float, exit_margin_m: float, confirm_frames: int = 2) -> None:
        self.enter_margin_m = max(0.0, float(enter_margin_m))
        self.exit_margin_m = max(self.enter_margin_m, float(exit_margin_m))
        self.confirm_frames = max(1, int(confirm_frames))
        self.reset()

    def reset(self) -> None:
        self.relation = PlacementRelation.UNKNOWN
        self._pending = PlacementRelation.UNKNOWN
        self._pending_count = 0

    def update(self, proposed: PlacementRelation, *, margin_m: Optional[float]) -> PlacementRelation:
        proposed = PlacementRelation(proposed)
        margin = float(margin_m) if margin_m is not None else None
        if self.relation in {PlacementRelation.ABOVE_ALIGNED, PlacementRelation.DESCENDING_CLEAR}:
            if proposed == PlacementRelation.ABOVE_UNALIGNED and margin is not None and margin > self.exit_margin_m:
                return self._confirm(proposed)
            if proposed in {PlacementRelation.RIM_CONTACT, PlacementRelation.LOST, PlacementRelation.UNKNOWN}:
                return self._confirm(proposed)
            return self.relation
        if proposed == PlacementRelation.ABOVE_ALIGNED and margin is not None and margin >= self.enter_margin_m:
            return self._confirm(proposed)
        return self._confirm(proposed)

    def _confirm(self, proposed: PlacementRelation) -> PlacementRelation:
        if proposed == self.relation:
            self._pending = proposed
            self._pending_count = 0
            return self.relation
        if proposed != self._pending:
            self._pending = proposed
            self._pending_count = 1
        else:
            self._pending_count += 1
        if self._pending_count >= self.confirm_frames:
            self.relation = proposed
            self._pending_count = 0
        return self.relation


class PlacementSpatialHarness:
    """Fuse stable metric placement evidence and compile one action."""

    def __init__(
        self,
        *,
        action_step_m: float = 0.01,
        enter_margin_m: float = 0.02,
        exit_margin_m: float = 0.01,
        confirm_frames: int = 2,
    ) -> None:
        self.action_step_m = max(0.0, float(action_step_m))
        self.hysteresis = PlacementHysteresis(
            enter_margin_m=enter_margin_m,
            exit_margin_m=exit_margin_m,
            confirm_frames=confirm_frames,
        )
        self.reset()

    def reset(self) -> None:
        self.hysteresis.reset()
        self.last_belief = PlacementBelief()
        self._last_candidate_key: Optional[tuple[Any, ...]] = None

    @staticmethod
    def classify(
        *,
        residual_world: Optional[Iterable[float]],
        uncertainty_m: Optional[Iterable[float]],
        containment_margin_m: Optional[float],
        rim_clearance_m: Optional[float],
        holding: Optional[bool],
        descending: bool = False,
        contact: bool = False,
        lost: bool = False,
        seated: bool = False,
        action_step_m: float = 0.01,
    ) -> PlacementRelation:
        if lost:
            return PlacementRelation.LOST
        if seated and holding:
            return PlacementRelation.SEATED_HELD
        if contact:
            return PlacementRelation.RIM_CONTACT
        residual = _finite_vector(residual_world, 3)
        uncertainty = _finite_vector(uncertainty_m, 3)
        margin = float(containment_margin_m) if containment_margin_m is not None else None
        if residual is None or margin is None:
            return PlacementRelation.UNKNOWN
        sigma = float(np.linalg.norm(uncertainty)) if uncertainty is not None else 0.0
        aligned = margin >= 2.0 * sigma + 0.5 * max(0.0, float(action_step_m))
        if not aligned:
            return PlacementRelation.ABOVE_UNALIGNED
        if descending:
            return PlacementRelation.DESCENDING_CLEAR
        return PlacementRelation.ABOVE_ALIGNED

    def update(self, belief: PlacementBelief) -> PlacementBelief:
        """Apply relation hysteresis while preserving the provider evidence."""

        candidate_key = (
            belief.instance_id,
            belief.grasp_epoch,
            belief.selected_candidate.candidate_id if belief.selected_candidate else None,
        )
        if candidate_key != self._last_candidate_key:
            self.hysteresis.reset()
            self._last_candidate_key = candidate_key
        stable = self.hysteresis.update(
            belief.relation,
            margin_m=belief.containment_margin_m,
        )
        result = replace(
            belief,
            relation=stable,
            diagnostics={
                **belief.diagnostics,
                "hysteresis_relation": stable.value,
                "candidate_key": list(candidate_key),
            },
        )
        self.last_belief = result
        return result

    def compile_action(
        self,
        *,
        relation: PlacementRelation,
        residual_world: Optional[Iterable[float]] = None,
        route_candidates: Optional[Iterable[dict[str, Any]]] = None,
    ) -> ControlDecision:
        """Compile a semantic relation into one signed, bounded atom."""

        relation = PlacementRelation(relation)
        if relation == PlacementRelation.ABOVE_ALIGNED:
            return ControlDecision(MOVE_DOWN, "placement relation is above and aligned; descend", None)
        if relation == PlacementRelation.DESCENDING_CLEAR:
            return ControlDecision(MOVE_DOWN, "descending with clear rim evidence", None)
        if relation == PlacementRelation.RIM_CONTACT:
            return ControlDecision(MOVE_UP, "rim contact requires reversible clearance", None)
        if relation in {PlacementRelation.SEATED_HELD, PlacementRelation.RELEASED_STABLE}:
            return ControlDecision(DONE, "seating evidence is ready for independent verification", None)
        if relation == PlacementRelation.LOST:
            return ControlDecision(STOP, "held instance was lost", None, "LOST_HOLD")
        if relation == PlacementRelation.UNKNOWN:
            return ControlDecision(STOP, "placement evidence is unknown; probe or re-observe", None)
        valid = [item for item in (route_candidates or ()) if isinstance(item, dict) and item.get("token")]
        if valid:
            valid.sort(key=lambda item: float(item.get("cosine", 0.0) or 0.0), reverse=True)
            if float(valid[0].get("cosine", 0.0) or 0.0) > 0.0:
                return ControlDecision(str(valid[0]["token"]).upper(), "signed metric placement residual", None)
        if relation == PlacementRelation.ABOVE_UNALIGNED:
            # An unaligned relation with no remaining horizontal residual is a
            # footprint-confidence boundary, not permission to freeze.  The
            # generic next atom is a bounded descent; the runtime will obtain
            # fresh evidence and can still recover if contact is observed.
            try:
                residual = np.asarray(tuple(residual_world), dtype=float).reshape(-1)
                horizontal_residual = float(np.linalg.norm(residual[:2]))
            except (TypeError, ValueError, IndexError):
                horizontal_residual = float("inf")
            if horizontal_residual <= self.action_step_m:
                return ControlDecision(
                    MOVE_DOWN,
                    "above relation has no remaining signed horizontal correction; descend for fresh evidence",
                    None,
                )
        return ControlDecision(STOP, "unaligned placement has no signed route correction", None)

    @staticmethod
    def semantic_action_for_relation(
        relation: PlacementRelation,
        *,
        has_signed_candidate: bool = False,
    ) -> SemanticPlacementAction:
        """Expose the small-agent choice without exposing signed motion tokens."""
        relation = PlacementRelation(relation)
        if relation == PlacementRelation.ABOVE_UNALIGNED:
            return (
                SemanticPlacementAction.CORRECT_LATERAL
                if has_signed_candidate
                else SemanticPlacementAction.PROBE
            )
        if relation in {PlacementRelation.ABOVE_ALIGNED, PlacementRelation.DESCENDING_CLEAR}:
            return SemanticPlacementAction.DESCEND
        if relation == PlacementRelation.RIM_CONTACT:
            return SemanticPlacementAction.RECOVER_CLEAR
        if relation in {PlacementRelation.SEATED_HELD, PlacementRelation.RELEASED_STABLE}:
            return SemanticPlacementAction.READY_TO_VERIFY
        return SemanticPlacementAction.PROBE

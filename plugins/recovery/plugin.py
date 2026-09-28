"""Recovery tool: conservative mechanical grasp-failure detection.

This tool is deliberately narrow. It does not inspect images, call a VLM, or own any
motion primitive. It watches the measured gripper width at stage boundaries and tells
the runner when a closed gripper is empty/lost, so the runner can reopen the gripper
and roll the active goal back to the relevant grasp stage.

The controller VLM owns the semantic decision to GRASP. This plugin only rejects a
clearly empty mechanical close (near-zero finger width) or a later clearly lost hold.
It must not require every object to occupy the current scene's holding-width band:
different object thicknesses legitimately produce different closed widths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from plugins.prompt_text import fragment


def _fragment(section: str) -> str:
    """This tool's prompt text lives in the co-located recovery.txt (the context
    wrapper and the per-event notes)."""
    return fragment(__file__, "recovery.txt", section)


RELEASE_TOKEN = "RELEASE"
GRASP_TOKEN = "GRASP"
STOP_TOKEN = "STOP"

@dataclass(frozen=True)
class RecoveryDecision:
    """A runner-level recovery request.

    ``token`` is an optional action override for the current step. ``release`` requests
    an immediate release after the just-executed token. ``rollback_index`` moves the
    active subgoal pointer before the next VLM decision.
    """

    event: str
    reason: str
    token: str | None = None
    release: bool = False
    rollback_index: int | None = None
    block_done: bool = False
    reset_history: bool = False
    grasp_empty: bool = False
    grasp_unverified: bool = False
    prompt_note: str = ""


class RecoveryPlugin:
    """Classify measured gripper width and request recovery interventions."""

    def __init__(
        self,
        enabled: bool = True,
        empty_width_m: float = 0.005,
        open_width_m: float = 0.06,
    ) -> None:
        self.enabled = bool(enabled)
        self.empty_width_m = max(0.0, float(empty_width_m))
        # Kept as a compatibility input for old robot configs. It is intentionally
        # not used to decide whether an object is held: that is a visual-agent task.
        self.open_width_m = max(self.empty_width_m, float(open_width_m))

    def reset(self) -> None:
        """Compatibility hook; grasp semantics are not episode-calibrated."""
        return None

    def render_prompt_context(self, note: str) -> str:
        """Return a short controller context line for the last recovery event."""
        if not self.enabled:
            return ""
        text = " ".join(str(note or "").split())
        return _fragment("context").replace("{note}", text) if text else ""

    def phase_from_width(self, width_m: Any) -> str:
        """Classify only the conservative ``empty`` vs ``nonempty`` mechanical fact."""
        try:
            width = float(width_m)
        except (TypeError, ValueError):
            return "unknown"
        if width <= self.empty_width_m:
            return "empty"
        return "nonempty"

    def clearly_empty(self, width_m: Any) -> bool:
        """Return only the conservative mechanical empty-close predicate."""
        return self.phase_from_width(width_m) == "empty"

    def arbitrate_transport_holding(
        self,
        decision: RecoveryDecision | None,
        *,
        stage: str,
        visual_holding_state: str,
    ) -> tuple[RecoveryDecision | None, dict[str, Any] | None]:
        """Reject a width-only lost-hold claim until temporal vision confirms LOST.

        Finger aperture is an auxiliary robot signal, not an object occupancy
        classifier.  During the route-aware TRANSPORT loop the visual holding
        arbiter has multi-frame co-motion evidence and must therefore win over a
        single near-zero width sample.  A later visual ``LOST`` decision is not
        suppressed and follows the normal release/reacquisition path.
        """
        if (
            decision is not None
            and str(stage or "").upper() == "TRANSPORT"
            and str(decision.event or "").lower() == "lost_grasp"
            and str(visual_holding_state or "").upper() != "LOST"
        ):
            return None, {
                "mechanical_event": decision.event,
                "mechanical_reason": decision.reason,
                "visual_holding_state": str(visual_holding_state or "UNKNOWN").upper(),
                "accepted": False,
                "reason": "width_only_loss_waiting_for_temporal_visual_confirmation",
            }
        return decision, None

    def agent_grasp_verification(
        self,
        *,
        verdict: str,
        current_index: int,
        subgoals: Sequence[Any],
        reasoning: str = "",
    ) -> RecoveryDecision | None:
        """Translate one visual verifier verdict into a bounded runner event.

        YES leaves the Agent's GRASP decision untouched. NO requests the normal
        release-and-retry path. UNKNOWN keeps the GRASP stage active and informs the
        next controller decision; it is never silently converted into empty.
        """
        normalized = str(verdict or "UNKNOWN").strip().upper()
        if normalized == "YES":
            return None
        if normalized == "NO":
            rollback = _nearest_reacquire_index(subgoals, current_index)
            return RecoveryDecision(
                event="agent_rejected_grasp",
                reason=(
                    "visual grasp verifier rejected the close"
                    + (f": {reasoning}" if reasoning else "")
                ),
                release=True,
                rollback_index=rollback,
                block_done=True,
                reset_history=True,
                grasp_empty=True,
                grasp_unverified=True,
                prompt_note=_prompt_note("empty_grasp"),
            )
        return RecoveryDecision(
            event="agent_grasp_unknown",
            reason=(
                "visual grasp verifier could not confirm the close"
                + (f": {reasoning}" if reasoning else "")
            ),
            block_done=True,
            grasp_unverified=True,
            prompt_note=_fragment("note_unverified_grasp"),
        )

    def before_decision(
        self,
        *,
        current_index: int,
        subgoals: Sequence[Any],
        measured_width_m: Any,
        gripper_closed: bool,
    ) -> RecoveryDecision | None:
        """Intervene before a VLM call based on the closed gripper's measured width.

        empty   -> reopen + roll back to the grasp stage (the close caught nothing).
        nonempty/unknown -> no semantic intervention; the Agent owns the visual
        judgment of whether the target is actually held.
        """
        if not self.enabled:
            return None
        if not gripper_closed:
            return None
        phase = self.phase_from_width(measured_width_m)

        if phase == "empty":
            rollback = _nearest_reacquire_index(subgoals, current_index)
            if rollback is None:
                return None
            stage = (
                _motion(subgoals[current_index])
                if 0 <= current_index < len(subgoals)
                else ""
            )
            event = "empty_grasp" if stage == GRASP_TOKEN else "lost_grasp"
            reason = (
                f"closed gripper width {float(measured_width_m):.4f}m <= "
                f"empty threshold {self.empty_width_m:.4f}m"
            )
            return RecoveryDecision(
                event=event,
                reason=reason,
                token=RELEASE_TOKEN,
                rollback_index=rollback,
                block_done=True,
                reset_history=True,
                grasp_empty=True,
                grasp_unverified=True,
                prompt_note=_prompt_note(event),
            )

        # Any non-empty width is deliberately left to the visual Agent; object
        # thickness is not encoded here.
        return None

    def after_step(
        self,
        *,
        token: str,
        result: Any,
        current_index: int,
        subgoals: Sequence[Any],
        measured_width_m: Any,
        subgoal_done: bool,
        gripper_closed: bool,
    ) -> RecoveryDecision | None:
        """Validate a just-executed step before the runner advances subgoals."""
        if not self.enabled:
            return None
        token = str(token or "").strip().upper()
        rollback = _nearest_reacquire_index(subgoals, current_index)
        if rollback is None:
            return None

        if bool(getattr(result, "grasp_empty", False)):
            return RecoveryDecision(
                event="empty_grasp",
                reason=str(getattr(result, "note", "") or "gripper closed empty"),
                rollback_index=rollback,
                block_done=True,
                reset_history=True,
                grasp_empty=True,
                grasp_unverified=True,
                prompt_note=_prompt_note("empty_grasp"),
            )

        stage = (
            _motion(subgoals[current_index])
            if 0 <= current_index < len(subgoals)
            else ""
        )
        phase = self.phase_from_width(measured_width_m)

        if stage == GRASP_TOKEN and subgoal_done and phase == "empty":
            reason = (
                "grasp DONE rejected because the measured width is a clearly empty close"
                f" (width={_fmt_width(measured_width_m)})"
            )
            return RecoveryDecision(
                event="unverified_grasp",
                reason=reason,
                release=bool(gripper_closed),
                rollback_index=current_index,
                block_done=True,
                reset_history=True,
                grasp_empty=(phase == "empty"),
                grasp_unverified=True,
                prompt_note=_prompt_note(
                    "empty_grasp" if phase == "empty" else "unverified_grasp"
                ),
            )

        if stage != GRASP_TOKEN and gripper_closed and phase == "empty":
            reason = (
                f"closed gripper became empty during {stage or 'post-grasp stage'} "
                f"(width={_fmt_width(measured_width_m)})"
            )
            return RecoveryDecision(
                event="lost_grasp",
                reason=reason,
                release=True,
                rollback_index=rollback,
                block_done=True,
                reset_history=True,
                grasp_empty=True,
                grasp_unverified=True,
                prompt_note=_prompt_note("lost_grasp"),
            )

        return None


def _nearest_reacquire_index(subgoals: Sequence[Any], current_index: int) -> int | None:
    """Return the latest APPROACH before the active grasp, when available.

    A lost hold means the object may have moved or fallen. Repeating GRASP at the
    old pose can therefore loop forever; re-entering APPROACH lets the visual Agent
    reacquire the current object location. If a plan has no explicit APPROACH, fall
    back to the nearest GRASP for compatibility with older plans.
    """
    upper = min(max(int(current_index), 0), len(subgoals) - 1)
    grasp_index = None
    for idx in range(upper, -1, -1):
        if _motion(subgoals[idx]) == GRASP_TOKEN:
            grasp_index = idx
            break
    if grasp_index is not None:
        for idx in range(grasp_index - 1, -1, -1):
            if _motion(subgoals[idx]) == "APPROACH":
                return idx
        return grasp_index
    return 0 if subgoals else None


def _motion(subgoal: Any) -> str:
    return str(getattr(subgoal, "motion", "") or "").strip().upper()


def _fmt_width(value: Any) -> str:
    try:
        return f"{float(value):.4f}m"
    except (TypeError, ValueError):
        return "unknown"


def _prompt_note(event: str) -> str:
    if event == "lost_grasp":
        return _fragment("note_lost_grasp")
    if event == "unverified_grasp":
        return _fragment("note_unverified_grasp")
    return _fragment("note_empty_grasp")

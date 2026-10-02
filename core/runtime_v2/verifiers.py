"""Evidence-only transition verifiers used by the runtime state store."""
from __future__ import annotations

from typing import Any, Optional

from .types import TransitionVerdict, Verdict


def verify_hold(
    *,
    frame_id: Optional[int],
    gripper_closed: bool,
    holding_evidence: Optional[dict[str, Any]],
    target_visible: bool,
    width_clearly_empty: bool,
) -> TransitionVerdict:
    evidence = dict(holding_evidence or {})
    state = str(evidence.get("state") or evidence.get("held_assessment") or "").upper()
    comotion = evidence.get("hold_comotion_score")
    try:
        comotion_value = float(comotion) if comotion is not None else None
    except (TypeError, ValueError):
        comotion_value = None

    positive = bool(
        gripper_closed
        and not width_clearly_empty
        and (state in {"HELD", "YES"} or (comotion_value is not None and comotion_value >= 0.6))
    )
    negative = bool(state in {"LOST", "NOT_HELD", "NO"})
    if positive:
        verdict, reason = Verdict.PASS, "multi-source holding evidence"
    elif negative and target_visible:
        verdict, reason = Verdict.FAIL, "visual holding evidence reports loss"
    else:
        verdict, reason = Verdict.UNKNOWN, "holding evidence incomplete or conflicting"
    return TransitionVerdict(verdict, reason, evidence, frame_id, frame_id is not None)


def verify_seated(
    *,
    frame_id: Optional[int],
    alignment: Optional[dict[str, Any]],
    holding_verdict: Verdict,
    contact_or_stall: bool,
) -> TransitionVerdict:
    evidence = dict(alignment or {})
    relation = str(evidence.get("relation") or "UNKNOWN").upper()
    if relation in {"SEATED_HELD", "RELEASED_STABLE"} and holding_verdict == Verdict.PASS:
        return TransitionVerdict(
            Verdict.PASS,
            "shared placement belief reports seated held payload",
            evidence,
            frame_id,
            frame_id is not None,
        )
    aligned = bool(evidence.get("aligned", False))
    rim_risk = bool(evidence.get("rim_contact_risk", False))
    descent_ready = bool(
        evidence.get("clearance_progress_ready", False)
        or evidence.get("opening_error_normalized", 1.0) in (0, 0.0)
    )
    if aligned and not rim_risk and holding_verdict == Verdict.PASS and (
        descent_ready or contact_or_stall
    ):
        verdict, reason = Verdict.PASS, "aligned inside opening with support evidence"
    elif rim_risk:
        verdict, reason = Verdict.FAIL, "receptacle rim contact risk"
    else:
        verdict, reason = Verdict.UNKNOWN, "placement support not yet established"
    return TransitionVerdict(verdict, reason, evidence, frame_id, frame_id is not None)

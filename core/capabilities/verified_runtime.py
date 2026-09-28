"""Verified physical-state runtime for lightweight embodied agents.

Phase 1 scope
-------------
This runtime externalizes only local APPROACH alignment. It deliberately does
NOT choose task goals, targets, grasp semantics, transport goals, or task
completion.

The high-level Agent still decides *what* to do. The runtime makes a bounded
physical option such as ALIGN(target) reliable using:

1. host-observed physical state,
2. calibrated geometry as an initial action prior,
3. actual post-action visual feedback,
4. a small online action-effect model.

Simulator object pose, task success state, BDDL coordinates, and task-specific
trajectories are never consumed here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


MOVE_LEFT = "MV_LEFT"
MOVE_RIGHT = "MV_RIGHT"
MOVE_FWD = "MV_FWD"
MOVE_BACK = "MV_BACK"

LATERAL_ACTIONS = (MOVE_LEFT, MOVE_RIGHT)
DEPTH_ACTIONS = (MOVE_FWD, MOVE_BACK)

OPPOSITE = {
    MOVE_LEFT: MOVE_RIGHT,
    MOVE_RIGHT: MOVE_LEFT,
    MOVE_FWD: MOVE_BACK,
    MOVE_BACK: MOVE_FWD,
}


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _error_tuple(value: Any) -> Optional[tuple[float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class PhysicalState:
    """Canonical read-only physical state used by the Phase-1 runtime."""

    frame_id: Optional[int]
    stage: str
    target: str
    camera: str

    visible: bool
    confidence: float
    source: str

    target_minus_eef_px: Optional[tuple[float, float]]
    alignment_error_px: Optional[float]
    alignment_ready: bool

    eef_height_m: Optional[float]

    horizontal_candidate: Optional[str]
    vertical_candidate: Optional[str]

    @classmethod
    def from_evidence(
        cls,
        evidence: dict[str, Any] | None,
        *,
        stage: str,
    ) -> "PhysicalState":
        evidence = evidence if isinstance(evidence, dict) else {}
        geometry = evidence.get("geometry")
        geometry = geometry if isinstance(geometry, dict) else {}

        error = _error_tuple(geometry.get("target_minus_eef_px"))
        error_norm = (
            max(abs(error[0]), abs(error[1]))
            if error is not None
            else None
        )

        candidates = geometry.get("calibrated_correction_candidates")
        if not isinstance(candidates, dict):
            candidates = geometry.get("correction_candidates")
        if not isinstance(candidates, dict):
            candidates = {}

        frame_raw = evidence.get("frame_id")
        try:
            frame_id = int(frame_raw) if frame_raw is not None else None
        except (TypeError, ValueError):
            frame_id = None

        horizontal = str(candidates.get("horizontal") or "").strip().upper()
        vertical = str(candidates.get("vertical") or "").strip().upper()

        return cls(
            frame_id=frame_id,
            stage=str(stage or "").strip().upper(),
            target=str(evidence.get("target") or ""),
            camera=str(evidence.get("camera") or ""),
            visible=bool(evidence.get("visible", False)),
            confidence=float(evidence.get("confidence", 0.0) or 0.0),
            source=str(evidence.get("source") or "unknown"),
            target_minus_eef_px=error,
            alignment_error_px=error_norm,
            alignment_ready=bool(geometry.get("alignment_ready", False)),
            eef_height_m=_float_or_none(geometry.get("eef_height_m")),
            horizontal_candidate=horizontal or None,
            vertical_candidate=vertical or None,
        )


class VerifiedEmbodiedRuntime:
    """Bounded external computation for physical closed-loop execution.

    Phase 1 implements ALIGN(target) during APPROACH.

    Important authority boundary:
      - Agent owns task semantics / target / stage.
      - Runtime owns bounded local alignment *inside* APPROACH.
      - Existing runner guards still own execution safety.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        mode: str = "shadow",
        confidence_threshold: float = 0.4,
        alignment_px: float = 13.5,
        final_height_max_m: Optional[float] = None,
        min_progress_px: float = 1.0,
        wrong_direction_px: float = 1.5,
        stall_limit: int = 3,
        max_option_steps: int = 40,
        effect_alpha: float = 0.5,
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = str(mode or "shadow").lower()
        if self.mode not in {"shadow", "active"}:
            raise ValueError("verified_runtime.mode must be shadow or active")

        self.confidence_threshold = float(confidence_threshold)
        self.alignment_px = max(1.0, float(alignment_px))
        self.final_height_max_m = (
            None
            if final_height_max_m is None
            else float(final_height_max_m)
        )

        self.min_progress_px = max(0.0, float(min_progress_px))
        self.wrong_direction_px = max(
            self.min_progress_px,
            float(wrong_direction_px),
        )
        self.stall_limit = max(1, int(stall_limit))
        self.max_option_steps = max(1, int(max_option_steps))
        self.effect_alpha = min(1.0, max(0.05, float(effect_alpha)))

        self.reset()

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
    ) -> Optional["VerifiedEmbodiedRuntime"]:
        section = cfg.get("verified_runtime")
        if not isinstance(section, dict) or not bool(
            section.get("enabled", False)
        ):
            return None

        capabilities = cfg.get("capabilities")
        capabilities = (
            capabilities if isinstance(capabilities, dict) else {}
        )

        return cls(
            enabled=True,
            mode=str(section.get("mode", "shadow")),
            confidence_threshold=float(
                section.get(
                    "confidence_threshold",
                    capabilities.get("confidence_threshold", 0.4),
                )
            ),
            alignment_px=float(
                section.get(
                    "alignment_px",
                    capabilities.get("alignment_ready_px", 13.5),
                )
            ),
            final_height_max_m=section.get(
                "final_height_max_m",
                capabilities.get("approach_completion_max_height_m"),
            ),
            min_progress_px=float(
                section.get("min_progress_px", 1.0)
            ),
            wrong_direction_px=float(
                section.get("wrong_direction_px", 1.5)
            ),
            stall_limit=int(section.get("stall_limit", 3)),
            max_option_steps=int(
                section.get("max_option_steps", 40)
            ),
            effect_alpha=float(section.get("effect_alpha", 0.5)),
        )

    def reset(self) -> None:
        # action -> observed [delta_error_x, delta_error_y]
        self.action_effects: dict[str, tuple[float, float]] = {}

        self._last_command: Optional[str] = None
        self._last_error: Optional[tuple[float, float]] = None
        self._last_identity: Optional[str] = None

        self._stall_count = 0
        self._option_steps = 0
        self._last_transition: dict[str, Any] = {
            "status": "UNINITIALIZED"
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "scope": "APPROACH_ALIGN_PHASE1",
            "confidence_threshold": self.confidence_threshold,
            "alignment_px": self.alignment_px,
            "final_height_max_m": self.final_height_max_m,
            "min_progress_px": self.min_progress_px,
            "wrong_direction_px": self.wrong_direction_px,
            "stall_limit": self.stall_limit,
            "max_option_steps": self.max_option_steps,
            "effect_alpha": self.effect_alpha,
        }

    @staticmethod
    def _identity(state: PhysicalState) -> str:
        return f"{state.stage}::{state.target}::{state.camera}"

    @staticmethod
    def _norm(error: tuple[float, float]) -> float:
        return max(abs(float(error[0])), abs(float(error[1])))

    def _update_effect_model(
        self,
        state: PhysicalState,
        previous_action: Optional[str],
    ) -> None:
        current = state.target_minus_eef_px
        previous_action = str(previous_action or "").strip().upper()

        if (
            current is None
            or self._last_error is None
            or self._last_command is None
            or previous_action != self._last_command
            or self._last_identity != self._identity(state)
        ):
            return

        delta = (
            float(current[0]) - float(self._last_error[0]),
            float(current[1]) - float(self._last_error[1]),
        )

        old = self.action_effects.get(previous_action)
        if old is None:
            updated = delta
        else:
            a = self.effect_alpha
            updated = (
                (1.0 - a) * old[0] + a * delta[0],
                (1.0 - a) * old[1] + a * delta[1],
            )

        self.action_effects[previous_action] = updated

        before = self._norm(self._last_error)
        after = self._norm(current)
        improvement = before - after

        if improvement >= self.min_progress_px:
            status = "IMPROVING"
            self._stall_count = 0
        elif improvement <= -self.wrong_direction_px:
            status = "WRONG_DIRECTION"
            self._stall_count += 1
        else:
            status = "NO_PROGRESS"
            self._stall_count += 1

        self._last_transition = {
            "status": status,
            "action": previous_action,
            "error_before_px": [
                round(self._last_error[0], 2),
                round(self._last_error[1], 2),
            ],
            "error_after_px": [
                round(current[0], 2),
                round(current[1], 2),
            ],
            "delta_error_px": [
                round(delta[0], 2),
                round(delta[1], 2),
            ],
            "improvement_px": round(improvement, 2),
            "stall_count": self._stall_count,
        }

    def _predicted_abs_error(
        self,
        token: str,
        *,
        axis: int,
        current_error: float,
    ) -> Optional[float]:
        effect = self.action_effects.get(token)
        if effect is None:
            return None
        return abs(float(current_error) + float(effect[axis]))

    def _choose_axis_action(
        self,
        *,
        pair: tuple[str, str],
        axis: int,
        current_error: float,
        calibrated_prior: Optional[str],
    ) -> tuple[str, str]:
        prior = str(calibrated_prior or "").strip().upper()
        if prior not in pair:
            prior = ""

        # 1. Once we have measured both actions, trust observed embodiment
        #    response rather than the textual camera convention.
        predictions = []
        for token in pair:
            predicted = self._predicted_abs_error(
                token,
                axis=axis,
                current_error=current_error,
            )
            if predicted is not None:
                predictions.append((predicted, token))

        if len(predictions) == 2:
            predictions.sort(key=lambda item: item[0])
            return predictions[0][1], "observed_action_effect"

        # 2. If the prior action has already been observed to move the error
        #    in the wrong direction, reverse it.
        if prior:
            predicted = self._predicted_abs_error(
                prior,
                axis=axis,
                current_error=current_error,
            )
            if (
                predicted is not None
                and predicted > abs(current_error)
            ):
                return OPPOSITE[prior], "prior_rejected_by_feedback"

            return prior, "calibrated_prior"

        # 3. If one action has been measured, use or invert it according to
        #    its observed effect.
        if len(predictions) == 1:
            predicted, token = predictions[0]
            if predicted < abs(current_error):
                return token, "single_observed_effect"
            return OPPOSITE[token], "single_effect_inverted"

        # 4. Last-resort bounded probe. No task/object coordinate is used.
        return pair[0], "bounded_probe"

    def _clear_pending_command(self) -> None:
        self._last_command = None
        self._last_error = None
        self._last_identity = None

    def observe(
        self,
        *,
        stage: str,
        evidence: dict[str, Any] | None,
        previous_action: Optional[str],
    ) -> dict[str, Any]:
        state = PhysicalState.from_evidence(
            evidence,
            stage=stage,
        )

        # First use the fresh observation to score the action that actually
        # executed on the preceding control step.
        self._update_effect_model(state, previous_action)

        result: dict[str, Any] = {
            "enabled": self.enabled,
            "mode": self.mode,
            "option": "ALIGN",
            "takeover": False,
            "suggested_token": None,
            "action_token": None,
            "status": "INACTIVE",
            "reason": "",
            "physical_state": asdict(state),
            "transition": dict(self._last_transition),
            "stall_count": self._stall_count,
            "option_steps": self._option_steps,
            "action_effects": {
                key: [round(value[0], 3), round(value[1], 3)]
                for key, value in self.action_effects.items()
            },
        }

        if not self.enabled:
            result["reason"] = "runtime_disabled"
            self._clear_pending_command()
            return result

        if state.stage != "APPROACH":
            result["reason"] = "stage_not_approach"
            self._clear_pending_command()
            self._option_steps = 0
            self._stall_count = 0
            return result

        if (
            not state.visible
            or state.confidence < self.confidence_threshold
            or state.source == "occlusion_memory"
        ):
            result["status"] = "TARGET_LOST"
            result["reason"] = "no_fresh_visible_target"
            self._clear_pending_command()
            return result

        error = state.target_minus_eef_px
        if error is None:
            result["status"] = "UNKNOWN"
            result["reason"] = "alignment_geometry_missing"
            self._clear_pending_command()
            return result

        if state.alignment_error_px is not None and (
            state.alignment_error_px <= self.alignment_px
        ):
            result["status"] = "ALIGNED"
            result["reason"] = "alignment_within_threshold"
            self._clear_pending_command()
            return result

        if self._option_steps >= self.max_option_steps:
            result["status"] = "OPTION_TIMEOUT"
            result["reason"] = "bounded_align_budget_exhausted"
            self._clear_pending_command()
            return result

        dx, dy = error

        axis: Optional[int] = None
        pair: Optional[tuple[str, str]] = None
        prior: Optional[str] = None
        axis_name = ""

        # High above the final approach band, only externalize the
        # unambiguous lateral image alignment. Screen vertical error can
        # be dominated by height/parallax, so do NOT turn it into a blind
        # FWD/BACK command.
        if abs(dx) > self.alignment_px:
            axis = 0
            pair = LATERAL_ACTIONS
            prior = state.horizontal_candidate
            axis_name = "horizontal"

        else:
            height_allows_depth = bool(
                self.final_height_max_m is None
                or (
                    state.eef_height_m is not None
                    and state.eef_height_m
                    <= self.final_height_max_m
                )
            )

            if abs(dy) > self.alignment_px and height_allows_depth:
                axis = 1
                pair = DEPTH_ACTIONS
                prior = state.vertical_candidate
                axis_name = "depth"
            elif abs(dy) > self.alignment_px:
                result["status"] = "WAITING_FOR_HEIGHT_TRANSITION"
                result["reason"] = (
                    "lateral_aligned_but_vertical_pixel_error_is_"
                    "ambiguous_above_final_height_band"
                )
                self._clear_pending_command()
                return result
            else:
                result["status"] = "ALIGNED"
                result["reason"] = "per_axis_alignment_within_threshold"
                self._clear_pending_command()
                return result

        assert axis is not None and pair is not None

        action, action_source = self._choose_axis_action(
            pair=pair,
            axis=axis,
            current_error=error[axis],
            calibrated_prior=prior,
        )

        # Repeated failure is not allowed to become another 200-step loop.
        # After a bounded number of non-improving observations, yield back
        # to the high-level Agent/recovery machinery.
        if self._stall_count >= 2 * self.stall_limit:
            result["status"] = "NO_PROGRESS"
            result["reason"] = (
                "local_align_failed_to_make_progress; "
                "yielding_to_high_level_agent"
            )
            self._clear_pending_command()
            return result

        # If the immediately preceding action was clearly wrong, explicitly
        # avoid blindly repeating the same direction.
        if (
            self._last_transition.get("status") == "WRONG_DIRECTION"
            and action == self._last_transition.get("action")
            and action in OPPOSITE
        ):
            action = OPPOSITE[action]
            action_source = "wrong_direction_reversal"

        self._option_steps += 1

        result.update(
            {
                "takeover": self.mode == "active",
                "suggested_token": action,
                "action_token": (
                    action if self.mode == "active" else None
                ),
                "status": "ALIGNING",
                "reason": (
                    f"{axis_name}_alignment_via_{action_source}"
                ),
                "axis": axis_name,
                "action_source": action_source,
                "option_steps": self._option_steps,
            }
        )

        # Stored only so the NEXT fresh observation can evaluate what the
        # action actually did. Learning is committed only if previous_action
        # confirms that this token was in fact executed.
        self._last_command = action
        self._last_error = error
        self._last_identity = self._identity(state)

        return result

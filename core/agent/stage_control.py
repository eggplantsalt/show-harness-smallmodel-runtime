from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.v0_types import SkillContext


class Controller:
    """One VLM call per step that returns the executed base/grasp token."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    @property
    def last_prompt(self) -> str:
        """The most recent fully-rendered controller prompt (for periodic logging)."""
        return getattr(self.agent, "last_prompt", "")

    def decide(
        self,
        ctx: SkillContext,
        recent_moves: str,
        previous_direction: str,
        gripper_state: str,
        recovery_context: str = "",
        prev_agentview: Any = None,
        capability_context: str = "",
    ):
        return self.agent.decide(
            task=ctx.task,
            subgoal=ctx.subgoal.to_prompt_dict(),
            recent_moves=recent_moves,
            previous_direction=previous_direction,
            gripper_state=gripper_state,
            agentview_image=ctx.agentview,
            wrist_image=ctx.wrist,
            # Frame captured BEFORE the previous action executed (action-ablation
            # blind review); None everywhere else, incl. the sim runner.
            prev_agentview_image=prev_agentview,
            proprio=ctx.proprio,
            recovery_context=recovery_context,
            capability_context=capability_context,
            debug=ctx.debug,
        )

    def verify_grasp(self, **kwargs):
        """Delegate the one-shot post-close visual check to the role Agent."""
        verifier = getattr(self.agent, "verify_grasp", None)
        if not callable(verifier):
            return None
        return verifier(**kwargs)

    def verify_place(self, **kwargs):
        """Delegate the one-shot low-placement visual check to the role Agent."""
        verifier = getattr(self.agent, "verify_place", None)
        if not callable(verifier):
            return None
        return verifier(**kwargs)

    def review_place_alignment(self, **kwargs):
        """Delegate the pre-placement alignment review to the role Agent."""
        reviewer = getattr(self.agent, "review_place_alignment", None)
        if not callable(reviewer):
            return None
        return reviewer(**kwargs)


@dataclass(frozen=True)
class StageControlSuite:
    controller: Controller

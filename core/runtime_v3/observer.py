"""Observation-only interface for image and robot-state evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class RobotObservation:
    observation_id: str
    frame_id: int
    images: Mapping[str, Any] = field(default_factory=dict)
    proprioception: Mapping[str, Any] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    fresh: bool = True
    done: bool = False


class Observer(Protocol):
    """Collect evidence only; this interface has no action method."""

    def observe(self, environment: Any) -> RobotObservation: ...

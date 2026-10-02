"""Structured experience-memory contract; persistence and learning are deferred."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class ExperienceRecord:
    task_id: str
    option_id: str
    expected_effect: Mapping[str, Any]
    observed_effect: Mapping[str, Any]
    outcome: str
    evidence_refs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None


class ExperienceStore(Protocol):
    """Storage interface only. It cannot change state, options, or policy."""

    def append(self, record: ExperienceRecord) -> None: ...

    def recent(self, task_id: str, limit: int = 5) -> list[ExperienceRecord]: ...

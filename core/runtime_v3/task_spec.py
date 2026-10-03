"""Minimal immutable semantic interface for Runtime V3 tasks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class GoalKind(str, Enum):
    ALIGN = "ALIGN"


@dataclass(frozen=True)
class EntitySpec:
    """Semantic description of one task entity; contains no physical plan."""

    key: str
    semantic_phrase: str
    role: str

    def __post_init__(self) -> None:
        for name in ("key", "semantic_phrase", "role"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class TaskSpec:
    """Task instruction plus the semantic entities needed by one skill."""

    instruction: str
    entities: tuple[EntitySpec, ...]
    focus_entity_key: str
    goal_kind: GoalKind = GoalKind.ALIGN

    def __post_init__(self) -> None:
        instruction = str(self.instruction).strip()
        entities = tuple(self.entities)
        focus = str(self.focus_entity_key).strip()
        try:
            goal = GoalKind(self.goal_kind)
        except ValueError as exc:
            raise ValueError("this Runtime interface supports only the ALIGN goal") from exc
        if not instruction:
            raise ValueError("instruction must be non-empty")
        if not entities or not all(isinstance(item, EntitySpec) for item in entities):
            raise ValueError("entities must contain at least one EntitySpec")
        keys = tuple(item.key for item in entities)
        if len(set(keys)) != len(keys):
            raise ValueError("entity keys must be unique")
        if focus not in keys:
            raise ValueError("focus_entity_key must name an entity in entities")
        object.__setattr__(self, "instruction", instruction)
        object.__setattr__(self, "entities", entities)
        object.__setattr__(self, "focus_entity_key", focus)
        object.__setattr__(self, "goal_kind", goal)

    @property
    def focus_entity(self) -> EntitySpec:
        return next(item for item in self.entities if item.key == self.focus_entity_key)


class ReferenceTaskCompiler:
    """Compile a frozen semantic binding from a task manifest entry."""

    _FORBIDDEN_FIELDS = frozenset({
        "direction", "distance", "pixel", "xyz", "coordinate", "coordinates",
        "scale", "threshold", "grasp_point", "action", "controller_action",
    })

    def compile(self, entry: Mapping[str, Any]) -> TaskSpec:
        instruction = entry.get("instruction")
        entities_raw = entry.get("entities")
        focus = entry.get("focus_entity_key", "target")
        goal = entry.get("goal_kind", GoalKind.ALIGN.value)
        if not isinstance(instruction, str) or not isinstance(entities_raw, Sequence):
            raise ValueError("manifest entry requires instruction and entities")
        entities = []
        for raw in entities_raw:
            if isinstance(raw, EntitySpec):
                entities.append(raw)
                continue
            if not isinstance(raw, Mapping):
                raise ValueError("manifest entities must be mappings or EntitySpec values")
            forbidden = self._FORBIDDEN_FIELDS.intersection(str(key).casefold() for key in raw)
            if forbidden:
                raise ValueError(f"semantic entity binding contains physical fields: {sorted(forbidden)}")
            entities.append(EntitySpec(
                key=raw.get("key", ""),
                semantic_phrase=raw.get("semantic_phrase", ""),
                role=raw.get("role", ""),
            ))
        return TaskSpec(instruction, tuple(entities), str(focus), GoalKind(goal))

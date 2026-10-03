"""Minimal immutable semantic interface for Runtime V3 tasks."""

from __future__ import annotations

import json
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
        forbidden = self._FORBIDDEN_FIELDS.intersection(
            str(key).casefold() for key in entry
        )
        if forbidden:
            raise ValueError(f"task binding contains physical fields: {sorted(forbidden)}")
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


class QwenTaskCompiler:
    """Bind one instruction to one semantic manipuland with one Qwen response."""

    _TOP_LEVEL_FIELDS = frozenset({"entities", "focus_entity_key", "goal_kind"})
    _ENTITY_FIELDS = frozenset({"key", "semantic_phrase", "role"})

    def __init__(self, client: Any, *, max_tokens: int = 128) -> None:
        self.client = client
        self.max_tokens = int(max_tokens)
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.last_raw_text: str | None = None

    def compile(self, instruction: str, *, agentview_image: Any = None) -> TaskSpec:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        prompt = (
            "Bind the primary manipuland named by the task instruction. Return exactly one "
            "JSON object and no prose, markdown, or extra keys. Use one entity, key=target, "
            "role=MANIPULAND, and a short semantic phrase copied from the instruction. "
            "goal_kind must be ALIGN. Never return coordinates, pixels, directions, distances, "
            "scales, thresholds, grasp points, robot actions, or controller fields.\n"
            'Required shape: {"entities":[{"key":"target",'
            '"semantic_phrase":"...","role":"MANIPULAND"}],'
            '"focus_entity_key":"target","goal_kind":"ALIGN"}\n'
            f"Task instruction: {instruction.strip()}"
        )
        self.last_raw_text = None
        try:
            response = self.client.complete_text(
                prompt,
                agentview_image=agentview_image,
                wrist_image=None,
                max_tokens=self.max_tokens,
                temperature=0.0,
                chat_template_kwargs={"enable_thinking": False, "thinking": False},
            )
        except Exception as exc:
            self.last_raw_text = getattr(exc, "raw_text", None)
            raise
        raw_text = str(getattr(response, "raw_text", ""))
        self.last_raw_text = raw_text
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError("Qwen semantic output is not strict JSON") from exc
        if not isinstance(parsed, Mapping) or set(parsed) != self._TOP_LEVEL_FIELDS:
            raise ValueError("Qwen semantic output does not match the required top-level schema")
        raw_entities = parsed.get("entities")
        if not isinstance(raw_entities, list) or len(raw_entities) != 1:
            raise ValueError("Qwen semantic output must contain exactly one primary entity")
        raw_entity = raw_entities[0]
        if not isinstance(raw_entity, Mapping) or set(raw_entity) != self._ENTITY_FIELDS:
            raise ValueError("Qwen entity does not match the required semantic-only schema")
        if any(not isinstance(raw_entity[field], str) for field in self._ENTITY_FIELDS):
            raise ValueError("Qwen entity fields must all be strings")
        if not isinstance(parsed.get("focus_entity_key"), str):
            raise ValueError("focus_entity_key must be a string")
        if parsed.get("goal_kind") != GoalKind.ALIGN.value:
            raise ValueError("Qwen task goal must be ALIGN")
        entity = EntitySpec(
            key=raw_entity["key"],
            semantic_phrase=raw_entity["semantic_phrase"],
            role=raw_entity["role"],
        )
        if entity.role != "MANIPULAND":
            raise ValueError("Qwen primary entity role must be MANIPULAND")
        return TaskSpec(
            instruction=instruction,
            entities=(entity,),
            focus_entity_key=parsed["focus_entity_key"],
            goal_kind=GoalKind.ALIGN,
        )

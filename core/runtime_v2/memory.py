"""Bounded visual memory for VCR-v2.

Images are kept by the episode logger; this class stores stable references and
compact evidence so a Qwen request can contain a small temporal window without
turning the runtime into a second policy memory.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace
from typing import Iterable, Optional
from uuid import uuid4

from .types import VisualMemoryEntry, jsonable


class VisualMemory:
    def __init__(self, *, max_entries_per_epoch: int = 8) -> None:
        self.max_entries_per_epoch = max(1, int(max_entries_per_epoch))
        self._entries: dict[tuple[Optional[str], int], deque[VisualMemoryEntry]] = defaultdict(
            lambda: deque(maxlen=self.max_entries_per_epoch)
        )
        self._keyframes: dict[tuple[Optional[str], int], deque[VisualMemoryEntry]] = defaultdict(
            lambda: deque(maxlen=4)
        )
        self._marked_events: set[tuple[Optional[str], int, str]] = set()
        self.episode_id = uuid4().hex

    def reset(self, *, episode_id: Optional[str] = None) -> None:
        self._entries.clear()
        self._keyframes.clear()
        self._marked_events.clear()
        self.episode_id = str(episode_id or uuid4().hex)

    def append(self, entry: VisualMemoryEntry) -> VisualMemoryEntry:
        if entry.episode_id not in {None, self.episode_id}:
            raise ValueError("visual memory entry belongs to a different episode")
        if entry.episode_id is None:
            entry = replace(entry, episode_id=self.episode_id)
        key = (entry.instance_id, int(entry.grasp_epoch))
        self._entries[key].append(entry)
        return entry

    def mark(
        self, *, instance_id: Optional[str], grasp_epoch: int, frame_id: int,
        tag: str, event_key: Optional[str] = None,
    ) -> None:
        entries = self._entries.get((instance_id, int(grasp_epoch)))
        if not entries:
            return
        if event_key is not None:
            event = (instance_id, int(grasp_epoch), str(event_key))
            if event in self._marked_events:
                return
            self._marked_events.add(event)
        latest = entries[-1]
        if latest.frame_id != int(frame_id):
            return
        tags = tuple(dict.fromkeys((*latest.tags, str(tag))))
        entries[-1] = replace(latest, tags=tags)
        keyframes = self._keyframes[(instance_id, int(grasp_epoch))]
        if keyframes and keyframes[-1].frame_id == latest.frame_id:
            keyframes[-1] = entries[-1]
        else:
            keyframes.append(entries[-1])

    def update_latest(self, *, instance_id: Optional[str], grasp_epoch: int, **fields) -> None:
        key = (instance_id, int(grasp_epoch))
        entries = self._entries.get(key)
        if not entries:
            return
        entries[-1] = replace(entries[-1], **fields)
        keyframes = self._keyframes.get(key)
        if keyframes and keyframes[-1].frame_id == entries[-1].frame_id:
            keyframes[-1] = entries[-1]

    def placement_bundle(
        self, *, instance_id: Optional[str], grasp_epoch: int, limit: int = 3,
        route_epoch: Optional[int] = None,
    ) -> list[dict]:
        """Select action context, one event keyframe, and the frozen live frame.

        The frame immediately before the current observation is the strongest
        default action-effect comparison. A distinct recent event keyframe adds
        context when the caller permits three rows; the current row is always
        last. This avoids letting one old keyframe crowd out the action's own
        pre-frame.
        """
        key = (instance_id, int(grasp_epoch))
        recent = self._entries.get(key)
        if not recent:
            return []
        current = recent[-1]
        max_frames = max(1, min(3, int(limit)))
        older_limit = max_frames - 1
        older_by_frame = {
            entry.frame_id: entry
            for entry in (*self._keyframes.get(key, ()), *recent)
            if entry.frame_id < current.frame_id
        }
        selected_older = []
        action_pre_frame = recent[-2] if len(recent) > 1 else None
        if action_pre_frame is not None and action_pre_frame.frame_id < current.frame_id:
            selected_older.append(action_pre_frame)
        if len(selected_older) < older_limit:
            distinct_keyframes = [
                entry for entry in self._keyframes.get(key, ())
                if entry.frame_id < current.frame_id
                and entry.frame_id not in {item.frame_id for item in selected_older}
            ]
            if distinct_keyframes:
                selected_older.append(distinct_keyframes[-1])
        if len(selected_older) < older_limit:
            remaining = [
                entry for frame, entry in older_by_frame.items()
                if frame not in {item.frame_id for item in selected_older}
            ]
            for entry in sorted(remaining, key=lambda item: item.frame_id, reverse=True):
                selected_older.append(entry)
                if len(selected_older) >= older_limit:
                    break
        selected = sorted(selected_older, key=lambda item: item.frame_id) + [current]
        bundle = [self._brief(entry) for entry in selected]
        for item in bundle:
            summary = item.get("decision_summary")
            if not isinstance(summary, dict):
                continue
            expired = int(current.frame_id) > int(summary.get("expires_after_frame", -1))
            wrong_route = route_epoch is not None and int(summary.get("route_epoch", -1)) != int(route_epoch)
            if expired or wrong_route:
                item["decision_summary"] = {}
        return bundle

    def record_decision(
        self, *, instance_id: Optional[str], grasp_epoch: int, frame_id: int,
        route_epoch: int, relation: str, reasoning: str, scene_description: Optional[dict] = None,
        evidence_for: Optional[list[dict]] = None,
        evidence_against: Optional[list[dict]] = None,
        missing_observation: str = "",
        expected_effect: str = "",
        failure_condition: str = "",
        lifetime_frames: int = 8,
    ) -> bool:
        """Keep an agent hypothesis with provenance and a short validity window."""
        entries = self._entries.get((instance_id, int(grasp_epoch)))
        if not entries or int(entries[-1].frame_id) != int(frame_id):
            return False
        summary = {
            "status": "UNVERIFIED_AGENT_HYPOTHESIS",
            "relation": str(relation or "UNKNOWN"),
            "selected_option": (
                str(scene_description.get("selected_option") or "UNKNOWN")
                if isinstance(scene_description, dict)
                else "UNKNOWN"
            ),
            "reasoning": str(reasoning or "")[:320],
            "scene_description": scene_description if isinstance(scene_description, dict) else {},
            "evidence_for": list(evidence_for or []),
            "evidence_against": list(evidence_against or []),
            "missing_observation": str(missing_observation or ""),
            "expected_effect": str(expected_effect or ""),
            "failure_condition": str(failure_condition or ""),
            "frame_id": int(frame_id),
            "route_epoch": int(route_epoch),
            "expires_after_frame": int(frame_id) + max(1, int(lifetime_frames)),
        }
        entries[-1] = replace(entries[-1], decision_summary=summary)
        keyframes = self._keyframes.get((instance_id, int(grasp_epoch)))
        if keyframes and keyframes[-1].frame_id == int(frame_id):
            keyframes[-1] = entries[-1]
        return True

    @staticmethod
    def _brief(entry: VisualMemoryEntry) -> dict:
        return {
            key: value for key, value in jsonable(entry).items()
            if key in {
                "episode_id", "instance_id", "grasp_epoch", "frame_id", "agentview_ref", "wrist_ref",
                "before_frame_id", "before_agentview_ref", "before_wrist_ref",
                "requested_action", "authorized_action", "executed_action", "motion_delta",
                "route_phase", "route_epoch", "placement_summary", "predicted_effect",
                "observed_effect", "effect_status", "decision_summary", "tags",
            }
        }

    def prompt_entries(self, *, instance_id: Optional[str], grasp_epoch: int, limit: int = 2) -> list[dict]:
        return [self._brief(item) for item in reversed(self.recent(instance_id=instance_id, grasp_epoch=grasp_epoch, limit=limit))]

    def recent(
        self,
        *,
        instance_id: Optional[str],
        grasp_epoch: int,
        limit: int = 4,
    ) -> list[VisualMemoryEntry]:
        entries = self._entries.get((instance_id, int(grasp_epoch)), ())
        return list(reversed(list(entries)[-max(1, int(limit)) :]))

    def refs(self, *, instance_id: Optional[str], grasp_epoch: int, limit: int = 4) -> tuple[str, ...]:
        refs: list[str] = []
        for entry in self.recent(instance_id=instance_id, grasp_epoch=grasp_epoch, limit=limit):
            refs.extend(x for x in (entry.agentview_ref, entry.wrist_ref) if x)
        return tuple(dict.fromkeys(refs))

    def summary(self, *, instance_id: Optional[str], grasp_epoch: int, limit: int = 4) -> list[dict]:
        return [jsonable(item) for item in self.recent(instance_id=instance_id, grasp_epoch=grasp_epoch, limit=limit)]

    def snapshot(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "max_entries_per_epoch": self.max_entries_per_epoch,
            "epochs": {
                f"{instance_id or 'unknown'}:{epoch}": [jsonable(item) for item in entries]
                for (instance_id, epoch), entries in self._entries.items()
            },
        }

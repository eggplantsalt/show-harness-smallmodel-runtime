"""Category-agnostic, CPU-only instance association for VCR-v2."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Any, Iterable, Optional

import numpy as np

from .types import EntityTrack, ObservationHealth


def bbox_tuple(value: Any) -> Optional[tuple[float, float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _center(box: tuple[float, float, float, float]) -> np.ndarray:
    return np.asarray(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0))


def _diagonal(box: tuple[float, float, float, float]) -> float:
    return float(np.linalg.norm((box[2] - box[0], box[3] - box[1])))


def _area(box: tuple[float, float, float, float]) -> float:
    return max(1.0, (box[2] - box[0]) * (box[3] - box[1]))


def appearance_histogram(
    image: Optional[np.ndarray], box: tuple[float, float, float, float], bins: int = 8
) -> tuple[float, ...]:
    """Small RGB histogram; no learned model and no task-specific appearance rule."""
    if image is None:
        return ()
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        return ()
    h, w = array.shape[:2]
    x1, y1, x2, y2 = (
        max(0, min(w, int(round(box[0])))),
        max(0, min(h, int(round(box[1])))),
        max(0, min(w, int(round(box[2])))),
        max(0, min(h, int(round(box[3])))),
    )
    if x2 <= x1 or y2 <= y1:
        return ()
    crop = np.asarray(array[y1:y2, x1:x2, :3], dtype=np.uint8)
    values: list[float] = []
    for channel in range(3):
        hist, _ = np.histogram(crop[..., channel], bins=bins, range=(0, 256))
        hist = hist.astype(float)
        hist /= max(1.0, float(hist.sum()))
        values.extend(hist.tolist())
    return tuple(values)


def _hist_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return float(np.abs(np.asarray(left) - np.asarray(right)).mean())


def candidate_dicts(tool: Any, *, camera: str = "unknown") -> list[dict[str, Any]]:
    """Return candidates expressed in the active camera's pixel coordinates.

    VisualHarness may include a secondary view for cross-view verification.  Its
    boxes are not commensurate with the primary image and must never enter the
    same candidate index space.
    """
    found: list[dict[str, Any]] = []
    camera = str(camera or "unknown").lower()

    def extend(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        candidates = value.get("candidates")
        if not isinstance(candidates, list):
            return False
        for item in candidates:
            if isinstance(item, dict):
                tagged = dict(item)
                tagged.setdefault("camera", camera)
                found.append(tagged)
        return bool(found)

    if isinstance(tool, dict):
        # A root candidate list always belongs to evidence["camera"].
        if not extend(tool):
            selected = str(tool.get("selected_camera") or "").lower()
            probe = str(tool.get("probe_camera") or "").lower()
            primary = str(tool.get("primary_camera") or "").lower()
            fallback = str(tool.get("fallback_camera") or "").lower()
            secondary = str(tool.get("secondary_camera") or "").lower()
            if selected == camera and probe == camera:
                extend(tool.get("probe"))
            elif selected == camera and fallback == camera:
                extend(tool.get("fallback"))
            elif primary == camera:
                extend(tool.get("primary"))
            elif secondary == camera:
                extend(tool.get("secondary"))
            elif selected == camera:
                # Older metadata may omit the explicit camera role. Prefer the
                # selected branch, but never recursively mix multiple views.
                extend(tool.get("probe")) or extend(tool.get("fallback"))
    unique: dict[tuple[float, float, float, float], dict[str, Any]] = {}
    for item in found:
        box = bbox_tuple(item.get("bbox_xyxy"))
        if box is None:
            continue
        old = unique.get(box)
        if old is None or float(item.get("score", 0.0) or 0.0) > float(
            old.get("score", 0.0) or 0.0
        ):
            unique[box] = item
    return list(unique.values())


@dataclass(frozen=True)
class AssociationResult:
    health: ObservationHealth
    track: Optional[EntityTrack]
    candidates: tuple[dict[str, Any], ...]
    reason: str
    decision_lock_active: bool = False


class EntityTracker:
    """Persistent identity lock with abstention on jumps or close alternatives."""

    _ids = count(1)

    def __init__(
        self,
        *,
        role: str,
        camera: str = "unknown",
        motion_gate_scale: float = 0.75,
        motion_gate_min_px: float = 10.0,
        ambiguity_margin: float = 0.15,
        max_missed_frames: int = 6,
    ) -> None:
        self.role = str(role)
        self.camera = str(camera)
        self.motion_gate_scale = max(0.1, float(motion_gate_scale))
        self.motion_gate_min_px = max(1.0, float(motion_gate_min_px))
        self.ambiguity_margin = max(0.0, float(ambiguity_margin))
        self.max_missed_frames = max(0, int(max_missed_frames))
        self.track: Optional[EntityTrack] = None
        self._decision_lock_frames = 0

    def reset(self) -> None:
        self.track = None
        self._decision_lock_frames = 0

    def commit_candidate(
        self,
        *,
        candidate: dict[str, Any],
        image: Optional[np.ndarray],
        frame_id: int,
        semantic_label: str,
    ) -> Optional[EntityTrack]:
        """Commit a bounded VLM candidate choice without changing instance id."""
        box = bbox_tuple(candidate.get("bbox_xyxy"))
        if box is None:
            return None
        score = float(candidate.get("score", 0.0) or 0.0)
        if self.track is None:
            self.track = self._new_track(
                label=semantic_label,
                box=box,
                score=score,
                source="runtime_v2_critical_decision",
                frame_id=frame_id,
                image=image,
            )
            self._decision_lock_frames = 2
            return self.track
        self.track.previous_bbox_xyxy = self.track.bbox_xyxy
        self.track.bbox_xyxy = box
        self.track.confidence = score
        self.track.source = "runtime_v2_critical_decision"
        self.track.last_confirmed_frame = frame_id
        self.track.appearance = appearance_histogram(image, box) or self.track.appearance
        self.track.association_confidence = 1.0
        self.track.missed_frames = 0
        self._decision_lock_frames = 2
        return self.track

    def _new_track(
        self,
        *,
        label: str,
        box: tuple[float, float, float, float],
        score: float,
        source: str,
        frame_id: int,
        image: Optional[np.ndarray],
    ) -> EntityTrack:
        return EntityTrack(
            instance_id=f"{self.role}-{next(self._ids):04d}",
            semantic_label=label,
            bbox_xyxy=box,
            confidence=score,
            source=source,
            last_confirmed_frame=frame_id,
            camera=self.camera,
            appearance=appearance_histogram(image, box),
        )

    def update(
        self,
        *,
        evidence: dict[str, Any],
        image: Optional[np.ndarray],
        frame_id: int,
        semantic_label: str,
    ) -> AssociationResult:
        candidates = candidate_dicts(evidence.get("tool"), camera=self.camera)
        observed_box = bbox_tuple(evidence.get("bbox_xyxy"))
        if not candidates and observed_box is not None:
            candidates = [
                {
                    "bbox_xyxy": list(observed_box),
                    "score": float(evidence.get("confidence", 0.0) or 0.0),
                    "label": semantic_label,
                }
            ]

        if self.track is None:
            if observed_box is None:
                return AssociationResult(
                    ObservationHealth.OCCLUDED, None, tuple(candidates), "no_initial_instance"
                )
            if len(candidates) > 1:
                # With no temporal reference, detector score is not identity.
                # Seed only after the bounded semantic resolver chooses a
                # candidate; this is especially important on camera handoff.
                return AssociationResult(
                    ObservationHealth.AMBIGUOUS,
                    None,
                    tuple(candidates),
                    "multiple_candidates_without_identity_reference",
                )
            self.track = self._new_track(
                label=semantic_label,
                box=observed_box,
                score=float(evidence.get("confidence", 0.0) or 0.0),
                source=str(evidence.get("source") or "unknown"),
                frame_id=frame_id,
                image=image,
            )
            return AssociationResult(
                ObservationHealth.VALID, self.track, tuple(candidates), "instance_seeded"
            )

        previous = self.track
        reference = previous.bbox_xyxy
        gate = max(self.motion_gate_min_px, self.motion_gate_scale * _diagonal(reference))
        scored: list[tuple[float, float, tuple[float, float, float, float], dict[str, Any], tuple[float, ...]]] = []
        for candidate in candidates:
            box = bbox_tuple(candidate.get("bbox_xyxy"))
            if box is None:
                continue
            distance = float(np.linalg.norm(_center(box) - _center(reference)))
            if distance > gate:
                continue
            area_cost = min(1.0, abs(np.log(_area(box) / _area(reference))))
            appearance = appearance_histogram(image, box)
            appearance_cost = min(1.0, _hist_distance(previous.appearance, appearance) * 8.0)
            normalized_distance = distance / max(1.0, gate)
            cost = 0.65 * normalized_distance + 0.20 * area_cost + 0.15 * appearance_cost
            score = float(candidate.get("score", 0.0) or 0.0)
            scored.append((cost, score, box, candidate, appearance))

        if not scored:
            previous.missed_frames += 1
            health = (
                ObservationHealth.OCCLUDED
                if previous.missed_frames <= self.max_missed_frames
                else ObservationHealth.AMBIGUOUS
            )
            return AssociationResult(health, previous, tuple(candidates), "no_candidate_in_motion_gate")

        scored.sort(key=lambda item: (item[0], -item[1]))
        best = scored[0]
        decision_locked = self._decision_lock_frames > 0
        if decision_locked:
            self._decision_lock_frames -= 1
        if (
            not decision_locked
            and len(scored) > 1
            and scored[1][0] - best[0] < self.ambiguity_margin
        ):
            return AssociationResult(
                ObservationHealth.AMBIGUOUS,
                previous,
                tuple(candidates),
                "association_margin_too_small",
            )

        cost, detector_score, box, _candidate, appearance = best
        previous.previous_bbox_xyxy = previous.bbox_xyxy
        previous.bbox_xyxy = box
        previous.confidence = detector_score
        previous.source = "runtime_v2_instance_associated"
        previous.last_confirmed_frame = frame_id
        previous.appearance = appearance or previous.appearance
        previous.association_confidence = max(0.0, min(1.0, 1.0 - cost))
        previous.missed_frames = 0
        return AssociationResult(
            ObservationHealth.VALID,
            previous,
            tuple(candidates),
            "instance_associated",
            decision_lock_active=decision_locked,
        )

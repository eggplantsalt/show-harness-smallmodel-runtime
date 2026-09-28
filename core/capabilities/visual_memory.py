"""CPU visual tracking and bounded episodic memory for the LIBERO pilot.

This module never reads simulator state.  It accepts only prepared RGB frames and
host-owned action metadata.  A tracked box is explicitly weaker than a fresh SAM3
grounding and expires after a small number of frames or a scene epoch change.
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional

import cv2
import numpy as np


def image_sha1(image: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


def _box_tuple(value: Any) -> Optional[tuple[int, int, int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (int(round(float(v))) for v in value)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _clip_box(box: tuple[int, int, int, int], width: int, height: int):
    x0, y0, x1, y1 = box
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return x0, y0, x1, y1


@dataclass
class TrackState:
    image: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    age: int = 0
    lost: bool = False


class CpuVisualTracker:
    """Small bounded template tracker used between semantic segmentations."""

    def __init__(self, *, min_match: float = 0.42, max_age: int = 4) -> None:
        self.min_match = float(min_match)
        self.max_age = max(1, int(max_age))
        self.state: Optional[TrackState] = None

    def reset(self) -> None:
        self.state = None

    def seed(
        self,
        image: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int],
        confidence: float,
    ) -> None:
        frame = np.ascontiguousarray(np.asarray(image, dtype=np.uint8)).copy()
        h, w = frame.shape[:2]
        self.state = TrackState(
            image=frame,
            bbox_xyxy=_clip_box(bbox_xyxy, w, h),
            confidence=max(0.0, min(1.0, float(confidence))),
        )

    def update(self, image: np.ndarray) -> Optional[dict[str, Any]]:
        if self.state is None:
            return None
        frame = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        if frame.ndim != 3 or frame.shape[2] != 3:
            return None
        previous = self.state
        previous_h, previous_w = previous.image.shape[:2]
        current_h, current_w = frame.shape[:2]
        if (previous_h, previous_w) != (current_h, current_w):
            self.reset()
            return None

        x0, y0, x1, y1 = previous.bbox_xyxy
        bw, bh = x1 - x0, y1 - y0
        if bw < 3 or bh < 3:
            self.reset()
            return None
        gray_previous = cv2.cvtColor(previous.image, cv2.COLOR_RGB2GRAY)
        gray_current = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        template = gray_previous[y0:y1, x0:x1]
        # Search a bounded neighborhood.  The tracker cannot jump across an image;
        # a semantic re-grounding is required after a lost target or large motion.
        pad_x = max(12, int(round(bw * 0.8)))
        pad_y = max(12, int(round(bh * 0.8)))
        sx0 = max(0, x0 - pad_x)
        sy0 = max(0, y0 - pad_y)
        sx1 = min(current_w, x1 + pad_x)
        sy1 = min(current_h, y1 + pad_y)
        search = gray_current[sy0:sy1, sx0:sx1]
        if search.shape[0] < bh or search.shape[1] < bw:
            self.reset()
            return None
        try:
            scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(scores)
        except cv2.error:
            self.reset()
            return None
        score = float(max_val)
        if not np.isfinite(score) or score < self.min_match:
            previous.age += 1
            previous.lost = True
            if previous.age > self.max_age:
                self.reset()
            return None
        nx0, ny0 = sx0 + int(max_loc[0]), sy0 + int(max_loc[1])
        bbox = _clip_box((nx0, ny0, nx0 + bw, ny0 + bh), current_w, current_h)
        confidence = max(0.0, min(1.0, 0.5 * previous.confidence + 0.5 * score))
        self.state = TrackState(
            image=frame.copy(), bbox_xyxy=bbox, confidence=confidence, age=previous.age + 1
        )
        return {
            "bbox_xyxy": list(bbox),
            "score": score,
            "confidence": confidence,
            "track_age": self.state.age,
            "source": "cpu_template_tracker",
        }


@dataclass
class MemoryEntry:
    frame_id: int
    camera: str
    image_hash: str
    stage: str
    target: str
    action: Optional[str]
    bbox_xyxy: Optional[list[int]]
    center_xy: Optional[list[float]]
    confidence: float
    source: str
    progress_delta_px: Optional[float]
    scene_epoch: int
    timestamp_s: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "camera": self.camera,
            "image_hash": self.image_hash,
            "stage": self.stage,
            "target": self.target,
            "action": self.action,
            "bbox_xyxy": self.bbox_xyxy,
            "center_xy": self.center_xy,
            "confidence": round(float(self.confidence), 4),
            "source": self.source,
            "progress_delta_px": (
                None
                if self.progress_delta_px is None
                else round(float(self.progress_delta_px), 3)
            ),
            "scene_epoch": self.scene_epoch,
            "timestamp_s": self.timestamp_s,
        }


class EpisodicVisualMemory:
    """Bounded, invalidatable memory of runtime visual evidence."""

    def __init__(self, *, max_entries: int = 12, ttl_frames: int = 4) -> None:
        self.max_entries = max(1, int(max_entries))
        self.ttl_frames = max(1, int(ttl_frames))
        self.entries: Deque[MemoryEntry] = deque(maxlen=self.max_entries)
        self.scene_epoch = 0
        self.last_center: Optional[np.ndarray] = None
        self.last_action: Optional[str] = None
        self.last_frame_id: Optional[int] = None

    def reset(self) -> None:
        self.entries.clear()
        self.scene_epoch = 0
        self.last_center = None
        self.last_action = None
        self.last_frame_id = None

    def invalidate_scene(self, reason: str = "scene_changed") -> None:
        self.scene_epoch += 1
        self.last_center = None
        self.entries.append(
            MemoryEntry(
                frame_id=int(self.last_frame_id or -1),
                camera="",
                image_hash="",
                stage="INVALIDATED",
                target="",
                action=None,
                bbox_xyxy=None,
                center_xy=None,
                confidence=0.0,
                source=reason,
                progress_delta_px=None,
                scene_epoch=self.scene_epoch,
            )
        )

    def append(
        self,
        *,
        frame_id: int,
        camera: str,
        image_hash_value: str,
        stage: str,
        target: str,
        action: Optional[str],
        bbox_xyxy: Optional[tuple[int, int, int, int]],
        confidence: float,
        source: str,
    ) -> MemoryEntry:
        center = None
        if bbox_xyxy is not None:
            x0, y0, x1, y1 = bbox_xyxy
            center = [(x0 + x1) / 2.0, (y0 + y1) / 2.0]
        current_center = np.asarray(center, dtype=float) if center else None
        progress = None
        if current_center is not None and self.last_center is not None:
            progress = float(np.linalg.norm(current_center - self.last_center))
        entry = MemoryEntry(
            frame_id=int(frame_id),
            camera=str(camera),
            image_hash=str(image_hash_value),
            stage=str(stage),
            target=str(target),
            action=action,
            bbox_xyxy=list(bbox_xyxy) if bbox_xyxy is not None else None,
            center_xy=center,
            confidence=float(confidence),
            source=str(source),
            progress_delta_px=progress,
            scene_epoch=self.scene_epoch,
        )
        self.entries.append(entry)
        if current_center is not None:
            self.last_center = current_center
        self.last_frame_id = int(frame_id)
        return entry

    def recent(self, *, target: str = "", frame_id: Optional[int] = None) -> list[MemoryEntry]:
        target = str(target)
        current_frame = self.last_frame_id if frame_id is None else int(frame_id)
        if current_frame is None:
            return []
        return [
            entry
            for entry in reversed(self.entries)
            if entry.target == target
            and entry.scene_epoch == self.scene_epoch
            and current_frame - entry.frame_id <= self.ttl_frames
        ]

    def context(self, *, target: str, frame_id: int) -> str:
        entries = self.recent(target=target, frame_id=frame_id)
        if not entries:
            return "VISUAL MEMORY: no current verified target evidence."
        latest = entries[0]
        recent_actions = [e.action for e in entries if e.action]
        repeated = ""
        if len(recent_actions) >= 3 and len(set(recent_actions[:3])) == 1:
            repeated = f" Repeated action warning: {recent_actions[0]} three times."
        center = latest.center_xy or []
        progress = (
            "unknown"
            if latest.progress_delta_px is None
            else f"{latest.progress_delta_px:.1f}px"
        )
        return (
            "HARNESS VISUAL EVIDENCE (host-generated, not Agent belief): "
            f"target={latest.target!r}; camera={latest.camera}; source={latest.source}; "
            f"center_xy={center}; bbox_xyxy={latest.bbox_xyxy}; "
            f"confidence={latest.confidence:.2f}; frame={latest.frame_id}; "
            f"motion_since_previous={progress}; scene_epoch={latest.scene_epoch}."
            f"{repeated} Treat missing or low-confidence evidence as unknown."
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "scene_epoch": self.scene_epoch,
            "last_frame_id": self.last_frame_id,
            "entries": [entry.as_dict() for entry in self.entries],
        }

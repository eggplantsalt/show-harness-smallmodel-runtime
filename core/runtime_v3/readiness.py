"""Task-independent scene-motion and same-entity observation readiness."""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np


SCENE_MOTION_MAX_NORMALIZED_RGB_DIFFERENCE = 0.00005
SCENE_MOTION_STABLE_FRAME_PAIRS = 3
SCENE_MOTION_THRESHOLD_SOURCE = (
    "M3.7 development HOLD captures: the largest per-task p95 of stable-tail "
    "full-frame normalized RGB differences was 0.0000351; 0.00005 is rounded "
    "above that value and below the measured settling-motion differences. "
    "Frozen before held-out evaluation."
)

ENTITY_OBSERVATION_WINDOW = 3
ENTITY_MAX_CENTROID_SHIFT_OVER_BBOX_DIAGONAL = 0.02
ENTITY_MAX_BBOX_EDGE_SHIFT_OVER_BBOX_DIAGONAL = 0.10
ENTITY_MIN_ADJACENT_MASK_IOU = 0.50
ENTITY_MAX_RELATIVE_AREA_CHANGE = 0.20
ENTITY_THRESHOLD_SOURCE = (
    "M3.7 development SAM control traces had stable-mask p95 normalized "
    "centroid changes below 0.0004, adjacent-mask IoU p10 above 0.999, and "
    "relative area changes below 0.001. Generic bounds of 0.02 bbox-diagonal "
    "centroid shift, 0.10 bbox-diagonal edge shift, 0.50 IoU, and 0.20 "
    "relative area change allow visibly non-identical masks; frozen before "
    "held-out evaluation."
)


class SceneMotionReady:
    """Measure only normalized full-frame canonical RGB change over time.

    The interface intentionally accepts no phrase, identity, mask, pose, depth,
    contact, or task input. Call ``update`` only after RobotReady has completed.
    """

    def __init__(
        self,
        *,
        max_normalized_rgb_difference: float = SCENE_MOTION_MAX_NORMALIZED_RGB_DIFFERENCE,
        stable_frame_pairs: int = SCENE_MOTION_STABLE_FRAME_PAIRS,
    ) -> None:
        if not np.isfinite(max_normalized_rgb_difference) or max_normalized_rgb_difference < 0:
            raise ValueError("scene motion threshold must be finite and nonnegative")
        if int(stable_frame_pairs) < 1:
            raise ValueError("scene motion window must contain at least one frame pair")
        self.max_normalized_rgb_difference = float(max_normalized_rgb_difference)
        self.stable_frame_pairs = int(stable_frame_pairs)
        self._previous: np.ndarray | None = None
        self._stable_scores: deque[float] = deque(maxlen=self.stable_frame_pairs)
        self.last_normalized_rgb_difference: float | None = None
        self.ready = False

    def reset(self) -> None:
        self._previous = None
        self._stable_scores.clear()
        self.last_normalized_rgb_difference = None
        self.ready = False

    def update(self, canonical_rgb: np.ndarray) -> bool:
        image = np.asarray(canonical_rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("canonical RGB must have shape HxWx3")
        if image.size == 0 or not np.issubdtype(image.dtype, np.number):
            raise ValueError("canonical RGB must be a nonempty numeric image")
        if not np.all(np.isfinite(image)):
            raise ValueError("canonical RGB must contain finite values")
        current = np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8, copy=False))
        if self._previous is None or self._previous.shape != current.shape:
            self.ready = False
            self._previous = current.copy()
            self._stable_scores.clear()
            self.last_normalized_rgb_difference = None
            return self.ready
        if self.ready:
            self._previous = current.copy()
            return True

        delta = np.abs(current.astype(np.int16) - self._previous.astype(np.int16))
        score = float(delta.mean() / 255.0)
        self.last_normalized_rgb_difference = score
        self._previous = current.copy()
        if score <= self.max_normalized_rgb_difference:
            self._stable_scores.append(score)
        else:
            self._stable_scores.clear()
        self.ready = len(self._stable_scores) >= self.stable_frame_pairs
        return self.ready

    def to_record(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "max_normalized_rgb_difference": self.max_normalized_rgb_difference,
            "required_stable_frame_pairs": self.stable_frame_pairs,
            "stable_frame_pair_count": len(self._stable_scores),
            "stable_frame_pair_scores": list(self._stable_scores),
            "last_normalized_rgb_difference": self.last_normalized_rgb_difference,
            "metric": "mean_absolute_full_frame_rgb_difference_divided_by_255",
            "roi": None,
            "threshold_source": SCENE_MOTION_THRESHOLD_SOURCE,
            "evidence_source": "canonical_rgb_temporal_observation",
        }


class EntityObservationReady:
    """Require a short stable window from one associated semantic entity.

    Identity continuity is supplied by the existing target association logic.
    This evidence layer rejects invalid/lost associations, measures normalized
    geometry and mask overlap, and selects a representative mask from its own
    same-entity window.
    """

    VALID_IDENTITY_STATUSES = frozenset({"ANCHORED", "SAME_TARGET"})

    def __init__(
        self,
        *,
        window_observations: int = ENTITY_OBSERVATION_WINDOW,
        max_centroid_shift_over_bbox_diagonal: float = ENTITY_MAX_CENTROID_SHIFT_OVER_BBOX_DIAGONAL,
        max_bbox_edge_shift_over_bbox_diagonal: float = ENTITY_MAX_BBOX_EDGE_SHIFT_OVER_BBOX_DIAGONAL,
        min_adjacent_mask_iou: float = ENTITY_MIN_ADJACENT_MASK_IOU,
        max_relative_area_change: float = ENTITY_MAX_RELATIVE_AREA_CHANGE,
    ) -> None:
        if int(window_observations) < 2:
            raise ValueError("entity observation window requires at least two observations")
        thresholds = (max_centroid_shift_over_bbox_diagonal,
                      max_bbox_edge_shift_over_bbox_diagonal,
                      min_adjacent_mask_iou, max_relative_area_change)
        if not all(np.isfinite(value) for value in thresholds):
            raise ValueError("entity observation thresholds must be finite")
        if (max_centroid_shift_over_bbox_diagonal < 0
                or max_bbox_edge_shift_over_bbox_diagonal < 0
                or not 0 <= min_adjacent_mask_iou <= 1
                or max_relative_area_change < 0):
            raise ValueError("entity observation thresholds are out of range")
        self.window_observations = int(window_observations)
        self.max_centroid_shift_over_bbox_diagonal = float(max_centroid_shift_over_bbox_diagonal)
        self.max_bbox_edge_shift_over_bbox_diagonal = float(max_bbox_edge_shift_over_bbox_diagonal)
        self.min_adjacent_mask_iou = float(min_adjacent_mask_iou)
        self.max_relative_area_change = float(max_relative_area_change)
        self._samples: deque[dict[str, Any]] = deque(maxlen=self.window_observations)
        self._context: tuple[str, str] | None = None
        self.last_interval: dict[str, Any] | None = None
        self.last_failure: str | None = None
        self.ready = False

    def reset(self) -> None:
        self._samples.clear()
        self._context = None
        self.last_interval = None
        self.last_failure = None
        self.ready = False

    def update(
        self,
        *,
        entity_key: str,
        grounding_query: str,
        identity_status: str,
        candidate_id: str | None,
        mask: np.ndarray | None,
        centroid_px: Sequence[float] | None,
        bbox_xyxy: Sequence[float] | None,
        mask_area_px: int | None,
        frame_id: int | None = None,
    ) -> bool:
        context = (str(entity_key), str(grounding_query))
        if self._context is not None and context != self._context:
            self.reset()
        self._context = context
        if self.ready and str(identity_status) in self.VALID_IDENTITY_STATUSES:
            return True
        if self.ready:
            self.reset()
            self._context = context
        if (str(identity_status) not in self.VALID_IDENTITY_STATUSES
                or mask is None or centroid_px is None or bbox_xyxy is None
                or mask_area_px is None or int(mask_area_px) <= 0):
            self._samples.clear()
            self.last_interval = None
            self.last_failure = (
                "IDENTITY_FAILURE" if str(identity_status) not in self.VALID_IDENTITY_STATUSES
                else "SEMANTIC_GROUNDING_FAILURE"
            )
            return False

        mask_array = np.asarray(mask, dtype=bool)
        centroid = np.asarray(centroid_px, dtype=float).reshape(-1)
        bbox = np.asarray(bbox_xyxy, dtype=float).reshape(-1)
        if mask_array.ndim != 2 or centroid.shape != (2,) or bbox.shape != (4,):
            self._samples.clear()
            self.last_failure = "ENTITY_OBSERVATION_NOT_READY"
            return False
        if not np.all(np.isfinite(centroid)) or not np.all(np.isfinite(bbox)):
            self._samples.clear()
            self.last_failure = "ENTITY_OBSERVATION_NOT_READY"
            return False
        x0, y0, x1, y1 = bbox.tolist()
        if x1 <= x0 or y1 <= y0 or not bool(mask_array.any()):
            self._samples.clear()
            self.last_failure = "ENTITY_OBSERVATION_NOT_READY"
            return False

        sample = {
            "entity_key": context[0],
            "grounding_query": context[1],
            "candidate_id": None if candidate_id is None else str(candidate_id),
            "identity_status": str(identity_status),
            "mask": mask_array.copy(),
            "centroid_px": centroid.tolist(),
            "bbox_xyxy": bbox.tolist(),
            "mask_area_px": int(mask_area_px),
            "frame_id": None if frame_id is None else int(frame_id),
        }
        self.last_interval = None
        self.last_failure = None
        stable = True
        if self._samples:
            previous = self._samples[-1]
            previous_box = previous["bbox_xyxy"]
            previous_diag = max(
                float(np.hypot(previous_box[2] - previous_box[0],
                               previous_box[3] - previous_box[1])), 1.0,
            )
            centroid_shift_ratio = float(
                np.linalg.norm(np.asarray(sample["centroid_px"])
                               - np.asarray(previous["centroid_px"])) / previous_diag
            )
            bbox_edge_shift_ratio = float(
                max(abs(left - right) for left, right in
                    zip(sample["bbox_xyxy"], previous_box)) / previous_diag
            )
            previous_mask = previous["mask"]
            if previous_mask.shape != mask_array.shape:
                mask_iou = 0.0
            else:
                union = int(np.logical_or(previous_mask, mask_array).sum())
                mask_iou = (float(np.logical_and(previous_mask, mask_array).sum() / union)
                            if union else 0.0)
            relative_area_change = abs(
                int(sample["mask_area_px"]) - int(previous["mask_area_px"])
            ) / max(int(previous["mask_area_px"]), 1)
            self.last_interval = {
                "centroid_shift_over_previous_bbox_diagonal": centroid_shift_ratio,
                "bbox_max_edge_shift_over_previous_bbox_diagonal": bbox_edge_shift_ratio,
                "adjacent_mask_iou": mask_iou,
                "relative_area_change": float(relative_area_change),
            }
            stable = (
                centroid_shift_ratio <= self.max_centroid_shift_over_bbox_diagonal
                and bbox_edge_shift_ratio <= self.max_bbox_edge_shift_over_bbox_diagonal
                and mask_iou >= self.min_adjacent_mask_iou
                and relative_area_change <= self.max_relative_area_change
            )
        if not stable:
            self._samples.clear()
            self.last_failure = "ENTITY_OBSERVATION_NOT_READY"
        self._samples.append(sample)
        self.ready = len(self._samples) >= self.window_observations
        return self.ready

    def representative_sample(self) -> Mapping[str, Any] | None:
        if not self._samples:
            return None
        masks = [sample["mask"] for sample in self._samples]
        pairwise_iou = np.eye(len(masks), dtype=float)
        for left in range(len(masks)):
            for right in range(left + 1, len(masks)):
                if masks[left].shape != masks[right].shape:
                    score = 0.0
                else:
                    union = int(np.logical_or(masks[left], masks[right]).sum())
                    score = (float(np.logical_and(masks[left], masks[right]).sum() / union)
                             if union else 0.0)
                pairwise_iou[left, right] = pairwise_iou[right, left] = score
        medoid_index = int(np.argmax(pairwise_iou.sum(axis=1)))
        median_centroid = np.median(
            np.asarray([sample["centroid_px"] for sample in self._samples], dtype=float), axis=0
        )
        selected = dict(self._samples[medoid_index])
        selected["reference_centroid_px"] = median_centroid.tolist()
        return selected

    def to_record(self) -> dict[str, Any]:
        representative = self.representative_sample()
        return {
            "ready": self.ready,
            "entity_key": self._context[0] if self._context else None,
            "grounding_query": self._context[1] if self._context else None,
            "stable_observation_count": len(self._samples),
            "required_observations": self.window_observations,
            "max_centroid_shift_over_bbox_diagonal": self.max_centroid_shift_over_bbox_diagonal,
            "max_bbox_edge_shift_over_bbox_diagonal": self.max_bbox_edge_shift_over_bbox_diagonal,
            "min_adjacent_mask_iou": self.min_adjacent_mask_iou,
            "max_relative_area_change": self.max_relative_area_change,
            "threshold_source": ENTITY_THRESHOLD_SOURCE,
            "last_visual_interval": self.last_interval,
            "last_failure": self.last_failure,
            "representative_reference": ({
                "method": "same-entity mask medoid with coordinate-wise median centroid",
                "candidate_id": representative.get("candidate_id"),
                "frame_id": representative.get("frame_id"),
                "centroid_px": representative.get("reference_centroid_px"),
                "bbox_xyxy": representative.get("bbox_xyxy"),
                "mask_area_px": representative.get("mask_area_px"),
            } if representative is not None else None),
            "evidence_source": "associated_sam_visual_observation",
        }

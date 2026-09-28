"""Harness-driven grounding, tracking, and bounded visual memory."""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from .camera_geometry import CameraCalibration, backproject_pixel_to_plane
from .sam3_client import Sam3Client
from .visual_memory import CpuVisualTracker, EpisodicVisualMemory, _box_tuple, image_sha1


class VisualHarness:
    """Automatic capability orchestration for one episode.

    It only returns evidence/context.  It does not choose, rewrite, or execute actions.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        sam3_enabled: bool = True,
        tracker_enabled: bool = True,
        memory_enabled: bool = True,
        geometry_enabled: bool = True,
        sam3_url: str = "http://127.0.0.1:8773/sse",
        sam3_python: str = "/root/autodl-tmp/openeta-services/sam3/.venv/bin/python",
        reacquire_every: int = 4,
        confidence_threshold: float = 0.5,
        sam3_ambiguity_margin: float = 0.05,
        commit_guard_enabled: bool = False,
        commit_guard_mode: str = "shadow",
        commit_guard_freshness_frames: int = 1,
        commit_guard_alignment_px: float = 24.0,
        alignment_ready_px: float = 16.0,
        approach_completion_guard_enabled: bool = False,
        approach_completion_guard_mode: str = "shadow",
        approach_completion_guard_freshness_frames: int = 1,
        grasp_agentview_fallback_enabled: bool = False,
        grasp_agentview_guard_enabled: bool = False,
        grasp_agentview_guard_mode: str = "shadow",
        grasp_agentview_guard_freshness_frames: int = 1,
        grasp_agentview_guard_alignment_px: float = 12.0,
        grasp_agentview_guard_height_m: float = 0.18,
        grasp_agentview_guard_min_height_m: Optional[float] = None,
        approach_completion_min_height_m: Optional[float] = None,
        approach_completion_max_height_m: Optional[float] = None,
        target_reference_height_m: Optional[float] = None,
        occlusion_hold_frames: int = 6,
    ) -> None:
        self.enabled = bool(enabled)
        self.sam3_enabled = bool(sam3_enabled)
        self.tracker_enabled = bool(tracker_enabled)
        self.memory_enabled = bool(memory_enabled)
        self.geometry_enabled = bool(geometry_enabled)
        self.reacquire_every = max(1, int(reacquire_every))
        self.confidence_threshold = float(confidence_threshold)
        self.sam3_ambiguity_margin = max(0.0, float(sam3_ambiguity_margin))
        self.commit_guard_enabled = bool(commit_guard_enabled)
        self.commit_guard_mode = str(commit_guard_mode or "shadow").lower()
        if self.commit_guard_mode not in {"shadow", "active"}:
            raise ValueError("commit_guard_mode must be 'shadow' or 'active'")
        self.commit_guard_freshness_frames = max(0, int(commit_guard_freshness_frames))
        self.commit_guard_alignment_px = max(0.0, float(commit_guard_alignment_px))
        self.alignment_ready_px = max(0.0, float(alignment_ready_px))
        self.approach_completion_guard_enabled = bool(approach_completion_guard_enabled)
        self.approach_completion_guard_mode = str(
            approach_completion_guard_mode or "shadow"
        ).lower()
        if self.approach_completion_guard_mode not in {"shadow", "active"}:
            raise ValueError("approach_completion_guard_mode must be 'shadow' or 'active'")
        self.approach_completion_guard_freshness_frames = max(
            0, int(approach_completion_guard_freshness_frames)
        )
        self.grasp_agentview_fallback_enabled = bool(grasp_agentview_fallback_enabled)
        self.grasp_agentview_guard_enabled = bool(grasp_agentview_guard_enabled)
        self.grasp_agentview_guard_mode = str(grasp_agentview_guard_mode or "shadow").lower()
        if self.grasp_agentview_guard_mode not in {"shadow", "active"}:
            raise ValueError("grasp_agentview_guard_mode must be 'shadow' or 'active'")
        self.grasp_agentview_guard_freshness_frames = max(
            0, int(grasp_agentview_guard_freshness_frames)
        )
        self.grasp_agentview_guard_alignment_px = max(
            0.0, float(grasp_agentview_guard_alignment_px)
        )
        self.grasp_agentview_guard_height_m = float(grasp_agentview_guard_height_m)
        self.grasp_agentview_guard_min_height_m = (
            None
            if grasp_agentview_guard_min_height_m is None
            else float(grasp_agentview_guard_min_height_m)
        )
        self.approach_completion_min_height_m = (
            None
            if approach_completion_min_height_m is None
            else float(approach_completion_min_height_m)
        )
        self.approach_completion_max_height_m = (
            None
            if approach_completion_max_height_m is None
            else float(approach_completion_max_height_m)
        )
        self.target_reference_height_m = (
            None
            if target_reference_height_m is None
            else float(target_reference_height_m)
        )
        self.occlusion_hold_frames = max(0, int(occlusion_hold_frames))
        self.sam3 = (
            Sam3Client(url=sam3_url, python=sam3_python)
            if self.enabled and self.sam3_enabled
            else None
        )
        self.tracker = CpuVisualTracker() if self.tracker_enabled else None
        self.opening_tracker = CpuVisualTracker() if self.tracker_enabled else None
        # MOVE/PLACE need two independent visual tracks: the receptacle is the
        # current subgoal target, while the object being carried comes from the
        # preceding GRASP subgoal.  Sharing one tracker would silently make the
        # destination bbox become the held-object bbox after the first update.
        self.held_tracker = CpuVisualTracker() if self.tracker_enabled else None
        self.memory = EpisodicVisualMemory() if self.memory_enabled else None
        self.last_stage_key: Optional[str] = None
        self.last_stage_identity: Optional[str] = None
        self.stage_camera_override: Optional[str] = None
        self.last_frame_id: Optional[int] = None
        self.last_grounding_frame: Optional[int] = None
        self.last_action: Optional[str] = None
        self._same_action_count = 0
        self.last_evidence: dict[str, Any] = {}
        self.last_visible_evidence: dict[str, Any] = {}
        self.held_stage_identity: Optional[str] = None
        self.held_last_grounding_frame: Optional[int] = None
        self.last_held_evidence: dict[str, Any] = {}
        self._held_instance_anchor_bbox: Optional[tuple[int, int, int, int]] = None
        self.held_rim_anchor_y: Optional[float] = None
        self.held_horizontal_stall = False
        self.opening_stage_identity: Optional[str] = None
        self.opening_last_grounding_frame: Optional[int] = None
        self.tool_calls = 0
        self.tool_failures = 0
        self.guard_shadow_events = 0
        self.guard_blocks = 0

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Optional["VisualHarness"]:
        section = cfg.get("capabilities")
        if not isinstance(section, dict) or not bool(section.get("enabled", False)):
            return None
        return cls(
            enabled=True,
            sam3_enabled=bool(section.get("sam3_enabled", True)),
            tracker_enabled=bool(section.get("tracker_enabled", True)),
            memory_enabled=bool(section.get("memory_enabled", True)),
            geometry_enabled=bool(section.get("geometry_enabled", True)),
            sam3_url=str(section.get("sam3_url", "http://127.0.0.1:8773/sse")),
            sam3_python=str(
                section.get(
                    "sam3_python",
                    "/root/autodl-tmp/openeta-services/sam3/.venv/bin/python",
                )
            ),
            reacquire_every=int(section.get("reacquire_every", 4)),
            confidence_threshold=float(section.get("confidence_threshold", 0.5)),
            sam3_ambiguity_margin=float(section.get("sam3_ambiguity_margin", 0.05)),
            commit_guard_enabled=bool(section.get("commit_guard_enabled", False)),
            commit_guard_mode=str(section.get("commit_guard_mode", "shadow")),
            commit_guard_freshness_frames=int(
                section.get("commit_guard_freshness_frames", 1)
            ),
            commit_guard_alignment_px=float(section.get("commit_guard_alignment_px", 24.0)),
            alignment_ready_px=float(section.get("alignment_ready_px", 16.0)),
            approach_completion_guard_enabled=bool(
                section.get("approach_completion_guard_enabled", False)
            ),
            approach_completion_guard_mode=str(
                section.get("approach_completion_guard_mode", "shadow")
            ),
            approach_completion_guard_freshness_frames=int(
                section.get("approach_completion_guard_freshness_frames", 1)
            ),
            grasp_agentview_fallback_enabled=bool(
                section.get("grasp_agentview_fallback_enabled", False)
            ),
            grasp_agentview_guard_enabled=bool(
                section.get("grasp_agentview_guard_enabled", False)
            ),
            grasp_agentview_guard_mode=str(
                section.get("grasp_agentview_guard_mode", "shadow")
            ),
            grasp_agentview_guard_freshness_frames=int(
                section.get("grasp_agentview_guard_freshness_frames", 1)
            ),
            grasp_agentview_guard_alignment_px=float(
                section.get("grasp_agentview_guard_alignment_px", 12.0)
            ),
            grasp_agentview_guard_height_m=float(
                section.get("grasp_agentview_guard_height_m", 0.18)
            ),
            grasp_agentview_guard_min_height_m=(
                None
                if section.get("grasp_agentview_guard_min_height_m") is None
                else float(section.get("grasp_agentview_guard_min_height_m"))
            ),
            approach_completion_min_height_m=(
                None
                if section.get("approach_completion_min_height_m") is None
                else float(section.get("approach_completion_min_height_m"))
            ),
            approach_completion_max_height_m=(
                None
                if section.get("approach_completion_max_height_m") is None
                else float(section.get("approach_completion_max_height_m"))
            ),
            target_reference_height_m=(
                None
                if section.get("target_reference_height_m") is None
                else float(section.get("target_reference_height_m"))
            ),
            occlusion_hold_frames=int(section.get("occlusion_hold_frames", 6)),
        )

    def reset(self) -> None:
        if self.tracker is not None:
            self.tracker.reset()
        if self.opening_tracker is not None:
            self.opening_tracker.reset()
        if self.held_tracker is not None:
            self.held_tracker.reset()
        if self.memory is not None:
            self.memory.reset()
        self.last_stage_key = None
        self.last_stage_identity = None
        self.stage_camera_override = None
        self.last_frame_id = None
        self.last_grounding_frame = None
        self.last_action = None
        self._same_action_count = 0
        self.last_evidence = {}
        self.last_visible_evidence = {}
        self.held_stage_identity = None
        self.held_last_grounding_frame = None
        self.last_held_evidence = {}
        self._held_instance_anchor_bbox = None
        self.held_rim_anchor_y = None
        self.held_horizontal_stall = False
        self.opening_stage_identity = None
        self.opening_last_grounding_frame = None
        self.tool_calls = 0
        self.tool_failures = 0
        self.guard_shadow_events = 0
        self.guard_blocks = 0

    def _select_view(
        self,
        stage: str,
        agentview: np.ndarray,
        wrist: Optional[np.ndarray],
    ) -> tuple[str, np.ndarray]:
        if str(stage).upper() in {"GRASP", "RELEASE"} and wrist is not None:
            return "wrist", wrist
        return "agentview", agentview

    @staticmethod
    def _target_prompt(target: str, affordance: Optional[str] = None) -> str:
        """Build one bounded semantic query from planner-provided language.

        The harness does not invent a task-specific object rule.  An affordance
        is already part of the planner's semantic subgoal (for example, the
        "body of the bottle"), and combining it with the target gives an open
        vocabulary segmenter useful visual context while retaining ambiguity.
        """
        target_text = " ".join(str(target or "").split())
        affordance_text = " ".join(str(affordance or "").split())
        if target_text and affordance_text and affordance_text.lower() not in target_text.lower():
            text = f"{target_text}, {affordance_text}"
        else:
            text = target_text or affordance_text
        return text[:160] or "target object"

    @classmethod
    def _target_query_variants(
        cls, target: str, affordance: Optional[str] = None
    ) -> list[str]:
        """Return bounded SAM3 query fallbacks without inventing object identity.

        SAM3 is often better at a compact visual noun phrase than at a planner
        phrase containing task language (for example ``salad dressing``).  The
        first query remains the exact planner target plus affordance.  Fallbacks
        only compress the same supplied words: an attribute such as ``green
        cap`` becomes ``green-capped`` and is combined with the target head noun.
        A generic head-noun query is last and still goes through the normal
        ambiguity check, so it cannot silently select one of several objects.
        """
        primary = cls._target_prompt(target, affordance)
        target_words = [
            word.strip(".,;:()[]{}")
            for word in " ".join(str(target or "").split()).split()
            if word.strip(".,;:()[]{}")
        ]
        affordance_words = [
            word.strip(".,;:()[]{}")
            for word in " ".join(str(affordance or "").split()).split()
            if word.strip(".,;:()[]{}")
        ]
        variants = [primary]
        if target_words:
            head = target_words[-1].lower()
            ignored = {
                "cap",
                "capped",
                "lid",
                "top",
                "body",
                "color",
                "colour",
                "of",
                "the",
                "and",
            }
            attributes = [
                word.lower()
                for word in affordance_words
                if word.lower() not in ignored
            ]
            if attributes:
                attribute = "-".join(attributes)
                if any(word.lower() in {"cap", "capped"} for word in affordance_words):
                    variants.append(f"{attribute}-capped {head}")
                variants.append(f"{attribute} {head}")
            if head not in {word.lower() for word in variants}:
                variants.append(head)

        result: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            normalized = " ".join(str(variant).split())[:160]
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                result.append(normalized)
        return result or ["target object"]

    def _sam3_ground(
        self, image: np.ndarray, target: str, affordance: Optional[str] = None
    ) -> tuple[Optional[tuple[int, int, int, int]], float, str, dict[str, Any]]:
        if self.sam3 is None:
            return None, 0.0, "sam3_disabled", {}
        started = time.perf_counter()
        queries = self._target_query_variants(target, affordance)
        attempts: list[dict[str, Any]] = []
        best_candidates: list[tuple[float, tuple[int, int, int, int], dict[str, Any]]] = []
        best_details: dict[str, Any] | None = None
        best_query = queries[0]

        for query in queries:
            result = self.sam3.segment(
                image,
                query,
                confidence_threshold=self.confidence_threshold,
            )
            self.tool_calls += 1
            if not isinstance(result, dict):
                self.tool_failures += 1
                return None, 0.0, "sam3_abstain", {
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
                    "query": query,
                    "query_attempts": attempts,
                    "confidence_threshold": self.confidence_threshold,
                    "abstain_reason": "invalid_response",
                    "error": "sam3_response_not_object",
                }
            if not bool(result.get("success", False)):
                self.tool_failures += 1
                return None, 0.0, "sam3_abstain", {
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
                    "query": query,
                    "query_attempts": attempts,
                    "confidence_threshold": self.confidence_threshold,
                    "abstain_reason": "request_failed",
                    "error": result.get("error") or result.get("content") or "sam3_request_failed",
                }
            details = result.get("details")
            if not isinstance(details, dict):
                self.tool_failures += 1
                return None, 0.0, "sam3_abstain", {
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
                    "query": query,
                    "query_attempts": attempts,
                    "confidence_threshold": self.confidence_threshold,
                    "abstain_reason": "missing_details",
                    "error": "sam3_details_missing",
                }
            detections = details.get("detections")
            attempt = {
                "query": query,
                "detection_count": len(detections) if isinstance(detections, list) else None,
            }
            attempts.append(attempt)
            if not isinstance(detections, list) or not detections:
                continue

            candidates: list[tuple[float, tuple[int, int, int, int], dict[str, Any]]] = []
            for detection in detections:
                if not isinstance(detection, dict):
                    continue
                candidate_bbox = _box_tuple(detection.get("bbox_xyxy"))
                if candidate_bbox is None:
                    continue
                try:
                    candidate_score = float(detection.get("score") or 0.0)
                except (TypeError, ValueError):
                    candidate_score = 0.0
                candidates.append((candidate_score, candidate_bbox, detection))
            candidates.sort(key=lambda item: item[0], reverse=True)
            if not candidates:
                continue
            if not best_candidates or candidates[0][0] > best_candidates[0][0]:
                best_candidates = candidates
                best_details = details
                best_query = query
            top_score = float(candidates[0][0])
            ambiguous = bool(
                len(candidates) >= 2
                and top_score - float(candidates[1][0]) < self.sam3_ambiguity_margin
            )
            if top_score >= self.confidence_threshold and not ambiguous:
                break

        base_meta: dict[str, Any] = {
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "query": best_query,
            "query_attempts": attempts,
            "fallback_used": best_query != queries[0],
            "confidence_threshold": self.confidence_threshold,
        }
        if not best_candidates:
            self.tool_failures += 1
            return None, 0.0, "sam3_abstain", {
                **base_meta,
                "abstain_reason": "no_detections_after_backend_threshold",
                "backend_metadata": (best_details or {}).get("metadata"),
                "error": None,
            }

        candidates = best_candidates
        score, bbox, chosen = candidates[0]
        candidate_summary = [
            {
                "score": round(float(item[0]), 4),
                "bbox_xyxy": list(item[1]),
                "label": item[2].get("label"),
            }
            for item in candidates[:5]
        ]
        if (
            len(candidates) >= 2
            and float(candidates[0][0]) - float(candidates[1][0])
            < self.sam3_ambiguity_margin
        ):
            self.tool_failures += 1
            return None, score, "sam3_abstain", {
                **base_meta,
                "abstain_reason": "ambiguous_top_detections",
                "detection_count": len(candidates),
                "candidate_count": len(candidates),
                "ambiguity_margin": self.sam3_ambiguity_margin,
                "candidates": candidate_summary,
            }
        if bbox is None or score < self.confidence_threshold:
            self.tool_failures += 1
            reason = "invalid_bbox" if bbox is None else "score_below_harness_threshold"
            return None, score, "sam3_abstain", {
                **base_meta,
                "abstain_reason": reason,
                "detection_count": len(candidates),
                "candidate_count": len(candidates),
                "candidates": candidate_summary,
                "raw_score": score,
                "bbox_xyxy": chosen.get("bbox_xyxy"),
                "label": chosen.get("label"),
            }
        return bbox, score, "sam3", {
            **base_meta,
            "detection_count": len(detections),
            "candidate_count": len(candidates),
            "candidates": candidate_summary,
            "area_px": chosen.get("area_px"),
            "label": chosen.get("label"),
        }

    def _sam3_ground_opening(
        self, image: np.ndarray, target: str
    ) -> tuple[Optional[tuple[int, int, int, int]], float, str, dict[str, Any]]:
        """Ground a receptacle's interior without falling back to its outer body.

        A normal target grounding is intentionally allowed to use semantic fallbacks
        (``basket`` -> ``green basket`` etc.).  That is wrong for placement: the
        outer receptacle silhouette is not the usable opening.  These queries stay
        tied to the planner target but are restricted to interior/opening language,
        and the auxiliary threshold is treated as evidence rather than a semantic
        success decision.
        """
        if self.sam3 is None:
            return None, 0.0, "sam3_disabled", {}
        target_text = " ".join(str(target or "").split())[:120]
        if not target_text:
            return None, 0.0, "sam3_abstain", {"abstain_reason": "empty_target"}
        queries = [
            f"{target_text} interior",
            f"opening of the {target_text}",
            f"{target_text} opening",
        ]
        threshold = max(0.25, min(self.confidence_threshold, 0.35))
        started = time.perf_counter()
        attempts: list[dict[str, Any]] = []
        best: tuple[float, tuple[int, int, int, int], dict[str, Any], str] | None = None
        for query in queries:
            result = self.sam3.segment(
                image, query, confidence_threshold=threshold
            )
            self.tool_calls += 1
            if not isinstance(result, dict) or not bool(result.get("success", False)):
                self.tool_failures += 1
                attempts.append({"query": query, "detection_count": None})
                continue
            details = result.get("details")
            detections = details.get("detections") if isinstance(details, dict) else None
            attempts.append(
                {
                    "query": query,
                    "detection_count": (
                        len(detections) if isinstance(detections, list) else None
                    ),
                }
            )
            if not isinstance(detections, list):
                continue
            for detection in detections:
                if not isinstance(detection, dict):
                    continue
                candidate_bbox = _box_tuple(detection.get("bbox_xyxy"))
                if candidate_bbox is None:
                    continue
                try:
                    score = float(detection.get("score") or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                if best is None or score > best[0]:
                    best = (score, candidate_bbox, detection, query)
        meta: dict[str, Any] = {
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "query": best[3] if best is not None else queries[0],
            "query_attempts": attempts,
            "confidence_threshold": threshold,
            "auxiliary_role": "receptacle_opening",
        }
        if best is None:
            self.tool_failures += 1
            return None, 0.0, "sam3_abstain", {
                **meta,
                "abstain_reason": "no_opening_detection",
            }
        score, bbox, chosen, _query = best
        return bbox, score, "sam3_opening", {
            **meta,
            "detection_count": len(detections) if isinstance(detections, list) else 0,
            "bbox_xyxy": list(bbox),
            "label": chosen.get("label"),
            "area_px": chosen.get("area_px"),
        }

    def _update_held_object(
        self,
        *,
        agentview: np.ndarray,
        stage: str,
        held_target: Optional[str],
        held_affordance: Optional[str],
        frame_id: int,
        previous_action: Optional[str] = None,
        handoff_evidence: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Ground the carried object independently of the receptacle.

        The caller supplies the semantic identity from the earlier GRASP subgoal;
        the harness does not guess an object name from pixels.  SAM3 provides fresh
        grounding at bounded intervals and a separate short-lived tracker bridges
        the intervening live frames.  This is visual evidence only: it does not
        declare that the object is held and it does not choose an action.
        """
        stage_name = str(stage).upper()
        if stage_name not in {"MOVE", "PLACE", "TRANSPORT"} or not str(held_target or "").strip():
            self.held_stage_identity = None
            self.held_last_grounding_frame = None
            self.last_held_evidence = {}
            self._held_instance_anchor_bbox = None
            self.held_rim_anchor_y = None
            self.held_horizontal_stall = False
            if self.held_tracker is not None:
                self.held_tracker.reset()
            return None

        held_prompt = self._target_prompt(held_target or "", held_affordance)
        identity = f"{stage_name}::{held_prompt}"
        if identity != self.held_stage_identity:
            self.held_stage_identity = identity
            self.held_last_grounding_frame = None
            self.last_held_evidence = {}
            self._held_instance_anchor_bbox = None
            self.held_rim_anchor_y = None
            self.held_horizontal_stall = False
            if self.held_tracker is not None:
                self.held_tracker.reset()
            # Bind TRANSPORT to the exact AgentView instance that converged in
            # GRASP.  A new semantic query may contain several same-category
            # objects and cannot by itself identify which one entered the hand.
            handoff_bbox = _box_tuple(
                (handoff_evidence or {}).get("bbox_xyxy")
                if isinstance(handoff_evidence, dict)
                else None
            )
            if stage_name == "TRANSPORT" and handoff_bbox is not None:
                self._held_instance_anchor_bbox = handoff_bbox
                self.last_held_evidence = {
                    "bbox_xyxy": list(handoff_bbox),
                    "confidence": float((handoff_evidence or {}).get("confidence", 0.0) or 0.0),
                    "source": "grasp_agentview_handoff",
                    "visible": True,
                }
                if self.held_tracker is not None:
                    self.held_tracker.seed(
                        agentview,
                        handoff_bbox,
                        float((handoff_evidence or {}).get("confidence", 0.0) or 0.0),
                    )

        should_ground = (
            self.held_last_grounding_frame is None
            or frame_id - int(self.held_last_grounding_frame) >= self.reacquire_every
            or (self.held_tracker is None)
            or getattr(self.held_tracker, "state", None) is None
        )
        bbox: Optional[tuple[int, int, int, int]] = None
        confidence = 0.0
        source = "unknown"
        tool: dict[str, Any] = {}
        if should_ground:
            bbox, confidence, source, tool = self._sam3_ground(
                agentview, held_target or "", held_affordance
            )
            self.held_last_grounding_frame = int(frame_id)
            # A periodic semantic query must not jump from the carried object to
            # another same-category detection elsewhere in the scene.  The held
            # object can move only through the bounded visual neighborhood between
            # two live frames; a far candidate is rejected and the short-term
            # tracker gets first chance to preserve continuity.  If both sources
            # fail, report unknown so Qwen does not act on a stale location.
            previous_bbox = _box_tuple(
                self.last_held_evidence.get("bbox_xyxy")
                if isinstance(self.last_held_evidence, dict)
                else None
            ) or self._held_instance_anchor_bbox
            # Associate all semantic candidates to the live instance track,
            # rather than accepting the detector's highest-confidence object.
            # This is category-agnostic and prevents a periodic refresh from
            # jumping to a visually similar object elsewhere in the scene.
            candidates = tool.get("candidates") if isinstance(tool, dict) else None
            if previous_bbox is not None and isinstance(candidates, list):
                previous_center = np.asarray(
                    [
                        (float(previous_bbox[0]) + float(previous_bbox[2])) / 2.0,
                        (float(previous_bbox[1]) + float(previous_bbox[3])) / 2.0,
                    ]
                )
                associated: list[tuple[float, tuple[int, int, int, int], float]] = []
                for candidate in candidates:
                    candidate_bbox = _box_tuple(
                        candidate.get("bbox_xyxy") if isinstance(candidate, dict) else None
                    )
                    if candidate_bbox is None:
                        continue
                    candidate_center = np.asarray(
                        [
                            (float(candidate_bbox[0]) + float(candidate_bbox[2])) / 2.0,
                            (float(candidate_bbox[1]) + float(candidate_bbox[3])) / 2.0,
                        ]
                    )
                    associated.append(
                        (
                            float(np.linalg.norm(candidate_center - previous_center)),
                            candidate_bbox,
                            float(candidate.get("score", 0.0) or 0.0),
                        )
                    )
                if associated:
                    distance_px, associated_bbox, associated_confidence = min(
                        associated, key=lambda item: item[0]
                    )
                    previous_diagonal = float(
                        np.linalg.norm(
                            [
                                previous_bbox[2] - previous_bbox[0],
                                previous_bbox[3] - previous_bbox[1],
                            ]
                        )
                    )
                    association_limit_px = max(8.0, 0.6 * previous_diagonal)
                    tool = dict(tool)
                    tool["instance_association"] = {
                        "reference_bbox_xyxy": list(previous_bbox),
                        "selected_bbox_xyxy": list(associated_bbox),
                        "distance_px": round(distance_px, 2),
                        "limit_px": round(association_limit_px, 2),
                    }
                    if distance_px <= association_limit_px:
                        bbox = associated_bbox
                        confidence = associated_confidence
                        source = "sam3_instance_associated"
                    else:
                        bbox = None
                        confidence = 0.0
                        source = "instance_association_rejected"
            if bbox is not None and previous_bbox is not None:
                previous_center = (
                    (float(previous_bbox[0]) + float(previous_bbox[2])) / 2.0,
                    (float(previous_bbox[1]) + float(previous_bbox[3])) / 2.0,
                )
                candidate_center = (
                    (float(bbox[0]) + float(bbox[2])) / 2.0,
                    (float(bbox[1]) + float(bbox[3])) / 2.0,
                )
                jump_px = float(
                    np.linalg.norm(
                        np.asarray(candidate_center) - np.asarray(previous_center)
                    )
                )
                previous_size = max(
                    float(previous_bbox[2] - previous_bbox[0]),
                    float(previous_bbox[3] - previous_bbox[1]),
                )
                max_jump_px = max(32.0, 1.75 * previous_size)
                if jump_px > max_jump_px:
                    rejected_bbox = bbox
                    tool = dict(tool)
                    tool["temporal_gate"] = {
                        "rejected": True,
                        "reason": "semantic_reacquire_jump",
                        "previous_bbox_xyxy": list(previous_bbox),
                        "candidate_bbox_xyxy": list(rejected_bbox),
                        "jump_px": round(jump_px, 2),
                        "max_jump_px": round(max_jump_px, 2),
                    }
                    bbox = None
                    confidence = 0.0
                    source = "temporal_gate_rejected"
                    if self.held_tracker is not None:
                        tracked_after_reject = self.held_tracker.update(agentview)
                        if tracked_after_reject is not None:
                            bbox = _box_tuple(
                                tracked_after_reject.get("bbox_xyxy")
                            )
                            confidence = float(
                                tracked_after_reject.get("confidence", 0.0)
                            )
                            source = "temporal_gate_tracker"
                            tool["temporal_gate"]["tracker_fallback"] = True
                        else:
                            self.held_last_grounding_frame = None
                            tool["temporal_gate"]["tracker_fallback"] = False
            if bbox is not None and self.held_tracker is not None:
                self.held_tracker.seed(agentview, bbox, confidence)
        elif self.held_tracker is not None:
            tracked = self.held_tracker.update(agentview)
            if tracked is not None:
                bbox = _box_tuple(tracked.get("bbox_xyxy"))
                confidence = float(tracked.get("confidence", 0.0))
                source = str(tracked.get("source", "cpu_template_tracker"))
                tool = {
                    "match_score": tracked.get("score"),
                    "track_age": tracked.get("track_age"),
                }
            else:
                # Force a semantic reacquisition on the next frame rather than
                # carrying a stale object location through a placement decision.
                self.held_last_grounding_frame = None
                source = "tracker_lost"

        result: dict[str, Any] = {
            "target": held_prompt,
            "bbox_xyxy": list(bbox) if bbox is not None else None,
            "confidence": round(float(confidence), 4),
            "source": source,
            "visible": bbox is not None,
            "grounding_frame": self.held_last_grounding_frame,
            "tool": tool,
        }
        previous_held_bbox = _box_tuple(
            self.last_held_evidence.get("bbox_xyxy")
            if isinstance(self.last_held_evidence, dict)
            else None
        )
        current_held_bbox = bbox
        previous_action_name = str(previous_action or "").strip().upper()
        if previous_held_bbox is not None and current_held_bbox is not None:
            previous_center_x = (
                float(previous_held_bbox[0]) + float(previous_held_bbox[2])
            ) / 2.0
            current_center_x = (
                float(current_held_bbox[0]) + float(current_held_bbox[2])
            ) / 2.0
            horizontal_delta_px = current_center_x - previous_center_x
            if previous_action_name in {
                "MV_LEFT",
                "MV_RIGHT",
                "MV_FWD",
                "MV_BACK",
            }:
                # A horizontal/depth correction that leaves the carried body's
                # image center unchanged is contact/loss evidence.  Keep it until
                # a later action produces visible object motion; this is not a
                # semantic object-size or scene-coordinate rule.
                self.held_horizontal_stall = abs(horizontal_delta_px) < 1.0
            elif abs(horizontal_delta_px) >= 1.0:
                self.held_horizontal_stall = False
            result["horizontal_delta_px"] = round(horizontal_delta_px, 2)
        result["previous_action"] = previous_action_name or None
        result["horizontal_motion_stalled"] = bool(self.held_horizontal_stall)
        if bbox is not None:
            self._held_instance_anchor_bbox = bbox
        result["instance_anchor_bbox_xyxy"] = (
            list(self._held_instance_anchor_bbox)
            if self._held_instance_anchor_bbox is not None
            else None
        )
        self.last_held_evidence = result
        return dict(result)

    @staticmethod
    def _view_relation(
        bbox: Optional[tuple[int, int, int, int]],
        confidence: float,
        source: str,
        geometry: Any,
    ) -> dict[str, Any]:
        """Describe one camera's target-to-EEF image relation without choosing an action."""
        result: dict[str, Any] = {
            "bbox_xyxy": list(bbox) if bbox is not None else None,
            "confidence": round(float(confidence), 4),
            "source": str(source),
            "visible": bbox is not None,
        }
        if bbox is None or not isinstance(geometry, dict):
            return result
        if geometry.get("eef_height_m") is not None:
            result["eef_height_m"] = geometry.get("eef_height_m")
        projected = geometry.get("pixel_xy")
        if not isinstance(projected, (list, tuple)) or len(projected) != 2:
            return result
        center = [
            (float(bbox[0]) + float(bbox[2])) / 2.0,
            (float(bbox[1]) + float(bbox[3])) / 2.0,
        ]
        error = [center[0] - float(projected[0]), center[1] - float(projected[1])]
        relation = {
            "horizontal": "right" if error[0] > 3 else "left" if error[0] < -3 else "aligned",
            "vertical": "down" if error[1] > 3 else "up" if error[1] < -3 else "aligned",
        }
        result.update(
            {
                "center_xy": [round(v, 2) for v in center],
                "eef_pixel_xy": [round(float(projected[0]), 2), round(float(projected[1]), 2)],
                "target_minus_eef_px": [round(v, 2) for v in error],
                "screen_relation": relation,
                "correction_candidates": {
                    "horizontal": (geometry.get("screen_direction_to_token") or {}).get(
                        relation["horizontal"]
                    ),
                    "vertical": (geometry.get("screen_direction_to_token") or {}).get(
                        relation["vertical"]
                    ),
                },
            }
        )
        return result

    def ground_auxiliary(
        self,
        *,
        agentview: np.ndarray,
        target: str,
        affordance: Optional[str] = None,
        include_opening: bool = False,
    ) -> dict[str, Any]:
        """Ground an auxiliary carried-object/destination pair without mutating tracks.

        Route planning needs both the object and its future receptacle at the
        instant a GRASP hands control to LIFT.  The normal stage update only
        grounds the current subgoal, so this small read-only helper performs the
        additional SAM3 queries and returns evidence in a stable shape.  It does
        not seed the regular destination/held trackers and never changes the raw
        frame used by perception.
        """
        if not self.enabled:
            return {"bbox_xyxy": None, "confidence": 0.0, "source": "disabled"}
        bbox, confidence, source, meta = self._sam3_ground(
            agentview, str(target or ""), affordance
        )
        result: dict[str, Any] = {
            "bbox_xyxy": list(bbox) if bbox is not None else None,
            "confidence": round(float(confidence), 4),
            "source": source,
            "tool": meta,
        }
        if include_opening:
            opening, opening_confidence, opening_source, opening_meta = (
                self._sam3_ground_opening(agentview, str(target or ""))
            )
            result.update(
                {
                    "opening_bbox_xyxy": (
                        list(opening) if opening is not None else None
                    ),
                    "opening_confidence": round(float(opening_confidence), 4),
                    "opening_source": opening_source,
                    "opening_tool": opening_meta,
                }
            )
        return result

    def update(
        self,
        *,
        agentview: np.ndarray,
        wrist: Optional[np.ndarray],
        stage: str,
        target: str,
        affordance: Optional[str] = None,
        frame_id: int,
        previous_action: Optional[str] = None,
        geometry: Optional[dict[str, Any]] = None,
        held_target: Optional[str] = None,
        held_affordance: Optional[str] = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        previous_secondary_view = self.last_evidence.get("secondary_view")
        target_prompt = self._target_prompt(target, affordance)
        stage_identity = f"{str(stage).upper()}::{target_prompt}"
        if stage_identity != self.last_stage_identity:
            self.stage_camera_override = None
            self._same_action_count = 0
            self.opening_stage_identity = None
            self.opening_last_grounding_frame = None
            if self.opening_tracker is not None:
                self.opening_tracker.reset()
        self.last_stage_identity = stage_identity
        camera, image = self._select_view(stage, agentview, wrist)
        if self.stage_camera_override == "agentview":
            camera, image = "agentview", agentview
        stage_key = f"{str(stage).upper()}::{target_prompt}::{camera}"
        stage_changed = stage_key != self.last_stage_key
        if stage_changed and self.memory is not None and self.last_stage_key is not None:
            self.memory.invalidate_scene("stage_changed")
        should_ground = stage_changed or self.last_grounding_frame is None
        if self.last_evidence and not bool(self.last_evidence.get("visible", False)):
            should_ground = True
        if (
            not should_ground
            and self.last_frame_id is not None
            and frame_id - self.last_grounding_frame >= self.reacquire_every
        ):
            should_ground = True

        bbox = None
        confidence = 0.0
        source = "unknown"
        tool_meta: dict[str, Any] = {}
        secondary_camera: Optional[str] = None
        secondary_bbox: Optional[tuple[int, int, int, int]] = None
        secondary_confidence = 0.0
        secondary_source = "unknown"
        secondary_meta: dict[str, Any] = {}
        # Preserve the current semantic instance across periodic detector refreshes.
        # A periodic SAM3 query is evidence refresh, not permission to switch to a
        # different same-category object. Reset the tracker only when the actual
        # stage/target/camera identity changes.
        previous_bbox_for_refresh = None
        if not stage_changed and isinstance(self.last_evidence, dict):
            previous_bbox_for_refresh = _box_tuple(
                self.last_evidence.get("bbox_xyxy")
            )

        if should_ground:
            if self.tracker is not None and stage_changed:
                self.tracker.reset()
            probe_wrist = bool(
                self.grasp_agentview_fallback_enabled
                and self.stage_camera_override == "agentview"
                and camera == "agentview"
                and wrist is not None
                and str(stage).upper() in {"GRASP", "RELEASE"}
            )
            if probe_wrist:
                wrist_bbox, wrist_confidence, wrist_source, wrist_meta = self._sam3_ground(
                    wrist, target, affordance
                )
                if wrist_bbox is not None:
                    self.stage_camera_override = "wrist"
                    camera, image = "wrist", wrist
                    stage_key = f"{str(stage).upper()}::{target_prompt}::wrist"
                    bbox, confidence, source, tool_meta = (
                        wrist_bbox,
                        wrist_confidence,
                        wrist_source,
                        {
                            "probe_camera": "wrist",
                            "probe": wrist_meta,
                            "selected_camera": "wrist",
                        },
                    )
                else:
                    bbox, confidence, source, agentview_meta = self._sam3_ground(
                        agentview, target, affordance
                    )
                    tool_meta = {
                        "probe_camera": "wrist",
                        "probe": wrist_meta,
                        "selected_camera": "agentview",
                        "fallback_reason": "wrist_no_detection",
                        "fallback": agentview_meta,
                    }
            else:
                bbox, confidence, source, tool_meta = self._sam3_ground(
                    image, target, affordance
                )
                if (
                    bbox is None
                    and camera == "wrist"
                    and str(stage).upper() in {"GRASP", "RELEASE"}
                    and self.grasp_agentview_fallback_enabled
                ):
                    fallback_bbox, fallback_confidence, fallback_source, fallback_meta = (
                        self._sam3_ground(agentview, target, affordance)
                    )
                    combined_meta = {
                        "primary_camera": "wrist",
                        "primary": tool_meta,
                        "fallback_camera": "agentview",
                        "fallback": fallback_meta,
                        "fallback_reason": "wrist_no_detection",
                    }
                    if fallback_bbox is not None:
                        self.stage_camera_override = "agentview"
                        camera, image = "agentview", agentview
                        stage_key = f"{str(stage).upper()}::{target_prompt}::agentview"
                        bbox = fallback_bbox
                        confidence = fallback_confidence
                        source = fallback_source
                    tool_meta = combined_meta
            # During APPROACH, periodic semantic refresh must remain attached to
            # the already tracked physical instance. SAM3 may return several objects
            # of the same category, or a fallback query such as "bottle" may rank a
            # different object highest. Associate candidates to the previous live
            # bbox instead of silently switching identity.
            if (
                not stage_changed
                and str(stage).upper() == "APPROACH"
                and camera == "agentview"
                and previous_bbox_for_refresh is not None
            ):
                candidates = (
                    tool_meta.get("candidates")
                    if isinstance(tool_meta, dict)
                    else None
                )

                if isinstance(candidates, list) and candidates:
                    previous_center = np.asarray(
                        [
                            (
                                float(previous_bbox_for_refresh[0])
                                + float(previous_bbox_for_refresh[2])
                            )
                            / 2.0,
                            (
                                float(previous_bbox_for_refresh[1])
                                + float(previous_bbox_for_refresh[3])
                            )
                            / 2.0,
                        ],
                        dtype=float,
                    )

                    associated = []
                    for candidate in candidates:
                        if not isinstance(candidate, dict):
                            continue
                        candidate_bbox = _box_tuple(candidate.get("bbox_xyxy"))
                        if candidate_bbox is None:
                            continue

                        candidate_center = np.asarray(
                            [
                                (
                                    float(candidate_bbox[0])
                                    + float(candidate_bbox[2])
                                )
                                / 2.0,
                                (
                                    float(candidate_bbox[1])
                                    + float(candidate_bbox[3])
                                )
                                / 2.0,
                            ],
                            dtype=float,
                        )

                        associated.append(
                            (
                                float(
                                    np.linalg.norm(
                                        candidate_center - previous_center
                                    )
                                ),
                                candidate_bbox,
                                float(candidate.get("score", 0.0) or 0.0),
                            )
                        )

                    if associated:
                        distance_px, associated_bbox, associated_confidence = min(
                            associated, key=lambda item: item[0]
                        )

                        previous_diagonal = float(
                            np.linalg.norm(
                                [
                                    previous_bbox_for_refresh[2]
                                    - previous_bbox_for_refresh[0],
                                    previous_bbox_for_refresh[3]
                                    - previous_bbox_for_refresh[1],
                                ]
                            )
                        )
                        association_limit_px = max(
                            8.0, 0.6 * previous_diagonal
                        )

                        tool_meta = dict(tool_meta)
                        tool_meta["instance_association"] = {
                            "reference_bbox_xyxy": list(
                                previous_bbox_for_refresh
                            ),
                            "selected_bbox_xyxy": list(associated_bbox),
                            "distance_px": round(distance_px, 2),
                            "limit_px": round(association_limit_px, 2),
                        }

                        if distance_px <= association_limit_px:
                            bbox = associated_bbox
                            confidence = associated_confidence
                            source = "sam3_instance_associated"
                        else:
                            # Detector wants to jump to another object. Keep the
                            # existing temporal track when it is still valid.
                            tracked_after_reject = (
                                self.tracker.update(image)
                                if self.tracker is not None
                                else None
                            )

                            tool_meta["instance_association"][
                                "semantic_jump_rejected"
                            ] = True

                            if tracked_after_reject is not None:
                                bbox = _box_tuple(
                                    tracked_after_reject.get("bbox_xyxy")
                                )
                                confidence = float(
                                    tracked_after_reject.get(
                                        "confidence", 0.0
                                    )
                                )
                                source = (
                                    "semantic_refresh_rejected_tracker"
                                )
                                tool_meta["instance_association"][
                                    "tracker_fallback"
                                ] = True
                            else:
                                bbox = None
                                confidence = 0.0
                                source = "instance_association_rejected"
                                tool_meta["instance_association"][
                                    "tracker_fallback"
                                ] = False

            if bbox is not None and self.tracker is not None:
                self.tracker.seed(image, bbox, confidence)
        elif self.tracker is not None:
            tracked = self.tracker.update(image)
            if tracked is not None:
                bbox = _box_tuple(tracked.get("bbox_xyxy"))
                confidence = float(tracked.get("confidence", 0.0))
                source = str(tracked.get("source", "cpu_template_tracker"))
                tool_meta = {"match_score": tracked.get("score"), "track_age": tracked.get("track_age")}
            else:
                source = "tracker_lost"
        else:
            source = "tracking_disabled"

        # For placement, replace the outer receptacle silhouette with a separate
        # opening track whenever the semantic opening query is available.  Keep the
        # outer bbox in metadata for obstacle/context inspection, but never use it
        # as the primary object-to-opening alignment target.
        outer_destination_bbox = bbox
        opening_bbox: Optional[tuple[int, int, int, int]] = None
        opening_confidence = 0.0
        opening_source = "opening_unknown"
        opening_meta: dict[str, Any] = {}
        if str(stage).upper() in {"MOVE", "PLACE", "TRANSPORT"} and camera == "agentview":
            opening_should_ground = bool(
                self.opening_last_grounding_frame is None
                or frame_id - int(self.opening_last_grounding_frame)
                >= self.reacquire_every
                or self.opening_tracker is None
                or getattr(self.opening_tracker, "state", None) is None
            )
            if opening_should_ground:
                (
                    opening_bbox,
                    opening_confidence,
                    opening_source,
                    opening_meta,
                ) = self._sam3_ground_opening(agentview, target)
                self.opening_last_grounding_frame = int(frame_id)
                if opening_bbox is not None and self.opening_tracker is not None:
                    self.opening_tracker.seed(
                        agentview, opening_bbox, opening_confidence
                    )
            elif self.opening_tracker is not None:
                tracked_opening = self.opening_tracker.update(agentview)
                if tracked_opening is not None:
                    opening_bbox = _box_tuple(tracked_opening.get("bbox_xyxy"))
                    opening_confidence = float(
                        tracked_opening.get("confidence", 0.0)
                    )
                    opening_source = "cpu_template_tracker_opening"
                    opening_meta = {
                        "match_score": tracked_opening.get("score"),
                        "track_age": tracked_opening.get("track_age"),
                    }
                else:
                    self.opening_last_grounding_frame = None
                    opening_source = "opening_tracker_lost"
            if opening_bbox is not None:
                if isinstance(tool_meta, dict):
                    tool_meta = dict(tool_meta)
                    tool_meta["outer_destination_bbox_xyxy"] = (
                        list(outer_destination_bbox)
                        if outer_destination_bbox is not None
                        else None
                    )
                    tool_meta["opening"] = opening_meta
                bbox = opening_bbox
                confidence = opening_confidence
                source = opening_source

        # GRASP is the one stage where the two cameras answer different spatial
        # questions: AgentView gives global XY relation, while Wrist gives local
        # relation to the finger entry/gap. Ground both when possible and expose the
        # pair as evidence; neither camera is allowed to overwrite the other.
        if should_ground and str(stage).upper() == "GRASP" and wrist is not None:
            secondary_camera = "agentview" if camera == "wrist" else "wrist"
            secondary_image = agentview if secondary_camera == "agentview" else wrist
            (
                secondary_bbox,
                secondary_confidence,
                secondary_source,
                secondary_meta,
            ) = self._sam3_ground(secondary_image, target, affordance)
            tool_meta["secondary_camera"] = secondary_camera
            tool_meta["secondary"] = secondary_meta

        if bbox is None:
            confidence = 0.0
            if self.memory is not None and self.last_evidence.get("visible", False):
                self.memory.invalidate_scene("target_lost")

            # A gripper can occlude the target exactly at the useful pre-grasp
            # pose.  Preserve only a very recent same-target bbox, explicitly
            # marked as memory rather than a new SAM3 detection.
            previous = self.last_visible_evidence
            previous_frame = previous.get("frame_id") if isinstance(previous, dict) else None
            previous_target = previous.get("target") if isinstance(previous, dict) else None
            try:
                occlusion_age = int(frame_id) - int(previous_frame)
            except (TypeError, ValueError):
                occlusion_age = self.occlusion_hold_frames + 1
            previous_bbox = _box_tuple(previous.get("bbox_xyxy")) if isinstance(previous, dict) else None
            can_hold = bool(
                self.occlusion_hold_frames > 0
                and previous_bbox is not None
                and previous_target == target_prompt
                and 0 <= occlusion_age <= self.occlusion_hold_frames
            )
            if can_hold:
                bbox = previous_bbox
                confidence = float(previous.get("confidence", 0.0) or 0.0)
                source = "occlusion_memory"
                tool_meta = {
                    "from_frame": int(previous_frame),
                    "age_frames": occlusion_age,
                    "reason": "recent_target_occluded_by_gripper",
                }
                if (
                    str(stage).upper() in {"GRASP", "RELEASE"}
                    and self.grasp_agentview_fallback_enabled
                    and previous.get("camera") == "agentview"
                ):
                    camera, image = "agentview", agentview
                    self.stage_camera_override = "agentview"
                    stage_key = f"{str(stage).upper()}::{target_prompt}::agentview"

        if should_ground and source != "occlusion_memory":
            # Freshness refers to a real SAM3/tracker grounding event, not to a
            # carried bbox.  The stage guards separately accept bounded occlusion.
            self.last_grounding_frame = int(frame_id)
        if self.memory is not None:
            entry = self.memory.append(
                frame_id=int(frame_id),
                camera=camera,
                image_hash_value=image_sha1(image),
                stage=str(stage),
                target=target_prompt,
                action=previous_action,
                bbox_xyxy=bbox,
                confidence=confidence,
                source=source,
            )
            progress = entry.progress_delta_px
        else:
            progress = None

        # In MOVE/PLACE, the destination bbox above and the carried-object bbox
        # answer different questions.  Compare their image-space relation before
        # deriving any destination safety cue; using the EEF projection alone can
        # report "inside" while the bottle/body is still visibly outside the rim.
        held_object = self._update_held_object(
            agentview=agentview,
            stage=stage,
            held_target=held_target,
            held_affordance=held_affordance,
            frame_id=int(frame_id),
            previous_action=previous_action,
            handoff_evidence=(
                previous_secondary_view
                if isinstance(previous_secondary_view, dict)
                and previous_secondary_view.get("camera") == "agentview"
                else None
            ),
        )
        geometry_evidence = {}
        if self.geometry_enabled and isinstance(geometry, dict):
            candidate = geometry.get(camera)
            if isinstance(candidate, dict):
                geometry_evidence = dict(candidate)
        projected = geometry_evidence.get("pixel_xy")
        if bbox is not None and isinstance(projected, (list, tuple)) and len(projected) == 2:
            target_center = [
                (float(bbox[0]) + float(bbox[2])) / 2.0,
                (float(bbox[1]) + float(bbox[3])) / 2.0,
            ]
            error = [target_center[0] - float(projected[0]), target_center[1] - float(projected[1])]
            geometry_evidence["target_center_xy"] = [round(v, 2) for v in target_center]
            geometry_evidence["target_minus_eef_px"] = [round(v, 2) for v in error]
            geometry_evidence["target_screen_relation"] = {
                "horizontal": "right" if error[0] > 3 else "left" if error[0] < -3 else "aligned",
                "vertical": "down" if error[1] > 3 else "up" if error[1] < -3 else "aligned",
            }
            direction_map = geometry_evidence.get("screen_direction_to_token")
            if isinstance(direction_map, dict):
                relation = geometry_evidence["target_screen_relation"]
                geometry_evidence["calibrated_correction_candidates"] = {
                    "horizontal": direction_map.get(relation["horizontal"]),
                    "vertical": direction_map.get(relation["vertical"]),
                }
            if str(stage).upper() in {"APPROACH", "MOVE", "PLACE", "TRANSPORT"}:
                geometry_evidence["alignment_ready"] = bool(
                    max(abs(float(error[0])), abs(float(error[1])))
                    <= self.alignment_ready_px
                )
                geometry_evidence["alignment_ready_threshold_px"] = self.alignment_ready_px

            # A calibrated reference-height back-projection separates world-X
            # approach error from the screen-down parallax caused by the object
            # being below the EEF.  The reference height is configuration, not a
            # read of the simulator object pose.
            # A bbox touching an image edge is only a partial observation.  Its
            # center is not a valid world-plane target for a Cartesian correction;
            # this matters especially for a receptacle at the edge of AgentView.
            bbox_complete = False
            try:
                bbox_complete = (
                    float(bbox[0]) > 2.0
                    and float(bbox[1]) > 2.0
                    and float(bbox[2]) < float(geometry_evidence.get("image_size", [256, 256])[0]) - 2.0
                    and float(bbox[3]) < float(geometry_evidence.get("image_size", [256, 256])[1]) - 2.0
                )
            except (TypeError, ValueError, IndexError):
                bbox_complete = False

            calibration_meta = geometry_evidence.get("camera_calibration")
            eef_world = geometry_evidence.get("eef_position_xyz")
            if (
                self.target_reference_height_m is not None
                and str(stage).upper() != "TRANSPORT"
                and isinstance(calibration_meta, dict)
                and isinstance(eef_world, (list, tuple))
                and len(eef_world) == 3
                and (
                    str(stage).upper() != "PLACE"
                    or bbox_complete
                )
            ):
                try:
                    calibration = CameraCalibration(
                        name=str(camera),
                        width=int(calibration_meta["width"]),
                        height=int(calibration_meta["height"]),
                        fovy_deg=float(calibration_meta["fovy_deg"]),
                        position_world=np.asarray(
                            calibration_meta["position_world"], dtype=float
                        ),
                        camera_to_world=np.asarray(
                            calibration_meta["camera_to_world"], dtype=float
                        ),
                        rotation_degrees=int(calibration_meta.get("rotation_degrees", 0)),
                        flip=str(calibration_meta.get("flip", "none")),
                    )
                    target_world = backproject_pixel_to_plane(
                        calibration,
                        target_center,
                        self.target_reference_height_m,
                    )
                except (TypeError, ValueError, KeyError, IndexError):
                    target_world = None
                if target_world is not None:
                    eef_world_array = np.asarray(eef_world, dtype=float)
                    xy_error = target_world[:2] - eef_world_array[:2]
                    geometry_evidence["target_reference_height_m"] = (
                        self.target_reference_height_m
                    )
                    geometry_evidence["target_xy_world_estimate"] = [
                        round(float(value), 4) for value in target_world
                    ]
                    geometry_evidence["target_minus_eef_xy_m"] = [
                        round(float(value), 4) for value in xy_error
                    ]
                    geometry_evidence["target_bbox_complete"] = bbox_complete
        secondary_evidence = None
        if secondary_camera is not None:
            secondary_geometry = (
                geometry.get(secondary_camera)
                if isinstance(geometry, dict)
                else None
            )
            secondary_evidence = self._view_relation(
                secondary_bbox,
                secondary_confidence,
                secondary_source,
                secondary_geometry,
            )
            secondary_evidence["camera"] = secondary_camera
            secondary_evidence["tool"] = secondary_meta
        elif str(stage).upper() == "GRASP" and isinstance(previous_secondary_view, dict):
            secondary_evidence = dict(previous_secondary_view)
            try:
                secondary_evidence["age_frames"] = int(frame_id) - int(
                    self.last_evidence.get("frame_id")
                )
            except (TypeError, ValueError):
                secondary_evidence["age_frames"] = None

        # A repeated action with a stable local target image is useful evidence in
        # GRASP: it says that the hand/object projection has converged, not that the
        # semantic close is automatically authorized.  Keep this camera- and object-
        # agnostic; the runner exposes it to Qwen as a reflection cue only.
        grasp_spatial_convergence: dict[str, Any] = {
            "eligible": False,
            "reason": "not_grasp_convergence",
            "same_action_count": int(self._same_action_count),
        }
        if str(stage).upper() == "GRASP" and bbox is not None:
            image_size = geometry_evidence.get("image_size", [256, 256])
            bbox_complete = False
            try:
                bbox_complete = (
                    float(bbox[0]) > 2.0
                    and float(bbox[1]) > 2.0
                    and float(bbox[2]) < float(image_size[0]) - 2.0
                    and float(bbox[3]) < float(image_size[1]) - 2.0
                )
            except (TypeError, ValueError, IndexError):
                bbox_complete = False
            local_progress = progress
            try:
                local_progress_value = (
                    None if local_progress is None else float(local_progress)
                )
            except (TypeError, ValueError):
                local_progress_value = None
            secondary_alignment = (
                secondary_evidence.get("target_minus_eef_px")
                if isinstance(secondary_evidence, dict)
                else None
            )
            secondary_alignment_known = bool(
                isinstance(secondary_alignment, (list, tuple))
                and len(secondary_alignment) == 2
            )
            secondary_aligned = bool(
                secondary_alignment_known
                and max(
                    abs(float(secondary_alignment[0])),
                    abs(float(secondary_alignment[1])),
                )
                <= self.alignment_ready_px
            )
            motion_threshold = max(3.0, self.alignment_ready_px * 0.30)
            stable_local_view = bool(
                local_progress_value is not None
                and local_progress_value <= motion_threshold
            )
            repeated_motion = bool(
                self._same_action_count >= 3
                and str(previous_action or "").strip().upper()
                in {"MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN"}
            )
            eligible = bool(
                self.last_grounding_frame is not None
                and bbox_complete
                and secondary_aligned
                and stable_local_view
                and repeated_motion
            )
            grasp_spatial_convergence = {
                "eligible": eligible,
                "reason": (
                    "global_agentview_aligned_and_local_wrist_converged"
                    if eligible
                    else "insufficient_dual_view_temporal_convergence"
                ),
                "primary_camera": camera,
                "primary_bbox_complete": bbox_complete,
                "local_progress_px": (
                    round(local_progress_value, 2)
                    if local_progress_value is not None
                    else None
                ),
                "local_motion_threshold_px": round(motion_threshold, 2),
                "secondary_camera": (
                    secondary_evidence.get("camera")
                    if isinstance(secondary_evidence, dict)
                    else None
                ),
                "secondary_alignment_px": (
                    list(secondary_alignment)
                    if secondary_alignment_known
                    else None
                ),
                "secondary_alignment_threshold_px": self.alignment_ready_px,
                "same_action": str(previous_action or "").strip().upper() or None,
                "same_action_count": int(self._same_action_count),
            }

        # During horizontal transit, the destination bbox is also a geometric
        # hazard boundary.  Once the calibrated EEF projection enters or nearly
        # enters that visible bbox, the held object's lowest point can meet a rim
        # before the center alignment is complete.  This is only a visual-safety
        # cue; it does not select MV_UP or any other action.
        destination_proximity: dict[str, Any] = {
            "eligible": False,
            "reason": "not_destination_proximity",
        }
        if str(stage).upper() in {"MOVE", "PLACE", "TRANSPORT"} and bbox is not None:
            projected = geometry_evidence.get("pixel_xy")
            if isinstance(projected, (list, tuple)) and len(projected) == 2:
                try:
                    eef_x, eef_y = float(projected[0]), float(projected[1])
                    x0, y0, x1, y1 = (float(value) for value in bbox)
                    inside = x0 <= eef_x <= x1 and y0 <= eef_y <= y1
                    dx = max(x0 - eef_x, 0.0, eef_x - x1)
                    dy = max(y0 - eef_y, 0.0, eef_y - y1)
                    distance = max(dx, dy)
                    proximity_threshold = max(8.0, self.alignment_ready_px * 0.60)
                    destination_proximity = {
                        "eligible": bool(distance <= proximity_threshold),
                        "reason": (
                            "eef_projection_overlaps_destination_bbox"
                            if inside
                            else "eef_projection_near_destination_bbox"
                        )
                        if distance <= proximity_threshold
                        else "eef_projection_clear_of_destination_bbox",
                        "eef_pixel_xy": [round(eef_x, 2), round(eef_y, 2)],
                        "destination_bbox_xyxy": list(bbox),
                        "inside_destination_bbox": bool(inside),
                        "distance_px": round(distance, 2),
                        "proximity_threshold_px": round(proximity_threshold, 2),
                        "same_action": str(previous_action or "").strip().upper() or None,
                        "same_action_count": int(self._same_action_count),
                    }
                except (TypeError, ValueError):
                    pass

        held_object_alignment: dict[str, Any] = {
            "known": False,
            "reason": "held_object_not_grounded",
        }
        if (
            str(stage).upper() in {"MOVE", "PLACE", "TRANSPORT"}
            and isinstance(held_object, dict)
            and held_object.get("visible")
            and bbox is not None
        ):
            try:
                hx0, hy0, hx1, hy1 = (
                    float(value) for value in held_object["bbox_xyxy"]
                )
                dx0, dy0, dx1, dy1 = (float(value) for value in bbox)
                held_center = [(hx0 + hx1) / 2.0, (hy0 + hy1) / 2.0]
                destination_center = [(dx0 + dx1) / 2.0, (dy0 + dy1) / 2.0]
                center_error = [
                    destination_center[0] - held_center[0],
                    destination_center[1] - held_center[1],
                ]
                direction_map = geometry_evidence.get("screen_direction_to_token")
                relation = {
                    "horizontal": (
                        "right" if center_error[0] > 3
                        else "left" if center_error[0] < -3
                        else "aligned"
                    ),
                    "vertical": (
                        "down" if center_error[1] > 3
                        else "up" if center_error[1] < -3
                        else "aligned"
                    ),
                }
                correction_candidates = {}
                if isinstance(direction_map, dict):
                    correction_candidates = {
                        "horizontal": direction_map.get(relation["horizontal"]),
                        "vertical": direction_map.get(relation["vertical"]),
                    }
                alignment_error = max(abs(center_error[0]), abs(center_error[1]))
                held_bbox_complete = bool(
                    hx0 > 2.0 and hy0 > 2.0
                    and hx1 < float(geometry_evidence.get("image_size", [256, 256])[0]) - 2.0
                    and hy1 < float(geometry_evidence.get("image_size", [256, 256])[1]) - 2.0
                )
                held_object_alignment = {
                    "known": True,
                    "held_target": held_object.get("target"),
                    "held_bbox_xyxy": [int(round(v)) for v in (hx0, hy0, hx1, hy1)],
                    "held_center_xy": [round(v, 2) for v in held_center],
                    "destination_center_xy": [round(v, 2) for v in destination_center],
                    "destination_minus_held_center_px": [
                        round(v, 2) for v in center_error
                    ],
                    "screen_relation": relation,
                    "correction_candidates": correction_candidates,
                    "alignment_px": round(alignment_error, 2),
                    "alignment_threshold_px": self.alignment_ready_px,
                    "aligned": bool(
                        held_bbox_complete and alignment_error <= self.alignment_ready_px
                    ),
                    "held_bbox_complete": held_bbox_complete,
                    "object_overlaps_destination_bbox": bool(
                        hx0 < dx1 and hx1 > dx0 and hy0 < dy1 and hy1 > dy0
                    ),
                    "held_object_lowest_y": round(hy1, 2),
                    "destination_bbox_xyxy": list(bbox),
                    "source": held_object.get("source"),
                    "confidence": held_object.get("confidence"),
                    "horizontal_motion_stalled": bool(
                        held_object.get("horizontal_motion_stalled", False)
                    ),
                    "horizontal_delta_px": held_object.get("horizontal_delta_px"),
                }
                # A carried body that already overlaps the receptacle bbox while
                # its center is still offset is a generic side/rim-contact cue.
                # It is deliberately an uncertainty signal, not a claim that a
                # collision definitely occurred; the Agent must inspect the live
                # frame and decide whether to lift, correct, or continue.
                held_object_alignment["rim_contact_risk"] = bool(
                    held_object_alignment["object_overlaps_destination_bbox"]
                    and not held_object_alignment["aligned"]
                    and hy0 < dy1
                    and hy1 > dy0
                )
                if not held_object_alignment["rim_contact_risk"]:
                    self.held_rim_anchor_y = None
                elif self.held_rim_anchor_y is None:
                    # Anchor the first risky observation.  Later upward image
                    # motion is evidence that a clearance action actually changed
                    # the spatial relation; it is not a hard-coded lift count.
                    self.held_rim_anchor_y = float(held_center[1])
                clearance_progress_px = max(
                    0.0,
                    float(self.held_rim_anchor_y) - float(held_center[1]),
                ) if self.held_rim_anchor_y is not None else 0.0
                held_object_alignment["vertical_clearance_progress_px"] = round(
                    clearance_progress_px, 2
                )
                held_object_alignment["clearance_progress_ready"] = bool(
                    held_object_alignment["rim_contact_risk"]
                    and clearance_progress_px
                    >= max(3.0, self.alignment_ready_px * 0.5)
                )
            except (TypeError, ValueError, KeyError, IndexError):
                held_object_alignment = {
                    "known": False,
                    "reason": "held_object_geometry_invalid",
                }
        if isinstance(destination_proximity, dict):
            destination_proximity["held_object_alignment"] = held_object_alignment
        self.last_stage_key = stage_key
        self.last_frame_id = int(frame_id)
        self.last_action = previous_action
        self.last_evidence = {
            "frame_id": int(frame_id),
            "camera": camera,
            "stage": str(stage),
            "target": target_prompt,
            "bbox_xyxy": list(bbox) if bbox is not None else None,
            "outer_destination_bbox_xyxy": (
                list(outer_destination_bbox)
                if outer_destination_bbox is not None
                and str(stage).upper() in {"MOVE", "PLACE", "TRANSPORT"}
                else None
            ),
            "confidence": round(float(confidence), 4),
            "source": source,
            "visible": bbox is not None,
            "progress_delta_px": progress,
            "tool": tool_meta,
            "geometry": geometry_evidence,
            "secondary_view": secondary_evidence,
            "held_object": held_object,
            "held_object_alignment": held_object_alignment,
            "grasp_spatial_convergence": grasp_spatial_convergence,
            "destination_proximity": destination_proximity,
            "tool_calls": self.tool_calls,
            "tool_failures": self.tool_failures,
        }
        if source == "occlusion_memory":
            self.last_evidence["occlusion_source_frame"] = tool_meta.get("from_frame")
        elif bbox is not None:
            self.last_visible_evidence = dict(self.last_evidence)
        return dict(self.last_evidence)

    def mark_action(self, token: str) -> None:
        normalized = str(token).strip().upper()
        if normalized and normalized == str(self.last_action or "").strip().upper():
            self._same_action_count += 1
        elif normalized:
            self._same_action_count = 1
        else:
            self._same_action_count = 0
        self.last_action = normalized

    def authorize(self, token: str) -> dict[str, Any]:
        """Check whether a semantic commit has fresh host evidence.

        Shadow mode records unsupported physical beliefs without changing execution;
        active mode lets the runner replace a blocked commit with a hold action.
        This is a narrow action gate, not a trajectory planner.
        """
        normalized = str(token or "").strip().upper()
        protected = {"GRASP", "RELEASE", "DONE"}
        if not self.commit_guard_enabled or normalized not in protected:
            return {"token": normalized, "authorized": True, "mode": "off"}
        frame_id = self.last_frame_id
        evidence = dict(self.last_evidence)
        fresh = (
            frame_id is not None
            and self.last_grounding_frame is not None
            and int(frame_id) - int(self.last_grounding_frame)
            <= self.commit_guard_freshness_frames
        )
        visible = bool(evidence.get("visible", False))
        confidence = float(evidence.get("confidence", 0.0) or 0.0)
        authorized = bool(fresh and visible and confidence >= self.confidence_threshold)
        alignment = evidence.get("geometry", {}).get("target_minus_eef_px")
        alignment_known = isinstance(alignment, (list, tuple)) and len(alignment) == 2
        if normalized == "GRASP" and authorized:
            # A visible target is not enough to authorize a physical close.  If
            # calibration evidence is available, require both image axes to be
            # within a small final-alignment window.  Missing geometry remains
            # unknown and therefore fails closed in active mode.
            authorized = bool(
                alignment_known
                and max(abs(float(alignment[0])), abs(float(alignment[1])))
                <= self.commit_guard_alignment_px
            )
        decision = {
            "token": normalized,
            "authorized": authorized,
            "mode": self.commit_guard_mode,
            "frame_id": frame_id,
            "fresh": fresh,
            "visible": visible,
            "confidence": round(confidence, 4),
            "alignment_px": list(alignment) if alignment_known else None,
            "alignment_threshold_px": self.commit_guard_alignment_px,
            "reason": "fresh_aligned_evidence" if authorized else "unsupported_commit",
        }
        if not authorized:
            if self.commit_guard_mode == "active":
                self.guard_blocks += 1
                decision["blocked"] = True
            else:
                self.guard_shadow_events += 1
                decision["blocked"] = False
        return decision

    def authorize_stage_completion(self, stage: str) -> dict[str, Any]:
        """Check whether host evidence supports ending the APPROACH stage.

        This guard is deliberately narrower than action selection: it only observes a
        fresh target bbox and the robot-only camera projection already used for prompt
        context.  It cannot authorize a GRASP or infer simulator object state.  Shadow
        mode measures whether the condition would have fired; active mode lets the
        runner emit the existing DONE token for the current subgoal.
        """
        normalized_stage = str(stage or "").strip().upper()
        decision: dict[str, Any] = {
            "stage": normalized_stage,
            "eligible": False,
            "applied": False,
            "mode": self.approach_completion_guard_mode
            if self.approach_completion_guard_enabled
            else "off",
            "reason": "disabled",
        }
        if not self.approach_completion_guard_enabled:
            return decision
        if normalized_stage != "APPROACH":
            decision["reason"] = "stage_not_supported"
            return decision

        frame_id = self.last_frame_id
        evidence = dict(self.last_evidence)
        fresh = bool(
            frame_id is not None
            and self.last_grounding_frame is not None
            and int(frame_id) - int(self.last_grounding_frame)
            <= self.approach_completion_guard_freshness_frames
        )
        occlusion_recent = bool(
            evidence.get("source") == "occlusion_memory"
            and frame_id is not None
            and evidence.get("occlusion_source_frame") is not None
            and int(frame_id) - int(evidence["occlusion_source_frame"])
            <= self.occlusion_hold_frames
        )
        visible = bool(evidence.get("visible", False))
        confidence = float(evidence.get("confidence", 0.0) or 0.0)
        geometry = evidence.get("geometry")
        alignment = geometry.get("target_minus_eef_px") if isinstance(geometry, dict) else None
        alignment_known = isinstance(alignment, (list, tuple)) and len(alignment) == 2
        alignment_ready = bool(isinstance(geometry, dict) and geometry.get("alignment_ready"))
        eef_height = geometry.get("eef_height_m") if isinstance(geometry, dict) else None
        try:
            eef_height_value = float(eef_height)
        except (TypeError, ValueError):
            eef_height_value = None
        height_ready = bool(
            eef_height_value is not None
            and (
                self.approach_completion_min_height_m is None
                or eef_height_value >= self.approach_completion_min_height_m
            )
            and (
                self.approach_completion_max_height_m is None
                or eef_height_value <= self.approach_completion_max_height_m
            )
        )
        eligible = bool(
            (fresh or occlusion_recent)
            and visible
            and confidence >= self.confidence_threshold
            and alignment_known
            and alignment_ready
            and height_ready
        )
        decision.update(
            {
                "eligible": eligible,
                "frame_id": frame_id,
                "grounding_frame": self.last_grounding_frame,
                "fresh": fresh,
                "occlusion_recent": occlusion_recent,
                "visible": visible,
                "source": evidence.get("source"),
                "confidence": round(confidence, 4),
                "alignment_px": list(alignment) if alignment_known else None,
                "alignment_threshold_px": self.alignment_ready_px,
                "eef_height_m": round(eef_height_value, 4)
                if eef_height_value is not None
                else None,
                "height_min_m": self.approach_completion_min_height_m,
                "height_max_m": self.approach_completion_max_height_m,
                "reason": "fresh_visible_aligned_evidence"
                if eligible
                else "insufficient_approach_evidence",
            }
        )
        if eligible and self.approach_completion_guard_mode == "active":
            decision["applied"] = True
        return decision

    def authorize_grasp(self, stage: str) -> dict[str, Any]:
        """Authorize GRASP from AgentView when Wrist is occluded or out of view.

        The eye-in-hand camera is a supplementary near-contact sensor in LIBERO: its
        mounting and gripper occlusion can make a physically graspable object disappear.
        AgentView remains sufficient when the target is visibly aligned with the calibrated
        EEF projection and the robot-only fingertip height is in the final grasp band.
        This uses no simulator object state or success signal.
        """
        normalized_stage = str(stage or "").strip().upper()
        decision: dict[str, Any] = {
            "stage": normalized_stage,
            "eligible": False,
            "applied": False,
            "mode": self.grasp_agentview_guard_mode
            if self.grasp_agentview_guard_enabled
            else "off",
            "reason": "disabled",
        }
        if not self.grasp_agentview_guard_enabled:
            return decision
        if normalized_stage != "GRASP":
            decision["reason"] = "stage_not_supported"
            return decision

        frame_id = self.last_frame_id
        evidence = dict(self.last_evidence)
        fresh = bool(
            frame_id is not None
            and self.last_grounding_frame is not None
            and int(frame_id) - int(self.last_grounding_frame)
            <= self.grasp_agentview_guard_freshness_frames
        )
        occlusion_recent = bool(
            evidence.get("source") == "occlusion_memory"
            and frame_id is not None
            and evidence.get("occlusion_source_frame") is not None
            and int(frame_id) - int(evidence["occlusion_source_frame"])
            <= self.occlusion_hold_frames
        )
        camera = str(evidence.get("camera", ""))
        visible = bool(evidence.get("visible", False))
        confidence = float(evidence.get("confidence", 0.0) or 0.0)
        geometry = evidence.get("geometry")
        alignment = geometry.get("target_minus_eef_px") if isinstance(geometry, dict) else None
        evidence_source = evidence.get("source")
        evidence_age = 0
        # When Wrist is the primary SAM3 view, use a same-frame AgentView grounding
        # for the global grasp guard. Wrist remains in the prompt as local evidence;
        # this only prevents an occluded/offset wrist view from vetoing an otherwise
        # visually aligned global close.
        secondary = evidence.get("secondary_view")
        if (
            camera != "agentview"
            and isinstance(secondary, dict)
            and secondary.get("camera") == "agentview"
            and secondary.get("visible", False)
        ):
            camera = "agentview"
            visible = True
            confidence = float(secondary.get("confidence", 0.0) or 0.0)
            evidence_source = secondary.get("source")
            alignment = secondary.get("target_minus_eef_px")
            geometry = {
                "eef_height_m": secondary.get("eef_height_m")
            }
            try:
                evidence_age = int(secondary.get("age_frames", 0) or 0)
            except (TypeError, ValueError):
                evidence_age = 0
        alignment_known = isinstance(alignment, (list, tuple)) and len(alignment) == 2
        alignment_error = (
            max(abs(float(alignment[0])), abs(float(alignment[1])))
            if alignment_known
            else None
        )
        eef_height = geometry.get("eef_height_m") if isinstance(geometry, dict) else None
        try:
            eef_height_value = float(eef_height)
        except (TypeError, ValueError):
            eef_height_value = None
        height_ready = bool(
            eef_height_value is not None
            and (
                self.grasp_agentview_guard_min_height_m is None
                or eef_height_value >= self.grasp_agentview_guard_min_height_m
            )
            and eef_height_value <= self.grasp_agentview_guard_height_m
        )
        eligible = bool(
            (fresh or occlusion_recent)
            and evidence_age <= self.grasp_agentview_guard_freshness_frames
            and camera == "agentview"
            and visible
            and confidence >= self.confidence_threshold
            and alignment_known
            and alignment_error is not None
            and alignment_error <= self.grasp_agentview_guard_alignment_px
            and height_ready
        )
        decision.update(
            {
                "eligible": eligible,
                "frame_id": frame_id,
                "grounding_frame": self.last_grounding_frame,
                "fresh": fresh,
                "occlusion_recent": occlusion_recent,
                "camera": camera,
                "visible": visible,
                "source": evidence_source,
                "confidence": round(confidence, 4),
                "alignment_px": list(alignment) if alignment_known else None,
                "alignment_threshold_px": self.grasp_agentview_guard_alignment_px,
                "eef_height_m": round(eef_height_value, 4)
                if eef_height_value is not None
                else None,
                "height_threshold_m": self.grasp_agentview_guard_height_m,
                "height_min_m": self.grasp_agentview_guard_min_height_m,
                "wrist_required": False,
                "reason": "agentview_aligned_at_grasp_height"
                if eligible
                else "insufficient_agentview_grasp_evidence",
            }
        )
        if eligible and self.grasp_agentview_guard_mode == "active":
            decision["applied"] = True
        return decision

    def prompt_context(self) -> str:
        if not self.enabled or self.last_frame_id is None:
            return ""
        if self.memory is not None:
            context = self.memory.context(
                target=str(self.last_evidence.get("target", "")),
                frame_id=int(self.last_frame_id),
            )
        else:
            context = (
            "HARNESS VISUAL EVIDENCE: "
            + str(self.last_evidence)
            + " Treat missing or low-confidence evidence as unknown."
            )
        geometry = self.last_evidence.get("geometry")
        if geometry:
            context += (
                "\nHOST CALIBRATED GEOMETRY (robot proprioception + camera calibration, "
                "not target pose): " + str(geometry)
                + ". This is evidence; Agent still chooses the action."
            )
        stage = str(self.last_evidence.get("stage", "")).upper()
        if stage == "TRANSPORT":
            # TRANSPORT has its own short controller contract.  Do not reuse the
            # long stage-specific MOVE/PLACE prose or pre-grasp reference-height
            # fields: they previously conflicted with the receding-horizon intent.
            held = self.last_evidence.get("held_object")
            held_summary = None
            if isinstance(held, dict):
                held_summary = {
                    "target": held.get("target"),
                    "bbox_xyxy": held.get("bbox_xyxy"),
                    "confidence": held.get("confidence"),
                    "source": held.get("source"),
                }
            transport_geometry = {}
            if isinstance(geometry, dict):
                for key in (
                    "eef_position_xyz",
                    "pixel_xy",
                    "image_size",
                    "screen_direction_to_token",
                ):
                    if key in geometry:
                        transport_geometry[key] = geometry[key]
            compact = {
                "destination_opening_bbox_xyxy": self.last_evidence.get("bbox_xyxy"),
                "outer_destination_bbox_xyxy": self.last_evidence.get(
                    "outer_destination_bbox_xyxy"
                ),
                "destination_confidence": self.last_evidence.get("confidence"),
                "held_object": held_summary,
                "geometry": transport_geometry,
            }
            return (
                "TRANSPORT RAW-FRAME EVIDENCE (observations, not an action policy): "
                + str(compact)
            )
        secondary_view = self.last_evidence.get("secondary_view")
        if stage == "GRASP" and isinstance(secondary_view, dict):
            primary_relation = (geometry or {}).get("target_screen_relation")
            context += (
                "\nGRASP DUAL-VIEW EVIDENCE: AgentView is the global object/EEF relation; "
                "Wrist is the local finger-gap relation. The current primary camera is "
                f"{self.last_evidence.get('camera')!r}, with target relation "
                f"{primary_relation}; the secondary {secondary_view.get('camera')!r} view "
                f"reports {secondary_view}. Compare both images before GRASP. In particular, "
                "a Wrist target below the finger entry is a depth/approach cue, while a target "
                "centered in the finger gap at the final height supports GRASP; do not confuse "
                "an AgentView/Wrist pixel relation with a hidden object pose."
            )
        convergence = self.last_evidence.get("grasp_spatial_convergence")
        if (
            stage == "GRASP"
            and isinstance(convergence, dict)
            and convergence.get("eligible")
        ):
            context += (
                "\nGRASP SPATIAL CONVERGENCE CUE (reflection only): the global "
                "AgentView relation is already inside the calibrated alignment "
                f"window ({convergence.get('secondary_alignment_px')} px), while the "
                "local Wrist target bbox is complete and has changed only "
                f"{convergence.get('local_progress_px')} px after "
                f"{convergence.get('same_action_count')} consecutive "
                f"{convergence.get('same_action')} moves. The latest image does not "
                "support blindly repeating that direction. Inspect the two actual "
                "views: if the target body is entering the finger span and there is "
                "no visual contradiction, choose GRASP now; otherwise choose a "
                "different visually justified correction or reacquire. This cue does "
                "not assert that the object is held and does not choose the action."
            )
        if stage in {"APPROACH", "GRASP"} and isinstance(geometry, dict):
            try:
                eef_height_value = float(geometry.get("eef_height_m"))
            except (TypeError, ValueError):
                eef_height_value = None
            if (
                eef_height_value is not None
                and self.grasp_agentview_guard_min_height_m is not None
                and eef_height_value < self.grasp_agentview_guard_min_height_m
            ):
                context += (
                    "\nGRASP HEIGHT SAFETY: the EEF is already below the calibrated "
                    f"final grasp band ({self.grasp_agentview_guard_min_height_m:.3f}m). "
                    "Do not issue MV_DOWN or GRASP; issue MV_UP until the EEF returns "
                    "to the band. A target appearing below the EEF in AgentView can be "
                    "camera parallax, not evidence that another descent is needed."
                )
        if (
            self.grasp_agentview_fallback_enabled
            and str(self.last_evidence.get("stage", "")).upper() in {"GRASP", "RELEASE"}
            and self.stage_camera_override == "agentview"
            and self.last_evidence.get("camera") == "agentview"
        ):
            context += (
                "\nGRASP CAMERA FALLBACK: SAM3 found no target in Wrist, so the current "
                "bbox/alignment comes from AgentView. Use AgentView only for global "
                "X/Y correction until the target is visibly in Wrist; do not interpret "
                "robot-only geometry as a target location."
            )
            alignment = geometry.get("target_minus_eef_px") if isinstance(geometry, dict) else None
            if (
                isinstance(alignment, (list, tuple))
                and len(alignment) == 2
                and max(abs(float(alignment[0])), abs(float(alignment[1])))
                <= self.alignment_ready_px
            ):
                eef_height = geometry.get("eef_height_m") if isinstance(geometry, dict) else None
                try:
                    eef_height_value = float(eef_height)
                except (TypeError, ValueError):
                    eef_height_value = None
                height_in_band = bool(
                    eef_height_value is not None
                    and (
                        self.grasp_agentview_guard_min_height_m is None
                        or eef_height_value >= self.grasp_agentview_guard_min_height_m
                    )
                    and eef_height_value <= self.grasp_agentview_guard_height_m
                )
                if (
                    self.grasp_agentview_guard_enabled
                    and height_in_band
                    and max(abs(float(alignment[0])), abs(float(alignment[1])))
                    <= self.grasp_agentview_guard_alignment_px
                ):
                    context += (
                        " AgentView now shows the target aligned at final grasp height; "
                        "Wrist is occluded by its camera mounting and is not required. "
                        "Issue GRASP now; do not issue another MV_DOWN."
                    )
                elif eef_height_value is not None and not height_in_band:
                    context += (
                        " The EEF is outside the calibrated final grasp band; use "
                        "MV_UP if below it or MV_DOWN if above it, then reassess. "
                        "Do not treat a lower target pixel as sufficient evidence for "
                        "MV_DOWN."
                    )
                else:
                    context += (
                        " AgentView alignment is already within the horizontal/depth "
                        f"tolerance ({self.alignment_ready_px:.1f}px): do not keep issuing "
                        "MV_FWD or MV_BACK; use MV_DOWN to lower the gripper into Wrist "
                        "view, then reassess before GRASP."
                    )
        if stage == "APPROACH" and isinstance(geometry, dict):
            try:
                approach_height = float(geometry.get("eef_height_m"))
            except (TypeError, ValueError):
                approach_height = None
            minimum = self.approach_completion_min_height_m
            maximum = self.approach_completion_max_height_m
            in_final_band = bool(
                approach_height is not None
                and (minimum is None or approach_height >= minimum)
                and (maximum is None or approach_height <= maximum)
            )
            alignment = geometry.get("target_minus_eef_px")
            if (
                approach_height is not None
                and maximum is not None
                and approach_height > maximum
                and isinstance(alignment, (list, tuple))
                and len(alignment) == 2
            ):
                horizontal_error = abs(float(alignment[0]))
                vertical_error = abs(float(alignment[1]))
                if (
                    horizontal_error <= self.alignment_ready_px
                    and vertical_error > max(horizontal_error, self.alignment_ready_px)
                ):
                    context += (
                        "\nAPPROACH HIGH-POSE SPATIAL DISAMBIGUATION: the EEF is at "
                        f"{approach_height:.3f}m, above the final band ending at "
                        f"{maximum:.3f}m. The calibrated AgentView projection shows only "
                        f"{horizontal_error:.1f}px horizontal error but "
                        f"{vertical_error:.1f}px screen-down error. At this high pose, "
                        "that large screen-down offset is primarily the height/depth "
                        "projection, not evidence for repeated MV_FWD/MV_BACK. Choose "
                        "MV_DOWN to enter the final height band; reassess the remaining "
                        "depth correction only after the new frame. This is a spatial "
                        "cue, not a scripted trajectory."
                    )
            if in_final_band:
                relation = geometry.get("target_screen_relation", {})
                candidates = geometry.get("calibrated_correction_candidates", {})
                context += (
                    "\nAPPROACH HEIGHT CUE: the robot-only EEF height is already inside "
                    "the configured final pre-grasp band. A target appearing lower in "
                    "AgentView is now a depth/alignment question, not evidence to keep "
                    f"descending. Reassess the visual relation {relation}; if the target "
                    f"remains below, the calibrated depth candidate is {candidates.get('vertical')}. "
                    "Choose the action from the live image and do not oscillate MV_DOWN/MV_UP "
                    "around an already valid grasp height."
                )
        if stage in {"MOVE", "PLACE", "TRANSPORT"} and isinstance(geometry, dict):
            alignment = geometry.get("target_minus_eef_px")
            if isinstance(alignment, (list, tuple)) and len(alignment) == 2:
                aligned = bool(geometry.get("alignment_ready", False))
                relation = geometry.get("target_screen_relation", {})
                bbox = self.last_evidence.get("bbox_xyxy")
                image_size = geometry.get("image_size", [256, 256])
                clipped = False
                try:
                    clipped = (
                        float(bbox[0]) <= 2.0
                        or float(bbox[1]) <= 2.0
                        or float(bbox[2]) >= float(image_size[0]) - 2.0
                        or float(bbox[3]) >= float(image_size[1]) - 2.0
                    )
                except (TypeError, ValueError, IndexError):
                    clipped = False
                if not aligned:
                    context += (
                        "\nPLACEMENT ALIGNMENT CUE: the host projection currently measures "
                        f"the receptacle center {list(alignment)} pixels from the EEF; the "
                        f"visible relation is {relation}. This is not aligned evidence. "
                        "Inspect the held object's body and the receptacle opening in the "
                        "AgentView yourself before DONE or MV_DOWN; the Agent chooses the "
                        "correct axis/action."
                    )
                if clipped:
                    context += (
                        "\nRECEPTACLE VIEW LIMITATION: the current visual bbox touches an "
                        "image edge, so its host center is incomplete. Do not treat the "
                        "clipped bbox center as proof of a successful placement; use the "
                        "full AgentView and the visible object/opening relation to decide "
                        "whether to move, lift, or keep inspecting."
                    )
        held_alignment = self.last_evidence.get("held_object_alignment")
        if (
            stage in {"MOVE", "PLACE", "TRANSPORT"}
            and isinstance(held_alignment, dict)
            and held_alignment.get("known")
        ):
            context += (
                "\nHELD-OBJECT SPATIAL EVIDENCE: the carried object's body was grounded "
                "separately from the receptacle. Its center is "
                f"{held_alignment.get('held_center_xy')} and the destination center is "
                f"{held_alignment.get('destination_center_xy')}; destination-minus-object "
                f"error={held_alignment.get('destination_minus_held_center_px')} px, "
                f"relation={held_alignment.get('screen_relation')}, correction candidates="
                f"{held_alignment.get('correction_candidates')}, object_aligned="
                f"{held_alignment.get('aligned')}, rim_contact_risk="
                f"{held_alignment.get('rim_contact_risk')}, vertical_clearance_progress="
                f"{held_alignment.get('vertical_clearance_progress_px')} px, clearance_ready="
                f"{held_alignment.get('clearance_progress_ready')}. Use the carried object's body and the "
                "opening—not the EEF projection alone—to judge MOVE completion. The "
                "harness does not choose the direction."
            )
            if held_alignment.get("rim_contact_risk"):
                context += (
                    " The carried body bbox currently overlaps the destination bbox while "
                    "the body center is still offset. Treat this as possible side/rim "
                    "contact, not as permission to keep pushing laterally: inspect the "
                    "lowest point and rim in the fresh AgentView. If contact or clearance "
                    "uncertainty is visible, choose MV_UP first, then reassess the new "
                    "spatial relation. If the evidence says clearance is ready, reassess "
                    "whether a horizontal move away from the contact is now justified; "
                    "this cue does not prescribe a recovery direction or release."
                )
        destination_proximity = self.last_evidence.get("destination_proximity")
        if (
            stage == "MOVE"
            and isinstance(destination_proximity, dict)
            and destination_proximity.get("eligible")
        ):
            location = (
                "inside"
                if destination_proximity.get("inside_destination_bbox")
                else "near"
            )
            context += (
                "\nMOVE DESTINATION-RIM PROXIMITY EVIDENCE: the calibrated EEF "
                f"projection is {location} the visible destination bbox "
                f"(distance={destination_proximity.get('distance_px')} px). This "
                "does not mean the held object is centered or safe: inspect the live "
                "AgentView for the object's lowest point, the opening, and the near "
                "rim. If the object is touching/overlapping the rim or clearance is "
                "uncertain, choose MV_UP and inspect a fresh frame before more "
                "horizontal motion; otherwise continue only if the image clearly shows "
                "a collision-free route. This is a safety cue, not a scripted recovery "
                "action, and RELEASE remains forbidden in MOVE."
            )
            if destination_proximity.get("inside_destination_bbox"):
                context += (
                    "\nMOVE DESTINATION-OVERLAP REVIEW: the calibrated EEF projection is "
                    "inside the visible destination bbox. This is only an EEF hazard cue, "
                    "not proof that the carried object is aligned. If the held-object "
                    "evidence still shows a clear same-direction offset, continue the "
                    "visually justified correction toward the opening; if the object body "
                    "is already safely over the opening, choose DONE to let PLACE take "
                    "over. Do not repeat lateral motion after the object itself is aligned, "
                    "and do not release in MOVE."
                )
        return context

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "sam3_enabled": self.sam3_enabled,
            "tracker_enabled": self.tracker_enabled,
            "memory_enabled": self.memory_enabled,
            "geometry_enabled": self.geometry_enabled,
            "alignment_ready_px": self.alignment_ready_px,
            "target_reference_height_m": self.target_reference_height_m,
            "occlusion_hold_frames": self.occlusion_hold_frames,
            "grasp_agentview_fallback_enabled": self.grasp_agentview_fallback_enabled,
            "approach_completion_guard": {
                "enabled": self.approach_completion_guard_enabled,
                "mode": self.approach_completion_guard_mode,
                "freshness_frames": self.approach_completion_guard_freshness_frames,
            },
            "grasp_agentview_guard": {
                "enabled": self.grasp_agentview_guard_enabled,
                "mode": self.grasp_agentview_guard_mode,
                "freshness_frames": self.grasp_agentview_guard_freshness_frames,
                "alignment_px": self.grasp_agentview_guard_alignment_px,
                "height_m": self.grasp_agentview_guard_height_m,
                "min_height_m": self.grasp_agentview_guard_min_height_m,
            },
            "tool_calls": self.tool_calls,
            "tool_failures": self.tool_failures,
            "commit_guard": {
                "enabled": self.commit_guard_enabled,
                "mode": self.commit_guard_mode,
                "alignment_px": self.commit_guard_alignment_px,
                "shadow_events": self.guard_shadow_events,
                "blocks": self.guard_blocks,
            },
            "last_evidence": self.last_evidence,
            "last_visible_evidence": self.last_visible_evidence,
            "memory": self.memory.snapshot() if self.memory is not None else None,
        }

    def close(self) -> None:
        if self.sam3 is not None:
            self.sam3.close()

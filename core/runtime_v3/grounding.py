"""Small semantic candidate-pool contract for binding EntitySpec to RGB."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


MAX_SEMANTIC_REGIONS = 4
DEFAULT_GROUNDING_POOL_K = 4
DEFAULT_DUPLICATE_MASK_IOU = 0.90
MIN_NORMALIZED_MASK_AREA = 0.00001
MAX_NORMALIZED_MASK_AREA = 0.95


@dataclass(frozen=True)
class GroundingCandidate:
    """One deployable visual mask proposal; it contains no simulator identity."""

    candidate_id: str
    mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    proposal_score: float | None
    source: str
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        candidate_id = str(self.candidate_id)
        if not candidate_id.startswith("C") or not candidate_id[1:].isdigit():
            raise ValueError("candidate_id must use the visible C<number> form")
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2 or not bool(mask.any()):
            raise ValueError("candidate mask must be a nonempty two-dimensional raster")
        if len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain four half-open pixel bounds")
        x0, y0, x1, y1 = (int(value) for value in self.bbox_xyxy)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("candidate bbox must have positive width and height")
        score = self.proposal_score
        if score is not None and not math.isfinite(float(score)):
            raise ValueError("proposal_score must be finite when supplied")
        sources = tuple(dict.fromkeys(str(item) for item in (self.sources or (self.source,))))
        if any(not item for item in sources):
            raise ValueError("candidate sources must be non-empty")
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "mask", np.ascontiguousarray(mask))
        object.__setattr__(self, "bbox_xyxy", (x0, y0, x1, y1))
        object.__setattr__(self, "proposal_score", None if score is None else float(score))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "sources", sources)


@dataclass(frozen=True)
class EntityGroundingEvidence:
    """Result of semantic candidate choice before Runtime temporal verification."""

    entity_key: str
    semantic_query: str
    candidate_id: str | None
    decision: str
    valid: bool
    source: str
    invalid_reason: str | None = None
    candidate_count: int = 0
    proposal_region_count: int = 0
    agent_calls: int = 0

    def __post_init__(self) -> None:
        if not str(self.entity_key).strip() or not str(self.semantic_query).strip():
            raise ValueError("grounding evidence requires an entity key and semantic query")
        if self.decision not in {"SELECT", "NO_MATCH", "INVALID"}:
            raise ValueError("unsupported semantic grounding decision")
        if self.valid and (self.decision != "SELECT" or self.candidate_id is None):
            raise ValueError("valid evidence must select an existing candidate")
        if self.decision != "SELECT" and self.candidate_id is not None:
            raise ValueError("non-selection evidence cannot contain a candidate ID")
        if self.candidate_count < 0 or self.proposal_region_count < 0 or self.agent_calls < 0:
            raise ValueError("grounding counts must be nonnegative")


class CandidatePoolBuilder:
    """Apply only generic mask-quality, area, overlap, and bounded-K rules."""

    def __init__(
        self,
        *,
        max_candidates: int = DEFAULT_GROUNDING_POOL_K,
        duplicate_mask_iou: float = DEFAULT_DUPLICATE_MASK_IOU,
        min_normalized_area: float = MIN_NORMALIZED_MASK_AREA,
        max_normalized_area: float = MAX_NORMALIZED_MASK_AREA,
    ) -> None:
        if int(max_candidates) < 1:
            raise ValueError("max_candidates must be positive")
        if int(max_candidates) > DEFAULT_GROUNDING_POOL_K:
            raise ValueError(f"max_candidates cannot exceed the frozen K={DEFAULT_GROUNDING_POOL_K}")
        if not 0 <= float(duplicate_mask_iou) <= 1:
            raise ValueError("duplicate_mask_iou must lie in [0, 1]")
        if not 0 <= float(min_normalized_area) < float(max_normalized_area) <= 1:
            raise ValueError("normalized mask area limits are invalid")
        self.max_candidates = int(max_candidates)
        self.duplicate_mask_iou = float(duplicate_mask_iou)
        self.min_normalized_area = float(min_normalized_area)
        self.max_normalized_area = float(max_normalized_area)
        self.last_counts: dict[str, int] = {}

    def build(
        self, candidates: Sequence[GroundingCandidate], *, image_shape: Sequence[int]
    ) -> tuple[GroundingCandidate, ...]:
        if len(image_shape) != 2:
            raise ValueError("image_shape must be (height, width)")
        height, width = int(image_shape[0]), int(image_shape[1])
        if height <= 0 or width <= 0:
            raise ValueError("image dimensions must be positive")
        total_pixels = height * width
        filtered: list[GroundingCandidate] = []
        for item in candidates:
            if not isinstance(item, GroundingCandidate) or item.mask.shape != (height, width):
                continue
            area_fraction = float(item.mask.sum()) / total_pixels
            if not self.min_normalized_area <= area_fraction <= self.max_normalized_area:
                continue
            x0, y0, x1, y1 = item.bbox_xyxy
            if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                continue
            filtered.append(item)

        ranked = sorted(
            filtered,
            key=lambda item: (
                item.proposal_score is None,
                -(item.proposal_score or 0.0),
                -int(item.mask.sum()),
                item.source,
            ),
        )
        deduplicated: list[GroundingCandidate] = []
        for candidate in ranked:
            duplicate_index = next(
                (index for index, kept in enumerate(deduplicated)
                 if _mask_iou(candidate.mask, kept.mask) >= self.duplicate_mask_iou),
                None,
            )
            if duplicate_index is None:
                deduplicated.append(candidate)
                continue
            kept = deduplicated[duplicate_index]
            sources = tuple(dict.fromkeys((*kept.sources, *candidate.sources)))
            deduplicated[duplicate_index] = replace(
                kept,
                source="both" if len(sources) > 1 else sources[0],
                sources=sources,
                proposal_score=max(
                    (score for score in (kept.proposal_score, candidate.proposal_score)
                     if score is not None),
                    default=None,
                ),
            )

        selected = deduplicated[:self.max_candidates]
        final = tuple(
            replace(candidate, candidate_id=f"C{index}")
            for index, candidate in enumerate(selected)
        )
        self.last_counts = {
            "raw_proposals": len(candidates),
            "filtered_proposals": len(filtered),
            "deduplicated_proposals": len(deduplicated),
            "final_candidates": len(final),
            "candidate_pool_k": self.max_candidates,
        }
        return final


class QwenSemanticRegionProposer:
    """One strict RGB-only query for coarse semantic regions to refine with SAM."""

    PROMPT_VERSION = "m3.8-zero-shot-region-v1"

    def __init__(self, client: Any, *, max_tokens: int = 256) -> None:
        self.client = client
        self.max_tokens = int(max_tokens)
        self.last_raw_text = ""
        self.last_error: str | None = None

    def propose(
        self, image: np.ndarray, *, task_instruction: str, semantic_phrase: str
    ) -> tuple[tuple[float, float, float, float], ...]:
        rgb = _validate_rgb(image)
        prompt = (
            "Identify plausible visible image region proposals for the named task entity. Use only the supplied "
            "RGB image. The target may be a small packaged, boxed, canned, bottled, or other ordinary object; "
            "do not require its printed label to be readable. If one visible object is a likely match, propose it. "
            "If several visible objects are plausible or you are uncertain, include up to four plausible regions "
            "so SAM and a separate semantic selector can resolve them. Use decision=NO_MATCH with an empty list "
            "only when no plausible visible region exists. Return strict JSON with decision=PROPOSE and the "
            "regions. Each bbox is [left,top,right,bottom] normalized independently to the full image width and "
            "height on a 0..1000 scale. Include visible object pixels with modest context; do not crop away or "
            "describe the scene. These boxes are visual prompts for segmentation only. Do not return robot, "
            "motion, action, distance, scale, or world-coordinate fields.\n"
            f"Task instruction: {str(task_instruction).strip()}\n"
            f"EntitySpec semantic phrase: {str(semantic_phrase).strip()}"
        )
        schema = {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["PROPOSE", "NO_MATCH"]},
                "regions": {
                    "type": "array", "maxItems": MAX_SEMANTIC_REGIONS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "bbox_norm_1000": {
                                "type": "array", "minItems": 4, "maxItems": 4,
                                "items": {"type": "number", "minimum": 0, "maximum": 1000},
                            }
                        },
                        "required": ["bbox_norm_1000"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["decision", "regions"],
            "additionalProperties": False,
        }
        self.last_raw_text, self.last_error = "", None
        try:
            response = self.client.complete_json(
                prompt, agentview_image=rgb, wrist_image=None, schema=schema,
                max_tokens=self.max_tokens, temperature=0.0,
                chat_template_kwargs={"enable_thinking": False, "thinking": False},
            )
            self.last_raw_text = str(getattr(response, "raw_text", ""))
            parsed = json.loads(self.last_raw_text)
            if not isinstance(parsed, Mapping) or set(parsed) != {"decision", "regions"}:
                raise ValueError("region proposal schema mismatch")
            decision, regions = parsed["decision"], parsed["regions"]
            if not isinstance(regions, list) or len(regions) > MAX_SEMANTIC_REGIONS:
                raise ValueError("region proposal count is outside the bounded schema")
            if decision == "NO_MATCH":
                if regions:
                    raise ValueError("NO_MATCH must contain an empty region list")
                return ()
            if decision != "PROPOSE" or not regions:
                raise ValueError("region proposal decision is invalid or empty")
            boxes = []
            for region in regions:
                if not isinstance(region, Mapping) or set(region) != {"bbox_norm_1000"}:
                    raise ValueError("region item schema mismatch")
                raw = region["bbox_norm_1000"]
                if (not isinstance(raw, list) or len(raw) != 4
                        or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(float(v)) or not 0 <= float(v) <= 1000
                               for v in raw)):
                    raise ValueError("normalized region box is invalid")
                x0, y0, x1, y1 = (float(v) / 1000.0 for v in raw)
                if x1 <= x0 or y1 <= y0:
                    raise ValueError("normalized region box has no area")
                boxes.append((x0, y0, x1, y1))
            return tuple(boxes)
        except Exception as exc:  # fail closed; semantic proposals are never retried
            self.last_error = f"{type(exc).__name__}: {exc}"
            return ()


class QwenSemanticCandidateSelector:
    """Select only an existing C-number ID or NO_MATCH from full-context RGB."""

    PROMPT_VERSION = "m3.8-zero-shot-detail-card-v2"

    def __init__(self, client: Any, *, max_tokens: int = 64) -> None:
        self.client = client
        self.max_tokens = int(max_tokens)
        self.last_raw_text = ""
        self.last_record: dict[str, Any] = {}

    def select(
        self,
        image: np.ndarray,
        contact_sheet: np.ndarray,
        candidates: Sequence[GroundingCandidate],
        *,
        task_instruction: str,
        semantic_phrase: str,
    ) -> tuple[str | None, str]:
        _validate_rgb(image)
        _validate_rgb(contact_sheet)
        ids = [candidate.candidate_id for candidate in candidates]
        if not ids:
            return None, "NO_MATCH"
        if len(ids) > DEFAULT_GROUNDING_POOL_K or len(set(ids)) != len(ids):
            return None, "INVALID"
        prompt = (
            "Compare the full canonical RGB image with the candidate cards. Each card's left panel shows the "
            "full scene with one numbered mask; its right panel magnifies that same candidate. Select only a "
            "candidate whose visible appearance supports the named task entity. Read legible product labels; "
            "when text is too small, compare visible shape, color, and packaging cues across the scene. A generic "
            "package shape alone is not enough. Candidate IDs are arbitrary: do not prefer an earlier ID or a "
            "proposal order. Choose NO_MATCH if no clear semantic match is shown or similar candidates cannot "
            "be distinguished. Return exactly one JSON "
            "object containing only decision and candidate_id. decision is SELECT or NO_MATCH; candidate_id "
            "must be one supplied C-number ID for SELECT and null for NO_MATCH. Do not return coordinates, "
            "boxes, direction, scale, pose, distance, robot action, or extra fields.\n"
            f"Task instruction: {str(task_instruction).strip()}\n"
            f"EntitySpec semantic phrase: {str(semantic_phrase).strip()}\n"
            f"Available candidate IDs: {json.dumps(ids)}"
        )
        schema = {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["SELECT", "NO_MATCH"]},
                "candidate_id": {"type": ["string", "null"],
                                  "enum": ids + [None]},
            },
            "required": ["decision", "candidate_id"],
            "additionalProperties": False,
        }
        self.last_raw_text = ""
        try:
            response = self.client.complete_json(
                prompt, agentview_image=image, wrist_image=contact_sheet, schema=schema,
                max_tokens=self.max_tokens, temperature=0.0,
                chat_template_kwargs={"enable_thinking": False, "thinking": False},
            )
            self.last_raw_text = str(getattr(response, "raw_text", ""))
            parsed = json.loads(self.last_raw_text)
            if (not isinstance(parsed, Mapping)
                    or set(parsed) != {"decision", "candidate_id"}):
                raise ValueError("semantic selection schema mismatch")
            decision, candidate_id = parsed["decision"], parsed["candidate_id"]
            if decision == "NO_MATCH" and candidate_id is None:
                return None, "NO_MATCH"
            if decision == "SELECT" and candidate_id in ids:
                return str(candidate_id), "SELECT"
            raise ValueError("semantic selector returned an unknown or inconsistent candidate")
        except Exception as exc:  # one call; malformed output is not repaired
            self.last_record = {"error": f"{type(exc).__name__}: {exc}"}
            return None, "INVALID"


def make_candidate_contact_sheet(
    image: np.ndarray,
    candidates: Sequence[GroundingCandidate],
    *,
    columns: int = 2,
    selected_candidate_id: str | None = None,
    selection_decision: str | None = None,
) -> np.ndarray:
    """Show full-scene localization and a magnified view for each numbered mask."""
    rgb = _validate_rgb(image)
    source = Image.fromarray(rgb, mode="RGB")
    header_h = 32 if selection_decision is not None else 0
    if not candidates:
        if not header_h:
            return rgb.copy()
        sheet = Image.new("RGB", (source.width, source.height + header_h), "white")
        sheet.paste(source, (0, header_h))
        ImageDraw.Draw(sheet).text(
            (8, 9), f"Qwen decision: {selection_decision} | candidate pool empty",
            fill="black",
        )
        return np.asarray(sheet, dtype=np.uint8)
    thumb_w = min(256, max(192, source.width))
    thumb_h = max(1, round(source.height * thumb_w / source.width))
    detail_size = 192
    label_h = 28
    card_gap = 8
    card_w = thumb_w + detail_size + card_gap
    card_h = max(thumb_h, detail_size) + label_h
    columns = max(1, int(columns))
    rows = (len(candidates) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * card_w, header_h + rows * card_h), "white")
    if header_h:
        ImageDraw.Draw(sheet).text(
            (8, 9),
            f"Qwen decision: {selection_decision}"
            + (f" {selected_candidate_id}" if selected_candidate_id is not None else ""),
            fill="black",
        )
    for index, candidate in enumerate(candidates):
        tint = (255, 45 + (index * 29) % 180, 45, 48)
        full_overlay = source.convert("RGBA")
        full_mask = np.zeros((*candidate.mask.shape, 4), dtype=np.uint8)
        full_mask[candidate.mask] = tint
        full_overlay = Image.alpha_composite(full_overlay, Image.fromarray(full_mask, mode="RGBA"))
        full_draw = ImageDraw.Draw(full_overlay)
        full_draw.rectangle(candidate.bbox_xyxy, outline=(0, 190, 220, 255), width=3)
        full_overlay.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)

        image_h, image_w = candidate.mask.shape
        x0, y0, x1, y1 = candidate.bbox_xyxy
        box_w, box_h = max(1, x1 - x0), max(1, y1 - y0)
        side = min(min(image_w, image_h), max(24, box_w, box_h) +
                   2 * max(8, int(max(box_w, box_h) * 0.65)))
        center_x, center_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        crop_x0 = max(0, min(image_w - side, int(round(center_x - side / 2.0))))
        crop_y0 = max(0, min(image_h - side, int(round(center_y - side / 2.0))))
        crop_box = (crop_x0, crop_y0, crop_x0 + side, crop_y0 + side)
        crop = source.crop(crop_box).convert("RGBA")
        local_mask = candidate.mask[crop_y0:crop_y0 + side, crop_x0:crop_x0 + side]
        local_overlay = np.zeros((side, side, 4), dtype=np.uint8)
        local_overlay[local_mask] = tint
        crop = Image.alpha_composite(crop, Image.fromarray(local_overlay, mode="RGBA"))
        ImageDraw.Draw(crop).rectangle(
            (x0 - crop_x0, y0 - crop_y0, x1 - crop_x0, y1 - crop_y0),
            outline=(0, 190, 220, 255), width=max(2, side // 96),
        )
        crop = crop.resize((detail_size, detail_size), Image.Resampling.LANCZOS)

        card_x = (index % columns) * card_w
        card_y = header_h + (index // columns) * card_h
        sheet.paste(full_overlay.convert("RGB"), (card_x, card_y))
        sheet.paste(crop.convert("RGB"), (card_x + thumb_w + card_gap, card_y))
        draw = ImageDraw.Draw(sheet)
        is_selected = candidate.candidate_id == selected_candidate_id
        if is_selected:
            draw.rectangle(
                (card_x, card_y, card_x + card_w - 1,
                 card_y + max(thumb_h, detail_size) + label_h - 1),
                outline=(30, 170, 70), width=4,
            )
        draw.text(
            (card_x + 6, card_y + max(thumb_h, detail_size) + 7),
            candidate.candidate_id + (" | SELECTED" if is_selected else ""),
            fill="black",
        )
    return np.asarray(sheet, dtype=np.uint8)


def candidate_from_detection(
    detection: Mapping[str, Any], *, candidate_id: str, image_shape: Sequence[int], source: str
) -> GroundingCandidate | None:
    encoded = detection.get("mask")
    if not isinstance(encoded, Mapping) or not isinstance(encoded.get("base64"), str):
        return None
    import base64
    import io
    try:
        with Image.open(io.BytesIO(base64.b64decode(encoded["base64"], validate=True))) as mask_image:
            mask = np.asarray(mask_image.convert("L"), dtype=np.uint8) > 0
    except (ValueError, OSError, TypeError):
        return None
    if mask.shape != tuple(int(value) for value in image_shape) or not bool(mask.any()):
        return None
    ys, xs = np.nonzero(mask)
    score_raw = detection.get("score")
    try:
        score = float(score_raw) if score_raw is not None else None
    except (TypeError, ValueError):
        score = None
    try:
        return GroundingCandidate(
            candidate_id, mask,
            (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)),
            score, source,
        )
    except ValueError:
        return None


@dataclass(frozen=True)
class SemanticGroundingResult:
    candidates: tuple[GroundingCandidate, ...]
    selected: GroundingCandidate | None
    evidence: EntityGroundingEvidence
    baseline_response: Mapping[str, Any]
    candidate_sheet: np.ndarray | None
    proposal_response_text: str
    selector_response_text: str
    pool_counts: Mapping[str, int]


class SemanticGroundingBinder:
    """Proposal -> SAM refinement -> semantic choice, with no control authority."""

    def __init__(
        self,
        *,
        qwen_client: Any,
        sam_client: Any,
        task_instruction: str,
        confidence_threshold: float = 0.05,
        max_candidates: int = DEFAULT_GROUNDING_POOL_K,
    ) -> None:
        self.qwen_client = qwen_client
        self.sam_client = sam_client
        self.task_instruction = str(task_instruction)
        self.confidence_threshold = float(confidence_threshold)
        self.proposer = QwenSemanticRegionProposer(qwen_client)
        self.selector = QwenSemanticCandidateSelector(qwen_client)
        self.pool_builder = CandidatePoolBuilder(max_candidates=max_candidates)
        self.agent_calls = 0

    def ground(
        self,
        image: np.ndarray,
        *,
        entity_key: str,
        semantic_phrase: str,
        semantic_query: str,
    ) -> SemanticGroundingResult:
        calls_before = self.agent_calls
        rgb = _validate_rgb(image)
        height, width = rgb.shape[:2]
        baseline = self.sam_client.segment(
            rgb, semantic_query, confidence_threshold=self.confidence_threshold
        )
        raw_candidates: list[GroundingCandidate] = []
        baseline_details = baseline.get("details") if isinstance(baseline, Mapping) else None
        detections = baseline_details.get("detections") if isinstance(baseline_details, Mapping) else None
        if bool(baseline.get("success")) and isinstance(detections, list):
            for index, detection in enumerate(detections):
                if isinstance(detection, Mapping):
                    candidate = candidate_from_detection(
                        detection, candidate_id=f"C{index}", image_shape=(height, width),
                        source="text_sam",
                    )
                    if candidate is not None:
                        raw_candidates.append(candidate)

        regions = self.proposer.propose(
            rgb, task_instruction=self.task_instruction, semantic_phrase=semantic_phrase,
        )
        self.agent_calls += 1
        for region_index, (x0, y0, x1, y1) in enumerate(regions):
            center_x = (x0 + x1) * 0.5 * width
            center_y = (y0 + y1) * 0.5 * height
            response = self.sam_client.segment_points(
                rgb, [{"x": center_x, "y": center_y, "label": 1}],
            )
            details = response.get("details") if isinstance(response, Mapping) else None
            point_detections = details.get("detections") if isinstance(details, Mapping) else None
            if bool(response.get("success")) and isinstance(point_detections, list):
                for candidate_index, detection in enumerate(point_detections):
                    if isinstance(detection, Mapping):
                        candidate = candidate_from_detection(
                            detection, candidate_id=f"C{candidate_index}",
                            image_shape=(height, width), source="qwen_region_sam_point",
                        )
                        if candidate is not None:
                            raw_candidates.append(candidate)

        candidates = self.pool_builder.build(raw_candidates, image_shape=(height, width))
        if not candidates:
            evidence = EntityGroundingEvidence(
                entity_key, semantic_query, None, "NO_MATCH", False, "proposal_pool",
                invalid_reason="GROUNDING_PROPOSAL_MISS", candidate_count=0,
                proposal_region_count=len(regions), agent_calls=self.agent_calls - calls_before,
            )
            return SemanticGroundingResult(
                candidates, None, evidence, baseline,
                make_candidate_contact_sheet(rgb, (), selection_decision="NO_MATCH"),
                self.proposer.last_raw_text,
                "", dict(self.pool_builder.last_counts),
            )

        sheet = make_candidate_contact_sheet(rgb, candidates)
        self.agent_calls += 1
        candidate_id, decision = self.selector.select(
            rgb, sheet, candidates,
            task_instruction=self.task_instruction,
            semantic_phrase=semantic_phrase,
        )
        selected = next((item for item in candidates if item.candidate_id == candidate_id), None)
        valid = decision == "SELECT" and selected is not None
        audit_sheet = make_candidate_contact_sheet(
            rgb, candidates, selected_candidate_id=(selected.candidate_id if valid and selected else None),
            selection_decision=decision,
        )
        source = selected.source if selected is not None else "semantic_selector"
        reason = None if valid else (
            "SEMANTIC_SELECTION_NO_MATCH" if decision == "NO_MATCH"
            else "semantic_selection_schema_invalid"
        )
        evidence = EntityGroundingEvidence(
            entity_key, semantic_query, selected.candidate_id if valid and selected else None,
            decision if decision in {"SELECT", "NO_MATCH"} else "INVALID", bool(valid), source,
            invalid_reason=reason, candidate_count=len(candidates),
            proposal_region_count=len(regions), agent_calls=self.agent_calls - calls_before,
        )
        return SemanticGroundingResult(
            candidates, selected if valid else None, evidence, baseline, audit_sheet,
            self.proposer.last_raw_text, self.selector.last_raw_text,
            dict(self.pool_builder.last_counts),
        )


def _validate_rgb(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3 or not value.size:
        raise ValueError("image must be a nonempty HxWx3 RGB array")
    if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
        raise ValueError("RGB image must contain finite numeric values")
    return np.ascontiguousarray(np.clip(value, 0, 255).astype(np.uint8))


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return 0.0
    union = int(np.logical_or(left, right).sum())
    return float(np.logical_and(left, right).sum() / union) if union else 0.0

#!/usr/bin/env python3
"""M3.7 HOLD-only readiness forensics and offline SAM phrase replay.

The capture phase has no target phrase and no SAM call. It records canonical
agentview RGB, RGB-only frame-difference metrics, and simulator object pose in
separate diagnostic artifacts. The replay phase runs SAM against those exact
saved frames and never advances the simulator.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.capabilities.sam3_client import Sam3Client
from core.config import load_yaml
from core.runtime_v3.adapters.libero_env import LiberoEnvironmentAdapter
from core.runtime_v3.adapters.libero_observation import LiberoObservationAdapter
from core.runtime_v3.canonical_image import CanonicalImageAdapter
from core.runtime_v3.object_relative import (
    associate_target_candidate,
    make_target_identity_anchor,
    segmentation_from_response,
)
from core.runtime_v3.scene_settling import (
    SCENE_READY_MAX_BBOX_EDGE_SHIFT_PX,
    SCENE_READY_MAX_CENTROID_SHIFT_PX,
    SCENE_READY_MAX_MASK_AREA_CHANGE_PX,
    SCENE_READY_WINDOW_OBSERVATIONS,
    SceneReadyEvidence,
)
from core.runtime_v3.temporal_calibration import run_v3_tick
from interpreters.libero_atomic_controller import LiberoAtomicController


SUITE = "LIBERO_OBJECT"
TASKS = {
    0: {"phrase": "alphabet soup", "qwen_phrase": "the alphabet soup"},
    2: {"phrase": "salad dressing"},
    6: {"phrase": "butter"},
    7: {"phrase": "milk"},
}
ROBOT_READY_HOLD_TICKS = 4
DIAGNOSTIC_TAIL_HOLD_TICKS = 80
CONTROL_TICK_M = 0.005


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable({name: getattr(value, name) for name in value.__dataclass_fields__})
    return repr(value)


def _write_json(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(record), stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _new_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = base / f"run_{stamp}_{uuid.uuid4().hex[:8]}"
    target.mkdir(parents=False, exist_ok=False)
    return target


def _configure_local_proxy_bypass(url: str) -> None:
    if urlparse(url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        return
    entries = [part.strip() for key in ("NO_PROXY", "no_proxy")
               for part in os.environ.get(key, "").split(",") if part.strip()]
    entries.extend(("127.0.0.1", "localhost", "::1"))
    value = ",".join(dict.fromkeys(entries))
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def _body_name_from_bddl(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(r"\(:obj_of_interest\s+([^\s()]+)", text, flags=re.IGNORECASE)
    if match is None:
        raise RuntimeError(f"BDDL has no :obj_of_interest entry: {path}")
    return f"{match.group(1)}_main"


def _body_id(environment: LiberoEnvironmentAdapter, body_name: str) -> int:
    model = environment.env.sim.model
    if callable(getattr(model, "body_name2id", None)):
        return int(model.body_name2id(body_name))
    return int(model.body(body_name).id)


def _pose(environment: LiberoEnvironmentAdapter, body_id: int) -> list[float]:
    xyz = np.asarray(environment.env.sim.data.xpos[body_id], dtype=float).reshape(3)
    if not np.all(np.isfinite(xyz)):
        raise ValueError("diagnostic target position is not finite")
    return xyz.tolist()


def _rgb_diff(previous: np.ndarray | None, current: np.ndarray) -> dict[str, float] | None:
    if previous is None:
        return None
    delta = np.abs(current.astype(np.int16) - previous.astype(np.int16)).astype(np.float32) / 255.0
    pixel = delta.max(axis=2)
    return {
        "mean_abs_rgb_delta_0_1": float(delta.mean()),
        "median_max_channel_delta_0_1": float(np.median(pixel)),
        "p95_max_channel_delta_0_1": float(np.percentile(pixel, 95)),
        "p99_max_channel_delta_0_1": float(np.percentile(pixel, 99)),
        "fraction_pixels_changed_gt_2_255": float(np.mean(pixel > (2.0 / 255.0))),
    }


def _hold(
    environment: LiberoEnvironmentAdapter,
    observer: LiberoObservationAdapter,
    controller: LiberoAtomicController,
    *,
    task_id: int,
    workspace: tuple[float, float],
) -> None:
    before = int(environment.step_count)
    result = run_v3_tick(
        environment,
        observer,
        controller,
        task_id=f"{SUITE}:{task_id}",
        token=None,
        direction_unit=None,
        commanded_step_m=CONTROL_TICK_M,
        reset=False,
        workspace_z_bounds_m=workspace,
    )
    if (int(result.get("actions", 0)) != 1
            or result.get("approved_action") != "CALIBRATION_HOLD"
            or not result.get("backend_execution")
            or int(environment.step_count) != before + 1):
        raise RuntimeError(f"diagnostic expected exactly one Executor HOLD: {result}")


def _capture_task(
    *, task_id: int, output: Path, config: Mapping[str, Any],
    resolution: int, tail_ticks: int,
) -> dict[str, Any]:
    task_dir = output / f"task_{task_id}"
    rgb_dir = task_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=False)
    workspace_cfg = config.get("workspace_z_bounds_m", {})
    workspace = (float(workspace_cfg.get("min", 0.02)),
                 float(workspace_cfg.get("max", 0.60)))
    environment = LiberoEnvironmentAdapter.create(
        suite_name=SUITE, task_id=task_id, init_state_index=0, seed=0,
        camera_height=resolution, camera_width=resolution, horizon=256,
    )
    observer = LiberoObservationAdapter(
        max_eef_z_m=workspace[1], min_eef_z_m=workspace[0],
        safe_lift_step_m=CONTROL_TICK_M,
    )
    controller = LiberoAtomicController(
        move_vectors=config["move_vectors"], step_m=CONTROL_TICK_M,
        sim_steps_per_decision=1,
        position_scale_m=float(config.get("position_scale_m", 0.05)),
    )
    try:
        environment.reset()
        body_name = _body_name_from_bddl(environment.handle.bddl_file)
        body_id = _body_id(environment, body_name)
        robot_trace: list[dict[str, Any]] = []
        initial_obs = observer.observe(environment)
        del initial_obs
        robot_trace.append({
            "environment_step": int(environment.step_count),
            "target_world_position_m": _pose(environment, body_id),
            "eef_position_xyz_m": list(observer.last_raw.eef_position_xyz),
        })
        for _ in range(ROBOT_READY_HOLD_TICKS):
            _hold(environment, observer, controller, task_id=task_id, workspace=workspace)
            observer.observe(environment)
            robot_trace.append({
                "environment_step": int(environment.step_count),
                "target_world_position_m": _pose(environment, body_id),
                "eef_position_xyz_m": list(observer.last_raw.eef_position_xyz),
            })

        canonical = CanonicalImageAdapter()
        rows: list[dict[str, Any]] = []
        previous_rgb: np.ndarray | None = None
        for tail_tick in range(tail_ticks + 1):
            if tail_tick:
                _hold(environment, observer, controller, task_id=task_id, workspace=workspace)
                observer.observe(environment)
            assert observer.last_raw is not None
            image = canonical.transform_image(observer.last_raw.agentview_rgb)
            pose = _pose(environment, body_id)
            image_path = rgb_dir / f"tick_{tail_tick:03d}.png"
            Image.fromarray(image, mode="RGB").save(image_path, optimize=True)
            delta = _rgb_diff(previous_rgb, image)
            previous_rgb = image
            rows.append({
                "tail_tick": tail_tick,
                "environment_step": int(environment.step_count),
                "simulation_time_s": float(environment.env.sim.data.time),
                "canonical_rgb_path": str(image_path),
                "rgb_motion": delta,
                "target_world_position_m": pose,
                "eef_position_xyz_m": list(observer.last_raw.eef_position_xyz),
                "diagnostic_only_oracle": True,
            })
        target = np.asarray([row["target_world_position_m"] for row in rows], dtype=float)
        delta = np.diff(target, axis=0)
        norms = np.linalg.norm(delta, axis=1)
        robot_target = np.asarray([row["target_world_position_m"] for row in robot_trace], dtype=float)
        result = {
            "suite": SUITE,
            "task_id": task_id,
            "task_instruction": environment.task_description,
            "target_body_name_diagnostic_only": body_name,
            "init_state_index": 0,
            "seed": 0,
            "resolution": [resolution, resolution],
            "robot_ready": {
                "completed": True,
                "hold_ticks": ROBOT_READY_HOLD_TICKS,
                "trace": robot_trace,
                "target_translation_during_robot_ready_m": (
                    (robot_target[-1] - robot_target[0]).tolist()
                ),
            },
            "diagnostic_tail_ticks": tail_ticks,
            "runtime_scene_ready_horizon_ticks": 40,
            "diagnostic_only": True,
            "oracle_inputs_to_runtime": False,
            "frames": rows,
            "post_robot_ready_oracle_motion_summary": {
                "per_tick_translation_m_median": float(np.median(norms)) if norms.size else None,
                "per_tick_translation_m_p95": float(np.percentile(norms, 95)) if norms.size else None,
                "per_tick_translation_m_max": float(np.max(norms)) if norms.size else None,
                "net_translation_from_tail_start_m": float(np.linalg.norm(target[-1] - target[0])),
                "max_excursion_from_tail_start_m": float(
                    np.linalg.norm(target - target[0], axis=1).max()
                ),
            },
        }
        _write_json(task_dir / "capture.json", result)
        return result
    finally:
        environment.close()


def _load_rgb(frame: Mapping[str, Any]) -> np.ndarray:
    with Image.open(str(frame["canonical_rgb_path"])) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _candidate_row(candidate: Any) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "rank": candidate.rank,
        "backend_index": candidate.backend_index,
        "score": candidate.score,
        "mask_area_px": candidate.area_px,
        "bbox_xyxy": list(candidate.bbox_xyxy) if candidate.bbox_xyxy is not None else None,
        "centroid_px": list(candidate.centroid_px) if candidate.centroid_px is not None else None,
        "mask_available": candidate.mask is not None,
    }


def _mask_iou(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None or first.shape != second.shape:
        return None
    union = int(np.logical_or(first, second).sum())
    if union == 0:
        return None
    return float(np.logical_and(first, second).sum() / union)


def _replay_phrase(
    *, task_id: int, phrase: str, frames: Sequence[Mapping[str, Any]],
    sam3: Sam3Client, output: Path,
) -> dict[str, Any]:
    phrase_key = re.sub(r"[^a-z0-9]+", "_", phrase.casefold()).strip("_")
    phrase_dir = output / f"task_{task_id}" / "sam" / phrase_key
    phrase_dir.mkdir(parents=True, exist_ok=False)
    trace: list[dict[str, Any]] = []
    anchor = None
    evidence = SceneReadyEvidence()
    previous_mask: np.ndarray | None = None
    previous_centroid: np.ndarray | None = None
    previous_area: int | None = None
    scene_ready_tick: int | None = None
    selected_indices = {0, 1, 2, 4, 10, 20, 40, 60, 80}
    overlays: dict[int, Path] = {}
    masks_for_sheet: dict[int, np.ndarray | None] = {}

    for frame in frames:
        tick = int(frame["tail_tick"])
        image = _load_rgb(frame)
        response = sam3.segment(image, phrase, confidence_threshold=0.05)
        segmentation = segmentation_from_response(response, image.shape[:2])
        identity_status = "ANCHOR_UNAVAILABLE"
        associated = None
        association_metrics: list[Mapping[str, Any]] = []
        if anchor is None:
            anchor = make_target_identity_anchor(
                segmentation, target_phrase=phrase, frame_id=tick,
            )
            if anchor is not None:
                identity_status = "ANCHORED"
                associated = next((item for item in segmentation.candidates
                                   if item.candidate_id == segmentation.selected_candidate_id), None)
        else:
            association = associate_target_candidate(anchor, segmentation.candidates)
            identity_status = association.status
            associated = association.candidate
            association_metrics = list(association.candidate_metrics)

        centroid = (np.asarray(associated.centroid_px, dtype=float)
                    if associated is not None and associated.centroid_px is not None else None)
        bbox = associated.bbox_xyxy if associated is not None else None
        area = int(associated.area_px) if associated is not None and associated.area_px else None
        mask = associated.mask if associated is not None else None
        mask_iou_previous = _mask_iou(previous_mask, mask)
        if centroid is not None and previous_centroid is not None and previous_mask is not None:
            ys, xs = np.nonzero(previous_mask)
            # The current associated object's previous bounding-box scale is the
            # normalization denominator; it is independent of absolute image size.
            diag = float(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
            centroid_shift_px = float(np.linalg.norm(centroid - previous_centroid))
            centroid_shift_over_bbox_diag = centroid_shift_px / diag if diag > 0 else None
        else:
            centroid_shift_px = None
            centroid_shift_over_bbox_diag = None
        relative_area_change = (
            abs(area - previous_area) / max(previous_area, 1)
            if area is not None and previous_area is not None else None
        )
        ready = evidence.update(
            target_identity_status=identity_status,
            centroid_px=associated.centroid_px if associated is not None else None,
            bbox_xyxy=bbox,
            mask_area_px=area,
        )
        if ready and scene_ready_tick is None:
            scene_ready_tick = tick

        details = response.get("details") if isinstance(response, Mapping) else None
        detections = details.get("detections") if isinstance(details, Mapping) else None
        row = {
            "tail_tick": tick,
            "environment_step": frame["environment_step"],
            "phrase": phrase,
            "success": bool(response.get("success")),
            "response_content": response.get("content"),
            "error": response.get("error"),
            "failure_reason": details.get("reason") if isinstance(details, Mapping) else None,
            "sam_metadata": details.get("metadata") if isinstance(details, Mapping) else None,
            "candidate_count": len(detections) if isinstance(detections, list) else 0,
            "candidates": [_candidate_row(candidate) for candidate in segmentation.candidates],
            "raw_top_candidate_id": segmentation.selected_candidate_id,
            "raw_top_candidate_score": segmentation.quality_score,
            "identity_status": identity_status,
            "associated_candidate_id": associated.candidate_id if associated is not None else None,
            "association_candidates": association_metrics,
            "centroid_px": list(associated.centroid_px) if associated is not None else None,
            "bbox_xyxy": list(bbox) if bbox is not None else None,
            "mask_area_px": area,
            "mask_iou_previous_associated": mask_iou_previous,
            "centroid_displacement_px": centroid_shift_px,
            "centroid_displacement_over_previous_bbox_diag": centroid_shift_over_bbox_diag,
            "relative_area_change": relative_area_change,
            "old_scene_ready": bool(ready),
            "old_scene_ready_evidence": evidence.to_record(),
        }
        trace.append(row)
        masks_for_sheet[tick] = mask

        if tick in selected_indices:
            overlay = Image.fromarray(image, mode="RGB")
            if mask is not None:
                rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                rgba[mask] = (255, 40, 40, 100)
                overlay = Image.alpha_composite(overlay.convert("RGBA"), Image.fromarray(rgba, mode="RGBA")).convert("RGB")
            draw = ImageDraw.Draw(overlay)
            if bbox is not None:
                draw.rectangle(tuple(bbox), outline=(255, 235, 0), width=3)
            if centroid is not None:
                x, y = (float(value) for value in centroid)
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(0, 255, 255), width=3)
            draw.rectangle((0, 0, image.shape[1], 56), fill=(0, 0, 0))
            draw.text((8, 6), f"{phrase} | tail {tick} | ready={ready} | {identity_status}", fill="white")
            draw.text((8, 30), f"n={row['candidate_count']} score={row['raw_top_candidate_score']} area={area}", fill="white")
            overlay_path = phrase_dir / f"overlay_tick_{tick:03d}.png"
            overlay.save(overlay_path)
            overlays[tick] = overlay_path

        if mask is not None and centroid is not None and area is not None:
            previous_mask = mask
            previous_centroid = centroid
            previous_area = area
        else:
            previous_mask = None
            previous_centroid = None
            previous_area = None

    ready = [row for row in trace if row["old_scene_ready"]]
    valid = [row for row in trace if row["identity_status"] in {"ANCHORED", "SAME_TARGET"}]
    ious = [row["mask_iou_previous_associated"] for row in valid
            if row["mask_iou_previous_associated"] is not None]
    normalized_jitter = [row["centroid_displacement_over_previous_bbox_diag"] for row in valid
                         if row["centroid_displacement_over_previous_bbox_diag"] is not None]
    relative_area = [row["relative_area_change"] for row in valid
                     if row["relative_area_change"] is not None]
    result = {
        "task_id": task_id,
        "phrase": phrase,
        "frame_count": len(trace),
        "same_saved_frame_sequence": True,
        "old_scene_ready": {
            "ready": scene_ready_tick is not None,
            "first_ready_tail_tick": scene_ready_tick,
            "timeout_at_40_holds": scene_ready_tick is None or scene_ready_tick > 40,
            "horizon_40_sample_ready_count": sum(
                bool(row["old_scene_ready"]) for row in trace[:41]
            ),
        },
        "grounding": {
            "any_candidate_count_positive": any(row["candidate_count"] > 0 for row in trace),
            "all_frames_candidate_count_positive": all(row["candidate_count"] > 0 for row in trace),
            "associated_frame_count": len(valid),
            "identity_lost_frame_count": sum(row["identity_status"] == "TARGET_IDENTITY_LOST" for row in trace),
            "anchor_unavailable_frame_count": sum(row["identity_status"] == "ANCHOR_UNAVAILABLE" for row in trace),
        },
        "stability_summary": {
            "associated_mask_iou_to_previous_median": float(np.median(ious)) if ious else None,
            "associated_mask_iou_to_previous_p10": float(np.percentile(ious, 10)) if ious else None,
            "normalized_centroid_jitter_median": float(np.median(normalized_jitter)) if normalized_jitter else None,
            "normalized_centroid_jitter_p95": float(np.percentile(normalized_jitter, 95)) if normalized_jitter else None,
            "relative_area_change_median": float(np.median(relative_area)) if relative_area else None,
            "relative_area_change_p95": float(np.percentile(relative_area, 95)) if relative_area else None,
        },
        "trace": trace,
        "contact_sheet": None,
    }
    _write_json(phrase_dir / "sam_trace.json", result)
    result["contact_sheet"] = str(_contact_sheet(task_id, phrase, frames, trace, overlays,
                                                    output / f"task_{task_id}" / "contacts"))
    _write_json(phrase_dir / "sam_trace.json", result)
    return result


def _contact_sheet(
    task_id: int, phrase: str, frames: Sequence[Mapping[str, Any]],
    trace: Sequence[Mapping[str, Any]], overlays: Mapping[int, Path], output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = {int(row["tail_tick"]): row for row in trace}
    ticks = [tick for tick in (0, 1, 2, 4, 10, 20, 40, 60, 80) if tick in overlays]
    thumb_w, thumb_h, label_h = 256, 256, 46
    columns = 3
    row_count = (len(ticks) + columns - 1) // columns
    chart_h = 180
    sheet = Image.new("RGB", (columns * thumb_w, row_count * (thumb_h + label_h) + chart_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, tick in enumerate(ticks):
        with Image.open(overlays[tick]) as image:
            thumb = image.convert("RGB").resize((thumb_w, thumb_h))
        x = (index % columns) * thumb_w
        y = (index // columns) * (thumb_h + label_h)
        sheet.paste(thumb, (x, y))
        row = rows[tick]
        bbox = row.get("bbox_xyxy")
        center = row.get("centroid_px")
        label = (
            f"t{tick} ready={row['old_scene_ready']} id={row['identity_status']}\n"
            f"IoU={row['mask_iou_previous_associated']} d/diag={row['centroid_displacement_over_previous_bbox_diag']}\n"
            f"area={row['mask_area_px']} bbox={bbox} c={center}"
        )
        draw.text((x + 4, y + thumb_h + 3), label, fill=(0, 0, 0))

    chart_top = row_count * (thumb_h + label_h) + 6
    chart_left, chart_right = 48, sheet.width - 20
    chart_bottom = sheet.height - 25
    draw.text((chart_left, chart_top), "Blue: normalized full-frame RGB difference  |  Red: oracle target displacement", fill=(0, 0, 0))
    draw.line((chart_left, chart_top + 28, chart_left, chart_bottom), fill=(0, 0, 0), width=1)
    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=(0, 0, 0), width=1)
    frame_by_tick = {int(frame["tail_tick"]): frame for frame in frames}
    diff_values = [frame_by_tick[t].get("rgb_motion", {}).get("mean_abs_rgb_delta_0_1")
                   for t in sorted(frame_by_tick) if frame_by_tick[t].get("rgb_motion")]
    target_positions = np.asarray([frame["target_world_position_m"] for frame in frames], dtype=float)
    oracle = np.linalg.norm(np.diff(target_positions, axis=0), axis=1).tolist()
    max_diff = max((float(v) for v in diff_values if v is not None), default=1e-9)
    max_oracle = max(oracle, default=1e-9)
    points_diff: list[tuple[float, float]] = []
    points_oracle: list[tuple[float, float]] = []
    diff_rows = [frame_by_tick[t] for t in sorted(frame_by_tick) if frame_by_tick[t].get("rgb_motion")]
    for i, frame in enumerate(diff_rows):
        x = chart_left + (i / max(1, len(diff_rows) - 1)) * (chart_right - chart_left)
        value = float(frame["rgb_motion"]["mean_abs_rgb_delta_0_1"])
        y = chart_bottom - value / max_diff * (chart_bottom - (chart_top + 36))
        points_diff.append((x, y))
    for i, value in enumerate(oracle):
        x = chart_left + (i / max(1, len(oracle) - 1)) * (chart_right - chart_left)
        y = chart_bottom - value / max_oracle * (chart_bottom - (chart_top + 36))
        points_oracle.append((x, y))
    if len(points_diff) > 1:
        draw.line(points_diff, fill=(35, 85, 205), width=2)
    if len(points_oracle) > 1:
        draw.line(points_oracle, fill=(215, 50, 45), width=2)
    draw.text((chart_left, chart_bottom + 3), "0", fill=(0, 0, 0))
    draw.text((chart_right - 20, chart_bottom + 3), str(len(frames) - 1), fill=(0, 0, 0))
    target = output_dir / f"task_{task_id}_{re.sub(r'[^a-z0-9]+', '_', phrase.casefold()).strip('_')}.png"
    sheet.save(target)
    return target


def _phrase_comparison(task_dir: Path, reference: Mapping[str, Any], qwen: Mapping[str, Any]) -> Path:
    ref_rows = {int(row["tail_tick"]): row for row in reference["trace"]}
    qwen_rows = {int(row["tail_tick"]): row for row in qwen["trace"]}
    picks = (0, 1, 2, 4, 10, 20, 40, 60, 80)
    cell_w, cell_h = 192, 236
    sheet = Image.new("RGB", (cell_w * 4, cell_h * len(picks)), "white")
    draw = ImageDraw.Draw(sheet)
    headers = ("same RGB", "alphabet soup", "the alphabet soup", "measures")
    for index, title in enumerate(headers):
        draw.text((index * cell_w + 4, 2), title, fill=(0, 0, 0))
    ref_phrase = task_dir / "sam" / "alphabet_soup" / "overlays"
    # Contact overlays are stored in the phrase directory itself.
    ref_phrase = task_dir / "sam" / "alphabet_soup"
    qwen_phrase = task_dir / "sam" / "the_alphabet_soup"
    for row_index, tick in enumerate(picks):
        y = row_index * cell_h + 20
        rgb_path = task_dir / "rgb" / f"tick_{tick:03d}.png"
        for col, path in enumerate((rgb_path, ref_phrase / f"overlay_tick_{tick:03d}.png",
                                    qwen_phrase / f"overlay_tick_{tick:03d}.png")):
            if path.exists():
                with Image.open(path) as image:
                    thumb = image.convert("RGB").resize((cell_w, 192))
                sheet.paste(thumb, (col * cell_w, y))
        ref_row, qwen_row = ref_rows[tick], qwen_rows[tick]
        label = (
            f"tick {tick}\nref: {ref_row['identity_status']} ready={ref_row['old_scene_ready']} "
            f"n={ref_row['candidate_count']} iou={ref_row['mask_iou_previous_associated']}\n"
            f"qwen: {qwen_row['identity_status']} ready={qwen_row['old_scene_ready']} "
            f"n={qwen_row['candidate_count']} iou={qwen_row['mask_iou_previous_associated']}"
        )
        draw.text((3 * cell_w + 4, y + 2), label, fill=(0, 0, 0))
        draw.text((4, y + 195), f"tail={tick}", fill=(0, 0, 0))
    target = task_dir / "contacts" / "task_0_phrase_comparison_same_frames.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target)
    return target


def run_capture(args: argparse.Namespace) -> int:
    config = load_yaml(args.config)
    output = _new_run_dir(Path(args.output_dir).expanduser())
    _write_json(output / "RUN_SETUP_READY.json", {
        "status": "RUN_SETUP_READY",
        "phase": "HOLD_ONLY_CANONICAL_RGB_AND_ORACLE_CAPTURE",
        "tasks": args.tasks,
        "init_states": [0],
        "robot_ready_hold_ticks": ROBOT_READY_HOLD_TICKS,
        "diagnostic_tail_hold_ticks": args.tail_ticks,
        "alignment_actions_authorized": False,
        "sam_calls_during_capture": 0,
        "oracle_used_by_runtime": False,
        "formal_readiness_result": False,
    })
    summaries = []
    for task_id in args.tasks:
        print(f"CAPTURE task_id={task_id}", flush=True)
        summaries.append(_capture_task(
            task_id=task_id, output=output, config=config,
            resolution=args.resolution, tail_ticks=args.tail_ticks,
        ))
    result = {
        "status": "CAPTURE_COMPLETE",
        "run_dir": str(output),
        "sam_calls_during_capture": 0,
        "runtime_received_oracle": False,
        "tasks": [{key: value for key, value in summary.items() if key != "frames"}
                  for summary in summaries],
    }
    _write_json(output / "capture_summary.json", result)
    print(json.dumps(_jsonable(result), indent=2, ensure_ascii=False))
    print(f"RUN_DIR={output}")
    return 0


def run_replay(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    _configure_local_proxy_bypass(args.sam3_url)
    sam3 = Sam3Client(url=args.sam3_url, python=args.sam3_python,
                      timeout_s=args.sam3_timeout_s, max_attempts=1)
    rows: list[dict[str, Any]] = []
    try:
        for task_id in args.tasks:
            capture_path = run_dir / f"task_{task_id}" / "capture.json"
            capture = json.loads(capture_path.read_text(encoding="utf-8"))
            frames = capture["frames"]
            phrases = [TASKS[task_id]["phrase"]]
            if task_id == 0:
                phrases.append(TASKS[task_id]["qwen_phrase"])
            for phrase in phrases:
                print(f"SAM_REPLAY task_id={task_id} phrase={phrase!r} frames={len(frames)}", flush=True)
                rows.append(_replay_phrase(
                    task_id=task_id, phrase=phrase, frames=frames,
                    sam3=sam3, output=run_dir,
                ))
    finally:
        sam3.close()
    task0_rows = {row["phrase"]: row for row in rows if row["task_id"] == 0}
    phrase_compare = None
    if "alphabet soup" in task0_rows and "the alphabet soup" in task0_rows:
        phrase_compare = {
            "same_rgb_sequence": True,
            "same_frame_count": task0_rows["alphabet soup"]["frame_count"] == task0_rows["the alphabet soup"]["frame_count"],
            "reference_ready_tail_tick": task0_rows["alphabet soup"]["old_scene_ready"]["first_ready_tail_tick"],
            "qwen_ready_tail_tick": task0_rows["the alphabet soup"]["old_scene_ready"]["first_ready_tail_tick"],
            "reference_associated_frames": task0_rows["alphabet soup"]["grounding"]["associated_frame_count"],
            "qwen_associated_frames": task0_rows["the alphabet soup"]["grounding"]["associated_frame_count"],
            "contact_sheet": str(_phrase_comparison(run_dir / "task_0", task0_rows["alphabet soup"], task0_rows["the alphabet soup"])),
        }
    summary = {
        "status": "OFFLINE_SAM_REPLAY_COMPLETE",
        "run_dir": str(run_dir),
        "alignment_actions": 0,
        "runtime_received_oracle": False,
        "query_comparison": phrase_compare,
        "phrases": rows,
    }
    task_suffix = "-".join(str(task_id) for task_id in args.tasks)
    summary_path = run_dir / f"sam_replay_summary_tasks_{task_suffix}.json"
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/robot_libero_clean_qwen3vl.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "rollouts/runtime_v3_m3_7_readiness"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--sam3-url", default="http://127.0.0.1:8773/sse")
    parser.add_argument("--sam3-python", default="/root/autodl-tmp/openeta-services/sam3/.venv/bin/python")
    parser.add_argument("--sam3-timeout-s", type=float, default=120.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--tail-ticks", type=int, default=DIAGNOSTIC_TAIL_HOLD_TICKS)
    parser.add_argument("--tasks", type=lambda text: [int(part.strip()) for part in text.split(",")],
                        default=[0, 2, 6, 7])
    parser.add_argument("--phase", choices=("capture", "replay"), default="capture")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.phase == "capture":
        return run_capture(args)
    if args.run_dir is None:
        raise SystemExit("--phase replay requires --run-dir")
    return run_replay(args)


if __name__ == "__main__":
    raise SystemExit(main())

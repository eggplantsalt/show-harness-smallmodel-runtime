#!/usr/bin/env python3
"""Audit LIBERO image orientation and direct-render SAM3 resolution behavior."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.capabilities.sam3_client import Sam3Client
from core.config import load_secrets_env, load_yaml
from core.runtime_v3.adapters.libero_env import LiberoEnvironmentAdapter
from core.runtime_v3.adapters.libero_observation import LiberoObservationAdapter
from core.runtime_v3.canonical_image import CanonicalImageAdapter
from core.runtime_v3.object_relative import TargetSegmentation, segmentation_from_response
from core.runtime_v3.temporal_calibration import run_v3_tick
from core.record.images import image_to_data_url
from core.sim.launch import build_config, make_vlm_client
from interpreters.libero_atomic_controller import LiberoAtomicController


SUITE = "LIBERO_OBJECT"
TASK_ID = 2
INIT_STATES = (0, 1, 2)
TARGET_PHRASE = "salad dressing"
PRE_SETTLE_TICKS = 4
RESOLUTIONS = (512, 768)
CANONICAL = CanonicalImageAdapter()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/robot_libero_clean_qwen3vl.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "rollouts/runtime_v3_object_relative_alignment"))
    parser.add_argument("--sam3-url", default="http://127.0.0.1:8773/sse")
    parser.add_argument("--sam3-python", default="/root/autodl-tmp/openeta-services/sam3/.venv/bin/python")
    parser.add_argument("--sam3-timeout-s", type=float, default=120.0)
    parser.add_argument("--vlm-backend", default=None)
    parser.add_argument("--vlm-url", default=os.environ.get("VLM_URL") or os.environ.get("VLLM_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL"))
    return parser


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _write_json(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(record), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _configure_local_sam3_proxy_bypass(url: str) -> None:
    if urlparse(str(url)).hostname not in {"127.0.0.1", "localhost", "::1"}:
        return
    entries = []
    for name in ("NO_PROXY", "no_proxy"):
        entries.extend(value.strip() for value in os.environ.get(name, "").split(",") if value.strip())
    entries.extend(("127.0.0.1", "localhost", "::1"))
    value = ",".join(dict.fromkeys(entries))
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def _new_run_dir(base: str | Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(base).expanduser() / f"perception_audit_{stamp}_{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def _capture_frame(*, size: int, init_state: int, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    environment = LiberoEnvironmentAdapter.create(
        suite_name=SUITE, task_id=TASK_ID, init_state_index=init_state, seed=0,
        camera_height=size, camera_width=size, horizon=32,
    )
    try:
        controller = LiberoAtomicController(
            move_vectors=config["move_vectors"], step_m=0.005, sim_steps_per_decision=1,
            position_scale_m=float(config.get("position_scale_m", 0.05)),
        )
        observer = LiberoObservationAdapter(
            min_eef_z_m=0.02, max_eef_z_m=0.60, safe_lift_step_m=0.005,
        )
        hold_cycles = []
        reset = True
        for _ in range(PRE_SETTLE_TICKS):
            result = run_v3_tick(
                environment, observer, controller, task_id=f"{SUITE}:{TASK_ID}",
                token=None, direction_unit=None, commanded_step_m=0.005, reset=reset,
                workspace_z_bounds_m=(0.02, 0.60),
            )
            reset = False
            if result["actions"] != 1 or not result["backend_execution"]:
                raise RuntimeError(f"HOLD settle tick failed: {result}")
            hold_cycles.append(result)
        observation = observer.observe(environment)
        raw = np.ascontiguousarray(observation.images["agentview"])
        return raw, {
            "hold_cycles": hold_cycles,
            "source_shape": list(raw.shape),
            "source_dtype": str(raw.dtype),
            "environment_observation_index": observer.observation_index,
            "environment_step": observer.last_raw.environment_step if observer.last_raw else None,
        }
    finally:
        environment.close()


def _save_orientation_sheet(raw: np.ndarray, output: Path) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    variants = {
        "raw": np.ascontiguousarray(raw),
        "identity": np.ascontiguousarray(raw.copy()),
        "flip_vertical": CANONICAL.transform_image(raw),
        "flip_horizontal": np.ascontiguousarray(np.fliplr(raw)),
        "rotate_180": np.ascontiguousarray(np.rot90(raw, 2)),
    }
    paths = {}
    for name, image in variants.items():
        path = output / f"{name}.png"
        Image.fromarray(image, mode="RGB").save(path)
        paths[name] = str(path)
    tile = 384
    sheet = Image.new("RGB", (tile * 2, tile * 2 + 72), "white")
    labels = ("RAW / IDENTITY", "VERTICAL FLIP", "HORIZONTAL FLIP", "ROTATE 180")
    keys = ("raw", "flip_vertical", "flip_horizontal", "rotate_180")
    draw = ImageDraw.Draw(sheet)
    for index, (key, label) in enumerate(zip(keys, labels)):
        x, y = (index % 2) * tile, (index // 2) * tile
        sheet.paste(Image.fromarray(variants[key]).resize((tile, tile)), (x, y))
        draw.rectangle((x, y + tile, x + tile, y + tile + 36), fill="white")
        draw.text((x + 8, y + tile + 9), label, fill="black")
    contact_path = output / "orientation_contact_sheet.png"
    sheet.save(contact_path)
    paths["contact_sheet"] = str(contact_path)
    _write_json(output / "orientation_audit.json", {
        "suite": SUITE, "task_id": TASK_ID, "seed": 0, "init_state_index": 0,
        "settle_hold_ticks": PRE_SETTLE_TICKS, "source_resolution_width_height": [raw.shape[1], raw.shape[0]],
        "source_shape": list(raw.shape), "canonical_transform": CANONICAL.orientation,
        "identity_is_raw_copy": bool(np.array_equal(variants["raw"], variants["identity"])),
        "robot_alignment_actions": 0, "artifacts": paths,
    })
    return paths


def _save_candidates(image: np.ndarray, segmentation: TargetSegmentation, output: Path) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    overlay = Image.fromarray(image, mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    colors = ((255, 40, 40, 75), (30, 200, 255, 75), (30, 230, 80, 75),
              (255, 190, 0, 75), (230, 30, 210, 75))
    rows = []
    for index, candidate in enumerate(segmentation.candidates):
        row = {
            "candidate_id": candidate.candidate_id, "rank": candidate.rank,
            "backend_index": candidate.backend_index, "score": candidate.score,
            "area_px": candidate.area_px, "centroid_px": candidate.centroid_px,
            "bbox_xyxy": candidate.bbox_xyxy,
        }
        if candidate.mask is not None:
            mask_path = output / f"candidate_{index:02d}_mask.png"
            Image.fromarray(candidate.mask.astype(np.uint8) * 255, mode="L").save(mask_path)
            row["mask_path"] = str(mask_path)
            tint = np.zeros((*candidate.mask.shape, 4), dtype=np.uint8)
            tint[candidate.mask] = colors[index % len(colors)]
            overlay = Image.alpha_composite(overlay, Image.fromarray(tint, mode="RGBA"))
        if candidate.bbox_xyxy is not None:
            x0, y0, x1, y1 = candidate.bbox_xyxy
            draw = ImageDraw.Draw(overlay)
            color = (255, 240, 0, 255) if index == 0 else (255, 255, 255, 255)
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=2)
            draw.text((x0, max(0, y0 - 12)), f"id={candidate.candidate_id} s={candidate.score}", fill=color)
        rows.append(row)
    overlay.convert("RGB").save(output / "all_candidates_overlay.png")
    _write_json(output / "candidates.json", {
        "candidate_count": len(segmentation.candidates), "top_candidate_id": segmentation.selected_candidate_id,
        "candidates": rows,
    })
    return rows


def _capture_resolution_rows(
    *, output: Path, config: dict[str, Any], sam3: Sam3Client,
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    rows = []
    qwen_source = None
    for size in RESOLUTIONS:
        for init_state in INIT_STATES:
            trial_dir = output / "resolution_audit" / str(size) / f"init_state_{init_state}"
            trial_dir.mkdir(parents=True, exist_ok=False)
            raw, capture_meta = _capture_frame(size=size, init_state=init_state, config=config)
            canonical = CANONICAL.transform_image(raw)
            raw_path, canonical_path = trial_dir / "raw.png", trial_dir / "canonical.png"
            Image.fromarray(raw, mode="RGB").save(raw_path)
            Image.fromarray(canonical, mode="RGB").save(canonical_path)
            response = sam3.segment(canonical, TARGET_PHRASE, confidence_threshold=0.05)
            segmentation = segmentation_from_response(response, canonical.shape[:2])
            candidates = _save_candidates(canonical, segmentation, trial_dir / "sam3_candidates")
            details = response.get("details") if isinstance(response, dict) else None
            metadata = details.get("metadata") if isinstance(details, dict) else None
            reported_size = metadata.get("image_size") if isinstance(metadata, dict) else None
            top = segmentation.candidates[0] if segmentation.candidates else None
            row = {
                "init_state_index": init_state,
                "configured_renderer_width_height": [size, size],
                "observed_source_width_height": [int(raw.shape[1]), int(raw.shape[0])],
                "sam3_input_width_height": [int(canonical.shape[1]), int(canonical.shape[0])],
                "sam3_reported_width_height": reported_size,
                "canonical_transform": CANONICAL.orientation,
                "candidate_count": len(segmentation.candidates),
                "target_detected": bool(segmentation.visible),
                "top_score": top.score if top else None,
                "top_area_px": top.area_px if top else None,
                "top_area_fraction": (float(top.area_px / (size * size)) if top and top.area_px else None),
                "top_centroid_px": top.centroid_px if top else None,
                "top_centroid_normalized": ([top.centroid_px[0] / size, top.centroid_px[1] / size]
                                             if top and top.centroid_px else None),
                "top_bbox_xyxy": top.bbox_xyxy if top else None,
                "all_candidates": candidates,
                "capture": capture_meta,
                "response_error": response.get("error") if not segmentation.candidates else None,
                "artifacts": {"raw_rgb": str(raw_path), "canonical_rgb": str(canonical_path),
                              "candidate_overlay": str(trial_dir / "sam3_candidates/all_candidates_overlay.png"),
                              "candidate_metadata": str(trial_dir / "sam3_candidates/candidates.json")},
            }
            _write_json(trial_dir / "audit.json", row)
            rows.append(row)
            if size == 768 and init_state == 0:
                qwen_source = raw.copy()
                _save_orientation_sheet(raw, output / "orientation_audit")
    return rows, qwen_source


def _qwen_ocr_sanity(
    *, args: argparse.Namespace, config: dict[str, Any], raw_image: np.ndarray, output: Path,
) -> dict[str, Any]:
    load_secrets_env()
    vlm_args = argparse.Namespace(
        task_suite_name=None, task_id=None, episode_index=None, max_steps=None,
        loop_period_s=None, log_dir=None, vlm_backend=args.vlm_backend,
        vlm_url=args.vlm_url, model=args.model,
    )
    resolved = build_config(vlm_args, config)
    client = make_vlm_client(vlm_args, resolved)
    prompt = "What prominent English word is printed on the red carton? Answer one word."
    results = {}
    try:
        for name, image in (("raw_orientation", raw_image),
                            ("canonical_orientation", CANONICAL.transform_image(raw_image))):
            data_url = image_to_data_url(image)
            encoded = data_url.split(",", 1)[1]
            png_bytes = base64.b64decode(encoded)
            with Image.open(io.BytesIO(png_bytes)) as wire_png:
                wire_size = list(wire_png.size)
            response = client.complete_text(
                prompt, image, wrist_image=None, max_tokens=32, temperature=0.0,
                debug=True, image_detail="high", agentview_label="Image A: agentview RGB",
            )
            results[name] = {
                "response": response.raw_text,
                "input_width_height": [int(image.shape[1]), int(image.shape[0])],
                "serialized_png_width_height": wire_size,
                "serialized_png_sha256": hashlib.sha256(png_bytes).hexdigest(),
                "robot_actions": 0,
                "payload_metadata": response.payload,
            }
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    record = {
        "mode": "NO_ACTION_ORIENTATION_OCR_SANITY",
        "model": getattr(client, "model", args.model),
        "prompt": prompt,
        "results": results,
        "robot_actions": 0,
        "image_source": "direct 768x768 task-2 init-state-0 simulator render after 4 HOLD ticks",
    }
    _write_json(output / "qwen_orientation_sanity.json", record)
    return record


def _resolution_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for size in RESOLUTIONS:
        group = [row for row in rows if row["configured_renderer_width_height"][0] == size]
        scores = [row["top_score"] for row in group if row["top_score"] is not None]
        counts = [row["candidate_count"] for row in group]
        result[str(size)] = {
            "renderer_and_sam_size": [size, size],
            "target_detection_count": sum(bool(row["target_detected"]) for row in group),
            "target_detection_rate": sum(bool(row["target_detected"]) for row in group) / len(group) if group else None,
            "candidate_counts_by_init": counts,
            "mean_candidate_count": float(np.mean(counts)) if counts else None,
            "mean_top_score": float(np.mean(scores)) if scores else None,
            "top_normalized_centroids_by_init": [row["top_centroid_normalized"] for row in group],
            "top_mask_area_fractions_by_init": [row["top_area_fraction"] for row in group],
        }
    return result


def main() -> int:
    args = _parser().parse_args()
    config = load_yaml(args.config)
    if config.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(config["libero_dir"])
    _configure_local_sam3_proxy_bypass(args.sam3_url)
    output = _new_run_dir(args.output_dir)
    sam3 = Sam3Client(url=args.sam3_url, python=args.sam3_python, timeout_s=args.sam3_timeout_s,
                      max_attempts=1)
    blockers = []
    rows = []
    qwen_record = None
    try:
        rows, qwen_source = _capture_resolution_rows(output=output, config=config, sam3=sam3)
        if qwen_source is not None:
            try:
                qwen_record = _qwen_ocr_sanity(args=args, config=config,
                                               raw_image=qwen_source, output=output)
            except Exception as exc:
                blockers.append(f"qwen_ocr: {type(exc).__name__}: {exc}")
                _write_json(output / "qwen_orientation_sanity.json", {
                    "status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}",
                    "robot_actions": 0,
                })
    except Exception as exc:
        blockers.append(f"resolution_audit: {type(exc).__name__}: {exc}")
    finally:
        sam3.close()
    summary = {
        "status": "COMPLETED" if len(rows) == len(RESOLUTIONS) * len(INIT_STATES) and not blockers else "PARTIAL",
        "branch": "runtime-v3", "baseline_commit": "d8b52dbb5ddaa0f4417aa6a0ee0de4b97ab80ba1",
        "suite": SUITE, "task_id": TASK_ID, "seed": 0, "init_states": list(INIT_STATES),
        "target_phrase": TARGET_PHRASE, "settle_hold_ticks": PRE_SETTLE_TICKS,
        "resolution_rows": rows, "resolution_summary": _resolution_summary(rows),
        "qwen_orientation_sanity": qwen_record,
        "canonical_transform": CANONICAL.orientation,
        "simulator_resize_or_upsample": False, "robot_alignment_actions": 0,
        "oracle_diagnostic_only": False, "oracle_used_by_runtime": False,
        "blockers": blockers,
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"ARTIFACT_DIR={output}")
    return 0 if summary["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

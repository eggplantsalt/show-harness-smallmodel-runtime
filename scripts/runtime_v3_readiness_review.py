#!/usr/bin/env python3
"""Render readable post-hoc contact sheets from M3.7 Stage A records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


def _pair_metrics(sample: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    evidence = sample.get("entity_observation_evidence", {}) or {}
    interval = evidence.get("last_visual_interval") or {}
    return (
        interval.get("centroid_shift_over_previous_bbox_diagonal"),
        interval.get("adjacent_mask_iou"),
        interval.get("bbox_max_edge_shift_over_previous_bbox_diagonal"),
        interval.get("relative_area_change"),
    )


def _chart(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
           motion: Sequence[float | None], oracle: Sequence[float]) -> None:
    left, top, right, bottom = box
    draw.text((left, top), "Blue: normalized RGB difference   Red: oracle target displacement (diagnostic)", fill="black")
    chart_top = top + 22
    draw.line((left, chart_top, left, bottom), fill="black")
    draw.line((left, bottom, right, bottom), fill="black")
    finite_motion = [float(value) for value in motion if value is not None]
    max_motion = max(max(finite_motion, default=1e-9), 0.00005)
    max_oracle = max(max((float(value) for value in oracle), default=1e-9), 0.001)

    def points(values: Sequence[float | None], max_value: float):
        result = []
        for index, value in enumerate(values):
            if value is None:
                continue
            x = left + index / max(1, len(values) - 1) * (right - left)
            y = bottom - min(float(value) / max_value, 1.0) * (bottom - chart_top)
            result.append((x, y))
        return result

    rgb_points = points(motion, max_motion)
    oracle_points = points(oracle, max_oracle)
    if len(rgb_points) > 1:
        draw.line(rgb_points, fill=(35, 85, 205), width=2)
    if len(oracle_points) > 1:
        draw.line(oracle_points, fill=(210, 45, 45), width=2)
    draw.text((left, bottom + 3), "observation 0", fill="black")
    draw.text((right - 90, bottom + 3), f"{max(len(motion), len(oracle)) - 1}", fill="black")


def _make_sheet(episode: Mapping[str, Any], source_sheet: Path, output: Path,
                *, title: str, raw_phrase_note: str | None = None) -> Path:
    trace = list(episode.get("readiness_trace", []))
    if not trace or not source_sheet.exists():
        raise ValueError(f"episode trace or source contact sheet missing: {source_sheet}")
    with Image.open(source_sheet) as source:
        source = source.convert("RGB")
        if source.size[0] != 960:
            raise ValueError(f"unexpected original contact-sheet width: {source.size}")
        tile_height, tile_label = 320, 58
        tile_stride = tile_height + tile_label

        ticks = sorted(set(value for value in (0, 1, 2, 3, 4, 10, 20, 40, len(trace) - 1)
                           if 0 <= value < len(trace)))
        cell_w, cell_h, label_h = 256, 256, 70
        cols = 3
        rows = math.ceil(len(ticks) / cols)
        chart_h = 190
        top_h = 48
        sheet = Image.new("RGB", (cols * cell_w, top_h + rows * (cell_h + label_h) + chart_h), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 4), title, fill="black")
        if raw_phrase_note:
            draw.text((4, 24), raw_phrase_note, fill="black")
        content_top = top_h
        for index, tick in enumerate(ticks):
            source_x = (index % 3) * 320
            source_y = (index // 3) * tile_stride
            panel = source.crop((source_x, source_y, source_x + tile_height,
                                 source_y + tile_height)).resize((cell_w, cell_h))
            x0 = (index % cols) * cell_w
            y0 = content_top + (index // cols) * (cell_h + label_h)
            sheet.paste(panel, (x0, y0))
            sample = trace[tick]
            centroid, iou, edges, area = _pair_metrics(sample)
            rgb = (sample.get("scene_motion_evidence", {}) or {}).get(
                "last_normalized_rgb_difference"
            )
            fmt = lambda value: "n/a" if value is None else f"{float(value):.4g}"
            draw.text((x0 + 3, y0 + cell_h + 2),
                      f"t={tick} ground={int(sample.get('grounding_success', False))} "
                      f"identity={int(sample.get('identity_valid', False))} "
                      f"ready={int(sample.get('scene_motion_ready', False))}/"
                      f"{int(sample.get('entity_observation_ready', False))}",
                      fill="black")
            draw.text((x0 + 3, y0 + cell_h + 18),
                      f"RGB Δ={fmt(rgb)}", fill="black")
            draw.text((x0 + 3, y0 + cell_h + 34),
                      f"c/diag={fmt(centroid)} IoU={fmt(iou)}", fill="black")
            draw.text((x0 + 3, y0 + cell_h + 50),
                      f"edge/diag={fmt(edges)} areaΔ={fmt(area)}",
                      fill="black")

        chart_top = content_top + rows * (cell_h + label_h) + 5
        oracle_record = episode.get("oracle_diagnostic_only", {}) or {}
        _chart(draw, (42, chart_top, sheet.width - 8, sheet.height - 23),
               [row.get("scene_motion_evidence", {}).get("last_normalized_rgb_difference")
                for row in trace],
               oracle_record.get("per_observation_translation_m", []))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def render(stage_a_dir: Path, output_dir: Path, tasks: Sequence[int]) -> list[str]:
    summary = json.loads((stage_a_dir / "summary.json").read_text(encoding="utf-8"))
    outputs: list[str] = []
    for task_id in tasks:
        episode = next((row for row in summary["episodes"]
                        if int(row["task_id"]) == int(task_id)
                        and int(row["init_state_index"]) == 0), None)
        if episode is None:
            raise ValueError(f"no init-state-0 episode for task {task_id}")
        source = Path(episode["contact_sheet"])
        path = output_dir / f"task_{task_id}_readiness_review.png"
        _make_sheet(episode, source, path, title=(
            f"LIBERO_OBJECT task {task_id}: {episode['task_phrase']} — M3.7 readiness"
        ))
        outputs.append(str(path))
        if task_id == 0:
            qwen_path = output_dir / "task_0_qwen_phrase_normalized_review.png"
            _make_sheet(
                episode, source, qwen_path,
                title="LIBERO_OBJECT task 0: same Stage A RGB and selected mask",
                raw_phrase_note=(
                    "raw Qwen phrase: 'the alphabet soup' -> grounding query: 'alphabet soup'"
                ),
            )
            outputs.append(str(qwen_path))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tasks", nargs="+", type=int, default=[0, 6, 1, 8])
    args = parser.parse_args()
    paths = render(args.stage_a_dir.resolve(), args.output_dir.resolve(), args.tasks)
    print(json.dumps({"reviewed_contact_sheets": paths}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare frozen V2.2 seating requests without stepping the simulator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.runtime_v2.placement_memory import build_placement_panel  # noqa: E402
from core.vlm.roles import ControllerAgent  # noqa: E402
from core.vlm.vlm_client import VLMClient  # noqa: E402


def prepare_case(run_dir: Path) -> tuple[dict, np.ndarray]:
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines() if line]
    transport = [event for event in events if event.get("stage") == "TRANSPORT"]
    candidates = [event for event in transport if ((event.get("evidence") or {}).get("visual_route") or {}).get("progress", {}).get("contact_candidate")]
    current = candidates[0] if candidates else next(
        (event for event in reversed(transport) if ((event.get("evidence") or {}).get("placement_belief") or {}).get("fresh")),
        transport[-1],
    )
    current_frame = int(current["frame_id"])
    belief = current.get("belief_after") or {}
    target = belief.get("target") or {}
    instance_id = str(target.get("instance_id") or "")
    epoch = int(belief.get("grasp_epoch", -1))
    eligible = [
        event for event in transport
        if int(event.get("frame_id", -1)) <= current_frame
        and (event.get("belief_after") or {}).get("grasp_epoch") == epoch
        and ((event.get("belief_after") or {}).get("target") or {}).get("instance_id") == instance_id
    ]
    seating = next((event for event in eligible if (((event.get("evidence") or {}).get("visual_route") or {}).get("route") or {}).get("active_leg") in {"PRE_DESCENT", "DESCENT"}), None)
    selected = {current_frame: current}
    if seating is not None and int(seating["frame_id"]) < current_frame:
        selected[int(seating["frame_id"])] = seating
    prior = next((event for event in reversed(eligible) if int(event["frame_id"]) < current_frame), None)
    if prior is not None:
        selected[int(prior["frame_id"])] = prior
    bundle = []
    by_frame = {int(event["frame_id"]): event for event in eligible}
    for frame, event in sorted(selected.items())[-3:]:
        route = (event.get("evidence") or {}).get("visual_route") or {}
        placement = route.get("placement_belief") or {}
        previous = by_frame.get(frame - 1)
        current_eef = (((event.get("belief_after") or {}).get("eef_xyz") or {}).get("value"))
        previous_eef = (((previous or {}).get("belief_after") or {}).get("eef_xyz") or {}).get("value")
        motion_delta = (
            [float(now) - float(before) for before, now in zip(previous_eef[:3], current_eef[:3])]
            if isinstance(previous_eef, list) and isinstance(current_eef, list)
            and len(previous_eef) >= 3 and len(current_eef) >= 3 else None
        )
        bundle.append({
            "instance_id": instance_id, "grasp_epoch": epoch, "frame_id": frame,
            "agentview_ref": f"images/raw_agentview/{frame:04d}.png",
            "wrist_ref": f"images/raw_wrist/{frame:04d}.png",
            "executed_action": event.get("executed_action"),
            "motion_delta": motion_delta,
            "route_phase": (route.get("route") or {}).get("active_leg"),
            "placement_summary": {key: placement.get(key) for key in ("rim_clearance_m", "containment_margin_m", "uncertainty_m", "fresh", "conflicts")},
        })
    panel, panel_meta = build_placement_panel(run_dir, bundle, instance_id=instance_id, grasp_epoch=epoch, current_frame=current_frame)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    subgoals = json.loads((run_dir / "subgoals.json").read_text()).get("planner_subgoals", [])
    place = next((item for item in subgoals if item.get("motion") in {"TRANSPORT", "MOVE", "PLACE"}), {})
    return {
        "run_dir": str(run_dir), "frame_id": current_frame, "task": metadata.get("task_description", "place the held object"),
        "target": place.get("target", "receptacle"), "affordance": place.get("affordance", "opening"),
        "placement_evidence": ((current.get("evidence") or {}).get("visual_route") or {}).get("placement_belief") or {},
        "contact_candidate": ((current.get("evidence") or {}).get("visual_route") or {}).get("progress", {}).get("contact_candidate"),
        "contact_or_stall": ((current.get("evidence") or {}).get("visual_route") or {}).get("progress", {}).get("contact_or_stall"),
        "bundle": bundle, "panel_meta": panel_meta,
    }, panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--images-only", action="store_true")
    args = parser.parse_args()
    cases = [prepare_case(path.resolve()) for path in args.run_dirs]
    results = []
    if not args.images_only:
        import requests
        try:
            response = requests.get(f"{args.base_url}/models", timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise SystemExit(f"Qwen endpoint unavailable: {exc}")
        client = VLMClient(args.base_url, args.model, "EMPTY", 90, 256, 0.0)
        agent = ControllerAgent(client, "", "")
    for case, panel in cases:
        row = {key: case[key] for key in ("run_dir", "frame_id", "contact_candidate", "contact_or_stall", "panel_meta")}
        if not args.images_only:
            frame = case["frame_id"]
            with Image.open(Path(case["run_dir"]) / f"images/raw_agentview/{frame:04d}.png") as source:
                agentview = np.asarray(source.convert("RGB"))
            with Image.open(Path(case["run_dir"]) / f"images/raw_wrist/{frame:04d}.png") as source:
                wrist = np.asarray(source.convert("RGB"))
            for mode in ("legacy", "off", "single", "double"):
                row[mode] = agent.verify_place(
                    task=case["task"], target=case["target"], affordance=case["affordance"],
                    agentview_image=agentview, wrist_image=wrist, placement_v22=True,
                    placement_evidence=case["placement_evidence"], memory_panel=panel,
                    memory_bundle=case["bundle"], reflection_mode=mode,
                    reflection_trigger="contact_risk" if case["contact_candidate"] else "seating_check",
                )
        results.append(row)
    report = {"cases": results}
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output + "\n")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

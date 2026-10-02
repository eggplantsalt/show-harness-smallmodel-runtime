#!/usr/bin/env python3
"""Replay a saved Show-Harness episode through VCR-v2 without simulator access."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime_v2 import VerifiedCapabilityRuntime  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _image(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return np.asarray(Image.open(path).convert("RGB"))


def replay(run_dir: Path) -> dict[str, Any]:
    steps_path = run_dir / "steps.jsonl"
    if not steps_path.is_file():
        raise FileNotFoundError(steps_path)
    runtime = VerifiedCapabilityRuntime(mode="shadow")
    rows = [json.loads(line) for line in steps_path.read_text().splitlines() if line.strip()]
    events = []
    identity_switches = []
    sensor_fault_actions = []
    previous_runtime_id = None
    previous_recorded_action = None
    for row in rows:
        index = int(row.get("i", row.get("step_idx", len(events))))
        capability = row.get("capability")
        capability = dict(capability) if isinstance(capability, dict) else {}
        capability.pop("verified_runtime", None)
        stage = str(row.get("stage") or capability.get("stage") or "")
        eef = row.get("fingertip") or row.get("eef")
        eef_xyz = None
        if isinstance(eef, (list, tuple)) and len(eef) >= 3:
            eef_xyz = tuple(float(value) for value in eef[:3])
        result = runtime.observe_frame(
            stage=stage,
            evidence=capability,
            previous_action=previous_recorded_action,
            agentview=_image(run_dir / "images" / "agentview" / f"{index:04d}.png"),
            wrist=_image(run_dir / "images" / "wrist" / f"{index:04d}.png"),
            eef_xyz=eef_xyz,
            gripper_closed=str(row.get("grip") or "").upper() == "CLOSE",
            gripper_width_m=row.get("w"),
        )
        # Reuse only the recorded visual verifier verdict (never simulator
        # truth) so a historical post-GRASP YES can seed the same semantic
        # transition that the live V2 critical-decision path would commit.
        grasp_verification = row.get("grasp_verification")
        if isinstance(grasp_verification, dict):
            verdict = str(grasp_verification.get("decision") or "UNKNOWN")
            runtime.apply_critical_decision(verdict)
        target = result.get("belief", {}).get("target") or {}
        runtime_id = target.get("instance_id")
        if previous_runtime_id and runtime_id and runtime_id != previous_runtime_id:
            identity_switches.append(index)
        if runtime_id:
            previous_runtime_id = runtime_id
        if (
            result.get("observation_health") == "SENSOR_FAULT"
            and result.get("action_token") not in {"STOP", None}
        ):
            sensor_fault_actions.append(index)
        events.append(
            {
                "step": index,
                "stage": stage,
                "recorded_action": row.get("act"),
                "runtime_action": result.get("action_token"),
                "health": result.get("observation_health"),
                "option": result.get("option"),
                "target_instance_id": runtime_id,
                "target_bbox_xyxy": target.get("bbox_xyxy"),
                "failure": result.get("failure"),
            }
        )
        previous_recorded_action = row.get("act")
    return {
        "runtime_version": runtime.runtime_version,
        "source": str(run_dir),
        "steps": len(events),
        "identity_switch_steps": identity_switches,
        "sensor_fault_action_steps": sensor_fault_actions,
        "events": events,
    }


def main() -> int:
    args = parse_args()
    report = replay(args.run_dir.resolve())
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

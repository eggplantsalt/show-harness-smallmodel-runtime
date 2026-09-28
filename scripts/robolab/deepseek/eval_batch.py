#!/usr/bin/env python3
"""Reproducible serial zero-shot evaluation for the SiliconFlow RoboLab setup.

Every expected attempt is written to manifest.json before its subprocess starts.
The denominator therefore includes API failures, simulator crashes, and missing
summaries; no stdout parsing or automatic retry can turn a failure into success.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TASKS = ("GrabAFruitTask",)


def wilson(successes: int, total: int) -> tuple[float, float]:
    if not total:
        return 0.0, 0.0
    p, z = successes / total, 1.959963984540054
    d = 1 + z * z / total
    c = (p + z * z / (2 * total)) / d
    r = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / d
    return max(0.0, c - r), min(1.0, c + r)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3, help="Attempts per task; must be positive")
    parser.add_argument("--seed", type=int, default=0, help="First independent process seed")
    parser.add_argument("--task", action="append", dest="tasks", help="Repeat for each RoboLab task")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--output", type=Path, default=ROOT / "results/robolab_siliconflow")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed process")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("--episodes and --max-steps must be positive")
    tasks = tuple(args.tasks or DEFAULT_TASKS)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output / f"batch_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "manifest.json"
    attempts = [
        {"task": task, "episode_index": index, "seed": args.seed + index,
         "status": "pending", "summary": None, "returncode": None}
        for task in tasks for index in range(args.episodes)
    ]
    manifest = {"protocol": "one fresh Isaac process per task/seed; failures remain in denominator",
                "created_utc": stamp, "tasks": list(tasks), "episodes_per_task": args.episodes,
                "max_steps": args.max_steps, "attempts": attempts}

    def save() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    save()
    overall_ok = True
    for attempt in attempts:
        task, idx, seed = attempt["task"], attempt["episode_index"], attempt["seed"]
        log_root = run_dir / "rollouts" / task / f"episode_{idx:03d}_seed_{seed}"
        logfile = run_dir / "logs" / task / f"episode_{idx:03d}_seed_{seed}.log"
        logfile.parent.mkdir(parents=True, exist_ok=True)
        command = [
            os.environ.get("ROBO_PYTHON", sys.executable), "-u", "scripts/run_robolab_zeroshot.py",
            "--robot-config", "configs/robot_robolab_deepseek.yaml", "--task", task,
            "--seed", str(seed), "--episode-index", str(idx), "--episodes", "1",
            "--max-steps", str(args.max_steps), "--log-dir", str(log_root),
            "--prompt-log-every", "1",
        ]
        attempt.update(status="running", command=command, log=str(logfile.relative_to(run_dir)), started_utc=datetime.now(timezone.utc).isoformat())
        save()
        with logfile.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
        attempt.update(returncode=completed.returncode, ended_utc=datetime.now(timezone.utc).isoformat())
        summaries = sorted(log_root.rglob("summary.json"))
        if len(summaries) == 1:
            try:
                summary = json.loads(summaries[0].read_text(encoding="utf-8"))
                success = summary.get("success")
                if not isinstance(success, bool):
                    raise ValueError("summary success is not boolean")
                attempt.update(status="success" if success else "task_failed", success=success,
                               end_reason=summary.get("end_reason"), summary=str(summaries[0].relative_to(run_dir)))
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                attempt.update(status="invalid_summary", error=str(exc))
        elif not summaries:
            attempt.update(status="missing_summary", error="runner produced no summary.json")
        else:
            attempt.update(status="ambiguous_summary", error=f"found {len(summaries)} summaries")
        if attempt["status"] != "success":
            overall_ok = False
        save()
        if not args.keep_going and attempt["status"] in {"missing_summary", "invalid_summary", "ambiguous_summary"}:
            break

    rows = []
    for task in tasks:
        subset = [a for a in attempts if a["task"] == task]
        n, k = len(subset), sum(a.get("success") is True for a in subset)
        lo, hi = wilson(k, n)
        rows.append({"task": task, "successes": k, "expected": n, "rate": k / n if n else 0,
                     "wilson95_low": lo, "wilson95_high": hi,
                     "non_success": n-k})
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["task"])
        writer.writeheader(); writer.writerows(rows)
    (run_dir / "summary.json").write_text(json.dumps({"per_task": rows, "manifest": "manifest.json"}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {run_dir / 'summary.json'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

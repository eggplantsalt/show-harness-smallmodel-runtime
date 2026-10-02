#!/usr/bin/env python3
"""Run the parallel LIBERO Verified Capability Runtime v2 profile."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "robot_libero_clean_qwen3vl_runtime_v2.yaml"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    if "--robot-config" not in sys.argv:
        sys.argv[1:1] = ["--robot-config", str(DEFAULT_CONFIG)]
    from scripts.run_libero_zeroshot import main as run

    return run()


if __name__ == "__main__":
    raise SystemExit(main())

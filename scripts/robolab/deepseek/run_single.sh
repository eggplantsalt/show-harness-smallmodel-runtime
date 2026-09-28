#!/usr/bin/env bash
# One task, with all supplied runner arguments forwarded verbatim.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common_env.sh"
require_robolab_runtime
cd "$SHOW_HARNESS_ROOT"
exec "$ROBO_PYTHON" -u scripts/run_robolab_zeroshot.py \
  --robot-config "$SHOW_HARNESS_ROOT/configs/robot_robolab_deepseek.yaml" "$@"

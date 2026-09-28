#!/usr/bin/env bash
# Run from any directory. The Python driver records every expected episode first.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common_env.sh"
require_robolab_runtime
cd "$SHOW_HARNESS_ROOT"
exec "$ROBO_PYTHON" -u scripts/robolab/deepseek/eval_batch.py "$@"

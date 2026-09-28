#!/usr/bin/env bash
# Explicit installation only: does not start Isaac Sim, call an API, or run experiments.
set -euo pipefail

SHOWHARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ROBOLAB_ROOT="${ROBOLAB_ROOT:-$(dirname "$SHOWHARNESS_ROOT")/RoboLab}"
ISAAC_STACK="${ISAAC_STACK:-isaac50}"

case "${1:-}" in
  --help|-h)
    cat <<'EOF'
Usage: bash scripts/robolab/deepseek/setup_env.sh [--extras-only]

No arguments: uv sync --frozen the RoboLab stack, then add harness client dependencies.
--extras-only: preserve an existing RoboLab environment; only add client dependencies.
Environment: ROBOLAB_ROOT, ISAAC_STACK=isaac50|isaac51, ROBO_PYTHON (extras-only).
This script never launches a simulation or sends an API request.
EOF
    exit 0 ;;
  ""|--extras-only) ;;
  *) echo "Unknown option: $1. Use --help." >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo "Too many arguments." >&2; exit 2; }
[[ "$ISAAC_STACK" == isaac50 || "$ISAAC_STACK" == isaac51 ]] || {
  echo "ISAAC_STACK must be isaac50 or isaac51." >&2; exit 2;
}
[[ -f "$ROBOLAB_ROOT/pyproject.toml" ]] || {
  echo "Missing RoboLab checkout: $ROBOLAB_ROOT. See docs/robolab_deepseek.zh-CN.md." >&2
  exit 1
}
command -v uv >/dev/null || {
  echo "uv is missing. Install it using the documentation, then repeat this command." >&2
  exit 1
}

# User's AutoDL rule: disable proxies before every pip / uv dependency install.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ "${1:-}" != --extras-only ]]; then
  [[ -f "$ROBOLAB_ROOT/uv.lock" ]] || {
    echo "Missing uv.lock: a reproducible Isaac installation needs RoboLab's lockfile." >&2
    exit 1
  }
  # Use only one of the mutually exclusive Isaac extras. uv provisions Python 3.11.
  (cd "$ROBOLAB_ROOT" && UV_PROJECT_ENVIRONMENT="$ROBOLAB_ROOT/.venv" \
    uv sync --frozen --python 3.11 --extra "$ISAAC_STACK")
  ROBO_PYTHON="$ROBOLAB_ROOT/.venv/bin/python"
else
  ROBO_PYTHON="${ROBO_PYTHON:-$ROBOLAB_ROOT/.venv/bin/python}"
fi
[[ -x "$ROBO_PYTHON" ]] || { echo "Missing Python interpreter: $ROBO_PYTHON" >&2; exit 1; }
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
uv pip install --python "$ROBO_PYTHON" \
  -r "$SHOWHARNESS_ROOT/requirements/requirements-robolab-client.txt"
printf 'Environment prepared: %s\nNext: docs/robolab_deepseek.zh-CN.md (preflight).\n' "$ROBO_PYTHON"

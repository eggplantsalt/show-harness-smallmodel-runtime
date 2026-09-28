#!/usr/bin/env bash
set -euo pipefail

# The existing LIBERO venv is owned by OpenETA; Show-Harness itself is imported
# directly from this checkout.  No dataset download or pip install is performed.
show_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
libero_root="${LIBERO_DIR:-$(cd "$show_root/../OpenETA/vendor/LIBERO" && pwd)}"
libero_python="${LIBERO_PYTHON:-$show_root/../OpenETA/sim/venvs/libero/bin/python}"

export LIBERO_DIR="$libero_root"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="$show_root:$libero_root${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$libero_python" ]]; then
  echo "LIBERO interpreter not found: $libero_python" >&2
  echo "Set LIBERO_PYTHON to the existing LIBERO venv's python." >&2
  exit 2
fi

exec "$libero_python" "$show_root/scripts/run_libero_zeroshot.py" "$@"

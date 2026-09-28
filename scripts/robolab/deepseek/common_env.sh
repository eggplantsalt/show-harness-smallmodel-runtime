#!/usr/bin/env bash
# Source this file. It configures paths/network variables; it never installs or runs Isaac.
# Keep credentials out of xtrace even if a caller enabled `bash -x`.
set +x

SHOW_HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export SHOW_HARNESS_ROOT
export ROBOLAB_ROOT="${ROBOLAB_ROOT:-$(dirname "$SHOW_HARNESS_ROOT")/RoboLab}"
export ROBO_PYTHON="${ROBO_PYTHON:-$ROBOLAB_ROOT/.venv/bin/python}"
export SILICONFLOW_ENV_FILE="${SILICONFLOW_ENV_FILE:-${HOME}/.config/show-harness/siliconflow.env}"

if [[ -f "$SILICONFLOW_ENV_FILE" ]]; then
  # This is a user-owned shell env file: put only trusted KEY=value declarations here.
  # Export declarations without printing their values, and preserve the caller's -a.
  _show_harness_allexport=0
  [[ $- == *a* ]] && _show_harness_allexport=1
  set -a
  source "$SILICONFLOW_ENV_FILE"
  if [[ "$_show_harness_allexport" == 0 ]]; then
    set +a
  fi
  unset _show_harness_allexport
fi

# Existing local Show-Harness deployments commonly keep credentials in this ignored
# file. Accept it as a migration fallback so preflight and run scripts resolve the
# same key source; a user-owned SILICONFLOW_ENV_FILE always takes precedence.
if [[ -z "${SILICONFLOW_API_KEY:-}" && -f "$SHOW_HARNESS_ROOT/configs/secrets.env" ]]; then
  _show_harness_allexport=0
  [[ $- == *a* ]] && _show_harness_allexport=1
  set -a
  source "$SHOW_HARNESS_ROOT/configs/secrets.env"
  if [[ "$_show_harness_allexport" == 0 ]]; then
    set +a
  fi
  unset _show_harness_allexport
fi

# SiliconFlow and pip use direct connections. HF uses the mirror first.
# /etc/network_turbo belongs ONLY around git clone, not this runtime environment.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="$SHOW_HARNESS_ROOT:$ROBOLAB_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

require_robolab_runtime() {
  if [[ ! -d "$ROBOLAB_ROOT/robolab" ]]; then
    printf 'RoboLab checkout missing: %s/robolab; set ROBOLAB_ROOT.\n' "$ROBOLAB_ROOT" >&2
    return 1
  fi
  if [[ ! -x "$ROBO_PYTHON" ]]; then
    printf 'RoboLab Python missing: %s; finish setup or set ROBO_PYTHON.\n' "$ROBO_PYTHON" >&2
    return 1
  fi
  if [[ -z "${SILICONFLOW_API_KEY:-}" ]]; then
    printf 'SILICONFLOW_API_KEY is unset; configure %s without putting the key in the repository.\n' "$SILICONFLOW_ENV_FILE" >&2
    return 1
  fi
  if [[ "${OMNI_KIT_ACCEPT_EULA:-}" != YES ]]; then
    printf 'Read NVIDIA Isaac Sim license terms, then explicitly export OMNI_KIT_ACCEPT_EULA=YES before simulation.\n' >&2
    return 1
  fi
}

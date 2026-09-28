#!/usr/bin/env bash
set -euo pipefail

# Clean no-LoRA Qwen3-VL-8B server for the LIBERO research profile.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}" \
PORT="${PORT:-8001}" \
GPU_UTIL="${GPU_UTIL:-0.78}" \
MAX_LEN="${MAX_LEN:-4096}" \
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}" \
TEMPERATURE="${TEMPERATURE:-0}" \
ENFORCE_EAGER="${ENFORCE_EAGER:-1}" \
bash "${ROOT}/scripts/serve_vlm.sh"

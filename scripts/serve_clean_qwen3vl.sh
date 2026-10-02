#!/usr/bin/env bash
set -euo pipefail

# Clean no-LoRA Qwen3-VL-8B server for the LIBERO research profile.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
if [[ "${MODEL}" == "Qwen/Qwen3-VL-8B-Thinking" ]]; then
  # vLLM's per-request thinking_token_budget is currently supported by the V1 runner.
  export VLLM_USE_V2_MODEL_RUNNER=0
  DEFAULT_TEMPERATURE=1.0
  DEFAULT_TOP_P=0.95
  DEFAULT_TOP_K=20
  # Qwen3-VL Thinking emits an assistant `<think>` prefix from its template.
  # vLLM's qwen3 parser does not split this checkpoint's channel reliably;
  # deepseek_r1 recognizes the same `<think>...</think>` boundary.
  DEFAULT_REASONING_PARSER=deepseek_r1
  DEFAULT_MAX_LEN=16384
else
  DEFAULT_TEMPERATURE=0
  DEFAULT_TOP_P=1.0
  DEFAULT_TOP_K=-1
  DEFAULT_REASONING_PARSER=""
  DEFAULT_MAX_LEN=4096
fi
MODEL="${MODEL}" \
GPU="${GPU:-0}" \
PORT="${PORT:-8001}" \
GPU_UTIL="${GPU_UTIL:-0.78}" \
MAX_LEN="${MAX_LEN:-${DEFAULT_MAX_LEN}}" \
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}" \
TEMPERATURE="${TEMPERATURE:-${DEFAULT_TEMPERATURE}}" \
TOP_P="${TOP_P:-${DEFAULT_TOP_P}}" \
TOP_K="${TOP_K:-${DEFAULT_TOP_K}}" \
REASONING_PARSER="${REASONING_PARSER-${DEFAULT_REASONING_PARSER}}" \
ENFORCE_EAGER="${ENFORCE_EAGER:-1}" \
bash "${ROOT}/scripts/serve_vlm.sh"

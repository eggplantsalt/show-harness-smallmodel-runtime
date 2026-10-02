#!/usr/bin/env bash
set -euo pipefail

# Official Hugging Face model only; the mirror follows workspace AGENTS.md.
# The local hf CLI currently follows this repository's Xet redirect to a path it
# cannot resume reliably.  Use the mirror's normal resolve endpoint instead; curl
# preserves partial shards with HTTP Range and keeps the exact Hub revision below.
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
case "${MODEL}" in
  Qwen/Qwen3-VL-8B-Instruct)
    DEFAULT_REVISION="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    ;;
  Qwen/Qwen3-VL-8B-Thinking)
    DEFAULT_REVISION="92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b"
    ;;
  *)
    echo "ERROR: supported clean models are Qwen/Qwen3-VL-8B-Instruct and Qwen/Qwen3-VL-8B-Thinking; got ${MODEL}" >&2
    exit 2
    ;;
esac
REVISION="${MODEL_REVISION:-${DEFAULT_REVISION}}"
MODEL_DIR="${MODEL_DIR:-${HF_HOME}/hub/${MODEL}}"

mkdir -p "${MODEL_DIR}"
files=(
  .gitattributes
  README.md
  chat_template.json
  config.json
  generation_config.json
  merges.txt
  model-00001-of-00004.safetensors
  model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors
  model-00004-of-00004.safetensors
  model.safetensors.index.json
  preprocessor_config.json
  tokenizer.json
  tokenizer_config.json
  video_preprocessor_config.json
  vocab.json
)

for file in "${files[@]}"; do
  dest="${MODEL_DIR}/${file}"
  url="${HF_ENDPOINT}/${MODEL}/resolve/${REVISION}/${file}"
  if [[ -f "${dest}" ]]; then
    case "${file}" in
      model-00001-of-00004.safetensors) remote_size=4902275944 ;;
      model-00002-of-00004.safetensors) remote_size=4915962496 ;;
      model-00003-of-00004.safetensors) remote_size=4999831048 ;;
      model-00004-of-00004.safetensors) remote_size=2716270024 ;;
      *)
        remote_size="$(curl --fail --location --silent --show-error --head \
          --connect-timeout 30 "${url}" \
          | awk 'BEGIN {IGNORECASE=1} /^content-length:/ {gsub(/[^0-9]/, "", $2); n=$2} END {print n+0}')"
        ;;
    esac
    local_size="$(stat -c '%s' "${dest}")"
    if [[ "${remote_size}" != "0" && "${local_size}" == "${remote_size}" ]]; then
      echo "[hf-mirror] ${file} already complete (${local_size} bytes)"
      continue
    fi
  fi
  echo "[hf-mirror] ${file}"
  curl --fail --location --retry 5 --retry-all-errors --retry-delay 5 \
    --connect-timeout 30 --speed-limit 1024 --speed-time 120 \
    -C - -o "${dest}" "${url}"
done

echo "Downloaded ${MODEL} revision ${REVISION} to ${MODEL_DIR}"

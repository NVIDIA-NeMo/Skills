#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:/nemo_run/code"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd /nemo_run/code

parser_args=()
if [[ -n "${REASONING_PARSER_NAME:-qwen3}" ]]; then
  parser_args+=(--reasoning-parser "${REASONING_PARSER_NAME:-qwen3}")
fi

python3 -m nemo_skills.inference.server.serve_vllm \
  --model "${MODEL}" \
  --num_gpus 1 \
  --num_nodes 1 \
  --port "${PORT}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_CALL_PARSER:-qwen3_coder}" \
  --max-model-len "${MAX_MODEL_LEN:-65536}" \
  --load-format safetensors \
  "${parser_args[@]}"

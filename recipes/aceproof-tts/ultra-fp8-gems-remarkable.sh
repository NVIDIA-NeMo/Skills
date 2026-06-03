#!/bin/bash
set -euo pipefail

base_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$base_dir"

PYTHON_BIN="${PYTHON_BIN:-$base_dir/.venv/bin/python}"

"$PYTHON_BIN" recipes/aceproof-tts/pipeline/run_pipeline.py \
  --config recipes/aceproof-tts/configs/aceproof-tts-gems-remarkable-ultra-fp8.yaml \
  "$@"

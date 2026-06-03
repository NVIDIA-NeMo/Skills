#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON:-$repo_root/.venv/bin/python}"

cd "$repo_root"
exec "$python_bin" recipes/aceproof-tts/pipeline/run_pipeline.py \
  --config recipes/aceproof-tts/configs/aceproof-tts-proofbench133-ultra-fp8-r1.yaml \
  "$@"

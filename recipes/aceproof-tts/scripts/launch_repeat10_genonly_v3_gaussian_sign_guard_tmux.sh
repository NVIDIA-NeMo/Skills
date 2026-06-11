#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/igitman/workspace/NeMo-Skills-aceproof-mp
PY=/home/igitman/workspace/NeMo-Skills/.venv/bin/python
SESSION=repeat10-genonly-v3-gaussian-sign-verify
INPUT=outputs/repeat10-genonly-candidate-verification/current_verify_v3_x8.jsonl
OUT=outputs/repeat10-genonly-candidate-verification/v3_gaussian_sign_guard_x8
LOG=outputs/repeat10-genonly-candidate-verification/v3-gaussian-sign-guard-x8.log

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  exit 1
fi

cmd="cd '$ROOT'; export OPENAI_API_KEY=\"\${OPENAI_API_KEY:-dummy}\"; export NEMO_SKILLS_OPENAI_AIOHTTP=\"\${NEMO_SKILLS_OPENAI_AIOHTTP:-1}\"; export NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT=\"\${NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT:-65536}\"; export PYTHONUNBUFFERED=1; export PATH='$(dirname "$PY")':\$PATH; mkdir -p outputs/repeat10-genonly-candidate-verification; '$PY' recipes/aceproof-tts/pipeline/run_verify_sidecar.py --input_file '$INPUT' --output_dir '$OUT' --prompt_config_path recipes/aceproof-tts/prompts/proof_verification_gaussian_sign_branch_guard.yaml --expname repeat10_genonly_v3_gaussian_sign_verify --max_concurrent_requests 256 2>&1 | tee '$LOG'; echo '[done]' \$?; sleep infinity"
tmux new-session -d -s "$SESSION" -n gaussian-sign "$cmd"
tmux set-option -t "$SESSION" remain-on-exit on >/dev/null
echo "Launched tmux session: $SESSION"
echo "Output: $OUT"

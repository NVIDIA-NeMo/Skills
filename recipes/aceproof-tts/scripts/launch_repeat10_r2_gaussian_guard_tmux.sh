#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/igitman/workspace/NeMo-Skills-aceproof-mp
PY=/home/igitman/workspace/NeMo-Skills/.venv/bin/python
SESSION=repeat10-r2-gaussian-guard-verify
LOG_DIR="$ROOT/outputs/repeat10-r2-candidate-verification"
LOG="$LOG_DIR/gaussian-guard-x16.log"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  exit 1
fi
mkdir -p "$LOG_DIR"
cmd="cd '$ROOT'; export OPENAI_API_KEY=\"\${OPENAI_API_KEY:-dummy}\"; export NEMO_SKILLS_OPENAI_AIOHTTP=\"\${NEMO_SKILLS_OPENAI_AIOHTTP:-1}\"; export NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT=\"\${NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT:-65536}\"; export PYTHONUNBUFFERED=1; export PATH='$(dirname "$PY")':\$PATH; '$PY' recipes/aceproof-tts/pipeline/run_verify_sidecar.py --input_file outputs/repeat10-r2-candidate-verification/current_verify_x16.jsonl --output_dir outputs/repeat10-r2-candidate-verification/gaussian_guard_x16 --prompt_config_path recipes/aceproof-tts/prompts/proof_verification_gaussian_unit_guard.yaml --expname repeat10_r2_gaussian_guard_verify --max_concurrent_requests 256 2>&1 | tee '$LOG'; echo '[done]' \$?; sleep infinity"
tmux new-session -d -s "$SESSION" -n gaussian-guard "$cmd"
tmux set-option -t "$SESSION" remain-on-exit on >/dev/null
echo "Launched tmux session: $SESSION"
echo "Log: $LOG"

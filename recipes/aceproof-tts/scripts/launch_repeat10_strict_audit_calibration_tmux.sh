#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/igitman/workspace/NeMo-Skills-aceproof-mp
PY=/home/igitman/workspace/NeMo-Skills/.venv/bin/python
SESSION=repeat10-strict-audit-calibration
LOG_DIR="$ROOT/outputs/repeat10-strict-audit-calibration"
LOG="$LOG_DIR/strict-audit-calibration.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 1
fi

mkdir -p "$LOG_DIR"
cmd="cd '$ROOT'; export OPENAI_API_KEY=\"\${OPENAI_API_KEY:-dummy}\"; export NEMO_SKILLS_OPENAI_AIOHTTP=\"\${NEMO_SKILLS_OPENAI_AIOHTTP:-1}\"; export NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT=\"\${NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT:-65536}\"; export PYTHONUNBUFFERED=1; export PATH='$(dirname "$PY")':\$PATH; '$PY' recipes/aceproof-tts/pipeline/run_verify_sidecar.py --input_file outputs/repeat10-strict-audit-calibration/input_verify_x8.jsonl --output_dir outputs/repeat10-strict-audit-calibration/strict_audit_x8 --expname repeat10_strict_audit_calibration 2>&1 | tee '$LOG'; echo '[done]' \$?; sleep infinity"

tmux new-session -d -s "$SESSION" -n strict-audit "$cmd"
tmux set-option -t "$SESSION" remain-on-exit on >/dev/null

echo "Launched tmux session: $SESSION"
echo "Log: $LOG"
echo "Attach: tmux attach -t $SESSION"

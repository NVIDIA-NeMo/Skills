#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/igitman/workspace/NeMo-Skills-aceproof-mp
PY=/home/igitman/workspace/NeMo-Skills/.venv/bin/python
CONFIG_DIR="$ROOT/recipes/aceproof-tts/configs/repeat10-mp-remaining6-r2"
LOG_DIR="$ROOT/outputs/repeat10-ultra-mp-remaining6-r2-launch-logs"
SESSION="repeat10-mp-remaining6-r2"
SKIP_NAME="${SKIP_NAME:-repeat10-ultra-mp-remaining6-r2-temp10-claimcert}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 1
fi

mkdir -p "$LOG_DIR"
cd "$ROOT"
: > "$LOG_DIR/tmux-windows.txt"
first=1
for cfg in $(find "$CONFIG_DIR" -maxdepth 1 -name '*.yaml' | sort); do
  name=$(basename "$cfg" .yaml)
  if [[ "$name" == "$SKIP_NAME" ]]; then
    echo "Skipping active diagnostic config: $name"
    continue
  fi
  short=${name#repeat10-ultra-mp-remaining6-r2-}
  log="$LOG_DIR/$name.log"
  cmd="cd '$ROOT'; export OPENAI_API_KEY=\"\${OPENAI_API_KEY:-dummy}\"; export NEMO_SKILLS_OPENAI_AIOHTTP=\"\${NEMO_SKILLS_OPENAI_AIOHTTP:-1}\"; export NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT=\"\${NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT:-65536}\"; export PYTHONUNBUFFERED=1; export PATH='$(dirname "$PY")':\$PATH; '$PY' recipes/aceproof-tts/pipeline/run_streaming.py --config '$cfg' 2>&1 | tee '$log'; echo '[done]' \$?; sleep infinity"
  if [[ "$first" == 1 ]]; then
    tmux new-session -d -s "$SESSION" -n "$short" "$cmd"
    first=0
  else
    tmux new-window -t "$SESSION" -n "$short" "$cmd"
  fi
  echo "$short $cfg $log" >> "$LOG_DIR/tmux-windows.txt"
done

tmux set-option -t "$SESSION" remain-on-exit on >/dev/null

echo "Launched tmux session: $SESSION"
echo "Windows: $LOG_DIR/tmux-windows.txt"
echo "Attach: tmux attach -t $SESSION"

#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/igitman/workspace/NeMo-Skills-aceproof-mp
PY=/home/igitman/workspace/NeMo-Skills/.venv/bin/python
CONFIG_DIR="$ROOT/recipes/aceproof-tts/configs/repeat10-mp-remaining6-r2"
LOG_DIR="$ROOT/outputs/repeat10-ultra-mp-remaining6-r2-launch-logs"

if [[ "${1:-}" != "--run" ]]; then
  echo "Prepared launch for repeat10 remaining6 R2 batch."
  echo "This script is guarded and has not contacted the endpoint."
  echo "To launch all configs: $0 --run"
  echo
  echo "Configs:"
  find "$CONFIG_DIR" -maxdepth 1 -name '*.yaml' | sort
  exit 0
fi

mkdir -p "$LOG_DIR"
cd "$ROOT"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export NEMO_SKILLS_OPENAI_AIOHTTP="${NEMO_SKILLS_OPENAI_AIOHTTP:-1}"
export NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT="${NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT:-65536}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PATH="$(dirname "$PY"):$PATH"
: > "$LOG_DIR/pids.txt"

for cfg in $(find "$CONFIG_DIR" -maxdepth 1 -name '*.yaml' | sort); do
  name=$(basename "$cfg" .yaml)
  log="$LOG_DIR/$name.log"
  if [[ -s "$log" ]] && grep -q "Submitted streaming driver" "$log" 2>/dev/null; then
    echo "Skipping already submitted $name -> $log"
    continue
  fi
  echo "Launching $name -> $log"
  nohup "$PY" recipes/aceproof-tts/pipeline/run_streaming.py --config "$cfg" > "$log" 2>&1 &
  pid=$!
  echo "$pid $name $cfg $log" >> "$LOG_DIR/pids.txt"
done

echo "Launched $(wc -l < "$LOG_DIR/pids.txt") local drivers."
echo "PID file: $LOG_DIR/pids.txt"
echo "Logs: $LOG_DIR"

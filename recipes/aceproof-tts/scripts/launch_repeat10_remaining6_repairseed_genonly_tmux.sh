#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/igitman/workspace/NeMo-Skills-aceproof-mp
PY=/home/igitman/workspace/NeMo-Skills/.venv/bin/python
SESSION=repeat10-remaining6-repairseed-genonly
INPUT=recipes/aceproof-tts/dataset/repeat10-remaining6-repairseeds-20260610.jsonl
PROMPT=recipes/aceproof-tts/prompts/proof_generation_repair_seed.yaml
OUT_ROOT="$ROOT/outputs/repeat10-ultra-mp-remaining6-repairseed-genonly"
LOG_ROOT="$ROOT/outputs/repeat10-ultra-mp-remaining6-repairseed-genonly-launch-logs"
N_PARALLEL="${N_PARALLEL:-128}"
MAX_CONCURRENT="${MAX_CONCURRENT:-512}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 1
fi

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

make_cmd() {
  local temp_label="$1"
  local temperature="$2"
  local out="$OUT_ROOT/$temp_label/repairseed"
  local log="$LOG_ROOT/$temp_label-repairseed.log"
  cat <<CMD
cd '$ROOT'
export OPENAI_API_KEY="\${OPENAI_API_KEY:-dummy}"
export NEMO_SKILLS_OPENAI_AIOHTTP="\${NEMO_SKILLS_OPENAI_AIOHTTP:-1}"
export NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT="\${NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT:-65536}"
export PYTHONUNBUFFERED=1
export PATH='$(dirname "$PY")':\$PATH
mkdir -p '$out' '$LOG_ROOT'
'$PY' recipes/aceproof-tts/pipeline/prepare_round1.py --input_paths '$INPUT' --output_dir '$out' --n_parallel_proof_gen '$N_PARALLEL' --prompt_config_path '$PROMPT' --interleave_rows
'$PY' recipes/aceproof-tts/pipeline/run_proof_gen_sidecar.py --input_file '$out/rounds/R1/proof_gen/input.jsonl' --output_dir '$out/rounds/R1/proof_gen' --prompt_config_path '$PROMPT' --expname repeat10_repairseed_${temp_label} --temperature '$temperature' --max_concurrent_requests '$MAX_CONCURRENT' 2>&1 | tee '$log'
echo '[done]' \$?
sleep infinity
CMD
}

tmux new-session -d -s "$SESSION" -n temp10 "$(make_cmd temp10 1.0)"
tmux new-window -t "$SESSION" -n temp08 "$(make_cmd temp08 0.8)"
tmux set-option -t "$SESSION" remain-on-exit on >/dev/null

echo "Launched tmux session: $SESSION"
echo "Output root: $OUT_ROOT"

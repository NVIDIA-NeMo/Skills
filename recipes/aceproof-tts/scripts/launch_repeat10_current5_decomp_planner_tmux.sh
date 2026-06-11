#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/igitman/workspace/NeMo-Skills-aceproof-mp
PY=/home/igitman/workspace/NeMo-Skills/.venv/bin/python
SESSION=repeat10-current5-decomp-planner
INPUT=recipes/aceproof-tts/dataset/repeat10-newckpt-current-unsolved5-20260610.jsonl
PROMPT=recipes/aceproof-tts/prompts/proof_generation_decomposition_planner.yaml
OUT_ROOT="$ROOT/outputs/repeat10-ultra-mp-current5-decomp-planner"
LOG_ROOT="$ROOT/outputs/repeat10-ultra-mp-current5-decomp-planner-launch-logs"
N_PARALLEL="${N_PARALLEL:-32}"
MAX_CONCURRENT="${MAX_CONCURRENT:-256}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 1
fi

mkdir -p "$OUT_ROOT/temp10/planner" "$LOG_ROOT"

cmd=$(cat <<CMD
cd '$ROOT'
export OPENAI_API_KEY="\${OPENAI_API_KEY:-dummy}"
export NEMO_SKILLS_OPENAI_AIOHTTP="\${NEMO_SKILLS_OPENAI_AIOHTTP:-1}"
export NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT="\${NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT:-65536}"
export PYTHONUNBUFFERED=1
export PATH='$(dirname "$PY")':\$PATH
'$PY' recipes/aceproof-tts/pipeline/prepare_round1.py --input_paths '$INPUT' --output_dir '$OUT_ROOT/temp10/planner' --n_parallel_proof_gen '$N_PARALLEL' --prompt_config_path '$PROMPT' --interleave_rows
'$PY' recipes/aceproof-tts/pipeline/run_proof_gen_sidecar.py --input_file '$OUT_ROOT/temp10/planner/rounds/R1/proof_gen/input.jsonl' --output_dir '$OUT_ROOT/temp10/planner/rounds/R1/proof_gen' --prompt_config_path '$PROMPT' --expname repeat10_current5_decomp_planner_temp10 --temperature 1.0 --max_concurrent_requests '$MAX_CONCURRENT' 2>&1 | tee '$LOG_ROOT/temp10-planner.log'
echo '[done]' \$?
sleep infinity
CMD
)

tmux new-session -d -s "$SESSION" -n planner "$cmd"
tmux set-option -t "$SESSION" remain-on-exit on >/dev/null

echo "Launched tmux session: $SESSION"
echo "Output root: $OUT_ROOT"

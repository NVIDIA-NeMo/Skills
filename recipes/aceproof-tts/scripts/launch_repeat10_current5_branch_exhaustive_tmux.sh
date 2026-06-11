#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/igitman/workspace/NeMo-Skills-aceproof-mp
PY=/home/igitman/workspace/NeMo-Skills/.venv/bin/python
SESSION=repeat10-current5-branch-exhaustive
INPUT=recipes/aceproof-tts/dataset/repeat10-newckpt-current-unsolved5-20260610.jsonl
PROMPT=recipes/aceproof-tts/prompts/proof_generation_branch_exhaustive.yaml
OUT="$ROOT/outputs/repeat10-ultra-mp-current5-branch-exhaustive/temp10/branch"
LOG="$ROOT/outputs/repeat10-ultra-mp-current5-branch-exhaustive-launch-logs/temp10-branch.log"
N_PARALLEL="${N_PARALLEL:-128}"
MAX_CONCURRENT="${MAX_CONCURRENT:-512}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 1
fi

cmd="cd '$ROOT'; export OPENAI_API_KEY=\"\${OPENAI_API_KEY:-dummy}\"; export NEMO_SKILLS_OPENAI_AIOHTTP=\"\${NEMO_SKILLS_OPENAI_AIOHTTP:-1}\"; export NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT=\"\${NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT:-65536}\"; export PYTHONUNBUFFERED=1; export PATH='$(dirname "$PY")':\$PATH; mkdir -p '$OUT' '$(dirname "$LOG")'; '$PY' recipes/aceproof-tts/pipeline/prepare_round1.py --input_paths '$INPUT' --output_dir '$OUT' --n_parallel_proof_gen '$N_PARALLEL' --prompt_config_path '$PROMPT' --interleave_rows; '$PY' recipes/aceproof-tts/pipeline/run_proof_gen_sidecar.py --input_file '$OUT/rounds/R1/proof_gen/input.jsonl' --output_dir '$OUT/rounds/R1/proof_gen' --prompt_config_path '$PROMPT' --expname repeat10_current5_branch_exhaustive_temp10 --temperature 1.0 --max_concurrent_requests '$MAX_CONCURRENT' 2>&1 | tee '$LOG'; echo '[done]' \$?; sleep infinity"
tmux new-session -d -s "$SESSION" -n branch "$cmd"
tmux set-option -t "$SESSION" remain-on-exit on >/dev/null

echo "Launched tmux session: $SESSION"
echo "Output: $OUT"

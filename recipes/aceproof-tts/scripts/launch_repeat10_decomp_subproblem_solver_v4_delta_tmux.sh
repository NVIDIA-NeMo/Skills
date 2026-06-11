#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/igitman/workspace/NeMo-Skills-aceproof-mp
PY=/home/igitman/workspace/NeMo-Skills/.venv/bin/python
SESSION=repeat10-decomp-subproblem-solver-v4-delta
INPUT=outputs/repeat10-decomp-subproblem-solver/current_subproblems_v4_delta.jsonl
PROMPT=recipes/aceproof-tts/prompts/proof_generation_subproblem_solver.yaml
OUT="$ROOT/outputs/repeat10-decomp-subproblem-solver/v4_delta_solver"
LOG="$ROOT/outputs/repeat10-decomp-subproblem-solver/v4-delta-solver.log"
MAX_CONCURRENT="${MAX_CONCURRENT:-64}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 1
fi

cmd="cd '$ROOT'; export OPENAI_API_KEY=\"\${OPENAI_API_KEY:-dummy}\"; export NEMO_SKILLS_OPENAI_AIOHTTP=\"\${NEMO_SKILLS_OPENAI_AIOHTTP:-1}\"; export NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT=\"\${NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT:-65536}\"; export PYTHONUNBUFFERED=1; export PATH='$(dirname "$PY")':\$PATH; mkdir -p '$OUT' outputs/repeat10-decomp-subproblem-solver; '$PY' recipes/aceproof-tts/pipeline/run_proof_gen_sidecar.py --input_file '$INPUT' --output_dir '$OUT' --prompt_config_path '$PROMPT' --expname repeat10_decomp_subproblem_solver_v4_delta --temperature 1.0 --max_concurrent_requests '$MAX_CONCURRENT' 2>&1 | tee '$LOG'; echo '[done]' \$?; sleep infinity"
tmux new-session -d -s "$SESSION" -n solver "$cmd"
tmux set-option -t "$SESSION" remain-on-exit on >/dev/null

echo "Launched tmux session: $SESSION"
echo "Output: $OUT"

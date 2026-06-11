#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/igitman/workspace/NeMo-Skills-aceproof-mp
PY=/home/igitman/workspace/NeMo-Skills/.venv/bin/python
SESSION="${SESSION:-repeat10-current2-028-c50-focus}"
INPUT=recipes/aceproof-tts/dataset/repeat10-newckpt-current-unsolved2-028-c50-20260611.jsonl
OUT_ROOT="$ROOT/outputs/repeat10-ultra-mp-current2-028-c50-focus"
LOG_ROOT="$ROOT/outputs/repeat10-ultra-mp-current2-028-c50-focus-launch-logs"
N_PARALLEL="${N_PARALLEL:-64}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_CONCURRENT="${MAX_CONCURRENT:-16}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 1
fi

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

arms=(
  "proofonly:recipes/aceproof-tts/prompts/proof_generation_proof_only.yaml"
  "obligation:recipes/aceproof-tts/prompts/proof_generation_obligation_certificate.yaml"
  "invariantextremal:recipes/aceproof-tts/prompts/proof_generation_invariant_extremal.yaml"
  "decompose:recipes/aceproof-tts/prompts/proof_generation_decompose.yaml"
)

make_cmd() {
  local arm="$1"
  local prompt="$2"
  local out="$OUT_ROOT/temp10/$arm"
  local log="$LOG_ROOT/temp10-$arm.log"
  cat <<CMD
cd '$ROOT'
export OPENAI_API_KEY="\${OPENAI_API_KEY:-dummy}"
export NEMO_SKILLS_OPENAI_AIOHTTP="\${NEMO_SKILLS_OPENAI_AIOHTTP:-1}"
export NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT="\${NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT:-65536}"
export NEMO_SKILLS_TRANSIENT_ERROR_RETRIES="\${NEMO_SKILLS_TRANSIENT_ERROR_RETRIES:-24}"
export PYTHONUNBUFFERED=1
export PATH='$(dirname "$PY")':\$PATH
mkdir -p '$out' '$LOG_ROOT'
'$PY' recipes/aceproof-tts/pipeline/prepare_round1.py --input_paths '$INPUT' --output_dir '$out' --n_parallel_proof_gen '$N_PARALLEL' --prompt_config_path '$prompt' --interleave_rows
'$PY' recipes/aceproof-tts/pipeline/run_proof_gen_sidecar.py --input_file '$out/rounds/R1/proof_gen/input.jsonl' --output_dir '$out/rounds/R1/proof_gen' --prompt_config_path '$prompt' --expname repeat10_current2_028_c50_${arm}_temp10 --temperature '$TEMPERATURE' --max_concurrent_requests '$MAX_CONCURRENT' 2>&1 | tee '$log'
echo '[done]' \$?
sleep infinity
CMD
}

first=1
for spec in "${arms[@]}"; do
  arm="${spec%%:*}"
  prompt="${spec#*:}"
  cmd="$(make_cmd "$arm" "$prompt")"
  if [[ "$first" == 1 ]]; then
    tmux new-session -d -s "$SESSION" -n "$arm" "$cmd"
    first=0
  else
    tmux new-window -t "$SESSION" -n "$arm" "$cmd"
  fi
done

tmux set-option -t "$SESSION" remain-on-exit on >/dev/null

echo "Launched tmux session: $SESSION"
echo "Windows: ${#arms[@]}"
echo "Output root: $OUT_ROOT"

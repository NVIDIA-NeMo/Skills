#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/igitman/workspace/NeMo-Skills-aceproof-mp
PY=/home/igitman/workspace/NeMo-Skills/.venv/bin/python
SESSION="${SESSION:-repeat10-genonly-v5-109-030-verifiers}"
INPUT=outputs/repeat10-genonly-candidate-verification/current_verify_v5_109_030_filtered_x8.jsonl
LOG_ROOT="$ROOT/outputs/repeat10-genonly-candidate-verification"
MAX_CONCURRENT="${MAX_CONCURRENT:-32}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 1
fi

mkdir -p "$LOG_ROOT"

arms=(
  "strict:recipes/aceproof-tts/prompts/proof_verification_strict_audit.yaml:v5_109_030_strict_audit_x8"
  "theorem:recipes/aceproof-tts/prompts/proof_verification_theorem_gate.yaml:v5_109_030_theorem_gate_x8"
  "congruence:recipes/aceproof-tts/prompts/proof_verification_congruence_guard.yaml:v5_109_030_congruence_guard_x8"
  "gaussiansign:recipes/aceproof-tts/prompts/proof_verification_gaussian_sign_branch_guard.yaml:v5_109_030_gaussian_sign_guard_x8"
)

make_cmd() {
  local arm="$1"
  local prompt="$2"
  local out="$3"
  local log="$LOG_ROOT/v5-109-030-${arm}.log"
  cat <<CMD
cd '$ROOT'
export OPENAI_API_KEY="\${OPENAI_API_KEY:-dummy}"
export NEMO_SKILLS_OPENAI_AIOHTTP="\${NEMO_SKILLS_OPENAI_AIOHTTP:-1}"
export NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT="\${NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT:-65536}"
export NEMO_SKILLS_TRANSIENT_ERROR_RETRIES="\${NEMO_SKILLS_TRANSIENT_ERROR_RETRIES:-24}"
export PYTHONUNBUFFERED=1
export PATH='$(dirname "$PY")':\$PATH
'$PY' recipes/aceproof-tts/pipeline/run_verify_sidecar.py --input_file '$INPUT' --output_dir 'outputs/repeat10-genonly-candidate-verification/$out' --prompt_config_path '$prompt' --expname repeat10_genonly_v5_109_030_${arm}_verify --max_concurrent_requests '$MAX_CONCURRENT' 2>&1 | tee '$log'
echo '[done]' \$?
sleep infinity
CMD
}

first=1
for spec in "${arms[@]}"; do
  IFS=: read -r arm prompt out <<<"$spec"
  cmd="$(make_cmd "$arm" "$prompt" "$out")"
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

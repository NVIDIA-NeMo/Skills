#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/igitman/workspace/NeMo-Skills-aceproof-mp
PY=/home/igitman/workspace/NeMo-Skills/.venv/bin/python
SESSION="${SESSION:-repeat10-legacy-028-c50-verifiers}"
INPUT=outputs/repeat10-genonly-candidate-verification/current_verify_028_c50_legacy_filtered_x8.jsonl
LOG_ROOT="$ROOT/outputs/repeat10-genonly-candidate-verification"
MAX_CONCURRENT="${MAX_CONCURRENT:-16}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 1
fi

mkdir -p "$LOG_ROOT"

arms=(
  "strict:recipes/aceproof-tts/prompts/proof_verification_strict_audit.yaml:legacy_028_c50_strict_audit_x8"
  "theorem:recipes/aceproof-tts/prompts/proof_verification_theorem_gate.yaml:legacy_028_c50_theorem_gate_x8"
  "congruence:recipes/aceproof-tts/prompts/proof_verification_congruence_guard.yaml:legacy_028_c50_congruence_guard_x8"
)

make_cmd() {
  local arm="$1"
  local prompt="$2"
  local out="$3"
  local log="$LOG_ROOT/legacy-028-c50-${arm}.log"
  cat <<CMD
cd '$ROOT'
export OPENAI_API_KEY="\${OPENAI_API_KEY:-dummy}"
export NEMO_SKILLS_OPENAI_AIOHTTP="\${NEMO_SKILLS_OPENAI_AIOHTTP:-1}"
export NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT="\${NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT:-65536}"
export NEMO_SKILLS_TRANSIENT_ERROR_RETRIES="\${NEMO_SKILLS_TRANSIENT_ERROR_RETRIES:-24}"
export PYTHONUNBUFFERED=1
export PATH='$(dirname "$PY")':\$PATH
'$PY' recipes/aceproof-tts/pipeline/run_verify_sidecar.py --input_file '$INPUT' --output_dir 'outputs/repeat10-genonly-candidate-verification/$out' --prompt_config_path '$prompt' --expname repeat10_legacy_028_c50_${arm}_verify --max_concurrent_requests '$MAX_CONCURRENT' 2>&1 | tee '$log'
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

#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:/nemo_run/code"
export HYDRA_FULL_ERROR=1
cd /nemo_run/code

extra_refine_args=()
if [[ -n "${ATTEMPT0_BANK:-}" ]]; then
  extra_refine_args+=(++refine_attempt0_bank_file="${ATTEMPT0_BANK}")
fi
if [[ -n "${REFINE_RESUME_BANK:-}" ]]; then
  if [[ -n "${ATTEMPT0_BANK:-}" ]]; then
    echo "ATTEMPT0_BANK and REFINE_RESUME_BANK are mutually exclusive" >&2
    exit 2
  fi
  extra_refine_args+=(++refine_resume_bank_file="${REFINE_RESUME_BANK}")
fi

python -m nemo_skills.inference.eval.swebench \
  ++skip_filled=True \
  ++max_concurrent_requests="${MAX_CONCURRENT_REQUESTS:-50}" \
  ++reuse_preinstalled_setup=True \
  ++agent_timeout="${AGENT_TIMEOUT:-1800}" \
  ++max_refine_rounds="${MAX_REFINE_ROUNDS:-2}" \
  ++refine_strategy="${REFINE_STRATEGY:-compact_raw}" \
  ++carry_over_token_budget="${CARRY_OVER_TOKEN_BUDGET:-30000}" \
  ++refine_verify_feedback_chars="${REFINE_VERIFY_FEEDBACK_CHARS:-6000}" \
  ++refine_failure_snippet_chars="${REFINE_FAILURE_SNIPPET_CHARS:-3000}" \
  ++input_file="${INPUT_FILE}" \
  ++output_file="${RUN_DIR}/eval-results/swe-bench/output.jsonl" \
  ++num_chunks="${NUM_CHUNKS}" \
  ++chunk_id="${CHUNK_ID}" \
  ++server.port="${PORT}" \
  ++server.model="${MODEL}" \
  ++server.server_type=vllm \
  ++eval_config.split=default \
  ++agent_framework="${AGENT_FRAMEWORK:-swe_agent_refine}" \
  ++inference.temperature="${TEMPERATURE:-0.6}" \
  ++inference.top_p="${TOP_P:-0.95}" \
  ++inference.top_k="${TOP_K:-20}" \
  ++inference.min_p="${MIN_P:-0.0}" \
  ++inference.repetition_penalty="${REPETITION_PENALTY:-1.0}" \
  ++inference.random_seed="${RANDOM_SEED:-2}" \
  ++agent_max_turns="${AGENT_MAX_TURNS:-100}" \
  ++server.host=127.0.0.1 \
  "${extra_refine_args[@]}"

touch "${RUN_DIR}/eval-results/swe-bench/output_chunk_${CHUNK_ID}.jsonl.done"

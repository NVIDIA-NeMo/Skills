#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

for name in MODEL RUN_DIR INPUT_FILE SKILLS_CODE VLLM_IMAGE SKILLS_IMAGE RUNTIME_CACHE_ARCHIVE NUM_CHUNKS PORT_BASE; do
  if [[ -z "${!name:-}" ]]; then
    echo "Required environment variable is unset: ${name}" >&2
    exit 2
  fi
done

export PYTHONUNBUFFERED=1
export SLURM_UNBUFFEREDIO=1
export HF_HOME=${HF_HOME:-${RUN_DIR}/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${RUN_DIR}/.cache/huggingface/datasets}
export MAX_REFINE_ROUNDS=${MAX_REFINE_ROUNDS:-2}
export REFINE_STRATEGY=${REFINE_STRATEGY:-compact_raw}
export AGENT_FRAMEWORK=${AGENT_FRAMEWORK:-swe_agent_refine}
export AGENT_MAX_TURNS=${AGENT_MAX_TURNS:-100}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export ATTEMPT0_BANK=${ATTEMPT0_BANK:-}
export REFINE_RESUME_BANK=${REFINE_RESUME_BANK:-}
export AGENT_TIMEOUT=${AGENT_TIMEOUT:-1800}
export MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS:-50}
export CARRY_OVER_TOKEN_BUDGET=${CARRY_OVER_TOKEN_BUDGET:-30000}
export REFINE_VERIFY_FEEDBACK_CHARS=${REFINE_VERIFY_FEEDBACK_CHARS:-6000}
export REFINE_FAILURE_SNIPPET_CHARS=${REFINE_FAILURE_SNIPPET_CHARS:-3000}
export TEMPERATURE=${TEMPERATURE:-0.6}
export TOP_P=${TOP_P:-0.95}
export TOP_K=${TOP_K:-20}
export MIN_P=${MIN_P:-0.0}
export REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
export RANDOM_SEED=${RANDOM_SEED:-2}
export TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen3_coder}
export REASONING_PARSER_NAME=${REASONING_PARSER_NAME:-qwen3}

test -f "${MODEL}/config.json"
test -f "${INPUT_FILE}"
test -f "${RUNTIME_CACHE_ARCHIVE}"
test -d "${SKILLS_CODE}/nemo_skills"
mkdir -p "${RUN_DIR}/eval-logs" "${RUN_DIR}/eval-results/swe-bench" "${HF_HOME}" "${HF_DATASETS_CACHE}"

required_nodes=$(((NUM_CHUNKS + GPUS_PER_NODE - 1) / GPUS_PER_NODE))
if ((SLURM_NNODES < required_nodes)); then
  echo "Need ${required_nodes} nodes for ${NUM_CHUNKS} chunks, got ${SLURM_NNODES}" >&2
  exit 2
fi

setup_root="/tmp/${USER:-swe}-swe-runtime-${SLURM_JOB_ID}"
srun --nodes="${required_nodes}" --ntasks="${required_nodes}" --ntasks-per-node=1 \
  --cpus-per-task=2 --distribution=block:block \
  --output="${RUN_DIR}/eval-logs/setup_node%N_%j.log" \
  bash -lc "
    set -euo pipefail
    mkdir -p '${setup_root}'
    tar -xf '${RUNTIME_CACHE_ARCHIVE}' -C '${setup_root}'
    test -x '${setup_root}/SWE-agent/venv/bin/python'
    test -x '${setup_root}/SWE-bench/venv/bin/python'
    test -f '${setup_root}/.swebench_setup_ready'
  "

server_mounts="${SKILLS_CODE}:/nemo_run/code"
eval_mounts="${SKILLS_CODE}:/nemo_run/code,${setup_root}:/root"
if [[ -n "${CONTAINER_MOUNTS:-}" ]]; then
  server_mounts="${CONTAINER_MOUNTS},${server_mounts}"
  eval_mounts="${CONTAINER_MOUNTS},${eval_mounts}"
fi
if [[ -n "${REASONING_PARSER_FILE:-}" ]]; then
  test -f "${REASONING_PARSER_FILE}"
  reasoning_parser_target=${REASONING_PARSER_TARGET:-/usr/local/lib/python3.12/dist-packages/vllm/reasoning/qwen3_reasoning_parser.py}
  server_mounts="${server_mounts},${REASONING_PARSER_FILE}:${reasoning_parser_target}"
fi

srun --nodes="${required_nodes}" --ntasks="${NUM_CHUNKS}" --ntasks-per-node="${GPUS_PER_NODE}" \
  --cpus-per-task=2 --gpus-per-task=1 --gpu-bind=single:1 --distribution=block:block \
  --output="${RUN_DIR}/eval-logs/server_chunk%t_%j.log" \
  --container-image="${VLLM_IMAGE}" \
  --container-mounts="${server_mounts}" \
  --container-workdir=/nemo_run/code \
  --wait=60 --kill-on-bad-exit=1 --no-container-mount-home --mpi=pmix \
  --container-env=HF_HOME,HF_DATASETS_CACHE,MODEL,PORT_BASE,MAX_MODEL_LEN,TOOL_CALL_PARSER,REASONING_PARSER_NAME,SLURM_PROCID \
  bash /nemo_run/code/nemo_skills/inference/eval/launchers/swebench_refine/packed_server_worker.sh &
server_pid=$!

cleanup() {
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT

sleep 30
if ! kill -0 "${server_pid}" 2>/dev/null; then
  wait "${server_pid}" || true
  echo "Packed vLLM server step exited before evaluation start" >&2
  exit 1
fi

set +e
srun --nodes="${required_nodes}" --ntasks="${NUM_CHUNKS}" --ntasks-per-node="${GPUS_PER_NODE}" \
  --cpus-per-task=6 --distribution=block:block --overlap \
  --output="${RUN_DIR}/eval-logs/eval_chunk%t_%j.log" \
  --container-image="${SKILLS_IMAGE}" \
  --container-mounts="${eval_mounts}" \
  --container-workdir=/nemo_run/code \
  --kill-on-bad-exit=1 --no-container-mount-home --mpi=pmix \
  --container-env=HF_HOME,HF_DATASETS_CACHE,MODEL,PORT_BASE,RUN_DIR,NUM_CHUNKS,INPUT_FILE,ATTEMPT0_BANK,REFINE_RESUME_BANK,AGENT_FRAMEWORK,MAX_REFINE_ROUNDS,REFINE_STRATEGY,AGENT_MAX_TURNS,AGENT_TIMEOUT,MAX_CONCURRENT_REQUESTS,CARRY_OVER_TOKEN_BUDGET,REFINE_VERIFY_FEEDBACK_CHARS,REFINE_FAILURE_SNIPPET_CHARS,TEMPERATURE,TOP_P,TOP_K,MIN_P,REPETITION_PENALTY,RANDOM_SEED,SLURM_PROCID \
  bash /nemo_run/code/nemo_skills/inference/eval/launchers/swebench_refine/packed_eval_worker.sh
eval_code=$?
set -e

cleanup
trap - EXIT
if ((eval_code != 0)); then
  echo "One or more evaluation chunks failed; refusing final merge" >&2
  exit "${eval_code}"
fi

python3 "${SCRIPT_DIR}/merge_results.py"

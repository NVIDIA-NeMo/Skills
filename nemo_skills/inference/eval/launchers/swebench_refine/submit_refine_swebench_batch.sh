#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)

ACCOUNT=${ACCOUNT:-}
PARTITION=${PARTITION:-interactive}
WALLTIME=${WALLTIME:-04:00:00}
MODEL=${MODEL:-}
RUN_DIR=${RUN_DIR:-}
INPUT_FILE=${INPUT_FILE:-}
SKILLS_CODE=${SKILLS_CODE:-${REPO_ROOT}}
VLLM_IMAGE=${VLLM_IMAGE:-}
SKILLS_IMAGE=${SKILLS_IMAGE:-}
RUNTIME_CACHE_ARCHIVE=${RUNTIME_CACHE_ARCHIVE:-}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-}
REASONING_PARSER_FILE=${REASONING_PARSER_FILE:-}
ATTEMPT0_BANK=${ATTEMPT0_BANK:-}
REFINE_RESUME_BANK=${REFINE_RESUME_BANK:-}
AGENT_FRAMEWORK=${AGENT_FRAMEWORK:-swe_agent_refine}
MAX_REFINE_ROUNDS=${MAX_REFINE_ROUNDS:-2}
REFINE_STRATEGY=${REFINE_STRATEGY:-compact_raw}
NUM_CHUNKS=${NUM_CHUNKS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
PORT_BASE=${PORT_BASE:-24600}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
AGENT_MAX_TURNS=${AGENT_MAX_TURNS:-100}
JOB_NAME=${JOB_NAME:-swe-refine}
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: submit_refine_swebench_batch.sh [options]

Required:
  --account NAME
  --model PATH
  --run-dir PATH
  --input-file PATH
  --vllm-image PATH
  --skills-image PATH
  --runtime-cache PATH
  --container-mounts MOUNTS   Comma-separated Pyxis mounts needed by the data/model/images.

Common options:
  --partition NAME            Default: interactive
  --time HH:MM:SS             Default: 04:00:00
  --skills-code PATH          Default: repository containing this script
  --num-chunks N              Default: 8 (one chunk per GPU)
  --agent-framework NAME      swe_agent_refine (default) or swe_agent
  --no-refine                 Alias for --agent-framework swe_agent
  --max-refine-rounds N       Default: 2
  --refine-strategy NAME      Default: compact_raw
  --refine-attempt0-bank-file PATH
  --refine-resume-bank-file PATH
  --reasoning-parser-file PATH
  --context-k N               Context length in Ki tokens; default: 64
  --turns N                   Agent turn limit; default: 100
  --port-base N               Default: 24600
  --job-name NAME             Default: swe-refine
  --dry-run                   Validate and print the sbatch command without submitting.
EOF
}

while (($#)); do
  case "$1" in
    --account) ACCOUNT=$2; shift 2 ;;
    --partition) PARTITION=$2; shift 2 ;;
    --time) WALLTIME=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --run-dir) RUN_DIR=$2; shift 2 ;;
    --input-file) INPUT_FILE=$2; shift 2 ;;
    --skills-code) SKILLS_CODE=$2; shift 2 ;;
    --vllm-image) VLLM_IMAGE=$2; shift 2 ;;
    --skills-image) SKILLS_IMAGE=$2; shift 2 ;;
    --runtime-cache) RUNTIME_CACHE_ARCHIVE=$2; shift 2 ;;
    --container-mounts) CONTAINER_MOUNTS=$2; shift 2 ;;
    --reasoning-parser-file) REASONING_PARSER_FILE=$2; shift 2 ;;
    --refine-attempt0-bank-file) ATTEMPT0_BANK=$2; shift 2 ;;
    --refine-resume-bank-file) REFINE_RESUME_BANK=$2; shift 2 ;;
    --agent-framework) AGENT_FRAMEWORK=$2; shift 2 ;;
    --no-refine) AGENT_FRAMEWORK=swe_agent; shift ;;
    --max-refine-rounds) MAX_REFINE_ROUNDS=$2; shift 2 ;;
    --refine-strategy) REFINE_STRATEGY=$2; shift 2 ;;
    --num-chunks) NUM_CHUNKS=$2; shift 2 ;;
    --context-k) MAX_MODEL_LEN=$(($2 * 1024)); shift 2 ;;
    --turns) AGENT_MAX_TURNS=$2; shift 2 ;;
    --port-base) PORT_BASE=$2; shift 2 ;;
    --job-name) JOB_NAME=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for name in ACCOUNT MODEL RUN_DIR INPUT_FILE VLLM_IMAGE SKILLS_IMAGE RUNTIME_CACHE_ARCHIVE CONTAINER_MOUNTS; do
  if [[ -z "${!name}" ]]; then
    echo "Missing required option: ${name}" >&2
    usage >&2
    exit 2
  fi
done
if [[ -n "${ATTEMPT0_BANK}" && -n "${REFINE_RESUME_BANK}" ]]; then
  echo "Attempt-0 and resume banks are mutually exclusive" >&2
  exit 2
fi
if [[ "${AGENT_FRAMEWORK}" != swe_agent_refine && "${AGENT_FRAMEWORK}" != swe_agent ]]; then
  echo "Unsupported agent framework: ${AGENT_FRAMEWORK}" >&2
  exit 2
fi
if [[ "${AGENT_FRAMEWORK}" == swe_agent && ( -n "${ATTEMPT0_BANK}" || -n "${REFINE_RESUME_BANK}" ) ]]; then
  echo "Attempt-0 and resume banks require swe_agent_refine" >&2
  exit 2
fi

NODES=$(((NUM_CHUNKS + GPUS_PER_NODE - 1) / GPUS_PER_NODE))

export MODEL RUN_DIR INPUT_FILE SKILLS_CODE VLLM_IMAGE SKILLS_IMAGE
export RUNTIME_CACHE_ARCHIVE CONTAINER_MOUNTS REASONING_PARSER_FILE
export ATTEMPT0_BANK REFINE_RESUME_BANK AGENT_FRAMEWORK MAX_REFINE_ROUNDS REFINE_STRATEGY
export NUM_CHUNKS GPUS_PER_NODE PORT_BASE MAX_MODEL_LEN AGENT_MAX_TURNS

sbatch_command=(sbatch
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --nodes="${NODES}"
  --ntasks-per-node="${GPUS_PER_NODE}"
  --gpus-per-node="${GPUS_PER_NODE}"
  --time="${WALLTIME}"
  --job-name="${JOB_NAME}"
  --output="${RUN_DIR}/slurm-%j.out"
  --export=ALL
  "${SCRIPT_DIR}/eval_refine.sbatch")

if ((DRY_RUN)); then
  printf 'Environment: AGENT_FRAMEWORK=%q MAX_REFINE_ROUNDS=%q REFINE_STRATEGY=%q NUM_CHUNKS=%q\n' \
    "${AGENT_FRAMEWORK}" "${MAX_REFINE_ROUNDS}" "${REFINE_STRATEGY}" "${NUM_CHUNKS}"
  printf 'Command:'
  printf ' %q' "${sbatch_command[@]}"
  printf '\n'
else
  mkdir -p "${RUN_DIR}"
  "${sbatch_command[@]}"
fi

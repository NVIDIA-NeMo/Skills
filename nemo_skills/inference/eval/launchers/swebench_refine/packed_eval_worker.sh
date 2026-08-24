#!/usr/bin/env bash
set -euo pipefail

export CHUNK_ID=${SLURM_PROCID}
export PORT=$((PORT_BASE + CHUNK_ID))
exec bash /nemo_run/code/nemo_skills/inference/eval/launchers/swebench_refine/eval.sh

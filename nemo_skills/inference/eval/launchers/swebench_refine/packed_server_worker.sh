#!/usr/bin/env bash
set -euo pipefail

export PORT=$((PORT_BASE + SLURM_PROCID))
exec bash /nemo_run/code/nemo_skills/inference/eval/launchers/swebench_refine/server.sh

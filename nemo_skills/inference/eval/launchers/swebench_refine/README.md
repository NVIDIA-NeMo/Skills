# SWE-bench Refine Slurm Launcher

This directory contains one packed-GPU launch path for direct single-turn evaluation, Refine evaluation, and continuation scaling. One vLLM server and one evaluator run on each GPU; chunk outputs are validated and merged only after every chunk writes a completion marker.

## Prerequisites

- A Slurm cluster with Pyxis/Enroot and nested Apptainer support.
- A Hugging Face model directory containing `config.json`.
- A prepared SWE-bench JSONL input whose container paths are visible inside the evaluator container.
- vLLM and NeMo Skills container images.
- A runtime cache containing SWE-agent and SWE-bench virtual environments.
- Any benchmark image directories included in `--container-mounts`.

Do not commit the runtime cache, model, input/output JSONL files, trajectories, or Slurm logs.

## Build the runtime cache

The cache builder intentionally refuses to overwrite an existing archive. Its defaults pin the SWE-agent and SWE-bench commits used by the existing Refine evaluation runtime; override them explicitly when testing a newer runtime.

- SWE-agent: `3ea751c087f32b16e039a2233dd6eefecef325d5`
- SWE-bench: `c6cd63f0f48956fa3db9ba1e0fadff3a4cd2e2c6`

```bash
export RUNTIME_CACHE_ARCHIVE=/path/to/swe_runtime_cache.tar
export SKILLS_IMAGE=/path/to/nemo-skills.sqsh
# Optional overrides:
# export SWE_AGENT_COMMIT=<commit>
# export SWE_BENCH_COMMIT=<commit>

sbatch \
  --account <account> \
  --partition=interactive \
  --nodes=1 \
  --gpus-per-node=1 \
  --time=00:30:00 \
  nemo_skills/inference/eval/launchers/swebench_refine/build_runtime_cache.sbatch
```

## Start an R2 evaluation

`--max-refine-rounds 2` means round 0 plus one eval-time Refine attempt.

```bash
nemo_skills/inference/eval/launchers/swebench_refine/submit_refine_swebench_batch.sh \
  --account <account> \
  --partition interactive \
  --model /path/to/model \
  --run-dir /path/to/output/run-r2 \
  --input-file /path/to/swe-bench/default.jsonl \
  --vllm-image /path/to/vllm.sqsh \
  --skills-image /path/to/nemo-skills.sqsh \
  --runtime-cache /path/to/swe_runtime_cache.tar \
  --container-mounts /path/visible/on/host:/path/visible/in/container \
  --num-chunks 16 \
  --context-k 64 \
  --turns 100 \
  --max-refine-rounds 2 \
  --refine-strategy compact_raw
```

Use a comma-separated mount list when the model, data, output, and SWE-bench images live under different roots. Paths supplied as `--model`, `--run-dir`, and `--input-file` must resolve to the same location inside the containers.

## Start a direct single-turn evaluation

Use the same entrypoint with `--no-refine`. This selects `agent_framework=swe_agent`; no Refine bank or continuation state is involved.

```bash
nemo_skills/inference/eval/launchers/swebench_refine/submit_refine_swebench_batch.sh \
  <the same cluster/model/input options> \
  --run-dir /path/to/output/single-turn \
  --context-k 64 \
  --turns 100 \
  --no-refine
```

## Continue an existing chain

Continuation reuses resolved rows and resumes unresolved rows from the previous output. For example, continuing an R2 result through R5:

```bash
nemo_skills/inference/eval/launchers/swebench_refine/submit_refine_swebench_batch.sh \
  <the same cluster/model/input options> \
  --run-dir /path/to/output/run-r5 \
  --max-refine-rounds 5 \
  --refine-resume-bank-file /path/to/output/run-r2/eval-results/swe-bench/output.jsonl
```

For a controlled strategy comparison, use `--refine-attempt0-bank-file` instead. Attempt-0 and resume banks are mutually exclusive.

## Outputs

The final merged file is:

```text
<run-dir>/eval-results/swe-bench/output.jsonl
```

Per-chunk outputs, completion markers, server/evaluator logs, trajectories, verifier output, and Refine prompt artifacts remain under the same run directory for auditing.

## Reproducing an evaluation

Record the following alongside a reported score:

- NeMo Skills Git commit and whether the worktree was clean.
- Model/checkpoint identifier.
- Input JSONL identity or checksum.
- SWE-agent and SWE-bench commits used to build the runtime cache.
- Container image identifiers.
- Refine strategy, maximum rounds, context length, turn limit, sampling seed, and sampling parameters.
- Whether the run started from round 0, an attempt-0 bank, or a resume bank.

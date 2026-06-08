# AceProof-TTS — Multiplexer streaming workflow

A streaming alternative to the blocking SLURM DAG (`pipeline/run_pipeline.py`). Instead
of staged `prepare → proof_gen → aggregate → verify → … → finalize` jobs, a single
long-running **local driver** advances every problem independently against a **persistent
multiplexer fleet** (one OpenAI gateway, two profiles), streaming each result as soon as
the problem is done. Output stays schema-compatible with `pipeline/finalize_results.py`
(`rounds/R*/{proof_gen,verify,refine}/output.jsonl` + `proof_pool/`).

## What happens internally

- **Per-problem orchestrator** (`pipeline/streaming_orchestrator.py`): each problem runs
  its own `gen → verify → score → (solved? / refine)` loop. Problems stream out
  independently rather than at a global stage barrier.
- **Early-stop verification**: each proof is verified `min_verifications_per_proof` times
  first; the remaining verifications are cancelled if the partial mean score is already
  `< 1.0` (a proof can only be declared *solved* after exhausting its full
  `n_verification_per_proof` budget — early-stop never declares solved, to avoid a
  length bias).
- **Solved-and-stop**: the first fully-correct proof cancels all of that problem's
  outstanding gen/verify work.
- **Two profiles = phase-decoupled fleets**: `aceproof-ultra-nvfp4-proofgen` (long
  context, `--max-num-seqs 8`) and `aceproof-ultra-nvfp4-verify` (shorter, wide batch
  `--max-num-seqs 64`). The driver fronts them with two OpenAI clients sharing `base_url`,
  differing only in `model`.
- **Role-separated concurrency lanes** (`scripts/streaming_generation.py`): gen / verify /
  refine each get their own `asyncio.Semaphore` (`gen_max_concurrent`,
  `verify_max_concurrent`, `refine_max_concurrent`). This is essential — with one shared
  FIFO semaphore the ~25k gen tasks (created before any verify) monopolise it and
  verification starves.
- **Autoscaling**: the multiplexer autoscaler sizes each profile to its own queue depth
  (`desired = ceil((pending+running)/slots_per_replica)`, capped at `max_replicas`),
  launches only where SLURM can start now, auto-relaunches on walltime expiry, and reaps
  to 0 when idle. Verify is the heavy stage, so it gets the larger cap.

> Note on savings: cancellation today stops the *driver* from waiting (wall-clock-to-result),
> but server-side GPU-free-on-cancel is not yet shipped in the multiplexer, so early-stop /
> solved-and-stop do not yet reduce GPU-hours. Frame comparisons as **wall-clock-to-result**.

## Prerequisites

- **Branch base**: this recipe depends on the high-concurrency inference work on
  `imoshkov/aceproof-tts-georgea` — in particular `openai: native AsyncOpenAI + aiohttp
  fast-path` (the `NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT=65536` connector). On plain `main` the
  HTTP client caps concurrency and the streaming lanes can't open up.
- **venv** `.venv/bin/python` (uv-managed, no `pip`; install extras with
  `VIRTUAL_ENV=.venv uv pip install <pkg>`). The `openai[aiohttp]` extra
  (`httpx-aiohttp`) must be installed — a missing extra hard-fails the aiohttp fast-path.
- A running multiplexer stack (queue_server `:8000`, controller `:9000`) reachable from the
  dev box, plus `NEMO_SKILLS_CONFIG_DIR` pointing at the cluster configs.

## Multiplexer setup (apply to your multiplexer checkout)

These configs are **not** in this repo (they live in the multiplexer repo). Recreate them.

**1. Profiles** — two sizing variants of `ultra`, pinned to one cluster.
`profiles/aceproof-ultra-nvfp4-proofgen.yaml`:

```yaml
name: aceproof-ultra-nvfp4-proofgen
extends: ultra
allowed_clusters: [aws-dfw]          # pin placement (autoscaler reads this; for manual scale_fleet use --clusters)
defaults:
  walltime: 04:00:00
  concurrency: 8
  partition: batch_long
  nodes: 1
  gpus: 4
  tp: 4
  vllm_args:
    --gpu-memory-utilization: "0.90"
    --max-num-seqs: "8"
    --max-model-len: "524288"
  env:
    VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"
```

`profiles/aceproof-ultra-nvfp4-verify.yaml` is identical except `concurrency: 64` and
`--max-num-seqs: "64"` (shorter verifications, wide batch).

**2. Autoscaler policy** — per-profile caps in `autoscale.yaml`. Each replica is 4 GPU
(1 node, TP=4). Example for a 512-GPU budget biased to verify (the heavy stage):

```yaml
profiles:
  aceproof-ultra-nvfp4-proofgen:
    max_replicas: 48
    slots_per_replica: 8
    max_launching: 8
  aceproof-ultra-nvfp4-verify:
    max_replicas: 80
    slots_per_replica: 64
    max_launching: 16
    idle_timeout_s: 1800   # verify demand lags gen at start / between rounds; don't reap the fleet mid-run
```

**3a. Start the autoscaler** (standalone service against the existing stack):

```bash
cd <multiplexer>
AUTOSCALE_PROFILES="aceproof-ultra-nvfp4-proofgen,aceproof-ultra-nvfp4-verify" \
AUTOSCALE_CONTROLLER_URL="http://127.0.0.1:9000" \
QUEUE_URL="http://<DEVBOX_LAN_IP>:8000" \
NEMO_SKILLS_CONFIG_DIR="<.../cluster_configs>" \
AUTOSCALE_PORT=9300 \
  python -m uvicorn nemo_multiplexer.autoscale_service:app --app-dir <multiplexer> \
    --host 127.0.0.1 --port 9300 --workers 1
```

> **Gotcha (important):** `QUEUE_URL` is baked into every launched replica as the address
> its worker uses to reach the gateway. It **must be the dev box's routable LAN IP**
> (e.g. `http://10.31.195.156:8000`), *not* `127.0.0.1` — on a remote compute node
> `127.0.0.1` is the node itself, so workers can't register and the GPUs sit idle.

**3b. Or bring the fleet up manually** (no autoscaler):

```bash
export NEMO_SKILLS_CONFIG_DIR=<.../cluster_configs>
python -m nemo_multiplexer.scale_fleet add aceproof-ultra-nvfp4-proofgen --n 48 --clusters aws-dfw
python -m nemo_multiplexer.scale_fleet add aceproof-ultra-nvfp4-verify   --n 16 --clusters aws-dfw
# readiness: curl http://127.0.0.1:8000/<profile>/admin/workers   (state ready)
```

## Run the driver

The driver runs locally (`pipeline.cluster: none`), hits the local gateway
`http://127.0.0.1:8000/v1`; the fleet does the GPU work remotely.

```bash
cd <NeMo-Skills>
export PATH=$PWD/.venv/bin:$PATH     # nemo-run's local wrapper calls bare `python`
OPENAI_API_KEY=dummy NEMO_SKILLS_OPENAI_AIOHTTP=1 \
  .venv/bin/python recipes/aceproof-tts/pipeline/run_streaming.py \
    --config recipes/aceproof-tts/configs/aceproof-tts-gems-remarkable-ultra-nvfp4-streaming.yaml \
    --input_paths <local.jsonl> --output_dir <local_dir> --expname <name>
```

A re-launch against the same `--output_dir` resumes at **problem granularity**
(`++skip_filled=True` skips finalized problems; in-progress problems restart).

For a quick end-to-end check use `configs/aceproof-tts-smoke-cmh-streaming.yaml`
(tiny: `n_gen=4`, `n_verify=4`, `min_verify=2`, `max_rounds=1`).

## Key knobs (`configs/...-streaming.yaml`)

| knob | value | note |
|---|---|---|
| `scaling.n_parallel_proof_gen` | 128 | matches the blocking baseline |
| `scaling.n_verification_per_proof` | 64 | full budget; early-stop trims it |
| `scaling.max_rounds` | 2 | R1 → R2 refinement |
| `streaming.min_verifications_per_proof` | 8 | early-stop floor |
| `streaming.gen_max_concurrent` | 512 | gen lane (≈ proofgen slots) |
| `streaming.verify_max_concurrent` | 5120 | verify lane — **must be ≥ verify fleet slots** (80×64) or the driver caps verify |
| `streaming.refine_max_concurrent` | 512 | refine lane |
| inference | temp 1.0, top_p 0.95, gen 500k / verify 200k / refine 480k tokens | matches baseline |

## Gotchas

- `QUEUE_URL` must be the routable dev-box IP for replicas (see above).
- `verify_max_concurrent` must match the verify fleet capacity; otherwise the driver lane,
  not the fleet, is the bottleneck.
- `OPENAI_API_KEY=dummy` is required (SDK needs a non-empty key; the gateway ignores it).
- SLURM / `nmux` / `scale_fleet` commands trigger cluster work — run in an approved shell.

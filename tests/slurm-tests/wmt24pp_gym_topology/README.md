# wmt24pp_gym_topology

Validates the NeMo-Gym `wmt_translation` resource server's streaming xCOMET-XXL
actor pool + per-row `comet_score` plumbing across five distinct vLLM server
topologies. Each topology stresses a different slice of the `vllm_dp_ray` +
`extra_gpu` Ray scheduling code and the COMET pool initialization path in
`Gym/resources_servers/wmt_translation/app.py`.

## Topologies

| Config | TP | DP | Model nodes | COMET nodes | Total | Model |
|--------|---:|---:|------------:|------------:|------:|-------|
| C1     |  8 |  1 |           1 |           1 |     2 | Nemotron-3-Nano |
| C2     |  8 |  4 |           4 |           1 |     5 | Nemotron-3-Nano |
| C3     | 16 |  2 |           4 |           1 |     5 | DeepSeek-V2-Lite |
| C4     | 16 |  1 |           2 |           1 |     3 | DeepSeek-V2-Lite |
| C5     |  8 |  1 |           1 |           2 |     3 | Nemotron-3-Nano |

C3 and C4 use DeepSeek-V2-Lite because Nemotron-3-Nano's hybrid (Mamba +
attention) layers fail vLLM's `MambaMixer2` `num_groups % tp_size` assertion
at TP=16. DSV2-Lite has 16 KV heads + 16 attention heads — the only public
model on this cluster that runs cross-node TP=16 cleanly.

## What it checks

For each topology the test asserts the streaming COMET invariant on the
resulting `rollouts.jsonl`:

> `rollout.comet_score is None` ⟺ `rollout.generation == ""`

i.e. every row whose model produced a non-empty translation gets a finite
xCOMET score, and every `None` is exactly the `verify()` early-return path
for empty generations. It also bounds the score to a sane band (`[-1.5, 1.5]`,
covering raw and normalized xCOMET output) and asserts at least half the rows
were scored (catches collapsed-generation regressions).

Failures from all five topologies are aggregated via `soft_assert` and
reported together at the end, so one bad config doesn't mask others.

## Cluster prerequisites

The test reuses the recipe's existing setup. Cluster config must provide:

- A workspace mount (typically `/workspace` per `slurm-tests/README.md`).
- The vllm + nemo-rl + sandbox containers from `cluster_configs/<cluster>.yaml`.
- `HF_HOME` populated with the three required models (or a network path that
  can fetch them on cold-load):
  - `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (gated; used by C1/C2/C5)
  - `deepseek-ai/DeepSeek-V2-Lite` (used by C3/C4)
  - `Unbabel/XCOMET-XXL` (xCOMET also goes online to `facebook/xlm-roberta-xxl`
    on first load; the actor pool retries 429s — see `wmt_translation/app.py`).

## Running

```bash
python tests/slurm-tests/wmt24pp_gym_topology/run_test.py \
    --cluster <your-cluster> \
    --workspace /workspace/nemo-skills-slurm-ci/wmt24pp_gym_topology \
    --expname_prefix wmt24pp_gym_topology_$(date +%Y-%m-%d)
```

The five topology jobs submit in parallel; a sixth `check-results` job is
chained `run_after` all five. Wall time depends on cluster availability —
each topology takes 10–15 min (weight load + 250 rollouts + COMET cold load).

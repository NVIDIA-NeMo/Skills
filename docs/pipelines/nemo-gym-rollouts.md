# NeMo-Gym rollouts

!!! info

    This pipeline starting script is [nemo_skills/pipeline/nemo_gym_rollouts.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/pipeline/nemo_gym_rollouts.py)

    All extra parameters are passed through to [NeMo-Gym](https://github.com/NVIDIA-NeMo/Gym)'s
    `ng_run` and `ng_collect_rollouts`.

[NeMo-Gym](https://github.com/NVIDIA-NeMo/Gym) is where new benchmarks and environments are being
added, and it's the recommended place to run and contribute evaluations. If you're already using
Nemo-Skills for orchestration — cluster configs, self-hosted servers, sandboxes, slurm scheduling —
you don't have to choose: `ns nemo_gym_rollouts` runs a Gym benchmark using the Skills pipeline
machinery.

The command orchestrates four things for you:

1. a policy server (self-hosted vLLM/TensorRT-LLM/sglang, or a pre-hosted endpoint),
2. an optional [sandbox](../basics/sandbox.md) for code execution,
3. the NeMo-Gym resource servers (`ng_run`),
4. the rollout collection client (`ng_collect_rollouts`).

Rollouts are written to `<output_dir>/rollouts.jsonl`.

## Self-hosted model

=== "ns interface"

    ```bash
    ns nemo_gym_rollouts \
        --cluster=local \
        --config_paths="ns_tools/configs/ns_tools.yaml,math_with_judge/configs/math_with_judge.yaml" \
        --input_file=/workspace/data/example.jsonl \
        --output_dir=/workspace/rollouts \
        --model=/hf_models/Qwen2.5-1.5B-Instruct \
        --server_type=vllm \
        --server_gpus=1 \
        --with_sandbox \
        +agent_name=ns_tools_simple_agent \
        +limit=10 \
        +num_samples_in_parallel=3
    ```

=== "python interface"

    ```python
    from nemo_skills.pipeline.cli import nemo_gym_rollouts, wrap_arguments

    nemo_gym_rollouts(
        ctx=wrap_arguments(
            "+agent_name=ns_tools_simple_agent "
            "+limit=10 "
            "+num_samples_in_parallel=3 "
        ),
        cluster="local",
        config_paths="ns_tools/configs/ns_tools.yaml,math_with_judge/configs/math_with_judge.yaml",
        input_file="/workspace/data/example.jsonl",
        output_dir="/workspace/rollouts",
        model="/hf_models/Qwen2.5-1.5B-Instruct",
        server_type="vllm",
        server_gpus=1,
        with_sandbox=True,
    )
    ```

## Pre-hosted model

Point at an existing endpoint with `--server_address` and skip the self-hosted server entirely.
`--policy_model_name` is required in this case.

```bash
ns nemo_gym_rollouts \
    --cluster=local \
    --config_paths="ns_tools/configs/ns_tools.yaml" \
    --input_file=/workspace/data/example.jsonl \
    --output_dir=/workspace/rollouts \
    --server_address=https://integrate.api.nvidia.com/v1 \
    --policy_model_name=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --policy_api_key=$NVIDIA_API_KEY \
    +agent_name=ns_tools_simple_agent
```

## Gym arguments

Everything prefixed with `+` or `++` is passed straight through to `ng_run` and
`ng_collect_rollouts` as a [Hydra](https://hydra.cc/) override. The ones you'll reach for most:

- **+agent_name**: required — which Gym agent to run.
- **+limit**: cap the number of input samples.
- **+num_samples_in_parallel**: concurrent in-flight requests.
- **+num_repeats**: run each prompt N times, e.g. `+num_repeats=4` for mean@4.

## Scaling across jobs

Use `--num_random_seeds` to fan rollout collection out over several independent jobs. Each job gets
its own server and sandbox (on distinct ports, so they can share a node) and writes to
`rollouts-rs{i}.jsonl`:

```bash
ns nemo_gym_rollouts \
    --cluster=slurm \
    --num_random_seeds=8 \
    ...
```

- **--starting_seed**: offset the seed numbering, e.g. to continue a previous run.
- **--random_seeds**: run an explicit subset instead, e.g. `--random_seeds='0,2,5,7'`.
- **--rerun_done**: by default seeds with an existing output file are skipped; pass this to redo them.

## Containers

The Gym step uses `cluster_config['containers']['nemo-gym']` when your
[cluster config](../basics/cluster-configs.md) defines it, and falls back to
`containers['nemo-rl']` otherwise. Override per-run with `--gym_container`.

If NeMo-Gym is installed somewhere other than the container default, point at it with `--gym_path`.
By default we use the nemo-skills code packaged to `/nemo_run/code`; set
`--use_mounted_nemo_skills=False` to use the version bundled in the Gym container instead.

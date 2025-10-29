# Pipelines

## Basics

Nemo-Skills has a large collection of building blocks that you can use to construct various pipelines to improve LLMs.
All of the "pipeline" scripts are located in the [nemo_skills/pipeline](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/pipeline)
folder and have a unified interface that help us connect them together.

Each pipeline script is a wrapper that accepts *wrapper* arguments that tell us how to orchestrate the job. These
arguments are directly listed in the corresponding Python function or visible when you run `ns <wrapper script> --help`.
Any other arguments that you pass to the wrapper script are directly passed into the *main* job that the wrapper
launches. These arguments are never checked when you submit a job, so if you have some mistake in them, you will only
know about that when the job starts running. For most of our *main* scripts we use [Hydra](https://hydra.cc/) and thus
their arguments typically start with `++arg_name`. If you're using Python API you would need to specify all *main* arguments with
`ctx=wrap_arguments("...")` interface for technical reasons.

This might sound a little complicated, so let's see how it works through an example from the [Getting Started Tutorial](../basics/index.md).

=== "ns interface"

    ```bash
    ns generate \
        --cluster=local \
        --server_type=trtllm \
        --model=/hf_models/Qwen2.5-1.5B-Instruct \
        --server_gpus=1 \
        --output_dir=/workspace/generation-local-trtllm \
        --input_file=/workspace/input.jsonl \
        ++prompt_config=/workspace/prompt.yaml
    ```

=== "python interface"

    ```python
    from nemo_skills.pipeline.cli import wrap_arguments, generate

    generate(
        cluster="local",
        server_type="trtllm",
        model="/hf_models/Qwen2.5-1.5B-Instruct",
        server_gpus=1,
        output_dir="/workspace/generation-local-trtllm",
        input_file="/workspace/input.jsonl",
        ctx=wrap_arguments(
            "++prompt_config=/workspace/prompt.yaml "
        ),
    )
    ```

In this command all arguments starting with `--` are *wrapper* arguments and everything starting with `++` are *main* arguments.
If you run `ns generate --help` you will see all the ones with `--` listed there (and more), but not the `++` ones.
The help output also contains this message that specifies which underlying *main* script we run for this command and how
to check its arguments

```bash
`python -m nemo_skills.inference.generate --help` for other supported arguments
```

You can also open that script's code in
[nemo_skills/inference/generate.py](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/inference/generate.py)
and see all arguments and logic there.

You can chain multiple pipelines together to set proper slurm dependencies using `--run_after` parameter.
There is an example in [tutorial](../basics/index.md#slurm-inference) or in
[training documentation](training.md#chaining-pipelines-with-python).


## Common parameters

Many of our scripts have a shared set of common parameters that we list here.

### All pipeline scripts

All scripts inside pipeline folder have the following parameters.

- **--cluster**: You always need to specify a cluster config that will be used to
  control where the job is executed.
- **--config_dir**: By default we search for cluster configs inside `cluster_configs`
  local folder, but you can control where they are located with this parameter.
  You can also use `NEMO_SKILLS_CONFIG_DIR` environment variable for this purpose.
- **--log_dir**: Can be used to customize the location of slurm logs.
- **--expname**: You can always specify an experiment name, which is a
  [NeMo-Run](https://github.com/NVIDIA-NeMo/Run) concept. This will control where
  the metadata is stored, the slurm job name and allows you to chain jobs one
  after the other using the `--run_after` argument.
- **--run_after**: Can be used in conjunction with `--expname` to chain jobs to
  run one after another (only applicable on slurm). E.g. run training job with
  `--expname my-training-run` and then launch an eval with `--run_after my-training-run`.
- **--mount_paths**: Can be used to mount additional paths to the cluster config dynamically.
  This is useful if you want to access some data that is not mounted in cluster config. E.g. use
  `--mount_paths /my/remote/workspace:/workspace` to mount `/workspace` folder from the host
  machine to the slurm job.
- **--check_mounted_paths**: This flag offers a few different capabilities for convenience:
    - Check if the paths specified in the script are mounted correctly. This is useful if you want to make
    sure that the paths that are mounted are available  on remote machine before running the job.
    E.g. use `--check_mounted_paths` to check if `/my/remote/workspace` folder from the host machine
    is a folder that exists and can be mounted.
    - In many cases, if the directory does not exist, we will create it for you. This is useful for
    output and log directories.
    - If paths are provided but not mounted, often times we will dynamically mount them for you.
- **--partition**: Can be used to run in a specific slurm partition (e.g. commonly used
  to launch interactive jobs).
- **--not_exclusive**: Can be used if you want to request a part o the slurm node. By default
  we set `exclusive=True`.
- **--time_min**: Can be used to specify minimum time after which the job might be killed by slurm.
  Specify in the following format `00:30:00` (for 30 minutes). Using a lower value will help jobs
  get scheduled faster.
- **--reuse_code** / **--reuse_code_exp**: Can be used to specify another experiment and reuse
  its code (to avoid re-packaing/uploading to cluster). If running from Python we will automatically
  reuse the last submitted experiment in the current Python session.

### Generation scripts

All of the scripts that involve LLM data generation accept a common set of parameters.

- **--model**: Either path to the model file or an API model name.
- **--server_type**: `nemo`, `trtllm`, `vllm` or `openai`. This is used on the client side
  to correctly format a request to a particular server. This needs to match model
  checkpoint format if self-hosting the model or has to be `openai` for both Nvidia
  NIM API as well as the OpenAI API.
- **--server_address**: Only relevant for API models. E.g. use
  `https://integrate.api.nvidia.com/v1` for Nvidia API and
  `https://api.openai.com/v1` for OpenAI API.
- **--server_gpus**: Number of GPUs needed to host a model (only applicable to self-hosted models).
- **--server_nodes**: Number of nodes needed to host a model (only applicable to self-hosted models).
- **--server_args**: Any other arguments you need to pass to a corresponding server.
  E.g. use `--server_args="--gpu-memory-utilization=0.99"` to change gpu memory utilization of a
  vLLM server.

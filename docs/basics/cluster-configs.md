# Cluster configs

All of the [pipeline scripts](../pipelines/index.md) accept `--cluster` argument which you can use
to control where the job gets executed (you need a "local" cluster config to run jobs locally as well).
That argument picks up one of the configs inside your local
[cluster_configs](https://github.com/NVIDIA-NeMo/Skills/tree/main/cluster_configs)
folder by default, but you can specify another location with `--config_dir` or set it in `NEMO_SKILLS_CONFIG_DIR` env variable.
You can also use `NEMO_SKILLS_CONFIG` env variable instead of the `--cluster` parameter.
The cluster config defines an executor (local or slurm), mounts for data/model access and (slurm-only) various parameters
such as account, partition, ssh-tunnel arguments and so on.

The recommended way to launch jobs on slurm is by running all commands locally and specifying `ssh_tunnel` portion in cluster config
to let [NeMo-Run](https://github.com/NVIDIA/NeMo-Run) know how to connect there.
But if you prefer to run from the cluster directly, you can install Nemo-Skills there
and then only specify `job_dir` parameter without using `ssh_tunnel` section in the config.

You can see example configs in [cluster_configs](https://github.com/NVIDIA-NeMo/Skills/tree/main/cluster_configs) folder.
To create a new config you can either rename and modify one of the examples or run

```bash
ns setup
```

that will help to create all necessary configs step-by-step.

## Environment variables

You can define environment variables in the cluster config file, which will be set inside the container.

```yaml
env_vars:
  - MYENVVAR  # will pick the value from env
  - MYENVVAR2=my_value  # will use my_value
```

If an environment variable is required, and you want us to fail if it's not provided,
you can use `required_env_vars` instead. One thing to note is that `required_env_vars` does not support
passing values directly, so you must provide them via environment variable only.


Depending on which pipelines you run, you might need to define the following environment variables

``` bash
# only needed for training (can opt-out with --disable_wandb)
export WANDB_API_KEY=...
# only needed if using gated models, like llama3.1
export HF_TOKEN=...
# only needed if running inference with OpenAI models
export OPENAI_API_KEY=...
# only needed if running inference with Azure OpenAI models
export AZURE_OPENAI_API_KEY=...
# only needed if running inference with Nvidia NIM models
export NVIDIA_API_KEY=...
```


## Useful tips

Here are some suggestions on what can be defined in cluster configs for different use-cases

1. Set `HF_HOME` environment variable to ensure all HuggingFace downloads are cached

2. If you want to have a custom version of one of the underlying libraries that we use
   (e.g. [NeMo](https://github.com/NVIDIA/NeMo) or [verl](https://github.com/volcengine/verl)),
   you can clone it locally (or on cluster if using slurm), make your changes and then override in the container with

      ```yaml
      mounts:
         - <your path>/NeMo:/opt/NeMo
         - <your path>/verl:/opt/verl
      ```

3. You can specify custom containers - our code should work out-of-the-box or with very little changes with different
   versions of inference libraries (e.g. [vLLM](https://github.com/vllm-project/vllm)) or training libraries
   (e.g. [NeMo](https://github.com/NVIDIA/NeMo)). If you get some errors, you might also need to modify the entry-point
   scripts we use, e.g.
   [nemo_skills/inference/server/serve_vllm.py](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/inference/server/serve_vllm.py)
   or [nemo_skills/training/start_sft.py](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/training/start_sft.py)

4. For slurm clusters it's recommended to [build .sqsh files](https://github.com/NVIDIA/enroot/blob/master/doc/cmd/import.md#example)
   for all containers and reference the cluster path

# Training using NeMo-RL

!!! info

    This pipeline starting script is [nemo_skills/pipeline/nemo_rl/sft.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/pipeline/nemo_rl/sft.py)

    All extra parameters are passed to [nemo_skills/training/nemo_rl/start_sft.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/training/nemo_rl/start_sft.py)


## Preparing the data

Before running the training we need to prepare the data in the right format. Here is an example command

```bash
python -m nemo_skills.training.prepare_data \
    ++input_files="<path to the generated synthetic data>/output-rs*.jsonl"> \
    ++output_path=sft-data.jsonl \
    ++prompt_config=generic/math \
    ++tokenizer=meta-llama/Llama-3.1-8B-Instruct
```

!!! tip

    Many scripts accept `++input_files` argument. You can use any glob patterns there and also
    reference multiple files/patterns separated by space or comma.

If you want to run that command inside container or on cluster, add `ns run_cmd --cluster=...` in the beginning.

You need to pass in the config and tokenizer so that we can format the data accordingly. There are many more parameters
that data preparation script supports which you can see
[here](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/training/data_preparation_utils/config/math_sft.yaml).
We are using [SDP library](https://github.com/NVIDIA/NeMo-speech-data-processor) for preparing the data, so it's
a good idea to check their documentation to understand how this config is structured.


## Running training

We use [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) to run LLM training,
so you can check their documentation to learn about all supported parameters.

Here is an example of how to run a training job.
It supports models in the Hugging Face format directly.

In the example below, we use Megatron as the backend.
NeMo-RL supports two training backends: fsdp and megatron. fsdp is typically slower but supports more models. If your model is supported by megatron we recommend using it.

```bash
ns nemo_rl sft \
    --cluster=slurm \
    --expname=my-training-job \
    --output_dir=/workspace/my-training-job/checkpoints \
    --hf_model=meta-llama/Llama-3.1-8B \
    --num_nodes=8 \
    --num_gpus=8 \
    --num_training_jobs=4 \
    --backend=megatron \
    --training_data=/data/sft-data.jsonl
```

This will run training on 8 nodes of 8 GPUs, using 4 dependent slurm jobs.
You can adjust the number of epochs and steps as shown below.
The training will finish once either the specified number of epochs or steps is reached.

```bash
    ++sft.max_num_epochs=2 \
    ++sft.max_num_steps=1000 \
```

It is also recommended to tune the micro batch size, max sequence length and parallelism parameters for optimal performance. If sequence packing is enabled (default) it's recommended to keep micro batch size as 1 and instead increase sequence packing length when GPU memory isn't used fully.

For dense models (e.g., Qwen3-8B), adjusting these settings can significantly improve training efficiency.

```bash
    ++policy.train_global_batch_size=32 \
    ++policy.train_micro_batch_size=1 \
    ++policy.tensor_model_parallel_size=4 \
```

For MoE models (e.g., Qwen3-30B-A3B), you can also adjust additional MoE-specific parameters to further optimize performance.

```bash
    ++policy.megatron_cfg.expert_model_parallel_size=2 \
    ++policy.megatron_cfg.expert_tensor_parallel_size=4
```


We also support sequence packing and context parallel, especially for training sequences > 4k or so, it's recommended to use sequence packing and context parallel.
By default, our sft config set sequence_packing as True.
```bash
   ++policy.sequence_packing.enabled=True \
   ++policy.context_parallel_size=4
```


The training script will automatically convert the final saved checkpoint into the HuggingFace format.
If you want to convert non-final checkpoint, use `--conversion_step=X`.
If you want to average a subset of checkpoints, add `--average_steps` parameter (e.g. --average_steps=100,200,300).

## Chaining pipelines with Python

Typically after training we want to follow up with evaluation. You can schedule
an evaluation job right away by providing a `--run_after=my-training-job` argument
which will appropriately set slurm dependencies. Here is how you can chain the commands
to schedule evaluation after training
(whenever you need to run multiple commands, it's more convenient to use python interface)

```python
from nemo_skills.pipeline.cli import wrap_arguments, sft_nemo_rl, eval

expname = "my-training-job"
cluster = "slurm"
output_dir = f"/workspace/{expname}/checkpoints"

sft_nemo_rl(
    ctx=wrap_arguments(""),
    cluster=cluster,
    expname=expname,
    output_dir=output_dir,
    hf_model="meta-llama/Llama-3.1-8B",
    num_nodes=8,
    num_gpus=8,
    num_training_jobs=4,
    training_data="/data/sft-data.jsonl",
)

eval(
    ctx=wrap_arguments(""),
    cluster=cluster,
    model=f"{output_dir}/final_hf_model",
    server_type="trtllm",
    output_dir=f"{output_dir}/results/",
    benchmarks="gsm8k,hendrycks_math",
    server_gpus=8,
    run_after=expname,
)
```


# Training using NeMo-Aligner

!!! info

    This pipeline starting script is [nemo_skills/pipeline/train.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/pipeline/train.py)

    All extra parameters are passed to either [nemo_skills/training/start_sft.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/training/start_sft.py) or [nemo_skills/training/start_dpo.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/training/start_dpo.py)


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

!!! note

    Even though we support both SFT and DPO training, the data preparation is currently only implemented
    for SFT jobs. For DPO, you'd need to manually prepare the data according to the
    [NeMo-Aligner documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/dpo.html#dpo-model-training)


## Running training

We use [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner/) to run LLM training,
so you can check their documentation to learn about all supported parameters.

Here is an example of how to run a training job.

```bash
ns train \
    --cluster=slurm \
    --expname=my-training-job \
    --output_dir=/workspace/my-training-job/checkpoints \
    --nemo_model=/nemo_models/llama3.1-8b-base \
    --num_nodes=8 \
    --num_gpus=8 \
    --num_training_jobs=4 \
    --training_data=/data/sft-data.jsonl
```

This will run training on 8 nodes of 8 GPUs, using 4 dependent slurm jobs.
By default we are training for 2 epochs, saving checkpoints every 1000 steps,
but you can adjust these values. It's also recommended to tune micro batch size
and tensor parallel parameters for optimal performance. E.g. these are good
defaults for an 8B model size

```bash
    ++model.data.train_ds.micro_batch_size=4 \
    ++model.tensor_model_parallel_size=4
```

You can customize any of the SFT parameters by directly providing them, e.g.
to disable wandb logging and add dropout use

```bash
   --disable_wandb \
   ++model.ffn_dropout=0.1 \
   ++model.attention_dropout=0.1 \
   ++model.hidden_dropout=0.1
```

The training script will average all of your generated checkpoints upon completion
(we found this to consistently increase the downstream accuracy). If you want to
only average a subset of checkpoint, add `--average_steps` parameter (e.g. if you
want to disable averaging, set it to the last training step). If you only want
to average the checkpoints of the finished job, set `--num_training_jobs=0`.

## Chaining pipelines with Python

Typically after training we want to follow up with evaluation. You can schedule
an evaluation job right away by providing a `--run_after=my-training-job` argument
which will appropriately set slurm dependencies. Here is how you can chain the commands
to schedule checkpoint conversion and evaluation after training
(whenever you need to run multiple commands, it's more convenient to use python interface)

```python
from nemo_skills.pipeline.cli import wrap_arguments, train, convert, eval

expname = "my-training-job"
cluster = "slurm"
output_dir = f"/workspace/{expname}/checkpoints"

train(
    ctx=wrap_arguments(""),
    cluster=cluster,
    expname=expname,
    output_dir=output_dir,
    nemo_model="/nemo_models/llama3.1-8b-base",
    num_nodes=8,
    num_gpus=8,
    num_training_jobs=4,
    training_data="/data/sft-data.jsonl",
)

convert(
    ctx=wrap_arguments(""),
    cluster=cluster,
    input_model=f"{output_dir}/model-averaged-nemo",
    output_model=f"{output_dir}/model-averaged-hf",
    expname=f"{expname}-to-hf",
    run_after=expname,
    convert_from="nemo",
    convert_to="hf",
    model_type="llama",
    num_gpus=8,
    tokenizer="meta-llama/Meta-Llama-3.1-8B",
)

eval(
    ctx=wrap_arguments(""),
    cluster=cluster,
    model=f"{output_dir}/model-averaged-hf",
    server_type="trtllm",
    output_dir=f"{output_dir}/results/",
    benchmarks="gsm8k,math",
    server_gpus=8,
    run_after=f"{expname}-to-hf",
)
```

## Using sequence packing and context parallel

When training on sequences >4k or so, it's recommended to use sequence packing and context parallel.
Here is an example how to do that. Most of the parameters don't need to change, but
the `global_batch_size` might need to be adjusted to be n times smaller than without packing
where n is the average number of sequences per pack, that packing script outputs, e.g.

```
[NeMo I 2025-01-16 13:57:37 prepare_packed_ft_dataset:165] Packing sequences to length 16384...
[NeMo I 2025-01-16 15:06:24 prepare_packed_ft_dataset:182] Packing is 98.23% efficient
[NeMo I 2025-01-16 15:06:24 prepare_packed_ft_dataset:183] >>>>> For pack size 16384, average number of sequences per pack is n = 3.669 <<<<<
```

Here is an example of running packing and training.

```python
from nemo_skills.pipeline.cli import wrap_arguments, train, run_cmd

expname = "my-training-job"
cluster = "slurm"
output_dir = f"/workspace/{expname}/checkpoints"

# your memory consumption will be similar to a job with
# `pack_seq_length / context_parallel` sequences without packing
pack_seq_length = 16384
context_parallel = 4

original_bs = 512
avg_sequences_per_pack = 3.7
# you need to make sure this is divisible by your data parallel rank,
# so might need to round to a power of 2
packed_bs = original_bs // avg_sequences_per_pack

# Make sure that train_ds.file_names is included in the bucket e.g., [/data/sft-data.jsonl]
packing_cmd = (
    f"python /nemo_run/code/nemo_skills/training/prepare_packed_ft_dataset.py "
    f"    ++model.data.train_ds.file_names=[/data/sft-data.jsonl] "
    f"    ++model.data.train_ds.max_seq_length={pack_seq_length} "
    f"    ++model.context_parallel_size={context_parallel} "
    f"    ++tokenizer_path=/hf_models/Meta-Llama-3.1-8B "
    f"    ++output_dir=/data "
    f"    ++pack_sizes=[{pack_seq_length}] "
    f"    ++model.data.train_ds.hf_dataset=True "
)

run_cmd(
    ctx=wrap_arguments(packing_cmd),
    cluster=cluster,
    expname=f"{expname}-packing",
    container="nemo", # please use "nemo container" for packed data prepration
    # this is a cpu-only operation, so if a cluster has a good cpu partition, it can be used
    # note that this is an expensive operation requiring a lot of CPUs and RAM
)


# The `packing_cmd` generates three files when `pack_seq_length=16384` is used, for example:

#  `packed_16384_seed0.input_ids.npy`
#  `packed_16384_seed0.loss_mask.npy`
#  `packed_16384_seed0.seq_start_id.npy`

# For training, set training_data=packed_16384_seed0.npy
# Refer to the _load_dataset_alt function in nemo_skills/training/gpt_sft_dataset.py for details on why this is required.

train(
    ctx=wrap_arguments(
        f"++model.data.train_ds.packed_sequence=True "
        f"++model.data.train_ds.micro_batch_size=1 "  # should always be 1 for packed jobs
        f"++model.data.train_ds.global_batch_size={packed_bs} "
        f"++model.context_parallel_size={context_parallel} "
        f"++model.data.train_ds.max_seq_length={pack_seq_length} "
        # all other parameters are generally the same as for the non-packed job with
        # max seq length = packed_seq_length / context_parallel
        # and keep in mind that each step now processes avg_sequences_per_pack * packed_bs examples
    ),
    cluster=cluster,
    expname=expname,
    run_after=f"{expname}-packing",
    output_dir=output_dir,
    nemo_model="/nemo_models/llama3.1-8b-base",
    num_nodes=8,
    num_gpus=8,
    num_training_jobs=4,
    training_data=f"/data/packed_{pack_seq_length}_seed0.npy",
)

# can follow up with the same convert/eval steps as above
```

If your data size is very large (i.e. >1M samples), you might run out of memory when doing packing on full data.
If that's the case, it's recommended to split data into smaller chunks and then merge them using
[nemo_skills/training/merge_packed_data.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/training/merge_packed_data.py)

Example command:

```bash
python nemo_skills/training/merge_packed_data.py \
    --input_prefixes <chunk 1 folder>/packed_24576_seed0 <chunk 2 folder>/packed_24576_seed0 \
    --output_prefix <final data folder>/packed_24576_seed0
```

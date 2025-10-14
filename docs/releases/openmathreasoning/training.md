# Model training

We assume you have `/workspace` defined in your [cluster config](../../basics/cluster-configs.md) and
that data and models will be downloaded to that folder.

## Download data and convert to SFT format

Get the data from [HuggingFace](https://huggingface.co/datasets/nvidia/OpenMathReasoning) and convert
to the SFT format that NeMo-Aligner understands.
This might take a while (depending on your network connection) and will use a significant amount of RAM.

```python
from functools import partial
from datasets import load_dataset
from nemo_skills.prompt.utils import get_prompt

def apply_format(elem, prompt, is_tir):
    if is_tir:
        if 'Remaining code executions: ' not in elem['output']:
            assert 'You have run out of code executions!' in elem['output']
            total_code_executions = 1
        else:
            total_code_executions = int(elem['output'].split('Remaining code executions: ')[1].split()[0][0]) + 1
        elem['input'] = prompt.fill({'problem': elem['input'], 'total_code_executions': total_code_executions})
    else:
        elem['input'] = prompt.fill({'problem': elem['input']})
    elem['output'] = prompt.format_assistant_response(elem['output'])
    return elem

dataset = load_dataset("nvidia/OpenMathReasoning")

for inference_mode in ["cot", "tir", "genselect"]:
    dataset[inference_mode] = dataset[inference_mode].rename_column("problem", "input")
    dataset[inference_mode] = dataset[inference_mode].rename_column("generated_solution", "output")

    code_tags = None
    if inference_mode == 'cot':
        prompt_config = 'generic/math'
    if inference_mode == 'tir':
        prompt_config = 'openmath/tir'
        code_tags = 'openmath'
    if inference_mode == 'genselect':  # already formatted
        prompt_config = {'user': '{problem}'}
    prompt = get_prompt(prompt_config, tokenizer='Qwen/Qwen2.5-32B-Instruct', code_tags=code_tags, system_message="")
    func = partial(apply_format, prompt=prompt, is_tir=(inference_mode == 'tir'))
    dataset[inference_mode] = dataset[inference_mode].map(func, num_proc=20)

dataset["cot"].to_json("omr-cot.jsonl")
dataset["tir"].to_json("omr-tir.jsonl")
dataset["genselect"].to_json("omr-genselect.jsonl")
```

If you want to train on all the data, mix it together running the following commands

```bash
cat omr-cot.jsonl omr-tir.jsonl omr-genselect.jsonl > omr-all.jsonl
shuf -o omr-all.jsonl omr-all.jsonl
```

## Pack sequences

We use NeMo-Aligner's sequence packing for training as our data has sequences of very different lengths.
Run the following code to prepare SFT data in the packed format. We split the data into multiple chunks
to speed up the process and then merge them back. Packing is computationally intensive and will use a substantial
amount of CPU and RAM resources. If you run into memory errors, increase number of splits.

First, split the data
```bash
split -l $((($(wc -l < omr-all.jsonl) + 7) / 8)) --numeric-suffixes=1 --additional-suffix='.jsonl' omr-all.jsonl 'omr-all-split'
```

Then run packing. It's recommended to run it on cluster to parallelize the process, but can also be run locally
by changing `cluster` parameter. The split files are assumed to be in `/workspace/openmathreasoning-sft` folder.

```python
from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

pack_seq_length = 32768
# NOTE: change to 4 for training 32B model!
# you need to prepare a separate data version whenever you change CP
context_parallel = 2
# all Qwen models use the same tokenizer, so any can be used here
hf_model = 'Qwen2.5-14B'
cluster = 'slurm'

num_splits = 8

for i in range(1, num_splits + 1):
    data_path = f"/workspace/openmathreasoning-sft/omr-all-split{i:02d}.jsonl"
    out_path = f"/workspace/openmathreasoning-sft/packed-data{i:02d}/"
    packing_cmd = (
        f"mkdir -p {out_path} && "
        f"python nemo_skills/training/prepare_packed_ft_dataset.py "
        f"    ++model.data.train_ds.file_names=[{data_path}] "
        f"    ++model.data.train_ds.max_seq_length={pack_seq_length} "
        f"    ++model.context_parallel_size={context_parallel} "
        f"    ++tokenizer_path=/hf_models/{hf_model} "
        f"    ++output_dir={out_path} "
        f"    ++pack_sizes=[{pack_seq_length}] "
        f"    ++model.data.train_ds.hf_dataset=True "
    )

    run_cmd(
        ctx=wrap_arguments(packing_cmd),
        cluster=cluster,
        expname=f"sequence-packing-{i}",
        log_dir=f"{out_path}",
        container="nemo",
        # this is a cpu-only operation, so if a cluster has a good cpu partition, it can be used
        # note that this is an expensive operation requiring a lot of CPUs and RAM
    )

# merging back
prefixes = [
    f"/workspace/openmathreasoning-sft/packed-data{i:02d}/packed_{pack_seq_length}_seed0"
    for i in range(1, num_splits + 1)
]
merging_cmd = (
    f"python nemo_skills/training/merge_packed_data.py "
    f"    --input_prefixes {' '.join([prefix for prefix in prefixes])} "
    f"    --output_prefix /workspace/openmathreasoning-sft/packed-data-all/packed_{pack_seq_length}_seed0"
)

run_cmd(
    ctx=wrap_arguments(merging_cmd),
    cluster=cluster,
    expname=f"sequence-packing-merging",
    run_after=[f"sequence-packing-{i}" for i in range(1, num_splits + 1)],
    log_dir=f"/workspace/openmathreasoning-sft/packed-data-all",
    container="nemo",
)
```

## Prepare base model

Download the base model and convert it to NeMo format. We used the following base models

* [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)
* [Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B)
* [Qwen2.5-14B](https://huggingface.co/Qwen/Qwen2.5-14B)
* [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B)

Here is an example of commands for Qwen2.5-Math-1.5B

```bash
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen2.5-Math-1.5B --local-dir Qwen2.5-Math-1.5B
```

For 1.5B and 7B models we use "Math" models, so we also need to update their rope base and max positional embeddings.
For 14B and 32B you should not do that!

```bash
sed -i 's/"max_position_embeddings": 4096,/"max_position_embeddings": 131072,/g' Qwen2.5-Math-1.5B/config.json
sed -i 's/"rope_theta": 10000,/"rope_theta": 500000.0,/g' Qwen2.5-Math-1.5B/config.json
```

```bash
ns convert \
    --cluster=slurm \
    --input_model=/workspace/Qwen2.5-Math-1.5B \
    --output_model=/workspace/qwen2.5-math-1.5b-nemo \
    --convert_from=hf \
    --convert_to=nemo \
    --model_type=qwen \
    --num_gpus=1 \
    --hf_model_name=Qwen/Qwen2.5-Math-1.5B
```

## Run training

Run the training (assuming slurm configuration here with the same folder structure). If your cluster has strict
timeout policy, you can run multiple dependent jobs with `--dependent_jobs=N`.

```bash
ns train \
    --cluster=slurm \
    --expname=openmathreasoning-repro-1.5b \
    --output_dir=/workspace/openmathreasoning-sft/checkpoints \
    --nemo_model=/workspace/qwen2.5-math-1.5b-nemo \
    --num_nodes=64 \
    --num_gpus=8 \
    --average_steps=7500,15000,22500,30000 \
    --training_data=/workspace/openmathreasoning-sft/packed-data-all/packed_32768_seed0.npy \
    ++model.data.train_ds.max_seq_length=32768 \
    ++model.data.train_ds.micro_batch_size=1 \
    ++model.data.train_ds.global_batch_size=256 \
    ++model.tensor_model_parallel_size=1 \
    ++model.context_parallel_size=2 \
    ++model.data.train_ds.packed_sequence=True \
    ++model.optim.lr=3e-4 \
    ++model.optim.sched.min_lr=3e-7 \
    ++model.optim.sched.warmup_steps=3000 \
    ++trainer.sft.save_interval=7500 \
    ++trainer.sft.max_steps=30000 \
    ++trainer.sft.max_epochs=100
```

Note that while we set batch size to be 256, the *real* batch size is about 4 times bigger as there are approximately
4 examples packed together in each element of the packed data.

For other models change the above parameters according to this table. Don't forget to re-pack the data when changing CP!

|                       | **lr** | **min_lr** | **TP** | **CP** |
| --------------------- | ------ | ---------- | ------ | ------ |
| **Qwen2.5-Math-1.5B** | 3e-4   | 3e-7       | 1      | 2      |
| **Qwen2.5-Math-7B**   | 2e-4   | 2e-7       | 4      | 2      |
| **Qwen2.5-14B**       | 1e-4   | 1e-7       | 8      | 2      |
| **Qwen2.5-32B**       | 1e-4   | 1e-7       | 8      | 4      |


If you want to follow up with checkpoint conversion and evaluation, see
[training docs](../../pipelines/training.md#chaining-pipelines-with-python) for an example of how to do it
through a convenient Python API.


## Second-round SFT

!!! note

    After release we realized that we didn't do filtering for TIR and GenSelect subsets. If you want
    to reproduce our results exactly, modify the code below to only apply filtering on the CoT subset
    and use original TIR and GenSelect subsets. In this case also change training duration to be 10000
    steps and update average steps and warmup accordingly.

    For best results though, we recommend doing filtering on all subsets. To do that, run the
    commands below without changes.

In our paper we also did a second round SFT for all models except 32B. All the commands stay the same
except the following changes to initial data preparation as well as a change to train for 3000 steps
instead of 30000 used in the first-round SFT.

```bash
    --nemo_model=/workspace/openmathreasoning-sft/checkpoints/model-averaged-nemo \
    --training_data=<path to the new packed data> \
    --average_steps=750,1500,2250,3000 \
    ++model.optim.sched.warmup_steps=300 \
    ++trainer.sft.max_steps=3000
```

Here is the code that can be used to prepare the second-round SFT data

```python
from functools import partial
from datasets import load_dataset
from transformers import AutoTokenizer
from nemo_skills.prompt.utils import get_prompt

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B")

def apply_format(elem, prompt, is_tir):
    if is_tir:
        if 'Remaining code executions: ' not in elem['output']:
            assert 'You have run out of code executions!' in elem['output']
            total_code_executions = 1
        else:
            total_code_executions = int(elem['output'].split('Remaining code executions: ')[1].split()[0][0]) + 1
        elem['input'] = prompt.fill({'problem': elem['input'], 'total_code_executions': total_code_executions})
    else:
        elem['input'] = prompt.fill({'problem': elem['input']})
    elem['output'] = prompt.format_assistant_response(elem['output'])
    return elem

def filter_func(example, inference_mode):
    olymp_sources = ['aops_c5_contests_amp_programs', 'aops_c6_high_school_olympiads']
    if example['problem_source'] not in olymp_sources:
        return False
    if example['pass_rate_72b_tir'] == 'n/a' or float(example['pass_rate_72b_tir']) > 0.3:
        return False
    if inference_mode == 'genselect':  # no length-based filtering for genselect
        return True
    return len(tokenizer.encode(example['output'])) >= 5000

dataset = load_dataset("nvidia/OpenMathReasoning")

for inference_mode in ["cot", "tir", "genselect"]:
    dataset[inference_mode] = dataset[inference_mode].rename_column("problem", "input")
    dataset[inference_mode] = dataset[inference_mode].rename_column("generated_solution", "output")

    code_tags = None
    if inference_mode == 'cot':
        prompt_config = 'generic/math'
    if inference_mode == 'tir':
        prompt_config = 'openmath/tir'
        code_tags = 'openmath'
    if inference_mode == 'genselect':  # already formatted
        prompt_config = {'user': '{problem}'}
    func = partial(filter_func, inference_mode=inference_mode)
    dataset[inference_mode] = dataset[inference_mode].filter(func, num_proc=20)
    prompt = get_prompt(prompt_config, tokenizer='Qwen/Qwen2.5-32B-Instruct', code_tags=code_tags, system_message="")
    func = partial(apply_format, prompt=prompt, is_tir=(inference_mode == 'tir'))
    dataset[inference_mode] = dataset[inference_mode].map(func, num_proc=20)

dataset["cot"].to_json("omr-cot-round2.jsonl")
dataset["tir"].to_json("omr-tir-round2.jsonl")
dataset["genselect"].to_json("omr-genselect-round2.jsonl")
```

Since the data is relatively small, you don't need to split it and can pack the full file directly.
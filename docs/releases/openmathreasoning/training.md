# Model training

We assume you have `/workspace` defined in your [cluster config](../../basics/cluster-configs.md) and
that data and models will be downloaded to that folder.

## Download data and convert to SFT format

Get the data from [HuggingFace](https://huggingface.co/datasets/nvidia/OpenMathReasoning) and convert it to the SFT JSONL format expected by the NeMo-RL SFT.
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


## Prepare base model

Download the base model. We used the following base models

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


## Run training

Run the training (assuming slurm configuration here with the same folder structure). If your cluster has strict
timeout policy, you can run multiple dependent jobs with `--num_training_jobs=N`.


```bash
ns nemo_rl sft \
    --cluster=slurm \
    --expname=openmathreasoning-repro-1.5b \
    --output_dir=/workspace/openmathreasoning-sft/checkpoints \
    --hf_model=/workspace/Qwen2.5-Math-1.5B  \
    --num_nodes=64 \
    --num_gpus=8 \
    --backend=megatron \
    --average_steps=7500,15000,22500,30000 \
    --training_data=/workspace/openmathreasoning-sft/omr-all.jsonl \
    ++policy.max_total_sequence_length=32768 \
    ++policy.train_micro_batch_size=1 \
    ++policy.train_global_batch_size=1024 \
    ++policy.tensor_model_parallel_size=1 \
    ++policy.context_parallel_size=2 \
    ++policy.lr=3e-4 \
    ++policy.min_lr=3e-7 \
    ++policy.megatron_cfg.scheduler.lr_warmup_iters=3000 \
    ++policy.megatron_cfg.scheduler.lr_warmup_init=0 \
    ++checkpointing.save_period=7500 \
    ++sft.max_num_steps=30000 \
    ++sft.max_num_epochs=100
```



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
    --hf_model=/workspace/openmathreasoning-sft/final_hf_model \
    --training_data=<path to the new data> \
    --average_steps=750,1500,2250,3000 \
    ++policy.megatron_cfg.scheduler.lr_warmup_iters=300 \
    ++policy.megatron_cfg.scheduler.lr_warmup_init=0 \
    ++sft.max_num_steps=3000
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
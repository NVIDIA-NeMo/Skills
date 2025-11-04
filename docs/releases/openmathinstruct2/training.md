# Model training

We assume you have `/workspace` defined in your [cluster config](../../basics/cluster-configs.md) and are
executing all commands from that folder locally. Change all commands accordingly
if running on slurm or using different paths.

## Download data

Get the data from [HuggingFace](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2).
This might take 20-30 minutes (or more depending on your network connection) and will use ~20Gb of RAM.

```python
import json

from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset('nvidia/OpenMathInstruct-2', split='train')

print("Converting dataset to jsonl format")
output_file = "openmathinstruct2.jsonl"
with open(output_file, 'w', encoding='utf-8') as f:
    for item in tqdm(dataset):
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Conversion complete. Output saved as {output_file}")
```

You can also download a subset of the data by using e.g. `split='train_5M'` that we used to train 70B model.
See the dataset page for more details about this.

## Convert to SFT format

Convert the data into the SFT format that NeMo-RL understands.

```bash
ns run_cmd --cluster=local \
python -m nemo_skills.training.prepare_data \
    ++prompt_config=generic/math \
    ++preprocessed_dataset_files=/workspace/openmathinstruct2.jsonl \
    ++output_key=generated_solution \
    ++output_path=/workspace/openmathinstruct2-sft.jsonl \
    ++tokenizer="meta-llama/Meta-Llama-3.1-8B" \
    ++filters.drop_multi_boxed=false \
    ++filters.trim_prefix=false \
    ++filters.trim_solutions=false \
    ++filters.drop_incorrect_arithmetic=false \
    ++filters.split_arithmetic=false \
    ++filters.remove_contaminated=false
```


## Run training

Run the training (assuming slurm configuration here with the same folder structure). If your cluster has strict
timeout policy, you can run multiple dependent jobs with `--dependent_jobs=N`.

```bash
ns nemo_rl sft \
    --cluster=slurm \
    --expname=openmathinstruct2-repro-8b \
    --output_dir=/workspace/openmathinstruct2-repro/checkpoints \
    --hf_model=meta-llama/Llama-3.1-8B  \
    --num_nodes=8 \
    --num_gpus=8 \
    --backend=megatron \
    --average_steps=10000,20000,30000,40000,50000,60000 \
    --training_data=/workspace/openmathinstruct2-sft.jsonl \
    ++policy.train_micro_batch_size=8 \
    ++policy.train_global_batch_size=512 \
    ++policy.tensor_model_parallel_size=4 \
    ++policy.pipeline_model_parallel_size=1 \
    ++policy.lr=2e-5 \
    ++checkpointing.save_period=10000 \
    ++sft.max_num_steps=60000 \
    ++sft.max_num_epochs=100 \
    ++policy.sequence_packing.enabled=False
```

For 70B model, we used 5M data subset and the following parameters, but training
it longer is likely going to improve results.


```bash
ns nemo_rl sft \
    --cluster=slurm \
    --expname=openmathinstruct2-repro-70b \
     --output_dir=/workspace/openmathinstruct2-repro-70b/checkpoints \
    --hf_model=meta-llama/Llama-3.1-70B  \
    --num_nodes=32 \
    --num_gpus=8 \
    --backend=megatron \
    --average_steps=3330,6660,9990,13320,16650,20000 \
    --training_data=/workspace/openmathinstruct2-sft-5M.jsonl \
    ++policy.train_micro_batch_size=1 \
    ++policy.train_global_batch_size=512 \
    ++policy.tensor_model_parallel_size=8 \
    ++policy.pipeline_model_parallel_size=2 \
    ++policy.lr=1e-5 \
    ++checkpointing.save_period=3330 \
    ++sft.max_num_steps=20000 \
    ++sft.max_num_epochs=100 \
    ++policy.sequence_packing.enabled=False
```


If you have a job timeout, it's necessary to set the maximum time per run to 40 minutes
before the timeout to allow for the final checkpoint to be saved. E.g. if your timeout is 4 hours,
add `++checkpointing.checkpoint_must_save_by=00:03:20:00`


If you want to follow up with evaluation, see [training docs](../../pipelines/training.md#chaining-pipelines-with-python) for an example of how to do it through a convenient Python API.


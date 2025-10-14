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

Convert the data into the SFT format that NeMo-Aligner understands.

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

## Prepare base model

The instructions below are for Llama3.1-8B, but the same commands should work for 70B model as well.

```bash
ns convert \
    --cluster=local \
    --input_model=meta-llama/Llama-3.1-8B \
    --output_model=/workspace/llama3.1-8b-nemo \
    --convert_from=hf \
    --convert_to=nemo \
    --model_type=llama \
    --num_gpus=1 \
    --hf_model_name=meta-llama/Llama-3.1-8B
```

## Run training

Run the training (assuming slurm configuration here with the same folder structure). If your cluster has strict
timeout policy, you can run multiple dependent jobs with `--dependent_jobs=N`.

```bash
ns train \
    --cluster=slurm \
    --expname=openmathinstruct2-repro-8b \
    --output_dir=/workspace/openmathinstruct2-repro/checkpoints \
    --nemo_model=/workspace/llama3.1-8b-nemo \
    --num_nodes=8 \
    --num_gpus=8 \
    --average_steps=10000,20000,30000,40000,50000,60000 \
    --training_data=/workspace/openmathinstruct2-sft.jsonl \
    ++model.data.train_ds.micro_batch_size=8 \
    ++model.tensor_model_parallel_size=4 \
    ++model.pipeline_model_parallel_size=1 \
    ++model.optim.lr=2e-5 \
    ++trainer.sft.save_interval=10000 \
    ++trainer.sft.max_steps=60000 \
    ++trainer.sft.max_epochs=100
```

For 70B model, we used 5M data subset and the following parameters, but training
it longer is likely going to improve results.

```bash
ns train \
    --cluster=slurm \
    --expname=openmathinstruct2-repro-70b \
    --output_dir=/workspace/openmathinstruct2-repro-70b/checkpoints \
    --nemo_model=/workspace/llama3.1-70b-nemo \
    --num_nodes=32 \
    --num_gpus=8 \
    --average_steps=3330,6660,9990,13320,16650,20000 \
    --training_data=/workspace/openmathinstruct2-sft-5M.jsonl \
    ++model.data.train_ds.micro_batch_size=1 \
    ++model.tensor_model_parallel_size=8 \
    ++model.pipeline_model_parallel_size=2 \
    ++model.optim.lr=1e-5 \
    ++trainer.sft.save_interval=3330 \
    ++trainer.sft.max_steps=20000 \
    ++trainer.sft.max_epochs=100
```

If you have a job timeout, it's necessary to set the maximum time per run to 40 minutes
before the timeout to allow for the final checkpoint to be saved. E.g. if your timeout is 4 hours,
add `++exp_manager.max_time_per_run=00:03:20:00`


If you want to follow up with checkpoint conversion and evaluation, see
[training docs](../../pipelines/training.md#chaining-pipelines-with-python) for an example of how to do it
through a convenient Python API.

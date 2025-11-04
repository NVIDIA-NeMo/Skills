# Training using verl

!!! info

    The pipeline starting script is

    * [nemo_skills/pipeline/verl/ppo.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/pipeline/verl/ppo.py)

    All extra parameters are passed to

    * [verl.trainer.main_ppo](https://github.com/volcengine/verl/blob/main/verl/trainer/main_ppo.py)




## PPO with verl

Here is an example of running PPO job with verl.
You can use [nemo_skills/training/verl/prepare_data.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/training/verl/prepare_data.py) to convert
our standard [SFT data format](./training.md#preparing-the-data) into parquet.

```python
from nemo_skills.pipeline.cli import wrap_arguments, ppo_verl

ppo_verl(
    ctx=wrap_arguments(
        '++trainer.save_freq=10 '
        '++data.train_batch_size=32 '
        '++data.filter_prompts=False '
        '++actor_rollout_ref.rollout.gpu_memory_utilization=0.7 '
        '++data.max_response_length=12000 '
        '++actor_rollout_ref.rollout.n=64 '
        '++actor_rollout_ref.rollout.tensor_model_parallel_size=2 '
    ),
    cluster="slurm",
    expname="test-verl-ppo",
    output_dir="/workspace/test-verl-ppo",
    hf_model="/hf_models/Qwen2.5-1.5B-Instruct",
    prompt_data="/data/rl-data.parquet",
    num_gpus=8,
    num_nodes=2,
)
```
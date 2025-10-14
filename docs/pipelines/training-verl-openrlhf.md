# Training using verl or OpenRLHF

!!! info

    Depending on the algorithm/framework, this pipeline starting script is

    * [nemo_skills/pipeline/openrlhf/sft.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/pipeline/openrlhf/sft.py)

    * [nemo_skills/pipeline/openrlhf/ppo.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/pipeline/openrlhf/sft.py)

    * [nemo_skills/pipeline/verl/ppo.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/pipeline/verl/ppo.py)

    All extra parameters are passed to

    * [openrlhf.cli.train_sft](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/cli/train_sft.py)

    * [openrlhf.cli.train_ppo_ray](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/cli/train_ppo_ray.py)

    * [verl.trainer.main_ppo](https://github.com/volcengine/verl/blob/main/verl/trainer/main_ppo.py)

!!! warning

    OpenRLHF support is experimental and incomplete. We use the following
    custom fork and it might not be easy to switch to official repositories versions.

    * OpenRLHF: https://github.com/Kipok/OpenRLHF

    The documentation here is incomplete and we advise you to open an issue if you
    plan to try something that is not covered below to get additional support.

    For OpenRLHF, please use the following non-default container `vllm: igitman/nemo-skills-vllm:0.6.0`

## SFT with OpenRLHF

Here is an example of running SFT job with OpenRLHF.
Our standard [SFT data format](./training.md#preparing-the-data) can be
used here.

```bash
from nemo_skills.pipeline.cli import wrap_arguments, sft_openrlhf

sft_openrlhf(
    ctx=wrap_arguments(""),
    cluster="slurm",
    expname="test-openrlhf-sft",
    output_dir="/workspace/test-openrlhf-sft",
    hf_model="/hf_models/Qwen2.5-1.5B-Instruct",
    training_data="/data/sft-data.jsonl",
    num_gpus=8,
    num_nodes=2,
    num_training_jobs=1,
)
```

## PPO with OpenRLHF

Here is an example of running PPO job with OpenRLHF.
Our standard [SFT data format](./training.md#preparing-the-data) can be
used here.

```python
from nemo_skills.pipeline.cli import wrap_arguments, ppo_openrlhf

ppo_openrlhf(
    ctx=wrap_arguments(
        "--ref_num_gpus_per_node=4 "
        "--actor_num_gpus_per_node=4 "
        "--vllm_num_engines=2 "
        "--vllm_tensor_parallel_size=2 "
        "--ref_num_nodes=1 "
        "--actor_num_nodes=1 "
        "--colocate_actor_ref "
        "--advantage_estimator=reinforce "
        "--remote_rm_url /nemo_run/code/nemo_skills/training/openrlhf/math_reward.py "
    ),
    cluster="slurm",
    expname="test-openrlhf-ppo",
    output_dir="/workspace/test-openrlhf-ppo",
    hf_model="/hf_models/Qwen2.5-1.5B-Instruct",
    prompt_data="/data/rl-data.jsonl",
    num_gpus=8,
    num_nodes=2,
    # this is used for the LLM judge
    server_gpus=8,
    server_type='trtllm',
    server_model='/hf_models/Qwen2.5-32B-Instruct',
    num_training_jobs=1,
)
```

## PPO with verl

Here is an example of running PPO job with verl.
You can use [nemo_skills/training/verl/prepare_data.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/training/verl/prepare_data.py) to convert
our standard [SFT data format](./training.md#preparing-the-data) into parquet.

```python
from nemo_skills.pipeline.cli import wrap_arguments, ppo_verl

ppo_verl(
    ctx=wrap_arguments(
        '++trainer.save_freq=0 '
        '++data.train_batch_size=32 '
        '++reward_model.compute_score=math-judge '
        '++reward_model.reward_manager=batched '
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
    # this is used for the LLM judge
    server_gpus=8,
    server_type='trtllm',
    server_model='/hf_models/Qwen2.5-32B-Instruct',
    num_training_jobs=1,
)
```
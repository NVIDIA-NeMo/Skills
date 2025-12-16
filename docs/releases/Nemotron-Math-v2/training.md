# Model training

We assume you have `/workspace` defined in your [cluster config](../../basics/cluster-configs.md) and
that data and models will be downloaded to that folder, and you already follow [dataset.md](dataset.md) to get all SFT data ready.


## Prepare base model

Download the base model.

* [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)
* [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)

Here is an example of commands for Qwen3-30B-A3B
```bash
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3-30B-A3B --local-dir Qwen3-30B-A3B
```


## Run training

Run the training (assuming slurm configuration here with the same folder structure). If your cluster has strict
timeout policy, you can run multiple dependent jobs with `dependent_jobs=N`.

The following example shows the training script for Qwen3-30B-A3B. You can modify it accordingly for Qwen3-8B.

```python
from nemo_skills.pipeline.cli  import sft_nemo_rl, wrap_arguments
cluster = 'slurm'
tp = 8
cp = 8
pp = 1
etp = 1
emp = 8
save_period=600
max_steps = 7200
batch_size=2048
num_training_jobs=10
warmup=0
partition = 'interactive'
backend='megatron'
lr=2e-4
min_lr=2e-4
sft_nemo_rl(
    ctx=wrap_arguments(
        '++sft.max_num_epochs=2000 '
        f'++sft.max_num_steps={max_steps} '
        '++data.force_reprocess=false '
        '++data.num_workers=10 '
        f'++policy.megatron_cfg.tensor_model_parallel_size={tp} '
        f'++policy.megatron_cfg.context_parallel_size={cp} '
        f'++policy.megatron_cfg.expert_model_parallel_size={emp} '
        f'++policy.megatron_cfg.expert_tensor_parallel_size={etp} '
        f'++policy.megatron_cfg.pipeline_model_parallel_size={pp} '
        f'++policy.sequence_parallel=True '
        f'++policy.megatron_cfg.bias_activation_fusion=True '
        f'++policy.megatron_cfg.apply_rope_fusion=True '
        f'++checkpointing.save_period={save_period} '
        f'++policy.train_global_batch_size={batch_size} '
        f'++policy.max_total_sequence_length=131072 '
        f'++policy.megatron_cfg.optimizer.lr={lr} '
        '++policy.megatron_cfg.optimizer.bf16=True '
        f'++policy.megatron_cfg.optimizer.min_lr={min_lr} '
        f'++policy.megatron_cfg.scheduler.lr_warmup_iters={warmup} '
        f'++policy.megatron_cfg.scheduler.lr_decay_iters={max_steps} '
        '++policy.megatron_cfg.scheduler.lr_warmup_init=1e-7 '
        '++policy.megatron_cfg.scheduler.lr_decay_style=cosine '
        '++logger.swanlab_enabled=false '
        '++checkpointing.checkpoint_must_save_by=00:03:35:00 '
    ),
    cluster=cluster,
    wandb_project='sft-Qwen3-30B-A3B',
    expname='nemo-rl-sft-Qwen3-30B-A3B',
    backend='megatron',
    output_dir='/workspace/final_sft_model',
    hf_model='/workspace/Qwen3-30B-A3B',
    training_data='/workspace/sft.jsonl',
    num_gpus=8,
    num_nodes=32,
    dependent_jobs=num_training_jobs,
)

```

### Training configuration by model and bucket length


| Model         | Context length | TP | CP | PP | ETP | EMP |
|---------------|----------------|----|----|----|-----|-----|
| Qwen3-30B-A3B | 16k            | 4  | 2  | 1  | 1   | 4   |
| Qwen3-30B-A3B | 32k            | 4  | 4  | 1  | 1   | 8   |
| Qwen3-30B-A3B | 64k            | 4  | 8  | 1  | 1   | 8   |
| Qwen3-30B-A3B | 128k           | 4  | 8  | 1  | 1   | 8   |
| Qwen3-8B      | 16k            | 2  | 2  | 1  | -   | -   |
| Qwen3-8B      | 32k            | 2  | 4  | 1  | -   | -   |
| Qwen3-8B      | 64k            | 4  | 4  | 1  | -   | -   |
| Qwen3-8B      | 128k           | 8  | 8  | 1  | -   | -   |




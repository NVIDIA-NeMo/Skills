# Model evaluation

Here are the commands you can run to reproduce our evaluation numbers.


## Prepare evaluation data

```bash
ns prepare_data comp-math-24-25 hle
```

## For comp-math-24-25
Below are the evaluation commands for three reasoning modes (high, medium, and low), with and without Python TIR.

```python
from nemo_skills.pipeline.cli import eval, wrap_arguments

cluster = 'slurm'
modes = [
    'low',
    'medium',
    'high'
]
max_length = 120000
model_path = '/workspace/final_sft_model'
for mode in modes:
    output_python = f"/workspace/final_sft_model/{mode}/with-python/"
    output_no_python = f"/workspace/final_sft_model/{mode}/no-python/"

    #with Python TIR evaluation
    eval(
        ctx=wrap_arguments(
            f"++inference.tokens_to_generate=120000 "
            f"++inference.temperature=1.0 "
            f"++inference.top_p=1.0 "
            "++max_concurrent_requests=1024 "
            "++prompt_config=gpt-oss/math "
            "++code_tags=gpt-oss "
            "++code_execution=true "
            "++server.code_execution.max_code_executions=100 "
            "++server.enable_soft_fail=True "
            "++inference.endpoint_type=text "
            f"++chat_template_kwargs.reasoning_effort={mode} "
            "++chat_template_kwargs.builtin_tools=[python] "
        ),
        cluster=cluster,
        expname=f"with-python-comp-math",
        model=model_path,
        server_type='vllm',
        server_gpus=8,
        benchmarks="comp-math-24-25:16",
        output_dir=output_python,
        server_args="--async-scheduling",
        with_sandbox=True,
    )

    #without Python TIR evaluation
    eval(
        ctx=wrap_arguments(
            f"++inference.tokens_to_generate=120000 "
            f"++inference.temperature=1.0 "
            f"++inference.top_p=1.0 "
            "++max_concurrent_requests=1024 "
            "++prompt_config=gpt-oss/math "
            "++inference.endpoint_type=text "
            "++server.enable_soft_fail=True "
            f"++chat_template_kwargs.reasoning_effort={mode} "
        ),
        cluster=cluster,
        expname=f"no-python-comp-math",
        model=model_path,
        server_type='vllm',
        server_gpus=8,
        benchmarks="comp-math-24-25:16",
        output_dir=output_no_python,
        server_args="--async-scheduling",
    )
```

## For hle-math
For hle-math it's necessary to run LLM-as-a-judge step to get accurate evaluation results. We use the [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) model as the judge which can be specified as follows.

```python
from nemo_skills.pipeline.cli import eval, wrap_arguments

cluster = 'slurm'
num_gpus = 8
modes = ['low', 'medium', 'high']
num_chunks = {
    'high': 4,
    'medium': 0,
    'low': 0
}

model_path = '/workspace/final_sft_model'
for mode in modes:
    output_python = f"/workspace/final_sft_model/{mode}/with-python/"
    output_no_python = f"/workspace/final_sft_model/{mode}/no-python/"
    eval(
        ctx=wrap_arguments(
            f"++inference.tokens_to_generate=120000 "
            "++inference.temperature=1.0 "
            "++inference.top_p=1.0 "
            "++max_concurrent_requests=1024 "
            "++prompt_config=gpt-oss/math "
            "++code_tags=gpt-oss "
            "++code_execution=true "
            "++server.code_execution.max_code_executions=100 "
            "++server.enable_soft_fail=True "
            "++inference.endpoint_type=text "
            f"++chat_template_kwargs.reasoning_effort={mode} "
            "++chat_template_kwargs.builtin_tools=[python] "
        ),
        cluster=cluster,
        expname=f"with-python-hle-math",
        model=model_path,
        server_type='vllm',
        server_gpus=8,
        benchmarks="hle:4",
        num_chunks=num_chunks[mode],
        judge_model='/workspace/Qwen2.5-32B-Instruct',
        judge_server_type="sglang",
        judge_server_gpus=8,
        extra_judge_args="++inference.tokens_to_generate=4096 ++server.enable_soft_fail=True",
        split='math',
        output_dir=output_python,
        server_args="--async-scheduling",
        with_sandbox=True,
    )

    eval(
        ctx=wrap_arguments(
            f"++inference.tokens_to_generate=120000 "
            "++inference.temperature=1.0 "
            "++inference.top_p=1.0 "
            "++max_concurrent_requests=1024 "
            "++prompt_config=gpt-oss/math "
            "++inference.endpoint_type=text "
            "++server.enable_soft_fail=True "
            f"++chat_template_kwargs.reasoning_effort={mode} "
        ),
        cluster=cluster,
        expname=f"no-python-hle-math",
        model=model_path,
        server_type='vllm',
        server_gpus=8,
        benchmarks="hle:4",
        num_chunks=num_chunks[mode],
        judge_model='/workspace/Qwen2.5-32B-Instruct',
        judge_server_type="sglang",
        judge_server_gpus=8,
        extra_judge_args="++inference.tokens_to_generate=4096 ++server.enable_soft_fail=True",
        split='math',
        output_dir=output_no_python,
        server_args="--async-scheduling",
    )
```
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from enum import Enum
from functools import partial
from pathlib import Path
from typing import List

import typer

from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils import (
    add_task,
    check_mounts,
    get_cluster_config,
    get_exp,
    parse_kwargs,
    resolve_mount_paths,
    run_exp,
)
from nemo_skills.utils import get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


def get_hf_to_trtllm_cmd(
    input_model,
    output_model,
    model_type,
    hf_model_name,
    dtype,
    num_gpus,
    num_nodes,
    calib_dataset,
    calib_size,
    extra_arguments,
    trt_prepare_args,
    trt_reuse_tmp_engine,
):
    dtype = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
        "fp8": "fp8",
    }[dtype]

    tmp_engine_dir = f"{output_model}-tmp-ckpt"

    setup_cmd = "export PYTHONPATH=$PYTHONPATH:/nemo_run/code && cd /nemo_run/code && "

    if dtype == "fp8":
        hf_to_trtllm_cmd = (
            f"python -m nemo_skills.conversion.hf_to_trtllm_quantize "
            f"    --model_dir {input_model} "
            f"    --dtype auto "
            f"    --qformat {dtype} "
            f"    --output_dir {tmp_engine_dir} "
            f"    --calib_size {calib_size} "
            f"    --calib_dataset {calib_dataset} "
            f"    --batch_size 4 "
            f"    --tp_size {num_gpus} "
            f"    --pp_size {num_nodes} "
            f"    {trt_prepare_args} "
        )
        trtllm_build_cmd = (
            f"trtllm-build "
            f"    --checkpoint_dir {tmp_engine_dir} "
            f"    --output_dir {output_model} "
            f"    --gemm_plugin auto "
            f"    --use_paged_context_fmha enable "
            f"    --max_batch_size 512 "
            f"    --max_input_len 4096 "
            f"    --max_seq_len 8192 "
            f"    --max_num_tokens 8192 "
            f"    {extra_arguments} && "
            f"cp {input_model}/tokenizer* {output_model} "
        )
    if trt_reuse_tmp_engine:
        cmd = (
            setup_cmd + f"if [ ! -f {tmp_engine_dir}/config.json ]; then {hf_to_trtllm_cmd}; fi && {trtllm_build_cmd}"
        )
    else:
        cmd = setup_cmd + hf_to_trtllm_cmd + " && " + trtllm_build_cmd

    return cmd


def get_hf_to_megatron_cmd(
    input_model, output_model, model_type, hf_model_name, dtype, num_gpus, num_nodes, extra_arguments
):
    # megatron-lm uses hacky import logic, so would need to copy a lot of files to move conversion on our side
    # for now just assuming it's available in /opt/Megatron-LM in whatever container is used
    cmd = (
        f"export PYTHONPATH=$PYTHONPATH:/opt/Megatron-LM && "
        f"export CUDA_DEVICE_MAX_CONNECTIONS=1 && "
        f"cd /opt/Megatron-LM && "
        f"python tools/checkpoint/convert.py "
        f"    --model-type GPT "
        f"    --loader llama_mistral "
        f"    --model-size llama3 "
        f"    --load-dir {input_model} "
        f"    --saver core "
        f"    --save-dir {output_model} "
        f"    --checkpoint-type hf "
        f"    --tokenizer-model {hf_model_name} "
        f"    --bf16 "
        f"    --target-tensor-parallel-size {num_gpus} "  # TODO: is there a way to not specify this?
        f"    --target-pipeline-parallel-size {num_nodes} "
        f"    {extra_arguments} "
    )

    return cmd


class SupportedTypes(str, Enum):
    llama = "llama"
    qwen = "qwen"
    deepseek_v3 = "deepseek_v3"


class SupportedFormatsTo(str, Enum):
    hf = "hf"
    trtllm = "trtllm"
    megatron = "megatron"


class SupportedFormatsFrom(str, Enum):
    hf = "hf"


class SupportedDtypes(str, Enum):
    bf16 = "bf16"
    fp16 = "fp16"
    fp32 = "fp32"
    fp8 = "fp8"


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def convert(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    input_model: str = typer.Option(...),
    model_type: SupportedTypes = typer.Option(..., help="Type of the model"),
    output_model: str = typer.Option(..., help="Where to put the final model"),
    convert_from: SupportedFormatsFrom = typer.Option(..., help="Format of the input model"),
    convert_to: SupportedFormatsTo = typer.Option(..., help="Format of the output model"),
    trt_prepare_args: str = typer.Option(
        "", help="Arguments to pass to the first step of trtllm conversion (that builds tmp engine)"
    ),
    trt_reuse_tmp_engine: bool = typer.Option(True, help="Whether to reuse the tmp engine for the final conversion"),
    hf_model_name: str = typer.Option(None, help="Name of the model on Hugging Face Hub to convert to/from"),
    dtype: SupportedDtypes = typer.Option("bf16", help="Data type"),
    calib_dataset: str = typer.Option(
        None,
        help="(Required for dtype=fp8) HuggingFace dataset to use for FP8 calibration",
    ),
    calib_size: int = typer.Option(
        4096,
        help="Optional number of samples to use from the calibration dataset (if dtype=fp8)",
    ),
    expname: str = typer.Option("conversion", help="NeMo-Run experiment name"),
    num_nodes: int = typer.Option(1, help="Number of nodes to use"),
    num_gpus: int = typer.Option(..., help="Number of GPUs per node"),
    partition: str = typer.Option(
        None, help="Can specify if need interactive jobs or a specific non-default partition"
    ),
    qos: str = typer.Option(None, help="Specify Slurm QoS, e.g. to request interactive nodes"),
    time_min: str = typer.Option(None, help="If specified, will use as a time-min slurm parameter"),
    mount_paths: str = typer.Option(None, help="Comma separated list of paths to mount on the remote machine"),
    run_after: List[str] = typer.Option(
        None, help="Can specify a list of expnames that need to be completed before this one starts"
    ),
    reuse_code: bool = typer.Option(
        True,
        help="If True, will reuse the code from the provided experiment. "
        "If you use it from Python, by default the code will be re-used from "
        "the last submitted experiment in the current Python session, so set to False to disable "
        "(or provide reuse_code_exp to override).",
    ),
    reuse_code_exp: str = typer.Option(
        None,
        help="If specified, will reuse the code from this experiment. "
        "Can provide an experiment name or an experiment object if running from code.",
    ),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    log_dir: str = typer.Option(None, help="Can specify a custom location for slurm logs."),
    exclusive: bool | None = typer.Option(None, help="If set will add exclusive flag to the slurm job."),
    check_mounted_paths: bool = typer.Option(False, help="Check if mounted paths are available on the remote machine"),
    skip_hf_home_check: bool | None = typer.Option(
        None,
        help="If True, skip checking that HF_HOME env var is defined in the cluster config.",
    ),
    installation_command: str | None = typer.Option(
        None,
        help="An installation command to run before main job. Only affects main task (not server or sandbox). "
        "You can use an arbitrary command here and we will run it on a single rank for each node. "
        "E.g. 'pip install my_package'",
    ),
    dry_run: bool = typer.Option(False, help="If True, will not run the job, but will validate all arguments."),
    sbatch_kwargs: str = typer.Option(
        "",
        help="Additional sbatch kwargs to pass to the job scheduler. Values should be provided as a JSON string or as a `dict` if invoking from code.",
    ),
    _reuse_exp: str = typer.Option(None, help="Internal option to reuse an experiment object.", hidden=True),
    _task_dependencies: List[str] = typer.Option(
        None, help="Internal option to specify task dependencies.", hidden=True
    ),
):
    """Convert a checkpoint from one format to another.

    All extra arguments are passed directly to the underlying conversion script (see their docs).
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = f"{' '.join(ctx.args)}"
    LOG.info("Starting conversion job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    try:
        model_type = model_type.value
        convert_from = convert_from.value
        convert_to = convert_to.value
        dtype = dtype.value
    except AttributeError:
        pass

    # Validate dtype-related requirements
    if dtype == "fp8":
        if not calib_dataset:
            raise ValueError("--calib_dataset is required when dtype is 'fp8'")
        if convert_to != "trtllm":
            raise ValueError("FP8 dtype is only supported when converting to TensorRT LLM (convert_to='trtllm')")

    if convert_to == "trtllm" and dtype != "fp8":
        raise ValueError("Conversion to TensorRT-LLM is no longer needed, please use HF checkpoint directly!")

    if convert_to != "trtllm" and hf_model_name is None:
        raise ValueError("--hf_model_name is required")

    if convert_to in ["hf"] and model_type == "deepseek_v3":
        raise ValueError("Conversion to HF is not yet supported for DeepSeek v3 models")

    if convert_to == "megatron":
        if convert_from != "hf":
            raise ValueError("Conversion to Megatron is only supported from HF models")
        if model_type != "llama":
            raise ValueError("Conversion to Megatron is only supported for Llama models")
        if dtype != "bf16":
            # TODO: that's probably not true, but need to figure out how it's passed
            raise ValueError("Conversion to Megatron is only supported for bf16 models")

    # Prepare cluster config and mount paths
    cluster_config = get_cluster_config(cluster, config_dir)
    cluster_config = resolve_mount_paths(cluster_config, mount_paths)

    if convert_from == "hf" and not input_model.startswith("/"):
        # For HF, we don't need to check the input_model path if the huggingface model name is used
        output_model, log_dir = check_mounts(
            cluster_config,
            log_dir=log_dir,
            mount_map={output_model: "/output_model"},
            check_mounted_paths=check_mounted_paths,
        )

    else:
        input_model, output_model, log_dir = check_mounts(
            cluster_config,
            log_dir=log_dir,
            mount_map={input_model: "/input_model", output_model: "/output_model"},
            check_mounted_paths=check_mounted_paths,
        )

    if log_dir is None:
        log_dir = str(Path(output_model) / "conversion-logs")

    conversion_cmd_map = {
        ("hf", "megatron"): get_hf_to_megatron_cmd,
        ("hf", "trtllm"): partial(
            get_hf_to_trtllm_cmd,
            calib_dataset=calib_dataset,
            calib_size=calib_size,
            trt_prepare_args=trt_prepare_args,
            trt_reuse_tmp_engine=trt_reuse_tmp_engine,
        ),
    }
    container_map = {
        ("hf", "megatron"): cluster_config["containers"]["megatron"],
        ("hf", "trtllm"): cluster_config["containers"]["trtllm"],
    }
    conversion_cmd = conversion_cmd_map[(convert_from, convert_to)](
        input_model=input_model,
        output_model=output_model,
        model_type=model_type,
        hf_model_name=hf_model_name,
        dtype=dtype,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        extra_arguments=extra_arguments,
    )

    with get_exp(expname, cluster_config, _reuse_exp) as exp:
        LOG.info("Launching task with command %s", conversion_cmd)
        prev_task = add_task(
            exp,
            cmd=conversion_cmd,
            task_name=expname,
            log_dir=log_dir,
            container=container_map[(convert_from, convert_to)],
            num_gpus=num_gpus,
            num_nodes=1,  # always running on a single node, might need to change that in the future
            num_tasks=1,
            cluster_config=cluster_config,
            partition=partition,
            run_after=run_after,
            reuse_code=reuse_code,
            reuse_code_exp=reuse_code_exp,
            sbatch_kwargs=parse_kwargs(sbatch_kwargs, exclusive=exclusive, qos=qos, time_min=time_min),
            installation_command=installation_command,
            task_dependencies=_task_dependencies,
            skip_hf_home_check=skip_hf_home_check,
        )
        run_exp(exp, cluster_config, dry_run=dry_run)

    if _reuse_exp:
        return [prev_task]
    return exp


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()

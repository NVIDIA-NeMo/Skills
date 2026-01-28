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
import copy
import hashlib
import logging
import os
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Optional, Union

from nemo_skills.pipeline.utils.cluster import get_tunnel
from nemo_skills.pipeline.utils.mounts import get_unmounted_path
from nemo_skills.pipeline.utils.server import get_free_port
from nemo_skills.utils import get_chunked_filename, get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def normalize_models_config(
    model: Optional[Union[str, List[str]]],
) -> List[str]:
    """
    Normalize model specification to list.

    Handles both scalar and list inputs:
    - CLI (Typer): Converts single values to single-element lists automatically
    - Python API: Accepts both strings and lists

    Args:
        model: Model path(s) - string or list from Python API, list from CLI

    Returns:
        List of model paths

    Raises:
        ValueError: If model is None or empty
    """
    if model is None:
        raise ValueError("Must specify --model")

    # Handle string (Python API with single model)
    if isinstance(model, str):
        return [model]

    # Handle list
    if len(model) == 0:
        raise ValueError("Must specify --model")
    return list(model)


def normalize_parameter(
    param_value: Any,
    num_models: int,
    param_name: str,
) -> List[Any]:
    """
    Normalize a parameter to a per-model list.

    Handles both scalar and list inputs for flexible usage:
    - CLI (Typer): Converts single values to single-element lists automatically
    - Python API: Accepts both scalars and lists directly

    Broadcast logic:
    - Scalar value: Broadcast to all models [value] * num_models
    - Single-element list: Broadcast to all models
    - Multi-element list: Must match num_models exactly

    Args:
        param_value: Parameter value (scalar or list)
        num_models: Number of models
        param_name: Name of parameter (for error messages)

    Returns:
        List of parameter values (one per model)

    Raises:
        ValueError: If list length doesn't match num_models
    """
    if not isinstance(param_value, list):
        return [param_value] * num_models

    if len(param_value) == num_models:
        return list(param_value)

    if len(param_value) == 1:
        return param_value * num_models

    raise ValueError(
        f"Parameter {param_name} has {len(param_value)} values but {num_models} models specified. "
        f"Must be 1 value (broadcast) or {num_models} values (per-model)."
    )


def build_requirements_venv_cmd(requirements: list[str]) -> str:
    req_lines = [str(line).strip() for line in requirements if str(line).strip()]
    requirements_content = "\n".join(req_lines)

    if not requirements_content:
        raise ValueError("Requirements content is empty.")

    requirements_hash = hashlib.sha256(requirements_content.encode("utf-8")).hexdigest()[:12]
    venv_root = Path("/tmp/nemo_skills/venvs")
    venv_dir = venv_root / f"reqs-{requirements_hash}"
    reqs_root = Path("/tmp/nemo_skills/requirements")
    reqs_file = reqs_root / f"reqs-{requirements_hash}.txt"

    venv_root_arg = shlex.quote(str(venv_root))
    venv_dir_arg = shlex.quote(str(venv_dir))
    reqs_root_arg = shlex.quote(str(reqs_root))
    reqs_file_arg = shlex.quote(str(reqs_file))

    return (
        f"REQS_DIR={reqs_root_arg} && REQS_FILE={reqs_file_arg} && "
        'mkdir -p "$REQS_DIR" && '
        'if [ ! -f "$REQS_FILE" ]; then '
        "cat <<'NEMO_SKILLS_REQS_EOF' > \"$REQS_FILE\"\n"
        f"{requirements_content}\n"
        "NEMO_SKILLS_REQS_EOF\n"
        "fi && "
        f"VENV_ROOT={venv_root_arg} && VENV_DIR={venv_dir_arg} && "
        'READY_FILE="$VENV_DIR/.nemo_skills_ready" && '
        'LOCK_DIR="$VENV_DIR.lock" && '
        'mkdir -p "$VENV_ROOT" && '
        'if [ ! -f "$READY_FILE" ]; then '
        '  if mkdir "$LOCK_DIR" 2>/dev/null; then '
        '    if ! uv venv --system-site-packages "$VENV_DIR"; then rmdir "$LOCK_DIR"; exit 1; fi; '
        '    . "$VENV_DIR/bin/activate"; '
        '    if ! uv pip install -r "$REQS_FILE"; then rmdir "$LOCK_DIR"; exit 1; fi; '
        '    touch "$READY_FILE"; '
        '    rmdir "$LOCK_DIR"; '
        "  else "
        '    while [ ! -f "$READY_FILE" ]; do sleep 1; done; '
        "  fi; "
        "fi && "
        '. "$VENV_DIR/bin/activate"'
    )


def get_chunked_rs_filename(
    output_dir: str,
    random_seed: int = None,
    chunk_id: int = None,
) -> str:
    """
    Return a path of the form: {output_dir}/output[-rsSEED][-chunkK].jsonl
    """
    if random_seed is not None:
        base_filename = f"output-rs{random_seed}.jsonl"
    else:
        base_filename = "output.jsonl"

    # If chunking is enabled, add the chunk suffix
    if chunk_id is not None:
        base_filename = get_chunked_filename(chunk_id, base_filename)
    return os.path.join(output_dir, base_filename)


def get_expected_done_files(output_dir, random_seeds, chunk_ids):
    """
    Returns a mapping of (seed, chunk_id) to expected .done file paths
    """
    file_map = {}
    for seed in random_seeds:
        for chunk_id in chunk_ids:
            output_file = get_chunked_rs_filename(output_dir, random_seed=seed, chunk_id=chunk_id)
            file_map[(seed, chunk_id)] = f"{output_file}.done"
    return file_map


def get_remaining_jobs(cluster_config, output_dir, random_seeds, chunk_ids, rerun_done):
    """
    Determines which jobs still need to be run based on missing .done files.
    Returns a mapping from random_seed to list of chunk_ids that need processing.
    """
    if rerun_done:
        return {seed: copy.deepcopy(chunk_ids) for seed in random_seeds}

    status_dir = get_unmounted_path(cluster_config, output_dir)
    expected_files = get_expected_done_files(output_dir, random_seeds, chunk_ids)
    check_commands = []
    for (seed, chunk_id), filepath in expected_files.items():
        unmounted_path = filepath.replace(output_dir, status_dir)
        # Create identifiers that can be parsed from output
        seed_str = "NONE" if seed is None else str(seed)
        chunk_str = "NONE" if chunk_id is None else str(chunk_id)
        check_commands.append(f'if [ ! -f "{unmounted_path}" ]; then echo "MISSING:{seed_str}:{chunk_str}"; fi')

    # Process commands in batches to avoid "Argument list too long" error
    # Use a conservative batch size that works well even with long paths
    batch_size = 30  # Very conservative to handle long file paths

    outputs = []
    total_files = len(check_commands)
    LOG.debug(f"Checking {total_files} files in batches of {batch_size}...")

    for i in range(0, len(check_commands), batch_size):
        batch = check_commands[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(check_commands) + batch_size - 1) // batch_size

        if total_files > 100:  # Show progress for large file sets
            LOG.debug(f"Processing batch {batch_num}/{total_batches}...")

        command = f"bash -c '{'; '.join(batch)}'"

        try:
            if cluster_config["executor"] == "slurm":
                out = get_tunnel(cluster_config).run(command).stdout.strip()
            else:
                out = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE).stdout.decode("utf-8")
            if out:  # Only append non-empty outputs
                outputs.append(out)
        except Exception as e:
            # If even batched commands fail, try processing one by one
            LOG.warning(f"Batch {batch_num} failed: {e}. Falling back to individual file checks.")
            for j, cmd in enumerate(batch):
                single_command = f"bash -c '{cmd}'"
                try:
                    if cluster_config["executor"] == "slurm":
                        out = get_tunnel(cluster_config).run(single_command).stdout.strip()
                    else:
                        out = subprocess.run(
                            single_command, shell=True, check=True, stdout=subprocess.PIPE
                        ).stdout.decode("utf-8")
                    if out:
                        outputs.append(out)
                except Exception as inner_e:
                    error_msg = f"Failed to check file {i + j + 1}/{total_files}: {inner_e}"
                    LOG.error(error_msg)
                    raise RuntimeError(f"{error_msg}. Unable to determine job status reliably.")

    output = "\n".join(outputs)

    # Parse results into a mapping of missing jobs
    missing_jobs = defaultdict(list)
    for line in output.splitlines():
        if line.startswith("MISSING:"):
            _, seed_str, chunk_str = line.split(":")
            seed = None if seed_str == "NONE" else int(seed_str)
            chunk = None if chunk_str == "NONE" else int(chunk_str)
            missing_jobs[seed].append(chunk)

    done_jobs = defaultdict(list)
    for seed, chunk_id in expected_files.keys():
        if chunk_id not in missing_jobs[seed]:
            done_jobs[seed].append(chunk_id)

    done_jobs_str = ", ".join(
        [
            (
                f"{seed}"
                if not any(chunk is not None for chunk in chunks)
                else f"{seed} (chunks: {', '.join(str(chunk) for chunk in chunks if chunk is not None)})"
            )
            for seed, chunks in done_jobs.items()
            if chunks
        ]
    )
    missing_jobs_str = ", ".join(
        [
            (
                f"{seed}"
                if not any(chunk is not None for chunk in chunks)
                else f"{seed} (chunks: {', '.join(str(chunk) for chunk in chunks if chunk is not None)})"
            )
            for seed, chunks in missing_jobs.items()
            if chunks
        ]
    )

    if missing_jobs_str:
        # only printing this if there are some missing and some done
        if done_jobs_str:
            LOG.warning(
                "The following jobs are incomplete and will be launched: seeds %s",
                missing_jobs_str,
            )
            LOG.warning(
                "The following jobs are completed and will be skipped (to override set --rerun_done): seeds %s",
                done_jobs_str,
            )
    else:
        LOG.warning("All jobs are completed. No jobs will be launched (to override set --rerun_done).")

    return missing_jobs


def separate_hydra_args(extra_arguments: str) -> tuple[str, str]:
    """
    Separate Hydra config args (--config-*, --cfg, --info, etc.) and
    other Hydra flags from override args (++*).

    Args:
        extra_arguments: String containing mixed Hydra and override arguments

    Returns:
        Tuple of (hydra_config_args, override_args) as strings

    Examples:
        Empty input:
            separate_hydra_args("") -> ("", "")

        With config path/name and overrides:
            separate_hydra_args(
                "--config-path /cfg --config-name my_cfg ++inference.temperature=0.7"
            ) -> (
                " --config-path /cfg --config-name my_cfg",
                " ++inference.temperature=0.7",
            )

        With Hydra flags that take values (space and equals forms):
            separate_hydra_args("--cfg all --info=plugins ++x=1") -> (
                " --cfg all --info=plugins",
                " ++x=1",
            )

        With Hydra flags without values:
            separate_hydra_args("--run --multirun ++y=2") -> (
                " --run --multirun",
                " ++y=2",
            )

        Mixed value formats and directories:
            separate_hydra_args(
                "--config-path=/a --config-dir /b ++foo=bar"
            ) -> (
                " --config-path=/a --config-dir /b",
                " ++foo=bar",
            )
    """
    hydra_config_args = ""
    override_args = ""

    if not extra_arguments:
        return hydra_config_args, override_args

    # Hydra flags that do not take a value
    hydra_flags_no_value = {
        "--help",
        "--hydra-help",
        "--version",
        "--resolve",
        "--run",
        "--multirun",
        "--shell-completion",
    }

    # Hydra flags that take a value (accept both --flag=value and "--flag value")
    hydra_flags_with_value = {
        "--config-path",
        "--config-name",
        "--config-dir",
        "--package",
        "--experimental-rerun",
        "--cfg",
        "--info",
    }

    args_parts = shlex.split(extra_arguments)
    i = 0
    while i < len(args_parts):
        arg = args_parts[i]

        # Exact match for no-value hydra flags
        if arg in hydra_flags_no_value:
            hydra_config_args += f" {arg}"
            i += 1
            continue

        # Match hydra flags that accept values (either --flag=value or "--flag value")
        is_with_value_flag = False
        for flag in hydra_flags_with_value:
            if arg.startswith(flag):
                is_with_value_flag = True
                # Handle both formats
                if "=" in arg:
                    hydra_config_args += f" {arg}"
                else:
                    hydra_config_args += f" {arg}"
                    if i + 1 < len(args_parts) and not args_parts[i + 1].startswith("-"):
                        i += 1
                        hydra_config_args += f" {args_parts[i]}"
                break

        if not is_with_value_flag:
            # Not a recognized hydra flag → treat as override
            override_args += f" {arg}"

        i += 1

    return hydra_config_args, override_args


def get_generation_cmd(
    output_dir,
    input_file=None,
    input_dir=None,
    extra_arguments="",
    random_seed=None,
    chunk_id=None,
    num_chunks=None,
    preprocess_cmd=None,
    postprocess_cmd=None,
    wandb_parameters=None,
    with_sandbox: bool = False,
    script: str = "nemo_skills.inference.generate",
    requirements: Optional[list[str]] = None,
    # Optional: for multi-model generation
    server_addresses: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    server_types: Optional[List[str]] = None,
):
    """Construct the generation command for language model inference.

    Supports both single-model and multi-model generation. For multi-model:
    - server_addresses: List of server addresses (one per model)
    - model_names: List of model names (one per model)
    - server_types: List of server types (one per model)

    For single-model, server config is passed via extra_arguments.
    If requirements are provided, a per-requirements uv venv is prepared
    and activated before running the generation command.
    """
    if input_file is None and input_dir is None:
        raise ValueError("Either input_file or input_dir must be provided.")
    if input_file is not None and input_dir is not None:
        raise ValueError("Please provide either input_file or input_dir, not both.")

    # in this case we are running on the output of another generate command
    # and doing 1-1 mapping of random seeds
    if input_dir is not None:
        if random_seed is None:
            raise ValueError("If input_dir is provided, random_seed must also be specified.")
        input_file = f"{input_dir}/output-rs{random_seed}.jsonl"

    # First get the unchunked filename for the output file
    output_file = get_chunked_rs_filename(
        output_dir=output_dir,
        random_seed=random_seed,
    )
    # Preamble for generation commands: added at executor/declarative level
    cmd = "export HYDRA_FULL_ERROR=1 && "

    # Separate Hydra config args (--config-*) from override args (++)
    hydra_config_args, override_args = separate_hydra_args(extra_arguments)

    # Handle file paths vs module names
    common_args = f"++skip_filled=True ++input_file={input_file} ++output_file={output_file}"
    if script.endswith(".py") or os.sep in script:
        # It's a file path, run it directly with .py extension
        script_path = script if script.endswith(".py") else f"{script}.py"
        cmd += f"python {script_path} {hydra_config_args} {common_args} "
    else:
        # It's a module name, use -m flag
        cmd += f"python -m {script} {hydra_config_args} {common_args} "

    # Add multi-model configuration if provided
    if server_addresses is not None and model_names is not None:
        num_models = len(model_names)
        if num_models > 1:
            # Multi-model: pass server configuration as lists
            model_names_arg = ",".join(model_names)
            cmd += f"++server.model=[{model_names_arg}] "

            server_types_arg = ",".join(server_types)
            cmd += f"++server.server_type=[{server_types_arg}] "

            server_addresses_arg = ",".join(server_addresses)
            cmd += f"++server.base_url=[{server_addresses_arg}] "

    job_end_cmd = ""

    if random_seed is not None and input_dir is None:  # if input_dir is not None, we default to greedy generations
        cmd += (
            f"    ++inference.random_seed={random_seed} "
            f"    ++inference.temperature=0.7 "
            f"    ++inference.top_k=-1 "
            f"    ++inference.top_p=0.95 "
        )

    if with_sandbox:
        cmd += "++wait_for_sandbox=true "

    if chunk_id is not None:
        cmd += f" ++num_chunks={num_chunks} ++chunk_id={chunk_id} "
        output_file = get_chunked_rs_filename(output_dir, random_seed=random_seed, chunk_id=chunk_id)
        donefiles = []
        # we are always waiting for all chunks in num_chunks, no matter chunk_ids in
        # the current run (as we don't want to merge partial jobs)
        for cur_chunk_id in range(num_chunks):
            filename = get_chunked_rs_filename(output_dir=output_dir, random_seed=random_seed, chunk_id=cur_chunk_id)
            donefile = f"{filename}.done"
            donefiles.append(donefile)

        if job_end_cmd:
            job_end_cmd += f" && touch {donefiles[chunk_id]} "
        else:
            job_end_cmd = f"touch {donefiles[chunk_id]} "

        # getting file name as if there is no chunking since that's where we want to merge
        merged_output_file = get_chunked_rs_filename(output_dir=output_dir, random_seed=random_seed)
        merge_cmd = (
            f"python -m nemo_skills.inference.merge_chunks {merged_output_file} "
            f"{' '.join([f[:-5] for f in donefiles])}"
        )
        if postprocess_cmd:
            postprocess_cmd = shlex.quote(postprocess_cmd)
            merge_cmd = f"{merge_cmd} -- {postprocess_cmd}"
        postprocess_cmd = f"{job_end_cmd} && {merge_cmd}"

    else:  # only writing a single status file
        if job_end_cmd:
            job_end_cmd += f" && touch {output_file}.done "
        else:
            job_end_cmd = f"touch {output_file}.done "

        if postprocess_cmd:
            postprocess_cmd = f"{job_end_cmd} && {postprocess_cmd}"
        else:
            postprocess_cmd = job_end_cmd

    if override_args:
        cmd += f" {override_args} "

    if requirements:
        requirements_cmd = build_requirements_venv_cmd(requirements)
        preprocess_cmd = f"{requirements_cmd} && {preprocess_cmd}" if preprocess_cmd else requirements_cmd

    return wrap_cmd(
        cmd=cmd,
        preprocess_cmd=preprocess_cmd,
        postprocess_cmd=postprocess_cmd,
        random_seed=random_seed,
        wandb_parameters=wandb_parameters,
    )


def wrap_cmd(cmd, preprocess_cmd, postprocess_cmd, random_seed=None, wandb_parameters=None):
    if preprocess_cmd:
        if random_seed is not None:
            preprocess_cmd = preprocess_cmd.format(random_seed=random_seed)
        cmd = f" {preprocess_cmd} && {cmd} "
    if postprocess_cmd:
        if random_seed is not None:
            postprocess_cmd = postprocess_cmd.format(random_seed=random_seed)
        cmd = f" {cmd} && {postprocess_cmd} "
    if wandb_parameters:
        log_wandb_cmd = (
            f"python -m nemo_skills.inference.log_samples_wandb "
            f"    {wandb_parameters['samples_file']} "
            f"    --name={wandb_parameters['name']} "
            f"    --project={wandb_parameters['project']} "
        )
        if wandb_parameters["group"] is not None:
            log_wandb_cmd += f" --group={wandb_parameters['group']} "
        cmd = f"{cmd} && {log_wandb_cmd} "
    return cmd


def configure_client(
    *,  # Force keyword arguments
    model: str,
    server_type: str,
    server_gpus: int,
    server_nodes: int,
    server_address: str,
    server_args: str,
    server_entrypoint: str | None,
    get_random_port: bool,
    extra_arguments: str,
    server_container: str | None = None,
):
    """
    Utility function to configure a client for the model inference server.

    Args:
        model: Mounted Path to the model to evaluate.
        server_type: String name of the server type.
        server_address: URL of the server hosting the model.
        server_gpus: Number of GPUs to use for the server.
        server_nodes: Number of nodes to use for the server.
        server_args: Additional arguments for the server.
        server_entrypoint: Entry point for the server command (will use default if None).
        get_random_port: Whether to get a random port for the server.
        extra_arguments: Extra arguments to pass to the command.
        server_container: Container to use for the server.

    Returns:
        A tuple containing:
            - server_config: Configuration for the server.
            - server_address: Address of the server.
            - extra_arguments: Updated extra arguments for the command.
    """
    # Check if user already specified server.server_type in extra_arguments
    server_type_arg = "" if "++server.server_type=" in extra_arguments else f"++server.server_type={server_type} "
    # Only add server_type if user didn't specify it (allows vllm_multimodal override)
    extra_arguments = server_type_arg + extra_arguments

    if server_gpus:  # we need to host the model
        server_port = get_free_port(strategy="random") if get_random_port else 5000
        assert server_gpus is not None, "Need to specify server_gpus if hosting the model"
        server_address = f"localhost:{server_port}"

        server_config = {
            "model_path": model,
            "server_type": server_type,
            "num_gpus": server_gpus,
            "num_nodes": server_nodes,
            "server_args": server_args,
            "server_entrypoint": server_entrypoint,
            "server_port": server_port,
        }
        if server_container:
            server_config["container"] = server_container
        extra_arguments = (
            f"++server.host=127.0.0.1 ++server.port={server_port} ++server.model={model} {extra_arguments}"
        )
    else:  # model is hosted elsewhere
        server_config = None
        extra_arguments = f"++server.base_url={server_address} ++server.model={model} {extra_arguments}"
    return server_config, server_address, extra_arguments

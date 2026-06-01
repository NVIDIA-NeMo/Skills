# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Gym-backend dispatcher for `ns eval --backend=gym`.

This module is a self-contained alternative to the Skills-shaped flow in
`eval.py`. The Skills dispatcher's pipeline lookup (`prepare_eval_commands`)
walks `nemo_skills/dataset/<bench>/__init__.py` to derive defaults — that's
fine for benchmarks that exist in both projects, but it's a hard dependency
for the Skills-only path. Gym-only benchmarks have no Skills dataset module
and can only ship through this dispatcher.

Design rationale: keeping the two paths separated rather than threading
`if backend == gym` branches into Skills' command-prep helpers means:

* Skills' code paths stay unmodified while the dual-backend window is open
  (no risk of Skills regressions from Gym-shaped patches).
* Gym-only benchmarks need exactly one new file to register: an entry in
  `pipeline/utils/gym/registry.py`. No shim dataset modules.
* When Skills backend is sunset, we delete `_eval_skills` + the Skills
  dataset-module loader + the translator wholesale. The Gym dispatcher
  stays as-is; no untangling of accumulated `if backend == gym` special
  cases.
* Errors at the boundary use Gym vocabulary (registry entry / input file /
  resource server) rather than Skills vocabulary leaking through ("EVAL_SPLIT
  not found in dataset module ...").
"""

from __future__ import annotations

import difflib
import logging
import os
import subprocess
from typing import List, Optional

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.pipeline.utils.declarative import (
    Command,
    CommandGroup,
    HardwareConfig,
    Pipeline,
)
from nemo_skills.pipeline.utils.gym import (
    GymBenchmarkConfig,
    get_gym_config,
    is_registered,
    registered_benchmarks,
)
from nemo_skills.pipeline.utils.scripts import GymEvalClientScript, SandboxScript, ServerScript

LOG = logging.getLogger(__name__)

# Default file inside the Gym install dir where `ng_collect_rollouts` reads
# its input. Resolved on the cluster to `${GYM_PATH}/<input_jsonl_fpath>`.
_DEFAULT_GYM_PATH = "/opt/Gym"


def _did_you_mean(name: str) -> Optional[str]:
    """Suggest the closest registered benchmark name for a typo'd input."""
    matches = difflib.get_close_matches(name, registered_benchmarks(), n=1, cutoff=0.6)
    return matches[0] if matches else None


def _validate_benchmarks(requested: List[str]) -> List[str]:
    """Validate every requested benchmark is in the Gym registry.

    Raises with a clear message + "did you mean" suggestion for typos.
    Returns the canonical list (stripped of the optional `:N` suffix that
    `prepare_eval_commands` uses for per-benchmark num-samples overrides
    — `_eval_gym` honors that via Gym's `+num_repeats=N` instead).
    """
    canonical = []
    missing = []
    for raw in requested:
        name = raw.split(":", 1)[0]
        if is_registered(name):
            canonical.append(name)
        else:
            missing.append(name)
    if missing:
        hint = ""
        for m in missing:
            close = _did_you_mean(m)
            if close:
                hint += f"\n  '{m}' — did you mean '{close}'?"
        raise ValueError(
            f"--backend=gym does not yet support benchmark(s): {missing}."
            f"{hint}\n"
            f"Registered benchmarks ({len(registered_benchmarks())}): "
            f"{registered_benchmarks()}.\n"
            f"For Skills-only benchmarks: re-run with --backend=skills. "
            f"To add a new Gym benchmark: extend "
            f"nemo_skills/pipeline/utils/gym/registry.py."
        )
    return canonical


def _preflight_input_file(cluster_config: dict, cfg: GymBenchmarkConfig, benchmark: str) -> None:
    """Cheap pre-submit check that the Gym input JSONL exists on the cluster.

    Skips when:
    - executor is "none" / "local" (no remote filesystem to check)
    - the user hasn't asked us to verify mounts (`check_mounted_paths=False`)
    - we can't resolve the mounted /opt/Gym path back to a host path

    The check is best-effort; the SLURM job's `ng_collect_rollouts` will
    surface a clean "file not found" if the preflight is skipped or fails
    silently.
    """
    if cluster_config.get("executor") in (None, "none", "local"):
        return
    # Resolve /opt/Gym on the host. Look through the cluster_config mounts for
    # a "<host_path>:/opt/Gym" entry; fall back to host==/opt/Gym (which only
    # works on the executor node itself).
    host_gym = None
    for m in cluster_config.get("mounts", []):
        if isinstance(m, str) and m.endswith(":/opt/Gym"):
            host_gym = m.split(":", 1)[0]
            break
    if host_gym is None:
        LOG.debug("Skipping Gym input-file preflight: no /opt/Gym mount in cluster_config.")
        return
    host_input = f"{host_gym}/{cfg.input_jsonl_fpath}"
    tunnel = cluster_config.get("ssh_tunnel") or {}
    if not tunnel.get("host"):
        return
    ssh_cmd = [
        "ssh",
        "-i",
        tunnel.get("identity", os.path.expanduser("~/.ssh/id_ed25519")),
        "-o",
        "BatchMode=yes",
        f"{tunnel['user']}@{tunnel['host']}",
        f"test -f {host_input}",
    ]
    try:
        subprocess.check_call(ssh_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, OSError) as exc:
        raise RuntimeError(
            f"Gym input file missing on cluster for benchmark '{benchmark}':\n"
            f"  {host_input}\n\n"
            f"Bootstrap with:\n"
            f"  ssh {tunnel['user']}@{tunnel['host']} "
            f"'cd {host_gym} && ng_prepare_benchmark "
            f'"+config_paths=[{",".join(cfg.config_paths)}]" '
            f"+use_cached_prepared_benchmarks=true'"
        ) from exc


def _build_gym_units(
    output_dir: str,
    starting_seed: int,
    num_random_seeds: int,
    extra_arguments: str,
    wandb_parameters: Optional[dict],
    requires_sandbox: bool,
) -> List[dict]:
    """One unit per random seed. `GymEvalClientScript` invokes
    `ng_collect_rollouts` once per unit, writing `rollouts-rs<seed>.jsonl`.
    """
    units = []
    for seed_idx in range(num_random_seeds or 1):
        seed = starting_seed + seed_idx if num_random_seeds else None
        units.append(
            {
                # `input_file` is consumed by Skills' generation flow; for
                # Gym it's ignored — `GymEvalClientScript` uses
                # `gym_input_jsonl_fpath` from the registry instead. The key
                # must be present to satisfy the dataclass schema.
                "input_file": "",
                "output_dir": output_dir,
                "extra_arguments": extra_arguments,
                "random_seed": seed,
                "chunk_id": None,
                "num_chunks": None,
                "script": None,
                "requirements": None,
                "wandb_parameters": wandb_parameters,
                "with_sandbox": requires_sandbox,
            }
        )
    return units


def eval_gym(
    *,
    ctx,
    cluster: Optional[str],
    output_dir: str,
    expname: str,
    benchmarks: str,
    model: str,
    server_type: str,
    server_address: Optional[str],
    server_gpus: Optional[int],
    server_nodes: Optional[int],
    server_args: Optional[str],
    server_entrypoint: Optional[str],
    server_container: Optional[str],
    partition: Optional[str],
    account: Optional[str],
    log_dir: Optional[str],
    starting_seed: int,
    num_random_seeds: int,
    extra_arguments: str,
    wandb_parameters: Optional[dict],
    single_node_mode: bool,
    with_sandbox: bool,
    keep_mounts_for_sandbox: bool,
    sandbox_container: Optional[str],
    sandbox_mounts: Optional[List[str]],
    main_container: Optional[str],
    mount_paths: Optional[List[str]],
    check_mounted_paths: bool,
    config_dir: Optional[str],
    run_after: Optional[List[str]],
    dependent_jobs: int,
    sbatch_kwargs: dict,
    installation_command: Optional[str],
    reuse_code: bool,
    reuse_code_exp: Optional[str],
    skip_hf_home_check: bool,
    dry_run: bool,
    _reuse_exp=None,
    _task_dependencies: Optional[list] = None,
):
    """Submit a Gym-backend eval pipeline.

    Mirrors the public surface of `nemo_skills.pipeline.eval.eval` for the
    arguments that have a meaningful interpretation under
    `--backend=gym`. Arguments that don't apply (judge_*, log_samples,
    extra_benchmark_map, generation_module, …) are intentionally not
    accepted: they're either Skills-specific or handled by the Gym
    resource_server itself.
    """
    # ------------------------------------------------------------------ #
    # 1. Registry validation                                              #
    # ------------------------------------------------------------------ #
    if " " in str(benchmarks):
        raise ValueError("benchmarks should be separated with commas")
    requested = [b for b in benchmarks.split(",") if b]
    benchmark_names = _validate_benchmarks(requested)

    # ------------------------------------------------------------------ #
    # 2. Cluster / mount resolution (mirrors `eval()` for shared helpers) #
    # ------------------------------------------------------------------ #
    cluster_config = pipeline_utils.get_cluster_config(cluster, config_dir)
    cluster_config = pipeline_utils.resolve_mount_paths(
        cluster_config, mount_paths, create_remote_dir=check_mounted_paths
    )

    if log_dir is None:
        log_dir = f"{output_dir}/eval-logs"

    # `check_mounts` returns `*new_paths, log_dir` as a flat tuple. We pass
    # exactly one mount entry (output_dir), so we get a 2-tuple back.
    output_dir, log_dir = pipeline_utils.check_mounts(
        cluster_config,
        log_dir=log_dir,
        mount_map={output_dir: None},
        check_mounted_paths=check_mounted_paths,
    )

    # ------------------------------------------------------------------ #
    # 3. Model server normalization                                       #
    # ------------------------------------------------------------------ #
    models_list = pipeline_utils.normalize_models_config(model)
    num_models = len(models_list)
    server_types_list = pipeline_utils.normalize_parameter(server_type, num_models, "server_type")
    server_gpus_list = pipeline_utils.normalize_parameter(server_gpus, num_models, "server_gpus")
    server_nodes_list = pipeline_utils.normalize_parameter(server_nodes, num_models, "server_nodes")
    server_args_list = pipeline_utils.normalize_parameter(server_args, num_models, "server_args")
    server_entrypoints_list = pipeline_utils.normalize_parameter(server_entrypoint, num_models, "server_entrypoint")
    server_containers_list = pipeline_utils.normalize_parameter(server_container, num_models, "server_container")
    server_addresses_list = (
        pipeline_utils.normalize_parameter(server_address, num_models, "server_address")
        if server_address is not None
        else [None] * num_models
    )
    for model_idx in range(num_models):
        if not (server_gpus_list[model_idx] is not None and int(server_gpus_list[model_idx] or 0) > 0):
            if not server_addresses_list[model_idx]:
                raise ValueError(
                    f"Model {model_idx} is not self-hosted (server_gpus=0/None) but server_address is missing. "
                    "Please provide --server-address (one per model, or a single value to broadcast)."
                )

    # ------------------------------------------------------------------ #
    # 4. Per-benchmark preflight                                          #
    # ------------------------------------------------------------------ #
    for benchmark in benchmark_names:
        cfg = get_gym_config(benchmark)
        if check_mounted_paths:
            _preflight_input_file(cluster_config, cfg, benchmark)

    sequential = cluster_config["executor"] in ["local", "none"]

    # ------------------------------------------------------------------ #
    # 5. Build one SLURM job per benchmark                                #
    # ------------------------------------------------------------------ #
    jobs: list[dict] = []
    job_names: list[str] = []
    if _task_dependencies is None:
        _task_dependencies = []

    with pipeline_utils.get_exp(expname, cluster_config, _reuse_exp) as exp:
        for bench_idx, benchmark in enumerate(benchmark_names):
            cfg = get_gym_config(benchmark)
            task_name = f"{expname}-job{bench_idx}-{benchmark}"

            # One unit per random seed. `GymEvalClientScript` runs
            # `ng_collect_rollouts` per unit, writing `rollouts-rs<seed>.jsonl`.
            units = _build_gym_units(
                output_dir=f"{output_dir}/eval-results/{benchmark}",
                starting_seed=starting_seed,
                num_random_seeds=num_random_seeds,
                extra_arguments=extra_arguments or "",
                wandb_parameters=wandb_parameters,
                requires_sandbox=cfg.requires_sandbox or with_sandbox,
            )

            # Build ServerScripts for self-hosted models.
            server_scripts: list[ServerScript | None] = []
            for model_idx in range(num_models):
                if server_gpus_list[model_idx] is not None and int(server_gpus_list[model_idx] or 0) > 0:
                    server_scripts.append(
                        ServerScript(
                            server_type=server_types_list[model_idx],
                            model_path=models_list[model_idx],
                            cluster_config=cluster_config,
                            num_gpus=server_gpus_list[model_idx],
                            num_nodes=server_nodes_list[model_idx],
                            server_args=server_args_list[model_idx] or "",
                            server_entrypoint=server_entrypoints_list[model_idx],
                            port=None,
                            allocate_port=True,
                        )
                    )
                else:
                    server_scripts.append(None)

            # Sandbox per benchmark — auto-on for benchmarks that declare
            # requires_sandbox=True in the registry, or globally via the
            # --with_sandbox CLI flag.
            sandbox_enabled = (cfg.requires_sandbox or with_sandbox) is True
            sandbox_script = None
            if sandbox_enabled:
                env_overrides = list(cfg.sandbox_env_vars or ())
                sandbox_script = SandboxScript(
                    cluster_config=cluster_config,
                    keep_mounts=keep_mounts_for_sandbox,
                    allocate_port=True,
                    env_overrides=env_overrides,
                )
                sandbox_script.span_group_nodes = True

            # Build the Gym client script.
            client_script = GymEvalClientScript(
                units=units,
                config_paths=list(cfg.config_paths),
                agent_name=cfg.agent_name,
                gym_input_jsonl_fpath=cfg.input_jsonl_fpath,
                gym_prompt_config=cfg.prompt_config,
                extra_overrides=tuple(cfg.extra_overrides),
                single_node_mode=single_node_mode,
                with_sandbox=sandbox_enabled,
                servers=server_scripts,
                server_addresses_prehosted=server_addresses_list,
                model_names=models_list,
                server_types=server_types_list,
                sandbox=sandbox_script,
                installation_command=installation_command,
            )

            # Group 0: (optional server0) + (optional sandbox) + client.
            group0_components = []
            group0_server = server_scripts[0] if server_scripts else None
            group_gpus = 0
            group_nodes = 1
            group_tasks = 1
            if group0_server is not None:
                group0_components.append(
                    Command(
                        script=group0_server,
                        container=server_containers_list[0] or cluster_config["containers"][server_types_list[0]],
                        name=f"{task_name}_model_0_server",
                    )
                )
                group_gpus = int(server_gpus_list[0])
                group_nodes = int(server_nodes_list[0])
                group_tasks = int(group0_server.num_tasks)

            if sandbox_script is not None:
                group0_components.append(
                    Command(
                        script=sandbox_script,
                        container=sandbox_container or cluster_config["containers"]["sandbox"],
                        name=f"{task_name}_sandbox",
                        mounts=sandbox_mounts,
                    )
                )

            client_container = (
                main_container
                or cluster_config["containers"].get("nemo-gym")
                or cluster_config["containers"]["nemo-rl"]
            )
            group0_components.append(
                Command(
                    script=client_script,
                    container=client_container,
                    name=f"{task_name}",
                )
            )

            groups = [
                CommandGroup(
                    commands=group0_components,
                    hardware=HardwareConfig(
                        partition=partition,
                        account=account,
                        num_gpus=group_gpus,
                        num_nodes=group_nodes,
                        num_tasks=group_tasks,
                        sbatch_kwargs=sbatch_kwargs,
                    ),
                    name=f"{task_name}_group0",
                    log_dir=log_dir,
                )
            ]

            # Extra groups for additional hosted models.
            for model_idx in range(1, num_models):
                srv = server_scripts[model_idx]
                if srv is None:
                    continue
                groups.append(
                    CommandGroup(
                        commands=[
                            Command(
                                script=srv,
                                container=server_containers_list[model_idx]
                                or cluster_config["containers"][server_types_list[model_idx]],
                                name=f"{task_name}_model_{model_idx}_server",
                            )
                        ],
                        hardware=HardwareConfig(
                            partition=partition,
                            account=account,
                            num_gpus=int(server_gpus_list[model_idx]),
                            num_nodes=int(server_nodes_list[model_idx]),
                            num_tasks=int(srv.num_tasks),
                            sbatch_kwargs=sbatch_kwargs,
                        ),
                        name=f"{task_name}_model_{model_idx}_group",
                        log_dir=log_dir,
                    )
                )

            base_deps = list(_task_dependencies)
            if run_after:
                base_deps.extend(run_after if isinstance(run_after, list) else [run_after])

            prev_job = None
            for dep_idx in range(dependent_jobs + 1):
                internal_job_name = f"{task_name}-dep{dep_idx}" if dep_idx > 0 else task_name
                job_deps = base_deps if dep_idx == 0 and base_deps else ([prev_job] if dep_idx > 0 else None)
                job_spec = {"name": internal_job_name, "dependencies": job_deps}
                if len(groups) > 1:
                    job_spec["groups"] = groups
                else:
                    job_spec["group"] = groups[0]
                jobs.append(job_spec)
                job_names.append(internal_job_name)
                prev_job = job_spec

        if not jobs:
            return None

        pipeline = Pipeline(
            name=expname,
            cluster_config=cluster_config,
            jobs=jobs,
            reuse_code=reuse_code,
            reuse_code_exp=reuse_code_exp,
            skip_hf_home_check=skip_hf_home_check,
        )
        # Pipeline.run with `_reuse_exp` only registers tasks on the
        # experiment; the caller is responsible for the actual SLURM submit.
        # Matches the pattern in `eval.py` line 1086 (`pipeline_utils.run_exp`
        # at end of `with get_exp(...)` block).
        handles = pipeline.run(dry_run=dry_run, _reuse_exp=exp, sequential=sequential)
        pipeline_utils.run_exp(exp, cluster_config, dry_run=dry_run)
        return handles if _reuse_exp else None

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

"""In-process vLLM launcher with the DP-on-Ray placement-group patch.

Identical interface to ``serve_vllm.py`` but starts vLLM's OpenAI API server
in the current Python process (not via subprocess), so the monkey-patches
applied before ``run_server()`` take effect inside the Ray DP coordination
path.

Motivation
----------
vLLM's ``CoreEngineActorManager.create_dp_placement_groups`` auto-discovers
every node registered with the Ray cluster and creates one placement group
per node, then asserts ``len(pgs) == dp_size``. That fails whenever a Ray
cluster has more nodes than DP replicas — for example, a 5-node allocation
running DP=4 with the 5th node reserved for a neural-eval Ray task.

NeMo Gym's ``LocalVLLMModel`` already carries the fix as two in-process
monkey-patches (``_patch_create_dp_placement_groups`` and
``_patch_init_data_parallel``). This wrapper ports them so pipelines that
serve vLLM via ``nemo_skills.inference.server`` (``nemo_gym_rollouts``,
``generate``, ``eval``, ...) can use the extra-node topology too.

The patches are intended to be temporary — see upstream vLLM issue
https://github.com/vllm-project/vllm/issues/32400 ("Ray cluster (5 Node),
Deploy DP=2, Creating too many placement groups") for the discussion of
the long-term fix.

Caveats and version coupling (vLLM 0.18.1)
-------------------------------------------
This file is tightly coupled to vLLM's internal placement-group code.
Symbols imported from ``vllm.v1.engine.utils`` (``CoreEngineActorManager``,
``current_platform``) and ``vllm.v1.engine.core`` (``DPEngineCoreProc``)
are private and have shifted between vLLM minor releases. Last validated
against vLLM 0.18.1 with the wmt24pp_gym_topology slurm test (5
topologies × {Nemotron-3-Nano, DeepSeek-V2-Lite}). After any vLLM bump:
  * Re-run ``tests/slurm-tests/wmt24pp_gym_topology`` end-to-end — it
    catches every regression mode we've seen (engine init, head-PG
    layout, hung api_server, lost extra_gpu node, missed per-row scores).
  * Re-read ``new_create_dp_placement_groups`` against the current vLLM
    version of the same function — the patch shadows the original, so
    silent drift won't fail loudly until a topology that exercises the
    diff hits production.

Coordination with ``get_ray_server_cmd``
-----------------------------------------
The "extra_gpu" trick this file relies on is implemented in
``nemo_skills/pipeline/utils/server.py:get_ray_server_cmd``: nodes beyond
``serving_nodes`` join Ray with ``--num-gpus=0
--resources='{"extra_gpu": <gpus_per_node>}'``. Their physical GPUs are
hidden from Ray's normal accounting (so vLLM's compiled-DAG node scan
doesn't trip on them) but are reachable to Ray tasks that explicitly
request ``resources={"extra_gpu": 1}`` plus the
``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`` env flag. If that
launch contract changes (e.g. someone drops ``--num-gpus=0``), the COMET
actor pool silently never schedules and downstream resource servers fall
back to end-of-batch dispatch — the wmt24pp slurm test catches this as
``0/N rows with comet_score``.

Private Ray API
---------------
``ray._private.services.get_node_ip_address`` (used to set
``VLLM_DP_MASTER_IP``) is a private import and could be moved or removed
in any Ray release. Both the DP=1 fast path and ``_reserve_head_placement_group``
use it — keep them in lockstep so they break together if Ray renames it.

Usage
-----
Drop-in replacement for ``serve_vllm.py``. Select via
``server_type=vllm_dp_ray`` when configuring a NeMo Skills pipeline.
Accepts the same ``--model``, ``--num_gpus``, ``--num_nodes``, ``--port``
arguments plus a passthrough of any extra vLLM args.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import signal
import sys
from typing import Sequence


def _apply_vllm_patches() -> None:
    """Monkey-patch vLLM's DP placement-group + DP-init logic.

    Mirrors ``LocalVLLMModelActor._patch_create_dp_placement_groups`` and
    ``_patch_init_data_parallel`` from NeMo Gym, adapted for the in-process
    serve path (no Ray actor wrapper).

    Must be called AFTER ``ray.init(...)`` connects to the cluster and
    AFTER the head placement group has been reserved.
    """
    import ray
    from ray.util.placement_group import PlacementGroup
    from vllm.v1.engine.core import DPEngineCoreProc
    from vllm.v1.engine.core import logger as dp_logger
    from vllm.v1.engine.utils import (
        CoreEngineActorManager,
        current_platform,
        envs,
        logger,
    )

    head_pg: PlacementGroup = _HEAD_PG_REF["pg"]
    server_name: str = _HEAD_PG_REF["server_name"]

    # --- create_dp_placement_groups ------------------------------------------------

    def new_create_dp_placement_groups(vllm_config):
        from ray._private.state import available_resources_per_node, total_resources_per_node

        logger.info("Creating placement groups for data parallel (patched)")
        dp_master_ip = vllm_config.parallel_config.data_parallel_master_ip
        dp_size = vllm_config.parallel_config.data_parallel_size
        dp_size_local = vllm_config.parallel_config.data_parallel_size_local

        if dp_size_local != 1:
            raise NotImplementedError(
                f"serve_vllm_dp_ray does not support data-parallel-size-local={dp_size_local}. "
                "Each DP replica must occupy its own node(s); set --data-parallel-size-local=1."
            )

        available_resources = available_resources_per_node()

        # Strip per-PG synthetic resource keys (e.g. "node:<ip>_group_<hex>")
        # that Ray adds once placement groups exist, so the assertion
        # `len(node_ip_keys) == 1` downstream stays valid.
        for node_hex_id, node_resources in list(available_resources.items()):
            available_resources[node_hex_id] = {
                resource_id: resource
                for resource_id, resource in node_resources.items()
                if "_group_" not in resource_id
            }

        world_size = vllm_config.parallel_config.world_size

        # Start with the pre-reserved head PG (DP rank 0) instead of
        # auto-discovering every Ray node.
        placement_groups: list[PlacementGroup] = [head_pg]
        local_dp_ranks: list[int] = [0]

        dp_master_ip_key = f"node:{dp_master_ip}"
        nodes = sorted(available_resources.values(), key=lambda x: dp_master_ip_key not in x)
        assert len(nodes) > 0, "No nodes with resources found in Ray cluster."
        assert dp_master_ip_key in nodes[0], f"The DP master node (ip: {dp_master_ip}) is missing or dead"
        device_str = current_platform.ray_device_key

        n_node_devices: list[int] = [
            int(node_resources[device_str]) for node_resources in nodes if device_str in node_resources
        ]

        if dp_size == 1:
            total_nodes = total_resources_per_node().values()
            total_n_node_devices: list[int] = [
                int(node_resources[device_str]) for node_resources in total_nodes if device_str in node_resources
            ]
            max_device_per_node = max(total_n_node_devices)
        else:
            assert n_node_devices, f"No {device_str} found in Ray cluster."
            max_device_per_node = max(n_node_devices)

        pack_strategy = envs.VLLM_RAY_DP_PACK_STRATEGY
        _supported_pack_strategies = ("strict", "fill", "span")
        if pack_strategy not in _supported_pack_strategies:
            raise ValueError(
                f"{envs.VLLM_RAY_DP_PACK_STRATEGY} is not supported. "
                f"Set VLLM_RAY_DP_PACK_STRATEGY to one of {_supported_pack_strategies}"
            )

        all2all_backend = vllm_config.parallel_config.all2all_backend
        if pack_strategy == "fill" and (
            all2all_backend == "deepep_high_throughput" or all2all_backend == "deepep_low_latency"
        ):
            raise ValueError(
                "DeepEP kernels require EP ranks on the same node, "
                "which VLLM_RAY_DP_PACK_STRATEGY=fill does not guarantee. "
                "Use VLLM_RAY_DP_PACK_STRATEGY=strict instead."
            )

        if pack_strategy in ("strict", "fill"):
            placement_strategy = "STRICT_PACK"
        else:
            placement_strategy = "PACK"
            assert world_size > max_device_per_node, (
                f"World size {world_size} smaller than max devices per node "
                f"{max_device_per_node}. Set VLLM_RAY_DP_PACK_STRATEGY to "
                "'strict' or 'fill'."
            )
            if dp_size == 1:
                assert set(total_n_node_devices) == {max_device_per_node}, f"Nodes are not homogenous, {nodes}"
            else:
                assert set(n_node_devices) == {max_device_per_node}, f"Nodes are not homogenous, {nodes}"
            assert world_size % max_device_per_node == 0, (
                f"For multi-node DP, world_size ({world_size}) must be a "
                f"multiple of devices per node ({max_device_per_node})."
            )
            # Head PG already covers one DP rank's worth of compute, so
            # the remaining check is against (dp_size - 1) groups.
            assert len(n_node_devices) * max_device_per_node >= world_size * (dp_size - 1), (
                f"Not enough available nodes ({len(n_node_devices)}) and "
                f"devices per node ({max_device_per_node}) to satisfy "
                f"world_size={world_size} and dp_size={dp_size}"
            )
            assert dp_size_local == 1, (
                f"data-parallel-size-local {dp_size_local} should be 1 with VLLM_RAY_DP_PACK_STRATEGY=span."
            )

        for _ in range(dp_size - 1):
            bundles = [{device_str: 1.0}] * world_size + [{"CPU": 1.0}]
            pg_name = f"{server_name}_dp_rank_{len(placement_groups)}"
            pg = ray.util.placement_group(
                name=pg_name,
                strategy=placement_strategy,
                bundles=bundles,
            )
            placement_groups.append(pg)
            local_dp_ranks.append(0)

        if len(placement_groups) < dp_size:
            raise ValueError(
                f"Not enough resources to allocate {dp_size} placement groups, "
                f"only created {len(placement_groups)}. "
                f"Available resources: {available_resources}"
            )
        assert len(placement_groups) == dp_size, (
            f"Created {len(placement_groups)} DP placement groups, expected {dp_size}"
        )
        assert len(local_dp_ranks) == dp_size, (
            f"local_dp_ranks length {len(local_dp_ranks)} does not match expected {dp_size}"
        )

        return placement_groups, local_dp_ranks

    CoreEngineActorManager.create_dp_placement_groups = new_create_dp_placement_groups

    # --- DPEngineCoreProc._init_data_parallel --------------------------------------

    def new_init_data_parallel(self, vllm_config):
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        dp_size = vllm_config.parallel_config.data_parallel_size
        local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local

        # Original vLLM: `assert dp_size > 1`. Our head-PG path makes
        # DP=1 a valid configuration (single replica, still Ray-coordinated).
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        if vllm_config.kv_transfer_config is not None:
            vllm_config.kv_transfer_config.engine_id = f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
            dp_logger.debug(
                "Setting kv_transfer_config.engine_id to %s",
                vllm_config.kv_transfer_config.engine_id,
            )

        self.dp_rank = dp_rank
        self.dp_group = vllm_config.parallel_config.stateless_init_dp_group()

    DPEngineCoreProc._init_data_parallel = new_init_data_parallel


# Populated by ``main()`` before the patches are applied.
_HEAD_PG_REF: dict = {}


def _reserve_head_placement_group(
    server_name: str,
    tp_size: int,
    pp_size: int,
    pack_strategy: str,
    num_gpus_per_node: int,
    timeout_s: int = 300,
):
    """Pre-reserve DP rank 0's placement group with only bundle 0 pinned to this node.

    For single-node DP replicas (``world_size <= num_gpus_per_node``), STRICT_PACK
    keeps the whole PG on this node. For multi-node DP replicas
    (``world_size > num_gpus_per_node``), strategy falls back to PACK and only
    the first bundle is pinned — Ray's PACK minimises spread while honouring
    the pin, so the PG anchors here and spills onto additional nodes as needed.

    The api_server runs in the caller's process on this node (rank 0), so
    colocating bundle 0 with the api_server minimises coordination RPC hops
    to DP rank 0's first GPU worker.
    """
    import ray
    from ray._private.services import get_node_ip_address

    node_ip = get_node_ip_address()
    node_resource_key = f"node:{node_ip}"
    world_size = tp_size * pp_size

    if world_size > num_gpus_per_node:
        placement_strategy = "PACK"
    else:
        placement_strategy = "STRICT_PACK" if pack_strategy in ("strict", "fill") else "PACK"

    # Pin only bundle 0 to this node (anchors the PG + colocates with
    # api_server). Remaining bundles float — PACK keeps them on the same
    # node when it fits, spills to additional nodes when it doesn't.
    bundles: list[dict] = [{"GPU": 1.0, node_resource_key: 0.01}]
    bundles += [{"GPU": 1.0} for _ in range(world_size - 1)]
    bundles += [{"CPU": 1.0}]

    pg = ray.util.placement_group(
        name=f"{server_name}_dp_rank_0",
        strategy=placement_strategy,
        bundles=bundles,
    )
    try:
        ray.get(pg.ready(), timeout=timeout_s)
    except ray.exceptions.GetTimeoutError as e:
        raise RuntimeError(
            f"Head placement group not ready after {timeout_s}s. "
            f"strategy={placement_strategy}, world_size={world_size}, "
            f"num_gpus_per_node={num_gpus_per_node}. "
            "Check Ray cluster resources: `ray status` from inside the job."
        ) from e

    return pg, node_ip


def _patch_signal_for_thread_safety() -> None:
    """No-op ``signal.signal`` when running off the main thread.

    vLLM's ``launcher.serve_http`` installs SIGINT/SIGTERM handlers via
    ``loop.add_signal_handler`` + ``signal.signal``. Neither is safe when
    a nested event loop runs off the main thread. When we're on the main
    thread (the normal case), this is a no-op so we don't clobber signal
    handling for subprocess / asyncio cleanup elsewhere in the process.
    """
    import threading

    if threading.current_thread() is threading.main_thread():
        return
    signal.signal = lambda *a, **kw: None


def _build_vllm_argv(args: argparse.Namespace, extra: Sequence[str]) -> list[str]:
    """Build the argv that vLLM's ``make_arg_parser()`` will parse.

    Note: ``--tensor-parallel-size`` / ``--pipeline-parallel-size`` /
    ``--data-parallel-size`` must be passed explicitly via ``extra`` (i.e.
    via ``server_args`` in the pipeline). We do NOT auto-infer them from
    ``num_gpus × num_nodes`` because the vllm_dp_ray topology
    (DP replicas × TP/PP per replica × extra Ray-only nodes) cannot be
    derived from that product alone.

    We do inject ``--distributed-executor-backend=ray`` when the user hasn't
    specified it — the whole point of this entrypoint is Ray-based DP/TP
    coordination, and vLLM defaults to mp backend which then rejects any
    ``world_size > num_gpus_per_node`` config.
    """
    argv = [
        f"--model={args.model}",
        f"--served-model-name={args.model}",
        "--trust-remote-code",
        "--host=0.0.0.0",
        f"--port={args.port}",
    ]
    if args.no_verbose:
        argv.extend(["--disable-log-requests", "--disable-log-stats"])
    has_backend = any(
        tok == "--distributed-executor-backend" or tok.startswith("--distributed-executor-backend=") for tok in extra
    )
    if not has_backend:
        argv.append("--distributed-executor-backend=ray")
    has_dp_local = any(
        tok in ("--data-parallel-size-local", "--data_parallel_size_local")
        or tok.startswith("--data-parallel-size-local=")
        or tok.startswith("--data_parallel_size_local=")
        for tok in extra
    )
    if not has_dp_local:
        # serve_vllm_dp_ray places each DP replica on its own node(s); vLLM
        # otherwise defaults data_parallel_size_local to data_parallel_size,
        # which packs all replicas onto the coordinator's node and then
        # tries to advertise GPU indices it doesn't have.
        argv.append("--data-parallel-size-local=1")
    has_dp_backend = any(
        tok in ("--data-parallel-backend", "--data_parallel_backend")
        or tok.startswith("--data-parallel-backend=")
        or tok.startswith("--data_parallel_backend=")
        for tok in extra
    )
    if not has_dp_backend:
        # vLLM's launch_core_engines selects CoreEngineActorManager (Ray-actor-based
        # DP engines, what our create_dp_placement_groups monkey-patch expects) only
        # when parallel_config.data_parallel_backend == "ray". Default is "mp", which
        # routes through CoreEngineProcManager and requires a separate `vllm serve`
        # invocation per node — not what we want for Ray-orchestrated multi-node DP.
        argv.append("--data-parallel-backend=ray")
    argv.extend(extra)
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve vLLM model with DP-on-Ray placement-group patch")
    parser.add_argument("--model", help="Path to the model or a HF model name")
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--num_nodes", type=int, required=False, default=1)
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--no_verbose", action="store_true")
    parser.add_argument(
        "--server_name",
        type=str,
        default="nemo_skills_vllm",
        help="Prefix for placement group names (for debugging in Ray dashboard)",
    )
    args, unknown = parser.parse_known_args()

    print(f"Deploying model {args.model} via serve_vllm_dp_ray")
    print("Starting OpenAI server (in-process, with DP-on-Ray PG patch)")

    # ------------------------------------------------------------------------------
    # Parse vLLM args first so we know TP / PP / DP sizing before Ray init.
    # ------------------------------------------------------------------------------
    from vllm.entrypoints.openai.api_server import (
        FlexibleArgumentParser,
        cli_env_setup,
        make_arg_parser,
        validate_parsed_serve_args,
    )

    cli_env_setup()
    vllm_parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    vllm_parser = make_arg_parser(vllm_parser)
    vllm_args = vllm_parser.parse_args(_build_vllm_argv(args, unknown))
    validate_parsed_serve_args(vllm_args)

    tp_size = getattr(vllm_args, "tensor_parallel_size", 1)
    pp_size = getattr(vllm_args, "pipeline_parallel_size", 1)
    dp_size = getattr(vllm_args, "data_parallel_size", 1)
    world_size = tp_size * pp_size

    # Auto-select pack strategy based on whether a DP replica fits on one
    # node. Upstream vLLM's patched create_dp_placement_groups requires
    # VLLM_RAY_DP_PACK_STRATEGY=span for multi-node DP replicas and
    # strict/fill for single-node replicas. Caller can still override.
    default_pack = "span" if world_size > args.num_gpus else "strict"
    pack_strategy = os.environ.setdefault("VLLM_RAY_DP_PACK_STRATEGY", default_pack)

    print(
        f"[serve_vllm_dp_ray] tp={tp_size} pp={pp_size} dp={dp_size} "
        f"world_size={world_size} num_gpus_per_node={args.num_gpus} "
        f"pack_strategy={pack_strategy}"
    )

    if world_size > args.num_gpus and world_size % args.num_gpus != 0:
        raise ValueError(
            f"world_size={world_size} (tp={tp_size} × pp={pp_size}) must be an integer "
            f"multiple of num_gpus_per_node={args.num_gpus} for multi-node DP replicas."
        )

    # ------------------------------------------------------------------------------
    # Attach to the existing Ray cluster (started by get_ray_server_cmd for
    # multi-node deployments; auto-started if single-node).
    # ------------------------------------------------------------------------------
    import ray

    ray.init(address="auto" if args.num_nodes > 1 else None, ignore_reinit_error=True)

    # The dp_size==1 fast path bifurcates by whether the single replica
    # fits on one node or not. The gate condition matters; do NOT relax
    # it without re-validating both branches end-to-end. Validation lives
    # in tests/slurm-tests/wmt24pp_gym_topology (configs C1 + C4).
    #
    # SINGLE-NODE DP=1 (this branch): TP fits on one node. vLLM's
    # ray_executor path handles this fine without our patches. Pre-reserving
    # a head PG here would just steal the GPUs vLLM is about to ask for
    # (engine then errors with "Current node has no GPU available"). We
    # only set VLLM_DP_MASTER_IP so the CoreEngineActorManager assertion
    # passes, and let vLLM drive. Validated by C1 (TP=8 DP=1 + 1 extra
    # COMET node).
    #
    # CROSS-NODE DP=1 (the else branch below, falls through to the
    # DP>1-style head-PG path): TP > num_gpus_per_node, replica spans 2+
    # nodes. The UNPATCHED vLLM ``create_dp_placement_groups`` uses
    # ``available_resources_per_node``, which sees 0 GPUs on our extra_gpu
    # nodes (those nodes joined Ray with ``--num-gpus=0`` per
    # get_ray_server_cmd). max_device_per_node miscomputes, the engine
    # init succeeds and KV cache allocates, but the OpenAI api_server
    # hangs after — never binds the listener. Observed in C4 (TP=16 DP=1
    # cross-node) on every run before this gate landed. The PATCHED path
    # uses ``total_resources_per_node`` for the dp_size==1 branch (in
    # ``new_create_dp_placement_groups`` above), which sees through the
    # extra-node masking.
    if dp_size == 1 and world_size <= args.num_gpus:
        print("[serve_vllm_dp_ray] dp_size=1 single-node: deferring to vLLM's native ray_executor path")
        from ray._private.services import get_node_ip_address

        os.environ["VLLM_DP_MASTER_IP"] = get_node_ip_address()
        _patch_signal_for_thread_safety()
        from vllm.entrypoints.openai.api_server import run_server

        print(f"[serve_vllm_dp_ray] vLLM argv: {shlex.join(sys.argv[1:])}")
        asyncio.run(run_server(vllm_args))
        return

    # ------------------------------------------------------------------------------
    # Reserve DP rank 0's placement group on THIS node and set
    # VLLM_DP_MASTER_IP so DP engines find the coordinator.
    # ------------------------------------------------------------------------------
    head_pg, node_ip = _reserve_head_placement_group(
        server_name=args.server_name,
        tp_size=tp_size,
        pp_size=pp_size,
        pack_strategy=pack_strategy,
        num_gpus_per_node=args.num_gpus,
    )
    os.environ["VLLM_DP_MASTER_IP"] = node_ip
    print(f"[serve_vllm_dp_ray] Reserved head PG on {node_ip} (DP rank 0)")

    _HEAD_PG_REF["pg"] = head_pg
    _HEAD_PG_REF["server_name"] = args.server_name

    # ------------------------------------------------------------------------------
    # Apply monkey-patches, then launch vLLM's OpenAI api_server in-process.
    # ------------------------------------------------------------------------------
    _apply_vllm_patches()
    _patch_signal_for_thread_safety()

    from vllm.entrypoints.openai.api_server import run_server

    print(f"[serve_vllm_dp_ray] vLLM argv: {shlex.join(sys.argv[1:])}")

    asyncio.run(run_server(vllm_args))


if __name__ == "__main__":
    main()

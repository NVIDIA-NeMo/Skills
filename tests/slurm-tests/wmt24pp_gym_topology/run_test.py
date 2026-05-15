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
"""Slurm test: NeMo-Gym wmt24pp rollouts across 5 server topologies.

Each topology exercises a distinct slice of the vllm_dp_ray + extra_gpu
COMET pool wiring. The Gym-side wmt_translation server dispatches a
streaming xCOMET-XXL future inside ``verify()`` and awaits it before
returning, so every rollouts.jsonl row should carry a non-None
``comet_score`` whenever the model produced a non-empty translation.

Topologies (see Gym/resources_servers/wmt_translation/app.py for the
streaming actor pool + per-row score plumbing this test verifies):

  C1  TP=8  DP=1                              ( 2 nodes: 1 model + 1 extra_gpu)
  C2  TP=8  DP=4   (1 replica per node)        ( 5 nodes: 4 model + 1 extra_gpu)
  C3  TP=16 DP=2   (each replica spans 2 nodes) ( 5 nodes: 4 model + 1 extra_gpu)
  C4  TP=16 DP=1   (single replica spans 2 nodes) ( 3 nodes: 2 model + 1 extra_gpu)
  C5  TP=8  DP=1   + 2 COMET nodes (16-actor pool) ( 3 nodes: 1 model + 2 extra_gpu)

Topologies C1 / C2 / C5 use NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (validated
translation model). C3 / C4 use deepseek-ai/DeepSeek-V2-Lite — the only
public model on this cluster that supports cross-node TP=16 (16 KV heads
+ 16 attention heads, no Mamba layers) without a vLLM KV-replication gap.
DSV2-Lite isn't a translation specialist; per-row comet_score will be low
on average but the test cares only that every non-empty row gets a score.

Each topology writes rollouts.jsonl into ``<workspace>/<config>/`` and
the final job runs ``check_results.py`` after all 5 topologies finish.
"""

import argparse

from nemo_skills.pipeline.cli import nemo_gym_rollouts, run_cmd, wrap_arguments

# Common Hydra overrides shared by every topology.
_COMMON_CTX = (
    "+agent_name=wmt24pp_wmt_translation_simple_agent "
    "+prompt_config=benchmarks/wmt24pp/prompts/default.yaml "
    "+num_repeats=1 "
    "+limit=250 "
    "++wmt24pp_wmt_translation_resources_server.resources_servers.wmt_translation.compute_comet=true "
    "++responses_create_params.temperature=1.0 "
    "++responses_create_params.top_p=0.95 "
    "++responses_create_params.max_output_tokens=2048 "
    "++policy_model.responses_api_models.vllm_model.extra_body.top_k=20 "
)

_NEMOTRON = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
_DSV2_LITE = "deepseek-ai/DeepSeek-V2-Lite"


def _server_args(tp_size: int, dp_size: int, reasoning_parser: bool) -> str:
    """vLLM CLI args shared across topologies; reasoning_parser only for Nemotron."""
    args = (
        f"--tensor-parallel-size {tp_size} "
        f"--data-parallel-size {dp_size} "
        "--data-parallel-size-local 1 "
        "--data-parallel-backend ray "
        "--distributed-executor-backend ray "
        "--api-server-count 1 "
        "--trust-remote-code --dtype auto --enforce-eager "
    )
    if reasoning_parser:
        args += "--reasoning-parser deepseek_r1 "
    return args


def _submit(
    *,
    name: str,
    workspace: str,
    cluster: str,
    expname_prefix: str,
    model: str,
    tp_size: int,
    dp_size: int,
    num_extra_nodes: int,
    num_samples_in_parallel: int,
    gpus_per_node: int = 8,
    partition: str | None = None,
    extra_ctx: str = "",
    reasoning_parser: bool = False,
) -> str:
    """Submit one topology configuration; return its expname for run_after wiring.

    ``gpus_per_node`` defaults to 8 (the H100 cluster shape we develop on)
    but flows through to both the SLURM allocation (``server_gpus``) and the
    derived COMET sharding so this test runs unchanged on 4-GPU/node nodes:

      * ``nodes_per_replica = ceil(world_size / gpus_per_node)`` — replica
        spans ``ceil(TP*PP / gpus_per_node)`` nodes (TP=16 → 2 nodes on 8-GPU,
        4 nodes on 4-GPU).
      * ``comet_num_shards = num_extra_nodes * gpus_per_node`` — actor pool
        saturates the extra_gpu nodes regardless of cluster shape.

    Per-config ``extra_ctx`` overrides win over the auto-computed default,
    so callers can clamp the actor pool down for memory-constrained nodes
    or up for benchmarking — but most callers should leave it alone.
    """
    nodes_per_replica = max(1, -(-tp_size // gpus_per_node))  # ceil divide
    num_nodes = dp_size * nodes_per_replica + num_extra_nodes
    comet_num_shards = num_extra_nodes * gpus_per_node
    expname = f"{expname_prefix}_{name}"
    output_dir = f"{workspace}/{name}"

    ctx_str = (
        _COMMON_CTX
        + f"+num_samples_in_parallel={num_samples_in_parallel} "
        + (
            "++wmt24pp_wmt_translation_resources_server.resources_servers."
            f"wmt_translation.comet_num_shards={comet_num_shards} "
        )
        + extra_ctx
    )

    nemo_gym_rollouts(
        ctx=wrap_arguments(ctx_str),
        cluster=cluster,
        config_paths=("benchmarks/wmt24pp/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml"),
        input_file="benchmarks/wmt24pp/data/wmt24pp_benchmark.jsonl",
        output_dir=output_dir,
        log_dir=f"{output_dir}/logs",
        model=model,
        server_type="vllm_dp_ray",
        server_gpus=gpus_per_node,
        server_nodes=num_nodes,
        server_args=_server_args(tp_size, dp_size, reasoning_parser),
        partition=partition,
        exclusive=True,
        expname=expname,
    )
    return expname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, help="Workspace dir for all 5 topology outputs")
    parser.add_argument("--cluster", required=True, help="Cluster config name")
    parser.add_argument("--expname_prefix", required=True, help="Prefix for the 5 + 1 SLURM experiments")
    parser.add_argument(
        "--gpus_per_node",
        type=int,
        default=8,
        help="GPUs per cluster node (default 8). Drives both replica sizing "
        "(nodes_per_replica = ceil(TP/gpus_per_node)) and the COMET actor "
        "count (comet_num_shards = num_extra_nodes * gpus_per_node).",
    )
    parser.add_argument(
        "--partition",
        default=None,
        help="SLURM partition. Defaults to the cluster config's default partition.",
    )
    args = parser.parse_args()

    expnames = [
        _submit(
            name="c1",
            workspace=args.workspace,
            cluster=args.cluster,
            expname_prefix=args.expname_prefix,
            gpus_per_node=args.gpus_per_node,
            partition=args.partition,
            model=_NEMOTRON,
            tp_size=8,
            dp_size=1,
            num_extra_nodes=1,
            num_samples_in_parallel=64,
            reasoning_parser=True,
        ),
        _submit(
            name="c2",
            workspace=args.workspace,
            cluster=args.cluster,
            expname_prefix=args.expname_prefix,
            gpus_per_node=args.gpus_per_node,
            partition=args.partition,
            model=_NEMOTRON,
            tp_size=8,
            dp_size=4,
            num_extra_nodes=1,
            num_samples_in_parallel=256,  # 64 concurrent × 4 DP engines
            reasoning_parser=True,
        ),
        _submit(
            name="c3",
            workspace=args.workspace,
            cluster=args.cluster,
            expname_prefix=args.expname_prefix,
            gpus_per_node=args.gpus_per_node,
            partition=args.partition,
            model=_DSV2_LITE,
            tp_size=16,
            dp_size=2,
            num_extra_nodes=1,
            num_samples_in_parallel=128,  # 64 concurrent × 2 DP engines
            reasoning_parser=False,
        ),
        _submit(
            name="c4",
            workspace=args.workspace,
            cluster=args.cluster,
            expname_prefix=args.expname_prefix,
            gpus_per_node=args.gpus_per_node,
            partition=args.partition,
            model=_DSV2_LITE,
            tp_size=16,
            dp_size=1,
            num_extra_nodes=1,
            num_samples_in_parallel=64,
            reasoning_parser=False,
        ),
        _submit(
            name="c5",
            workspace=args.workspace,
            cluster=args.cluster,
            expname_prefix=args.expname_prefix,
            gpus_per_node=args.gpus_per_node,
            partition=args.partition,
            model=_NEMOTRON,
            tp_size=8,
            dp_size=1,
            num_extra_nodes=2,
            num_samples_in_parallel=64,
            # comet_num_shards auto-derives to num_extra_nodes * gpus_per_node
            # (16 on 8-GPU/node, 8 on 4-GPU/node) — saturates whichever cluster.
            reasoning_parser=True,
        ),
    ]

    # Schedule the result-checker after all 5 topology jobs complete.
    checker_cmd = f"python tests/slurm-tests/wmt24pp_gym_topology/check_results.py --workspace {args.workspace}"
    run_cmd(
        ctx=wrap_arguments(checker_cmd),
        cluster=args.cluster,
        expname=args.expname_prefix + "-check-results",
        log_dir=f"{args.workspace}/check-results-logs",
        run_after=expnames,
    )


if __name__ == "__main__":
    main()

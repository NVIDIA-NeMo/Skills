#!/usr/bin/env python3
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

"""Pipeline+KubernetesBackend smoke test for single-node and multi-node jobs.

Exercises the actual Pipeline → KubernetesBackend code path (not raw kubectl)
to validate that:
1. Pipeline._convert_groups_to_job_spec() produces correct multi-node JobSpecs
2. KubernetesBackend.submit_job() creates Headless Service + Indexed Job
3. NCCL_DEBUG=INFO shows correct GPU topology and transport

Usage:
    # Dry-run only (validates manifests without submitting)
    python pipeline_smoke_test.py --dry-run

    # Single-node 2-GPU smoke test (submit to cluster)
    python pipeline_smoke_test.py --mode single --gpus 2

    # Multi-node 2x2 smoke test (submit to cluster)
    python pipeline_smoke_test.py --mode multi --nodes 2 --gpus 2

    # With custom namespace
    python pipeline_smoke_test.py --mode single --namespace my-ns --gpus 2
"""

import argparse
import os
import sys
from pathlib import Path

import nemo_run as run

# Ensure the repository root is importable when this script is executed directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nemo_skills.pipeline.utils.declarative import (  # noqa: E402
    Command,
    CommandGroup,
    HardwareConfig,
    Pipeline,
)

# Inline training script for the smoke test (PyTorch DDP, not full NeMo SFT)
_SINGLE_NODE_TRAIN_TEMPLATE = (
    "export NCCL_DEBUG=INFO\n"
    "export NCCL_DEBUG_SUBSYS=INIT,NET\n"
    "cat > /tmp/smoke_train.py << 'PYEOF'\n"
    "import os, torch, torch.distributed as dist\n"
    "from torch.nn.parallel import DistributedDataParallel as DDP\n"
    "import torch.nn as nn\n"
    "\n"
    'dist.init_process_group(backend="nccl")\n'
    "rank = dist.get_rank()\n"
    'local_rank = int(os.environ.get("LOCAL_RANK", 0))\n'
    'device = torch.device(f"cuda:{{local_rank}}")\n'
    "torch.cuda.set_device(device)\n"
    'print(f"[Rank {{rank}}] GPU: {{torch.cuda.get_device_name(device)}}")\n'
    "\n"
    "model = DDP(nn.Linear(64, 64).to(device), device_ids=[local_rank])\n"
    "x = torch.randn(16, 64, device=device)\n"
    "loss = model(x).sum()\n"
    "loss.backward()\n"
    "\n"
    "tensor = torch.ones(1, device=device) * rank\n"
    "dist.all_reduce(tensor)\n"
    'print(f"[Rank {{rank}}] all-reduce sum={{tensor.item()}}")\n'
    "dist.barrier()\n"
    "\n"
    "if rank == 0:\n"
    '    print("=== SMOKE TEST PASSED ===")\n'
    '    print(f"  World size: {{dist.get_world_size()}}")\n'
    '    print(f"  GPU: {{torch.cuda.get_device_name(0)}}")\n'
    "dist.destroy_process_group()\n"
    "PYEOF\n"
    "torchrun --nproc_per_node={gpus} --master_port=29500 /tmp/smoke_train.py"
)

MULTI_NODE_TRAIN_CMD = """
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
cat > /tmp/multinode_smoke_train.py << 'PYEOF'
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn

dist.init_process_group(backend='nccl')
rank = int(os.environ.get('RANK', 0))
local_rank = int(os.environ.get('LOCAL_RANK', 0))
device = torch.device(f'cuda:{{local_rank}}')
torch.cuda.set_device(device)
print(f'[Rank {{rank}}] Node {{os.environ.get("NODE_RANK","?")}} GPU: {{torch.cuda.get_device_name(device)}}')

model = DDP(nn.Linear(64, 64).to(device), device_ids=[local_rank])
x = torch.randn(16, 64, device=device)
loss = model(x).sum()
loss.backward()

tensor = torch.ones(1, device=device) * rank
dist.all_reduce(tensor)
print(f'[Rank {{rank}}] all-reduce sum={{tensor.item()}}')
dist.barrier()

if rank == 0:
    print('=== MULTI-NODE SMOKE TEST PASSED ===')
    print(f'  World size: {{dist.get_world_size()}}')
    print(f'  GPU: {{torch.cuda.get_device_name(0)}}')
dist.destroy_process_group()
PYEOF
torchrun \\
  --nproc_per_node={gpus} \\
  --nnodes={nodes} \\
  --node_rank=${{NODE_RANK:-0}} \\
  --master_addr=${{MASTER_ADDR:-localhost}} \\
  --master_port=${{MASTER_PORT:-29500}} \\
  /tmp/multinode_smoke_train.py
"""


def build_pipeline(mode, namespace, gpus, nodes, image):
    """Build a Pipeline using the declarative API."""
    if mode == "single":
        cmd = _SINGLE_NODE_TRAIN_TEMPLATE.format(gpus=gpus)
        hw = HardwareConfig(num_gpus=gpus, num_nodes=1)
    else:
        cmd = MULTI_NODE_TRAIN_CMD.format(gpus=gpus, nodes=nodes)
        hw = HardwareConfig(num_gpus=gpus, num_nodes=nodes)

    script = run.Script(inline=cmd)
    command = Command(script=script, container="nemo-skills", name="trainer")
    group = CommandGroup(
        commands=[command],
        hardware=hw,
        name=f"smoke-{mode}",
        log_dir="/tmp/smoke-logs",
    )

    cluster_config = {
        "executor": "kubernetes",
        "namespace": namespace,
        "containers": {"nemo-skills": image},
        "skip_hf_home_check": True,
        "default_timeout": "15m",
        "env_vars": ["NCCL_DEBUG=INFO"],
    }

    pipeline = Pipeline(
        name=f"smoke-test-{mode}",
        cluster_config=cluster_config,
        jobs=[{"name": f"smoke-{mode}", "group": group}],
    )

    return pipeline, cluster_config


def main():
    parser = argparse.ArgumentParser(description="Pipeline+KubernetesBackend smoke test")
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifests only")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--nodes", type=int, default=2, help="Used for --mode multi only")
    parser.add_argument("--image", default=os.environ.get("PYTORCH_IMAGE", "nvcr.io/nvidia/pytorch:25.03-py3"))
    args = parser.parse_args()

    if args.mode == "multi" and args.nodes < 2:
        parser.error("--mode multi requires --nodes >= 2")

    effective_nodes = 1 if args.mode == "single" else args.nodes

    print(f"Mode: {args.mode} | Namespace: {args.namespace} | GPUs: {args.gpus} | Nodes: {effective_nodes}")
    print(f"Image: {args.image}")
    print(f"Dry run: {args.dry_run}")
    print()

    pipeline, cluster_config = build_pipeline(
        args.mode,
        args.namespace,
        args.gpus,
        effective_nodes,
        args.image,
    )

    if args.dry_run:
        # Dry run exercises the full Pipeline → JobSpec conversion
        print("=== DRY RUN: Validating manifest generation ===")
        result = pipeline.run(dry_run=True)
        print("Dry run completed successfully.")

        # Also validate the JobSpec directly
        groups = [pipeline.jobs[0]["group"]]
        job_spec = pipeline._convert_groups_to_job_spec(
            job_name="smoke-test",
            groups=groups,
            log_dir="/tmp/logs",
        )
        print(f"JobSpec.num_nodes = {job_spec.num_nodes}")
        print(f"JobSpec.is_multi_node = {job_spec.is_multi_node}")
        print(f"JobSpec.containers = {len(job_spec.containers)}")
        print(f"JobSpec.containers[0].resources.gpus = {job_spec.containers[0].resources.gpus}")
        if job_spec.is_multi_node:
            print("Multi-node: Headless Service + Indexed Job will be created")
        print("\n=== MANIFEST VALIDATION PASSED ===")
        return

    # Real submission through Pipeline + KubernetesBackend
    print("=== SUBMITTING JOB VIA PIPELINE + KUBERNETES BACKEND ===")
    result = pipeline.run(dry_run=False)

    if result is None:
        print("ERROR: Pipeline returned None (unexpected)")
        sys.exit(1)

    # result is a dict of {job_name: JobHandle}
    print(f"Submitted {len(result)} job(s)")
    for name, handle in result.items():
        print(f"  {name}: job_id={handle.job_id}")
        svc = handle.metadata.get("headless_service")
        if svc:
            print(f"    headless_service={svc}")

    # Monitor and collect logs
    from nemo_skills.pipeline.backends import JobStatus, get_backend

    backend = get_backend(cluster_config)

    for name, handle in result.items():
        print(f"\nWaiting for job '{name}' to complete...")
        status = backend.wait_for_completion(handle, timeout=900)
        print(f"Job '{name}' status: {status.value}")

        if status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
            print(f"\n--- Logs from {name} ---")
            for line in backend.get_logs(handle):
                print(line, end="" if line.endswith("\n") else "\n")

        if status == JobStatus.FAILED:
            print(f"\nERROR: Job '{name}' failed")
            sys.exit(1)

    print("\n=== ALL JOBS COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()

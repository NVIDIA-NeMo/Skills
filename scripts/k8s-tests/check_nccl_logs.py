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

"""NCCL log validator for verifying distributed training communication.

Parses NCCL_DEBUG=INFO logs and checks:
- NCCL initialization and version
- GPU detection (count and model)
- Transport selection (NVLink, IB, TCP, etc.)
- Ring/tree topology setup
- Multi-node communication (for multi-node jobs)

Usage:
    # From kubectl logs
    kubectl logs <pod-name> | python check_nccl_logs.py

    # From a log file
    python check_nccl_logs.py --log-file /path/to/nccl.log

    # Specify expected node/GPU counts
    python check_nccl_logs.py --expected-nodes 2 --expected-gpus-per-node 8 --log-file logs.txt
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class NCCLCheckResult:
    """Result of NCCL log validation."""

    nccl_initialized: bool = False
    nccl_version: Optional[str] = None
    gpu_count: int = 0
    gpu_model: Optional[str] = None
    transport_used: List[str] = field(default_factory=list)
    nvlink_detected: bool = False
    ib_detected: bool = False
    tcp_fallback: bool = False
    ring_topology: bool = False
    tree_topology: bool = False
    world_size: int = 0
    ranks_seen: set = field(default_factory=set)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def parse_nccl_logs(log_text: str) -> NCCLCheckResult:
    """Parse NCCL_DEBUG=INFO log output and extract key indicators."""
    result = NCCLCheckResult()

    for line in log_text.splitlines():
        # NCCL version detection
        if "NCCL version" in line or "NCCL INFO NCCL" in line:
            result.nccl_initialized = True
            version_match = re.search(r"NCCL version (\S+)", line)
            if version_match:
                result.nccl_version = version_match.group(1)

        # NCCL init
        if "NCCL INFO" in line and "Init" in line:
            result.nccl_initialized = True

        # GPU count and model
        gpu_match = re.search(r"(\d+) GPUs? detected", line, re.IGNORECASE)
        if gpu_match:
            result.gpu_count = max(result.gpu_count, int(gpu_match.group(1)))

        if "CUDA Dev" in line or "GPU" in line:
            model_match = re.search(r"(H100|A100|H200|B200|GB200|V100|A10G|L40)", line)
            if model_match:
                result.gpu_model = model_match.group(1)

        # Transport detection
        if "NVLink" in line or "NVLS" in line:
            result.nvlink_detected = True
            if "NVLink" not in result.transport_used:
                result.transport_used.append("NVLink")

        if "IB" in line and ("NCCL INFO" in line or "transport" in line.lower()):
            result.ib_detected = True
            if "IB" not in result.transport_used:
                result.transport_used.append("IB")

        if re.search(r"NET.*Socket|TCP", line) and "NCCL" in line:
            result.tcp_fallback = True
            if "TCP/Socket" not in result.transport_used:
                result.transport_used.append("TCP/Socket")

        # Topology
        if "Ring" in line and "NCCL" in line:
            result.ring_topology = True
        if "Tree" in line and "NCCL" in line:
            result.tree_topology = True

        # World size / rank info
        world_match = re.search(r"nranks (\d+)|world_size[=: ]+(\d+)|WORLD_SIZE[=: ]+(\d+)", line)
        if world_match:
            for g in world_match.groups():
                if g:
                    result.world_size = max(result.world_size, int(g))

        rank_match = re.search(r"Rank (\d+)", line)
        if rank_match:
            result.ranks_seen.add(int(rank_match.group(1)))

        # Errors
        if "NCCL WARN" in line or "NCCL ERROR" in line:
            result.errors.append(line.strip())

    return result


def validate_result(
    result: NCCLCheckResult,
    expected_nodes: Optional[int] = None,
    expected_gpus_per_node: Optional[int] = None,
) -> tuple:
    """Validate NCCL check result against expectations.

    Returns (passed: bool, messages: list[str])
    """
    messages = []
    passed = True

    # Check NCCL initialization
    if not result.nccl_initialized:
        messages.append("FAIL: NCCL did not initialize. Check NCCL_DEBUG=INFO is set.")
        passed = False
    else:
        ver = result.nccl_version or "unknown"
        messages.append(f"OK: NCCL initialized (version: {ver})")

    # Check GPU detection
    if result.gpu_count > 0:
        messages.append(f"OK: {result.gpu_count} GPU(s) detected (model: {result.gpu_model or 'unknown'})")
    elif result.nccl_initialized:
        messages.append("WARN: GPU count not found in logs (may be normal for some NCCL versions)")

    # Check transport
    if result.nvlink_detected:
        messages.append("OK: NVLink transport detected (optimal intra-node)")
    elif result.nccl_initialized:
        messages.append("WARN: NVLink NOT detected. May be using PCIe (check GPU topology)")

    if expected_nodes and expected_nodes > 1:
        if result.ib_detected:
            messages.append("OK: InfiniBand transport detected (optimal inter-node)")
        elif result.tcp_fallback:
            messages.append("WARN: TCP/Socket transport detected (IB preferred for multi-node)")
        elif result.nccl_initialized:
            messages.append("WARN: No inter-node transport detected in logs")

    # Check topology
    if result.ring_topology or result.tree_topology:
        topologies = []
        if result.ring_topology:
            topologies.append("Ring")
        if result.tree_topology:
            topologies.append("Tree")
        messages.append(f"OK: Topology detected: {', '.join(topologies)}")

    # Check world size / expected nodes
    if expected_nodes and expected_gpus_per_node:
        expected_world = expected_nodes * expected_gpus_per_node
        if result.world_size > 0:
            if result.world_size == expected_world:
                messages.append(
                    f"OK: World size {result.world_size} matches expected ({expected_nodes} nodes x {expected_gpus_per_node} GPUs)"
                )
            else:
                messages.append(f"FAIL: World size {result.world_size} != expected {expected_world}")
                passed = False

    # Check for NCCL errors
    if result.errors:
        messages.append(f"FAIL: {len(result.errors)} NCCL error(s)/warning(s):")
        for err in result.errors[:5]:
            messages.append(f"  - {err}")
        passed = False

    # Transport summary
    if result.transport_used:
        messages.append(f"Transport summary: {', '.join(result.transport_used)}")

    return passed, messages


def main():
    parser = argparse.ArgumentParser(description="Validate NCCL logs for distributed training")
    parser.add_argument("--log-file", help="Path to log file (reads stdin if not specified)")
    parser.add_argument("--expected-nodes", type=int, help="Expected number of nodes")
    parser.add_argument("--expected-gpus-per-node", type=int, default=8, help="Expected GPUs per node (default: 8)")
    args = parser.parse_args()

    if args.log_file:
        with open(args.log_file) as f:
            log_text = f.read()
    else:
        log_text = sys.stdin.read()

    result = parse_nccl_logs(log_text)
    passed, messages = validate_result(
        result,
        expected_nodes=args.expected_nodes,
        expected_gpus_per_node=args.expected_gpus_per_node,
    )

    print("=" * 60)
    print("NCCL Log Validation Report")
    print("=" * 60)
    for msg in messages:
        print(msg)
    print("=" * 60)
    print(f"Result: {'PASS' if passed else 'FAIL'}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

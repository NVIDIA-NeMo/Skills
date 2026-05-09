# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""End-to-end slurm-test for the Ray executor.

Validates that a cluster_config with ``executor: ray`` can submit a job
through NeMo-Skills, the job actually executes on the Ray cluster, and the
dependency-chained check_results step runs after.

Cluster requirements (provided by the cluster_config passed via --cluster):
- ``executor: ray``
- A Ray cluster reachable via ``ray.address`` (typically a Ray-on-Slurm setup
  where Slurm allocated the nodes and Ray started a head + workers inside
  that allocation; alternatively a standalone Ray endpoint).
- A ``/workspace`` mount visible to Ray workers.

What this test does:
1. Submits a tiny ``run_cmd`` task via the Ray executor that writes a marker
   file + a status JSON to the workspace. This exercises:
     - cluster_config["executor"] == "ray" routing in get_executor()
     - add_task() Ray short-circuit → RayJobClient.submit_job()
     - shared-FS visibility (the worker writes to /workspace, head reads it)
2. Schedules check_results.py as a dependent task — exercises Ray dependency
   chaining via submission IDs.

This is intentionally minimal — proving the executor RUNS end-to-end is the
point, not measuring model quality. Quality metrics belong in the SFT/eval
slurm tests, not here.
"""

import argparse
import shlex

from nemo_skills.pipeline.cli import run_cmd, wrap_arguments
from nemo_skills.pipeline.utils.cluster import get_cluster_config


def submit_ray_smoke_task(workspace, cluster, expname_prefix):
    """Submit a trivial Ray job that writes marker files to the workspace."""
    expname = f"{expname_prefix}-ray-smoke"
    marker_dir = f"{workspace}/ray_smoke_output"
    qdir = shlex.quote(marker_dir)
    qdone = shlex.quote(f"{marker_dir}/done.marker")
    qts = shlex.quote(f"{marker_dir}/timestamp.txt")
    qstatus = shlex.quote(f"{marker_dir}/status.json")
    # Pass status.json path as sys.argv[1] rather than embedding it in the
    # python -c source — avoids the quoting trap when workspace contains a
    # single quote, space, or other shell metacharacter.
    py_code = (
        "import json, sys, platform; "
        'json.dump({"host": platform.node(), "python": sys.version}, '
        'open(sys.argv[1], "w"))'
    )
    smoke_cmd = (
        f"mkdir -p {qdir} && "
        f"echo 'ray_executor_smoke ok' > {qdone} && "
        f"date -u +%Y-%m-%dT%H:%M:%SZ > {qts} && "
        f"python -c {shlex.quote(py_code)} {qstatus}"
    )

    run_cmd(
        ctx=wrap_arguments(smoke_cmd),
        cluster=cluster,
        expname=expname,
        log_dir=f"{workspace}/ray-smoke-logs",
    )

    return expname


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, help="Workspace directory (mounted into Ray workers)")
    parser.add_argument("--cluster", required=True, help="Cluster config name with executor: ray")
    parser.add_argument("--expname_prefix", required=True, help="Experiment name prefix")

    args = parser.parse_args()

    cluster_config = get_cluster_config(cluster=args.cluster)
    executor = cluster_config.get("executor")
    if executor != "ray":
        raise ValueError(
            f"qwen3_4b_ray_executor slurm-test requires cluster_config with "
            f"executor: ray, got executor={executor!r} from cluster {args.cluster!r}"
        )

    smoke_expname = submit_ray_smoke_task(
        workspace=args.workspace,
        cluster=args.cluster,
        expname_prefix=args.expname_prefix,
    )

    # schedule a dependent check_results job that asserts the marker files exist.
    checker_cmd = f"python tests/slurm-tests/qwen3_4b_ray_executor/check_results.py --workspace {args.workspace}"

    run_cmd(
        ctx=wrap_arguments(checker_cmd),
        cluster=args.cluster,
        expname=args.expname_prefix + "-check-results",
        log_dir=f"{args.workspace}/check-results-logs",
        run_after=[smoke_expname],
    )


if __name__ == "__main__":
    main()

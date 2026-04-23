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

import shlex
import subprocess

import nemo_skills.pipeline.utils.scripts.eval as eval_scripts
from nemo_skills.pipeline.utils import eval as eval_utils
from nemo_skills.pipeline.utils.scripts import EvalClientScript


def test_eval_client_script_parallel_fails_if_any_unit_fails(monkeypatch, tmp_path):
    """Ensure eval units fail if any command fails in parallel mode."""
    failed_marker = tmp_path / "failed.txt"
    succeeded_marker = tmp_path / "succeeded.txt"

    monkeypatch.setattr(eval_scripts, "get_generation_cmd", lambda *, command, **kwargs: command)
    monkeypatch.setattr(eval_scripts, "wrap_python_path", lambda cmd: cmd)

    units = [
        {"command": f"echo failed > {shlex.quote(str(failed_marker))}; sleep 0.1; exit 1"},
        {"command": f"echo succeeded > {shlex.quote(str(succeeded_marker))}; sleep 0.2; exit 0"},
    ]
    client_script = EvalClientScript(units=units, single_node_mode="parallel")

    cmd, _ = client_script.inline()
    result = subprocess.run(["bash", "-lc", cmd], check=False)

    assert result.returncode == 1
    assert failed_marker.read_text().strip() == "failed"
    assert succeeded_marker.read_text().strip() == "succeeded"


def test_prepare_eval_commands_propagates_cli_with_sandbox_to_generation_cmd(monkeypatch):
    """Ensure `--with-sandbox` is treated as an override when building eval commands.

    Previously, if a benchmark had `REQUIRES_SANDBOX` unset and the user passed
    `--with-sandbox`, the sandbox sidecar was still launched because `add_task()`
    ORed the two flags together. This checks that the prepared eval generation
    unit keeps `with_sandbox=True` all the way into `get_generation_cmd`.
    """
    benchmark_args = eval_utils.BenchmarkArgs(
        name="aime25",
        input_file="/tmp/aime25.jsonl",
        generation_args="",
        judge_args="",
        judge_pipeline_args={},
        requires_sandbox=False,
        keep_mounts_for_sandbox=False,
        generation_module="nemo_skills.inference.generate",
        num_samples=0,
        num_chunks=None,
        eval_subfolder="eval-results/aime25",
    )

    monkeypatch.setattr(eval_utils, "add_default_args", lambda *args, **kwargs: [benchmark_args])
    monkeypatch.setattr(eval_utils.pipeline_utils, "get_remaining_jobs", lambda **kwargs: {None: [None]})

    captured = {}

    def fake_get_generation_cmd(*args, **kwargs):
        captured["with_sandbox"] = kwargs["with_sandbox"]
        return "echo generation"

    monkeypatch.setattr("nemo_skills.pipeline.utils.scripts.eval.get_generation_cmd", fake_get_generation_cmd)

    _, job_batches = eval_utils.prepare_eval_commands(
        cluster_config={"executor": "none"},
        benchmarks_or_groups="aime25",
        split=None,
        num_jobs=1,
        starting_seed=0,
        output_dir="/tmp/out",
        num_chunks=None,
        chunk_ids=None,
        rerun_done=False,
        extra_arguments="",
        data_dir=None,
        exclusive=False,
        with_sandbox=True,
        keep_mounts_for_sandbox=False,
        wandb_parameters=None,
        eval_requires_judge=False,
    )

    units = [vars(unit).copy() for unit in job_batches[0][0]]
    client_script = EvalClientScript(units=units)
    client_script.inline()

    assert captured["with_sandbox"] is True

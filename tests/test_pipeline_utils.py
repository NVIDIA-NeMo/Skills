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

import os
import tempfile
from unittest.mock import MagicMock, patch

from nemo_skills.pipeline.utils.declarative import Command
from nemo_skills.pipeline.utils.generation import (
    get_chunked_rs_filename,
    get_expected_done_files,
    get_remaining_jobs,
    separate_hydra_args,
)


def create_done_files(output_dir, seed_chunk_pairs):
    """Helper to create .done files for given seed/chunk pairs."""
    for seed, chunk in seed_chunk_pairs:
        filename = get_chunked_rs_filename(output_dir, random_seed=seed, chunk_id=chunk)
        done_file = f"{filename}.done"
        os.makedirs(os.path.dirname(done_file), exist_ok=True)
        with open(done_file, "w") as f:
            f.write("")


def test_get_chunked_rs_filename():
    """Test filename generation with different parameters."""
    assert get_chunked_rs_filename("/tmp/output", random_seed=42) == "/tmp/output/output-rs42.jsonl"
    assert (
        get_chunked_rs_filename("/tmp/output", random_seed=42, chunk_id=5) == "/tmp/output/output-rs42_chunk_5.jsonl"
    )
    assert get_chunked_rs_filename("/tmp/output", chunk_id=5) == "/tmp/output/output_chunk_5.jsonl"
    assert get_chunked_rs_filename("/tmp/output") == "/tmp/output/output.jsonl"


def test_get_expected_done_files():
    """Test expected done file mapping generation."""
    output_dir = "/tmp/output"
    random_seeds = [0, 1]
    chunk_ids = [0, 1, 2]

    file_map = get_expected_done_files(output_dir, random_seeds, chunk_ids)

    assert len(file_map) == 6  # 2 seeds × 3 chunks
    assert file_map[(0, 0)] == "/tmp/output/output-rs0_chunk_0.jsonl.done"
    assert file_map[(1, 2)] == "/tmp/output/output-rs1_chunk_2.jsonl.done"


@patch("nemo_skills.pipeline.utils.generation.get_unmounted_path", lambda config, path: path)
def test_get_remaining_jobs_small():
    """Test with small number of files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cluster_config = {"executor": "local", "mounts": []}
        random_seeds = [0, 1]
        chunk_ids = [0, 1, 2]

        # Create some done files
        create_done_files(tmpdir, [(0, 0), (1, 1)])

        remaining = get_remaining_jobs(cluster_config, tmpdir, random_seeds, chunk_ids, rerun_done=False)

        assert sorted(remaining[0]) == [1, 2]
        assert sorted(remaining[1]) == [0, 2]


@patch("nemo_skills.pipeline.utils.generation.get_unmounted_path", lambda config, path: path)
def test_get_remaining_jobs_large():
    """Test with large number of files requiring batching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cluster_config = {"executor": "local", "mounts": []}
        # Create 960 files (8 seeds × 120 chunks)
        random_seeds = list(range(8))
        chunk_ids = list(range(120))

        # Mark every 3rd chunk as done
        done_pairs = [(seed, chunk) for seed in random_seeds for chunk in range(0, 120, 3)]
        create_done_files(tmpdir, done_pairs)

        remaining = get_remaining_jobs(cluster_config, tmpdir, random_seeds, chunk_ids, rerun_done=False)

        # Verify the results
        for seed in random_seeds:
            expected_remaining = [c for c in chunk_ids if c % 3 != 0]
            assert sorted(remaining[seed]) == sorted(expected_remaining)


@patch("nemo_skills.pipeline.utils.generation.get_unmounted_path", lambda config, path: path)
def test_get_remaining_jobs_rerun_done():
    """Test that rerun_done=True ignores existing done files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cluster_config = {"executor": "local", "mounts": []}
        random_seeds = [0, 1]
        chunk_ids = [0, 1, 2]

        # Create all done files
        for seed in random_seeds:
            for chunk in chunk_ids:
                create_done_files(tmpdir, [(seed, chunk)])

        remaining = get_remaining_jobs(cluster_config, tmpdir, random_seeds, chunk_ids, rerun_done=True)

        # All jobs should be marked as remaining
        for seed in random_seeds:
            assert sorted(remaining[seed]) == sorted(chunk_ids)


@patch("nemo_skills.pipeline.utils.generation.get_unmounted_path", lambda config, path: path)
def test_get_remaining_jobs_no_chunks():
    """Test with no chunking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cluster_config = {"executor": "local", "mounts": []}
        random_seeds = [0, 1, 2]
        chunk_ids = [None]

        # Create done file for seed 1
        create_done_files(tmpdir, [(1, None)])

        remaining = get_remaining_jobs(cluster_config, tmpdir, random_seeds, chunk_ids, rerun_done=False)

        assert None in remaining[0]
        assert None not in remaining[1]  # This one is done
        assert None in remaining[2]


@patch("nemo_skills.pipeline.utils.generation.get_unmounted_path", lambda config, path: path)
def test_batch_processing_fallback():
    """Test fallback to individual file checks when batch processing fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cluster_config = {"executor": "local", "mounts": []}
        random_seeds = list(range(2))
        chunk_ids = list(range(20))  # 40 files to trigger batching

        with patch("nemo_skills.pipeline.utils.generation.subprocess.run") as mock_run:
            # Setup mock responses
            side_effects = []

            # First batch fails
            side_effects.append(Exception("Batch command failed"))

            # Individual checks for first batch (30 files)
            for i in range(30):
                if i % 3 == 0:
                    side_effects.append(MagicMock(stdout=f"MISSING:{i // 20}:{i % 20}".encode()))
                else:
                    side_effects.append(MagicMock(stdout=b""))

            # Second batch fails
            side_effects.append(Exception("Batch command failed"))

            # Individual checks for remaining files
            for i in range(30, 40):
                if i % 3 == 0:
                    side_effects.append(MagicMock(stdout=f"MISSING:{i // 20}:{i % 20}".encode()))
                else:
                    side_effects.append(MagicMock(stdout=b""))

            mock_run.side_effect = side_effects

            remaining = get_remaining_jobs(cluster_config, tmpdir, random_seeds, chunk_ids, rerun_done=False)

            # Verify the correct files are marked as missing
            for seed in random_seeds:
                for chunk in chunk_ids:
                    file_idx = seed * 20 + chunk
                    if file_idx % 3 == 0:
                        assert chunk in remaining[seed]
                    else:
                        assert chunk not in remaining[seed]


@patch("nemo_skills.pipeline.utils.generation.get_tunnel")
@patch("nemo_skills.pipeline.utils.generation.get_unmounted_path", lambda config, path: path)
def test_slurm_execution(mock_get_tunnel):
    """Test execution on Slurm cluster."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cluster_config = {"executor": "slurm", "mounts": []}
        random_seeds = [0, 1]
        chunk_ids = list(range(50))  # Enough to trigger batching

        # Mock the tunnel response
        mock_tunnel = MagicMock()
        mock_tunnel.run.return_value.stdout = "MISSING:0:10\nMISSING:1:20"
        mock_get_tunnel.return_value = mock_tunnel

        remaining = get_remaining_jobs(cluster_config, tmpdir, random_seeds, chunk_ids, rerun_done=False)

        # Verify tunnel was called
        assert mock_get_tunnel.called
        assert 10 in remaining[0]
        assert 20 in remaining[1]


def test_separate_hydra_args_empty():
    """Test with empty string."""
    hydra_args, override_args = separate_hydra_args("")
    assert hydra_args == ""
    assert override_args == ""


def test_separate_hydra_args_only_hydra():
    """Test with only Hydra config args."""
    hydra_args, override_args = separate_hydra_args("--config-path /workspace/configs --config-name my_config")
    assert hydra_args == " --config-path /workspace/configs --config-name my_config"
    assert override_args == ""


def test_separate_hydra_args_only_overrides():
    """Test with only override args."""
    hydra_args, override_args = separate_hydra_args("++inference.temperature=0.7 ++inference.top_p=0.95")
    assert hydra_args == ""
    assert override_args == " ++inference.temperature=0.7 ++inference.top_p=0.95"


def test_separate_hydra_args_mixed():
    """Test with mixed Hydra and override args."""
    extra_args = (
        "--config-path /workspace/configs --config-name reasoning ++inference.temperature=0.7 ++parse_reasoning=True"
    )
    hydra_args, override_args = separate_hydra_args(extra_args)
    assert hydra_args == " --config-path /workspace/configs --config-name reasoning"
    assert override_args == " ++inference.temperature=0.7 ++parse_reasoning=True"


def test_separate_hydra_args_equals_format():
    """Test with --config-path=value format."""
    hydra_args, override_args = separate_hydra_args("--config-path=/workspace/configs --config-name=reasoning")
    assert hydra_args == " --config-path=/workspace/configs --config-name=reasoning"
    assert override_args == ""


def test_separate_hydra_args_mixed_formats():
    """Test with mixed formats (= and space separated)."""
    extra_args = "--config-path=/workspace/configs --config-name reasoning ++temperature=0.7"
    hydra_args, override_args = separate_hydra_args(extra_args)
    assert hydra_args == " --config-path=/workspace/configs --config-name reasoning"
    assert override_args == " ++temperature=0.7"


def test_separate_hydra_args_with_special_chars():
    """Test with override args containing special characters."""
    extra_args = "--config-path /configs --config-name test ++end_reasoning_string='END_TAG' ++stop_phrase='\\n\\n'"
    hydra_args, override_args = separate_hydra_args(extra_args)
    assert hydra_args == " --config-path /configs --config-name test"
    assert override_args == " ++end_reasoning_string=END_TAG ++stop_phrase=\\n\\n"


def test_separate_hydra_args_complex():
    """Test complex realistic scenario."""
    extra_args = (
        "--config-path /nemo_run/code/configs --config-name reasoning_config "
        "++prompt_config=generic/math-base ++inference.temperature=0.7 "
        "++inference.tokens_to_generate=2048 ++parse_reasoning=True "
        "++end_reasoning_string='END_TAG'"
    )
    hydra_args, override_args = separate_hydra_args(extra_args)
    assert hydra_args == " --config-path /nemo_run/code/configs --config-name reasoning_config"
    assert override_args == (
        " ++prompt_config=generic/math-base ++inference.temperature=0.7 "
        "++inference.tokens_to_generate=2048 ++parse_reasoning=True "
        "++end_reasoning_string=END_TAG"
    )


def test_separate_hydra_args_hydra_no_value_flags():
    """Hydra flags without values should be extracted correctly."""
    extra_args = "--help ++inference.temperature=0.7 --run ++other=1 --multirun"
    hydra_args, override_args = separate_hydra_args(extra_args)
    assert hydra_args == " --help --run --multirun"
    assert override_args == " ++inference.temperature=0.7 ++other=1"


def test_separate_hydra_args_hydra_with_value_flags_space_and_equals():
    """Hydra flags with values should support both space and equals formats."""
    extra_args = (
        "--cfg all --info=plugins --package mypkg ++x=1 --config-dir /workspace/configs ++y=2 --experimental-rerun=abc"
    )
    hydra_args, override_args = separate_hydra_args(extra_args)
    assert (
        hydra_args
        == " --cfg all --info=plugins --package mypkg --config-dir /workspace/configs --experimental-rerun=abc"
    )
    assert override_args == " ++x=1 ++y=2"


def test_separate_hydra_args_hydra_help_and_version():
    """Ensure misc hydra flags are captured and don't disturb overrides."""
    extra_args = "--hydra-help --version ++prompt_config=generic/math"
    hydra_args, override_args = separate_hydra_args(extra_args)
    assert hydra_args == " --hydra-help --version"
    assert override_args == " ++prompt_config=generic/math"


def test_separate_hydra_args_config_at_end():
    """Test with config args at the end."""
    extra_args = (
        "++inference.temperature=0.7 ++parse_reasoning=True --config-path /workspace/configs --config-name reasoning"
    )
    hydra_args, override_args = separate_hydra_args(extra_args)
    assert hydra_args == " --config-path /workspace/configs --config-name reasoning"
    assert override_args == " ++inference.temperature=0.7 ++parse_reasoning=True"


def test_separate_hydra_args_config_in_middle():
    """Test with config args in the middle."""
    extra_args = "++inference.temperature=0.7 --config-path /configs --config-name test ++parse_reasoning=True"
    hydra_args, override_args = separate_hydra_args(extra_args)
    assert hydra_args == " --config-path /configs --config-name test"
    assert override_args == " ++inference.temperature=0.7 ++parse_reasoning=True"


def test_separate_hydra_args_interspersed():
    """Test with config args interspersed with overrides."""
    extra_args = (
        "++prompt_config=generic/math "
        "--config-path /workspace/configs "
        "++inference.temperature=0.7 "
        "--config-name reasoning_config "
        "++inference.top_p=0.95"
    )
    hydra_args, override_args = separate_hydra_args(extra_args)
    assert hydra_args == " --config-path /workspace/configs --config-name reasoning_config"
    assert override_args == " ++prompt_config=generic/math ++inference.temperature=0.7 ++inference.top_p=0.95"


def test_separate_hydra_args_only_config_name():
    """Test with only config-name at the end."""
    extra_args = "++inference.temperature=0.7 ++tokens_to_generate=512 --config-name my_config"
    hydra_args, override_args = separate_hydra_args(extra_args)
    assert hydra_args == " --config-name my_config"
    assert override_args == " ++inference.temperature=0.7 ++tokens_to_generate=512"


def test_separate_hydra_args_with_spaces_in_values():
    """Test with parameter values containing spaces (using quotes)."""
    extra_args = '--config-path "/path with spaces" ++description="some text with spaces" --config-name my_config'
    hydra_args, override_args = separate_hydra_args(extra_args)
    assert hydra_args == " --config-path /path with spaces --config-name my_config"
    assert override_args == " ++description=some text with spaces"


def test_separate_hydra_args_with_quoted_special_chars():
    """Test with quoted values containing special characters."""
    extra_args = """--config-path /configs ++end_reasoning_string="END_TAG" ++prompt="Question: {question}" """
    hydra_args, override_args = separate_hydra_args(extra_args)
    assert hydra_args == " --config-path /configs"
    assert override_args == " ++end_reasoning_string=END_TAG ++prompt=Question: {question}"


# --- SandboxScript.keep_mounts wiring tests ---


@patch("nemo_skills.pipeline.utils.scripts.sandbox_command", return_value=("echo sandbox", {}))
@patch("nemo_skills.pipeline.utils.scripts.get_free_port", return_value=12345)
def test_sandbox_keep_mounts_false_produces_empty_mounts(mock_port, mock_cmd):
    """Default keep_mounts=False must produce mounts=[] so the sandbox is filesystem-isolated.

    This test revealed a pre-fix bug: Command.prepare_for_execution always emitted
    mounts=None regardless of SandboxScript.keep_mounts, silently granting the sandbox
    full cluster filesystem access even when the user explicitly left keep_mounts=False.
    """
    from nemo_skills.pipeline.utils.scripts import SandboxScript

    cluster_config = {"executor": "slurm", "containers": {"sandbox": "sandbox:latest"}}
    sandbox = SandboxScript(cluster_config=cluster_config, keep_mounts=False)
    cmd = Command(script=sandbox, container="sandbox:latest", name="sandbox")
    _, exec_config = cmd.prepare_for_execution(cluster_config)
    assert exec_config["mounts"] == [], (
        "keep_mounts=False must yield mounts=[] to isolate the sandbox from cluster filesystems"
    )


@patch("nemo_skills.pipeline.utils.scripts.sandbox_command", return_value=("echo sandbox", {}))
@patch("nemo_skills.pipeline.utils.scripts.get_free_port", return_value=12345)
def test_sandbox_keep_mounts_true_produces_none_mounts(mock_port, mock_cmd):
    """keep_mounts=True must produce mounts=None so the sandbox inherits cluster mounts."""
    from nemo_skills.pipeline.utils.scripts import SandboxScript

    cluster_config = {"executor": "slurm", "containers": {"sandbox": "sandbox:latest"}}
    sandbox = SandboxScript(cluster_config=cluster_config, keep_mounts=True)
    cmd = Command(script=sandbox, container="sandbox:latest", name="sandbox")
    _, exec_config = cmd.prepare_for_execution(cluster_config)
    assert exec_config["mounts"] is None, (
        "keep_mounts=True must yield mounts=None so get_executor inherits cluster config mounts"
    )


def test_non_sandbox_command_mounts_unchanged():
    """Non-SandboxScript commands must still produce mounts=None (inherit cluster mounts)."""
    import nemo_run as run

    script = run.Script(inline="echo hello")
    cmd = Command(script=script, container="nemo-skills:latest", name="client")
    _, exec_config = cmd.prepare_for_execution({"executor": "slurm"})
    assert exec_config["mounts"] is None, "Non-sandbox commands should inherit cluster mounts (mounts=None)"


# --- sandbox_mounts resolution: full 2x2 matrix (keep_mounts_for_sandbox x sandbox_mounts) ---
#
# Priority order: sandbox_mounts (if non-empty) > keep_mounts_for_sandbox > safe default ([])
#
# sandbox_mounts present  (any keep_mounts)      -> sandbox_mounts exactly  (cases 2 & 4)
# sandbox_mounts absent + keep_mounts=False      -> []     safe default     (case 1, existing test)
# sandbox_mounts absent + keep_mounts=True       -> None   inherit cluster  (case 3, existing test)


@patch("nemo_skills.pipeline.utils.scripts.sandbox_command", return_value=("echo sandbox", {}))
@patch("nemo_skills.pipeline.utils.scripts.get_free_port", return_value=12345)
def test_sandbox_mounts_used_when_keep_mounts_false(mock_port, mock_cmd):
    """Case 2: keep_mounts_for_sandbox=False + sandbox_mounts defined -> sandbox gets exactly those mounts.

    This is the primary new code path: sandbox_mounts in cluster config replaces the default
    empty-list fallback, giving the sandbox access to specific paths without granting full
    cluster filesystem access (which keep_mounts_for_sandbox=True would do).
    """
    from nemo_skills.pipeline.utils.scripts import SandboxScript

    cluster_config = {
        "executor": "slurm",
        "containers": {"sandbox": "sandbox:latest"},
        "sandbox_mounts": ["/data/sandbox:/sandbox-data", "/models/readonly:/models:ro"],
    }
    # keep_mounts_for_sandbox=False is the default; mirrors how generate.py constructs SandboxScript
    sandbox = SandboxScript(cluster_config=cluster_config, keep_mounts=False)
    cmd = Command(script=sandbox, container="sandbox:latest", name="sandbox")
    _, exec_config = cmd.prepare_for_execution(cluster_config)

    assert exec_config["mounts"] == ["/data/sandbox:/sandbox-data", "/models/readonly:/models:ro"], (
        "sandbox_mounts must be passed through exactly when keep_mounts_for_sandbox=False"
    )


@patch("nemo_skills.pipeline.utils.scripts.sandbox_command", return_value=("echo sandbox", {}))
@patch("nemo_skills.pipeline.utils.scripts.get_free_port", return_value=12345)
def test_sandbox_mounts_takes_precedence_over_keep_mounts_true(mock_port, mock_cmd):
    """Case 4: keep_mounts_for_sandbox=True + sandbox_mounts defined -> sandbox_mounts wins.

    sandbox_mounts in config takes precedence over keep_mounts_for_sandbox=True, making the
    risky 'inherit all cluster mounts' behaviour opt-out rather than opt-in when sandbox_mounts
    is explicitly configured.
    """
    from nemo_skills.pipeline.utils.scripts import SandboxScript

    cluster_config = {
        "executor": "slurm",
        "containers": {"sandbox": "sandbox:latest"},
        "sandbox_mounts": ["/data/sandbox:/sandbox-data"],
    }
    sandbox = SandboxScript(cluster_config=cluster_config, keep_mounts=True)
    cmd = Command(script=sandbox, container="sandbox:latest", name="sandbox")
    _, exec_config = cmd.prepare_for_execution(cluster_config)

    assert exec_config["mounts"] == ["/data/sandbox:/sandbox-data"], (
        "sandbox_mounts must take precedence over keep_mounts_for_sandbox=True"
    )


@patch("nemo_skills.pipeline.utils.scripts.sandbox_command", return_value=("echo sandbox", {}))
@patch("nemo_skills.pipeline.utils.scripts.get_free_port", return_value=12345)
def test_sandbox_mounts_env_var_expansion(mock_port, mock_cmd, monkeypatch):
    """sandbox_mounts entries with ${VAR} placeholders are resolved before being passed to executor."""
    from nemo_skills.pipeline.utils.scripts import SandboxScript

    monkeypatch.setenv("SANDBOX_DATA", "/cluster/storage/sandbox")
    cluster_config = {
        "executor": "slurm",
        "containers": {"sandbox": "sandbox:latest"},
        "sandbox_mounts": ["${SANDBOX_DATA}:/sandbox-data:ro"],
    }
    sandbox = SandboxScript(cluster_config=cluster_config, keep_mounts=False)
    cmd = Command(script=sandbox, container="sandbox:latest", name="sandbox")
    _, exec_config = cmd.prepare_for_execution(cluster_config)

    assert exec_config["mounts"] == ["/cluster/storage/sandbox:/sandbox-data:ro"]


@patch("nemo_skills.pipeline.utils.scripts.sandbox_command", return_value=("echo sandbox", {}))
@patch("nemo_skills.pipeline.utils.scripts.get_free_port", return_value=12345)
def test_sandbox_mounts_absent_falls_back_to_empty(mock_port, mock_cmd):
    """sandbox_mounts absent with keep_mounts=False falls back to [] (no filesystem access).

    Explicit counterpart to test_sandbox_mounts_used_when_keep_mounts_false: confirms the
    fallback path is taken when sandbox_mounts is not in the cluster config at all, even when
    a regular mounts section is present.
    """
    from nemo_skills.pipeline.utils.scripts import SandboxScript

    cluster_config = {
        "executor": "slurm",
        "containers": {"sandbox": "sandbox:latest"},
        "mounts": ["/models:/models"],  # regular mounts present but no sandbox_mounts
    }
    sandbox = SandboxScript(cluster_config=cluster_config, keep_mounts=False)
    cmd = Command(script=sandbox, container="sandbox:latest", name="sandbox")
    _, exec_config = cmd.prepare_for_execution(cluster_config)

    assert exec_config["mounts"] == [], (
        "without sandbox_mounts, keep_mounts=False must fall back to [] not the regular cluster mounts"
    )


# --- get_executor: mounts parameter overrides cluster config mounts ---


@patch("nemo_skills.pipeline.utils.exp.get_packager")
@patch("nemo_skills.pipeline.utils.exp.resolve_container_image", return_value="nemo:latest")
def test_get_executor_mounts_none_falls_back_to_config(mock_resolve, mock_packager):
    """get_executor with mounts=None should use cluster config mounts."""
    from nemo_skills.pipeline.utils.exp import get_executor

    cluster_config = {"executor": "local", "mounts": ["/host/models:/models"]}
    executor = get_executor(
        cluster_config=cluster_config,
        container="nemo:latest",
        num_nodes=1,
        tasks_per_node=1,
        gpus_per_node=0,
        job_name="test",
        log_dir="/tmp",
        mounts=None,
    )
    assert executor.volumes == ["/host/models:/models"]


@patch("nemo_skills.pipeline.utils.exp.get_packager")
@patch("nemo_skills.pipeline.utils.exp.resolve_container_image", return_value="nemo:latest")
def test_get_executor_empty_mounts_overrides_config(mock_resolve, mock_packager):
    """get_executor with mounts=[] should use empty list, not cluster config mounts."""
    from nemo_skills.pipeline.utils.exp import get_executor

    cluster_config = {"executor": "local", "mounts": ["/host/models:/models"]}
    executor = get_executor(
        cluster_config=cluster_config,
        container="nemo:latest",
        num_nodes=1,
        tasks_per_node=1,
        gpus_per_node=0,
        job_name="test",
        log_dir="/tmp",
        mounts=[],
    )
    assert executor.volumes == []


@patch("nemo_skills.pipeline.utils.exp.get_packager")
@patch("nemo_skills.pipeline.utils.exp.resolve_container_image", return_value="nemo:latest")
def test_get_executor_explicit_mounts_overrides_config(mock_resolve, mock_packager):
    """get_executor with explicit mounts should use those, not cluster config mounts."""
    from nemo_skills.pipeline.utils.exp import get_executor

    cluster_config = {"executor": "local", "mounts": ["/host/models:/models"]}
    executor = get_executor(
        cluster_config=cluster_config,
        container="nemo:latest",
        num_nodes=1,
        tasks_per_node=1,
        gpus_per_node=0,
        job_name="test",
        log_dir="/tmp",
        mounts=["/sandbox/data:/data:ro"],
    )
    assert executor.volumes == ["/sandbox/data:/data:ro"]


# --- add_task: sandbox mount resolution ---


@patch("nemo_skills.pipeline.utils.exp.get_executor")
@patch("nemo_skills.pipeline.utils.exp.get_free_port", return_value=12345)
def test_add_task_sandbox_mounts_used_when_configured(mock_port, mock_get_executor):
    """add_task: sandbox_mounts in config takes precedence over keep_mounts_for_sandbox."""
    from types import SimpleNamespace

    from nemo_skills.pipeline.utils.exp import add_task

    mock_get_executor.return_value = MagicMock()
    exp = SimpleNamespace(add=MagicMock(return_value="task_handle"))
    cluster_config = {
        "executor": "local",
        "containers": {"sandbox": "sandbox:latest"},
        "sandbox_mounts": ["/host/data:/sandbox/data:ro"],
    }

    add_task(
        exp=exp,
        cmd="echo hello",
        task_name="test-task",
        cluster_config=cluster_config,
        container="main:latest",
        log_dir="/tmp/logs",
        with_sandbox=True,
        keep_mounts_for_sandbox=False,
        skip_hf_home_check=True,
        reuse_code=False,
    )

    sandbox_mounts = mock_get_executor.call_args_list[-1].kwargs["mounts"]
    assert sandbox_mounts == ["/host/data:/sandbox/data:ro"]


@patch("nemo_skills.pipeline.utils.exp.get_executor")
@patch("nemo_skills.pipeline.utils.exp.get_free_port", return_value=12345)
def test_add_task_keep_mounts_for_sandbox_true_no_sandbox_mounts(mock_port, mock_get_executor):
    """add_task: keep_mounts_for_sandbox=True with no sandbox_mounts → None (inherit cluster mounts)."""
    from types import SimpleNamespace

    from nemo_skills.pipeline.utils.exp import add_task

    mock_get_executor.return_value = MagicMock()
    exp = SimpleNamespace(add=MagicMock(return_value="task_handle"))
    cluster_config = {
        "executor": "local",
        "containers": {"sandbox": "sandbox:latest"},
    }

    add_task(
        exp=exp,
        cmd="echo hello",
        task_name="test-task",
        cluster_config=cluster_config,
        container="main:latest",
        log_dir="/tmp/logs",
        with_sandbox=True,
        keep_mounts_for_sandbox=True,
        skip_hf_home_check=True,
        reuse_code=False,
    )

    sandbox_mounts = mock_get_executor.call_args_list[-1].kwargs["mounts"]
    assert sandbox_mounts is None


@patch("nemo_skills.pipeline.utils.exp.get_executor")
@patch("nemo_skills.pipeline.utils.exp.get_free_port", return_value=12345)
def test_add_task_no_sandbox_mounts_falls_back_to_empty(mock_port, mock_get_executor):
    """add_task: no sandbox_mounts + keep_mounts_for_sandbox=False → [] (safe default)."""
    from types import SimpleNamespace

    from nemo_skills.pipeline.utils.exp import add_task

    mock_get_executor.return_value = MagicMock()
    exp = SimpleNamespace(add=MagicMock(return_value="task_handle"))
    cluster_config = {
        "executor": "local",
        "containers": {"sandbox": "sandbox:latest"},
    }

    add_task(
        exp=exp,
        cmd="echo hello",
        task_name="test-task",
        cluster_config=cluster_config,
        container="main:latest",
        log_dir="/tmp/logs",
        with_sandbox=True,
        keep_mounts_for_sandbox=False,
        skip_hf_home_check=True,
        reuse_code=False,
    )

    sandbox_mounts = mock_get_executor.call_args_list[-1].kwargs["mounts"]
    assert sandbox_mounts == []

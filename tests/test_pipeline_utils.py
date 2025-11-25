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

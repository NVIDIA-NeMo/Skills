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

import os
import tempfile
from unittest.mock import patch

import yaml
from typer.testing import CliRunner

from nemo_skills.pipeline.cli import app

runner = CliRunner()


def test_local_setup_crashes_without_login_session():
    """ns setup crashes in containers/VMs when creating a local config.

    os.getlogin() is called on line 106 to compute the default mount path.
    In environments without a login session (containers, VMs, systemd
    services), this raises OSError before the user can provide input.

    See: https://github.com/NVIDIA-NeMo/Skills/issues/1269
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("os.getlogin", side_effect=OSError("[Errno 6] No such device or address")):
            prompts = "\n".join(
                [
                    tmpdir,  # config dir
                    "local",  # config type
                    "local",  # config name
                    "/tmp:/workspace",  # mounts (never reached — default eval crashes)
                    "",  # HF_HOME
                    "",  # env vars
                    "n",  # pull containers
                    "n",  # create another
                ]
            )
            result = runner.invoke(app, ["setup"], input=prompts)

            assert result.exit_code != 0, (
                f"Expected crash from os.getlogin() OSError, but exited with code {result.exit_code}.\n"
                f"Output: {result.output}"
            )
            assert isinstance(result.exception, OSError), (
                f"Expected OSError, got {type(result.exception).__name__}: {result.exception}"
            )


def test_slurm_setup_crashes_without_login_session():
    """ns setup crashes in containers/VMs when creating a slurm config with SSH.

    os.getlogin() is called on line 175 as the default SSH username.
    The mounts prompt is unaffected (default is None for slurm), but
    the SSH username prompt crashes.

    See: https://github.com/NVIDIA-NeMo/Skills/issues/1269
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("os.getlogin", side_effect=OSError("[Errno 6] No such device or address")):
            prompts = "\n".join(
                [
                    tmpdir,  # config dir
                    "slurm",  # config type
                    "slurm",  # config name
                    "/lustre:/lustre",  # mounts (OK — default is None for slurm)
                    "",  # HF_HOME
                    "",  # env vars
                    "y",  # SSH access
                    "cluster.example.com",  # SSH hostname
                    "testuser",  # SSH username (never reached — default eval crashes)
                    "",  # SSH key
                    "/tmp/jobs",  # job dir
                    "myaccount",  # account
                    "batch",  # partition
                    "",  # timeouts
                    "n",  # create another
                ]
            )
            result = runner.invoke(app, ["setup"], input=prompts)

            assert result.exit_code != 0, (
                f"Expected crash from os.getlogin() OSError, but exited with code {result.exit_code}.\n"
                f"Output: {result.output}"
            )
            assert isinstance(result.exception, OSError), (
                f"Expected OSError, got {type(result.exception).__name__}: {result.exception}"
            )


def test_local_setup_succeeds_with_login_session():
    """Sanity check: ns setup works when os.getlogin() is available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("os.getlogin", return_value="testuser"):
            prompts = "\n".join(
                [
                    tmpdir,  # config dir
                    "local",  # config type
                    "local",  # config name
                    "/tmp:/workspace",  # mounts
                    "",  # HF_HOME (skip)
                    "",  # env vars
                    "n",  # pull containers
                    "n",  # create another
                ]
            )
            result = runner.invoke(app, ["setup"], input=prompts)

            assert result.exit_code == 0, (
                f"Expected success, but exited with code {result.exit_code}.\n"
                f"Output: {result.output}\nException: {result.exception}"
            )

            config_file = os.path.join(tmpdir, "local.yaml")
            assert os.path.exists(config_file)
            with open(config_file) as f:
                config = yaml.safe_load(f)
            assert config["executor"] == "local"
            assert "/tmp:/workspace" in config["mounts"]

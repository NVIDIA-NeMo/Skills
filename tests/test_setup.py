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


def test_local_setup_without_login_session():
    """ns setup should work in containers/VMs even when os.getlogin() fails.

    Regression test for https://github.com/NVIDIA-NeMo/Skills/issues/1269
    The default mount path now uses os.path.expanduser('~') instead of
    os.getlogin(), so it works in environments without a login session.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("os.getlogin", side_effect=OSError("[Errno 6] No such device or address")):
            prompts = "\n".join(
                [
                    tmpdir,  # config dir
                    "local",  # config type
                    "local",  # config name
                    "/tmp:/workspace",  # mounts
                    "",  # HF_HOME
                    "",  # env vars
                    "n",  # pull containers
                    "n",  # create another
                ]
            )
            result = runner.invoke(app, ["setup"], input=prompts)

            assert result.exit_code == 0, (
                f"Expected setup to succeed without os.getlogin(), but exited with code {result.exit_code}.\n"
                f"Output: {result.output}\nException: {result.exception}"
            )

            config_file = os.path.join(tmpdir, "local.yaml")
            assert os.path.exists(config_file)
            with open(config_file) as f:
                config = yaml.safe_load(f)
            assert config["executor"] == "local"
            assert "/tmp:/workspace" in config["mounts"]


def test_slurm_setup_without_login_session():
    """ns setup should work for slurm+SSH even when os.getlogin() fails.

    Regression test for https://github.com/NVIDIA-NeMo/Skills/issues/1269
    The default SSH username now uses getpass.getuser() instead of
    os.getlogin(), so it works in environments without a login session.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("os.getlogin", side_effect=OSError("[Errno 6] No such device or address")):
            prompts = "\n".join(
                [
                    tmpdir,  # config dir
                    "slurm",  # config type
                    "slurm",  # config name
                    "/lustre:/lustre",  # mounts
                    "",  # HF_HOME
                    "",  # env vars
                    "y",  # SSH access
                    "cluster.example.com",  # SSH hostname
                    "testuser",  # SSH username
                    "",  # SSH key
                    "/tmp/jobs",  # job dir
                    "myaccount",  # account
                    "batch",  # partition
                    "",  # timeouts
                    "n",  # create another
                ]
            )
            result = runner.invoke(app, ["setup"], input=prompts)

            assert result.exit_code == 0, (
                f"Expected setup to succeed without os.getlogin(), but exited with code {result.exit_code}.\n"
                f"Output: {result.output}\nException: {result.exception}"
            )

            config_file = os.path.join(tmpdir, "slurm.yaml")
            assert os.path.exists(config_file)
            with open(config_file) as f:
                config = yaml.safe_load(f)
            assert config["executor"] == "slurm"
            assert config["ssh_tunnel"]["user"] == "testuser"


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

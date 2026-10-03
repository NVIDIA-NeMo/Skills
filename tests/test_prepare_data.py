# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import importlib
import shlex
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

prepare_module = importlib.import_module("nemo_skills.pipeline.prepare_data")


@pytest.fixture
def submitted_command(monkeypatch):
    monkeypatch.delenv("NEMO_SKILLS_CONFIG", raising=False)
    monkeypatch.delenv("NEMO_SKILLS_CONFIG_DIR", raising=False)
    monkeypatch.delenv("NEMO_SKILLS_EXTRA_BENCHMARK_MAP", raising=False)
    monkeypatch.setattr(prepare_module, "get_dataset_module", lambda dataset: (SimpleNamespace(), None))
    run_cmd = Mock(return_value="submitted")
    monkeypatch.setattr(prepare_module, "_run_cmd", run_cmd)
    return run_cmd


@pytest.mark.parametrize("data_dir", ["prepared", "prepared data"])
def test_local_data_dir_without_cluster(submitted_command, data_dir):
    ctx = SimpleNamespace(args=["gsm8k"])
    assert prepare_module.prepare_data(ctx=ctx, data_dir=data_dir) == "submitted"
    kwargs = submitted_command.call_args.kwargs
    assert kwargs["cluster"]["executor"] == "none"
    source = prepare_module.get_dataset_path("gsm8k")
    assert f"cp -r {shlex.quote(str(source) + '/.')} {shlex.quote(data_dir + '/gsm8k/')}" in kwargs["command"]
    assert "/nemo_run/code" not in kwargs["command"]
    assert kwargs["log_dir"] == data_dir
    assert ctx.args == []


def test_local_external_dataset_uses_host_path(submitted_command, monkeypatch):
    dataset = "/local/external benchmark"
    monkeypatch.setattr(prepare_module, "get_dataset_name", lambda name: "external benchmark")
    ctx = SimpleNamespace(args=[dataset])
    prepare_module.prepare_data(ctx=ctx, data_dir="prepared data")
    command = submitted_command.call_args.kwargs["command"]
    assert shlex.quote(dataset) in command
    source = str(prepare_module.get_dataset_path(dataset)) + "/."
    assert f"cp -r {shlex.quote(source)} {shlex.quote('prepared data/external benchmark/')}" in command


@pytest.mark.parametrize("executor", ["local", "slurm"])
def test_container_data_dir_preserves_container_paths(submitted_command, monkeypatch, executor):
    monkeypatch.setattr(prepare_module, "get_cluster_config", lambda *args, **kwargs: {"executor": executor})
    prepare_module.prepare_data(ctx=SimpleNamespace(args=["gsm8k"]), cluster="configured", data_dir="/data")
    command = submitted_command.call_args.kwargs["command"]
    assert "cp -r /nemo_run/code/nemo_skills/dataset/gsm8k/. /data/gsm8k/" in command
    assert submitted_command.call_args.kwargs["cluster"]["executor"] == executor


def test_data_dir_respects_environment_cluster(submitted_command, monkeypatch, tmp_path):
    config = tmp_path / "cluster.yaml"
    config.write_text("executor: local\ncontainers: {}\n", encoding="utf-8")
    monkeypatch.setenv("NEMO_SKILLS_CONFIG", str(config))
    prepare_module.prepare_data(ctx=SimpleNamespace(args=["gsm8k"]), data_dir="/data")
    assert submitted_command.call_args.kwargs["cluster"]["executor"] == "local"
    assert "/nemo_run/code/nemo_skills/dataset/gsm8k/." in submitted_command.call_args.kwargs["command"]


def test_slurm_still_requires_data_dir(submitted_command, monkeypatch):
    monkeypatch.setattr(prepare_module, "get_cluster_config", lambda *args, **kwargs: {"executor": "slurm"})
    with pytest.raises(ValueError, match="Data directory is required"):
        prepare_module.prepare_data(ctx=SimpleNamespace(args=["gsm8k"]), cluster="configured")
    submitted_command.assert_not_called()


def test_local_without_data_dir_does_not_copy(submitted_command):
    prepare_module.prepare_data(ctx=SimpleNamespace(args=["gsm8k"]))
    assert "cp -r" not in submitted_command.call_args.kwargs["command"]

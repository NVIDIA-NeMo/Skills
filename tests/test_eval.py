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

import asyncio
import json
import os
import re
import shlex
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import hydra
import pytest
from omegaconf import ListConfig

import nemo_skills.pipeline.utils.scripts.eval as eval_scripts
from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig
from nemo_skills.pipeline.eval import eval as eval_fn
from nemo_skills.pipeline.utils import eval as eval_utils
from nemo_skills.pipeline.utils.scripts import EvalClientScript
from nemo_skills.utils import nested_dataclass


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


def _normalize_multi_model_value(value):
    if isinstance(value, ListConfig):
        return list(value)
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _normalize_server_address(address: str) -> str:
    if not address.startswith(("http://", "https://")):
        address = f"http://{address}"
    if not address.endswith("/v1"):
        address = f"{address}/v1"
    return address


def _contains_expected_answer(generation: str, expected_answer: str) -> bool:
    return re.search(rf"\b{re.escape(expected_answer)}\b", generation) is not None


@nested_dataclass(kw_only=True)
class MultiModelEvalSmokeConfig(GenerationTaskConfig):
    prompt_format: str = "openai"

    def __post_init__(self):
        base_urls = _normalize_multi_model_value(self.server.get("base_url"))
        models = _normalize_multi_model_value(self.server.get("model"))
        server_types = _normalize_multi_model_value(self.server.get("server_type"))

        if len(base_urls) != 2:
            raise ValueError(f"Expected exactly 2 server.base_url values, got {len(base_urls)}")

        if len(models) == 1:
            models = models * len(base_urls)
        elif len(models) != len(base_urls):
            raise ValueError(f"Expected server.model to have 1 or {len(base_urls)} values, got {len(models)}")

        if len(server_types) == 1:
            server_types = server_types * len(base_urls)
        elif len(server_types) != len(base_urls):
            raise ValueError(
                f"Expected server.server_type to have 1 or {len(base_urls)} values, got {len(server_types)}"
            )

        self.server["base_url"] = base_urls
        self.server["model"] = models
        self.server["server_type"] = server_types
        super().__post_init__()


_MULTI_MODEL_EVAL_SMOKE_CONFIG_NAME = "test_eval_multi_model_smoke_config"
hydra.core.config_store.ConfigStore.instance().store(
    name=_MULTI_MODEL_EVAL_SMOKE_CONFIG_NAME,
    node=MultiModelEvalSmokeConfig,
)


class MultiModelEvalSmokeTask(GenerationTask):
    """Tiny test-only generation task that verifies two live model endpoints work."""

    def setup_prompt(self):
        return None

    def log_example_prompt(self, data):
        return

    def setup_llm(self):
        from nemo_skills.inference.model import get_model

        self.server_addresses = list(self.cfg.server["base_url"])
        self.model_names = list(self.cfg.server["model"])
        self.server_types = list(self.cfg.server["server_type"])

        data_dir = str(Path(self.cfg.input_file).parent)
        output_dir = str(Path(self.cfg.output_file).parent)

        self.clients = []
        for address, model_name, server_type in zip(self.server_addresses, self.model_names, self.server_types):
            server_config = dict(self.cfg.server)
            server_config["base_url"] = _normalize_server_address(address)
            server_config["model"] = model_name
            server_config["server_type"] = server_type
            self.clients.append(
                get_model(
                    **server_config,
                    tokenizer=self.tokenizer,
                    data_dir=data_dir,
                    output_dir=output_dir,
                )
            )

        return self.clients[0]

    def wait_for_server(self):
        return

    async def process_single_datapoint(self, data_point, all_data, prompt_format=None):
        if is_dataclass(self.cfg.inference):
            inference_params = asdict(self.cfg.inference)
        else:
            inference_params = dict(self.cfg.inference)

        outputs = await asyncio.gather(
            *(client.generate_async(prompt=data_point["messages"], **inference_params) for client in self.clients)
        )
        generations = [str(output["generation"]).strip() for output in outputs]
        expected_answer = str(data_point["expected_answer"])

        if not all(_contains_expected_answer(generation, expected_answer) for generation in generations):
            raise AssertionError(f"Expected both generations to contain {expected_answer!r}, got: {generations}")

        return {
            "generation": generations[0],
            "generation_model_0": generations[0],
            "generation_model_1": generations[1],
            "predicted_answer": expected_answer,
            "expected_answer": expected_answer,
            "symbolic_correct": True,
        }


GENERATION_TASK_CLASS = MultiModelEvalSmokeTask


@hydra.main(version_base=None, config_name=_MULTI_MODEL_EVAL_SMOKE_CONFIG_NAME)
def run_multi_model_eval_smoke(cfg: MultiModelEvalSmokeConfig):
    cfg = MultiModelEvalSmokeConfig(_init_nested=True, **cfg)
    task = MultiModelEvalSmokeTask(cfg)
    task.generate()


def _write_multi_model_benchmark(benchmark_dir: Path) -> None:
    benchmark_dir.mkdir()
    (benchmark_dir / "__init__.py").write_text('METRICS_TYPE = "math"\n', encoding="utf-8")
    sample = {
        "id": "sample-0",
        "problem": "What is 2 + 2?",
        "expected_answer": "4",
        "messages": [{"role": "user", "content": "What is 2 + 2? Reply with only the digit 4."}],
    }
    (benchmark_dir / "test.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")


@pytest.mark.timeout(300)
@pytest.mark.skipif("NVIDIA_API_KEY" not in os.environ, reason="requires NVIDIA_API_KEY")
def test_eval_multi_model_generation_module_smoke(tmp_path):
    benchmark_dir = tmp_path / "multi_model_eval_smoke"
    output_dir = tmp_path / "out"
    _write_multi_model_benchmark(benchmark_dir)

    eval_fn(
        ctx=SimpleNamespace(
            args=[
                "++max_concurrent_requests=1",
                "++inference.timeout=120",
                "++server.max_retries=1",
            ]
        ),
        output_dir=str(output_dir),
        benchmarks=str(benchmark_dir),
        model=[
            "nvidia/nemotron-3-nano-30b-a3b",
            "nvidia/nemotron-3-nano-30b-a3b",
        ],
        server_type=["openai", "openai"],
        server_address=[
            "https://integrate.api.nvidia.com/v1",
            "https://integrate.api.nvidia.com/v1",
        ],
        generation_module=str(Path(__file__).resolve()),
        auto_summarize_results=False,
    )

    output_file = output_dir / "eval-results" / benchmark_dir.name / "output.jsonl"
    with output_file.open(encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]

    assert len(data) == 1
    assert data[0]["predicted_answer"] == "4"
    assert data[0]["symbolic_correct"] is True
    assert _contains_expected_answer(data[0]["generation_model_0"], "4")
    assert _contains_expected_answer(data[0]["generation_model_1"], "4")


if __name__ == "__main__":
    run_multi_model_eval_smoke()

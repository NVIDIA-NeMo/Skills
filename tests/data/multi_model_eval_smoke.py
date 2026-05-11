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
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import hydra
from omegaconf import ListConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig
from nemo_skills.utils import nested_dataclass


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


_MULTI_MODEL_EVAL_SMOKE_CONFIG_NAME = "multi_model_eval_smoke_config"
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

        expected_answer = str(data_point["expected_answer"])
        prompt = [
            {
                "role": "user",
                "content": (
                    "This is a NeMo-Skills multi-model eval smoke test. "
                    f"Reply with exactly this text and nothing else: {expected_answer}"
                ),
            }
        ]
        outputs = await asyncio.gather(
            *(client.generate_async(prompt=prompt, **inference_params) for client in self.clients)
        )
        generations = [str(output["generation"]).strip() for output in outputs]
        model_0_exact_match = generations[0] == expected_answer
        model_1_exact_match = generations[1] == expected_answer

        if not (model_0_exact_match and model_1_exact_match):
            raise AssertionError(f"Expected both generations to equal {expected_answer!r}, got: {generations}")

        return {
            "generation": f"\\boxed{{{expected_answer}}}",
            "generation_model_0": generations[0],
            "generation_model_1": generations[1],
            "model_0_exact_match": model_0_exact_match,
            "model_1_exact_match": model_1_exact_match,
            "predicted_answer": expected_answer,
            "expected_answer": expected_answer,
            "symbolic_correct": True,
        }


GENERATION_TASK_CLASS = MultiModelEvalSmokeTask


@hydra.main(version_base=None, config_name=_MULTI_MODEL_EVAL_SMOKE_CONFIG_NAME)
def generate(cfg: MultiModelEvalSmokeConfig):
    cfg = MultiModelEvalSmokeConfig(_init_nested=True, **cfg)
    task = MultiModelEvalSmokeTask(cfg)
    task.generate()


if __name__ == "__main__":
    generate()

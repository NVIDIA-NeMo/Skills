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

import importlib.util
import inspect
import logging
import os
import sys
from dataclasses import asdict, field, is_dataclass

import hydra

from nemo_skills.inference.generate import (
    GenerationTask,
    GenerationTaskConfig,
    InferenceConfig,
)
from nemo_skills.inference.model import server_params
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    setup_logging,
)


@nested_dataclass(kw_only=True)
class ScriptInferenceConfig(InferenceConfig):
    pass


@nested_dataclass(kw_only=True)
class ScriptGenerationConfig(GenerationTaskConfig):
    inference: ScriptInferenceConfig = field(default_factory=ScriptInferenceConfig)
    model_name: str | None = None
    prompt_format: str = "openai"
    script_program_path: str | None = None
    # Arbitrary kwargs passed to `module.process_single`
    script_config: dict = field(default_factory=dict)

    def _get_disallowed_params(self):
        return [
            ("prompt_config", None),
            ("generation_key", "generation"),
            ("prompt_format", "openai"),
            ("code_execution", False),
        ]


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_script_generation_config", node=ScriptGenerationConfig)


class ScriptGenerationTask(GenerationTask):
    def log_example_prompt(self, data):
        return

    def setup_prompt(self):
        # Ignore parent prompt handling
        return None

    def setup_llm(self):
        # Configure networking for high-throughput requests via LiteLLM
        nemo_llm = super().setup_llm()

        # Load the script program module once
        module_name = os.path.splitext(os.path.basename(self.cfg.script_program_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, self.cfg.script_program_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self._script_module = module

        if is_dataclass(self.cfg.inference):
            inference_params = asdict(self.cfg.inference)
        else:
            # Already a dict from Hydra
            inference_params = dict(self.cfg.inference)
        # Take random seed out of inference params, we manually add it if needed
        assert "random_seed" in inference_params, "Random seed must be specified in inference params"
        self.random_seed = inference_params.pop("random_seed")

        self.llm_kwargs = inference_params
        return nemo_llm

    async def process_single_datapoint(self, data_point, all_data, prompt_format=None):
        # Delegate processing to the user-provided script program
        # Get the function signature to check if llm is a parameter
        sig = inspect.signature(self._script_module.process_single)
        kwargs = {"datapoint": data_point, **self.cfg.script_config}
        if "llm" in sig.parameters:
            kwargs["llm"] = self.llm
            kwargs["llm_kwargs"] = self.llm_kwargs
        if "random_seed" in sig.parameters:
            kwargs["random_seed"] = self.random_seed
        async with self.semaphore:
            result = await self._script_module.process_single(**kwargs)
            if "generation" not in result:
                result["generation"] = "dummy generation key"  # To avoid error in dumping

        return result


GENERATION_TASK_CLASS = ScriptGenerationTask


@hydra.main(version_base=None, config_name="base_script_generation_config")
def script_generation(cfg: ScriptGenerationConfig):
    cfg = ScriptGenerationConfig(_init_nested=True, **cfg)
    LOG = logging.getLogger(get_logger_name(__file__))
    LOG.info("Config used: %s", cfg)

    task = ScriptGenerationTask(cfg)
    task.generate()


if __name__ == "__main__":
    HELP_MESSAGE = get_help_message(
        ScriptGenerationConfig,
        server_params=server_params(),
    )
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        script_generation()

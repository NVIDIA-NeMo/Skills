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

import logging
import sys
import time
from dataclasses import asdict, field
from pathlib import Path

import hydra

from nemo_skills.inference.generate import (
    GenerationTask,
    GenerationTaskConfig,
    InferenceConfig,
)
from nemo_skills.inference.model import server_params
from nemo_skills.inference.model.nim_utils import ASRExtraConfig, TTSExtraConfig
from nemo_skills.utils import get_help_message, nested_dataclass, setup_logging

LOG = logging.getLogger(__name__)


@nested_dataclass(kw_only=True)
class RivaGenerateConfig(GenerationTaskConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)

    generation_type: str = "tts"  # 'tts' or 'asr'
    voice: str = "Magpie-Multilingual.EN-US.Sofia"
    tts_output_dir: str = "/tmp/tts_outputs"
    language_code: str = "en-US"
    sample_rate_hz: int = 22050

    generation_key: str = "result"
    prompt_config: str | None = None
    prompt_format: str = "openai"  # Bypass prompt validation for TTS/ASR


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_riva_generate_config", node=RivaGenerateConfig)


class RivaGenerationTask(GenerationTask):
    def __init__(self, cfg: RivaGenerateConfig):
        super().__init__(cfg)

    def wait_for_server(self):
        """Override to check Riva HTTP health endpoint."""
        http_port = int(self.cfg.server.get("port", "8000"))
        host = self.cfg.server.get("host", "127.0.0.1")

        LOG.warning(f"Checking Riva NIM server health at {host}:{http_port}")
        import requests

        # Try multiple common health check endpoints
        health_endpoints = [
            "/v1/health/ready",
            "/v1/health/live",
            "/health/ready",
            "/health",
            "/v1/models",
        ]

        max_attempts = 120  # Wait up to 20 minutes
        sleep_time_between_attempts = 10
        sleep_time_between_endpoints = 3

        for attempt in range(max_attempts):
            for endpoint in health_endpoints:
                try:
                    time.sleep(sleep_time_between_endpoints)
                    response = requests.get(f"http://{host}:{http_port}{endpoint}", timeout=5)
                    if response.status_code == 200:
                        LOG.warning(f"Server is ready at {host}:{http_port} (endpoint: {endpoint})")
                        # Give it a few extra seconds to fully stabilize
                        return
                except requests.exceptions.RequestException:
                    # Try next endpoint
                    continue

            # All endpoints failed, log and retry
            LOG.warning(f"Health check attempt {attempt + 1}/{max_attempts} - all endpoints failed")
            if attempt < max_attempts - 1:
                time.sleep(sleep_time_between_attempts)

        raise ValueError(
            f"Server health check did not pass after {max_attempts * sleep_time_between_attempts * len(health_endpoints) * sleep_time_between_endpoints} seconds. Check your server configuration."
        )

    def setup_llm(self):
        host = self.cfg.server.get("host", "127.0.0.1")
        grpc_port = str(int(self.cfg.server.get("port", "8000")) + 1)

        if self.cfg.generation_type == "tts":
            from nemo_skills.inference.model.tts_nim import TTSNIMModel

            Path(self.cfg.tts_output_dir).mkdir(parents=True, exist_ok=True)
            return TTSNIMModel(
                host=host,
                port=grpc_port,
                model="riva-tts",  # Required by BaseModel
                voice=self.cfg.voice,
                language_code=self.cfg.language_code,
                sample_rate_hz=self.cfg.sample_rate_hz,
                output_dir=self.cfg.tts_output_dir,
            )
        elif self.cfg.generation_type == "asr":
            from nemo_skills.inference.model.asr_nim import ASRNIMModel

            return ASRNIMModel(
                host=host,
                port=grpc_port,
                model="riva-asr",  # Required by BaseModel
                language_code=self.cfg.language_code,
            )
        else:
            raise ValueError(f"Invalid generation type: {self.cfg.generation_type}")

    def setup_prompt(self):
        return None

    def fill_prompt(self, data_point, all_data):
        if self.cfg.generation_type == "tts":
            return data_point.get("text", data_point.get("prompt", ""))
        else:
            return data_point.get("audio_path", data_point.get("audio_file", ""))

    def log_example_prompt(self, data):
        if data:
            LOG.info(f"Example input: {self.fill_prompt(data[0], data)}")

    async def process_single_datapoint(self, data_point, all_data):
        prompt = self.fill_prompt(data_point, all_data)
        if not prompt:
            raise ValueError(f"Empty input for datapoint {data_point}")

        # At this stagee extra_body can contain only items picked from the command line
        extra_body = dict(self.cfg.inference.extra_body) if self.cfg.inference.extra_body else {}

        # Now we want to check that all fields in data_point except main one are present
        # in TTS or ASR configs and override extra_body from the command line
        data_params = dict(data_point)
        for key in data_point.keys():
            if key in ["text", "prompt", "audio_path", "audio_file"] or key[0] == "_":
                data_params.pop(key, None)

        if self.cfg.generation_type == "tts":
            config = TTSExtraConfig(**data_params)
        elif self.cfg.generation_type == "asr":
            config = ASRExtraConfig(**data_params)
        else:
            raise ValueError(f"Invalid generation type: {self.cfg.generation_type}")

        extra_body.update({k: v for k, v in asdict(config).items() if v is not None})

        return await self.llm.generate_async(
            prompt=prompt, tokens_to_generate=1, temperature=0.0, extra_body=extra_body, **self.extra_generate_params
        )


GENERATION_TASK_CLASS = RivaGenerationTask


@hydra.main(version_base=None, config_name="base_riva_generate_config")
def generate(cfg: RivaGenerateConfig):
    cfg = RivaGenerateConfig(_init_nested=True, **cfg)
    task = RivaGenerationTask(cfg)
    task.generate()


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(get_help_message(RivaGenerateConfig, server_params=server_params()))
    else:
        setup_logging()
        generate()

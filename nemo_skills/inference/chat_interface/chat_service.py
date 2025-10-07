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

from __future__ import annotations

import asyncio
import logging
from typing import Iterator

from nemo_skills.inference.chat_interface.core import AppConfig, CodeExecStatus, ModelLoader, PromptManager

logger = logging.getLogger(__name__)


class ChatService:
    """Combines LLM, prompt, and (optionally) sandbox into a streaming chat."""

    def __init__(self, loader: ModelLoader, prompts: PromptManager):
        self._loader = loader
        self._prompts = prompts

    def stream_chat(
        self,
        turns: list[dict],
        tokens_to_generate: int,
        temperature: float,
        status: CodeExecStatus,
        prompt_config_override: str | None = None,
    ) -> Iterator[str]:
        """Yield the bot response incrementally as plain text chunks."""

        use_code = status == CodeExecStatus.ENABLED
        llm = self._loader.code_llm if use_code else self._loader.generic_llm
        if llm is None:
            raise RuntimeError("No active LLM available.")

        prompt_obj = self._prompts.get(use_code, prompt_config_override)
        prompt_kwargs = {
            "turns": turns,
        }

        try:
            prompt_filled = prompt_obj.fill(prompt_kwargs, multi_turn_key="turns")
        except Exception as e:  # noqa: BLE001
            logger.exception("Prompt filling failed")
            raise RuntimeError(f"Error preparing prompt: {e}") from e

        extra_params = prompt_obj.get_code_execution_args() if use_code else {}
        stream_iter_list = asyncio.run(
            llm.generate_async(
                prompt=prompt_filled,
                tokens_to_generate=int(tokens_to_generate),
                temperature=float(temperature),
                stream=True,
                stop_phrases=prompt_obj.stop_phrases or [],
                **extra_params,
            )
        )
        if not stream_iter_list:
            raise RuntimeError("LLM did not return a stream iterator.")

        for delta in stream_iter_list:
            yield delta["generation"]


class AppContext:
    """Composition root injected into UI callbacks."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.loader = ModelLoader(cfg)
        self.prompts = PromptManager(cfg)

        self.chat = ChatService(self.loader, self.prompts)

        # Load only the models required by the declared capabilities.
        if self.cfg.supported_modes in ("cot", "both"):
            self.loader.load_generic()
        if self.cfg.supported_modes in ("tir", "both"):
            self.loader.load_code_and_sandbox()

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

from .openai import OpenAIModel


class AzureOpenAIModel(OpenAIModel):
    MODEL_PROVIDER = "azure"

    def __init__(
        self,
        *args,
        api_version: str = "2024-12-01-preview",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.litellm_kwargs["api_version"] = api_version

    def _get_api_key(self, api_key: str | None, api_key_env_var: str | None, base_url: str) -> str | None:
        # OPENAI_API_KEY is used by non-Azure OpenAI models
        api_key = super(OpenAIModel, self)._get_api_key(api_key, api_key_env_var, base_url)

        if api_key is None:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY is required for Azure models and could not be found.")
        return api_key

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

from nemo_skills.utils import python_doc_to_cmd_help

from .azure import AzureOpenAIModel

# Base classes
from .base import BaseModel

# Code execution
from .code_execution import CodeExecutionConfig, CodeExecutionWrapper
from .context_retry import ContextLimitRetryConfig
from .gemini import GeminiModel
from .megatron import MegatronModel

# Online GenSelect
from .online_genselect import OnlineGenSelectConfig, OnlineGenSelectWrapper
from .openai import OpenAIModel

# Tool Calling
from .tool_call import ToolCallingWrapper

# Utilities
from .vllm import VLLMModel

# Model implementations


# Model registry
models = {
    "trtllm": VLLMModel,
    "megatron": MegatronModel,
    "openai": OpenAIModel,
    "azureopenai": AzureOpenAIModel,
    "gemini": GeminiModel,
    "vllm": VLLMModel,
    "sglang": VLLMModel,
}


def get_model(server_type, tokenizer=None, **kwargs):
    """A helper function to make it easier to set server through cmd."""
    model_class = models[server_type.lower()]
    return model_class(tokenizer=tokenizer, **kwargs)


def get_code_execution_model(server_type, tokenizer=None, code_execution=None, sandbox=None, **kwargs):
    """A helper function to make it easier to set server through cmd."""
    model = get_model(server_type=server_type, tokenizer=tokenizer, **kwargs)
    if code_execution is None:
        code_execution = {}
    code_execution_config = CodeExecutionConfig(**code_execution)
    return CodeExecutionWrapper(model=model, sandbox=sandbox, config=code_execution_config)


def get_online_genselect_model(model, tokenizer=None, online_genselect_config=None, **kwargs):
    """A helper function to create OnlineGenSelect model."""
    if isinstance(model, str):
        model = get_model(model=model, tokenizer=tokenizer, **kwargs)
    return OnlineGenSelectWrapper(model=model, cfg=online_genselect_config or OnlineGenSelectConfig())


def get_tool_calling_model(model, tool_config, tokenizer=None, additional_config=None, **kwargs):
    if isinstance(model, str):
        model = get_model(model=model, tokenizer=tokenizer, **kwargs)
    return ToolCallingWrapper(model, tool_config_yaml=tool_config, additional_config=additional_config)


def server_params():
    """Returns server documentation (to include in cmd help)."""
    # TODO: This needs a fix now
    prefix = f"\n        server_type: str = MISSING - Choices: {list(models.keys())}"
    return python_doc_to_cmd_help(BaseModel, docs_prefix=prefix, arg_prefix="server.")

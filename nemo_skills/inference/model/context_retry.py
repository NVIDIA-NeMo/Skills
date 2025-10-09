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
import copy
import functools
import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, Union

from nemo_skills.utils import get_logger_name

from .utils import ServerTokenizer, WrapperAutoTokenizer, is_context_window_exceeded_error

LOG = logging.getLogger(get_logger_name(__file__))


def parse_context_window_exceeded_error(error) -> Union[Dict[str, int], None]:
    """
    Extract token information from LiteLLM context window error messages.

    Returns:
        Dict with keys: max_context_length, message_tokens or message_tokens_overflow if the prompt tokens overflow the context limit
        None if parsing fails
    """

    error_str = str(error)

    # Pattern 1
    # - maximum context length is 40960 tokens and your request has 142 input tokens (1000000 > 40960 - 142)"
    # - This model's maximum context length is 40960 tokens. However, your request has 3000009 input tokens.
    pattern1 = re.compile(
        r"maximum context length is (\d+) tokens.*?"
        r"request has (\d+) input tokens",
        re.IGNORECASE | re.DOTALL,
    )

    # Check pattern 1
    match = pattern1.search(error_str)
    if match:
        max_context = int(match.group(1))
        message_tokens = int(match.group(2))
        return {
            "max_context_length": max_context,
            "message_tokens": message_tokens,
        }

    # Pattern 2: "The input (187537 tokens) is longer than the model's context length (131072 tokens)."
    pattern2 = re.compile(
        r"The input \((\d+) tokens\) is longer than the model's context length \((\d+) tokens\)", re.IGNORECASE
    )

    # Check pattern 2
    match = pattern2.search(error_str)
    if match:
        return {
            # Context length is the second group
            "max_context_length": int(match.group(2)),
            "message_tokens": int(match.group(1)),
        }

    # Pattern 3: Handle format like "45008 in the messages, 2048 in the completion"
    pattern3 = re.compile(
        r"maximum context length is (\d+) tokens.*?"
        r"you requested (\d+) tokens.*?"
        r"(\d+) in the messages, (\d+) in the completion",
        re.IGNORECASE | re.DOTALL,
    )

    # Pattern 4: "Requested token count exceeds the model's maximum context length of 131072 tokens. You requested a total of 1000159 tokens: 159 tokens from the input messages and 1000000 tokens for the completion."
    pattern4 = re.compile(
        r"maximum context length of (\d+) tokens.*?"
        r"total of (\d+) tokens.*?"
        r"(\d+) tokens from the input messages.*?"
        r"(\d+) tokens for the completion",
        re.IGNORECASE | re.DOTALL,
    )

    # Try patterns 3 and 4
    for pattern in [pattern3, pattern4]:
        match = pattern.search(error_str)
        if match:
            return {
                "max_context_length": int(match.group(1)),
                "message_tokens": int(match.group(3)),
            }

    # Pattern 5: "max_tokens must be at least 1, got -37673"
    pattern5 = re.compile(r"max_tokens must be at least 1, got (-\d+)", re.IGNORECASE)

    # Try pattern 5
    match = pattern5.search(error_str)
    if match:
        return {
            "message_tokens_overflow": abs(int(match.group(1))),
        }

    return None


@dataclass
class ContextLimitRetryConfig:
    """Configuration for context limit retry behavior, i.e., when the context limit is exceeded."""

    enable_soft_fail: bool = False  # If True, will enable soft fail or try to reduce the context by reducing the number of tokens to generate/prompt and perform the task
    strategy: str | None = (
        None  # Strategy to use when reducing the context - reduce_generation, reduce_prompt_from_start, reduce_prompt_from_end
    )
    num_special_tokens_budget: int = 50  # To account for the discrepancy when tokenizing a message content standalone and when tokenizing it as part of a message list. Keep it high to be safe.

    def __post_init__(self):
        """Validate configuration."""
        valid_strategies = ["reduce_generation", "reduce_prompt_from_start", "reduce_prompt_from_end"]
        if self.strategy is not None and self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")

        if self.enable_soft_fail:
            LOG.info(f"Soft fail enabled with strategy: {self.strategy}")

    @property
    def reduce_generate_tokens(self):
        """Reduce the number of tokens to generate."""
        if self.strategy is not None and self.strategy == "reduce_generation":
            LOG.info("Message is too long. Reducing the number of tokens to generate.")
            return True
        else:
            return False

    @property
    def reduce_prompt_from_start(self):
        """Remove tokens from the start of the prompt."""
        if self.strategy is not None and self.strategy == "reduce_prompt_from_start":
            LOG.info("Message is too long. Removing tokens from the start of the prompt.")
            return True
        else:
            return False

    @property
    def reduce_prompt_from_end(self):
        """Remove tokens from the end of the prompt."""
        if self.strategy is not None and self.strategy == "reduce_prompt_from_end":
            LOG.info("Message is too long. Removing tokens from the end of the prompt.")
            return True
        else:
            return False


def with_context_retry(func: Callable) -> Callable:
    """
    Decorator to add context limit retry logic to generate functions.
    Uses the model's context_limit_retry_config attribute.
    """

    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        config = getattr(self, "context_limit_retry_config")
        return await handle_context_retries_async(func, self, args, kwargs, config)

    @functools.wraps(func)
    def sync_wrapper(self, *args, **kwargs):
        config = getattr(self, "context_limit_retry_config")
        return handle_context_retries_sync(func, self, args, kwargs, config)

    # Return the appropriate wrapper based on whether the function is async
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


async def handle_context_retries_async(
    func: Callable, self, args: tuple, kwargs: dict, config: ContextLimitRetryConfig
) -> dict:
    """Async version of context retry logic."""
    try:
        result = await func(self, *args, **kwargs)
        return result
    except Exception as error:
        if not config.enable_soft_fail:
            raise error
        else:
            LOG.info("Soft fail is enabled.")
            if is_context_window_exceeded_error(error):
                LOG.info(f"Caught context window exceeded error: {error}")
                if config.strategy is None:
                    return return_empty_generation_with_error(
                        f"No strategy configured. {error}", error_reason="context_window_exceeded"
                    )
                modified_kwargs = _prepare_context_error_retry(kwargs, config, self.tokenizer, error)
                if modified_kwargs is None:
                    return return_empty_generation_with_error(f"Could not apply strategy. {error}")

                try:
                    result = await func(self, *args, **modified_kwargs)
                    return result
                except Exception as error:
                    LOG.warning(f"Caught an error. Returning empty generation. {error}")
                    # This error most likely is not related to the context window exceeded error.
                    return return_empty_generation_with_error(f"{error}")
            else:
                LOG.warning(f"Caught an error. Returning empty generation. {error}")
                return return_empty_generation_with_error(f"{error}")


def handle_context_retries_sync(
    func: Callable, self, args: tuple, kwargs: dict, config: ContextLimitRetryConfig
) -> dict:
    """Sync version of context retry logic."""
    try:
        result = func(self, *args, **kwargs)
        return result
    except Exception as error:
        if not config.enable_soft_fail:
            raise error
        else:
            LOG.info("Soft fail is enabled.")
            if is_context_window_exceeded_error(error):
                LOG.info(f"Caught context window exceeded error: {error}")
                if config.strategy is None:
                    return return_empty_generation_with_error(
                        f"No strategy configured. {error}", error_reason="context_window_exceeded"
                    )
                modified_kwargs = _prepare_context_error_retry(kwargs, config, self.tokenizer, error)
                if modified_kwargs is None:
                    return return_empty_generation_with_error(f"Could not apply strategy. {error}")

                try:
                    result = func(self, *args, **modified_kwargs)
                    return result
                except Exception as error:
                    LOG.warning(f"Caught an error. Returning empty generation. {error}")
                    # This error most likely is not related to the context window exceeded error.
                    return return_empty_generation_with_error(f"{error}")
            else:
                LOG.warning(f"Caught an error. Returning empty generation. {error}")
                return return_empty_generation_with_error(f"{error}")


def _prepare_context_error_retry(
    kwargs: dict,
    config: ContextLimitRetryConfig,
    tokenizer: Union[ServerTokenizer, WrapperAutoTokenizer],
    error: Exception,
) -> dict | None:
    """Prepare kwargs for context error retry based on configured strategy.

    Returns:
        Modified kwargs dict if successful, None if strategy couldn't be applied
    """
    parsed_error = parse_context_window_exceeded_error(error)
    if parsed_error is None:
        detailed_error = f"Not able to parse the context window exceeded error- {parsed_error}. Returning empty generation.\n\n{error}"
        raise ValueError(detailed_error)
    else:
        LOG.info(f"Parsed error: {parsed_error}")

    # Apply the configured strategy
    if config.reduce_generate_tokens:
        if "message_tokens_overflow" in parsed_error:
            detailed_error = f"Prompt tokens overflow. Reducing generation tokens will not help. Returning empty generation.\n\n{error}"
            LOG.warning(detailed_error)
            return None

        return _try_reduce_generation_tokens(kwargs, parsed_error, config, error)

    elif config.reduce_prompt_from_start or config.reduce_prompt_from_end:
        if tokenizer is None:
            # Without tokenizer, we can't trim the prompt.
            detailed_error = f"Tokenizer is not set. Cannot reduce prompt tokens. Please set the tokenizer in your eval/generate request.\n\n{error}"
            raise ValueError(detailed_error)

        return _try_reduce_prompt_tokens(kwargs, parsed_error, config, tokenizer, error)
    else:
        detailed_error = f"No valid strategy configured. Returning empty generation.\n\n{error}"
        LOG.warning(detailed_error)
        return None


def _try_reduce_generation_tokens(
    kwargs: dict, parsed_error: dict, config: ContextLimitRetryConfig, original_error: Exception
) -> dict:
    """Try to reduce the number of tokens to generate."""
    original_budget = kwargs.get("tokens_to_generate", None)
    max_context_length = parsed_error["max_context_length"]
    message_tokens = parsed_error["message_tokens"]

    # Assume the num_special_tokens_budget to be part of the calculation for safety
    safe_remaining_budget = max_context_length - config.num_special_tokens_budget
    if message_tokens >= safe_remaining_budget:
        detailed_error = f"Messages tokens are already at the max context length. Cannot reduce generate tokens.\n\n{original_error}"
        LOG.warning(detailed_error)
        return None

    reduced_generation_budget = safe_remaining_budget - message_tokens
    # This min operation is probably not needed but just in case
    if original_budget is not None:
        reduced_tokens = min(original_budget, reduced_generation_budget)
    else:
        reduced_tokens = reduced_generation_budget

    LOG.warning(f"Reducing tokens_to_generate to {reduced_tokens} to stay within the context window.")

    modified_kwargs = kwargs.copy()
    modified_kwargs["tokens_to_generate"] = reduced_tokens
    return modified_kwargs


def _try_reduce_prompt_tokens(
    kwargs: dict,
    parsed_error: dict,
    config: ContextLimitRetryConfig,
    tokenizer: Union[ServerTokenizer, WrapperAutoTokenizer],
    original_error: Exception,
) -> dict:
    """Try to reduce the number of tokens in the prompt."""
    if "message_tokens_overflow" in parsed_error:
        # We can just use this information to reduce the prompt tokens
        orig_prompt_num_tokens = len(tokenizer.encode(kwargs["prompt"], tools=kwargs.get("tools", None)))
        num_prompt_tokens_to_keep = orig_prompt_num_tokens - (
            parsed_error["message_tokens_overflow"] + kwargs["tokens_to_generate"] + config.num_special_tokens_budget
        )
        LOG.warning(f"Prompt tokens overflow. Reducing prompt tokens to {num_prompt_tokens_to_keep}.")

    else:
        max_context_length = parsed_error["max_context_length"]
        completion_tokens = kwargs["tokens_to_generate"]

        if completion_tokens is None:
            detailed_error = f"tokens_to_generate is not set. Cannot reduce prompt tokens.\n\n{original_error}"
            raise ValueError(detailed_error)

        # Assume the num_special_tokens_budget to be part of the calculation for safety. SGLang has thrown error for exact equality.
        safe_remaining_budget = max_context_length - config.num_special_tokens_budget
        if completion_tokens >= safe_remaining_budget:
            detailed_error = f"Completion tokens are already at the max context length. Cannot reduce prompt tokens.\n\n{original_error}"
            raise ValueError(detailed_error)

        num_prompt_tokens_to_keep = safe_remaining_budget - completion_tokens

    prompt = kwargs["prompt"]

    LOG.info(f"Num tokens to keep: {num_prompt_tokens_to_keep}")

    if isinstance(prompt, str):
        return _trim_string_prompt(kwargs, num_prompt_tokens_to_keep, config, tokenizer)
    elif isinstance(prompt, list):
        return _trim_list_prompt(kwargs, num_prompt_tokens_to_keep, config, tokenizer, original_error)
    else:
        LOG.warning(f"Unsupported prompt type: {type(prompt)}")
        return None


def _trim_string_prompt(
    kwargs: dict,
    num_tokens_to_keep: int,
    config: ContextLimitRetryConfig,
    tokenizer: Union[ServerTokenizer, WrapperAutoTokenizer],
) -> dict:
    """Trim a string prompt to fit within token limits."""
    encoded_prompt = tokenizer.encode(kwargs["prompt"])

    if config.reduce_prompt_from_start:
        trimmed_encoded = encoded_prompt[-num_tokens_to_keep:]
    else:  # reduce_prompt_from_end
        trimmed_encoded = encoded_prompt[:num_tokens_to_keep]

    trimmed_prompt = tokenizer.decode(trimmed_encoded)
    modified_kwargs = kwargs.copy()
    modified_kwargs["prompt"] = trimmed_prompt
    return modified_kwargs


def _trim_list_prompt(
    kwargs: dict,
    num_tokens_to_keep: int,
    config: ContextLimitRetryConfig,
    tokenizer: Union[ServerTokenizer, WrapperAutoTokenizer],
    original_error: Exception,
) -> dict:
    """Trim a list-based prompt to fit within token limits."""
    prompt_copy = copy.deepcopy(kwargs["prompt"])
    tools = kwargs.get("tools", None)

    if config.reduce_prompt_from_start:
        trimmed_messages = _trim_messages_from_start(prompt_copy, tools, num_tokens_to_keep, tokenizer)
    else:  # reduce_prompt_from_end
        trimmed_messages = _trim_messages_from_end(prompt_copy, tools, num_tokens_to_keep, tokenizer)

    if not trimmed_messages:
        detailed_error = f"Not able to trim the prompt. Returning empty generation.\n\n{original_error}"
        LOG.warning(detailed_error)
        return None

    modified_kwargs = kwargs.copy()
    modified_kwargs["prompt"] = trimmed_messages
    return modified_kwargs


def _trim_messages_from_end(
    messages: list,
    tools: Union[list[dict], None],
    remaining_token_budget: int,
    tokenizer: Union[ServerTokenizer, WrapperAutoTokenizer],
) -> list:
    """Trim messages from the end of the list."""
    trimmed_messages = []
    cumulative_tokens = 0  # Number of tokens in the previous messages

    for idx, message in enumerate(messages):
        # Check if adding this full message would exceed the limit
        test_messages = messages[: idx + 1]
        encoded = tokenizer.encode(test_messages, tools=tools)
        if encoded is None:
            continue

        prefix_token_count = len(encoded)
        LOG.info(f"Prefix token count: {prefix_token_count}")

        if prefix_token_count > remaining_token_budget:
            # Try to partially include this message: Remove tokens from the previous messages -> cumulative_tokens
            num_remaining_tokens = remaining_token_budget - cumulative_tokens
            LOG.info(f"Num remaining tokens: {num_remaining_tokens}")
            trimmed_content = get_trimmed_content(
                content=message["content"],
                num_remaining_tokens=num_remaining_tokens,
                tokenizer=tokenizer,
                trim_suffix=True,  # Since we're trimming from the end, we want to trim the suffix of the current message
            )
            if trimmed_content:  # Successfully trimmed the content of the current message
                message_copy = message.copy()
                message_copy["content"] = trimmed_content
                trimmed_messages.append(message_copy)
            break
        else:
            trimmed_messages.append(message)
            cumulative_tokens = prefix_token_count

    return trimmed_messages


def _trim_messages_from_start(
    messages: list,
    tools: Union[list[dict], None],
    remaining_token_budget: int,
    tokenizer: Union[ServerTokenizer, WrapperAutoTokenizer],
) -> list:
    """Returns the suffix of the current message list that fits within the token budget."""
    trimmed_message_list = []
    total_messages = len(messages)
    cumulative_tokens = 0  # For tracking the length of the previous suffix

    for idx in range(total_messages):
        # Test messages from this index to the end
        suffix_start_idx = -(idx + 1)
        test_messages = messages[suffix_start_idx:]
        encoded = tokenizer.encode(test_messages, tools=tools)

        if encoded is None:
            continue

        suffix_token_count = len(encoded)
        LOG.info(f"Length of encoded suffix: {suffix_token_count}")

        if suffix_token_count > remaining_token_budget:
            # Try to partially include the first message of this slice
            LOG.info(
                f"idx: {idx}, suffix_token_count: {suffix_token_count}, remaining_token_budget: {remaining_token_budget}, cumulative_tokens: {cumulative_tokens}"
            )
            remaining_tokens = remaining_token_budget - cumulative_tokens
            trimmed_content = get_trimmed_content(
                content=messages[suffix_start_idx]["content"],
                num_remaining_tokens=remaining_tokens,
                tokenizer=tokenizer,
                trim_suffix=False,  # Since we're trimming from the start, we want to trim the prefix of the current message
            )
            if trimmed_content:  # Successfully trimmed the content of the current message
                message_copy = messages[suffix_start_idx].copy()
                message_copy["content"] = trimmed_content
                return [message_copy] + trimmed_message_list
            else:
                return trimmed_message_list

        else:
            # Track the length of the current suffix
            cumulative_tokens = suffix_token_count
            trimmed_message_list = test_messages

    return []


def get_trimmed_content(
    content: str,
    num_remaining_tokens: int,
    tokenizer: Union[ServerTokenizer, WrapperAutoTokenizer],
    trim_suffix: bool = True,
) -> str:
    """
    Get the trimmed content of a message.
    """
    # Make sure we have a positive number of tokens to keep.
    if num_remaining_tokens > 0:
        encoded_content = tokenizer.encode(content)
        if trim_suffix:
            encoded_content = encoded_content[:num_remaining_tokens]
        else:
            encoded_content = encoded_content[-num_remaining_tokens:]
        trimmed_content = tokenizer.decode(encoded_content)
        return trimmed_content
    else:
        return None


def return_empty_generation_with_error(detailed_error: str, error_reason: str = "See detailed_error"):
    return {
        "generation": "",
        "num_generated_tokens": 0,
        "error": error_reason,
        "detailed_error": detailed_error,
    }

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import json
import logging
import random
import re
from dataclasses import asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from transformers import AutoTokenizer

from nemo_skills.code_execution.utils import format_code_output
from nemo_skills.prompt.few_shot_examples import examples_map
from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))


class BM25Retriever:
    def __init__(self, data_path: str, field: str):
        from rank_bm25 import BM25Okapi

        with open(data_path, "rt", encoding="utf-8") as fin:
            self.entries = [json.loads(x) for x in fin]

        corpus = [entry[field] for entry in self.entries]
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 1):
        tokenized_query = query.split(" ")
        return self.bm25.get_top_n(tokenized_query, self.entries, n=top_k)


@nested_dataclass(kw_only=True)
class FewShotExamplesConfig:
    prefix: str = ""
    template: str = ""
    suffix: str = ""

    examples_type: Optional[str] = None

    retrieval_field: Optional[str] = None  # e.g. question, reference_solution, etc.
    retrieval_file: Optional[str] = None  # needs to be provided if retrieval_field is not None
    retrieved_entries: int = 10  # need to set higher than few_shots to filter out exact matches
    retrieved_few_shots: int = 5
    randomize_retrieved_entries: bool = False
    max_retrieved_chars: int = 100000000  # no limit by default
    max_retrieved_chars_field: str = "reference_solution"
    retriever: Optional[Any] = None

    def __post_init__(self):
        """Error checks + building example_dicts and retriever if needed."""
        if self.examples_type is not None and self.retriever is not None:
            raise ValueError("examples_type and retriever cannot be used together")

        if self.retriever is not None:
            return

        if self.retrieval_field is not None:  # building retriever
            if self.retrieval_file is None:
                raise ValueError("retrieval_file must be provided if retrieval_field is not None")
            self.retriever = BM25Retriever(self.retrieval_file, field=self.retrieval_field)
        else:
            if self.retrieval_file is not None:
                raise ValueError("retrieval_field must be provided if retrieval_file is not None")


@nested_dataclass(kw_only=True)
class CodeTags:
    # used to execute code within these tags
    code_begin: str = "```python\n"
    code_end: str = "```\n"

    # used to extract the code output
    code_output_begin: str = "```output\n"
    code_output_end: str = "```\n"

    # used to post-process code output
    code_output_format: str = "qwen"


@nested_dataclass(kw_only=True)
class PromptConfig:
    user: str
    system: str | None = None
    code_tags: CodeTags = None
    few_shot_examples: FewShotExamplesConfig = field(default_factory=FewShotExamplesConfig)
    # VLM support: if set, the field name from input_dict containing the image path
    # When set, user content will be a multimodal list with image_url + text
    image_field: str | None = None
    # Whether to put image before or after the text in multimodal content
    image_position: str = "before"  # "before" or "after"


class Prompt:
    def __init__(self, config, tokenizer):
        # rebuilding prompt config to make sure post init is called again in
        # case some parameters were manually changed after the config was created
        self.config = PromptConfig(_init_nested=True, **asdict(config))
        self.tokenizer = tokenizer
        if self.tokenizer:
            # assuming it's the object already if not str
            if isinstance(self.tokenizer, str):
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer, trust_remote_code=True)

    def build_filled_example(self, example_dict: Dict[str, Any]) -> str:
        """Builds a filled example string based on the example dictionary."""

        # replacing code/code-output separators in the examples if present
        example_dict = example_dict.copy()
        if "solution" in example_dict and self.config.code_tags:

            def replace_code_output(match):
                code_output = match.group(2)
                formatted_output = format_code_output(
                    execution_dict={"process_status": "completed", "stdout": code_output, "stderr": ""},
                    code_output_begin=self.config.code_tags.code_output_begin,
                    code_output_end=self.config.code_tags.code_output_end,
                    code_output_format=self.config.code_tags.code_output_format,
                )
                return formatted_output

            pattern = r"({code_output_begin}\n)(.*?)({code_output_end})"
            example_dict["solution"] = re.sub(pattern, replace_code_output, example_dict["solution"], flags=re.DOTALL)

            example_dict["solution"] = example_dict["solution"].replace(
                "{code_begin}", self.config.code_tags.code_begin
            )
            example_dict["solution"] = example_dict["solution"].replace("{code_end}", self.config.code_tags.code_end)
            example_dict["solution"] = example_dict["solution"].replace("{code_output_begin}", "")
            example_dict["solution"] = example_dict["solution"].replace("{code_output_end}", "")

        return self.config.few_shot_examples.template.format(**example_dict)

    def build_examples_dict(self, input_dict):
        if self.config.few_shot_examples.examples_type:
            return examples_map[self.config.few_shot_examples.examples_type.format(**input_dict)]

        if self.config.few_shot_examples.retriever is None:
            return []

        example_dicts = self.config.few_shot_examples.retriever.retrieve(
            query=input_dict[self.config.few_shot_examples.retrieval_field],
            top_k=self.config.few_shot_examples.retrieved_entries,
        )
        reference = input_dict[self.config.few_shot_examples.retrieval_field]
        # filtering exact match if it's there
        while example_dicts and example_dicts[0][self.config.few_shot_examples.retrieval_field] == reference:
            example_dicts = example_dicts[1:]

        # removing too long solutions
        example_dicts = [
            example_dict
            for example_dict in example_dicts
            if len(example_dict[self.config.few_shot_examples.max_retrieved_chars_field])
            < self.config.few_shot_examples.max_retrieved_chars
        ]

        if len(example_dicts) < self.config.few_shot_examples.retrieved_few_shots:
            LOG.warning(
                'Too little examples (%d) found for the query "%s"',
                len(example_dicts),
                input_dict[self.config.few_shot_examples.retrieval_field],
            )

        # let's reverse the order to show the most relevant last
        examples = example_dicts[: self.config.few_shot_examples.retrieved_few_shots][::-1]
        if self.config.few_shot_examples.randomize_retrieved_entries:
            random.shuffle(examples)

        return examples

    def build_user_message(self, input_dict: Dict[str, str]) -> str:
        """Builds all examples string concatenated by delimiter."""
        example_dicts = self.build_examples_dict(input_dict)

        filled_examples = "".join([self.build_filled_example(example) for example in example_dicts])
        if not filled_examples:
            examples = ""
        else:
            examples = f"{self.config.few_shot_examples.prefix}{filled_examples}{self.config.few_shot_examples.suffix}"
        user_message = self.config.user.format(examples=examples, **input_dict)
        return user_message

    def get_code_execution_args(self):
        """Returns the code execution arguments."""
        if self.config.code_tags is None:
            raise ValueError(
                "Please provide 'code_tags' in your prompt configuration before calling get_code_execution_args()."
            )
        return {
            "code_begin": self.config.code_tags.code_begin,
            "code_end": self.config.code_tags.code_end,
            "code_output_begin": self.config.code_tags.code_output_begin,
            "code_output_end": self.config.code_tags.code_output_end,
            "code_output_format": self.config.code_tags.code_output_format,
        }

    def format_assistant_response(
        self, content: str, thinking: str | None = None, chat_template_kwargs: dict | None = None
    ) -> str:
        """Adds special tokens to the end of assistant response and formats thinking if provided"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not set.")

        messages = [{"role": "user", "content": ""}]

        chat_template_kwargs = chat_template_kwargs or {}
        user_string = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )

        messages.append({"role": "assistant", "content": content})
        if thinking is not None:
            messages[-1]["thinking"] = thinking
        assistant_string = self.tokenizer.apply_chat_template(messages, tokenize=False, **chat_template_kwargs)

        assert assistant_string.startswith(user_string), f"Something is wrong\n{user_string}\n||\n{assistant_string}"

        formatted_response = assistant_string[len(user_string) :]
        if thinking is not None:
            # Check that thinking is part of the assistant string
            # If not, the thinking string should be added to the "content" field during preprocessing
            # Tokenizers for models like Qwen3-4B don't add thinking to the assistant string by themselves
            assert thinking in assistant_string, (
                f"The thinking content is not part of the assistant string. We suggest you add the thinking string to the 'content' field during preprocessing.\n\nThinking string:{thinking}\n\nAssistant string:{formatted_response}"
            )

        return formatted_response

    def fill(
        self,
        input_dict: Dict[str, str],
        start_assistant_response_key: str | None = None,
        chat_template_kwargs: dict | None = None,
        format_as_string=False,
    ) -> str | List[dict]:
        """
        Fills the prompt with the input_dict.
        Operates in two modes:
        - If `self.tokenizer` is set, it will use it to format the prompt, returning a string.
        - If `self.tokenizer` is not set, it will assume chat format and return a list of dictionaries.

        Args:
            input_dict: The input dictionary to fill the prompt with.
            start_assistant_response_key: Whether to append the value of this key to the beginning of assistant response.
            chat_template_kwargs: Any extra parameters to pass to the tokenizer's apply_chat_template method.
            format_as_string: When False (default) we will just return a list of messages, when format_as_string is True, we will return a string using the tokenizer's apply_chat_template method (and fail if the tokenizer is not set).

        Returns:
            The filled prompt - either a string or a list of dictionaries.
        """

        if self.config.system is not None:
            messages = [
                {"role": "system", "content": self.config.system.format(**input_dict)},
            ]
        else:
            messages = []

        # Build user message content
        user_text = self.build_user_message(input_dict)

        # For VLM: if image_field is set, build multimodal content with image + text
        if self.config.image_field and self.config.image_field in input_dict:
            image_path = input_dict[self.config.image_field]
            text_part = {"type": "text", "text": user_text}
            image_part = {"type": "image_url", "image_url": {"url": image_path}}

            if self.config.image_position == "before":
                user_content = [image_part, text_part]
            elif self.config.image_position == "after":
                user_content = [text_part, image_part]
            else:
                raise ValueError(f"Invalid image_position '{self.config.image_position}'. Must be 'before' or 'after'")
        else:
            user_content = user_text

        messages.append({"role": "user", "content": user_content})

        if not format_as_string:
            if start_assistant_response_key:
                raise ValueError("start_assistant_response_key is not supported for chat template format.")

            if chat_template_kwargs:
                raise ValueError("chat_template_kwargs can only be used when format_as_string=True")

            return messages
        else:
            if self.tokenizer is None:
                raise ValueError("tokenizer is not set, can't format messages as a string")

            chat_template_kwargs = chat_template_kwargs or {}
            try:
                messages_string = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **chat_template_kwargs,
                )
            except ValueError as e:
                if "Cannot use chat template functions because tokenizer.chat_template is not set" in str(e):
                    # assuming that's a base model and we just need to add bos
                    if len(messages) != 1 or messages[0]["role"] != "user":
                        raise ValueError(
                            "The model doesn't support chat template, can't format messages which contain non-user values"
                        )
                    user_content = messages[0]["content"]
                    # Handle multimodal content - extract text for base models (no chat template)
                    if isinstance(user_content, list):
                        text_parts = [item["text"] for item in user_content if item.get("type") == "text"]
                        user_content = " ".join(text_parts)
                    if hasattr(self.tokenizer, "bos_token"):
                        messages_string = self.tokenizer.bos_token + user_content
                    else:
                        messages_string = user_content
                else:
                    raise e
            if start_assistant_response_key:
                messages_string += input_dict[start_assistant_response_key]
            return messages_string

    def __str__(self):
        return str(self.config)


def get_token_count(
    tokenizer,
    messages: Union[str, list[Union[dict, Any]]],
    tools: Union[list[dict], None] = None,
) -> int | None:
    """
    Count the number of tokens in a string or chat message list.

    Args:
        messages (str | list[dict]): Input text or chat messages.

    Returns:
        int | None: Token count, or None if no tokenizer is set.
    """

    def message_to_dict(orig_message: Any) -> Dict[str, Any]:
        message = {"role": orig_message.role}
        # Handle content
        if orig_message.content is not None:
            message["content"] = orig_message.content
        else:
            message["content"] = ""

        # Handle tool_calls
        if hasattr(orig_message, "tool_calls") and orig_message.tool_calls:
            message["tool_calls"] = []
            for tool_call in orig_message.tool_calls:
                # Check if tool_call is already a dict
                if isinstance(tool_call, dict):
                    # Already in dict format, use as-is
                    message["tool_calls"].append(tool_call)
                else:
                    # Convert object to dict
                    tool_call_dict = {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
                    }
                    message["tool_calls"].append(tool_call_dict)
        return message

    if tokenizer is None:
        return None

    if messages is None:
        return None

    if isinstance(messages, str):
        return len(tokenizer.encode(messages, add_special_tokens=False))
    elif isinstance(messages, list):
        # Convert messages to dicts if they are not already in dict format
        messages = [
            message if isinstance(message, dict) else message_to_dict(copy.deepcopy(message)) for message in messages
        ]
        try:
            result = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, tools=tools)
            # Handle newer HF tokenizer versions that return a BatchEncoding instead of a list
            if not isinstance(result, list):
                result = result["input_ids"]
            return len(result)

        except Exception as e:
            raise ValueError(f"Invalid chat message format: {e}")

    else:
        raise ValueError("messages must be a string or a list of dictionaries")


def get_config_path(config: str, config_dir: str | None = None, config_extension: str = "yaml") -> Path:
    if config_dir is None:
        config_dir = str(Path(__file__).parent.absolute() / "config")

    if config.endswith(f".{config_extension}"):
        config_path = Path(config).absolute()
        # If not found, try relative to repo root (works for external packages
        # whose code lives next to nemo_skills, e.g. in /nemo_run/code/)
        if not config_path.is_file():
            config_path = Path(__file__).parents[2].absolute() / config
    else:
        config_path = Path(config_dir) / f"{config}.{config_extension}"

    return config_path


def load_config(config: str, config_dir: str | None = None) -> dict:
    """
    Reads the prompt config/template from the yaml file.

    Args:
        config (str): The location of the prompt config file.
            If it ends with .yaml, it is treated as a path (absolute or relative to repo root).
            Otherwise, it is looked up relative to config_dir.
        config_dir (str): The dir to look for the config file.

    Returns:
        The loaded dictionary.
    """
    config_path = get_config_path(config, config_dir)

    with open(config_path, "rt", encoding="utf-8") as fin:
        return yaml.safe_load(fin)


def get_prompt(
    prompt_config: str | dict,
    tokenizer: Any | None = None,
    code_tags: str | dict | None = None,
    examples_type: str | None = None,
    system_message: str | None = None,
    config_dir: str | None = None,
    code_tags_dir: str | None = None,
) -> Prompt:
    if code_tags_dir is None:
        code_tags_dir = Path(__file__).parent.absolute() / "code_tags"

    if isinstance(prompt_config, str):
        config = load_config(prompt_config, config_dir)
    else:
        config = prompt_config

    if system_message is not None:
        config["system"] = system_message

    code_tags_obj = None
    if code_tags is not None:
        if isinstance(code_tags, str):
            code_tags_dict = load_config(code_tags, code_tags_dir)
        else:
            code_tags_dict = code_tags
        code_tags_obj = CodeTags(**code_tags_dict)

    prompt = Prompt(PromptConfig(**config, code_tags=code_tags_obj), tokenizer=tokenizer)

    if examples_type is not None:
        prompt.config.few_shot_examples.examples_type = examples_type

    return prompt

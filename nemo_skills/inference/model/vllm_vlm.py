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

import base64
import logging
import mimetypes
import os
from pathlib import Path

from nemo_skills.utils import get_logger_name

from .vllm import VLLMModel

LOG = logging.getLogger(get_logger_name(__file__))


def encode_image_to_base64(image_path: str) -> str:
    """Encode a local image file to base64 data URL.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64-encoded data URL string.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Determine MIME type
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        # Default to jpeg for unknown types
        mime_type = "image/jpeg"

    with open(path, "rb") as f:
        image_data = f.read()

    base64_data = base64.b64encode(image_data).decode("utf-8")
    return f"data:{mime_type};base64,{base64_data}"


def process_image_content(content: list | str, data_dir: str = "") -> list | str:
    """Process message content to handle image paths and URLs.

    Converts local file paths to base64 data URLs if needed.
    HTTP/HTTPS URLs and existing data URLs are passed through unchanged.

    Path resolution strategy:
    1. Absolute paths (e.g., /path/to/image.png) - used directly
    2. Relative paths with data_dir set - joined with data_dir
    3. Relative paths without data_dir - tried as-is (may work if CWD is correct)
    4. HTTP/HTTPS URLs - passed through unchanged
    5. Data URLs (base64) - passed through unchanged

    Args:
        content: Message content - either a string or list of content parts.
        data_dir: Base directory for resolving relative image paths.
                  Typically the parent directory of the input JSONL file.

    Returns:
        Processed content with images as proper URLs.
    """
    if isinstance(content, str):
        return content

    processed_content = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "image_url":
            image_url = item.get("image_url", {})
            url = image_url.get("url", "")

            # Check if it's a local file path that needs to be converted
            if url and not url.startswith(("data:", "http://", "https://")):
                # Resolve the path: use data_dir for relative paths, otherwise use as-is
                if not os.path.isabs(url) and data_dir:
                    resolved_path = os.path.join(data_dir, url)
                else:
                    resolved_path = url

                try:
                    base64_url = encode_image_to_base64(resolved_path)
                    processed_item = {
                        "type": "image_url",
                        "image_url": {"url": base64_url},
                    }
                    # Preserve any additional image_url properties (like detail)
                    for key in image_url:
                        if key != "url":
                            processed_item["image_url"][key] = image_url[key]
                    item = processed_item
                except FileNotFoundError:
                    LOG.error(
                        f"Image file not found: {resolved_path} "
                        f"(original path: {url}, data_dir: {data_dir or 'not set'})"
                    )
                    raise

            processed_content.append(item)
        else:
            processed_content.append(item)

    return processed_content


class VLLMVLMModel(VLLMModel):
    """VLLMModel with support for Vision-Language Model (VLM) image inputs.

    This model extends VLLMModel to handle image inputs in the OpenAI-compatible
    chat completions format. Images can be provided in message content as:
    - Base64-encoded data URLs
    - HTTP/HTTPS URLs
    - Local file paths (will be converted to base64 automatically)

    The model will process images and send them to the vLLM server which must
    be running a VLM model (e.g., LLaVA, Qwen2-VL, InternVL, Pixtral, etc.).

    Example:
        model = VLLMVLMModel(model="Qwen/Qwen2-VL-7B-Instruct")
        response = await model.generate_async(
            prompt=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is shown in this image?"},
                        {"type": "image_url", "image_url": {"url": "/path/to/image.jpg"}}
                    ]
                }
            ]
        )
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        LOG.info("Initialized VLM model with image input support")

    def _build_chat_request_params(
        self,
        messages: list[dict],
        stream: bool,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        timeout: int | None = None,
        top_logprobs: int | None = None,
        reasoning_effort: str | None = None,
        tools: list[dict] | None = None,
        extra_body: dict = None,
    ) -> dict:
        """Build chat request params with image processing.

        This method extends the parent to process any image content in messages,
        converting local file paths to base64 data URLs.
        """
        # Process messages to handle image content
        processed_messages = []
        for msg in messages:
            processed_msg = msg.copy()
            if "content" in processed_msg:
                processed_msg["content"] = process_image_content(processed_msg["content"], self.data_dir)
            processed_messages.append(processed_msg)

        # Call parent method with processed messages
        return super()._build_chat_request_params(
            messages=processed_messages,
            stream=stream,
            tokens_to_generate=tokens_to_generate,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            random_seed=random_seed,
            stop_phrases=stop_phrases,
            timeout=timeout,
            top_logprobs=top_logprobs,
            reasoning_effort=reasoning_effort,
            tools=tools,
            extra_body=extra_body,
        )

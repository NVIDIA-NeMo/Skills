#!/usr/bin/env python3
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

# ---- Fast JSON (prefer orjson for reading) ----
import argparse
import json as _json_std
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoTokenizer

try:  # pragma: no cover - best effort
    import orjson as _orjson  # type: ignore

    def _json_loads(s: str):
        return _orjson.loads(s)

except Exception:  # pragma: no cover
    _orjson = None

    def _json_loads(s: str):
        return _json_std.loads(s)


# Always use stdlib json for writing to avoid compact formatting from orjson
# (orjson omits spaces after colons/commas which produces non-standard
# tokenization when tool-call arguments are embedded via chat templates).
def _json_dumps(obj) -> str:
    return _json_std.dumps(obj, ensure_ascii=False)


# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------


def messages_to_string(
    messages: List[Dict[str, str]],
    tokenizer: Any,
    add_generation_prompt: bool,
    chat_template_kwargs: Dict[str, Any] | None = None,
) -> str:
    """
    Convert a list of messages to a formatted string using tokenizer's chat template.

    Args:
        messages: List of message dictionaries with fields like role, content, etc.
        tokenizer: Tokenizer object with apply_chat_template method.
        chat_template_kwargs: Optional kwargs to pass to apply_chat_template.

    Returns:
        Formatted string with tokenizer's chat template applied.
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required")

    # check for "harmony" style (GPT-OSS) templates and convert to "role"/"content" format if needed

    if "gpt-oss" in tokenizer.name_or_path.lower():
        is_gpt_oss = True
    else:
        is_gpt_oss = False

    if is_gpt_oss:  # replace the "reasoning_content" with a "thinking" key
        converted_messages = []
        for msg in messages:
            if msg.get("role") == "assistant" and "reasoning_content" in msg:
                reasoning = msg.pop("reasoning_content")
                msg["thinking"] = reasoning
            converted_messages.append(msg)
        messages = converted_messages

    # remove all fields with None value
    formatted_messages = []
    for msg in messages:
        formatted_msg = {k: v for k, v in msg.items() if v is not None}
        formatted_messages.append(formatted_msg)

    if chat_template_kwargs is None:
        chat_template_kwargs = {}
    elif not isinstance(chat_template_kwargs, dict):
        raise TypeError(f"chat_template_kwargs must be a mapping/dict, got {type(chat_template_kwargs).__name__}")

    return tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **chat_template_kwargs,
    )


def compute_token_length(text: str, tokenizer: AutoTokenizer) -> int:
    """Compute number of tokens for a given text."""
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def bucket_index(length: int, bucket_sizes: List[int]) -> int:
    """Return bucket index (upper bound) for given length."""
    for size in bucket_sizes:
        if length <= size:
            return size
    return -1  # overflow


def _parse_chat_template_kwargs_json(raw: str) -> Dict[str, Any]:
    try:
        parsed = _json_std.loads(raw)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f'--chat_template_kwargs must be a JSON object string (e.g. \'{{"reasoning_effort": "high"}}\'). Parse error: {exc}'
        ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"--chat_template_kwargs must be a JSON object, got {type(parsed).__name__}")
    return parsed


def extract_input_output_from_messages(
    obj: Dict[str, Any], tokenizer: AutoTokenizer, chat_template_kwargs: Dict[str, Any]
) -> tuple:
    """
    Extract input (system + user prompts) and output (reasoning, tool calls, etc.) from message object.

    Args:
        obj: Dictionary containing messages list or role/content fields
        tokenizer: Tokenizer to format messages
        chat_template_kwargs: Additional keyword arguments for chat template formatting

    Returns:
        Tuple of (input_string, output_string, input_token_length, output_token_length)
    """
    messages = list(obj.get("messages", []))

    if not messages:
        return "", "", 0, 0

    # Separate system/user messages from the rest

    input_messages = []

    for msg in messages:
        role = msg.get("role", "").lower()
        if role in ["system", "user"]:
            input_messages.append(msg)

    if input_messages:
        input_string = messages_to_string(
            input_messages, tokenizer, add_generation_prompt=True, chat_template_kwargs=chat_template_kwargs
        )
    else:
        input_string = ""

    output_string = messages_to_string(
        messages, tokenizer, add_generation_prompt=False, chat_template_kwargs=chat_template_kwargs
    )

    output_string = output_string.removeprefix(input_string).strip() if input_string else output_string.strip()

    # Calculate token lengths
    input_token_length = compute_token_length(input_string, tokenizer)
    output_token_length = compute_token_length(output_string, tokenizer)

    return input_string, output_string, input_token_length, output_token_length


# ------------------------------------------------------------
# Core logic
# ------------------------------------------------------------


def process_jsonl(
    input_path: str,
    output_dir: str,
    tokenizer: AutoTokenizer,
    chat_template_kwargs: Dict[str, Any],
    bucket_sizes: List[int] = None,
    bucket_field: str = "output_token_length",
):
    """Process messages, calculate token lengths, optionally bucket by token size."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Processing file: {input_path}")
    logging.info(f"Output directory: {output_dir}")

    # Prepare bucket files
    bucket_files = {}
    bucket_counts = {}
    if bucket_sizes:
        for b in bucket_sizes:
            bucket_path = output_dir / f"{input_path.stem}_bucket_{b}.jsonl"
            bucket_files[b] = open(bucket_path, "w", encoding="utf-8")
            bucket_counts[b] = 0
        overflow_path = output_dir / f"{input_path.stem}_bucket_overflow.jsonl"
        bucket_files["overflow"] = open(overflow_path, "w", encoding="utf-8")
        bucket_counts["overflow"] = 0
    else:
        output_path = output_dir / f"{input_path.stem}_with_tokens.jsonl"
        outfile = open(output_path, "w", encoding="utf-8")

    processed = 0
    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json_loads(line)

                # Extract input/output and calculate token lengths
                if (
                    "input" not in obj
                    or "output" not in obj
                    or "input_token_length" not in obj
                    or "output_token_length" not in obj
                ):
                    input_str, output_str, input_len, output_len = extract_input_output_from_messages(
                        obj, tokenizer, chat_template_kwargs
                    )
                    obj["input"] = input_str
                    obj["output"] = output_str
                    obj["input_token_length"] = input_len
                    obj["output_token_length"] = output_len
                    dumped = _json_dumps(obj)
                else:
                    dumped = line  # already processed

                # Determine bucket field and length
                length = obj.get(bucket_field, 0)

                if bucket_sizes:
                    b = bucket_index(length, bucket_sizes)
                    if b != -1:
                        bucket_files[b].write(dumped + "\n")
                        bucket_counts[b] += 1
                    else:
                        bucket_files["overflow"].write(dumped + "\n")
                        bucket_counts["overflow"] += 1
                else:
                    outfile.write(dumped + "\n")

                processed += 1
                if processed % 1000 == 0:
                    logging.info(f"Processed {processed} lines")

            except Exception as e:
                logging.info(f"Skipping line: {line[:100]}... due to error: {e}")
                logging.error(f"Error processing line: {e}")

    if bucket_sizes:
        for f in bucket_files.values():
            f.close()
    else:
        outfile.close()

    # ---- Summary logging ----
    logging.info(f"✅ Done! Processed {processed} examples total.")
    if bucket_sizes:
        logging.info("📊 Bucket distribution:")
        total = sum(bucket_counts.values())
        for b, count in bucket_counts.items():
            pct = (count / total * 100) if total > 0 else 0
            logging.info(f"  - Bucket {b:<10}: {count:>8} items ({pct:.1f}%)")
        logging.info(f"💾 Saved bucketed files in: {output_dir}")
    else:
        logging.info(f"💾 Saved to {output_path}")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process messages, calculate input/output token lengths, and optionally bucket by token size."
    )
    parser.add_argument("input_file", type=str, help="Path to input .jsonl file with messages")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed files")
    parser.add_argument(
        "--bucket_sizes",
        default=[16000, 32000, 64000],
        nargs="+",
        type=int,
        help="List of bucket size upper limits (e.g. 16000, 32000, 64000)",
    )
    parser.add_argument(
        "--bucket_field",
        type=str,
        default="output_token_length",
        help="Field to use for bucketing (output_token_length or input_token_length)",
    )
    parser.add_argument("--tokenizer_path", type=str, help="Model name for tokenizer")
    parser.add_argument(
        "--chat_template_kwargs",
        type=_parse_chat_template_kwargs_json,
        default={},
        help='Additional keyword arguments for chat template formatting as a JSON object string, e.g. {"reasoning_effort":"high"}',
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    chat_template_kwargs = args.chat_template_kwargs

    process_jsonl(
        args.input_file,
        args.output_dir,
        tokenizer,
        chat_template_kwargs,
        args.bucket_sizes,
        args.bucket_field,
    )

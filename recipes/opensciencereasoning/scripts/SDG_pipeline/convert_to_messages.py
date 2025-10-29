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

import argparse
import json as _json_std
import logging
import multiprocessing
import os
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Dict, List, Optional, Tuple

# --- Fast JSON (optional) ----------------------------------------------------
try:  # orjson is significantly faster; fallback to std json
    import orjson as _orjson  # type: ignore

    def _json_loads(s: str):
        return _orjson.loads(s)

    def _json_dumps(obj) -> str:
        return _orjson.dumps(obj).decode("utf-8")

except Exception:  # pragma: no cover - best effort
    _orjson = None

    def _json_loads(s: str):
        return _json_std.loads(s)

    def _json_dumps(obj) -> str:
        return _json_std.dumps(obj)


OUTPUT_PATTERN_RE = re.compile(
    r"<\|start\|>"  # start sentinel already prefixed for assistant output concat
    r"(?P<source_info>.*?)"
    r"<\|message\|>"
    r"(?P<content>.*?)"
    r"(?P<terminator><\|end\|>|<\|call\|>|<\|return\|>)",
    re.DOTALL,
)


# --- Core parsing -------------------------------------------------------------


def parse_line_to_openai_format(json_line: str) -> Tuple[List[Dict], Dict]:
    """
    Parses a single JSONL line into the OpenAI chat format, combining an
    assistant's reasoning with its subsequent tool call and fusing the final
    reasoning and content messages.

    Returns (messages, extra_fields) where extra_fields are all original JSON
    keys except 'input' and 'output'.
    """
    try:
        data = _json_loads(json_line)
        input_str = data["input"]
        output_str = data["output"]
    except Exception as e:  # broad for speed (JSONDecodeError/KeyError)
        raise ValueError(f"Invalid JSON or missing keys: {e}")

    # retain all other keys
    extra_fields = {k: v for k, v in data.items() if k not in ("input", "output")}

    messages: List[Dict] = []
    tool_call_counter = 0
    assistant_reasoning_buffer: Optional[Dict] = None

    # Parse 'input'+ 'output' for assistant and tool messages
    full_interaction_str = input_str + output_str
    matches = list(OUTPUT_PATTERN_RE.finditer(full_interaction_str))

    for i, match in enumerate(matches):
        parts = match.groupdict()
        source_info = parts["source_info"].strip()
        content = parts["content"].strip()
        terminator = parts["terminator"]

        if "system" in source_info:
            messages.append({"role": "system", "content": content})
            continue

        if "user" in source_info:
            messages.append({"role": "user", "content": content})
            continue

        # Flush buffer if it exists before processing a new message that isn't a tool call
        if assistant_reasoning_buffer and "to=python code" not in source_info:
            messages.append(assistant_reasoning_buffer)
            assistant_reasoning_buffer = None

        if "to=assistant" in source_info:  # tool response
            tool_name = "stateful_python_code_exec"
            tool_call_id = f"call_{tool_call_counter - 1}"
            messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": content})
        else:  # assistant side
            if "to=python code" in source_info:  # tool call
                if not assistant_reasoning_buffer:
                    assistant_reasoning_buffer = {"role": "assistant"}
                tool_call_id = f"call_{tool_call_counter}"
                tool_name = "stateful_python_code_exec"
                assistant_reasoning_buffer.setdefault("tool_calls", []).append(
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": _json_dumps({"code": content})},
                    }
                )
                messages.append(assistant_reasoning_buffer)
                assistant_reasoning_buffer = None
                tool_call_counter += 1
            else:  # reasoning / final answer
                is_final_answer = (terminator == "<|return|>") or (i == len(matches) - 1 and terminator != "<|call|>")
                if is_final_answer:
                    messages.append({"role": "assistant", "content": content})
                else:
                    assistant_reasoning_buffer = {"role": "assistant", "reasoning_content": content}

    if assistant_reasoning_buffer:  # flush
        messages.append(assistant_reasoning_buffer)

    # 3. Fuse the final two assistant messages if applicable
    if (
        len(messages) >= 2
        and messages[-1].get("role") == "assistant"
        and "content" in messages[-1]
        and messages[-2].get("role") == "assistant"
        and "reasoning_content" in messages[-2]
        and "tool_calls" not in messages[-2]
    ):
        messages[-1]["reasoning_content"] = messages[-2]["reasoning_content"]
        messages.pop(-2)

    # adding content = '' everywhere if it's missing
    for msg in messages:
        if msg.get("role") == "assistant" and "content" not in msg:
            msg["content"] = ""

    return messages, extra_fields


# --- Workers -----------------------------------------------------------------


def _process_batch(payloads: List[Tuple[int, int, str]]):  # batch variant to reduce IPC overhead
    out = []
    append = out.append
    for seq_id, original_line_number, line in payloads:
        try:
            openai_messages, extra = parse_line_to_openai_format(line)
            append((seq_id, _json_dumps({**extra, "messages": openai_messages}), None))
        except ValueError as e:
            append((seq_id, None, f"Skipping line {original_line_number} due to error: {e}"))
    return out


# --- Main --------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL to OpenAI format, combining reasoning, tool calls, and final answer."
    )
    parser.add_argument("input_file", help="Path to the input JSONL file. Use '-' to read from stdin.")
    parser.add_argument("output_file", help="Path to the output JSONL file. Use '-' to write to stdout.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of worker processes (default: CPU count - 1). Use 1 to disable parallelism.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1000000,
        help="Maximum number of in-flight lines being processed in parallel (default: 1000000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of lines to group into a single task (default: 100). Larger batches reduce overhead but increase latency.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50000,
        help="Log progress every N processed lines (default: 50000)",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("convert-tools2")
    start_time = time.time()
    logger.info(
        "Starting conversion: input=%s output=%s workers=%d buffer_size=%d batch_size=%d json_lib=%s",
        args.input_file,
        args.output_file,
        args.workers,
        args.buffer_size,
        args.batch_size,
        "orjson" if _orjson else "stdlib",
    )

    input_stream = open(args.input_file, "r", encoding="utf-8") if args.input_file != "-" else sys.stdin
    output_stream = open(args.output_file, "w", encoding="utf-8") if args.output_file != "-" else sys.stdout

    # Sequential path (still can benefit from orjson + precompiled regex)
    if args.workers == 1:
        processed = 0
        errors = 0
        with input_stream, output_stream:
            for i, line in enumerate(input_stream):
                if not line.strip():
                    continue
                try:
                    openai_messages, extra = parse_line_to_openai_format(line)
                    output_stream.write(_json_dumps({**extra, "messages": openai_messages}) + "\n")
                except ValueError as e:
                    errors += 1
                    logger.warning("Skipping line %d due to error: %s", i + 1, e)
                processed += 1
                if processed % args.log_interval == 0:
                    elapsed = time.time() - start_time
                    logger.info(
                        "Progress: processed=%d errors=%d elapsed=%.1fs rate=%.1f/s",
                        processed,
                        errors,
                        elapsed,
                        processed / elapsed if elapsed > 0 else 0,
                    )
        elapsed = time.time() - start_time
        logger.info(
            "Finished sequential conversion: processed=%d errors=%d elapsed=%.1fs rate=%.1f/s",
            processed,
            errors,
            elapsed,
            processed / elapsed if elapsed > 0 else 0,
        )
        return

    # Parallel (batched) path
    batch_size = args.batch_size

    with input_stream, output_stream, ProcessPoolExecutor(max_workers=args.workers) as executor:
        in_flight = {}
        pending_results: Dict[int, Tuple[Optional[str], Optional[str]]] = {}
        next_to_write = 0
        seq_id = 0
        processed = 0
        errors = 0
        current_batch: List[Tuple[int, int, str]] = []

        def submit_batch(batch: List[Tuple[int, int, str]]):
            if not batch:
                return
            fut = executor.submit(_process_batch, batch)
            in_flight[fut] = len(batch)  # track size for diagnostics

        for original_index, line in enumerate(input_stream):
            if not line.strip():
                continue
            current_batch.append((seq_id, original_index + 1, line))
            seq_id += 1
            if len(current_batch) >= batch_size:
                submit_batch(current_batch)
                current_batch = []
            # Flow control: ensure outstanding lines <= buffer_size
            if (seq_id - next_to_write) >= args.buffer_size:
                done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    results = fut.result()
                    in_flight.pop(fut, None)
                    for r in results:
                        s, out_line, err = r
                        pending_results[s] = (out_line, err)
                while next_to_write in pending_results:
                    out_line, err = pending_results.pop(next_to_write)
                    if out_line is not None:
                        output_stream.write(out_line + "\n")
                        processed += 1
                    if err:
                        logger.warning(err)
                        errors += 1
                    next_to_write += 1
                    if processed and processed % args.log_interval == 0:
                        elapsed = time.time() - start_time
                        logger.info(
                            "Progress: processed=%d errors=%d elapsed=%.1fs rate=%.1f/s inflight_batches=%d pending=%d",
                            processed,
                            errors,
                            elapsed,
                            processed / elapsed if elapsed > 0 else 0,
                            len(in_flight),
                            len(pending_results),
                        )

        # Submit any remaining partial batch
        if current_batch:
            submit_batch(current_batch)
            current_batch = []

        # Drain remaining futures
        while in_flight:
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                results = fut.result()
                in_flight.pop(fut, None)
                for r in results:
                    s, out_line, err = r
                    pending_results[s] = (out_line, err)
            while next_to_write in pending_results:
                out_line, err = pending_results.pop(next_to_write)
                if out_line is not None:
                    output_stream.write(out_line + "\n")
                    processed += 1
                if err:
                    logger.warning(err)
                    errors += 1
                next_to_write += 1
                if processed and processed % args.log_interval == 0:
                    elapsed = time.time() - start_time
                    logger.info(
                        "Progress: processed=%d errors=%d elapsed=%.1fs rate=%.1f/s inflight_batches=%d pending=%d",
                        processed,
                        errors,
                        elapsed,
                        processed / elapsed if elapsed > 0 else 0,
                        len(in_flight),
                        len(pending_results),
                    )

    elapsed = time.time() - start_time
    logger.info(
        "Finished parallel conversion: processed=%d errors=%d elapsed=%.1fs rate=%.1f/s",
        processed,
        errors,
        elapsed,
        processed / elapsed if elapsed > 0 else 0,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

import argparse

# Fast JSON (prefer orjson)
import json as _json_std
import logging
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover - best effort
    import orjson as _orjson  # type: ignore

    def _json_loads(s: str):
        return _orjson.loads(s)

    def _json_dumps(obj) -> str:
        return _orjson.dumps(obj).decode("utf-8")

except Exception:  # pragma: no cover
    _orjson = None

    def _json_loads(s: str):
        return _json_std.loads(s)

    def _json_dumps(obj) -> str:  # type: ignore
        return _json_std.dumps(obj)


from transformers import AutoTokenizer  # noqa: E402

# ----------------------------------------------------------------------------
# Tokenizer (lazy init in workers to avoid heavy fork overhead / repeated load)
# ----------------------------------------------------------------------------
MODEL_NAME = "/hf_models/Qwen2.5-32B-Instruct"
_TOKENIZER = None  # type: ignore
ADD_TOOLS = False  # Will be set in main() if input filename contains 'with-tool'


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        except Exception as e:  # pragma: no cover
            sys.stderr.write(
                f"Could not load tokenizer '{MODEL_NAME}'. Ensure internet access and transformers installed. Error: {e}\n"
            )
            raise
        tok.chat_template = coder_template  # set template once
        _TOKENIZER = tok
    return _TOKENIZER


coder_template = """{% macro render_extra_keys(json_dict, handled_keys) %}
    {%- if json_dict is mapping %}
        {%- for json_key in json_dict if json_key not in handled_keys %}
            {%- if json_dict[json_key] is mapping or (json_dict[json_key] is sequence and json_dict[json_key] is not string) %}
                {{- '\n<' ~ json_key ~ '>' ~ (json_dict[json_key] | tojson | safe) ~ '</' ~ json_key ~ '>' }}
            {%- else %}
                {{-'\n<' ~ json_key ~ '>' ~ (json_dict[json_key] | string) ~ '</' ~ json_key ~ '>' }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
{% endmacro %}

{%- if messages[0]["role"] == "system" %}
    {%- set system_message = messages[0]["content"] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
{%- endif %}

{%- if not tools is defined %}
    {%- set tools = [] %}
{%- endif %}

{%- if system_message is defined %}
    {{- "<|im_start|>system\n" + system_message }}
{%- else %}
    {%- if tools is iterable and tools | length > 0 %}
        {{- "<|im_start|>system\nYou are Qwen, a helpful AI assistant that can interact with a computer to solve tasks." }}
    {%- endif %}
{%- endif %}
{%- if tools is iterable and tools | length > 0 %}
    {{- "\n\n# Tools\n\nYou have access to the following functions:\n\n" }}
    {{- "<tools>" }}
    {%- for tool in tools %}
        {%- if tool.function is defined %}
            {%- set tool = tool.function %}
        {%- endif %}
        {{- "\n<function>\n<name>" ~ tool.name ~ "</name>" }}
        {%- if tool.description is defined %}
            {{- '\n<description>' ~ (tool.description | trim) ~ '</description>' }}
        {%- endif %}
        {{- '\n<parameters>' }}
        {%- if tool.parameters is defined and tool.parameters is mapping and tool.parameters.properties is defined and tool.parameters.properties is mapping %}
            {%- for param_name, param_fields in tool.parameters.properties|items %}
                {{- '\n<parameter>' }}
                {{- '\n<name>' ~ param_name ~ '</name>' }}
                {%- if param_fields.type is defined %}
                    {{- '\n<type>' ~ (param_fields.type | string) ~ '</type>' }}
                {%- endif %}
                {%- if param_fields.description is defined %}
                    {{- '\n<description>' ~ (param_fields.description | trim) ~ '</description>' }}
                {%- endif %}
                {%- set handled_keys = ['name', 'type', 'description'] %}
                {{- render_extra_keys(param_fields, handled_keys) }}
                {{- '\n</parameter>' }}
            {%- endfor %}
        {%- endif %}
        {% set handled_keys = ['type', 'properties'] %}
        {{- render_extra_keys(tool.parameters, handled_keys) }}
        {{- '\n</parameters>' }}
        {%- set handled_keys = ['type', 'name', 'description', 'parameters'] %}
        {{- render_extra_keys(tool, handled_keys) }}
        {{- '\n</function>' }}
    {%- endfor %}
    {{- "\n</tools>" }}
    {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
{%- endif %}
{%- if system_message is defined %}
    {{- '<|im_end|>\n' }}
{%- else %}
    {%- if tools is iterable and tools | length > 0 %}
        {{- '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in loop_messages %}
    {%- if message.role == "assistant" and message.tool_calls is defined and message.tool_calls is iterable and message.tool_calls | length > 0 %}
        {%- if message.content is string %}
            {%- set content = message.content %}
        {%- else %}
            {%- set content = '' %}
        {%- endif %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {{- '<|im_start|>' + message.role }}
        {{- '\n' + '<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') | trim + '\n' }}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
            {%- if tool_call.arguments is defined %}
                {%- for args_name, args_value in tool_call.arguments|items %}
                    {{- '<parameter=' + args_name + '>\n' }}
                    {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}
                    {{- args_value }}
                    {{- '\n</parameter>\n' }}
                {%- endfor %}
            {%- endif %}
            {{- '</function>\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "user" or message.role == "system" %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- if message.content is string %}
            {%- set content = message.content %}
        {%- else %}
            {%- set content = '' %}
        {%- endif %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {{- '<|im_start|>' + message.role + '\n' + '<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') + '<|im_end|>' + '\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.previtem and loop.previtem.role != "tool" %}
            {{- '<|im_start|>user\n' }}
        {%- endif %}
        {{- '<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>\n' }}
        {%- if not loop.last and loop.nextitem.role != "tool" %}
            {{- '<|im_end|>\n' }}
        {%- elif loop.last %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- else %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}"""

# (chat template will be attached during lazy tokenizer initialization)


def convert_using_chat_template(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Converts a list of OpenAI-formatted messages to the Qwen3-Coder format
    using the tokenizer's apply_chat_template method.

    Args:
        messages: A list of message dictionaries in OpenAI chat format.

    Returns:
        A dictionary containing the 'input' and 'output' strings.
    """
    # Find the split point: where the first assistant message appears
    first_assistant_idx = -1
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            first_assistant_idx = i
            break

    if first_assistant_idx == -1:
        raise ValueError("No assistant message found to create the 'output' part.")

    # Prepare messages for template application
    # The Qwen3 chat template expects the 'arguments' in tool_calls to be a dict, not a string.
    for msg in messages:
        if "tool_calls" in msg and msg["tool_calls"]:
            for tool_call in msg["tool_calls"]:
                if "function" in tool_call and isinstance(tool_call["function"].get("arguments"), str):
                    try:
                        tool_call["function"]["arguments"] = _json_loads(tool_call["function"]["arguments"])
                    except Exception:  # noqa: BLE001
                        sys.stderr.write(
                            f"Warning: Could not decode JSON string in tool_call arguments: {tool_call['function']['arguments']}\n"
                        )

    # 1. Generate the 'input' string (the prompt)
    # This includes all messages up to the first assistant turn.
    # `add_generation_prompt=True` adds `<|im_start|>assistant\n` at the end.
    tokenizer = _get_tokenizer()
    # Only include tools if ADD_TOOLS flag is set
    tools = None
    if True:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "stateful_python_code_exec",
                    "description": (
                        "Call this function to execute Python code in a stateful Jupyter notebook environment. "
                        "Python will respond with the output of the execution or time out after 120.0 seconds."
                    ),
                    "parameters": {
                        "properties": {"code": {"description": "Code to execute", "type": "string"}},
                        "required": ["code"],
                        "type": "object",
                    },
                },
            }
        ]
    prompt_messages = messages[:first_assistant_idx]
    input_str = tokenizer.apply_chat_template(prompt_messages, tools=tools, tokenize=False, add_generation_prompt=True)

    # 2. Generate the 'output' string (the full completion)
    # This includes the entire conversation history.
    # We then manually remove the 'input' part to isolate the assistant's output.
    full_conversation_str = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,  # Do not add the prompt token again
    )

    # The output is everything in the full conversation that comes after the input prompt
    if full_conversation_str.startswith(input_str):
        output_str = full_conversation_str[len(input_str) :]
    else:
        # Fallback for safety, though it should not be needed with correct template usage
        sys.stderr.write(
            "Warning: Full conversation does not start with the generated input prompt. Fallback logic used.\n"
        )
        assistant_and_tool_messages = messages[first_assistant_idx:]
        output_str = tokenizer.apply_chat_template(
            assistant_and_tool_messages, tools=tools, tokenize=False, add_generation_prompt=False
        )

    return {"input": input_str, "output": output_str}


# ----------------------------------------------------------------------------
# Parallel processing helpers
# ----------------------------------------------------------------------------


def _process_single_line(payload: Tuple[int, int, str]):
    seq_id, original_line_number, line = payload
    try:
        data = _json_loads(line)
        messages = data.get("messages")
        if not messages:
            raise ValueError("JSON object does not contain a 'messages' key.")
        qwen_data = convert_using_chat_template(messages)
        # add other fields
        keys_to_copy = [k for k in data.keys() if k != "messages"]
        for k in keys_to_copy:
            qwen_data[k] = data[k]
        return seq_id, _json_dumps(qwen_data), None
    except Exception as e:  # noqa: BLE001
        return seq_id, None, f"Skipping line {original_line_number} due to error: {e}"


def _process_batch(payloads: List[Tuple[int, int, str]]):
    out = []
    append = out.append
    get_tok = _get_tokenizer  # local for speed
    for seq_id, original_line_number, line in payloads:
        try:
            data = _json_loads(line)
            messages = data.get("messages")
            if not messages:
                raise ValueError("JSON object does not contain a 'messages' key.")
            # ensure tokenizer init once per worker
            get_tok()
            qwen_data = convert_using_chat_template(messages)
            keys_to_copy = [k for k in data.keys() if k != "messages"]
            for k in keys_to_copy:
                qwen_data[k] = data[k]
            append((seq_id, _json_dumps(qwen_data), None))
        except Exception as e:  # noqa: BLE001
            append((seq_id, None, f"Skipping line {original_line_number} due to error: {e}"))
    return out


def main():  # noqa: C901
    global MODEL_NAME, ADD_TOOLS  # declare globals before any usage
    """CLI entrypoint with optional parallelism."""
    parser = argparse.ArgumentParser(
        description="Convert an OpenAI-formatted JSONL file (messages) to Qwen3-Coder format (input/output)."
    )
    parser.add_argument("input_file", help="Path to the input JSONL file (OpenAI format). Use '-' for stdin.")
    parser.add_argument("output_file", help="Path to the output JSONL file. Use '-' for stdout.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of worker processes (default: CPU count - 1). Use 1 to disable parallelism.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Lines per task batch (default: 20). Larger reduces overhead, increases latency/memory.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=10000,
        help="Max in-flight lines (seq ids not yet written). Acts as backpressure (default: 10000).",
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
        default=5000,
        help="Progress log interval in processed lines (default: 5000).",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Override model name to load tokenizer (default: %(default)s).",
    )
    args = parser.parse_args()

    MODEL_NAME = args.model_name  # override after parsing
    try:
        from os import path as _path

        ADD_TOOLS = "with-tool" in _path.basename(args.input_file)
    except Exception:
        ADD_TOOLS = False

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.buffer_size < args.batch_size:
        parser.error("--buffer-size must be >= --batch-size")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("convert-qwen")

    start_time = time.time()
    logger.info(
        "Starting conversion input=%s output=%s workers=%d batch=%d buffer=%d json_lib=%s model=%s",
        args.input_file,
        args.output_file,
        args.workers,
        args.batch_size,
        args.buffer_size,
        "orjson" if _orjson else "stdlib",
        MODEL_NAME,
    )

    input_stream = open(args.input_file, "r", encoding="utf-8") if args.input_file != "-" else sys.stdin
    output_stream = open(args.output_file, "w", encoding="utf-8") if args.output_file != "-" else sys.stdout

    # Sequential path
    if args.workers == 1:
        processed = 0
        errors = 0
        with input_stream, output_stream:
            for i, line in enumerate(input_stream):
                if not line.strip():
                    continue
                try:
                    data = _json_loads(line)
                    messages = data.get("messages")
                    if not messages:
                        raise ValueError("JSON object does not contain a 'messages' key.")
                    qwen_data = convert_using_chat_template(messages)
                    output_stream.write(_json_dumps(qwen_data) + "\n")
                except Exception as e:  # noqa: BLE001
                    errors += 1
                    logger.warning("Skipping line %d due to error: %s", i + 1, e)
                processed += 1
                if processed % args.log_interval == 0:
                    elapsed = time.time() - start_time
                    logger.info(
                        "Progress processed=%d errors=%d elapsed=%.1fs rate=%.1f/s",
                        processed,
                        errors,
                        elapsed,
                        processed / elapsed if elapsed > 0 else 0,
                    )
        elapsed = time.time() - start_time
        logger.info(
            "Finished sequential processed=%d errors=%d elapsed=%.1fs rate=%.1f/s",
            processed,
            errors,
            elapsed,
            processed / elapsed if elapsed > 0 else 0,
        )
        return

    # Parallel path
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
            in_flight[fut] = len(batch)

        for original_index, line in enumerate(input_stream):
            if not line.strip():
                continue
            current_batch.append((seq_id, original_index + 1, line))
            seq_id += 1
            if len(current_batch) >= batch_size:
                submit_batch(current_batch)
                current_batch = []
            # flow control
            if (seq_id - next_to_write) >= args.buffer_size:
                done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    results = fut.result()
                    in_flight.pop(fut, None)
                    for s, out_line, err in results:
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
                            "Progress processed=%d errors=%d elapsed=%.1fs rate=%.1f/s inflight_batches=%d pending=%d",
                            processed,
                            errors,
                            elapsed,
                            processed / elapsed if elapsed > 0 else 0,
                            len(in_flight),
                            len(pending_results),
                        )

        if current_batch:
            submit_batch(current_batch)

        while in_flight:
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                results = fut.result()
                in_flight.pop(fut, None)
                for s, out_line, err in results:
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
                        "Progress processed=%d errors=%d elapsed=%.1fs rate=%.1f/s inflight_batches=%d pending=%d",
                        processed,
                        errors,
                        elapsed,
                        processed / elapsed if elapsed > 0 else 0,
                        len(in_flight),
                        len(pending_results),
                    )

    elapsed = time.time() - start_time
    logger.info(
        "Finished parallel processed=%d errors=%d elapsed=%.1fs rate=%.1f/s",
        processed,
        errors,
        elapsed,
        processed / elapsed if elapsed > 0 else 0,
    )


if __name__ == "__main__":
    main()

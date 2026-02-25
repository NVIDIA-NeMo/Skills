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
import json
import os
import re
from functools import partial  # <-- Import partial
from pathlib import Path

import pandas as pd
import torch

# LangChain and LangGraph for the ReAct Agent
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from scripts.tools import (
    ALL_TOOLS,
    apply_configuration,
    create_trouble_ticket,
    execute_remote_action,
    inspect_logs,
    orchestrate_workload,
    query_alarm,
    query_container_status,
    query_external_factors,
    query_performance,
    query_power_system,
    query_resource_health,
    query_rf_status,
    query_topology,
    run_diagnostics,
    test_connectivity,
    verify_recovery,
    verify_signaling_path,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def get_bound_tools(row: pd.Series) -> list:
    """
    Takes a data row and returns a list of tools with the 'row' argument
    pre-populated using functools.partial.
    """
    bound_tools = []
    for t in ALL_TOOLS:
        # Create a partial function that has the 'row' argument fixed.
        # The agent will only need to provide the other arguments.
        bound_tool = partial(t, row)
        # LangChain tools copy metadata; we need to ensure the original
        # name, description, and args_schema are preserved.
        bound_tool.name = t.name
        bound_tool.description = t.description
        bound_tool.args_schema = t.args_schema
        bound_tools.append(bound_tool)
    return bound_tools


# The rest of your utility functions (ensure_parent_dir, etc.) remain unchanged.
def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def load_processed_indices(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    processed = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not (line := line.strip()):
                continue
            try:
                obj = json.loads(line)
                if "row_index" in obj:
                    processed.add(int(obj["row_index"]))
            except json.JSONDecodeError:
                print(f"[WARNING] Skipping malformed line in output file: {line}")
    return processed


def append_jsonl_batch(output_path: Path, records: list):
    with output_path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_str(val):
    """Return a JSON-serializable string; avoid NaN/None from pandas."""
    if val is None or (isinstance(val, float) and (val != val or val == float("inf") or val == float("-inf"))):
        return "NotApplicable"
    return str(val)


def format_tool_response(string: str) -> str:
    safe = _safe_str(string)
    json_payload = json.dumps({"summary": safe}, ensure_ascii=False)
    return f"""<tool_response>\n{json_payload}</tool_response>\n"""


def _parse_tool_call_json(raw: str):
    """Parse tool_call content; tolerate unquoted keys (e.g. {name: "..."})."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Fix unquoted keys: word followed by colon -> "word":
    fixed = re.sub(r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)", r'\1"\2"\3', raw)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        raise


def main():
    parser = argparse.ArgumentParser(description="ReAct Agent for Processing Inputs from JSONL (resumable)")
    parser.add_argument("--input", required=True, help="Path to the JSONL file containing inputs")
    parser.add_argument(
        "--output", default="outputs/agent_responses.jsonl", help="Path to the JSONL file to append agent responses"
    )
    parser.add_argument(
        "--weights_dir", default="training/qwen3-32b-improved-hf", help="Local directory for fine-tuned HF weights"
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-32B", help="Tokenizer for the model")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of rows to process before writing to disk (default: 1)"
    )
    parser.add_argument(
        "--gpu",
        required=False,
        help="Optional GPU index (e.g. 0). If omitted, device_map='auto' is used to spread the model across all available GPUs (recommended for large models).",
    )
    parser.add_argument("--limit", required=False)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete any existing output file and start from scratch instead of resuming.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    ensure_parent_dir(output_path)
    if args.fresh and os.path.exists(output_path):
        os.remove(output_path)
    processed_indices = load_processed_indices(output_path)
    print(f"[INFO] Found {len(processed_indices)} already processed rows. Resuming...")

    print(f"[INFO] Loading dataset from {input_path}...")
    df = pd.read_json(str(input_path), lines=True)
    if args.limit:
        df = df.iloc[: int(args.limit)]
    df["row_index"] = df.index

    unprocessed_df = df[~df.index.isin(processed_indices)]
    if unprocessed_df.empty:
        print("[INFO] All rows have already been processed. Exiting.")
        return
    print(f"[INFO] {len(unprocessed_df)} rows remaining to process.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[INFO] Loading model and tokenizer from {args.weights_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Use all available GPUs by default (device_map="auto"); only pin to one GPU if --gpu is set
    device_map = "auto" if not args.gpu else None
    model = AutoModelForCausalLM.from_pretrained(
        args.weights_dir, torch_dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
    )
    if args.gpu:
        model.to(f"cuda:{str(args.gpu)}")
    elif torch.cuda.device_count() > 1:
        print(f"[INFO] Model spread across {torch.cuda.device_count()} GPUs (device_map=auto)")
    model.eval()

    hf_pipeline = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048, temperature=0.7, top_p=0.95
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    chat_llm = ChatHuggingFace(llm=llm)

    checkpointer = MemorySaver()

    output_batch = []
    react_agent = create_react_agent(chat_llm, tools=[], checkpointer=checkpointer)
    for index, row in tqdm(unprocessed_df.iterrows(), total=len(unprocessed_df), desc="Processing Rows"):
        # print(row)
        # exit()

        # --- Step 2: Get the row-specific tools using the helper function. ---
        # row_tools = get_bound_tools(row)

        config = {"configurable": {"thread_id": str(index)}}
        input_question = row["input"]
        system_message = row["system"]
        input_messages = [{"role": "system", "content": system_message}, {"role": "user", "content": input_question}]

        try:
            # We still invoke one-by-one, but the tool creation is now efficient.

            ## WE need to fix this. Add system message and tool calling messages
            # separate user input messages
            TOOLS = {
                "query_alarm": lambda args: query_alarm(**args),
                "query_resource_health": lambda args: query_resource_health(**args),
                "query_performance": lambda args: query_performance(**args),
                "query_topology": lambda args: query_topology(**args),
                "execute_remote_action": lambda args: execute_remote_action(**args),
                "apply_configuration": lambda args: apply_configuration(**args),
                "run_diagnostics": lambda args: run_diagnostics(**args),
                "inspect_logs": lambda args: inspect_logs(**args),
                "create_trouble_ticket": lambda args: create_trouble_ticket(**args),
                "verify_recovery": lambda args: verify_recovery(**args),
                "query_external_factors": lambda args: query_external_factors(**args),
                "orchestrate_workload": lambda args: orchestrate_workload(**args),
                "query_power_system": lambda args: query_power_system(**args),
                "query_rf_status": lambda args: query_rf_status(**args),
                "query_container_status": lambda args: query_container_status(**args),
                "verify_signaling_path": lambda args: verify_signaling_path(**args),
                "test_connectivity": lambda args: test_connectivity(**args),
            }

            end = False
            iterations = 0
            while not end:
                iterations += 1
                if iterations > 55:
                    break

                response = react_agent.invoke(
                    {"messages": input_messages}, config={"configurable": {**config["configurable"]}}
                )

                # Add a while statement to parse in tools or find the conclusion
                final_content = response["messages"][-1].content
                # parsed = tokenizer.parse_response(out_text)
                # print(parsed)
                # exit()
                last_message_start = final_content.rfind("<|im_start|>assistant")
                # last_message_end = final_content.find("<|im_end|>")
                tool_calls = re.search(
                    r"<tool_call>(.*?)</tool_call>",
                    final_content[last_message_start + len("<|im_start|>assistant") :],
                    flags=re.DOTALL,
                )
                # json_obj = react_to_json(response)
                # print(json.dumps(json_obj, indent=2))

                # print(final_content[last_message_start + len("<|im_start|>assistant"):][-20:])
                if tool_calls:
                    tool_called = tool_calls.group(1)
                    try:
                        data = _parse_tool_call_json(tool_called)
                    except json.JSONDecodeError as e:
                        print(f"\n[WARN] Row {index}: could not parse tool_call JSON: {e}")
                        break
                    tool_name = data.get("name") or data.get("function", {}).get("name")
                    if not tool_name:
                        print(f"[WARN] No tool name in parsed data: {data}")
                        break
                    arguments = data.get("arguments", data.get("parameters", {}))
                    if not isinstance(arguments, dict):
                        arguments = {}
                    if tool_name not in TOOLS:
                        print(f"[WARN] Unknown tool: {tool_name}")
                        continue
                    arguments["row"] = row
                    result = TOOLS[tool_name](arguments)
                    input_messages.append(
                        {
                            "role": "assistant",
                            "content": final_content[last_message_start + len("<|im_start|>assistant") :],
                        }
                    )
                    input_messages.append({"role": "user", "content": format_tool_response(_safe_str(result))})
                    print(f"✅ {tool_name} -> {result}")

                else:
                    input_messages.append(
                        {
                            "role": "assistant",
                            "content": final_content[last_message_start + len("<|im_start|>assistant") :],
                        }
                    )
                    end = True
                    # print("Conclusion Arrived")

            # if found conclusion, find the close code as well as the final_content

            # the final content needs to be repeatitively added.

            output_row = row.to_dict()
            output_row["agent_response"] = input_messages
            output_batch.append(output_row)

        except Exception as e:
            print(f"\n[ERROR] An error occurred on row {index}: {e}")
            err_row = row.to_dict()
            err_row["row_index"] = int(index)
            err_row["input"] = input_question
            err_row["error"] = repr(e)
            if input_messages:
                err_row["agent_response"] = input_messages
            output_batch.append(err_row)

        if len(output_batch) >= args.batch_size:
            append_jsonl_batch(output_path, output_batch)
            output_batch = []

    if output_batch:
        append_jsonl_batch(output_path, output_batch)

    print(f"[DONE] All responses have been written to: {output_path}")


if __name__ == "__main__":
    main()

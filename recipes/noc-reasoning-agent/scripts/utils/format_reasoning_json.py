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
import copy
import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer


def _incident_id(data):
    """Synthetic schema uses incident_identifier; legacy uses number."""
    incident_id = data.get("incident_identifier") or data.get("number")
    if incident_id is None:
        raise ValueError(f"Missing incident identifier in data: {list(data.keys())}")
    return incident_id


def _resolution_method(data):
    """Synthetic schema uses root_cause_secondary; legacy uses close_code."""
    return data.get("root_cause_secondary") or data.get("close_code", "")


def extract_formatted_json_steps(input_file):
    """
    Extracts a JSON array string from a larger block of text.

    Args:
        text (str): The raw text containing the JSON array.

    Returns:
        list: The parsed JSON object (a list of dictionaries).
        Returns None if no valid JSON array is found.
    """

    responses = {}
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if not data:
                    continue
                text = data["generation"]

                number = _incident_id(data)

                try:
                    # Find the starting position of the JSON array '['

                    start_index = text.rfind("<|message|>")
                    text = text[start_index + len("<|message|>") :]
                    start_index = text.find("[")
                    # Find the last position of the JSON array ']' to ensure we get the whole thing
                    end_index = text.rfind("]") + 1

                    if start_index != -1 and end_index != -1:
                        # Slice the string to get only the JSON part
                        json_string = text[start_index:end_index]

                        # Parse the JSON string into a Python object
                        parsed_json = json.loads(json_string)
                        responses[number] = parsed_json
                    else:
                        print(text)
                        print("Error: Could not find the start '[' or end ']' of the JSON array.")
                        continue
                except json.JSONDecodeError as e:
                    print(text)
                    print(f"Error decoding JSON: {e}")
            except json.JSONDecodeError:
                print(f"Skipping invalid line: {line.strip()}")

    return responses


def extract_final_thinking_processes(input_file):
    responses = {}
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data["generation"]
            number = _incident_id(data)
            step_number = data["step_number"]
            if number not in responses:
                responses[number] = {}

            thinking = text[text.rfind("final<|message|>") + len("final<|message|>") :]
            data["generation"] = thinking
            responses[number][step_number] = thinking

    return responses


def prepare_data_for_reasoning_traces(jsonl_file, input_file, output_file):
    formatted_steps_taken = extract_formatted_json_steps(input_file)
    new_jsonl = []

    incorrect_incidents = 0
    # Read the file line by line
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if not data:
                continue
            number = _incident_id(data)

            if number in formatted_steps_taken:
                formatted_steps = formatted_steps_taken[number]
                current_conclusion = ""
                for i in range(len(formatted_steps)):
                    sub_data = copy.deepcopy(data)
                    current_steps = formatted_steps[i]
                    sub_data["step_number"] = current_steps["step_number"]
                    sub_data["background_context"] = current_conclusion
                    conclusion_called = f"Step {current_steps['step_number']} {current_steps['sop_step_title']} {current_steps['status']}.\nAction taken: {current_steps['action_taken']}\n"
                    tool_response = ""
                    if current_steps["tool_call"]:
                        conclusion_called += f"Tool called: {current_steps['tool_call']}\n"
                        tool_response = f"Tool response: {current_steps['result']}\n"
                    else:
                        conclusion_called += "No tool call needed.\n"
                    sub_data["outcome"] = conclusion_called
                    new_jsonl.append(sub_data)
                    current_conclusion += conclusion_called + tool_response
                # data["formatted_steps"] = formatted_steps_taken[number]

                # new_jsonl.append(data)
            else:
                incorrect_incidents += 1

    # print(json.dumps(new_jsonl, indent = 4))
    print(f"{incorrect_incidents} incidents were not parsed correctly and discarded.")

    with open(output_file, "w", encoding="utf-8") as f:
        for line in new_jsonl:
            json.dump(line, f)
            f.write("\n")

    print(f"Wrote {len(new_jsonl)} entries to {output_file}")


def token_converting(string, model):
    """
    Converts a shorthand tool command like:
      Check_Alarm_Status[site-123]
    into a Qwen-32B compliant <tool_call> XML block.
    """
    if model != "qwen32":
        return string  # fallback for other models

    import re

    # --- 1. Parse tool name and the raw arguments inside [...] or (...) ---
    # Match "ToolName[args]" or "ToolName[ args ]"
    m = re.match(r"^\s*([A-Za-z_]\w*)\s*\[(.*)\]\s*$", str(string), re.DOTALL)

    if not m:
        # Also accept parenthesis format: ToolName(args) or ToolName()
        m = re.match(r"^\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$", str(string), re.DOTALL)

    if not m:
        m_no_args = re.match(r"^\s*([A-Za-z_]\w*)\s*[\[\(]\s*[\]\)]\s*$", str(string))
        if m_no_args:
            tool_name = m_no_args.group(1)
            raw_args = ""
        else:
            return string
    else:
        tool_name, raw_args = m.groups()

    # --- 2. Smart Splitter ---
    # Splits by commas, but ignores commas inside single/double quotes.
    # e.g. "dept, 'Error in rack 1, shelf 2'" -> ["dept", "'Error in rack 1, shelf 2'"]
    parts = re.split(r'\s*,\s*(?=(?:[^\'"]|\'[^\']*\'|"[^"]*")+$)', raw_args.strip()) if raw_args.strip() else []

    # --- 3. Normalize Tokens ---
    kv_args = {}
    pos_args = []

    for p in parts:
        if not p:
            continue
        # Check for key=value or key: value
        if ("=" in p or ":" in p) and not (p.startswith("'") or p.startswith('"')):
            k, v = re.split(r"\s*[:=]\s*", p, maxsplit=1)
            v = v.strip().strip('"').strip("'")
            kv_args[k.strip()] = v
        else:
            pos_args.append(p.strip().strip('"').strip("'"))

    # Helper to enforce positional argument counts
    def req_pos(n, arg_name="argument"):
        if len(pos_args) < n:
            raise ValueError(
                f"{tool_name} requires at least {n} value(s) (missing {arg_name}); got {len(pos_args)} in: {string}"
            )

    # --- 4. Tool-Specific Argument Mapping ---
    # When no arguments are provided (model used tool_name() format), use "all"
    # as default so the pipeline doesn't lose the entire incident.

    def _first_pos(key, named_key=None):
        """Return named arg, first positional arg, or 'all' as default."""
        if named_key:
            val = kv_args.get(named_key)
            if val:
                return val
        return pos_args[0] if pos_args else "all"

    arg_dict = {}

    if tool_name == "query_alarm":
        arg_dict = {"site_or_element_id": _first_pos("site_or_element_id")}

    elif tool_name == "query_resource_health":
        arg_dict = {"element_id": _first_pos("element_id")}

    elif tool_name == "query_performance":
        arg_dict = {"metric_type": _first_pos("metric_type")}

    elif tool_name == "query_topology":
        arg_dict = {"element_id": _first_pos("element_id")}

    elif tool_name == "execute_remote_action":
        elem = kv_args.get("element_id") or (pos_args[0] if pos_args else "all")
        act = kv_args.get("action") or (pos_args[1] if len(pos_args) > 1 else "default_action")
        arg_dict = {"element_id": elem, "action": act}

    elif tool_name == "apply_configuration":
        elem = kv_args.get("element_id") or (pos_args[0] if pos_args else "all")
        cfg = kv_args.get("config_type") or (pos_args[1] if len(pos_args) > 1 else None)
        arg_dict = {"element_id": elem}
        if cfg:
            arg_dict["config_type"] = cfg

    elif tool_name == "run_diagnostics":
        arg_dict = {"diagnostic_type": _first_pos("diagnostic_type")}

    elif tool_name == "inspect_logs":
        arg_dict = {"log_type": _first_pos("log_type")}

    elif tool_name == "create_trouble_ticket":
        pri = kv_args.get("priority") or (pos_args[0] if pos_args else "medium")
        team = kv_args.get("team") or (pos_args[1] if len(pos_args) > 1 else "unknown")
        details = kv_args.get("issue_details") or (
            ", ".join(pos_args[2:]) if len(pos_args) > 2 else "No details provided"
        )
        arg_dict = {"priority": pri, "team": team, "issue_details": details}

    elif tool_name == "verify_recovery":
        arg_dict = {"element_id": _first_pos("element_id")}

    elif tool_name == "query_external_factors":
        arg_dict = {"site_or_area": _first_pos("site_or_area")}

    elif tool_name == "orchestrate_workload":
        act = kv_args.get("action") or (pos_args[0] if pos_args else "default")
        typ = kv_args.get("type") or (pos_args[1] if len(pos_args) > 1 else None)
        arg_dict = {"action": act}
        if typ:
            arg_dict["type"] = typ

    elif tool_name == "query_power_system":
        arg_dict = {"target": _first_pos("target")}

    elif tool_name == "query_rf_status":
        arg_dict = {"sector_or_antenna_id": _first_pos("sector_or_antenna_id")}

    elif tool_name == "query_container_status":
        arg_dict = {"type": _first_pos("type")}

    elif tool_name == "verify_signaling_path":
        arg_dict = {"interface": _first_pos("interface")}

    elif tool_name == "test_connectivity":
        arg_dict = {"test_type": _first_pos("test_type")}

    # --- Fallback for unknown tools ---
    else:
        if kv_args:
            arg_dict = kv_args
        elif pos_args:
            arg_dict = {"args": pos_args} if len(pos_args) > 1 else {"argument": pos_args[0]}
        else:
            arg_dict = {}

    # --- 5. Construct XML Output ---
    json_call = {"name": tool_name, "arguments": arg_dict}
    return json_call


def merge_reasoning_steps(steps_taken, reasoning_steps, model="qwen32"):
    broken_numbers = []
    for number in steps_taken:
        if number in reasoning_steps:
            # fix tool calling
            try:
                for i in range(len(steps_taken[number])):
                    if steps_taken[number][i]["tool_call"]:
                        steps_taken[number][i]["tool_call"] = token_converting(
                            steps_taken[number][i]["tool_call"], model
                        )
                    steps_taken[number][i]["thinking"] = reasoning_steps[number][steps_taken[number][i]["step_number"]]
            except (KeyError, ValueError) as e:
                print(f"Error merging steps for incident {number}: {e}")
                broken_numbers.append(number)

    for number in broken_numbers:
        del steps_taken[number]

    return steps_taken


SFT_DUMMY_USER = "DUMMY_USER_FOR_SFT"
SFT_ASSISTANT_SENTINEL = "<<<ASSISTANT_SENTINEL>>>"


def compute_prefix_len_for_dummy_user(tokenizer):
    messages = [
        {"role": "user", "content": SFT_DUMMY_USER},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_special_tokens=False,
        add_generation_prompt=False,
    )

    idx = len(rendered)

    # Keep everything from the sentinel onward, drop everything before it
    return idx


def qwen_token_converter(data, full_reasoning_steps, tokenizer=None):
    curriculum_learning_stages = {}
    turn = 0
    total_tokens = 0
    pre_compute_idx = compute_prefix_len_for_dummy_user(tokenizer)
    current_assistant_content = [{"role": "user", "content": SFT_DUMMY_USER}]

    for i in range(len(full_reasoning_steps)):
        step = full_reasoning_steps[i]

        thinking = step.get("thinking", "")
        status = step.get("status", "")
        title = step.get("sop_step_title", "")
        action = step.get("action_taken", "")
        tool_call = step.get("tool_call", "")
        result = step.get("result", "")
        step_text = f"<think>\n{thinking} {status} {title}: {action}\n</think>\n"

        # Construct the text for this specific step
        # Note: We inject <think> tags here as part of the content
        response_message = [{"role": "user", "content": SFT_DUMMY_USER}]
        sub_data = copy.deepcopy(data)

        # --- CASE A: Tool Call Triggered ---
        if tool_call:
            # Response String
            response_message.append(
                {
                    "role": "assistant",
                    "content": step_text,
                    "tool_calls": [{"type": "function", "function": tool_call}],
                }
            )
            raw_response = tokenizer.apply_chat_template(
                response_message, tokenize=False, add_special_tokens=False, add_generation_prompt=False
            )
            cleaned_response = raw_response[pre_compute_idx:]
            sub_data["response"] = cleaned_response

            # Background String
            raw_background = tokenizer.apply_chat_template(
                current_assistant_content, tokenize=False, add_special_tokens=False, add_generation_prompt=False
            )
            cleaned_background = raw_background[pre_compute_idx:]
            sub_data["background"] = cleaned_background

            # Next Context
            current_assistant_content.append(
                {
                    "role": "assistant",
                    "content": step_text,
                    "tool_calls": [{"type": "function", "function": tool_call}],
                }
            )
            current_assistant_content.append({"role": "tool", "content": result})
            # print(raw)
            # print("----:")
            # print(cleaned)
            # exit()

            curriculum_learning_stages[turn] = sub_data
            turn += 1

        # --- CASE B: Final Conclusion ---
        elif i == len(full_reasoning_steps) - 1:
            total_tokens = len(
                tokenizer.apply_chat_template(current_assistant_content, tokenize=True, add_generation_prompt=False)
            )
            sub_data = copy.deepcopy(data)

            result = result if result else ""

            response_message.append(
                {
                    "role": "assistant",
                    "content": step_text + result + f"\nClose Code: [{_resolution_method(sub_data)}]",
                }
            )
            raw = tokenizer.apply_chat_template(
                response_message, tokenize=False, add_special_tokens=False, add_generation_prompt=False
            )
            cleaned = raw[pre_compute_idx:]
            sub_data["response"] = cleaned

            # Background String
            raw_background = tokenizer.apply_chat_template(
                current_assistant_content, tokenize=False, add_special_tokens=False, add_generation_prompt=False
            )
            cleaned_background = raw_background[pre_compute_idx:]
            sub_data["background"] = cleaned_background

            curriculum_learning_stages[turn] = sub_data

        # --- CASE C: Intermediate Step (just accumulation) ---
        else:
            # We already added to current_assistant_content at the top of loop
            pass

    # --- CASE D: Forced Conclusion ---
    # If the last step had a tool_call (Case B never triggered), append
    # an extra conclusion turn so the model learns to output a Close Code.
    if turn > 0 and (turn - 1) in curriculum_learning_stages and turn not in curriculum_learning_stages:
        close_code = _resolution_method(data)
        if close_code:
            total_tokens = len(
                tokenizer.apply_chat_template(current_assistant_content, tokenize=True, add_generation_prompt=False)
            )
            sub_data = copy.deepcopy(data)
            conclusion_msg = [
                {"role": "user", "content": SFT_DUMMY_USER},
                {
                    "role": "assistant",
                    "content": f"<think>\nAll troubleshooting steps have been completed and the incident has been resolved.\n</think>\n\nClose Code: [{close_code}]",
                },
            ]
            raw = tokenizer.apply_chat_template(
                conclusion_msg, tokenize=False, add_special_tokens=False, add_generation_prompt=False
            )
            sub_data["response"] = raw[pre_compute_idx:]
            raw_background = tokenizer.apply_chat_template(
                current_assistant_content, tokenize=False, add_special_tokens=False, add_generation_prompt=False
            )
            sub_data["background"] = raw_background[pre_compute_idx:]
            curriculum_learning_stages[turn] = sub_data

    return curriculum_learning_stages, total_tokens


def compile_reasoning(jsonl_file, input_file, output_dir, reasoning_jsonl, tokenizer_name="Qwen/Qwen3-32B"):
    # 1. LOAD TOKENIZER ONCE HERE
    tokenizer = None
    print("Loading Tokenizer (Qwen3-32B)...")
    # Trust remote code is often needed for Qwen tokenizers
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    formatted_steps_taken = extract_formatted_json_steps(input_file)
    formatted_reasoning_steps_taken = extract_final_thinking_processes(reasoning_jsonl)

    full_steps = merge_reasoning_steps(formatted_steps_taken, formatted_reasoning_steps_taken)

    all_tokens = []
    stages = {}
    incorrect_incidents = 0

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            data = json.loads(line)
            number = _incident_id(data)

            if number in full_steps:
                # 2. PASS TOKENIZER TO THE FUNCTION
                try:
                    steps_data, tokens = qwen_token_converter(data, full_steps[number], tokenizer)
                    for stage in steps_data:
                        if stage not in stages:
                            stages[stage] = []
                        stages[stage].append(steps_data[stage])

                    if tokens > 0:
                        all_tokens.append(tokens)
                except (KeyError, ValueError) as e:
                    print(f"Error for incident {number}: {e}")
                    incorrect_incidents += 1
            else:
                incorrect_incidents += 1

    # ... (Rest of your writing logic remains the same) ...
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(stages)):
        name = os.path.join(output_dir, f"iteration_{i}.jsonl")
        with open(name, "w", encoding="utf-8") as f:
            for line in stages[i]:
                json.dump(line, f)
                f.write("\n")

    print(f"CURRICULUM Info\n{'*' * 20}")
    print(f"There are currently {len(stages)} stages")
    print(f"{incorrect_incidents} incidents failed")


def main(jsonl_file, input_file, output_file, parse_types, reasoning_jsonl=None, output_dir=None):
    if parse_types == "steps_extraction":
        prepare_data_for_reasoning_traces(jsonl_file, input_file, output_file)
    elif parse_types == "compile_reasoning":
        if not reasoning_jsonl:
            raise ValueError("Please specify a reasoning jsonl file by specifying --reasoning_jsonl")
        compile_reasoning(jsonl_file, input_file, output_dir, reasoning_jsonl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and format reasoning steps from JSONL file.")
    parser.add_argument("--input", type=str, help="Path to the first JSONL file")
    parser.add_argument("--output", required=False, type=str)
    parser.add_argument("--jsonl_file", required=False, type=str)
    parser.add_argument("--parse_type", type=str)
    parser.add_argument("--output_dir", required=False)
    parser.add_argument("--reasoning_jsonl", required=False, type=str)

    parsing_types = ["steps_extraction", "compile_reasoning"]
    args = parser.parse_args()

    if args.parse_type not in parsing_types:
        raise ValueError(f"{args.parse_type} is not supported. Supported parse_types include {parsing_types}")

    main(args.jsonl_file, args.input, args.output, args.parse_type, args.reasoning_jsonl, args.output_dir)

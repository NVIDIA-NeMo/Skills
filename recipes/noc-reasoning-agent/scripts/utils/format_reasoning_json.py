import argparse
import copy
import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer


def _incident_id(data):
    """Synthetic schema uses incident_identifier; legacy uses number."""
    return data.get("incident_identifier") or data.get("number")


def _resolution_method(data):
    """Synthetic schema uses resolution_method; legacy uses close_code."""
    return data.get("resolution_method") or data.get("close_code", "")


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
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
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
    print(f"{incorrect_incidents} incidents were not parsed correctly and disgarded.")

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

    # --- 1. Parse tool name and the raw arguments inside [...] ---
    # Match "ToolName[args]" or "ToolName[ args ]"
    m = re.match(r"^\s*([A-Za-z_]\w*)\s*\[(.*)\]\s*$", str(string), re.DOTALL)

    if not m:
        # Handle case with no arguments, e.g., Check_Time[]
        m_no_args = re.match(r"^\s*([A-Za-z_]\w*)\s*\[\s*\]\s*$", str(string))
        if m_no_args:
            tool_name = m_no_args.group(1)
            raw_args = ""
        else:
            # If it doesn't match the syntax, return original string or raise error
            # returning string allows the LLM to fail gracefully or retry
            return string

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

    arg_dict = {}

    # 1. Check_Alarm_Status[<site_or_element_id>]
    if tool_name == "Check_Alarm_Status":
        val = kv_args.get("site_or_element_id") or (pos_args[0] if pos_args else None)
        if not val:
            req_pos(1, "site_or_element_id")
        arg_dict = {"site_or_element_id": val}

    # 2. Check_Element_Neighbors[<element_id>]
    elif tool_name == "Check_Element_Neighbors":
        val = kv_args.get("element_id") or (pos_args[0] if pos_args else None)
        if not val:
            req_pos(1, "element_id")
        arg_dict = {"element_id": val}

    # 3. Check_Element_Health[<element_id>]
    elif tool_name == "Check_Element_Health":
        val = kv_args.get("element_id") or (pos_args[0] if pos_args else None)
        if not val:
            req_pos(1, "element_id")
        arg_dict = {"element_id": val}

    # 4. Execute_Remote_Action[<element_id>, '<action>']
    elif tool_name == "Execute_Remote_Action":
        elem = kv_args.get("element_id")
        act = kv_args.get("action")

        if not elem and len(pos_args) > 0:
            elem = pos_args[0]
        if not act and len(pos_args) > 1:
            act = pos_args[1]

        if not elem or not act:
            raise ValueError(f"{tool_name} requires 'element_id' and 'action'.")
        arg_dict = {"element_id": elem, "action": act}

    # 5. Check_External_Issues[<site_or_area>]
    elif tool_name == "Check_External_Issues":
        val = kv_args.get("site_or_area") or (pos_args[0] if pos_args else None)
        if not val:
            req_pos(1, "site_or_area")
        arg_dict = {"site_or_area": val}

    # 6. Check_Apply_Configuration[<element_id>]
    elif tool_name == "Check_Apply_Configuration":
        val = kv_args.get("element_id") or (pos_args[0] if pos_args else None)
        if not val:
            req_pos(1, "element_id")
        arg_dict = {"element_id": val}

    # 7. Check_Performance['<kpi_metric_name>']
    elif tool_name == "Check_Performance":
        val = kv_args.get("kpi_metric_name") or (pos_args[0] if pos_args else None)
        if not val:
            req_pos(1, "kpi_metric_name")
        arg_dict = {"kpi_metric_name": val}

    # 8. Create_Ticket['<department_name>', '<issue_details>']
    elif tool_name == "Create_Ticket":
        dept = kv_args.get("department_name")
        details = kv_args.get("issue_details")

        # Handle positional logic
        if not dept and len(pos_args) >= 1:
            dept = pos_args[0]

        # If details weren't named, we assume everything after department is the details.
        # We join them back with commas in case the split separated a sentence.
        if not details and len(pos_args) >= 2:
            details = ", ".join(pos_args[1:])

        if not dept or not details:
            raise ValueError(f"{tool_name} requires 'department_name' and 'issue_details'.")

        arg_dict = {"department_name": dept, "issue_details": details}

    # 9. Orchestration_tool['<action_command>']
    elif tool_name == "Orchestration_tool":
        val = kv_args.get("action_command") or (pos_args[0] if pos_args else None)
        if not val:
            req_pos(1, "action_command")
        arg_dict = {"action_command": val}

    # 10. Triage_Toolkit_Tool['<issue_type>']
    elif tool_name == "Triage_Toolkit_Tool":
        val = kv_args.get("issue_type") or (pos_args[0] if pos_args else None)
        if not val:
            req_pos(1, "issue_type")
        arg_dict = {"issue_type": val}

    # 11. Check_remote_files[<element_id>]
    elif tool_name == "Check_remote_files":
        val = kv_args.get("element_id") or (pos_args[0] if pos_args else None)
        if not val:
            req_pos(1, "element_id")
        arg_dict = {"element_id": val}

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
            except Exception as e:
                print(e)
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
                except Exception as e:
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

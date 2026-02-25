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
import random
import re

import yaml
from scripts.tools import ALL_TOOLS_STRING


def extract_number_from_input(input_text):
    """
    Extracts the incident identifier from the 'input' field using regex.
    Supports synthetic IDs (e.g. INCME-100001) and legacy (INCWLS...).
    """
    # Match any non-whitespace after "Number:" (e.g. INCME-100001, INCWLS0873337)
    match = re.search(r"Number:\s*(\S+)", input_text)
    if match:
        return match.group(1)
    return None


def get_tools(text):
    matches = {}

    # Find all <tool_call>...</tool_call> blocks

    tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL)
    tool_response = re.findall(r"<tool_response>(.*?)</tool_response>", text, flags=re.DOTALL)
    # print(tool_calls)
    if len(tool_calls) != len(tool_response):
        raise ValueError(f"Mismatch: {len(tool_calls)} tool_calls vs {len(tool_response)} tool_responses")
    for i in range(len(tool_calls)):
        # try:
        tool_block = tool_calls[i]
        response_block = tool_response[i]
        # Extract the JSON portion inside the tags
        tool_json_str = tool_block.strip()

        tool_data = json.loads(tool_json_str)
        response = response_block.strip()
        tool_name = tool_data["name"]
        arguments = tool_data["arguments"]

        matches[tool_name] = {"arguments": arguments, "response": response}

    # except json.JSONDecodeError as e:
    #     print(f"Skipping invalid JSON: {e}")

    if not matches:
        # print("No tools!")
        return None, None
    # print(matches)
    return matches


def main(file1_path, file2_path, prompt_config, output_path="output.jsonl"):
    # Load first JSONL: keyed by 'number' (extracted if needed)
    data1 = {}
    with open(prompt_config, "r") as f:
        prompt_template = yaml.safe_load(f)

    system_prompt = prompt_template["system"]
    with open(file1_path, "r", encoding="utf-8") as f1:
        for line in f1:
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                    number = d.get("incident_identifier", d.get("number"))
                    if d.get("expected") or "Close Code: [" in d.get("response", ""):
                        matches = get_tools(d.get("initial_background", ""))
                        if matches == (None, None):
                            print(f"No tools for incident {number}, skipping")
                            continue
                            # print(data1[number])
                        d["tool_matches"] = matches
                        formatted_prompt = prompt_template["user"].format(**d)
                        if formatted_prompt.endswith("\n<think>\n"):
                            formatted_prompt = formatted_prompt[: -len("\n<think>\n")]
                        # print(system_prompt)
                        d["formatted_input"] = formatted_prompt
                        data1[number] = d

                    # num = d.get('number') or extract_number_from_input(d.get('input', ''))
                    # if num:
                    #     data1[num] = d
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file1: {e}")

    # Load second JSONL: keyed by 'number'
    data2 = {}
    with open(file2_path, "r", encoding="utf-8") as f2:
        for line in f2:
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                    input_string = d["input"]
                    output_string = d["output"]
                    match = re.search(r"Number:\s*(\S+)", input_string)
                    if match:
                        number = match.group(1)
                    else:
                        raise ValueError("No incident identifier match found in input")
                    data2[number] = [input_string, output_string]
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file2: {e}")

    # Build consolidated results for matching numbers

    results = []
    for num in data1.keys():
        if num in data2:
            used_tools = data1[num]["tool_matches"]

            consolidated = {}
            tools = ALL_TOOLS_STRING
            if used_tools is None:
                print("no tools!")
            else:
                for tool in tools:
                    if tool in used_tools:
                        consolidated[tool] = used_tools[tool]["response"]
            # print(consolidated)
            # print(used_tools)

            consolidated["system"] = system_prompt
            consolidated["input"] = data1[num]["formatted_input"]
            consolidated["expected"] = data1[num].get("expected", data2[num][1])
            # consolidated["output"] =

            results.append(consolidated)

    random.shuffle(results)
    with open(output_path, "w", encoding="utf-8") as out_file:
        for res in results:
            out_file.write(json.dumps(res) + "\n")
    print(f"Consolidated output written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate data from two JSONL files.")
    parser.add_argument("file1", help="Path to the first JSONL file (with input, output, expected_answer)")
    parser.add_argument("file2", help="Path to the second JSONL file (with check answers)")
    parser.add_argument("--prompt_config", default="data/prompts/prompt_incident.yaml")
    parser.add_argument(
        "--output", default="output.jsonl", help="Path to the output JSONL file (default: output.jsonl)"
    )
    args = parser.parse_args()
    main(args.file1, args.file2, args.prompt_config, args.output)

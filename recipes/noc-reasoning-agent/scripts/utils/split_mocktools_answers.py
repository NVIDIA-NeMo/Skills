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


def parse_generation(gen_str):
    """
    Parses a generation string into a dictionary where keys are before ':' and values are after.
    """
    parsed = {}
    if not gen_str:
        return parsed
    lines = gen_str.split("\n")
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            parsed[key] = value
    return parsed


def main(file1_path, file2_path, output_path=None):
    """
    Processes two JSONL files: extracts specified fields from the first file,
    includes 'generation' from the first file and 'generation2' from the second file (matched by 'number'),
    parses the 'generation2' into additional answer columns,
    and outputs the result as JSONL to stdout or a file.

    Args:
    file1_path (str): Path to the first JSONL file.
    file2_path (str): Path to the second JSONL file.
    output_path (str, optional): Path to the output JSONL file. If None, prints to stdout.
    """
    # Define the answer keys to extract
    answer_keys = [
        "Check_Triage_toolkit_answer",
        "Check_Alarm_Status_on_System_answer",
        "Remote_Connection_answer",
        "Remote_Device_Reboot_answer",
        "Check_Fiber_Issues_answer",
        "Check_Power_Issues_answer",
        "Check_Element_Neighbors_answer",
    ]

    # Load data from first file, keyed by 'number'
    data1 = {}
    with open(file1_path, "r", encoding="utf-8") as f1:
        for line_num, line in enumerate(f1, 1):
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                    num = d.get("number")
                    if num:
                        data1[num] = d
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file1 on line {line_num}: '{line}'. Error: {e}")

    # Load generations from second file, keyed by 'number'
    data2 = {}
    with open(file2_path, "r", encoding="utf-8") as f2:
        for line_num, line in enumerate(f2, 1):
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                    num = d.get("number")
                    if num:
                        data2[num] = d.get("generation", "")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file2 on line {line_num}: '{line}'. Error: {e}")

    # Build results for matching numbers
    results = []
    for num, d1 in data1.items():
        if num in data2:
            # Parse generation2
            parsed2 = parse_generation(data2[num])

            extracted = {
                "number": num,
                "u_region2": d1.get("u_region2", ""),
                "category": d1.get("category", ""),
                "subcategory": d1.get("subcategory", ""),
                "u_market2": d1.get("u_market2", ""),
                "u_ran_vendor": d1.get("u_ran_vendor", ""),
                "u_aoi2": d1.get("u_aoi2", ""),
                "priority": d1.get("priority", ""),
                "u_locations": d1.get("u_locations", ""),
                "short_description": d1.get("short_description", ""),
                "opened_at": d1.get("opened_at", ""),
                "follow_up": d1.get("follow_up", ""),
                "assignment_group": d1.get("assignment_group", ""),
                "problem code": d1.get("u_problem_code", ""),
                "generation": d1.get("generation", ""),
                "generation2": data2[num],
                "close_notes": d1.get("close_notes", ""),
            }

            # Add the parsed answer columns from generation2
            for key in answer_keys:
                extracted[key] = parsed2.get(key, "NotApplicable")

            results.append(extracted)

    # Output
    if output_path:
        with open(output_path, "w", encoding="utf-8") as out_file:
            for res in results:
                out_file.write(json.dumps(res) + "\n")
        print(f"Output written to {output_path}")
    else:
        for res in results:
            print(json.dumps(res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and combine data from two JSONL files.")
    parser.add_argument("file1", help="Path to the first JSONL file")
    parser.add_argument("file2", help="Path to the second JSONL file")
    parser.add_argument("--output", help="Optional path to output JSONL file (default: print to stdout)")
    args = parser.parse_args()
    main(args.file1, args.file2, args.output)

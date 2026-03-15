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
from pathlib import Path

import pandas as pd
import torch
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def load_processed_indices(output_path: Path):
    if not output_path.exists():
        return set()
    processed = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # we record row_index for resume
                if "row_index" in obj:
                    processed.add(int(obj["row_index"]))
            except json.JSONDecodeError:
                continue
    return processed


def append_jsonl(output_path: Path, obj: dict):
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="ReAct Agent for Processing Inputs from JSONL (resumable)")
    parser.add_argument("--input", help="Path to the JSONL file containing inputs and answers")
    parser.add_argument(
        "--output",
        default="outputs/agent_responses.jsonl",
        help="Path to the JSONL file to append agent responses (default: outputs/agent_responses.jsonl)",
    )
    parser.add_argument(
        "--weights_dir",
        default="training/qwen2.5-32b-improved-hf",
        help="Local directory for fine-tuned HF weights",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    ensure_parent_dir(output_path)

    # Resume support: read any already-processed rows
    processed_indices = load_processed_indices(output_path)
    print(f"[INFO] Will resume. Already processed rows: {len(processed_indices)}")

    # Load input JSONL
    df = pd.read_json(str(input_path), lines=True)

    # Load tokenizer and model once
    tokenizer = AutoTokenizer.from_pretrained(args.weights_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.weights_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # Create generation pipeline and LangChain wrappers
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    chat_llm = ChatHuggingFace(llm=llm)

    # Process each row, appending immediately to output
    for index, row in tqdm(df.iterrows()):
        if index in processed_indices:
            # Skip already done
            continue

        input_question = row.get("input", "")
        print(f"Processing input {index + 1}/{len(df)}")

        # --- Updated Tool Definitions ---

        @tool
        def Check_Alarm_Status(site_or_element_id: str) -> str:
            """
            Retrieves current alarm details, severity, and active time
            for a given site or network element.
            """
            return row.get("Check_Alarm_Status_answer", "NotApplicable")

        @tool
        def Check_Element_Neighbors(element_id: str) -> str:
            """
            Checks all adjacent and upstream devices of a target element
            to find common alarms affecting the area.
            """
            return row.get("Check_Element_Neighbors_answer", "NotApplicable")

        @tool
        def Check_Element_Health(element_id: str) -> str:
            """
            Polls the element (e.g., DU/RU) to retrieve key health
            metrics like cell status and radiation.
            """
            return row.get("Check_Element_Health_answer", "NotApplicable")

        @tool
        def Execute_Remote_Action(element_id: str, action: str) -> str:
            """
            Runs a specific remote command on an element.
            Example Actions: 'unlock_cell', 'restart_du', 'restore_ru'
            """
            return row.get("Execute_Remote_Action_answer", "NotApplicable")

        @tool
        def Check_External_Issues(site_or_area: str) -> str:
            """
            Scans external monitors (like DownDetector or topology maps)
            for area-wide issues like fiber cuts or power outages.
            """
            return row.get("Check_External_Issues_answer", "NotApplicable")

        @tool
        def Validate_And_Apply_Config(element_id: str) -> str:
            """
            Retrieves the element's configuration, validates it against the
            standard, and pushes a corrected config if a mismatch is found.
            """
            return row.get("Validate_And_Apply_Config_answer", "NotApplicable")

        @tool
        def Check_KPI_Performance(kpi_metric_name: str) -> str:
            """
            Fetches a specific KPI from monitoring tools to check if its
            trends are in line with expectations.
            Example Metric: 'PRACH success rate'
            """
            return row.get("Check_KPI_Performance_answer", "NotApplicable")

        @tool
        def Create_Ticket(department_name: str, issue_details: str) -> str:
            """
            Logs a new issue in the ticketing system and routes it
            to the correct department.
            """
            return row.get("Create_Ticket_answer", "NotApplicable")

        @tool
        def Execute_Orchestration_Action(action_command: str) -> str:
            """
            Runs an automated O-RAN orchestration task using Kubernetes/Helm.
            Example Action: 'delete_pod_xyz', 'reassign_ip_address'
            """
            return row.get("Execute_Orchestration_Action_answer", "NotApplicable")

        @tool
        def Run_Triage_Diagnostics(issue_type: str) -> str:
            """
            Executes diagnostic scripts specifically for container or
            pod-related issues.
            Example Issue Type: 'pod-crash-loop', 'container-networking'
            """
            return row.get("Run_Triage_Diagnostics_answer", "NotApplicable")

        @tool
        def Check_Remote_Dump_Files(element_id: str) -> str:
            """
            Connects to a device via SSH/Telnet to review system dump
            files for identified errors or issues.
            """
            return row.get("Check_Remote_Dump_Files_answer", "NotApplicable")

        # --- Updated List of Tools ---
        tools = [
            Check_Alarm_Status,
            Check_Element_Neighbors,
            Check_Element_Health,
            Execute_Remote_Action,
            Check_External_Issues,
            Validate_And_Apply_Config,
            Check_KPI_Performance,
            Create_Ticket,
            Execute_Orchestration_Action,
            Run_Triage_Diagnostics,
            Check_Remote_Dump_Files,
        ]

        # --- End of Updates ---

        llm_with_tools = chat_llm.bind_tools(tools)
        checkpointer = MemorySaver()
        react_agent = create_react_agent(llm_with_tools, tools, checkpointer=checkpointer)

        config = {"configurable": {"thread_id": str(index)}}
        input_messages = [{"role": "user", "content": input_question}]

        # Run and append immediately; if anything fails, record the error so we can resume later
        try:
            response = react_agent.invoke({"messages": input_messages}, config)
            final_content = response["messages"][-1].content
            output_row = row.to_dict()
            output_row["row_index"] = int(index)
            output_row["agent_response"] = final_content
            append_jsonl(output_path, output_row)
        except Exception as e:
            err_row = {
                "row_index": int(index),
                "input": input_question,
                "error": repr(e),
            }
            append_jsonl(output_path, err_row)
            # Optional: continue to next row
            continue

    print(f"[DONE] Wrote/updated: {output_path}")


if __name__ == "__main__":
    main()

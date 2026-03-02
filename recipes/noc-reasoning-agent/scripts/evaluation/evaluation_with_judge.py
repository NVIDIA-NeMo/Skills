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

import pandas as pd
from bert_score import score as bert_score
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from rouge_score import rouge_scorer
from tqdm import tqdm

# Parse arguments for input JSONL path
parser = argparse.ArgumentParser(description="Evaluation Pipeline for Agent Responses")
parser.add_argument("input_jsonl", help="Path to agent_responses.jsonl containing expected_answer and agent_response")
parser.add_argument(
    "--output_file", default="evaluation_results.json", help="Path to output (default: evaluation_results.json)"
)
parser.add_argument(
    "--nim_url", default="http://localhost:8000/v1", help="Base URL for NIM API (default: http://localhost:8000/v1)"
)
parser.add_argument("--model", default="openai/gpt-oss-120b", help="NIM model name (default: gpt-oss-120b)")
args = parser.parse_args()

# Load the input JSONL
print(f"Loading input JSONL: {args.input_jsonl}")
df = pd.read_json(args.input_jsonl, lines=True)
print(f"Loaded {len(df)} rows")


# Set up ChatNVIDIA LLM
llm = ChatNVIDIA(base_url=args.nim_url, model=args.model)

# Initialize ROUGE scorer
rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


# Function for LLM-as-judge evaluation
def llm_judge_final_output(expected, generated):
    prompt = f"""
    Evaluate how well the generated resolution at the end matches the expected resolution on a scale of 1-5:
    - 5: Perfect match in content.
    - 4: High similarity, minor differences.
    - 3: Moderate match, key elements present but some deviations.
    - 2: Low match, major differences.
    - 1: No match.

    Expected: {expected}
    Generated: {generated}

    Provide only the score (1-5) and a brief reasoning.
    Format: Score: X\nReasoning: ...
    """
    # response = llm.invoke(prompt)
    # output = response.content.strip()
    # score = int(output.split("Score:")[1].split("\n")[0].strip())
    # reasoning = output.split("Reasoning:")[1].strip()
    # return score, reasoning
    try:
        response = llm.invoke(prompt)
        output = response.content.strip()
        score = int(output.split("Score:")[1].split("\n")[0].strip())
        reasoning = output.split("Reasoning:")[1].strip()
        return score, reasoning
    except Exception as e:
        print(f"Error in LLM judge: {e}")
        return 0, "Error"


def llm_judge_reasoning(expected, generated):
    prompt = f"""
    Evaluate how well the generated reasoning is, including tools used, resolution matches the expected resolution on a scale of 1-5:
    - 5: Perfect match in content, structure, and actions.
    - 4: High similarity, minor differences.
    - 3: Moderate match, key elements present but some deviations.
    - 2: Low match, major differences.
    - 1: No match.

    Expected: {expected}
    Generated: {generated}

    Provide only the score (1-5) and a brief reasoning.
    Format: Score: X\nReasoning: ...
    """
    # response = llm.invoke(prompt)
    # output = response.content.strip()
    # score = int(output.split("Score:")[1].split("\n")[0].strip())
    # reasoning = output.split("Reasoning:")[1].strip()
    # return score, reasoning
    try:
        response = llm.invoke(prompt)
        output = response.content.strip()
        score = int(output.split("Score:")[1].split("\n")[0].strip())
        reasoning = output.split("Reasoning:")[1].strip()
        return score, reasoning
    except Exception as e:
        print(f"Error in LLM judge: {e}")
        return 0, "Error"


# First pass: extract generated reasoning parts for batched BERTScore
candidates = []
references = []
row_data = []
for index, row in df.iterrows():
    conclusion_expected = row["expected_answer"]
    reasoning_expected = row["output"]
    generated = row["agent_response"]

    if "Thought 1:" in generated:
        if generated.count("Thought 1:") == 1:
            _, reasoning_tail = generated.split("Thought 1:", -1)
            generated_reasoning_part = "Thought 1:" + reasoning_tail
        elif generated.count("Thought 1:") >= 2:
            second_idx = generated.find("Thought 1:", generated.find("Thought 1:") + 1)
            generated_reasoning_part = generated[second_idx:].strip()
    else:
        generated_reasoning_part = generated

    candidates.append(generated_reasoning_part)
    references.append(conclusion_expected + reasoning_expected)
    row_data.append((index, row, conclusion_expected, reasoning_expected, generated_reasoning_part))

# Batched BERTScore computation (single model load)
print("Computing BERTScore (batched)...")
_, _, F1_all = bert_score(candidates, references, lang="en", verbose=True)

# Second pass: compute ROUGE, LLM judge, and assemble output
evaluations = []
for i, (index, row, conclusion_expected, reasoning_expected, generated_reasoning_part) in enumerate(
    tqdm(row_data, desc="Evaluating")
):
    rouge_scores = rouge.score(conclusion_expected + reasoning_expected, generated_reasoning_part)
    rouge1 = rouge_scores["rouge1"].fmeasure
    rougeL = rouge_scores["rougeL"].fmeasure
    bert_f1 = F1_all[i].item()

    reasoning_judge_score, reasoning_judge_reason = llm_judge_reasoning(reasoning_expected, generated_reasoning_part)
    conclusion_judge_score, conclusion_judge_reason = llm_judge_final_output(
        conclusion_expected, generated_reasoning_part
    )

    output_row = row.to_dict()
    output_row["rouge1"] = rouge1
    output_row["rougeL"] = rougeL
    output_row["bertscore_f1"] = bert_f1
    output_row["llm_reasoning_judge_score"] = reasoning_judge_score
    output_row["llm_reasoning_judge_reasoning"] = reasoning_judge_reason
    output_row["llm_conclusion_judge_score"] = conclusion_judge_score
    output_row["llm_conclusion_judge_reasoning"] = conclusion_judge_reason
    evaluations.append(output_row)

# Save to output JSONL
output_df = pd.DataFrame(evaluations)
output_df.to_json(args.output_file, orient="records", lines=True)
print("Evaluations saved to evaluations.jsonl")

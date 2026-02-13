import argparse

import pandas as pd
from bert_score import score as bert_score
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from rouge_score import rouge_scorer
from tqdm import tqdm

# Parse arguments for input JSONL path
parser = argparse.ArgumentParser(description="Evaluation Pipeline for Agent Responses")
parser.add_argument("input_jsonl", help="Path to agent_responses.jsonl containing expected_answer and agent_response")
parser.add_argument("--output_file", help="Path to output")
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


# Loop over rows and evaluate
evaluations = []
for index, row in tqdm(df.iterrows(), total=len(df)):
    conclusion_expected = row["expected_answer"]
    reasoning_expected = row["output"]
    generated = row["agent_response"]

    if "Thought 1:" in generated:
        if generated.count("Thought 1:") == 1:
            question_part, reasoning_tail = generated.split("Thought 1:", -1)
            question_part = question_part.strip()
            generated_reasoning_part = "Thought 1:" + reasoning_tail
        elif generated.count("Thought 1:") >= 2:
            # Find where the 2nd "Thought 1:" starts
            second_idx = generated.find("Thought 1:", generated.find("Thought 1:") + 1)
            # Split into question and reasoning at the 2nd occurrence
            question_part = generated[:second_idx].strip()
            generated_reasoning_part = generated[second_idx:].strip()
    else:
        generated_reasoning_part = generated

    # Compute ROUGE
    rouge_scores = rouge.score(conclusion_expected + reasoning_expected, generated_reasoning_part)
    rouge1 = rouge_scores["rouge1"].fmeasure
    rougeL = rouge_scores["rougeL"].fmeasure

    # Compute BERTScore (requires torch)
    P, R, F1 = bert_score(
        [generated_reasoning_part], [conclusion_expected + reasoning_expected], lang="en", verbose=False
    )
    bert_f1 = F1.mean().item()

    # LLM Judge
    reasoning_judge_score, reasoning_judge_reason = llm_judge_reasoning(reasoning_expected, generated_reasoning_part)
    conclusion_judge_score, conclusion_judge_reason = llm_judge_final_output(
        conclusion_expected, generated_reasoning_part
    )

    # Add to output row
    output_row = row.to_dict()
    output_row["rouge1"] = rouge1
    output_row["rougeL"] = rougeL
    output_row["bertscore_f1"] = bert_f1
    output_row["llm_reasoning_judge_score"] = reasoning_judge_score
    output_row["llm_reasoning_judge_reasoning"] = reasoning_judge_reason
    output_row["llm_conclusion_judge_score"] = conclusion_judge_score
    output_row["llm_conclusion_judge_reasoning"] = conclusion_judge_reason
    if index == 1:
        print(output_row)
    evaluations.append(output_row)

# Save to output JSONL
output_df = pd.DataFrame(evaluations)
output_df.to_json(args.output_file, orient="records", lines=True)
print("Evaluations saved to evaluations.jsonl")

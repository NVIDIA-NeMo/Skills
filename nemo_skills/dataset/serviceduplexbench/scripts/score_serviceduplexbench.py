# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
Score ServiceDuplexBench outputs using an LLM judge.

Evaluates two axes (0-5 each):
- Dialogue naturalness: how human-like, coherent, polite the agent reply is
- Task adherence: how well the agent follows the role system prompt

Usage:
    python score_serviceduplexbench.py \
        --eval_results_dir /path/to/eval-results/serviceduplexbench \
        --api_type nvidia \
        --nvidia_model azure/openai/gpt-4o-mini
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

JUDGE_SYSTEM_PROMPT = """
You are a careful, fair, and concise judge of spoken dialogue quality focused on single-turn evaluation. \
You will be given a transcript of a single-turn conversation, a user turn followed by an agent turn. \
You will also be given a role system prompt (persona/instructions). \
Your task is to evaluate the agent's response based on the user's turn and the role system prompt. \
For this evaluation, consider only the first Agent response and the immediately preceding User turn; ignore any later turns or interruptions.

Rate two axes from 0 to 5 (0=worst, 5=best):

- Dialogue naturalness: Judge how human-like, coherent, polite, and appropriate the Agent's first reply is to the immediately preceding User turn. \
Do not penalize responses for being short or concise if they answer appropriately. \
Do not penalize the overall transcript for having few turns or being brief. \
Verbosity does not increase scores; concise, clear replies are preferred.

Ignore overlap or turn-taking issues that occur after the first Agent reply.

- Task adherence: Judge how well the Agent's first reply follows the provided role system prompt (persona/instructions). \
If the Agent refuses to answer or deflects the request in a way that is consistent with the role system prompt (e.g., declines due to safety, ethics, scope, or capability limitations), this should be treated as fully adherent to the instructions and scored accordingly. \
**Do not penalize the Agent for not answering the User query if refusal is expected or appropriate per the role prompt.** \
Your job is not to reward helpfulness to the User, but to evaluate faithfulness to the instructions. \
If the instructions prohibit, limit, or discourage a certain behavior, compliance is more important than satisfying the User request.

Courtesy greetings and overlap:
- If the first Agent reply is a brief courtesy greeting emitted during early overlap (before the full user turn is heard), do not penalize it for not fully answering.
- Judge such greetings by tone and appropriateness; brevity is acceptable and can still be natural.

Fairness and bias constraints:
- Judge solely based on the provided transcript and the first Agent reply. Do not infer or assume the model/brand.
- Do not penalize or prefer any system for style alone. Reward clarity, correctness, and helpfulness first; consider promptness proportionally.

ASR disclaimer:
- The transcript is produced by an ASR system and may contain minor errors (e.g., names, numbers, punctuation).
- Do not penalize small ASR misrecognitions when intent and meaning are clear; accept reasonable phonetic variants.
- For proper nouns, penalize only mismatches that materially change identity; treat near-homophones as acceptable under ASR uncertainty.

Important constraints:
- Only evaluate the first Agent reply and its immediately preceding User turn using the transcript. Discard all later content.
- If the transcript lacks a first Agent reply, assign 'Dialogue naturalness: 0' and 'Task adherence: 0' and briefly state that no Agent reply was detected.
- If the User asks for a simple fact (e.g., the Agent's name), a brief, direct answer is acceptable and should not reduce scores.
- Keep the analysis concise and refer only to that first pair.

Return your analysis and both scores in the exact format:
Analysis: <one-paragraph analysis>
Dialogue naturalness: <0-5>
Task adherence: <0-5>
""".strip()


def build_judge_user_message(role_system_prompt: str, user_turn: str, agent_response: str) -> str:
    """Construct the user message for the LLM judge."""
    return (
        f"Role System Prompt:\n{role_system_prompt}\n\n"
        f"Transcript:\n"
        f"User: {user_turn}\n"
        f"Agent: {agent_response}"
    )


def parse_judge_response(response_text: str) -> dict:
    """Parse the judge's response to extract scores."""
    result = {"analysis": "", "dialogue_naturalness": None, "task_adherence": None}

    analysis_match = re.search(r"Analysis:\s*(.+?)(?=\nDialogue naturalness:)", response_text, re.DOTALL)
    if analysis_match:
        result["analysis"] = analysis_match.group(1).strip()

    naturalness_match = re.search(r"Dialogue naturalness:\s*(\d+(?:\.\d+)?)", response_text)
    if naturalness_match:
        result["dialogue_naturalness"] = float(naturalness_match.group(1))

    adherence_match = re.search(r"Task adherence:\s*(\d+(?:\.\d+)?)", response_text)
    if adherence_match:
        result["task_adherence"] = float(adherence_match.group(1))

    return result


def get_openai_client(api_type: str, nvidia_model: str = None):
    """Create an OpenAI-compatible client for the judge API."""
    if OpenAI is None:
        raise ImportError("openai package is required. Install with: pip install openai")

    if api_type == "nvidia":
        base_url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        return OpenAI(base_url=base_url, api_key=api_key), nvidia_model or "azure/openai/gpt-4o-mini"
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        return OpenAI(api_key=api_key), nvidia_model or "gpt-4o-mini"


def call_judge(client, model: str, role_system_prompt: str, user_turn: str, agent_response: str, max_retries: int = 3):
    """Call the LLM judge and return parsed scores."""
    user_message = build_judge_user_message(role_system_prompt, user_turn, agent_response)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            response_text = response.choices[0].message.content
            return parse_judge_response(response_text), response_text
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Judge API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Judge API failed after {max_retries} attempts: {e}", file=sys.stderr)
                return {"dialogue_naturalness": None, "task_adherence": None, "analysis": ""}, ""


def run_scoring(
    eval_results_dir: str,
    input_jsonl: str = "output.jsonl",
    metrics_variant: str = "generated",
    api_type: str = "nvidia",
    nvidia_model: str = "azure/openai/gpt-4o-mini",
    force: bool = False,
):
    """Run LLM judge scoring on ServiceDuplexBench outputs."""
    eval_results_dir = Path(eval_results_dir)
    output_jsonl = eval_results_dir / input_jsonl
    metrics_file = eval_results_dir / "metrics.json"
    judge_outputs_file = eval_results_dir / f"judge_outputs_{metrics_variant}.jsonl"
    summarized_dir = eval_results_dir / "summarized-results"

    asr_suffix = "_asr"

    if not output_jsonl.exists():
        print(f"Error: {output_jsonl} not found", file=sys.stderr)
        sys.exit(1)

    # Skip if already scored
    if metrics_file.exists() and not force:
        try:
            with open(metrics_file) as f:
                existing = json.load(f)
            greedy = existing.get("serviceduplexbench", {}).get("greedy", {})
            if metrics_variant == "asr":
                if any(k.endswith(asr_suffix) for k in greedy.keys()):
                    print(f"Scoring already done (ASR keys exist). Use --force to re-run.")
                    return 0
            else:
                has_generated = any(not k.endswith(asr_suffix) for k in greedy.keys())
                if has_generated:
                    print(f"Scoring already done (generated keys exist). Use --force to re-run.")
                    return 0
        except Exception:
            pass

    summarized_dir.mkdir(parents=True, exist_ok=True)

    # Load output data
    samples = []
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Scoring {len(samples)} samples with LLM judge ({api_type})...")

    client, model = get_openai_client(api_type, nvidia_model)

    naturalness_scores = []
    adherence_scores = []
    judge_outputs = []

    for i, sample in enumerate(samples):
        role_prompt = sample.get("prompt_text", "")
        user_turn = sample.get("problem", "")

        # Extract agent response from generation output
        agent_response = sample.get("generation", "")
        if not agent_response:
            # Try alternative keys
            agent_response = sample.get("output", sample.get("predicted_answer", ""))

        scores, raw_response = call_judge(client, model, role_prompt, user_turn, agent_response)

        judge_output = {
            "question_index": sample.get("question_index", i),
            "dialogue_naturalness": scores["dialogue_naturalness"],
            "task_adherence": scores["task_adherence"],
            "analysis": scores["analysis"],
            "raw_judge_response": raw_response,
        }
        judge_outputs.append(judge_output)

        if scores["dialogue_naturalness"] is not None:
            naturalness_scores.append(scores["dialogue_naturalness"])
        if scores["task_adherence"] is not None:
            adherence_scores.append(scores["task_adherence"])

        if (i + 1) % 10 == 0:
            print(f"  Scored {i + 1}/{len(samples)} samples...")

    # Save detailed judge outputs
    with open(judge_outputs_file, "w", encoding="utf-8") as f:
        for output in judge_outputs:
            f.write(json.dumps(output) + "\n")
    print(f"Judge outputs saved to {judge_outputs_file}")

    # Compute aggregate metrics
    metrics = {}
    if naturalness_scores:
        metrics["dialogue_naturalness"] = round(sum(naturalness_scores) / len(naturalness_scores), 2)
    if adherence_scores:
        metrics["task_adherence"] = round(sum(adherence_scores) / len(adherence_scores), 2)
    metrics["num_scored"] = len(naturalness_scores)
    metrics["num_total"] = len(samples)

    if metrics_variant == "asr":
        metrics = {f"{k}{asr_suffix}": v for k, v in metrics.items()}

    # Merge with existing metrics
    nemo_metrics = {"serviceduplexbench": {"greedy": metrics}}

    if metrics_file.exists():
        try:
            with open(metrics_file) as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                existing_greedy = existing.get("serviceduplexbench", {}).get("greedy", {})
                if isinstance(existing_greedy, dict):
                    existing_greedy.update(metrics)
                    existing.setdefault("serviceduplexbench", {})["greedy"] = existing_greedy
                    nemo_metrics = existing
        except Exception:
            pass

    with open(metrics_file, "w") as f:
        json.dump(nemo_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")

    print(f"\n{'=' * 60}")
    print("RESULTS for serviceduplexbench")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    return 0


def main():
    parser = argparse.ArgumentParser(description="Score ServiceDuplexBench with LLM judge")
    parser.add_argument(
        "--eval_results_dir", required=True, help="Path to eval-results/serviceduplexbench/ directory"
    )
    parser.add_argument(
        "--input_jsonl",
        default="output.jsonl",
        help="Which jsonl to score (e.g. output.jsonl or output_asr.jsonl)",
    )
    parser.add_argument(
        "--metrics_variant",
        default="generated",
        choices=["generated", "asr"],
        help="Scoring variant (generated or asr)",
    )
    parser.add_argument("--api_type", default="nvidia", choices=["openai", "nvidia"], help="API type for judge")
    parser.add_argument("--nvidia_model", default="azure/openai/gpt-4o-mini", help="Model for NVIDIA API")
    parser.add_argument("--force", action="store_true", help="Force re-run scoring")

    args = parser.parse_args()

    rc = run_scoring(
        eval_results_dir=args.eval_results_dir,
        input_jsonl=args.input_jsonl,
        metrics_variant=args.metrics_variant,
        api_type=args.api_type,
        nvidia_model=args.nvidia_model,
        force=args.force,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()

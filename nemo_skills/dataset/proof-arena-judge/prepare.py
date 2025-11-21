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
from pathlib import Path

import datasets
import requests
from transformers import AutoTokenizer

JUDGEMENT_YES = "Judgement: Yes"
JUDGEMENT_NO = "Judgement: No"

SUBSET_PROOFS = "proofs"
MAX_QWEN_TOKENS = 10000

qwen3_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def prepare_data(output_path):
    # Load data
    imo_data = datasets.load_dataset("MathArena/imo_2025_outputs")["train"]
    usamo_data = datasets.load_dataset("MathArena/usamo_2025_outputs")["train"]
    imc_data = datasets.load_dataset("MathArena/imc_2025_outputs")["train"]
    openai_imo_data = load_openai_imo_proofs()
    gemini_imo_data = load_gemini_imo_proofs()
    processed_data = []
    processed_data.extend(openai_imo_data)
    processed_data.extend(gemini_imo_data)

    # Process usamo data
    processed_data.extend(process_imo_usamo_data(usamo_data, "usamo"))
    processed_data.extend(process_imo_usamo_data(imo_data, "imo"))
    processed_data.extend(process_imc_data(imc_data))

    # Shuffle the data
    for item in processed_data:
        item["problem"] = item["problem"].strip()
        item["proof"] = item["proof"].strip()
    random.seed(42)
    random.shuffle(processed_data)

    # Filter out too long proofs
    filtered_processed_data = [
        item for item in processed_data if len(qwen3_tokenizer.encode(item["proof"])) <= MAX_QWEN_TOKENS
    ]
    print(f"Filtered out {len(processed_data) - len(filtered_processed_data)} proofs due to length.")

    # Save the combined data
    full_output_path = Path(__file__).parent / output_path
    with open(full_output_path, "w") as f:
        for item in filtered_processed_data:
            f.write(json.dumps(item) + "\n")

    # Print statistics
    print(f"Statistics for {output_path}:")
    print(f"- {len([item for item in filtered_processed_data if item['subset_for_metrics'] == SUBSET_PROOFS])} proofs")
    print(
        f"Correct Proofs: {len([item for item in filtered_processed_data if item['expected_judgement'] == JUDGEMENT_YES and item['subset_for_metrics'] == SUBSET_PROOFS])}"
    )
    print(
        f"Incorrect Proofs: {len([item for item in filtered_processed_data if item['expected_judgement'] == JUDGEMENT_NO and item['subset_for_metrics'] == SUBSET_PROOFS])}"
    )
    print("-" * 20)


def load_jsonl(file_path):
    data = []
    full_path = Path(__file__).parent / file_path
    with open(full_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def grading_scheme_to_rubric(grading_scheme, desc_key="grading_scheme_desc"):
    if desc_key != "grading_scheme_desc":
        assert "grading_scheme_desc" not in grading_scheme[0], (
            "Something is off, make sure the grading scheme is loaded correctly."
        )
    return "\n".join([f"- {scheme[desc_key]}" for scheme in grading_scheme])


def load_openai_imo_proofs():
    # Download the proofs
    # https://github.com/aw31/openai-imo-2025-proofs/blob/main/problem_{1..5}.txt
    # Download the problems from MathArena/imo_2025
    imo_data = datasets.load_dataset("MathArena/imo_2025")["train"]
    # Load the proofs
    data = []
    for i in range(1, 6):
        url = f"https://raw.githubusercontent.com/aw31/openai-imo-2025-proofs/main/problem_{i}.txt"
        response = requests.get(url)
        assert int(imo_data[i - 1]["problem_idx"]) == i, (
            f"Problem index mismatch: {imo_data[i - 1]['problem_idx']} != {i}"
        )
        data.append(
            {
                "problem": imo_data[i - 1]["problem"],
                "proof": response.text,
                "rubric": grading_scheme_to_rubric(imo_data[i - 1]["grading_scheme"], desc_key="desc"),
                "expected_judgement": JUDGEMENT_YES,
                "subset_for_metrics": SUBSET_PROOFS,
                "metadata": {
                    "source": "openai_imo",
                    "problem_idx": i,
                    "model_id": "openai_imo",
                },
            }
        )

    print(f"Loaded {len(data)} proofs from OpenAI-IMO")
    return data


def load_gemini_imo_proofs():
    # Download the proofs
    # https://github.com/aw31/openai-imo-2025-proofs/blob/main/problem_{1..5}.txt
    # Download the problems from MathArena/imo_2025
    imo_data = datasets.load_dataset("MathArena/imo_2025")["train"]
    # Load the proofs
    data = []
    for i in range(1, 6):
        file = Path(__file__).parent / "gemini_imo_2025" / f"{i}.txt"
        with open(file, "r") as f:
            text = f.read()
            assert int(imo_data[i - 1]["problem_idx"]) == i, (
                f"Problem index mismatch: {imo_data[i - 1]['problem_idx']} != {i}"
            )
            data.append(
                {
                    "problem": imo_data[i - 1]["problem"],
                    "proof": text,
                    "rubric": grading_scheme_to_rubric(imo_data[i - 1]["grading_scheme"], desc_key="desc"),
                    "expected_judgement": JUDGEMENT_YES,
                    "subset_for_metrics": SUBSET_PROOFS,
                    "metadata": {
                        "source": "gemini_imo",
                        "problem_idx": i,
                        "model_id": "gemini_imo",
                    },
                }
            )
    print(f"Loaded {len(data)} proofs from Gemini-IMO")
    return data


def process_imo_usamo_data(raw_data, source):
    data = []
    for item in raw_data:
        points1, points2 = item["points_judge_1"], item["points_judge_2"]
        if points2 is None:
            points2 = points1
        if points1 < 6 and points2 < 6:
            judgement = JUDGEMENT_NO
        elif points1 == 7 and points2 == 7:
            judgement = JUDGEMENT_YES
        else:
            # Gray area, ignore
            continue
        data.append(
            {
                "problem": item["problem"],
                "proof": item["answer"].split("</think>")[-1].strip(),
                "expected_judgement": judgement,
                "rubric": grading_scheme_to_rubric(item["grading_details_judge_1"], desc_key="grading_scheme_desc"),
                "subset_for_metrics": SUBSET_PROOFS,
                "metadata": {
                    "source": source,
                    "problem_idx": item["problem_idx"],
                    "grading_details_judge_1": item["grading_details_judge_1"],
                    "grading_details_judge_2": item["grading_details_judge_2"],
                    "model_id": item["model_name"],
                },
            }
        )
    print(f"Loaded {len(data)} proofs from {source}")
    return data


def process_imc_data(raw_data):
    data = []
    for item in raw_data:
        points1 = item["points_judge_1"]
        if points1 < 8:
            judgement = JUDGEMENT_NO
        elif points1 == 10:
            judgement = JUDGEMENT_YES
        else:
            # Gray area, ignore
            continue
        data.append(
            {
                "problem": item["problem"],
                "proof": item["answer"].split("</think>")[-1].strip(),
                "expected_judgement": judgement,
                "subset_for_metrics": SUBSET_PROOFS,
                "rubric": grading_scheme_to_rubric(item["grading_details_judge_1"], desc_key="grading_scheme_desc"),
                "metadata": {
                    "source": "imc",
                    "problem_idx": item["problem_idx"],
                    "grading_details_judge_1": item["grading_details_judge_1"],
                    "model_id": item["model_name"],
                },
            }
        )
    print(f"Loaded {len(data)} proofs from IMC")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="test.jsonl", help="Path to save combined dataset")
    args = parser.parse_args()

    prepare_data(args.output_path)

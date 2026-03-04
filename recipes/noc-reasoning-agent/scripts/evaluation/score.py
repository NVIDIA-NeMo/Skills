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


def main():
    parser = argparse.ArgumentParser(description="Compute overall evaluation scores from JSONL")
    parser.add_argument("input_jsonl", help="Path to JSONL file containing rouge, bertscore, and judge scores")
    args = parser.parse_args()

    # Load the JSONL into a DataFrame
    df = pd.read_json(args.input_jsonl, lines=True)

    # Compute averages for the numeric columns
    metrics = ["rouge1", "rougeL", "bertscore_f1", "llm_reasoning_judge_score", "llm_conclusion_judge_score"]
    averages = df[metrics].mean()

    print("Overall Scores:")
    for metric, value in averages.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()

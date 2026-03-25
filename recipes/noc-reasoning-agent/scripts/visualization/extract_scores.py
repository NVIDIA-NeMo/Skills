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

import json

import pandas as pd

# Replace with your actual JSONL file path
file_path = "evaluations.jsonl"

# Load JSONL file
data = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Metrics to calculate averages for
metrics = ["rouge1", "rougeL", "bertscore_f1", "llm_judge_score"]

# Check which columns are available
available_metrics = [metric for metric in metrics if metric in df.columns]

if not available_metrics:
    raise ValueError("No required metrics found in the JSONL file!")

# Calculate averages
averages = df[available_metrics].mean()

# Display results
print("Average Metrics:")
for metric, avg in averages.items():
    print(f"{metric}: {avg:.4f}")


# Select only the relevant columns
columns_to_display = ["expected_answer", "agent_response", "llm_judge_reason"]
df_subset = df[columns_to_display].head(10)

# Display neatly
for idx, row in df_subset.iterrows():
    print(f"\n--- Sample {idx + 1} ---")
    print(f"True Answer (expected_answer): {row['expected_answer']}")
    print(f"Model Answer (agent_response): {row['agent_response']}")
    print(f"Judge Explanation (llm_judge_reason): {row['llm_judge_reason']}")

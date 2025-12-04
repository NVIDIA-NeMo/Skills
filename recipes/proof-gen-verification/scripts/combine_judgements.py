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
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--n_seeds", type=int, required=True)
    args = parser.parse_args()

    all_files = []
    for file in sorted(os.listdir(args.input_dir)):
        if file.endswith(".jsonl"):
            with open(os.path.join(args.input_dir, file), "r") as f:
                file_content = [json.loads(line) for line in f]
            all_files.append(file_content)

    assert all(len(all_files[0]) == len(file) for file in all_files)

    final_output = []
    keys_to_keep = ["problem", "proof", "expected_judgement", "subset_for_metrics", "metadata"]
    assert len(all_files) == args.n_seeds, "Number of seeds does not match number of files"
    for i in range(len(all_files[0])):
        # combine "judgement" into "judgement_candidates" and assert all other fields are equal
        final_output.append(
            {
                "judgement_candidates": [file[i]["judgement"] for file in all_files],
                "num_generated_tokens_list": [file[i]["num_generated_tokens"] for file in all_files],
                **{k: all_files[0][i][k] for k in all_files[0][i] if k in keys_to_keep},
            }
        )
        for j in range(len(all_files)):
            assert all(all_files[0][i][k] == all_files[j][i][k] for k in all_files[j][i] if k in keys_to_keep)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        for item in final_output:
            f.write(json.dumps(item) + "\n")

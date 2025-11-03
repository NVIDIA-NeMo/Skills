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
import functools
import json
import multiprocessing as mp

from tqdm import tqdm


def process_data(elem, target_model):
    # input:
    # ['output', 'category', 'license', 'reasoning', 'generator', 'used_in_training', 'version', 'system_prompt', 'messages',
    # 'generation_start_time', 'num_generated_tokens', 'finish_reason', 'generation', 'generation_end_time', 'generation_time']

    # output:
    # ['input', 'generator', 'split', 'reasoning', 'system_prompt', 'output_DeepSeek-R1', 'output_llama-3.1-nemotron-ultra-253b-v1']
    data = {}
    data["input"] = elem["messages"]
    data["generator"] = elem["generator"]
    data["split"] = elem["split"]
    data["reasoning"] = elem["reasoning"]
    data["system_prompt"] = elem["system_prompt"]
    model_1 = f"output_{data['generator']}"
    data[model_1] = elem["output"]
    model_2 = f"output_{target_model}"
    data[model_2] = elem["generation"]
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare data")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="The path that contains jsonl with responses from multiple models",
    )

    parser.add_argument(
        "--target_model", type=str, required=True, help="The model that was used to generate alternative answers"
    )

    parser.add_argument("--output_path", type=str, required=True, help="result json to feed to comparison script")

    args = parser.parse_args()

    data = []
    with open(f"{args.input_path}/output.jsonl", "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    num_workers = max(1, mp.cpu_count() - 1)
    print(f"Using {num_workers} workers for parallel processing")
    partial_func = functools.partial(process_data, target_model=args.target_model)

    # Create a pool of workers
    with mp.Pool(num_workers) as pool:
        # Process the dataset in parallel with a progress bar
        processed_data = list(
            tqdm(
                pool.imap(partial_func, data),
                total=len(data),
                desc="Processing data",
            )
        )

    print(f"Processed {len(processed_data)} elements")

    print(f"Dump {len(processed_data)} examples to {args.output_path}")
    with open(args.output_path, "w") as fout:
        json.dump(processed_data, fout)

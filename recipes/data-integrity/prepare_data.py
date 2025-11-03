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

from datasets import IterableDataset, load_dataset
from tqdm import tqdm


def process_data(elem, split):
    # rename "input" to "messages" and use ++prompt_format=openai and remove ++prompt_config to work with nemo-skills.
    elem["messages"] = elem.pop("input")
    system_message = {"role": "system", "content": elem["system_prompt"]}
    elem["messages"].append(system_message)
    elem["split"] = split
    return elem


def get_from_iterable(dataset: IterableDataset):
    examples = []
    for example in dataset:
        examples.append(example)
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare data")
    parser.add_argument(
        "--split", type=str, default="science", help="one of the following: 'safety','chat','science','code', 'math'"
    )

    parser.add_argument("--nrows", type=int, default=100, help="number of examples to download")

    parser.add_argument(
        "--generator", type=str, default="DeepSeek-R1", help="The model whose respose you want to compare"
    )

    parser.add_argument("--output", type=str, required=True, help="The download result")

    args = parser.parse_args()

    dataset_name = "nvidia/Llama-Nemotron-Post-Training-Dataset"

    dataset = load_dataset(dataset_name, "SFT", split=args.split, streaming=True)
    dataset = dataset.filter(lambda x: x["generator"] == args.generator)
    first_nrows = dataset.take(args.nrows)
    first_nrows = get_from_iterable(first_nrows)

    num_workers = max(1, mp.cpu_count() - 1)
    print(f"Using {num_workers} workers for parallel processing")

    partial_func = functools.partial(process_data, split=args.split)
    # Create a pool of workers
    with mp.Pool(num_workers) as pool:
        # Process the dataset in parallel with a progress bar
        processed_data = list(
            tqdm(
                pool.imap(partial_func, first_nrows),
                total=len(first_nrows),
                desc="Processing data",
            )
        )

    print(f"Processed {len(processed_data)} elements")

    output_fname = args.output
    print(f"Dump {len(processed_data)} examples to {output_fname}")
    with open(output_fname, "w") as fout:
        for d in processed_data:
            fout.write(json.dumps(d) + "\n")

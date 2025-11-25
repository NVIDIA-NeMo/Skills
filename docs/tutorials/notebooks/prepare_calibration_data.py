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

import gc
import os
from itertools import islice

from datasets import Dataset, disable_caching, load_dataset

from nemo_skills.prompt.utils import get_prompt

disable_caching()
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define paths and parameters
LOCAL_DATASET_PATH = "./calibration_dataset"
CALIB_DATASET_NAME = "nvidia/OpenMathReasoning"
CALIB_SPLIT = "tir"
CALIB_SIZE = 4096

# Load samples, format them, and save as a Parquet file
print(f"Loading and formatting {CALIB_SIZE} samples for calibration...")
ds_samples = load_dataset(CALIB_DATASET_NAME, split=CALIB_SPLIT, streaming=True)

prompt_template = get_prompt("generic/math", tokenizer="nvidia/OpenMath-Nemotron-14B-kaggle")

all_texts = []
for sample in islice(ds_samples, CALIB_SIZE):
    formatted_text = prompt_template.fill(
        {k: v for k, v in sample.items() if k in ["problem", "generated_solution"]},
    )
    all_texts.append(formatted_text)

calibration_dataset = Dataset.from_dict({"text": all_texts})
os.makedirs(LOCAL_DATASET_PATH, exist_ok=True)
calibration_dataset.to_parquet(f"{LOCAL_DATASET_PATH}/data.parquet")

# Free memory before exit
del all_texts, calibration_dataset, prompt_template, ds_samples
gc.collect()
print(f"Calibration dataset saved to {LOCAL_DATASET_PATH}/data.parquet", flush=True)
os._exit(0)

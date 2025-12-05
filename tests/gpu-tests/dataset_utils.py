# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path

# Datasets excluded from test_prepare_and_eval_all_datasets
# These don't support max_samples, require explicit parameters, or are very heavy to prepare
EXCLUDED_DATASETS = {
    "__pycache__",
    "ruler",
    "bigcodebench",
    "livecodebench",
    "livebench_coding",
    "livecodebench-pro",
    "livecodebench-cpp",
    "ioi24",
    "ioi25",
    "bfcl_v3",
    "bfcl_v4",
    "swe-bench",
    "aai",
    "human-eval",
    "human-eval-infilling",
    "mbpp",
    "mmau-pro",
    "aalcr",  # Has tokenization mismatch issues
}


def get_preparable_datasets():
    """Get list of datasets that can be prepared for testing (used by CI workflow)."""
    datasets_dir = Path(__file__).absolute().parents[2] / "nemo_skills" / "dataset"
    return sorted(
        dataset.name
        for dataset in datasets_dir.iterdir()
        if dataset.is_dir() and (dataset / "prepare.py").exists() and dataset.name not in EXCLUDED_DATASETS
    )


if __name__ == "__main__":
    # Allow running directly to print datasets (for CI workflow)
    print(" ".join(get_preparable_datasets()))

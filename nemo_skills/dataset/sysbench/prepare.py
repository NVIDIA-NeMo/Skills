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

import json
import logging
import urllib.request
from pathlib import Path

from nemo_skills.utils import get_logger_name, setup_logging

URL = "https://raw.githubusercontent.com/PKU-Baichuan-MLSystemLab/SysBench/main/datas/system_benchmark_eval_datas.json"


def run_preparation():
    """Prepare SysBench dataset for multi-turn evaluation.
    
    Each entry in the output represents one complete dialogue (system_id).
    The custom evaluator (sysbench.py) will handle:
    1. Generating assistant responses turn-by-turn
    2. Building history blocks for judging
    3. Formatting criteria for each turn
    """
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    source_file = data_dir / "system_benchmark_eval_datas.json"
    if not source_file.exists():
        print(f"Downloading SysBench dataset manifest from {URL}")
        urllib.request.urlretrieve(URL, source_file)
    output_file = data_dir / "test.jsonl"

    with open(source_file, "rt", encoding="utf-8") as fin:
        raw_entries = json.load(fin)

    print(f"Preparing {len(raw_entries)} SysBench dialogues")

    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in raw_entries:
            # Keep the full dialogue structure
            # The evaluator will generate each assistant turn and judge it
            prepared_entry = {
                "system_id": entry.get("system_id"),
                "domain": entry.get("领域"),
                "scene": entry.get("场景"),
                "system_prompt": entry.get("system_prompt"),
                "messages": entry.get("messages", []),  # Full conversation template
                "prompt_infos": entry.get("prompt_infos", {}),  # Criteria for each user prompt
                "rounds_related": entry.get("rounds_related"),
                "split": "test",
            }
            fout.write(json.dumps(prepared_entry, ensure_ascii=False) + "\n")
    print(f"Saved prepared SysBench data to {output_file}")


if __name__ == "__main__":
    setup_logging()
    run_preparation()

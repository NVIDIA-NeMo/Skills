# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from pathlib import Path

if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    with open(data_dir / "test.jsonl", "wt", encoding="utf-8") as fout:
        fout.write(
            json.dumps(
                {
                    "problem": "What is 2 + 2?",
                    "expected_answer": 4,
                }
            )
            + "\n"
        )

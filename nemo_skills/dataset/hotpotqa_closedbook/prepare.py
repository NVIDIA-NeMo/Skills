# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Closed-book variant uses the same validation data as hotpotqa (distractor setting).
# We reuse that file so there is only one real data preparation (in hotpotqa).

import shutil
import sys
from pathlib import Path

# Reuse the shared preparation so we don't require hotpotqa to be prepared first.
from nemo_skills.dataset.hotpotqa.prepare_utils import prepare_validation

if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / "validation.jsonl"

    hotpotqa_source = data_dir.parent / "hotpotqa" / "validation.jsonl"

    if hotpotqa_source.exists():
        shutil.copy2(hotpotqa_source, output_file)
        print(f"Copied {hotpotqa_source} -> {output_file}")
    else:
        # Same data; run shared preparation for hotpotqa then copy here.
        prepare_validation(hotpotqa_source)
        if not hotpotqa_source.exists():
            print("Preparation did not create the expected file.", file=sys.stderr)
            sys.exit(1)
        shutil.copy2(hotpotqa_source, output_file)
        print(f"Copied {hotpotqa_source} -> {output_file}")

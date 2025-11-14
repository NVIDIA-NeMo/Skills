# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Prepare Earnings-21 dataset for AudioBench evaluation."""

import sys
from pathlib import Path

# Import main prepare module
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from prepare import main as prepare_main

if __name__ == "__main__":
    # Override sys.argv to prepare only earnings datasets
    sys.argv = [
        sys.argv[0],
        "--datasets",
        "earnings21",
        "earnings22",
    ] + sys.argv[1:]  # Keep any user-provided args
    
    prepare_main()


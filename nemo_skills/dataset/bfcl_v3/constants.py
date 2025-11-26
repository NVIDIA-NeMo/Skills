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

# As we use BFCLv4 dataset for data processing
VERSION_PREFIX = "BFCL_v4"

TEST_COLLECTION_MAPPING = {
    "all": [
        "simple_python",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "simple_java",
        "simple_javascript",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
        "multi_turn_long_context",
    ],
    "multi_turn": [
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
        "multi_turn_long_context",
    ],
    "single_turn": [
        "simple_python",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "simple_java",
        "simple_javascript",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
    "live": [
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
    "non_live": [
        "simple_python",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "simple_java",
        "simple_javascript",
    ],
    "ast": [
        "simple_python",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "simple_java",
        "simple_javascript",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
    "non_python": [
        "simple_java",
        "simple_javascript",
    ],
    "python": [
        "simple_python",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
}

ALL_SCORING_CATEGORIES = TEST_COLLECTION_MAPPING["all"]

# Repo relative paths

DATA_FOLDER_PATH = Path("berkeley-function-call-leaderboard/bfcl_eval/data")

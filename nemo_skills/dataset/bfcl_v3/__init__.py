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
SPLITS = [
    "simple_python",  # Simple function calls
    "parallel",  # Multiple function calls in parallel
    "multiple",  # Multiple function calls in sequence
    "parallel_multiple",  # Multiple function calls in parallel and in sequence
    "simple_java",  # Java function calls
    "simple_javascript",  # JavaScript function calls
    "irrelevance",  # Function calls with irrelevant function documentation
    "live_simple",  # User-contributed simple function calls
    "live_multiple",  # User-contributed multiple function calls in sequence
    "live_parallel",  # User-contributed multiple function calls in parallel
    "live_parallel_multiple",  # User-contributed multiple function calls in parallel and in sequence
    "live_irrelevance",  # User-contributed function calls with irrelevant function documentation
    "live_relevance",  # User-contributed function calls with relevant function documentation
    "multi_turn_base",  # Base entries for multi-turn function calls
    "multi_turn_miss_func",  # Multi-turn function calls with missing function
    "multi_turn_miss_param",  # Multi-turn function calls with missing parameter
    "multi_turn_long_context",  # Multi-turn function calls with long context
]

IS_BENCHMARK_GROUP = True

SCORE_MODULE = "nemo_skills.dataset.bfcl_v3.bfcl_score"

BENCHMARKS = {f"bfcl_v3.{split}": {} for split in SPLITS}

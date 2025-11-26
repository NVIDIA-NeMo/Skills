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


SIMPLE_AST = [
    "simple_python",
    "simple_java",
    "simple_javascript",
]

OTHER_SINGLE_TURN_AST = [
    "parallel",
    "multiple",
    "parallel_multiple",
]

SINGLE_TURN_IRRELEVANCE = "irrelevance"

LIVE_SINGLE_TURN_AST = [
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
]

LIVE_SINGLE_TURN_RELEVANCE = "live_relevance"

LIVE_SINGLE_TURN_IRRELEVANCE = "live_irrelevance"

MULTI_TURN_AST = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]

# Global to track the expected max k across all subsets (None until set)
GLOBAL_MAX_K = None


def calculate_combined_accuracy(accuracy_dict_list: list[dict], weighted=False):
    total_count = 0
    total_div_count = 0  # Denominator for combined accuracy
    total_accuracy = 0

    for accuracy_dict in accuracy_dict_list:
        accuracy = accuracy_dict["accuracy"]
        count = accuracy_dict["num_entries"]

        total_count += count

        if weighted:
            total_div_count += count
            total_accuracy += accuracy * count
        else:
            # Unweighted accuracy
            total_div_count += 1
            total_accuracy += accuracy

    if total_count == 0:
        return {"accuracy": 0, "num_entries": 0}
    else:
        return {"accuracy": total_accuracy / total_div_count, "num_entries": total_count}


def get_accuracy_dict(metrics, category):
    # reporting aggregation for pass@1[avg-of-k] (for highest k) if available
    if f"bfcl_v3.{category}" in metrics:
        category_dict = metrics[f"bfcl_v3.{category}"]
    elif f"bfcl_v4.{category}" in metrics:
        category_dict = metrics[f"bfcl_v4.{category}"]
    else:
        raise ValueError(f"Metrics category {category} not found!")

    # Find all keys that match "pass@1[avg-of-{k}]"
    avg_keys = [key for key in category_dict.keys() if key.startswith("pass@1[avg-of-") and key.endswith("]")]

    # Determine k for this category: max k if avg keys present, else treat as k=1 (pass@1)
    k_for_category = 1
    selected_key = "pass@1"
    if avg_keys:
        ks = []
        for key in avg_keys:
            try:
                k_str = key.split("pass@1[avg-of-")[1].rstrip("]")
                k = int(k_str)
                ks.append((k, key))
            except ValueError:
                continue
        if ks:
            max_k, max_key = max(ks)
            k_for_category = max_k
            selected_key = max_key

    # Enforce global consistency of max k across subsets
    global GLOBAL_MAX_K
    if GLOBAL_MAX_K is None:
        GLOBAL_MAX_K = k_for_category
    elif GLOBAL_MAX_K != k_for_category:
        raise ValueError(
            f"Inconsistent max k across subsets: expected {GLOBAL_MAX_K}, "
            f"got {k_for_category} for category '{category}'. "
            "Check if all jobs have finished successfully. "
        )

    return category_dict[selected_key]


def calculate_non_live_single_turn_accuracy(metrics):
    # First calculate simple ast unweighted accuracy
    simple_ast_accuracy_dict = calculate_combined_accuracy(
        [get_accuracy_dict(metrics, category) for category in SIMPLE_AST], weighted=False
    )

    non_live_ast_accuracy_list = [simple_ast_accuracy_dict]
    for category in OTHER_SINGLE_TURN_AST:
        non_live_ast_accuracy_list.append(get_accuracy_dict(metrics, category))

    non_live_ast_accuracy = calculate_combined_accuracy(non_live_ast_accuracy_list, weighted=False)

    non_live_irrelevance_accuracy = get_accuracy_dict(metrics, SINGLE_TURN_IRRELEVANCE)

    overall_accuracy_non_live = calculate_combined_accuracy(
        non_live_ast_accuracy_list + [non_live_irrelevance_accuracy], weighted=False
    )

    return {
        "overall_non_live": overall_accuracy_non_live,
        "non_live_ast": non_live_ast_accuracy,
        "non_live_irrelevance": non_live_irrelevance_accuracy,
    }


def calculate_live_single_turn_accuracy(metrics):
    live_ast_accuracy_list = [get_accuracy_dict(metrics, category) for category in LIVE_SINGLE_TURN_AST]
    live_ast_accuracy = calculate_combined_accuracy(live_ast_accuracy_list, weighted=True)

    live_irrelevance_accuracy = get_accuracy_dict(metrics, LIVE_SINGLE_TURN_IRRELEVANCE)
    live_relevance_accuracy = get_accuracy_dict(metrics, LIVE_SINGLE_TURN_RELEVANCE)

    overall_accuracy_live = calculate_combined_accuracy(
        live_ast_accuracy_list + [live_irrelevance_accuracy, live_relevance_accuracy], weighted=True
    )

    return {
        "overall_live": overall_accuracy_live,
        "live_ast": live_ast_accuracy,
        "live_irrelevance": live_irrelevance_accuracy,
        "live_relevance": live_relevance_accuracy,
    }


def calculate_multi_turn_accuracy(metrics):
    multi_turn_accuracy_dict_list = [get_accuracy_dict(metrics, category) for category in MULTI_TURN_AST]
    overall_accuracy_multi_turn = calculate_combined_accuracy(multi_turn_accuracy_dict_list, weighted=False)

    return {
        "overall_multi_turn": overall_accuracy_multi_turn,
    }


def compute_score(metrics: dict):
    non_live_single_turn_accuracy = calculate_non_live_single_turn_accuracy(metrics)
    live_single_turn_accuracy = calculate_live_single_turn_accuracy(metrics)
    multi_turn_accuracy = calculate_multi_turn_accuracy(metrics)

    overall_accuracy = calculate_combined_accuracy(
        [
            non_live_single_turn_accuracy["overall_non_live"],
            live_single_turn_accuracy["overall_live"],
            multi_turn_accuracy["overall_multi_turn"],
        ],
        weighted=False,
    )

    return {
        "bfcl_v3": {
            "overall_accuracy": overall_accuracy,
            **non_live_single_turn_accuracy,
            **live_single_turn_accuracy,
            **multi_turn_accuracy,
        }
    }

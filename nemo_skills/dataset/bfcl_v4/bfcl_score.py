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


# TODO: refactor this to expose all metrics properly, currently for k>1 the reporting is partial


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

LIVE_SINGLE_TURN_AST = [
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
]

LIVE_SINGLE_TURN_RELEVANCE = "live_relevance"

HALLUCINATION = [
    "irrelevance",
    "live_irrelevance",
]

MULTI_TURN_AST = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]

MEMORY = [
    "memory_kv",
    "memory_vector",
    "memory_rec_sum",
]

WEB_SEARCH = [
    "web_search_base",
    "web_search_no_snippet",
]

FORMAT_SENSITIVITY = "format_sensitivity"


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


def get_accuracy_dict(metrics, category, optional=False):
    # reporting aggregation for pass@1[avg-of-k] (for highest k) if available
    if optional and f"bfcl_v4.{category}" not in metrics:
        category_dict = {}
    category_dict = metrics[f"bfcl_v4.{category}"]

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

    return {
        "overall_non_live": non_live_ast_accuracy,
    }


def calculate_live_single_turn_accuracy(metrics):
    live_ast_accuracy_list = [get_accuracy_dict(metrics, category) for category in LIVE_SINGLE_TURN_AST]
    live_ast_accuracy = calculate_combined_accuracy(live_ast_accuracy_list, weighted=True)

    live_relevance_accuracy = get_accuracy_dict(metrics, LIVE_SINGLE_TURN_RELEVANCE)

    return {
        "overall_live": live_ast_accuracy,
        "relevance": live_relevance_accuracy,
    }


def calculate_multi_turn_accuracy(metrics):
    multi_turn_accuracy_dict_list = [get_accuracy_dict(metrics, category) for category in MULTI_TURN_AST]
    overall_accuracy_multi_turn = calculate_combined_accuracy(multi_turn_accuracy_dict_list, weighted=False)

    return {
        "overall_multi_turn": overall_accuracy_multi_turn,
    }

def calculate_agentic_accuracy(metrics):
    memory_accuracy_list = [get_accuracy_dict(metrics, category) for category in MEMORY]
    overall_accuracy_memory = calculate_combined_accuracy(memory_accuracy_list, weighted=False)
    web_search_accuracy_list = [get_accuracy_dict(metrics, category) for category in WEB_SEARCH]
    overall_accuracy_web_search = calculate_combined_accuracy(web_search_accuracy_list, weighted=False)

    result_dict = {
        "overall_agentic": calculate_combined_accuracy([overall_accuracy_memory, overall_accuracy_web_search], weighted=False),
        "overall_memory": overall_accuracy_memory,
        "overall_web_search": overall_accuracy_web_search,
    }


    return result_dict


def calculate_hallucination_measurement(metrics):
    hallucination_accuracy_list = [get_accuracy_dict(metrics, category) for category in HALLUCINATION]
    overall_hallucination_accuracy = calculate_combined_accuracy(hallucination_accuracy_list, weighted=False)

    result_dict = {
        "overall_hallucination": overall_hallucination_accuracy
    }

    return result_dict


def compute_score(metrics: dict):
    non_live_single_turn_accuracy = calculate_non_live_single_turn_accuracy(metrics)
    live_single_turn_accuracy = calculate_live_single_turn_accuracy(metrics)
    multi_turn_accuracy = calculate_multi_turn_accuracy(metrics)
    agentic_accuracy = calculate_agentic_accuracy(metrics)
    hallucination_accuracy = calculate_hallucination_measurement(metrics)

    # Following the calculation guide from https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html
    overall_accuracy = (
        (agentic_accuracy["overall_agentic"] * 0.4) +
        (multi_turn_accuracy["overall_multi_turn"] * 0.3) +
        (live_single_turn_accuracy["overall_live"] * 0.1) +
        (non_live_single_turn_accuracy["overall_non_live"] * 0.1) +
        (hallucination_accuracy["overall_hallucination"] * 0.1)
    )

    res = {
        "overall_accuracy": overall_accuracy,
        **non_live_single_turn_accuracy,
        **live_single_turn_accuracy,
        **multi_turn_accuracy,
        **agentic_accuracy,
        **hallucination_accuracy,
    }

    return {"bfcl_v4": res}

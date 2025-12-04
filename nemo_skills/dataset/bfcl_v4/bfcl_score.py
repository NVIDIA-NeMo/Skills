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


from nemo_skills.dataset.bfcl_v3.bfcl_score import (
    calculate_combined_accuracy,
    calculate_multi_turn_accuracy,
    get_accuracy_dict,
)

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


def calculate_agentic_accuracy(metrics):
    memory_accuracy_list = [get_accuracy_dict(metrics, category) for category in MEMORY]
    overall_accuracy_memory = calculate_combined_accuracy(memory_accuracy_list, weighted=False)
    web_search_accuracy_list = [get_accuracy_dict(metrics, category) for category in WEB_SEARCH]
    overall_accuracy_web_search = calculate_combined_accuracy(web_search_accuracy_list, weighted=False)

    result_dict = {
        "overall_agentic": calculate_combined_accuracy(
            [overall_accuracy_memory, overall_accuracy_web_search], weighted=False
        ),
        "overall_memory": overall_accuracy_memory,
        "overall_web_search": overall_accuracy_web_search,
    }

    return result_dict


def calculate_hallucination_measurement(metrics):
    hallucination_accuracy_list = [get_accuracy_dict(metrics, category) for category in HALLUCINATION]
    overall_hallucination_accuracy = calculate_combined_accuracy(hallucination_accuracy_list, weighted=False)

    result_dict = {"overall_hallucination": overall_hallucination_accuracy}

    return result_dict


def compute_score(metrics: dict):
    non_live_single_turn_accuracy = calculate_non_live_single_turn_accuracy(metrics)
    live_single_turn_accuracy = calculate_live_single_turn_accuracy(metrics)
    multi_turn_accuracy = calculate_multi_turn_accuracy(metrics)
    agentic_accuracy = calculate_agentic_accuracy(metrics)
    hallucination_accuracy = calculate_hallucination_measurement(metrics)

    # Following the calculation guide from https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html
    overall_accuracy = (
        (agentic_accuracy["overall_agentic"]["accuracy"] * 0.4)
        + (multi_turn_accuracy["overall_multi_turn"]["accuracy"] * 0.3)
        + (live_single_turn_accuracy["overall_live"]["accuracy"] * 0.1)
        + (non_live_single_turn_accuracy["overall_non_live"]["accuracy"] * 0.1)
        + (hallucination_accuracy["overall_hallucination"]["accuracy"] * 0.1)
    )
    overall_num_entries = sum(
        [
            agentic_accuracy["overall_agentic"]["num_entries"],
            multi_turn_accuracy["overall_multi_turn"]["num_entries"],
            live_single_turn_accuracy["overall_live"]["num_entries"],
            non_live_single_turn_accuracy["overall_non_live"]["num_entries"],
            hallucination_accuracy["overall_hallucination"]["num_entries"],
        ]
    )

    res = {
        "overall_accuracy": {
            "accuracy": overall_accuracy,
            "num_entries": overall_num_entries,
        },
        **non_live_single_turn_accuracy,
        **live_single_turn_accuracy,
        **multi_turn_accuracy,
        **agentic_accuracy,
        **hallucination_accuracy,
    }

    return {"bfcl_v4": res}

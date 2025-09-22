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

# Scoring based on: https://artificialanalysis.ai/methodology/intelligence-benchmarking#intelligence-index-evaluation-suite-overview


def compute_score(metrics: dict):
    mmlu_pro = metrics["mmlu-pro"]["pass@1"]["symbolic_correct"]
    hle = metrics["hle"]["pass@1"]["judge_correct"]
    gpqa = metrics["gpqa"]["pass@1"]["symbolic_correct"]

    aime25 = metrics["aime24"]["pass@1[avg-of-10]"]["symbolic_correct"]

    scicode = metrics["scicode"]["pass@1[avg-of-3]"]["subtask_accuracy"]
    livecodebench = metrics["livecodebench"]["pass@1[avg-of-3]"]["accuracy"]

    ifbench = metrics["ifbench"]["pass@1[avg-of-5]"]["average_score"]

    aalcr = metrics["aalcr"]["pass@1[avg-of-3]"]["judge_correct"]

    math_score = aime25
    code_score = (scicode + livecodebench) / 2

    overall_score = (mmlu_pro + hle + gpqa + aime25 + scicode + livecodebench + ifbench + aalcr) / 8
    return {
        "overall_score": overall_score,
        "math_score": math_score,
        "code_score": code_score,
        "mmlu_pro": mmlu_pro,
        "hle": hle,
        "gpqa": gpqa,
        "aime25": aime25,
        "scicode": scicode,
        "livecodebench": livecodebench,
        "ifbench": ifbench,
        "aalcr": aalcr,
    }

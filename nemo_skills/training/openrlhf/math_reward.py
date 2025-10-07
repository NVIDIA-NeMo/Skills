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

import asyncio
import json
import os

import torch

from nemo_skills.evaluation.math_grader import extract_answer
from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from nemo_skills.inference.model import get_model
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import prefill_judgement


def reward_func(queries: list[str], prompts: list[str], prompt_metadata: list[dict]):
    """Will check if the predicted answer matches expected answer and return 1 or 0 accordingly.

    Args:
        queries: list of responses from an actor LLM
        prompts: list of prompts that queries are generated from
        prompt_metadata: any other keys from the original data file
            corresponding to each prompt/query (e.g. expected_answer)
    """
    data_points = []
    prefilled_judgements = []
    prefilled_indices = set()
    for idx, (metadata, query) in enumerate(zip(prompt_metadata, queries)):
        dp = {
            "problem": metadata["problem"],
            "expected_answer": metadata["expected_answer"],
            "predicted_answer": extract_answer(query),
        }
        judgement = prefill_judgement(dp)
        if judgement is not None:
            prefilled_judgements.append(judgement)
            prefilled_indices.add(idx)
        else:  # cannot prefill, will send to an LLM
            data_points.append(dp)

    host = os.getenv("SLURM_MASTER_NODE_HET_GROUP_0", "localhost")
    server_args = json.loads(os.getenv("REWARD_SERVER_ARGS", "{}"))
    llm = get_model(host=host, **server_args)
    # TODO: remove hardcoded qwen template
    prompt = get_prompt("judge/math", "Qwen/Qwen2.5-32B-Instruct")
    judge_prompts = [prompt.fill(dp) for dp in data_points]
    if len(judge_prompts) > 0:
        # Too slow, but we are no longer supporting openrlhf anyways.
        outputs = [
            asyncio.run(llm.generate_async(prompt=jp, stop_phrases=prompt.stop_phrases)) for jp in judge_prompts
        ]
    else:
        outputs = []
    judgements = []
    prefilled_idx = 0
    generation_idx = 0
    for idx in range(len(queries)):  # looping over all and selecting either prefilled or generated judgements
        if idx in prefilled_indices:
            judgements.append(prefilled_judgements[prefilled_idx])
            prefilled_idx += 1
        else:
            judgements.append(outputs[generation_idx]["generation"])
            generation_idx += 1
    is_correct_array = [is_correct_judgement(judgement) for judgement in judgements]
    return torch.tensor(is_correct_array, dtype=torch.float32)

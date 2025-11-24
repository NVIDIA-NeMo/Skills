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
import random

import omegaconf
from transformers import AutoTokenizer

from nemo_skills.evaluation.math_grader import extract_answer
from nemo_skills.inference.model import BaseModel

MAX_RETRIES = 20
MAX_QWEN_TOKENS = 10000

PROMPT_ADDITION = " Please put your final answer within \\boxed{...}."

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


async def _llm_call(llm: BaseModel, messages: list[dict], llm_kwargs: dict) -> str:
    response = await llm.generate_async(
        prompt=messages,
        **llm_kwargs,
        random_seed=random.randint(0, 1000000),
    )
    result = response["generation"]
    result = result.split("</think>")[-1].strip()
    if len(tokenizer.encode(result)) > MAX_QWEN_TOKENS:
        print(f"Token length {len(tokenizer.encode(result))} exceeds {MAX_QWEN_TOKENS}, skipping.")
        return None
    return result


def extract_corrects_incorrects(batch_results: list[str], gt_answer: str) -> tuple[list[str], list[str]]:
    corrects = []
    incorrects = []
    for result in batch_results:
        if result is None:
            continue
        pred_answer = extract_answer(result, extract_from_boxed=True)
        if not pred_answer or not pred_answer.isdigit():
            continue
        is_eq = int(pred_answer) == int(gt_answer)
        if is_eq:
            corrects.append(result)
        else:
            print(f"Predicted: {pred_answer} != Expected: {gt_answer}")
            incorrects.append(result)
    return corrects, incorrects


async def process_single(
    llm: BaseModel, datapoint: dict, prompt_config_path: str, n_pos_neg: int, llm_kwargs: dict
) -> dict:
    prompt_config = omegaconf.OmegaConf.load(prompt_config_path)
    prompt = prompt_config.user.format(problem=datapoint["problem"])
    messages = [
        {"role": "user", "content": prompt.strip() + PROMPT_ADDITION},
    ]
    positives, negatives = [], []
    for _ in range(MAX_RETRIES):
        batch_results = await asyncio.gather(*[_llm_call(llm, messages, llm_kwargs) for _ in range(4 * n_pos_neg)])
        new_positives, new_negatives = extract_corrects_incorrects(batch_results, datapoint["expected_answer"])
        positives.extend(new_positives)
        negatives.extend(new_negatives)
        if max(len(positives), len(negatives)) > 20 and min(len(positives), len(negatives)) == 0:
            print(
                f"{datapoint['source']}: {len(positives)} positives and {len(negatives)} negatives, there is no way the model changes its mind."
            )
            break
        if len(positives) >= n_pos_neg and len(negatives) >= n_pos_neg:
            break
    return {
        **datapoint,
        "positives": positives,
        "negatives": negatives,
    }

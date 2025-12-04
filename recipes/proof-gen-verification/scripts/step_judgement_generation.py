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
from collections import Counter
from functools import partial

import numpy as np
from omegaconf import OmegaConf

from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from nemo_skills.inference.model import BaseModel

is_correct_judgement_or_none = partial(is_correct_judgement, return_none=True)

VAR_TO_JUDGEMENT_MAP = {
    None: "None",
    True: "Judgement: Yes",
    False: "Judgement: No",
}


async def process_single(
    llm: BaseModel,
    datapoint: dict,
    step_mode: str,  # "step-based" or "lemma-based" or "truth-based"
    # For step-based mode:
    step_break_prompt_path: str,
    step_judge_prompt_path: str,
    # For lemma-based mode:
    lemma_break_prompt_path: str,
    lemma_judge_prompt_path: str,
    # For truth-based mode:
    truth_break_prompt_path: str,
    truth_judge_prompt_path: str,
    step_maj_n: int,
    llm_kwargs: dict,
    random_seed: int,
) -> dict:
    rng = np.random.RandomState(random_seed + 10000 * datapoint["_async_position"])

    if step_mode == "lemma-based":
        final_judgement_str, judgement_list, output_tokens = await process_lemma_judgement(
            llm, datapoint, lemma_break_prompt_path, lemma_judge_prompt_path, step_maj_n, llm_kwargs, rng
        )
    elif step_mode == "step-based":
        final_judgement_str, judgement_list, output_tokens = await process_step_judgement(
            llm, datapoint, step_break_prompt_path, step_judge_prompt_path, step_maj_n, llm_kwargs, rng
        )
    elif step_mode == "truth-based":
        final_judgement_str, judgement_list, output_tokens = await process_truth_judgement(
            llm, datapoint, truth_break_prompt_path, truth_judge_prompt_path, step_maj_n, llm_kwargs, rng
        )
    else:
        raise ValueError(f"Invalid step_mode: {step_mode}. Must be 'step-based', 'lemma-based', or 'truth-based'")

    return {
        **datapoint,
        "judgement": final_judgement_str,
        "judgement_list": judgement_list,
        "num_generated_tokens": output_tokens,
    }


def load_prompt_template(prompt_config_path):
    """Load the prompt template from the config file."""
    config = OmegaConf.load(prompt_config_path)
    assert getattr(config, "system", None) is None, "System prompt is not allowed"
    return config.user


def _surround_with_step_index(proof_steps: list[str], target_step_slice: slice):
    assert target_step_slice.start >= 0 and target_step_slice.stop <= len(proof_steps)
    assert target_step_slice.start < target_step_slice.stop
    new_proof = (
        proof_steps[: target_step_slice.start]
        + ["<judge_block>"]
        + proof_steps[target_step_slice.start : target_step_slice.stop]
        + ["</judge_block>"]
        + proof_steps[target_step_slice.stop :]
    )
    return "".join(new_proof)


async def _llm_call(llm: BaseModel, prompt: str, llm_kwargs: dict, req_seed: int, validation_fn=None):
    messages = [{"role": "user", "content": prompt}]
    for i in range(5):
        response = await llm.generate_async(
            prompt=messages,
            **llm_kwargs,
            random_seed=req_seed,
        )
        full_response = response["generation"]
        output_tokens = response["num_generated_tokens"]
        serialized_output = response["serialized_output"]
        llm_text = full_response.split("</think>")[-1].strip()
        if validation_fn is None:
            break
        if validation_fn(llm_text) is not None:
            break
        req_seed = req_seed + 1
        print(f"Invalid response, retrying... {i + 1} of 5")
    return llm_text, output_tokens, serialized_output


def _parse_lemmas(lemma_break_result: str):
    """Parse lemma break result to extract <problem>...</problem><proof>...</proof> pairs.

    Returns None if parsing fails (used for validation), otherwise returns list of lemmas.
    """
    lemmas = []
    remaining = lemma_break_result
    while "<problem>" in remaining:
        problem_start = remaining.find("<problem>")
        problem_end = remaining.find("</problem>")
        if problem_end == -1:
            return None

        proof_start = remaining.find("<proof>", problem_end)
        proof_end = remaining.find("</proof>", proof_start)
        if proof_start == -1 or proof_end == -1:
            return None

        problem_text = remaining[problem_start + len("<problem>") : problem_end].strip()
        proof_text = remaining[proof_start + len("<proof>") : proof_end].strip()

        # Validate that we actually got some text
        if not problem_text or not proof_text:
            return None

        lemmas.append({"problem": problem_text, "proof": proof_text})
        remaining = remaining[proof_end + len("</proof>") :]

    # Must have at least one lemma
    if len(lemmas) == 0:
        return None

    return lemmas


def _parse_truth_statements(truth_break_result: str):
    """Parse truth break result to extract <statement>...</statement> blocks.

    Returns None if parsing fails (used for validation),
    Returns "EARLY_REJECT" if the break step detected obvious fatal flaws,
    otherwise returns list of statements.
    """
    # Check if the break step returned an early rejection judgement
    if "<judgement>Judgement: No</judgement>" in truth_break_result:
        return "EARLY_REJECT"

    statements = []
    remaining = truth_break_result
    while "<statement>" in remaining:
        statement_start = remaining.find("<statement>")
        statement_end = remaining.find("</statement>")
        if statement_end == -1:
            return None

        statement_text = remaining[statement_start + len("<statement>") : statement_end].strip()

        # Validate that we actually got some text
        if not statement_text:
            return None

        statements.append({"statement": statement_text})
        remaining = remaining[statement_end + len("</statement>") :]

    # Must have at least one statement
    if len(statements) == 0:
        return None

    return statements


def _compute_majority_vote(step_results):
    """Compute majority vote from multiple judgement results."""
    judgement_votes = []
    all_step_judgements = []
    total_tokens = 0

    for judgement, judgement_tokens, serialized_output_i in step_results:
        total_tokens += judgement_tokens
        judgement_result = is_correct_judgement_or_none(judgement)
        judgement_votes.append(judgement_result)
        all_step_judgements.append(
            {
                "judgement_raw": judgement,
                "judgement": judgement_result,
                "serialized_output": serialized_output_i,
            }
        )

    vote_counter = Counter(judgement_votes)
    majority_judgement = vote_counter.most_common(1)[0][0]

    return majority_judgement, dict(vote_counter), all_step_judgements, total_tokens


async def process_step_judgement(
    llm: BaseModel,
    datapoint: dict,
    step_break_prompt_path: str,
    step_judge_prompt_path: str,
    step_maj_n: int,
    llm_kwargs: dict,
    rng: np.random.RandomState,
) -> dict:
    output_tokens = 0
    step_break_prompt = load_prompt_template(step_break_prompt_path).format(
        problem=datapoint["problem"], proof=datapoint["proof"]
    )
    step_break_result, step_break_tokens, serialized_output = await _llm_call(
        llm, step_break_prompt, llm_kwargs, rng.randint(0, 2**32)
    )
    output_tokens += step_break_tokens
    step_break_result = step_break_result.replace("</step>", "")  # sometimes the model adds </step> at the end
    proof_steps = step_break_result.split("<step>")
    proof_steps = list(filter(lambda x: x.strip(), proof_steps))

    # if the result it too large, or empty, we declare invalid
    if len("".join(proof_steps)) > 2 * len(datapoint["proof"]) or len(proof_steps) == 0:
        return VAR_TO_JUDGEMENT_MAP[None], [], output_tokens

    prompts = [
        load_prompt_template(step_judge_prompt_path).format(
            problem=datapoint["problem"], proof=_surround_with_step_index(proof_steps, slice(i, i + 1))
        )
        for i in range(len(proof_steps))
    ]

    llm_tasks = []
    for prompt in prompts:
        for _ in range(step_maj_n):
            llm_tasks.append(
                _llm_call(llm, prompt, llm_kwargs, rng.randint(0, 2**32), validation_fn=is_correct_judgement_or_none)
            )

    llm_results = await asyncio.gather(*llm_tasks)

    judgement_list = [{"proof_steps": proof_steps, "serialized_output": serialized_output}]

    for i in range(len(proof_steps)):
        step_results = llm_results[i * step_maj_n : (i + 1) * step_maj_n]
        majority_judgement, majority_votes, all_step_judgements, tokens_used = _compute_majority_vote(step_results)
        output_tokens += tokens_used

        judgement_list.append(
            {
                "judgement_prompt": prompts[i],
                "judgement": majority_judgement,
                "majority_votes": majority_votes,
                "all_judgements": all_step_judgements,
            }
        )

        if majority_judgement in [None, False]:
            return VAR_TO_JUDGEMENT_MAP[majority_judgement], judgement_list, output_tokens

    return VAR_TO_JUDGEMENT_MAP[True], judgement_list, output_tokens


async def process_lemma_judgement(
    llm: BaseModel,
    datapoint: dict,
    lemma_break_prompt_path: str,
    lemma_judge_prompt_path: str,
    step_maj_n: int,
    llm_kwargs: dict,
    rng: np.random.RandomState,
) -> dict:
    output_tokens = 0
    lemma_break_prompt = load_prompt_template(lemma_break_prompt_path).format(
        problem=datapoint["problem"], proof=datapoint["proof"]
    )
    lemma_break_result, lemma_break_tokens, serialized_output = await _llm_call(
        llm, lemma_break_prompt, llm_kwargs, rng.randint(0, 2**32), validation_fn=_parse_lemmas
    )
    output_tokens += lemma_break_tokens

    lemmas = _parse_lemmas(lemma_break_result)

    # if the result is too large, or empty, we declare invalid
    if lemmas is None:
        return VAR_TO_JUDGEMENT_MAP[None], [], output_tokens

    # Make sure length of any of the lemmas is not too large
    for lemma in lemmas:
        if len(lemma["problem"]) + len(lemma["proof"]) > 2 * len(datapoint["problem"] + datapoint["proof"]):
            return VAR_TO_JUDGEMENT_MAP[None], [], output_tokens

    # Create prompts for each lemma
    prompts = [
        load_prompt_template(lemma_judge_prompt_path).format(problem=lemma["problem"], proof=lemma["proof"])
        for lemma in lemmas
    ]

    llm_tasks = []
    for prompt in prompts:
        for _ in range(step_maj_n):
            llm_tasks.append(
                _llm_call(llm, prompt, llm_kwargs, rng.randint(0, 2**32), validation_fn=is_correct_judgement_or_none)
            )

    llm_results = await asyncio.gather(*llm_tasks)

    judgement_list = [{"lemmas": lemmas, "serialized_output": serialized_output}]

    for i in range(len(lemmas)):
        lemma_results = llm_results[i * step_maj_n : (i + 1) * step_maj_n]
        majority_judgement, majority_votes, all_lemma_judgements, tokens_used = _compute_majority_vote(lemma_results)
        output_tokens += tokens_used

        judgement_list.append(
            {
                "lemma_problem": lemmas[i]["problem"],
                "lemma_proof": lemmas[i]["proof"],
                "judgement_prompt": prompts[i],
                "judgement": majority_judgement,
                "majority_votes": majority_votes,
                "all_judgements": all_lemma_judgements,
            }
        )

        if majority_judgement in [None, False]:
            return VAR_TO_JUDGEMENT_MAP[majority_judgement], judgement_list, output_tokens

    return VAR_TO_JUDGEMENT_MAP[True], judgement_list, output_tokens


async def process_truth_judgement(
    llm: BaseModel,
    datapoint: dict,
    truth_break_prompt_path: str,
    truth_judge_prompt_path: str,
    step_maj_n: int,
    llm_kwargs: dict,
    rng: np.random.RandomState,
) -> dict:
    output_tokens = 0
    truth_break_prompt = load_prompt_template(truth_break_prompt_path).format(
        problem=datapoint["problem"], proof=datapoint["proof"]
    )
    truth_break_result, truth_break_tokens, serialized_output = await _llm_call(
        llm, truth_break_prompt, llm_kwargs, rng.randint(0, 2**32), validation_fn=_parse_truth_statements
    )
    output_tokens += truth_break_tokens

    statements = _parse_truth_statements(truth_break_result)

    # Check if the break step detected obvious fatal flaws and returned early rejection
    if statements == "EARLY_REJECT":
        return (
            VAR_TO_JUDGEMENT_MAP[False],
            [{"early_rejection": True, "serialized_output": serialized_output}],
            output_tokens,
        )

    # if the result is too large, or empty, we declare invalid
    if statements is None:
        return VAR_TO_JUDGEMENT_MAP[None], [], output_tokens

    # Make sure length of any of the statements is not too large
    for statement in statements:
        if len(statement["statement"]) > 2 * len(datapoint["problem"] + datapoint["proof"]):
            return VAR_TO_JUDGEMENT_MAP[None], [], output_tokens

    # Create prompts for each statement
    prompts = []
    for i, statement in enumerate(statements):
        # Build previous statements string
        if i == 0:
            previous_statements = "No previous statements."
        else:
            previous_stmts = [f"{j + 1}. {statements[j]['statement']}" for j in range(i)]
            previous_statements = "\n".join(previous_stmts)

        prompt = load_prompt_template(truth_judge_prompt_path).format(
            statement=statement["statement"], previous_statements=previous_statements
        )
        prompts.append(prompt)

    llm_tasks = []
    for prompt in prompts:
        for _ in range(step_maj_n):
            llm_tasks.append(
                _llm_call(llm, prompt, llm_kwargs, rng.randint(0, 2**32), validation_fn=is_correct_judgement_or_none)
            )

    llm_results = await asyncio.gather(*llm_tasks)

    judgement_list = [{"statements": statements, "serialized_output": serialized_output}]

    for i in range(len(statements)):
        statement_results = llm_results[i * step_maj_n : (i + 1) * step_maj_n]
        majority_judgement, majority_votes, all_statement_judgements, tokens_used = _compute_majority_vote(
            statement_results
        )
        output_tokens += tokens_used

        judgement_list.append(
            {
                "statement": statements[i]["statement"],
                "judgement_prompt": prompts[i],
                "judgement": majority_judgement,
                "majority_votes": majority_votes,
                "all_judgements": all_statement_judgements,
            }
        )

        if majority_judgement in [None, False]:
            return VAR_TO_JUDGEMENT_MAP[majority_judgement], judgement_list, output_tokens

    return VAR_TO_JUDGEMENT_MAP[True], judgement_list, output_tokens

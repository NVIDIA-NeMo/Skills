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
import copy
import logging
from functools import partial

import numpy as np
from omegaconf import OmegaConf

from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from nemo_skills.inference.model import BaseModel

is_correct_judgement_or_none = partial(is_correct_judgement, return_none=True)

TOURNAMENT_PROMPT = """[Instructions]
You are a senior mathematician. You are given a math problems, a proof and a list of {num_participants} judgements from different graders.
Each grader has graded the proof and provided a judgement. Each grader has provided a one-paragraph summary of their analysis, and a final judgement of whether the proof is correct or not.
Your task as a senior grader is to carefully analyze the summary from each grader and pick the best judgement from the list.

[Format]

Your output must end with the following tags:
<meta_summary>One-paragraph summary of your analysis</meta_summary>
<best_judgement>Index</best_judgement>

The best_judgement tag MUST ONLY contain an integer between 1 and {num_participants} (inclusive).

[Problem]
{problem}

[Proof]
{proof}"""

JUDGEMENT_SUMMARY_FORMAT = """[Grader {idx}]
<summary>{summary}</summary>
<judgement>{judgement}</judgement>"""

PROMPT_FORMATS = {
    "default": TOURNAMENT_PROMPT,
}


def load_prompt_template(prompt_config_path):
    """Load the prompt template from the config file."""
    config = OmegaConf.load(prompt_config_path)
    assert getattr(config, "system", None) is None, "System prompt is not allowed"
    return config.user


# TODO: GLM does not put </think> in the response, so we consider the full response for now.
async def _llm_call(llm: BaseModel, prompt: str, llm_kwargs: dict, req_seed: int):
    messages = [{"role": "user", "content": prompt}]
    response = await llm.generate_async(
        prompt=messages,
        **llm_kwargs,
        random_seed=req_seed,
    )
    full_response = response["generation"]
    output_tokens = response["num_generated_tokens"]
    llm_text = full_response.split("</think>")[-1].strip()
    return llm_text, output_tokens


def extract_judgement_summary_result(judgement: str):
    all_keys = ["<judgement>", "</judgement>", "<summary>", "</summary>"]
    if not all(key in judgement for key in all_keys):
        return None
    summary = judgement.split("<summary>")[-1].split("</summary>")[0].strip()
    judgement = judgement.split("<judgement>")[-1].split("</judgement>")[0].strip()
    judgement_is_correct = is_correct_judgement_or_none(judgement)
    if judgement_is_correct is None:
        return None
    return {
        "judgement": judgement,
        "summary": summary,
    }


def _create_tournament_prompt(
    tournament_judgements, judgement_summary_results, problem, proof, prompt_config_path: str
):
    """Create a tournament prompt for the given judgements."""
    tournament_prompt = (
        load_prompt_template(prompt_config_path)
        .format(num_participants=len(tournament_judgements), problem=problem, proof=proof)
        .strip()
    )

    # Add each judgement with its summary
    for idx, (i, judgement) in enumerate(tournament_judgements, 1):
        judgement_data = judgement_summary_results[i]
        tournament_prompt += "\n\n" + JUDGEMENT_SUMMARY_FORMAT.format(
            idx=idx, summary=judgement_data["summary"], judgement=judgement_data["judgement"]
        )

    return tournament_prompt


def _extract_winner_and_eliminate(tournament_result, tournament_judgements, req_seed: int):
    """Extract winner from tournament result and eliminate losers."""
    knocked_out_judgements = []
    if "<best_judgement>" in tournament_result and "</best_judgement>" in tournament_result:
        try:
            winner_idx_str = tournament_result.split("<best_judgement>")[-1].split("</best_judgement>")[0].strip()
            winner_idx = int(winner_idx_str) - 1  # Convert to 0-based index
            winner_original_idx = tournament_judgements[winner_idx][0]

            # Knock out all non-winners from this tournament
            for i, _ in tournament_judgements:
                if i != winner_original_idx:
                    knocked_out_judgements.append((i, f"Lost to judgement {winner_original_idx} in tournament"))
            return winner_original_idx, knocked_out_judgements, True
        except (ValueError, IndexError):
            pass

    # If we can't parse the result, randomly select a winner and eliminate the rest
    logging.warning(
        f"Tournament result parsing failed, randomly selecting a winner. Tournament result: {tournament_result}"
    )
    winner_original_idx = int(np.random.RandomState(req_seed).choice([x[0] for x in tournament_judgements]))
    for i, _ in tournament_judgements:
        if i != winner_original_idx:
            knocked_out_judgements.append((i, "Randomly eliminated (tournament result parsing failed)"))
    return winner_original_idx, knocked_out_judgements, False


async def _run_single_tournament(
    llm: BaseModel,
    tournament_judgements,
    judgement_summary_results,
    problem,
    proof,
    prompt_config_path: str,
    llm_kwargs: dict,
    req_seed: int,
):
    """Run a single tournament and return the winner and eliminated judgements."""
    tournament_prompt = _create_tournament_prompt(
        tournament_judgements, judgement_summary_results, problem, proof, prompt_config_path
    )
    num_generated_tokens = 0
    for i in range(5):
        new_req_seed = req_seed + i
        tournament_result, new_num_generated_tokens = await _llm_call(llm, tournament_prompt, llm_kwargs, new_req_seed)
        winner_idx, knocked_out, parsing_success = _extract_winner_and_eliminate(
            tournament_result, tournament_judgements, new_req_seed
        )
        num_generated_tokens += new_num_generated_tokens
        if parsing_success:
            break
        print(f"Tournament result parsing failed, retrying... {i + 1} of 5")
    touenament_log = {
        "participant_id": [x[0] for x in tournament_judgements],
        "num_generated_tokens": num_generated_tokens,
        "winner_idx": winner_idx,
        "knocked_out_ids": [x[0] for x in knocked_out],
    }

    return winner_idx, knocked_out, touenament_log


async def run_judgement_tournament(
    llm: BaseModel,
    datapoint: dict,
    n_judgements_per_tournament: int,
    prompt_config_path: str,
    llm_kwargs: dict,
    rng: np.random.RandomState,
) -> dict:
    knocked_out_judgements = []
    judgements_candids = list(enumerate(datapoint["judgement_candidates"]))
    judgement_summary_results = [extract_judgement_summary_result(judgement) for _, judgement in judgements_candids]
    all_tournament_logs = []
    # Initial filtering of invalid judgements
    for i, judgement_cand in judgements_candids:
        if judgement_summary_results[i] is None:
            knocked_out_judgements.append((i, "Invalid format (summary or judgement missing)"))

    while len(knocked_out_judgements) < len(judgements_candids) - 1:
        # Get remaining judgements
        knocked_out_judgements_indices = [ko[0] for ko in knocked_out_judgements]
        remaining_judgements = [
            (i, judgement) for i, judgement in judgements_candids if i not in knocked_out_judgements_indices
        ]
        # Shuffle remaining judgements
        rng.shuffle(remaining_judgements)

        if len(remaining_judgements) < 2:
            break

        # Group remaining judgements into tournament batches
        tournament_batches = []
        for i in range(0, len(remaining_judgements), n_judgements_per_tournament):
            batch = remaining_judgements[i : i + n_judgements_per_tournament]
            if len(batch) >= 2:  # Only run tournaments with at least 2 participants
                tournament_batches.append(batch)

        if not tournament_batches:
            break

        # Run all tournaments in parallel
        tournament_tasks = [
            _run_single_tournament(
                llm,
                batch,
                judgement_summary_results,
                datapoint["problem"],
                datapoint["proof"],
                prompt_config_path,
                llm_kwargs,
                rng.randint(0, 100000),
            )
            for batch in tournament_batches
        ]

        tournament_results = await asyncio.gather(*tournament_tasks)

        # Process all tournament results
        for winner_idx, eliminated, tournament_log in tournament_results:
            knocked_out_judgements.extend(eliminated)
            all_tournament_logs.append(tournament_log)
    # Find the final winner (the one not knocked out)
    remaining_judgements = [
        (i, judgement) for i, judgement in judgements_candids if i not in [ko[0] for ko in knocked_out_judgements]
    ]

    if remaining_judgements:
        assert len(remaining_judgements) == 1, "Expected exactly one remaining judgement"
        final_judgement_idx = remaining_judgements[0][0]
        final_judgement = judgements_candids[final_judgement_idx][1]
    else:
        final_judgement = "No valid judgement found"

    return final_judgement, knocked_out_judgements, all_tournament_logs


async def process_single(
    llm: BaseModel,
    datapoint: dict,
    n_judgements_per_tournament: int,
    max_seeds_to_use: int,
    prompt_config_path: str,
    llm_kwargs: dict,
    random_seed: int,
) -> dict:
    # Run Keeping only the first max_seeds_to_use judgements
    datapoint = copy.deepcopy(datapoint)
    datapoint["judgement_candidates"] = datapoint["judgement_candidates"][:max_seeds_to_use]
    datapoint["num_generated_tokens_list"] = datapoint["num_generated_tokens_list"][:max_seeds_to_use]
    rng = np.random.RandomState(random_seed + 10000 * datapoint["_async_position"])
    final_judgement_str, tournoment_logs, all_tournament_logs = await run_judgement_tournament(
        llm, datapoint, n_judgements_per_tournament, prompt_config_path, llm_kwargs, rng
    )
    return {
        **datapoint,
        "judgement": final_judgement_str,
        "tournament_logs": tournoment_logs,
        "all_tournament_logs": all_tournament_logs,
    }

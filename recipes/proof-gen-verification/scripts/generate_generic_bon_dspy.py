#!/usr/bin/env python3
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

"""
Generic BON evaluation using script generation.
Supports llm-as-a-judge (binary and scoring) and genselect (pairwise) evaluation methods.
"""

import asyncio
import copy
import itertools
import re
from functools import partial

import numpy as np
import omegaconf

from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from nemo_skills.inference.model import BaseModel
from nemo_skills.inference.tournament_utils import ProofKnockoutTournamentManager

is_correct_judgement_or_none = partial(is_correct_judgement, return_none=True)


async def process_single(
    llm: BaseModel,
    datapoint: dict,
    eval_type: str,  # "llm_as_judge_binary", "llm_as_judge_binary_v2", "llm_as_judge_binary_gt_proof", "llm_as_judge_scoring", "llm_as_judge_scoring_rubric_gt_proof", or "genselect"
    judgement_num_seeds: int,  # Number of seeds for llm-as-a-judge
    judgement_binary_prompt_config_path: str,  # LLM-as-a-judge binary prompt
    judgement_binary_prompt_config_path_v2: str,  # LLM-as-a-judge binary prompt v2
    judgement_binary_gt_proof_prompt_config_path: str,  # LLM-as-a-judge binary prompt with rubric
    judgement_scoring_prompt_config_path: str,  # LLM-as-a-judge scoring prompt (0-7)
    judgement_scoring_rubric_gt_proof_prompt_config_path: str,  # LLM-as-a-judge scoring prompt with rubric and ground truth proof
    genselect_prompt_config_path: str,  # Genselect prompt for pairwise comparisons
    llm_kwargs: dict,  # LLM kwargs
    random_seed: int,  # Random seed for the datapoint
) -> dict:
    datapoint = copy.deepcopy(datapoint)
    rng = np.random.RandomState(random_seed + 10000 * datapoint["_async_position"])

    proofs_list = datapoint.get("proofs", [])

    # Mapping of eval types to their prompt configs and score extractors
    llm_as_judge_configs = {
        "llm_as_judge_binary": (judgement_binary_prompt_config_path, extract_binary_correctness),
        "llm_as_judge_binary_v2": (judgement_binary_prompt_config_path_v2, extract_binary_correctness),
        "llm_as_judge_binary_gt_proof": (judgement_binary_gt_proof_prompt_config_path, extract_binary_correctness),
        "llm_as_judge_scoring": (judgement_scoring_prompt_config_path, extract_score_from_xml),
        "llm_as_judge_scoring_rubric_gt_proof": (
            judgement_scoring_rubric_gt_proof_prompt_config_path,
            extract_score_from_xml,
        ),
    }

    if eval_type in llm_as_judge_configs:
        # Run llm-as-a-judge evaluation
        prompt_config_path, score_extractor = llm_as_judge_configs[eval_type]
        results = await run_llm_as_judge(
            llm=llm,
            datapoint=datapoint,
            proofs_list=proofs_list,
            judgement_num_seeds=judgement_num_seeds,
            judgement_prompt_config_path=prompt_config_path,
            score_extractor=score_extractor,
            llm_kwargs=llm_kwargs,
            rng=rng,
            method_name=eval_type,
        )
    elif eval_type == "genselect":
        # Run pairwise genselect evaluation
        results = await run_genselect_pairwise(
            llm=llm,
            datapoint=datapoint,
            proofs_list=proofs_list,
            genselect_prompt_config_path=genselect_prompt_config_path,
            llm_kwargs=llm_kwargs,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")

    return {
        **datapoint,
        "evaluation_type": eval_type,
        "evaluation_results": results,
    }


async def _llm_call(llm: BaseModel, prompt: str, llm_kwargs: dict, req_seed: int):
    """Make an LLM call and return the response and metadata."""
    messages = [{"role": "user", "content": prompt}]
    response = await llm.generate_async(
        prompt=messages,
        **llm_kwargs,
        random_seed=req_seed,
    )
    full_response = response["generation"]
    output_tokens = response["num_generated_tokens"]
    llm_text = full_response.split("</think>")[-1].strip()
    return llm_text, {"num_generated_tokens": output_tokens}


def extract_score_from_xml(text: str) -> float:
    """Extract score from XML format (0-7 scale).

    Expected format: <score>N</score> where N is an integer in [0, 7]

    Returns:
        float: The extracted score (0-7), or None if not found or invalid
    """
    match = re.search(r"<score>(\d+)</score>", text)
    if match:
        score = int(match.group(1))
        if 0 <= score <= 7:
            return float(score)
    return None


def extract_binary_correctness(text: str) -> float:
    """Extract binary correctness from judgment text.

    Returns:
        float: 1.0 if correct, 0.0 if incorrect, None if invalid
    """
    result = is_correct_judgement_or_none(text)
    if result is True:
        return 1.0
    elif result is False:
        return 0.0
    return None


async def run_llm_as_judge(
    llm: BaseModel,
    datapoint: dict,
    proofs_list: list[str],
    judgement_num_seeds: int,
    judgement_prompt_config_path: str,
    score_extractor: callable,
    llm_kwargs: dict,
    rng: np.random.RandomState,
    method_name: str,
) -> dict:
    """Run llm-as-a-judge evaluation for all proofs.

    For each proof, generate multiple judgments with different seeds,
    and compute average scores using the provided score_extractor function.

    Args:
        llm: The language model to use
        datapoint: The full datapoint containing all fields (problem, rubric, ground_truth_proof, etc.)
        proofs_list: List of proof strings to evaluate
        judgement_num_seeds: Number of judgment seeds
        judgement_prompt_config_path: Path to judgment prompt config
        score_extractor: Function to extract score from judgment text (returns float or None)
        llm_kwargs: LLM generation kwargs
        rng: Random number generator
        method_name: Name of the method (for result dict)

    Returns:
        dict: Contains judgments for each proof with average scores
    """
    prompt_template = omegaconf.OmegaConf.load(judgement_prompt_config_path).user

    # Create tasks for all proofs and all seeds
    all_tasks = []
    task_metadata = []  # Track which proof and seed each task corresponds to

    for proof_idx, proof in enumerate(proofs_list):
        for seed_idx in range(judgement_num_seeds):
            # Pass all datapoint fields to the prompt template, plus the current proof
            # Extra kwargs are fine - format() will ignore fields not in the template
            assert "proof" not in datapoint, "proof should not be in datapoint"
            format_kwargs = {**datapoint, "proof": proof}

            prompt_formatted = prompt_template.format(**format_kwargs)
            task = _llm_call(llm, prompt_formatted, llm_kwargs, rng.randint(0, 100000))
            all_tasks.append(task)
            task_metadata.append((proof_idx, seed_idx))

    # Run all tasks in parallel
    results = await asyncio.gather(*all_tasks)

    # Organize results by proof
    proofs_judgements = [{"judgements": [], "scores": [], "num_generated_tokens": []} for _ in proofs_list]

    for (proof_idx, seed_idx), (judgement_text, aux_info) in zip(task_metadata, results):
        proofs_judgements[proof_idx]["judgements"].append(judgement_text)
        proofs_judgements[proof_idx]["num_generated_tokens"].append(aux_info["num_generated_tokens"])

        # Extract score using the provided extractor function
        score = score_extractor(judgement_text)
        proofs_judgements[proof_idx]["scores"].append(score)

    # Compute average scores
    for proof_data in proofs_judgements:
        scores = proof_data["scores"]
        valid_scores = [s for s in scores if s is not None]
        proof_data["average_score"] = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        proof_data["num_valid_scores"] = len(valid_scores)

    return {
        "method": method_name,
        "judgement_num_seeds": judgement_num_seeds,
        "proofs_judgements": proofs_judgements,
    }


async def run_genselect_pairwise(
    llm: BaseModel,
    datapoint: dict,
    proofs_list: list[str],
    genselect_prompt_config_path: str,
    llm_kwargs: dict,
    rng: np.random.RandomState,
) -> dict:
    """Run pairwise genselect evaluation (all C(n,2) comparisons).

    For each pair of proofs, run a tournament to determine the winner.

    Args:
        llm: The language model to use
        datapoint: The full datapoint containing all fields (problem, etc.)
        proofs_list: List of proof strings to compare
        genselect_prompt_config_path: Path to genselect prompt config
        llm_kwargs: LLM generation kwargs
        rng: Random number generator

    Returns:
        dict: Contains pairwise comparison results
    """
    n_proofs = len(proofs_list)

    # Generate all pairwise combinations
    pairs = list(itertools.combinations(range(n_proofs), 2))

    tournament_manager = ProofKnockoutTournamentManager(
        llm=llm,
        n_participants_per_tournament=2,
        prompt_config_path=genselect_prompt_config_path,
        llm_kwargs=llm_kwargs,
        rng=rng,
    )

    # Create tasks for all pairwise comparisons
    pairwise_tasks = []
    for idx1, idx2 in pairs:
        participants = [(idx1, proofs_list[idx1]), (idx2, proofs_list[idx2])]
        common_context = {"problem": datapoint["problem"]}

        task = tournament_manager.run_single_game(
            participants=participants,
            common_context=common_context,
            req_seed=rng.randint(0, 100000),
        )
        pairwise_tasks.append(task)

    # Run all pairwise comparisons in parallel
    pairwise_results = await asyncio.gather(*pairwise_tasks)

    # Process results to count wins for each proof
    pairwise_comparisons = []

    for (idx1, idx2), (winner_idx, _, game_metadata) in zip(pairs, pairwise_results):
        # winner_idx is the original index of the winner
        parsing_success = game_metadata["parsing_success"]
        pairwise_comparisons.append(
            {
                "proof1_idx": idx1,
                "proof2_idx": idx2,
                "winner_idx": winner_idx if parsing_success else None,
                "game_metadata": game_metadata,
            }
        )

    return {
        "method": "genselect",
        "n_proofs": n_proofs,
        "pairwise_comparisons": pairwise_comparisons,
    }

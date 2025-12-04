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
from functools import partial

import numpy as np
import omegaconf
from transformers import AutoTokenizer

from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from nemo_skills.inference.model import BaseModel
from nemo_skills.inference.tournament_utils import ProofKnockoutTournamentManager

is_correct_judgement_or_none = partial(is_correct_judgement, return_none=True)

# For filtering out too long solutions
MAX_QWEN_TOKENS = 10000
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


async def process_single(
    llm: BaseModel,
    datapoint: dict,
    proof_generation_prompt_config_path: str,  # Prover prompt
    max_num_solutions: int,  # How many solutions to generate
    proof_genselect_to_keep: int,  # How many solutions to keep after proof genselect
    proof_genselect_prompt_config_path: str,  # Proof Genselect prompt
    judgement_num_seeds: int,  # LLM-as-a-judge seed count
    judgement_prompt_config_path: str,  # LLM-as-a-judge prompt
    llm_kwargs: dict,  # LLM kwargs, including temperature, max_tokens, etc.
    random_seed: int,  # Random seed for the datapoint
) -> dict:
    """Process a single datapoint with parallelized proof processing."""
    datapoint = copy.deepcopy(datapoint)
    rng = np.random.RandomState(random_seed + 10000 * datapoint["_async_position"])

    print(f"Processing datapoint {datapoint['_async_position']}")
    # Step 1: Generate all proofs in parallel
    all_proofs_list: list[dict] = await generate_proofs(
        llm, datapoint, max_num_solutions, proof_generation_prompt_config_path, llm_kwargs, rng
    )
    print(f"Generated {len(all_proofs_list)} proofs for datapoint {datapoint['_async_position']}")

    # Step 2: Run proof genselect (tournament-based selection)
    selected_proofs_list, selected_proofs_aux_info = await run_proof_genselect(
        llm,
        datapoint["problem"],
        all_proofs_list,
        proof_genselect_to_keep,
        proof_genselect_prompt_config_path,
        llm_kwargs,
        rng,
    )
    print(f"Selected {len(selected_proofs_list)} proofs for datapoint {datapoint['_async_position']}")

    # Step 3: Process each selected proof with judgements in parallel
    proof_tasks = []
    for i, proof_data in enumerate(selected_proofs_list):
        # Create a new RNG for this proof
        proof_rng = np.random.RandomState(rng.randint(0, 2**32))
        task = process_single_proof_judgements(
            llm,
            datapoint["problem"],
            proof_data,
            judgement_num_seeds,
            judgement_prompt_config_path,
            llm_kwargs,
            proof_rng,
        )
        proof_tasks.append(task)

    proofs_with_judgements_list = await asyncio.gather(*proof_tasks)
    # Take max score proof as the llm as judge proof
    llm_as_judge_proof = max(proofs_with_judgements_list, key=lambda x: x["judgements_score"])["proof"]
    print(f"Finished processing datapoint {datapoint['_async_position']}")

    return {
        **datapoint,
        "original_proofs_list": all_proofs_list,
        "proof_genselect_aux": selected_proofs_aux_info,
        "proof_judgements_results": proofs_with_judgements_list,
        "llm_as_judge_selected_proof": llm_as_judge_proof,
    }


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
    return llm_text, {"num_generated_tokens": output_tokens}


async def generate_proofs(
    llm: BaseModel,
    datapoint: dict,
    max_num_solutions: int,
    prompt_config_path: str,
    llm_kwargs: dict,
    rng: np.random.RandomState,
) -> list[dict]:
    """Generate multiple proofs in parallel."""
    prompt_formatted = omegaconf.OmegaConf.load(prompt_config_path).user.format(problem=datapoint["problem"])

    # Create tasks for parallel proof generation
    tasks = [_llm_call(llm, prompt_formatted, llm_kwargs, rng.randint(0, 100000)) for _ in range(max_num_solutions)]

    # Run all proof generation tasks in parallel
    results = await asyncio.gather(*tasks)

    # Filter out too long proofs
    results = [result for result in results if len(tokenizer.encode(result[0])) <= MAX_QWEN_TOKENS]
    results_dict = [{"proof": result[0], "proof_gen_aux": result[1]} for i, result in enumerate(results)]
    return results_dict


async def run_proof_genselect(
    llm: BaseModel,
    problem: str,
    proofs_list: list[dict],
    proof_genselect_to_keep: int,
    proof_genselect_prompt_config_path: str,
    llm_kwargs: dict,
    rng: np.random.RandomState,
) -> tuple[list[dict], dict]:
    if proof_genselect_to_keep == -1:
        return proofs_list, {}

    if len(proofs_list) <= proof_genselect_to_keep:
        return proofs_list, {}

    # Create tournament manager for proof selection
    tournament_manager = ProofKnockoutTournamentManager(
        llm=llm,
        n_participants_per_tournament=2,
        prompt_config_path=proof_genselect_prompt_config_path,
        llm_kwargs=llm_kwargs,
        rng=rng,
    )

    # Prepare participants as list of proof strings
    participants = [proof_data["proof"] for proof_data in proofs_list]

    common_context = {"problem": problem}

    # Run tournament to select the best proofs
    winner_indices, tournament_logs = await tournament_manager.run_tournament(
        participants=participants, common_context=common_context, winner_count=proof_genselect_to_keep
    )

    # Get selected proofs using winner indices
    selected_proofs = [proofs_list[i] for i in winner_indices]

    aux_info = {
        "tournament_logs": tournament_logs,
        "original_count": len(proofs_list),
        "selected_count": len(selected_proofs),
        "winner_indices": winner_indices,
    }

    return selected_proofs, aux_info


async def process_single_proof_judgements(
    llm: BaseModel,
    problem: str,
    proof_data: dict,
    judgement_num_seeds: int,
    judgement_prompt_config_path: str,
    llm_kwargs: dict,
    rng: np.random.RandomState,
) -> dict:
    """Generate multiple judgements for a single proof using different seeds."""
    # Generate judgements for this proof
    judgement_tasks = []
    prompt_template = omegaconf.OmegaConf.load(judgement_prompt_config_path).user
    for seed in range(judgement_num_seeds):
        prompt_formatted = prompt_template.format(problem=problem, proof=proof_data["proof"])
        task = _llm_call(llm, prompt_formatted, llm_kwargs, rng.randint(0, 100000))
        judgement_tasks.append(task)

    # Run all judgement tasks in parallel
    judgement_results = await asyncio.gather(*judgement_tasks)

    judgements = []
    gen_tokens = []
    for result, aux in judgement_results:
        judgements.append(result)
        gen_tokens.append(aux["num_generated_tokens"])

    proof_with_judgements = {
        **proof_data,
        "judgements": judgements,
        "judgement_aux": {
            "num_participants": len(judgements),
            "num_generated_tokens_list": gen_tokens,
        },
        "judgements_score": compute_judgement_scores(judgements),
    }

    return proof_with_judgements


def compute_judgement_scores(judgements_list: list[str]) -> list[float]:
    """Num correct / total"""
    not_none_judgements = [judgement for judgement in judgements_list if judgement is not None]
    if not not_none_judgements:  # If all judgements are None, return lowest score
        return -1
    return sum([int(is_correct_judgement_or_none(judgement) is True) for judgement in not_none_judgements]) / len(
        not_none_judgements
    )

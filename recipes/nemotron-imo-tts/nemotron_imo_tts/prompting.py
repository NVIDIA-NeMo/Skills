# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Prompt templates and byte-exact rendering for generation, verification, refinement, and judging."""

from dataclasses import dataclass
from pathlib import Path

import yaml

CANONICAL_GENERATION_PROMPT = "proof_generation"
VERIFICATION_PROMPT = "proof_verification"
REFINEMENT_PROMPT = "proof_refinement"
JUDGE_PROMPT = "proof_imo_judge"


@dataclass(frozen=True)
class PromptSet:
    generation_order: tuple[str, ...]
    generation: dict[str, str]
    canonical_generation: str
    verification: str
    refinement: str
    judge: str | None


def load_template(path):
    """The ``user`` field of a prompt yaml file."""
    with open(path, encoding="utf-8") as f:
        document = yaml.safe_load(f)
    if not isinstance(document, dict) or not isinstance(document.get("user"), str):
        raise ValueError(f"Prompt file {path} must contain a string 'user' field")
    return document["user"]


def prompt_path(prompts_dir, prompt_id):
    return Path(prompts_dir) / f"{prompt_id}.yaml"


def load_prompt_set(prompts_dir, generation_prompt_ids, with_judge):
    generation = {prompt_id: load_template(prompt_path(prompts_dir, prompt_id)) for prompt_id in generation_prompt_ids}
    return PromptSet(
        generation_order=tuple(generation_prompt_ids),
        generation=generation,
        canonical_generation=load_template(prompt_path(prompts_dir, CANONICAL_GENERATION_PROMPT)),
        verification=load_template(prompt_path(prompts_dir, VERIFICATION_PROMPT)),
        refinement=load_template(prompt_path(prompts_dir, REFINEMENT_PROMPT)),
        judge=load_template(prompt_path(prompts_dir, JUDGE_PROMPT)) if with_judge else None,
    )


def render_generation(template, question):
    return template.format(question=question)


def render_verification(template, question, proof):
    return template.format(statement=question, proof=proof)


def format_proofs_to_refine(proof, critiques):
    """The candidate block of a refinement prompt, in the exact format of the report run."""
    ratings = [f"=== Evaluation {i} of Solution 0 ===\n{critique}" for i, critique in enumerate(critiques)]
    return f"--- Solution 0 ---\n{proof}\n\n" + "\n\n".join(ratings)


def render_refinement(prompts, question, proof, critiques):
    """Refinement instruction = the standard generation prompt; candidate = one proof with its critiques."""
    instruction = prompts.canonical_generation.format(question=question)
    return prompts.refinement.format(
        instruction=instruction, proofs_to_refine=format_proofs_to_refine(proof, critiques)
    )


def render_judge(template, question, proof):
    return template.format(problem=question, response=proof)


def user_message(text):
    return [{"role": "user", "content": text}]

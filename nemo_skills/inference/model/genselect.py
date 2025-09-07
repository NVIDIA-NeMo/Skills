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
import glob
import hashlib
import json
import logging
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Union

from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_logger_name, nested_dataclass, remove_thinking

from .base import BaseModel

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class GenSelectConfig:
    use_completions_api: bool = False
    tokenizer: str | None = None
    chat_template_kwargs: dict | None = None  # extra parameters to pass to the tokenizer's apply_chat_template method
    temperature: float = 0.6
    tokens_to_generate: int | None = None
    remove_thinking: bool = True  # Remove thinking tokens from the comparison key
    thinking_begin: str = "<think>"
    thinking_end: str = "</think>"

    # GenSelect Simple
    window_size: int = 8  # Number of solutions compared in a single request
    regex: str = r"Judg[e]?ment: (\d+)"
    prompt_config: str = "generic/genselect"
    comparison_key: str = "generation"  # Key used for comparing the different solutions
    filter_incomplete_solutions: bool = True  # Filter out incomplete solutions

    # Parameters specifically for Offline GenSelect
    generation_dir: str | None = None  # Assumes output-rs[random_seed].jsonl files in this directory
    num_initial_solutions: int | None = None  # If specified, will only consider this many solutions for GenSelect


class GenSelectWrapper:
    """
    Wrapper that generates multiple completions for a datapoint and uses GenSelect
    to choose the best one.
    """

    def __init__(self, model: BaseModel, orig_prompt_filler, cfg: GenSelectConfig):
        self.model = model
        self.orig_prompt_filler = orig_prompt_filler
        self.cfg = cfg

        # Load GenSelect prompt
        if self.cfg.use_completions_api:
            tokenizer = self.cfg.tokenizer or self.model.model_name_or_path
        else:
            tokenizer = None
        self.genselect_prompt = get_prompt(prompt_config=self.cfg.prompt_config, tokenizer=tokenizer)

        # Initialize the solutions if input_dir is provided
        if self.cfg.generation_dir is not None:
            LOG.info("Loading solutions from %s", self.cfg.generation_dir)
            self.prompt_to_solutions_dict = self._load_solutions(self.cfg.generation_dir)
            LOG.info("Loaded solutions for %d prompts", len(self.prompt_to_solutions_dict))

        # TODO: These calculations will change for GenSelect Competition
        if self.cfg.generation_dir is not None:
            self.cfg.max_concurrent_requests = 1
        else:
            # We will be generating the solutions in parallel
            self.cfg.max_concurrent_requests = self.cfg.window_size

    def _extract_judgment(self, generation: str, max_idx: int) -> Optional[int]:
        """Extract the judgment index from GenSelect generation."""
        judgment = None

        try:
            matches = re.findall(self.cfg.regex, generation)
            if matches:
                number = matches[-1]
                judgment = int(number)
                if judgment > max_idx:
                    judgment = None
            else:
                judgment = None
        except Exception:
            judgment = None

        if judgment is not None and judgment > max_idx:
            judgment = None

        return judgment

    def _format_solutions_for_genselect(self, solutions: List[Dict]) -> str:
        """Format solutions for GenSelect prompt."""
        formatted_solutions = []
        for i, solution in enumerate(solutions):
            formatted_solutions.append(f"Solution {i}: {solution[self.cfg.comparison_key]}")
        return "\n\n".join(formatted_solutions)

    async def _run_genselect(
        self, prompt: str, solutions: List[Dict], local_random: random.Random
    ) -> tuple[int, Dict]:
        """Run GenSelect to choose the best solution."""
        # Step 1: Format the solutions for GenSelect
        num_solutions = len(solutions)
        max_idx = num_solutions - 1
        solutions_text = self._format_solutions_for_genselect(solutions)

        genselect_input = {
            "problem": prompt,
            "solutions": solutions_text,
            "num_solutions": num_solutions,
            "max_idx": max_idx,
        }

        genselect_prompt = self.genselect_prompt.fill(genselect_input)

        # Step 2: Run Self-GenSelect
        genselect_result = await self.model.generate_async(
            prompt=genselect_prompt,
            tokens_to_generate=self.cfg.tokens_to_generate,
            temperature=self.cfg.temperature,
            remove_stop_phrases=True,
        )

        # Step 3: Extract the judgment from the GenSelect result
        judgment = self._extract_judgment(genselect_result["generation"], max_idx)
        if judgment is None:
            LOG.warning("GenSelect failed to produce valid judgment, falling back to random selection")
            judgment = local_random.randint(0, max_idx)

        return judgment, genselect_result

    async def generate_solutions(
        self,
        prompt: Union[str, List],
        local_random: random.Random,
        **solution_kwargs,
    ) -> Dict:
        """
        Generate multiple solutions and use Self-GenSelect to choose the best one.
        """
        # Generate multiple solutions
        tasks = []
        for _ in range(self.cfg.window_size):
            # Generate solutions with different seeds for diversity
            cur_random_seed = local_random.getrandbits(32)
            # Create a copy to avoid mutation issues
            current_kwargs = solution_kwargs.copy()
            current_kwargs["random_seed"] = cur_random_seed

            task = self.model.generate_async(prompt=prompt, **current_kwargs)
            tasks.append(task)

        generation_results = await asyncio.gather(*tasks)
        solutions = []
        for generation_result in generation_results:
            if self.cfg.remove_thinking:
                remove_thinking(
                    generation_result,
                    generation_key=self.cfg.comparison_key,
                    thinking_begin=self.cfg.thinking_begin,
                    thinking_end=self.cfg.thinking_end,
                )

            solutions.append(
                {
                    self.cfg.comparison_key: generation_result[self.cfg.comparison_key],
                    "output_dict": generation_result,
                }
            )

        local_random.shuffle(solutions)
        return solutions

    @classmethod
    def hash_prompt(cls, prompt: Union[str, List[dict]]) -> str:
        """Hash any data structure - handles strings, lists, dicts, etc."""
        return hashlib.md5(json.dumps(prompt, sort_keys=True, default=str).encode()).hexdigest()

    def _load_solutions(self, input_dir: str) -> Dict[str, List[Dict]]:
        """Load the solutions from the input directory."""
        prompt_to_solutions_dict = defaultdict(list)
        solution_files = glob.glob(os.path.join(input_dir, "output-rs*.jsonl"))

        # If num_initial_solutions is specified, only load the first num_initial_solutions solutions
        if self.cfg.num_initial_solutions is not None:
            # Sort the solution files to ensure consistent ordering
            solution_files.sort()
            solution_files = solution_files[: self.cfg.num_initial_solutions]

        if not solution_files:
            raise ValueError(f"No solutions found in {input_dir}")

        for input_file in solution_files:
            with open(input_file, "r") as f:
                for line in f:
                    data_point = json.loads(line)
                    # TODO: Making an assumptiont that the prompt doesn't require all the data for few-shot prompting
                    # Hashing the prompt to get the key for the solutions
                    prompt = self.hash_prompt(self.orig_prompt_filler(data_point, data=None))
                    prompt_to_solutions_dict[prompt].append(
                        {
                            self.cfg.comparison_key: data_point[self.cfg.comparison_key],
                            "output_dict": data_point,
                        }
                    )

        return prompt_to_solutions_dict

    async def generate_async(self, prompt: Union[str, List], **kwargs):
        """Generate a single solution using GenSelect."""

        local_random = random.Random(kwargs.get("random_seed", 0))

        # Step 1: Load/Generate the solutions
        if self.cfg.generation_dir is not None:
            # Already have the solutions in the input directory
            # Hashing the prompt to get the key for the solutions
            solutions = self.prompt_to_solutions_dict[self.hash_prompt(prompt)]
            local_random.shuffle(solutions)
            # After shuffling, only take the first window_size solutions
            solutions = solutions[: self.cfg.window_size]
        else:
            # Generate the solutions first
            solutions = await self.generate_solutions(prompt, local_random, **kwargs)

        total_num_generated_tokens = 0
        for solution in solutions:
            total_num_generated_tokens += solution["output_dict"].get("num_generated_tokens", 0)

        if self.cfg.filter_incomplete_solutions:
            # Remove unfinished solutions
            filtered_solutions = []
            for solution in solutions:
                # Check if thinking_begin is in the solution and thinking_end is not in the solution
                if (
                    self.cfg.thinking_begin in solution[self.cfg.comparison_key]
                    and self.cfg.thinking_end not in solution[self.cfg.comparison_key]
                ):
                    continue
                else:
                    filtered_solutions.append(solution)

            LOG.info(f"Filtered out {len(solutions) - len(filtered_solutions)} incomplete solutions")
            solutions = filtered_solutions

        if not solutions:
            return {
                self.cfg.comparison_key: "",
                "solution_list": [],
                "genselect_comparison": "",
                "genselect_num_generated_tokens": 0,
                "total_solution_generated_tokens": total_num_generated_tokens,
                "num_generated_tokens": total_num_generated_tokens,  # No additional tokens for genselect
                "num_best_solution_generated_tokens": 0,
            }

        # Step 2: Run GenSelect to choose the best solution
        best_index, genselect_result = await self._run_genselect(prompt, solutions, local_random)
        best_solution = solutions[best_index]

        # Return the best solution in the expected format
        result = {
            self.cfg.comparison_key: best_solution[self.cfg.comparison_key],
            "solution_list": [solution[self.cfg.comparison_key] for solution in solutions],
            "genselect_comparison": genselect_result["generation"],
        }

        if self.cfg.comparison_key != "generation":
            # Add the generation key to the result since it's required by inference/generate.py
            result["generation"] = best_solution["output_dict"]["generation"]

        result["total_solution_generated_tokens"] = total_num_generated_tokens

        # Add the tokens for genselect
        result["genselect_num_generated_tokens"] = genselect_result.get("num_generated_tokens", 0)

        # Add the tokens for all the solutions and genselect
        total_gen_tokens = result["total_solution_generated_tokens"] + result["genselect_num_generated_tokens"]

        # TODO: Decide what count of generated tokens do we want to report - the total or the best solution?
        # Current implementation returns the total number of generated tokens
        result["num_generated_tokens"] = total_gen_tokens

        # Add the tokens for the best solution
        result["num_best_solution_generated_tokens"] = best_solution["output_dict"].get("num_generated_tokens", 0)

        return result

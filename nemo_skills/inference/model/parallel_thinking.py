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
from dataclasses import field
from typing import Dict, List, Optional, Union

from transformers import AutoTokenizer

from nemo_skills.prompt.utils import get_prompt, get_token_count
from nemo_skills.utils import get_logger_name, nested_dataclass, parse_reasoning

from .base import BaseModel, EndpointType

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class GenSelectSpecificConfig:
    prompt_config: str = "generic/genselect"
    regex: str = r"Judg[e]?ment: (\d+)"


@nested_dataclass(kw_only=True)
class GenSynthesisSpecificConfig:
    prompt_config: str = "generic/gensynthesis"
    regex: str = r"<NEW_SOLUTION>(.*?)</NEW_SOLUTION>"


@nested_dataclass(kw_only=True)
class ParallelThinkingConfig:
    temperature: float = 0.6
    tokens_to_generate: int | None = None

    parse_reasoning: bool = False
    parse_reasoning_solutions: bool = True  # Whether to parse the reasoning for the solutions.
    end_reasoning_string: str = "</think>"
    endpoint_type: EndpointType = EndpointType.chat
    tokenizer: str | None = None
    chat_template_kwargs: dict = field(default_factory=dict)
    start_assistant_response_key: str | None = None  # whether to start assistant response with this key

    # Count the number of tokens in the prompt
    count_prompt_tokens: bool = False

    # GenSelect vs GenSynthesis
    mode: str | None = None  # genselect or gensynthesis

    genselect: GenSelectSpecificConfig = field(default_factory=GenSelectSpecificConfig)
    gensynthesis: GenSynthesisSpecificConfig = field(default_factory=GenSynthesisSpecificConfig)

    # Solution related parameters
    solution_length_cap: int | None = 16384  # If specified, will filter out solutions that are longer than this length
    window_size: int = 8  # Number of solutions compared in a single request
    solution_key: str = "generation"  # Key used for identifying the solution content
    filter_incomplete_solutions: bool = True  # Filter out incomplete solutions

    # Parameters specifically for Offline GenSelect/GenSynthesis
    generation_dir: str | None = None  # Assumes output-rs[random_seed].jsonl files in this directory
    num_initial_solutions: int | None = None  # If specified, will only consider this many solutions


class ParallelThinkingTask:
    """
    Wrapper that generates/loads multiple solutions for a datapoint and uses GenSelect or GenSynthesis
    to choose the best one or synthesize a new solution.
    """

    def __init__(self, model: BaseModel, tokenizer: str | None, orig_prompt_filler, cfg: ParallelThinkingConfig):
        self.model = model
        self.orig_prompt_filler = orig_prompt_filler
        self.cfg = cfg

        self.tokenizer = tokenizer

        # Load GenSelect/GenSynthesis prompt
        if self.cfg.mode == "genselect":
            self.parallel_thinking_prompt = get_prompt(
                prompt_config=self.cfg.genselect.prompt_config, tokenizer=self.tokenizer
            )
        elif self.cfg.mode == "gensynthesis":
            self.parallel_thinking_prompt = get_prompt(
                prompt_config=self.cfg.gensynthesis.prompt_config, tokenizer=self.tokenizer
            )
        else:
            raise ValueError(f"Invalid parallel thinking mode: {self.cfg.mode}")

        if self.cfg.count_prompt_tokens:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
            if self.hf_tokenizer is None:
                raise ValueError("Tokenizer could not be initialized. Needed for counting prompt tokens.")

        # Initialize the solutions if input_dir is provided
        if self.cfg.generation_dir is not None:
            LOG.info("Loading solutions from %s", self.cfg.generation_dir)
            self.prompt_to_solutions_dict = self._load_solutions(self.cfg.generation_dir)
            LOG.info("Loaded solutions for %d prompts", len(self.prompt_to_solutions_dict))

        # TODO: These calculations will change for Parallel Thinking competition setting
        if self.cfg.generation_dir is not None:
            self.cfg.max_concurrent_requests = 1
        else:
            # We will be generating the solutions in parallel
            self.cfg.max_concurrent_requests = self.cfg.window_size

    @classmethod
    def hash_prompt(cls, prompt: Union[str, List[dict]]) -> str:
        """Hash any data structure - handles strings, lists, dicts, etc."""
        return hashlib.md5(json.dumps(prompt, sort_keys=True, default=str).encode()).hexdigest()

    async def generate_solutions(
        self,
        prompt: Union[str, List],
        local_random: random.Random,
        **solution_kwargs,
    ) -> Dict:
        """
        Generate multiple solutions for input to Parallel Thinking.
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
            if self.cfg.parse_reasoning_solutions:
                orig_generation = generation_result[self.cfg.solution_key]
                parse_reasoning(
                    generation_result,
                    generation_key=self.cfg.solution_key,
                    end_reasoning_string=self.cfg.end_reasoning_string,
                )
                if generation_result[self.cfg.solution_key] == "":
                    # Revert to original generation, probably because reasoning is already parsed
                    generation_result[self.cfg.solution_key] = orig_generation

            if self.cfg.solution_length_cap is not None:
                if len(generation_result[self.cfg.solution_key]) > self.cfg.solution_length_cap:
                    LOG.debug(
                        f"Solution filtered out: length {len(generation_result[self.cfg.solution_key])} exceeds cap {self.cfg.solution_length_cap}"
                    )
                    continue

            solutions.append(
                {
                    self.cfg.solution_key: generation_result[self.cfg.solution_key],
                    "output_dict": generation_result,
                }
            )

        local_random.shuffle(solutions)
        return solutions

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
                    if self.cfg.parse_reasoning_solutions:
                        orig_generation = data_point[self.cfg.solution_key]
                        parse_reasoning(
                            data_point,
                            generation_key=self.cfg.solution_key,
                            end_reasoning_string=self.cfg.end_reasoning_string,
                        )
                        if data_point[self.cfg.solution_key] == "":
                            # Revert to original generation, probably because reasoning is already parsed
                            data_point[self.cfg.solution_key] = orig_generation

                    if self.cfg.solution_length_cap is not None:
                        if len(data_point[self.cfg.solution_key]) > self.cfg.solution_length_cap:
                            LOG.debug(
                                f"Solution filtered out: length {len(data_point[self.cfg.solution_key])} exceeds cap {self.cfg.solution_length_cap}"
                            )
                            continue

                    # TODO: Making an assumption that the prompt doesn't require all the data for few-shot prompting
                    # Hashing the prompt to get the key for the solutions
                    prompt = self.hash_prompt(self.orig_prompt_filler(data_point, data=None))
                    prompt_to_solutions_dict[prompt].append(
                        {
                            self.cfg.solution_key: data_point[self.cfg.solution_key],
                            "output_dict": data_point,
                        }
                    )

        return prompt_to_solutions_dict

    async def _get_multiple_solutions(
        self, prompt: Union[str, List], local_random: random.Random, **kwargs
    ) -> tuple[List[Dict], int]:
        """Return multiple solutions for the input prompt."""
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

        # Filter out incomplete solutions if specified
        if self.cfg.filter_incomplete_solutions:
            # Remove unfinished solutions
            filtered_solutions = []
            for solution in solutions:
                if solution[self.cfg.solution_key] == "":
                    LOG.warning("Solution is empty, skipping")
                    continue
                else:
                    filtered_solutions.append(solution)

            if len(filtered_solutions) < len(solutions):
                LOG.info(f"Filtered out {len(solutions) - len(filtered_solutions)} incomplete solutions")

            solutions = filtered_solutions

        total_num_generated_tokens = 0
        for solution in solutions:
            total_num_generated_tokens += solution["output_dict"].get("num_generated_tokens", 0)

        return solutions, total_num_generated_tokens

    async def _generate_parallel_thinking_contraction(self, prompt: str, solutions: List[Dict], **kwargs) -> Dict:
        """Output which combines the solutions into a single solution/selection."""

        num_solutions = len(solutions)
        max_idx = num_solutions - 1

        formatted_solutions = []
        for i, solution in enumerate(solutions):
            formatted_solutions.append(f"Solution {i}: {solution[self.cfg.solution_key]}")
        solutions_text = "\n\n".join(formatted_solutions)

        parallel_thinking_input = {
            "problem": prompt,
            "solutions": solutions_text,
            "num_solutions": num_solutions,
            "max_idx": max_idx,
        }

        parallel_thinking_prompt = self.parallel_thinking_prompt.fill(
            parallel_thinking_input,
            start_assistant_response_key=self.cfg.start_assistant_response_key,
            chat_template_kwargs=self.cfg.chat_template_kwargs,
        )

        LOG.info(f"Parallel thinking prompt:\n\n{parallel_thinking_prompt}")

        output_dict = {}
        if self.cfg.count_prompt_tokens:
            num_input_tokens = get_token_count(tokenizer=self.hf_tokenizer, messages=parallel_thinking_prompt)
            output_dict["num_input_tokens"] = num_input_tokens

        for duplicate_key in ["temperature", "tokens_to_generate", "prompt", "endpoint_type"]:
            kwargs.pop(duplicate_key, None)

        LOG.info(f"kwargs: {kwargs}")
        kwargs["endpoint_type"] = self.cfg.endpoint_type

        output_dict.update(
            await self.model.generate_async(
                prompt=parallel_thinking_prompt,
                # Overriding the tokens_to_generate, temperature
                tokens_to_generate=self.cfg.tokens_to_generate,
                temperature=self.cfg.temperature,
                **kwargs,
            )
        )
        return output_dict

    def _extract_selected_solution(self, generation: str, max_idx: int) -> Optional[int]:
        """Extract the selected solutions index from the GenSelect generation."""
        solution_idx = None

        try:
            matches = re.findall(self.cfg.genselect.regex, generation)
            if matches:
                number = matches[-1]
                solution_idx = int(number)
                if solution_idx > max_idx:
                    return None

        except Exception:
            return None

        return solution_idx

    def _extract_synthesized_solution(self, generation: str) -> str:
        """Extract the synthesized solution from the GenSynthesis result."""
        matches = re.findall(self.cfg.gensynthesis.regex, generation, re.DOTALL)
        if matches:
            return matches[-1].strip()  # Remove any trailing newlines
        else:
            return None

    async def _run_genselect(
        self, prompt: str, solutions: List[Dict], local_random: random.Random, **kwargs
    ) -> tuple[int, Dict]:
        """Run GenSelect to choose the best solution."""

        max_idx = len(solutions) - 1
        genselect_result = await self._generate_parallel_thinking_contraction(
            prompt=prompt, solutions=solutions, **kwargs
        )

        # Extract the judgment from the GenSelect result
        sel_solution_idx = self._extract_selected_solution(genselect_result["generation"], max_idx)
        if sel_solution_idx is None:
            LOG.warning("GenSelect failed to produce valid solution index, falling back to random selection")
            sel_solution_idx = local_random.randint(0, max_idx)
            genselect_result["generation_successful"] = False
        else:
            genselect_result["generation_successful"] = True

        return {
            self.cfg.solution_key: solutions[sel_solution_idx][self.cfg.solution_key],
            "parallel_thinking_result": genselect_result,
        }

    async def _run_gensynthesis(
        self, prompt: str, solutions: List[Dict], local_random: random.Random, **kwargs
    ) -> Dict:
        """Run GenSynthesis to synthesize a new solution from a list of candidate solutions."""

        gensynthesis_result = await self._generate_parallel_thinking_contraction(
            prompt=prompt, solutions=solutions, **kwargs
        )

        # Extract the synthesized solution from the GenSynthesis result
        synthesized_solution = self._extract_synthesized_solution(gensynthesis_result["generation"])
        if synthesized_solution is None:
            LOG.warning("GenSynthesis failed to produce valid solution, falling back to random selection")
            synthesized_solution = local_random.choice(solutions)[self.cfg.solution_key]
            # Add the boolean flag to aid analysis and debugging
            gensynthesis_result["generation_successful"] = False
        else:
            gensynthesis_result["generation_successful"] = True

        return {
            self.cfg.solution_key: synthesized_solution,
            "parallel_thinking_result": gensynthesis_result,
        }

    async def generate_async(self, prompt: Union[str, List], **kwargs):
        """Generate a single solution using parallel thinking."""

        result = {}  # Result dictionary
        local_random = random.Random(kwargs.get("random_seed", 0))

        # Step 1: Get the multiple solutions
        solutions, total_num_generated_tokens = await self._get_multiple_solutions(prompt, local_random, **kwargs)
        result["total_solution_generated_tokens"] = total_num_generated_tokens

        if solutions is None or len(solutions) == 0:
            output_dict = {
                self.cfg.solution_key: "",
                "solution_list": [],
                f"{self.cfg.mode}_comparison": "",
                f"{self.cfg.mode}_num_generated_tokens": 0,
                f"{self.cfg.mode}_successful": False,
                "total_solution_generated_tokens": total_num_generated_tokens,
                "num_generated_tokens": total_num_generated_tokens,  # No additional tokens for genselect/gensynthesis
                "num_best_solution_generated_tokens": 0,
            }

            # Required by inference/generate.py
            output_dict["generation"] = ""
            if self.cfg.count_prompt_tokens:
                # The input doesn't make sense for such cases where there are no solutions
                output_dict["num_input_tokens"] = 0

            LOG.warning("No solutions found for the prompt, returning empty output")
            return output_dict

        # Step 2: Run GenSelect/GenSynthesis

        # If the prompt is a list, we need to get the first message's content
        prompt_str = prompt if isinstance(prompt, str) else prompt[0]["content"]
        assert isinstance(prompt_str, str), "Prompt must be a string"

        if self.cfg.mode == "genselect":
            output_dict = await self._run_genselect(prompt_str, solutions, local_random, **kwargs)
            parallel_thinking_result = output_dict["parallel_thinking_result"]
        else:
            # GenSynthesis
            output_dict = await self._run_gensynthesis(prompt_str, solutions, local_random)
            parallel_thinking_result = output_dict["parallel_thinking_result"]

        result[f"{self.cfg.mode}_comparison"] = parallel_thinking_result["generation"]
        result[f"{self.cfg.mode}_successful"] = parallel_thinking_result["generation_successful"]
        result[f"{self.cfg.mode}_num_generated_tokens"] = parallel_thinking_result.get("num_generated_tokens", 0)

        # Add the tokens for all the solutions and parallel thinking
        total_gen_tokens = result["total_solution_generated_tokens"] + result[f"{self.cfg.mode}_num_generated_tokens"]

        # TODO: Decide what count of generated tokens do we want to report - the total or the best solution?
        # Current implementation returns the total number of generated tokens
        result["num_generated_tokens"] = total_gen_tokens
        if self.cfg.count_prompt_tokens:
            result["num_input_tokens"] = parallel_thinking_result["num_input_tokens"]

        result[self.cfg.solution_key] = output_dict[self.cfg.solution_key]
        result["solution_list"] = [solution[self.cfg.solution_key] for solution in solutions]

        if self.cfg.solution_key != "generation":
            # Add the generation key to the result since it's required by inference/generate.py
            # We're just copying the solution key to the generation key to avoid errors
            result["generation"] = result[self.cfg.solution_key]

        return result

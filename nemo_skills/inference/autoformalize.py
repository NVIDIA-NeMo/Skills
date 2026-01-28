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
import logging
import sys
from dataclasses import asdict, is_dataclass
from typing import List

import hydra
from openai import BadRequestError

from nemo_skills.code_execution.proof_utils import (
    extract_code,
    move_imports_to_beginning,
    refine_by_sorry,
    remove_comments,
)
from nemo_skills.code_execution.sandbox import sandbox_params
from nemo_skills.inference.model import server_params
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    parse_reasoning,
    setup_logging,
)

from .generate import GenerationTask, GenerationTaskConfig

LOG = logging.getLogger(get_logger_name(__file__))

reasoning_effort_list = ["low", "medium", "high"]


@nested_dataclass(kw_only=True)
class AutoformalizeConfig(GenerationTaskConfig):
    """LLM generation parameters."""

    # Lean 4 specific parameters
    refine_parsing_error_prompt_config: str | None = None  # prompt for refining the code
    refine_code_error_prompt_config: str | None = None  # prompt for refining the code
    refine_consistent_error_prompt_config: str | None = None  # prompt for refining the code
    refinement: bool = False  # whether to refine the code
    refinement_max_turns: int = 8  # maximum number of turns for refinement
    judge_enabled: bool = False  # whether to judge the code
    backtranslation_prompt_config: str | None = None  # prompt for backtranslation
    judge_prompt_config: str | None = None  # prompt for judging the code
    judge_exact_match: bool = (
        True  # recommend to set to true when using gpt-oss and should set to false if using deepseek
    )
    adaptive_reasoning: bool = False  # whether to adapt the reasoning effort
    parse_generation: bool = False  # whether to parse the generation


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_generation_config", node=AutoformalizeConfig)


class AutoformalizeTask(GenerationTask):
    def __init__(self, cfg: AutoformalizeConfig):
        """
        Class that represents a generation task. It implements a template of steps to generate solutions using LLMs.
        Individual functions can be overriden to customize the behavior of the generation task.

        Args:
            cfg: AutoformalizeConfig object with the configuration parameters or subclass.
        """
        super().__init__(cfg)
        if self.cfg.refinement:
            self.setup_refine_prompt()
        if self.cfg.judge_enabled:
            self.setup_judge_prompt()

    def setup_llm(self):
        if self.cfg.code_execution:
            raise ValueError(
                "Code execution is not supported for autoformalization. Use sandbox config for Lean4 execution."
            )
        llm = super().setup_llm()
        # Validate sandbox is configured - fail early during setup rather than during generation
        if self.sandbox is None:
            raise ValueError(
                "Sandbox is required for Lean4 code execution but was not configured. "
                "Please provide sandbox configuration."
            )
        return llm

    def setup_refine_prompt(self):
        assert self.cfg.refine_parsing_error_prompt_config is not None, (
            "refine_parsing_error_prompt_config is required when refinement is enabled. Please set refinement=False to disable refinement."
        )
        assert self.cfg.refine_code_error_prompt_config is not None, (
            "refine_code_error_prompt_config is required when refinement is enabled. Please set refinement=False to disable refinement."
        )
        self.refine_parsing_error_prompt = get_prompt(self.cfg.refine_parsing_error_prompt_config)
        self.refine_code_error_prompt = get_prompt(self.cfg.refine_code_error_prompt_config)
        if self.cfg.judge_enabled:
            assert self.cfg.refine_consistent_error_prompt_config is not None, (
                "refine_consistent_error_prompt_config is required when refinement is enabled and judge is enabled. Please set refinement=False to disable refinement."
            )
            self.refine_consistent_error_prompt = get_prompt(self.cfg.refine_consistent_error_prompt_config)

    def setup_judge_prompt(self):
        assert self.cfg.backtranslation_prompt_config is not None, (
            "backtranslation_prompt_config is required when judge is enabled. Please set judge_enabled=False to disable judge."
        )
        assert self.cfg.judge_prompt_config is not None, (
            "judge_prompt_config is required when judge is enabled. Please set judge_enabled=False to disable judge."
        )
        self.judge_prompt = get_prompt(self.cfg.judge_prompt_config)
        self.backtranslation_prompt = get_prompt(self.cfg.backtranslation_prompt_config)

    def _extract_code_sync(self, completion: str):
        try:
            code = extract_code(completion)
            if code == "None":
                return None, None
            clean_code = remove_comments(code)
            clean_code = move_imports_to_beginning(clean_code)
            clean_code = refine_by_sorry(clean_code)
        except (ValueError, TypeError, AttributeError) as e:
            LOG.debug("Code extraction failed: %s", e)
            return None, None
        else:
            return code, clean_code

    async def _extract_code(self, completion: str):
        # Offload the blocking work to another thread
        return await asyncio.to_thread(self._extract_code_sync, completion)

    async def _backtranslate_code(self, code: str) -> str:
        prompt = self.backtranslation_prompt.fill({"code": code})
        generation = await self._generate_single_completion(prompt)
        return generation.get("generation")

    async def _judge_backtranslation(self, backtranslation_result: str, data_point) -> str:
        prompt = self.judge_prompt.fill(
            {
                "backtranslation": backtranslation_result,
                "problem": data_point["problem"],
            }
        )
        generation = await self._generate_single_completion(prompt)
        return generation.get("generation")

    async def _judge_code(self, code: str | None, data_point) -> dict:
        results_dict = {}
        results_dict["code"] = code
        results_dict["passed_compile"] = False
        results_dict["backtranslation_result"] = None
        results_dict["judge_result"] = None
        results_dict["passed_compile_judge"] = False
        results_dict["feedback"] = None
        if code is None:
            results_dict["parse_error"] = True
            return results_dict
        else:
            results_dict["parse_error"] = False

        # execute_code returns (result_dict, session_id) tuple
        code_execution_result, _ = await self.sandbox.execute_code(
            remove_comments(code), language="lean4", timeout=600.0, max_output_characters=1000000
        )
        results_dict["code_execution_result"] = code_execution_result

        # Handle timeout (now indicated by process_status in the dict)
        if code_execution_result.get("process_status") == "timeout":
            results_dict["code_execution_result"] = {
                "process_status": "failed",
                "stdout": "Timeout error, please check for heavy computation, dead loop, etc.",
            }
        elif code_execution_result["process_status"] == "completed":
            results_dict["passed_compile"] = True
            if self.cfg.judge_enabled:
                backtranslation_result = await self._backtranslate_code(code)
                if backtranslation_result is not None:
                    results_dict["backtranslation_result"] = backtranslation_result
                    judge_result = await self._judge_backtranslation(backtranslation_result, data_point)
                    results_dict["judge_result"] = judge_result
                    if judge_result is not None:
                        if self.cfg.judge_exact_match:
                            if "true" == judge_result.lower().strip():
                                results_dict["passed_compile_judge"] = True
                        else:
                            if "true" in judge_result.lower().strip():
                                results_dict["passed_compile_judge"] = True
                    else:
                        LOG.warning("Judge failed but code compiled successfully")
                        results_dict["passed_compile_judge"] = False
                        results_dict["judge_result"] = "Backtranslation passed, but judge failed."
                else:
                    LOG.warning("Backtranslation failed for compiled code")
                    results_dict["backtranslation_result"] = "Backtranslation failed."
                    results_dict["passed_compile_judge"] = False
            else:
                results_dict["passed_compile_judge"] = True
        return results_dict

    def _construct_refine_prompt(self, results_dict):
        if results_dict["parse_error"]:
            # parse error
            prompt = self.refine_parsing_error_prompt.fill({})
        elif results_dict["passed_compile"]:
            # consistent error
            prompt = self.refine_consistent_error_prompt.fill({"reason": results_dict["judge_result"]})
        else:
            # code error
            prompt = self.refine_code_error_prompt.fill(
                {"error_message": results_dict["code_execution_result"]["stdout"]}
            )
        return prompt

    async def _generate_single_completion(self, prompt: List[str]):
        """Generate a single completion with semaphore-controlled concurrency."""
        if is_dataclass(self.cfg.inference):
            inference_params = asdict(self.cfg.inference)
        else:
            # Already a dict from Hydra
            inference_params = dict(self.cfg.inference)
        generation_params = {
            "prompt": prompt,
            "stop_phrases": [self.cfg.stop_phrase] if self.cfg.stop_phrase else None,
            **inference_params,
            **self.extra_generate_params,
        }

        # Use semaphore for concurrency control (inherited from GenerationTask)
        async with self.semaphore:
            generation = await self.llm.generate_async(**generation_params)
            if self.cfg.adaptive_reasoning:
                assert generation_params["extra_body"].get("reasoning_effort", None) is not None, (
                    "reasoning_effort is required when adaptive_reasoning is enabled"
                )
                reasoning_effort_index = reasoning_effort_list.index(
                    generation_params["extra_body"].get("reasoning_effort", None)
                )
                while len(generation["generation"]) == 0 and reasoning_effort_index > 0:
                    LOG.info(
                        "Reasoning effort is too high, reducing to %s",
                        reasoning_effort_list[reasoning_effort_index - 1],
                    )
                    reasoning_effort_index = reasoning_effort_index - 1
                    generation_params["extra_body"]["reasoning_effort"] = reasoning_effort_list[reasoning_effort_index]
                    generation = await self.llm.generate_async(**generation_params)

        if self.cfg.parse_generation:
            parse_reasoning(
                generation,
                self.cfg.generation_key,
                self.cfg.end_reasoning_string,
            )
        return generation

    async def _single_data_point_generate(self, data_point, data):
        results_dict = {}
        prompt_turn_list = self.fill_prompt(data_point, data)
        code_list = []
        unrefined_code_list = []
        results_dict_list = []
        assert isinstance(prompt_turn_list, list), "prompt_turn_list should be a list"
        results_dict["passed_compile_judge"] = False
        turn_idx = 0

        try:
            for turn_idx in range(self.cfg.refinement_max_turns):
                generation = await self._generate_single_completion(prompt_turn_list)
                prompt_turn_list += [{"role": "assistant", "content": generation["generation"]}]
                unrefined_code, code = await self._extract_code(generation["generation"])
                unrefined_code_list.append(unrefined_code)
                code_list.append(code)
                results_dict = await self._judge_code(code, data_point)
                if "reasoning_content" in generation:
                    results_dict["reasoning_content_generation"] = generation["reasoning_content"]
                results_dict_list.append(results_dict)
                if results_dict["passed_compile_judge"]:
                    break
                else:
                    if self.cfg.refinement and turn_idx < self.cfg.refinement_max_turns - 1:
                        prompt = self._construct_refine_prompt(results_dict)
                        results_dict["feedback"] = prompt
                        prompt_turn_list += prompt
                    else:
                        break
        except BadRequestError as e:
            LOG.warning("BadRequestError: %s", e)
        return {
            "code_list": code_list,
            "unrefined_code_list": unrefined_code_list,
            "results_dict_list": results_dict_list,
            "prompt_turn_list": prompt_turn_list,
            "turn_idx": turn_idx,
            "success": results_dict["passed_compile_judge"],
        }

    async def process_single_datapoint(self, data_point, all_data):
        result = await self._single_data_point_generate(data_point, all_data)
        result_dict = {"generation": result}
        return result_dict


GENERATION_TASK_CLASS = AutoformalizeTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name="base_generation_config")
def generate(cfg: AutoformalizeConfig):
    cfg = AutoformalizeConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = AutoformalizeTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    AutoformalizeConfig,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()

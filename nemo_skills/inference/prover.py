# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import List

import hydra

from nemo_skills.code_execution.sandbox import get_sandbox, sandbox_params
from nemo_skills.inference.model import get_model, server_params
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    remove_thinking,
    setup_logging,
)

from .generate import GenerateSolutionsConfig, GenerationTask
from .lean4_utils import *

LOG = logging.getLogger(get_logger_name(__file__))

reasoning_effort_list = [
    "low",
    "medium",
    "high",
]  # This is only used for adaptive reasoning with gpt-oss models


@nested_dataclass(kw_only=True)
class ProverConfig(GenerateSolutionsConfig):
    max_tokens: int = 40960  # model max tokens
    n_pass: int = 1  # number of passes to run the prover

    # Lean 4 specific parameters
    refinement: bool = False  # whether to refine the code
    refinement_max_turns: int = 2  # maximum number of turns for refinement
    refinement_prompt_config: str | None = None  # prompt for refining the code
    adaptive_reasoning: bool = False  # whether to adapt the reasoning effort
    parse_generation: bool = False  # whether to parse the generation
    remove_cot: bool = False  # whether to remove the cot from the generation
    # whether to delete the wrong turns from the generation
    delete_wrong_turns: bool = False

    def _post_init_validate_params(self):
        """Validate that certain parameters are restricted to certain values"""
        if self.prompt_format not in ["ns", "openai"]:
            raise ValueError(f"prompt_format must be either 'ns' or 'openai', got '{self.prompt_format}'")

        if self.prompt_format == "openai":
            assert self.prompt_config is None, "prompt_config is not supported for prompt_format == 'openai'"
        else:
            assert self.prompt_config is not None, "prompt_config is required when prompt_format == 'ns'"
        for param, default_value in self._get_disallowed_params():
            if getattr(self, param) != default_value:
                raise ValueError(f"{param} must be {default_value}")

        if self.n_pass > 32:
            raise ValueError("Please consider using num_random_seeds instead")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_prover_config", node=ProverConfig)


class ProverTask(GenerationTask):
    def __init__(self, cfg: ProverConfig):
        """
        Class that represents a generation task. It implements a template of steps to generate solutions using LLMs.
        Individual functions can be overriden to customize the behavior of the generation task.

        Args:
            cfg: GenerateSolutionsConfig object with the configuration parameters or subclass.
        """
        self.cfg = cfg
        # chat template kwargs goes either into extra body of inference or as a prompt parameter

        if self.cfg.chat_template_kwargs:
            if not self.cfg.use_completions_api:
                if "chat_template_kwargs" in self.cfg.inference.extra_body:
                    raise ValueError(
                        "chat_template_kwargs is provided in both inference.extra_body and as a separate argument. "
                        "You can only use one of them!"
                    )
                self.cfg.inference.extra_body = dict(self.cfg.inference.extra_body)
                self.cfg.inference.extra_body["chat_template_kwargs"] = dict(self.cfg.chat_template_kwargs)
                self.cfg.chat_template_kwargs = None

        self.llm = self.setup_llm()
        self.prompt = self.setup_prompt()
        if self.cfg.refinement:
            self.setup_refine_prompt()

        if self.cfg.code_execution:
            self.extra_generate_params = self.prompt.get_code_execution_args()
        else:
            self.extra_generate_params = {}

        LOG.info(
            "Async loop is maintaining %d generations in parallel. "
            "Use max_concurrent_requests to control the number of concurrent requests.",
            self.cfg.max_concurrent_requests,
        )

        self.semaphore = asyncio.Semaphore(self.cfg.max_concurrent_requests)

        # output_lock will be initialized when async_loop is called
        self.output_lock = None

        if self.cfg.delete_wrong_turns:
            assert self.cfg.remove_cot, "remove_cot is required when delete_wrong_turns is enabled"

    def log_example_prompt(self, data):
        return

    def setup_llm(self):
        if self.cfg.code_execution:
            raise ValueError("Code execution is not supported for prover")
        sandbox = get_sandbox(**self.cfg.sandbox) if self.cfg.sandbox is not None else None
        server = deepcopy(self.cfg.server)
        server["server_type"] = "autoformalization"
        llm = get_model(**server, sandbox=sandbox)
        return llm

    def setup_prompt(self):
        if self.cfg.prompt_format == "openai":
            return None
        if self.cfg.use_completions_api:
            tokenizer = self.cfg.tokenizer or self.cfg.server["model"]
        else:
            tokenizer = None
        prompt = get_prompt(
            prompt_config=self.cfg.prompt_config,
            tokenizer=tokenizer,
            code_tags=self.cfg.code_tags,
            examples_type=self.cfg.examples_type,
        )
        if self.cfg.system_message is not None:
            prompt.config.system = self.cfg.system_message
        LOG.info("Prompt used: %s", prompt)
        return prompt

    def setup_refine_prompt(self):
        assert (
            self.cfg.refinement_prompt_config is not None
        ), "refinement_prompt_config is required when refinement is enabled. Please set refinement=False to disable refinement."
        self.refine_prompt = get_prompt(self.cfg.refinement_prompt_config)

    # with adaptive reasoning
    async def _generate_single_completion(self, prompt: List[str], **kwargs):
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
        for key, value in kwargs.items():
            generation_params[key] = value
        generation = await self.llm.generate_async(**generation_params)
        if self.cfg.adaptive_reasoning:
            assert (
                generation_params["extra_body"].get("reasoning_effort", None) is not None
            ), "reasoning_effort is required when adaptive_reasoning is enabled"
            reasoning_effort_index = reasoning_effort_list.index(
                generation_params["extra_body"].get("reasoning_effort", None)
            )
            while len(generation["generation"]) == 0 and reasoning_effort_index > 0:
                print(f"Reasoning effort is too high, reducing to {reasoning_effort_list[reasoning_effort_index - 1]}")
                reasoning_effort_index = reasoning_effort_index - 1
                generation_params["extra_body"]["reasoning_effort"] = reasoning_effort_list[reasoning_effort_index]
                generation = await self.llm.generate_async(**generation_params)
        if self.cfg.parse_generation:
            remove_thinking(
                generation,
                self.cfg.generation_key,
                self.cfg.thinking_begin,
                self.cfg.thinking_end,
            )
        return generation

    # factor out his part so it won't become a bottleneck.
    async def _extract_and_replace_code(self, formal_statement, generation):
        code = extract_code(generation)
        full_code = replace_statement_in_proof(formal_statement, code)
        return code, full_code

    async def _signle_data_point_generate(self, data_point, data):
        formal_statement = (
            (data_point["header"].strip() + "\n")
            + data_point["informal_prefix"].strip()
            + ("\n" + data_point["formal_statement"].strip())
        )
        formal_statement = refine_by_sorry(formal_statement)
        prompt_turn_list = self.prompt.fill({"problem": formal_statement.strip()})

        full_prompt_turn_list = deepcopy(
            prompt_turn_list
        )  # We need to get a full copy of the prompt turn list for the final result in case remove_cot is enabled. This is only used to generate SFT data.
        promt_turn_list_list = (
            []
        )  # We need to store the prompt turn list for each turn for the final result in case delete_wrong_turns is enabled. This is only used to generate SFT data.
        base_prompt_turn_list = deepcopy(prompt_turn_list)

        code_list = []
        results_dict_list = []
        assert type(prompt_turn_list) == list, "prompt_turn_list should be a list"

        success = False
        for turn_idx in range(self.cfg.refinement_max_turns):
            results_dict = {}  # everything will be stored in this dict
            prefix_tokens = self.llm.tokenizer.apply_chat_template(
                prompt_turn_list, tokenize=True, add_generation_prompt=True
            )
            num_tokens_prefix = len(prefix_tokens)
            prefix = self.llm.tokenizer.apply_chat_template(
                prompt_turn_list, tokenize=False, add_generation_prompt=True
            )
            # We need to check if the prefix is too long, if it is, we need to break the loop
            if num_tokens_prefix > self.cfg.max_tokens:
                break

            generation = await self._generate_single_completion(
                prefix,
                tokens_to_generate=min(
                    self.cfg.max_tokens - num_tokens_prefix,
                    self.cfg.inference.tokens_to_generate,
                ),
            )

            new_prompt_turn_list = deepcopy(prompt_turn_list)
            new_prompt_turn_list += [{"role": "assistant", "content": generation["generation"]}]

            promt_turn_list_list.append(
                new_prompt_turn_list
            )  # This stores the latest turn list after each generation.

            code, full_code = await self._extract_and_replace_code(formal_statement, generation["generation"])
            code_list.append(full_code)
            results_dict["code"] = code  # We keep track of the uncleaned code.
            if self.cfg.remove_cot and not (
                code == "None" or "**Error**" in full_code
            ):  # check if successfully parse the code. We do not want to delete the turn if there is a parsing error.
                if self.cfg.delete_wrong_turns:
                    prompt_turn_list = deepcopy(base_prompt_turn_list) + [
                        {
                            "role": "assistant",
                            "content": f"```lean4\n{full_code.strip()}\n```",
                        }
                    ]  # only keep the latest turn
                else:
                    prompt_turn_list += [
                        {
                            "role": "assistant",
                            "content": f"```lean4\n{full_code.strip()}\n```",
                        }
                    ]
                full_prompt_turn_list += [{"role": "assistant", "content": generation["generation"]}]
            else:
                prompt_turn_list += [{"role": "assistant", "content": generation["generation"]}]
                full_prompt_turn_list += [{"role": "assistant", "content": generation["generation"]}]

            if code == "None" or "**Error**" in full_code:
                if code == "None":
                    execution_result = {
                        "process_status": "failed",
                        "stderr": "",
                        "stdout": "Parsing error. Cannot parse the code from output. Please try again and write the code in the format of ```lean4\n<code>\n```",
                    }
                elif "**Error**" in full_code:
                    execution_result = {
                        "process_status": "failed",
                        "stderr": "",
                        "stdout": full_code,
                    }
                results_dict["execution_result"] = execution_result
                results_dict["success"] = False
                feedback = self.refine_prompt.fill({"error_message": execution_result["stdout"]})
                results_dict["feedback"] = feedback[0]["content"]
            else:
                execution_result = await self.llm.sandbox.execute_lean4_code(
                    full_code, timeout=600.0, max_output_characters=1000000
                )
                results_dict["execution_result"] = execution_result
                if isinstance(execution_result, dict):
                    if (
                        execution_result["process_status"] == "completed"
                        and "sorry" not in execution_result["stdout"]
                        and "failed" not in execution_result["stdout"]
                    ):
                        results_dict["success"] = True
                    else:
                        error_list = parse_error(execution_result["stdout"])
                        error_message = get_error_str(full_code, error_list, error_thres=True)
                        # checking for sorry
                        if execution_result["process_status"] == "completed":
                            stdout = execution_result["stdout"].lower()
                            stderr = execution_result["stderr"].lower()
                            combined = stdout + "\n" + stderr
                            if re.search(r"\bsorry\b", combined) is not None:
                                error_message += "\nThe code contains 'sorry', which means the proof is incomplete."
                        feedback = self.refine_prompt.fill(
                            {
                                "error_message": "We use <error></error> to signal the position of the error. \n"
                                + error_message
                            }
                        )
                        results_dict["feedback"] = feedback[0]["content"]
                        results_dict["success"] = False
                # This is only used for the case when the code execution timed out.
                elif isinstance(execution_result, str):
                    execution_result = {
                        "process_status": "failed",
                        "stderr": "",
                        "stdout": execution_result,
                    }
                    results_dict["success"] = False
                    feedback = self.refine_prompt.fill(
                        {
                            "error_message": "The compilation timed out. There might be a heavy computation in the code or an endless loop."
                        }
                    )
                    results_dict["feedback"] = feedback[0]["content"]
                else:
                    raise ValueError(f"Unknown execution result type: {type(execution_result)}")

            results_dict_list.append(results_dict)

            if results_dict["success"]:
                # This is the case when the code execution is successful. The theorem is proved.
                break
            else:
                if self.cfg.refinement and turn_idx < self.cfg.refinement_max_turns - 1:
                    prompt_turn_list += feedback
                    full_prompt_turn_list += feedback
                else:
                    # Proving attempt failed.
                    break

        if len(results_dict_list) > 0 and results_dict_list[-1]["success"]:
            success = True

        # Usually only need prompt_turn_list for standard SFT, full_prompt_turn_list for SFT with remove_cot enabled, promt_turn_list_list for SFT with delete_wrong_turns enabled.
        return {
            "code_list": code_list,
            "results_dict_list": results_dict_list,
            "prompt_turn_list": prompt_turn_list,
            "turn_idx": turn_idx,
            "success": success,
            "full_prompt_turn_list": full_prompt_turn_list,
            "promt_turn_list_list": promt_turn_list_list,
        }

    async def pass_at_N(self, data_point, data, N=None):
        if N is None:
            N = self.cfg.n_pass

        new_results_dict = {"success": False}
        # results_dict_list = []
        for i in range(N):
            results_dict = await self._signle_data_point_generate(data_point, data)
            # results_dict_list.append(results_dict)

            if results_dict["success"]:
                new_results_dict["success"] = True
                break

        new_results_dict["results_dict_list"] = results_dict
        new_results_dict["n_pass"] = i + 1

        return new_results_dict

    async def process_single_datapoint(self, data_point, all_data):
        result = await self.pass_at_N(data_point, all_data)
        result_dict = {"generation": result}

        return result_dict


GENERATION_TASK_CLASS = ProverTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name="base_prover_config")
def generate(cfg: ProverConfig):
    cfg = ProverConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = ProverTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    ProverConfig,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()

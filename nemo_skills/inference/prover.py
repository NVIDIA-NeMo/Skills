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

import logging
import re
import sys
from copy import deepcopy
from dataclasses import asdict, is_dataclass

import hydra
from transformers import AutoTokenizer

from nemo_skills.code_execution.proof_utils import (
    extract_code,
    get_error_str,
    parse_error,
    refine_by_sorry,
    replace_statement_in_proof,
)
from nemo_skills.code_execution.sandbox import sandbox_params
from nemo_skills.inference.model import server_params
from nemo_skills.inference.model.base import EndpointType
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

reasoning_effort_list = [
    "low",
    "medium",
    "high",
]  # This is only used for adaptive reasoning with gpt-oss models


@nested_dataclass(kw_only=True)
class ProverConfig(GenerationTaskConfig):
    max_tokens: int = 40960  # model max tokens
    n_pass: int = 1  # number of passes to run the prover

    # Lean 4 specific parameters
    nemotron_refinement: bool = False  # whether to use single-turn nemotron-style refinement
    refinement: bool = False  # whether to refine the code
    refinement_max_turns: int = 2  # maximum number of turns for refinement
    refinement_prompt_config: str | None = None  # prompt for multi-turn refinement feedback
    # prompt for single-turn nemotron refinement (used when nemotron_refinement=True)
    nemotron_refinement_prompt_config: str | None = None
    adaptive_reasoning: bool = False  # whether to adapt the reasoning effort
    parse_generation: bool = False  # whether to parse the generation
    remove_cot: bool = False  # whether to remove the cot from the generation
    # whether to delete the wrong turns from the generation
    delete_wrong_turns: bool = False

    def _post_init_validate_params(self):
        """Validate that certain parameters are restricted to certain values"""
        if self.prompt_format == "openai":
            raise ValueError(
                "prompt_format='openai' is not supported for lean4_prover. Use prompt_format='ns' with a prompt_config."
            )
        if self.prompt_format != "ns":
            raise ValueError(f"prompt_format must be 'ns', got '{self.prompt_format}'")

        assert self.prompt_config is not None, "prompt_config is required for lean4_prover"

        for param, default_value in self._get_disallowed_params():
            if getattr(self, param) != default_value:
                raise ValueError(f"{param} must be {default_value}")

        if self.n_pass > 32:
            LOG.warning(
                "n_pass=%d exceeds recommended maximum of 32. Consider using num_random_seeds instead.", self.n_pass
            )


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_prover_config", node=ProverConfig)


class ProverTask(GenerationTask):
    def __init__(self, cfg: ProverConfig):
        """
        Class that represents a generation task. It implements a template of steps to generate solutions using LLMs.
        Individual functions can be overriden to customize the behavior of the generation task.

        Args:
            cfg: GenerationTaskConfig object with the configuration parameters or subclass.
        """
        super().__init__(cfg)

        # Initialize tokenizer for chat template application
        tokenizer_path = self.cfg.tokenizer or self.cfg.server.get("model")
        self.hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        if self.cfg.refinement:
            self.setup_refine_prompt()

        if self.cfg.delete_wrong_turns:
            assert self.cfg.remove_cot, "remove_cot is required when delete_wrong_turns is enabled"

    def log_example_prompt(self, data):
        return

    def setup_llm(self):
        if self.cfg.code_execution:
            raise ValueError("Code execution is not supported for prover. Use sandbox config for Lean4 execution.")
        return super().setup_llm()

    def setup_refine_prompt(self):
        assert self.cfg.refinement_prompt_config is not None, (
            "refinement_prompt_config is required when refinement is enabled. Please set refinement=False to disable refinement."
        )
        self.refine_prompt = get_prompt(self.cfg.refinement_prompt_config)

        if self.cfg.nemotron_refinement:
            assert self.cfg.nemotron_refinement_prompt_config is not None, (
                "nemotron_refinement_prompt_config is required when nemotron_refinement is enabled."
            )
            self.nemotron_refine_prompt = get_prompt(self.cfg.nemotron_refinement_prompt_config)

    async def _generate_single_completion(self, prompt: str, **kwargs):
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
        # Override endpoint_type to text since we already applied the chat template
        generation_params["endpoint_type"] = EndpointType.text
        for key, value in kwargs.items():
            generation_params[key] = value

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

    # factor out this part so it won't become a bottleneck.
    async def _extract_and_replace_code(self, formal_statement, generation):
        code = extract_code(generation)
        full_code = replace_statement_in_proof(formal_statement, code)
        return code, full_code

    def _transform_for_nemotron_refinement(self, proof_attempt: str, error_message: str) -> list[dict]:
        """Transform multi-turn refinement into single-turn nemotron-style prompt."""
        return self.nemotron_refine_prompt.fill(
            {
                "proof_attempt": proof_attempt,
                "error_message": error_message,
            }
        )

    def _parse_gpt_oss_output(self, content: str) -> tuple[str, str | None]:
        """Parse gpt-oss model output to extract thinking and final content.

        gpt-oss models output in the format:
        <|channel|>analysis<|message|>...thinking...<|end|><|start|>assistant<|channel|>final<|message|>...final...<|return|>

        The chat template expects analysis content in 'thinking' field and final content in 'content' field.

        Returns:
            tuple of (final_content, thinking_content or None)
        """
        import re

        # Check if the content contains gpt-oss channel tags
        if "<|channel|>" not in content:
            return content, None

        thinking = None
        final_content = content

        # Extract analysis/thinking content: between <|channel|>analysis<|message|> and <|end|>
        analysis_pattern = r"<\|channel\|>analysis[^<]*<\|message\|>(.*?)<\|end\|>"
        analysis_match = re.search(analysis_pattern, content, re.DOTALL)
        if analysis_match:
            thinking = analysis_match.group(1).strip()

        # Extract final content: after <|channel|>final<|message|> until <|return|> or end
        final_pattern = r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|$)"
        final_match = re.search(final_pattern, content, re.DOTALL)
        if final_match:
            final_content = final_match.group(1).strip()
        else:
            # If no final channel found, try to strip all channel tags and use what remains
            # This handles cases where the format might be slightly different
            final_content = re.sub(r"<\|[^|]+\|>", "", content).strip()

        return final_content, thinking

    def _make_assistant_message(self, content: str, reasoning_content: str | None = None) -> dict:
        """Create an assistant message dict, optionally with thinking/reasoning content.

        Some models (e.g., gpt-oss) output <|channel|> tags that need to be in a separate
        'thinking' field rather than in 'content' for the chat template to work correctly.

        If reasoning_content is not provided, attempts to parse it from content if the content
        contains gpt-oss channel tags.
        """
        # If reasoning_content not provided, try to parse from content
        if reasoning_content is None:
            content, reasoning_content = self._parse_gpt_oss_output(content)

        message = {"role": "assistant", "content": content}
        if reasoning_content:
            message["thinking"] = reasoning_content
        return message

    async def _single_data_point_generate(self, data_point, data):
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
        prompt_turn_list_list = []  # We need to store the prompt turn list for each turn for the final result in case delete_wrong_turns is enabled. This is only used to generate SFT data.
        base_prompt_turn_list = deepcopy(prompt_turn_list)

        code_list = []
        results_dict_list = []
        assert isinstance(prompt_turn_list, list), "prompt_turn_list should be a list"

        success = False
        turn_idx = 0
        last_proof_attempt = None  # Track for nemotron refinement
        last_error_message = None  # Track for nemotron refinement
        for turn_idx in range(self.cfg.refinement_max_turns):
            results_dict = {}  # everything will be stored in this dict
            if turn_idx != 0 and self.cfg.nemotron_refinement and last_proof_attempt and last_error_message:
                prepared_conversation = self._transform_for_nemotron_refinement(last_proof_attempt, last_error_message)
            else:
                prepared_conversation = prompt_turn_list
            prefix_tokens = self.hf_tokenizer.apply_chat_template(
                prepared_conversation, tokenize=True, add_generation_prompt=True
            )
            num_tokens_prefix = len(prefix_tokens)
            prefix = self.hf_tokenizer.apply_chat_template(
                prepared_conversation, tokenize=False, add_generation_prompt=True
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

            # Get reasoning_content if available (e.g., from gpt-oss models)
            reasoning_content = generation.get("reasoning_content")

            new_prompt_turn_list = deepcopy(prompt_turn_list)
            new_prompt_turn_list.append(self._make_assistant_message(generation["generation"], reasoning_content))

            prompt_turn_list_list.append(
                new_prompt_turn_list
            )  # This stores the latest turn list after each generation.

            code, full_code = await self._extract_and_replace_code(formal_statement, generation["generation"])
            last_proof_attempt = generation["generation"]  # Track for nemotron refinement
            code_list.append(full_code)
            results_dict["code"] = code  # We keep track of the uncleaned code.
            if self.cfg.remove_cot and not (
                code == "None" or "**Error**" in full_code
            ):  # check if successfully parse the code. We do not want to delete the turn if there is a parsing error.
                if self.cfg.delete_wrong_turns:
                    prompt_turn_list = deepcopy(base_prompt_turn_list) + [
                        self._make_assistant_message(f"```lean4\n{full_code.strip()}\n```")
                    ]  # only keep the latest turn
                else:
                    prompt_turn_list.append(self._make_assistant_message(f"```lean4\n{full_code.strip()}\n```"))
                full_prompt_turn_list.append(self._make_assistant_message(generation["generation"], reasoning_content))
            else:
                assistant_msg = self._make_assistant_message(generation["generation"], reasoning_content)
                prompt_turn_list.append(assistant_msg)
                full_prompt_turn_list.append(assistant_msg)

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
                else:
                    execution_result = {
                        "process_status": "failed",
                        "stderr": "",
                        "stdout": "Unknown error when parsing code.",
                    }
                results_dict["execution_result"] = execution_result
                results_dict["success"] = False
                last_error_message = execution_result["stdout"]  # Track for nemotron refinement
                feedback = self.refine_prompt.fill({"error_message": last_error_message})
                results_dict["feedback"] = feedback[0]["content"]
            else:
                if self.sandbox is None:
                    raise RuntimeError(
                        "Sandbox is required for Lean4 code execution but was not configured. "
                        "Please provide sandbox configuration."
                    )
                # execute_code returns (result_dict, session_id) tuple
                execution_result, _ = await self.sandbox.execute_code(
                    full_code, language="lean4", timeout=600.0, max_output_characters=1000000
                )
                results_dict["execution_result"] = execution_result
                # Handle timeout (now indicated by process_status in the dict)
                if execution_result.get("process_status") == "timeout":
                    results_dict["success"] = False
                    last_error_message = (
                        "The compilation timed out. There might be a heavy computation in the code or an endless loop."
                    )
                    feedback = self.refine_prompt.fill({"error_message": last_error_message})
                    results_dict["feedback"] = feedback[0]["content"]
                elif (
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
                    if error_message.strip() == "":  # something in stderr indicating failure
                        error_message = execution_result["stderr"][:1000]
                        if len(execution_result["stderr"]) > 1000:
                            error_message += "... (truncated)"

                    last_error_message = (
                        "We use <error></error> to signal the position of the error. \n" + error_message
                    )
                    feedback = self.refine_prompt.fill({"error_message": last_error_message})
                    results_dict["feedback"] = feedback[0]["content"]
                    results_dict["success"] = False

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

        # Usually only need prompt_turn_list for standard SFT, full_prompt_turn_list for SFT with remove_cot enabled, prompt_turn_list_list for SFT with delete_wrong_turns enabled.
        return {
            "code_list": code_list,
            "results_dict_list": results_dict_list,
            "prompt_turn_list": prompt_turn_list,
            "turn_idx": turn_idx,
            "success": success,
            "full_prompt_turn_list": full_prompt_turn_list,
            "prompt_turn_list_list": prompt_turn_list_list,
        }

    async def pass_at_N(self, data_point, data, N=None):
        if N is None:
            N = self.cfg.n_pass

        new_results_dict = {"success": False}
        for i in range(N):
            results_dict = await self._single_data_point_generate(data_point, data)

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

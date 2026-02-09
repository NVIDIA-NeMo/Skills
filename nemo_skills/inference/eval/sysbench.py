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
import json
import logging
import sys
from dataclasses import asdict, field

import hydra

from nemo_skills.inference.generate import (
    GenerationTask,
    GenerationTaskConfig,
    InferenceConfig,
)
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    parse_reasoning,
    setup_logging,
)

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class SysBenchGenerationConfig(GenerationTaskConfig):
    """SysBench multi-turn benchmark generation."""

    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    # Inference server configuration
    server: dict = field(default_factory=dict)

    def _post_init_validate_params(self):
        """Validate that certain parameters are restricted to certain values"""
        if self.prompt_format not in ["openai"]:
            raise ValueError(f"prompt_format must be 'openai' for SysBench, got '{self.prompt_format}'")

        if self.prompt_format == "openai":
            assert self.prompt_config is None, "prompt_config is not supported for prompt_format == 'openai'"

    def _get_disallowed_params(self):
        """Returns a list of parameters with their default values to check that they are not changed from the defaults"""
        return [
            ("prompt_config", None),
        ]


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_sysbench_generation_config", node=SysBenchGenerationConfig)


class SysBenchGenerationTask(GenerationTask):
    """SysBench multi-turn generation task.
    
    This task handles the multi-turn nature of SysBench:
    1. For each dialogue, iterate through user messages
    2. Generate assistant responses turn by turn
    3. Build history blocks for judging
    4. Format criteria for each turn
    """

    def __init__(self, cfg: SysBenchGenerationConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def log_example_prompt(self, data):
        """SysBench is a multi-turn benchmark, so we can't print a single prompt."""
        return

    def setup_prompt(self):
        """SysBench uses OpenAI chat format, no prompt template needed."""
        return None

    def _build_history_block(self, messages, current_turn_idx):
        """Build the history block for judge prompt (all turns before current).
        
        Args:
            messages: List of messages (system, user, assistant alternating)
            current_turn_idx: Index of the current assistant message being judged
        
        Returns:
            Formatted history string for judge prompt
        """
        history_parts = []
        round_num = 1
        
        # Iterate through pairs of user/assistant messages before the current turn
        # messages[0] is system, messages[1] is first user, messages[2] is first assistant, etc.
        for i in range(1, current_turn_idx, 2):
            if i + 1 <= current_turn_idx:
                user_msg = messages[i]
                assistant_msg = messages[i + 1] if i + 1 < current_turn_idx else None
                
                if assistant_msg:
                    history_parts.append(f"""<round-{round_num}>
<role:>
{user_msg["role"]}
</role>
<content>
{user_msg["content"]}
</content>
<role:>{assistant_msg["role"]}</role>
<content>
{assistant_msg["content"]}
</content>
</round-{round_num}>""")
                    round_num += 1
        
        return "\n\n".join(history_parts) if history_parts else ""

    def _format_criteria(self, criteria_dict):
        """Format criteria dict into the required string format for judge prompt.
        
        Args:
            criteria_dict: Dict mapping criterion ID to criterion info
        
        Returns:
            Formatted criteria string
        """
        criteria_lines = []
        for crit_id, crit_info in criteria_dict.items():
            criteria_lines.append(
                f"{crit_id}. {crit_info['criteria_content']} | {crit_info['criteria_type']}"
            )
        return "\n".join(criteria_lines)

    async def _generate_single_assistant_turn(self, messages):
        """Generate a single assistant response given the conversation history.
        
        Args:
            messages: List of message dicts in OpenAI format
            
        Returns:
            Dict with generated message and metadata
        """
        # Construct input dict for generation
        # SysBench messages already include the system message, so only prepend if not present
        if self.cfg.system_message and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": self.cfg.system_message}] + messages
        
        input_dict = {
            "prompt": messages,
            "include_response": True,
            **asdict(self.cfg.inference),
        }
        
        return_dict = {}
        if self.cfg.count_prompt_tokens:
            from nemo_skills.prompt.utils import get_token_count
            num_input_tokens = get_token_count(self.hf_tokenizer, messages=messages)
            return_dict["num_input_tokens"] = num_input_tokens
        
        # Query the LLM server
        try:
            output = await self.generate_with_semaphore(**input_dict)
        except Exception as error:
            from nemo_skills.inference.model.utils import is_context_window_exceeded_error
            if is_context_window_exceeded_error(error):
                LOG.warning(f"SysBench generation failed due to running out of context: {error}")
                return None
            else:
                raise error
        
        # Extract the response
        return_dict["num_generated_tokens"] = output.get("num_generated_tokens", 0)
        
        # Parse the generation output
        if "response" in output:
            # Server-side response
            response_message = output["response"].choices[0].message
            content = response_message.content
        elif "generation" in output:
            # Client-side response
            content = output["generation"]
        else:
            LOG.error("Unexpected output format from generation")
            return None
        
        return_dict["content"] = content
        return return_dict

    async def _generate_single_data_point_multi_turn(self, data_point):
        """Generate each turn of a SysBench dialogue.
        
        Args:
            data_point: Dict containing system_id, messages, prompt_infos, etc.
        
        Returns:
            List of dicts, one per turn, ready for judging
        """
        system_id = data_point["system_id"]
        messages_template = data_point["messages"]  # Original message template
        prompt_infos = data_point["prompt_infos"]  # Criteria for each user prompt
        domain = data_point.get("domain")
        scene = data_point.get("scene")
        
        # Build the actual conversation with generated assistant responses
        generated_messages = []
        turn_results = []
        
        # Add system message
        if messages_template and messages_template[0]["role"] == "system":
            generated_messages.append(messages_template[0])
        
        # Process each user/assistant turn
        for i in range(1, len(messages_template), 2):
            if i >= len(messages_template):
                break
            
            user_msg = messages_template[i]
            generated_messages.append(user_msg)
            
            # Generate assistant response
            assistant_response = await self._generate_single_assistant_turn(generated_messages)
            
            if assistant_response is None:
                LOG.warning(f"Failed to generate response for system_id {system_id}, turn {i // 2 + 1}")
                break
            
            # Parse reasoning tokens if configured (for reasoning models)
            raw_content = assistant_response["content"]
            if self.cfg.parse_reasoning:
                # Use parse_reasoning utility to extract non-thinking content
                temp_sample = {"generation": raw_content}
                parse_reasoning(temp_sample, "generation", self.cfg.end_reasoning_string)
                parsed_content = temp_sample["generation"]
                full_content = temp_sample.get("_full_generation", raw_content)
            else:
                parsed_content = raw_content
                full_content = raw_content
            
            # Add assistant message to conversation (use parsed content for history)
            assistant_msg = {
                "role": "assistant",
                "content": parsed_content
            }
            generated_messages.append(assistant_msg)
            
            # Build history block (all turns before current)
            history = self._build_history_block(generated_messages, len(generated_messages) - 1)
            
            # Get criteria for this user prompt
            user_prompt_content = user_msg["content"]
            prompt_info = prompt_infos[user_prompt_content]
            criteria = prompt_info["criteria"]
            formatted_criteria = self._format_criteria(criteria)
            alignment = prompt_info.get("alignment", "unknown")
            
            # Create a standalone result for this turn (for judge input)
            turn_result = {
                "system_id": system_id,
                "domain": domain,
                "scene": scene,
                "turn_idx": i // 2,
                "rounds_related": data_point.get("rounds_related", False),  # For SSR metric
                "alignment": alignment,  # For ISR metric (align/misalign)
                "system_prompt": generated_messages[0]["content"] if generated_messages[0]["role"] == "system" else "",
                "history": history,
                "current_user_role": user_msg["role"],
                "current_assistant_role": assistant_msg["role"],
                "current_user_content": user_prompt_content,  # Changed from final_user_prompt
                "current_assistant_content": parsed_content,  # Use parsed content (thinking removed if parse_reasoning=True)
                "criteria": formatted_criteria,
                "criteria_types": {k: v.get("criteria_type", "unknown") for k, v in criteria.items()},  # For CSR metric
                "prompt_infos": prompt_infos,  # Keep full prompt_infos for metrics
                "num_generated_tokens": assistant_response.get("num_generated_tokens", 0),
            }
            
            # Store full content with thinking for debugging (if different from parsed)
            if full_content != parsed_content:
                turn_result["_full_current_assistant_content"] = full_content
            
            if self.cfg.count_prompt_tokens:
                turn_result["num_input_tokens"] = assistant_response.get("num_input_tokens", 0)
            
            turn_results.append(turn_result)
        
        # Return list of turn results (each will be written as a separate line)
        return turn_results

    async def process_single_datapoint(self, data_point, all_data):
        """Process a single SysBench dialogue."""
        turn_results = await self._generate_single_data_point_multi_turn(data_point)
        # Wrap in a dict to satisfy parent class expectations
        return {"turn_results": turn_results, "system_id": data_point["system_id"]}
    
    async def postprocess_single_output(self, output, original_data_point):
        """Override to keep turn_results structure intact."""
        # Don't do the standard postprocessing - we'll handle it in dump_outputs
        # Just ensure generation_key is set for each turn if needed
        pass
    
    def dump_outputs(self, outputs, data_points, fout):
        """Override to write multiple lines per dialogue (one per turn)."""
        for output in outputs:
            # Check if this is a SysBench multi-turn output
            if "turn_results" in output:
                turn_results = output["turn_results"]
                generation_metadata = {
                    "generation_start_time": output.get("generation_start_time"),
                    "generation_end_time": output.get("generation_end_time"),
                    "generation_time": output.get("generation_time"),
                }
                
                # Write each turn as a separate line (without async_position since we handle ordering ourselves)
                for turn_result in turn_results:
                    turn_output = dict(turn_result)
                    
                    # Add generation metadata if configured
                    if self.cfg.add_generation_stats:
                        turn_output.update(generation_metadata)
                    
                    # Ensure generation_key is set
                    if "current_assistant_content" in turn_output:
                        turn_output[self.cfg.generation_key] = turn_output["current_assistant_content"]
                    
                    fout.write(json.dumps(turn_output) + "\n")
            else:
                # Standard single output
                fout.write(json.dumps(output) + "\n")
    
    def skip_completed_samples(self, data):
        """Override to handle multi-turn output when skipping completed samples.
        
        For SysBench, we can't use async_position since we write multiple lines per input.
        Instead, we check which system_ids have been completed.
        """
        from pathlib import Path
        import json
        
        if not self.cfg.skip_filled:
            return data
        
        output_file = Path(self.cfg.output_file)
        if not output_file.exists():
            return data
        
        # Collect completed system_ids
        completed_system_ids = set()
        with open(output_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    continue
                system_id = result.get("system_id")
                if system_id is not None:
                    completed_system_ids.add(system_id)
        
        if not completed_system_ids:
            return data
        
        # Filter out completed dialogues
        remaining_data = [d for d in data if d.get("system_id") not in completed_system_ids]
        
        LOG.info(f"Skipping {len(completed_system_ids)} completed dialogues, {len(remaining_data)} remaining")
        return remaining_data
    
    def restore_async_order(self):
        """Override to skip order restoration for SysBench.
        
        SysBench writes multiple lines per input dialogue, so we can't use the standard
        async position restoration (which assumes 1 output per input). The turns are already
        written in the correct order during generation.
        """
        # Simply rename the async file to the final output file
        import shutil
        from pathlib import Path
        
        async_file = Path(self.cfg.output_file + "-async")
        final_file = Path(self.cfg.output_file)
        
        if async_file.exists():
            shutil.move(str(async_file), str(final_file))
        
        self.cleanup_litellm_cache()


GENERATION_TASK_CLASS = SysBenchGenerationTask


@hydra.main(version_base=None, config_name="base_sysbench_generation_config")
def sysbench_generation(cfg: SysBenchGenerationConfig):
    cfg = SysBenchGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = SysBenchGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    SysBenchGenerationConfig,
    name="sysbench",
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        sysbench_generation()

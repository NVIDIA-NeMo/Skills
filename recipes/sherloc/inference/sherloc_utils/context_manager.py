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

"""Context window management for the SHERLOC localization agent.

The agent inspects a repository over many turns, so its dialogue grows with
every tool call. This module keeps that dialogue inside the model's context
window and intervenes when the agent stops making progress:

* proactive context checks before each generation
* a first-and-recent truncation strategy for the dialogue history
* injection of the final-turn instruction that requests the location submission
* detection of repeated tool calls and interventions that break such loops
* diagnosis of over-long responses

Token accounting here only sums counts already stored on a turn
(``_llm_tokens``, ``_tool_tokens``, ``_input_tokens``). This module does not
estimate those values. A missing count contributes zero. Callers may store an
estimate when no tokenizer count is available; the initial input and fallback
tool-output estimate both use four characters per token.
"""

import copy
import json
import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

LOG = logging.getLogger(__name__)


class ContextManager:
    """Static helpers for context budgeting, dialogue truncation and response diagnostics."""

    @staticmethod
    def count_dialogue_tokens(turns: List[Dict]) -> int:
        """Sum the recorded token counts of a dialogue history.

        Only counts already stored on a turn are summed. This method does not
        estimate. A turn without recorded counts contributes zero.

        Args:
            turns: Dialogue turns, each optionally carrying token counts

        Returns:
            Total number of tokens in the dialogue
        """
        return sum(ContextManager.get_turn_tokens(turn) for turn in turns if isinstance(turn, dict))

    @staticmethod
    def check_context_before_generation(data_point: Dict, cfg) -> Tuple[bool, Optional[str], Dict]:
        """Check whether the next generation still fits in the context window.

        Args:
            data_point: Current sample, whose ``turns`` hold the dialogue so far
            cfg: Generation config supplying ``max_seq_length``,
                ``context_safety_margin`` and ``inference.tokens_to_generate``

        Returns:
            Tuple of (will_fit, error_message, stats). ``error_message`` is
            ``None`` when the dialogue fits.
        """
        turns = data_point["turns"]
        current_tokens = ContextManager.count_dialogue_tokens(turns)

        safe_max = int(cfg.max_seq_length * cfg.context_safety_margin)
        available_tokens = safe_max - cfg.inference.tokens_to_generate

        stats = {
            "current_tokens": current_tokens,
            "available_tokens": available_tokens,
            "max_seq_length": cfg.max_seq_length,
            "tokens_to_generate": cfg.inference.tokens_to_generate,
            "safety_margin": cfg.context_safety_margin,
        }

        if current_tokens > available_tokens:
            error_msg = (
                f"Context too long: {current_tokens} tokens > {available_tokens} available "
                f"(max: {cfg.max_seq_length}, generate: {cfg.inference.tokens_to_generate}, safety: {cfg.context_safety_margin})"
            )
            return False, error_msg, stats

        return True, None, stats

    # First-and-recent truncation strategy
    @staticmethod
    def get_turn_tokens(turn: Dict) -> int:
        """Get the actual token count for a single turn.

        Args:
            turn: A dialogue turn dictionary with token counts

        Returns:
            Actual token count from the turn
        """
        # Sum up all actual token counts stored in the turn
        total = 0
        if "_llm_tokens" in turn:
            total += turn["_llm_tokens"]
        if "_tool_tokens" in turn:
            total += turn["_tool_tokens"]
        if "_input_tokens" in turn:
            total += turn["_input_tokens"]
        return total

    @staticmethod
    def first_and_recent_truncate(
        turns: List[Dict], max_seq_length: int, tokens_to_generate: int
    ) -> Tuple[List[Dict], Dict]:
        """
        Truncate dialogue keeping first turn and as many recent turns as possible.

        Strategy:
        1. Always keep the first turn (problem statement)
        2. Keep recent turns in reverse chronological order
        3. Remove middle turns as needed to fit within token limit

        Args:
            turns: List of dialogue turns
            max_seq_length: Maximum context length in tokens
            tokens_to_generate: Tokens to reserve for the next generation

        Returns:
            Tuple of (truncated turns, statistics dictionary)
        """
        if not turns:
            return turns, {"removed_turns": 0, "token_reduction": 0}

        # Calculate available space
        target_tokens = max_seq_length - tokens_to_generate

        # Deep copy turns to avoid modifying original
        first_turn = copy.deepcopy(turns[0])
        first_turn_tokens = ContextManager.get_turn_tokens(first_turn)

        # The first turn is mandatory, so an oversized first turn cannot be salvaged
        if first_turn_tokens > target_tokens:
            LOG.error(f"First turn alone ({first_turn_tokens} tokens) exceeds target ({target_tokens} tokens)")
            original_tokens = sum(ContextManager.get_turn_tokens(turn) for turn in turns)
            return [first_turn], {
                "original_turns": len(turns),
                "kept_turns": 1,
                "removed_turns": len(turns) - 1,
                "original_tokens": original_tokens,
                "final_tokens": first_turn_tokens,
                "token_reduction": 100.0,
                "kept_indices": [0],
                "warning": "First turn exceeds token limit",
            }

        # Start building result with first turn
        result_turns = [first_turn]
        used_tokens = first_turn_tokens

        # Add recent turns in reverse order (most recent first)
        kept_indices = [0]  # Track which turn indices we're keeping

        for i in range(len(turns) - 1, 0, -1):  # Start from last turn, go backwards, skip first
            turn = copy.deepcopy(turns[i])
            turn_tokens = ContextManager.get_turn_tokens(turn)

            # Check if adding this turn would exceed limit
            if used_tokens + turn_tokens <= target_tokens:
                result_turns.insert(1, turn)  # Insert after first turn
                used_tokens += turn_tokens
                kept_indices.insert(1, i)
            else:
                # Can't fit any more turns
                break

        # Calculate statistics
        original_tokens = sum(ContextManager.get_turn_tokens(t) for t in turns)
        removed_turns = len(turns) - len(result_turns)
        token_reduction = ((original_tokens - used_tokens) / original_tokens * 100) if original_tokens > 0 else 0

        # Log the truncation
        if removed_turns > 0:
            LOG.info(f"First-and-recent truncation: {len(turns)} → {len(result_turns)} turns")
            LOG.info(f"Kept turns: {kept_indices}")
            LOG.info(f"Token reduction: {original_tokens} → {used_tokens} ({token_reduction:.1f}%)")
        else:
            LOG.debug(f"No truncation needed: {used_tokens} tokens <= {target_tokens} target")

        stats = {
            "original_turns": len(turns),
            "kept_turns": len(result_turns),
            "removed_turns": removed_turns,
            "original_tokens": original_tokens,
            "final_tokens": used_tokens,
            "token_reduction": token_reduction,
            "kept_indices": kept_indices,
        }

        return result_turns, stats

    @staticmethod
    def get_truncation_preview(turns: List[Dict], max_seq_length: int, tokens_to_generate: int) -> str:
        """Describe what first_and_recent_truncate would drop, without dropping it.

        Args:
            turns: List of dialogue turns
            max_seq_length: Maximum context length in tokens
            tokens_to_generate: Tokens to reserve for the next generation

        Returns:
            Human readable summary of the truncation, intended for logs.
        """
        _, stats = ContextManager.first_and_recent_truncate(turns, max_seq_length, tokens_to_generate)

        if stats["removed_turns"] == 0:
            return "No truncation needed"

        preview = f"Would truncate {stats['removed_turns']} turns:\n"
        preview += "  Keep: Turn 0 (problem statement)\n"

        # Show which turns would be kept
        for idx in stats["kept_indices"][1:]:  # Skip first turn
            preview += f"  Keep: Turn {idx}"
            if idx == len(turns) - 1:
                preview += " (most recent)"
            preview += "\n"

        # Show which turns would be removed
        removed_indices = [i for i in range(len(turns)) if i not in stats["kept_indices"]]
        if removed_indices:
            preview += f"  Remove: Turns {removed_indices}\n"

        preview += f"  Token reduction: {stats['token_reduction']:.1f}%"

        return preview

    # Final turn prompt injection
    @staticmethod
    def inject_final_turn_instruction(turns: List[Dict], is_final_turn: bool) -> List[Dict]:
        """
        Inject a special instruction when it's the final turn.

        Args:
            turns: List of dialogue turns
            is_final_turn: Whether this is the last allowed turn

        Returns:
            Modified turns with final turn instruction if applicable
        """
        if not is_final_turn or not turns:
            return turns

        # Deep copy to avoid modifying original
        modified_turns = copy.deepcopy(turns)

        # The wording mirrors the response format required by the system prompt,
        # so the final turn does not introduce a second, competing format.
        instruction_text = (
            "\n\n**Interaction protocol update**\n"
            "You have reached the maximum number of tool calls. You **must** now reply with `<findings>` and `<locations>` blocks.\n\n"
            "**Strict rules for this final response:**\n"
            "- Assistant message must follow the EXACT structure: `<think>...</think>` followed by `<findings>...</findings>` and `<locations>...</locations>`\n"
            "- **DO NOT** issue any more `<tool_call>` blocks\n"
            "- **DO NOT** include any text outside the required tags\n"
            "- In `<findings>`, provide bullet points explaining why each location needs modification\n"
            "- In `<locations>`, emit **every** file and line range that needs editing\n"
            "- If uncertain about exact lines, include your best assessment\n\n"
            "**Output format:**\n"
            "```xml\n"
            "<think>\n"
            "Based on my investigation, I have identified the following locations that need editing...\n"
            "</think>\n"
            "\n"
            "<findings>\n"
            "• Location explanation: Why this specific location requires modification\n"
            "• Root cause: What the underlying issue is\n"
            "• Solution idea: How the fix should be approached (without showing code)\n"
            "</findings>\n"
            "\n"
            "<locations>\n"
            "path/to/file.py:L<start>-L<end>\n"
            "another/file.rs:L<start>-L<end>\n"
            "</locations>\n"
            "```\n\n"
            "Remember: It's better to over-inspect and over-include than to miss a required edit location."
        )

        # Find the last turn with inputs
        last_input_idx = -1
        for i in range(len(modified_turns) - 1, -1, -1):
            if "inputs" in modified_turns[i] and modified_turns[i]["inputs"]:
                last_input_idx = i
                break

        if last_input_idx >= 0:
            # Append instruction to the last input
            modified_turns[last_input_idx]["inputs"] += instruction_text
            # Mark that this turn contains final turn instruction
            modified_turns[last_input_idx]["contains_final_turn_instruction"] = True
            LOG.info("Injected final turn instruction")
        else:
            # If no inputs found, create a new turn with the instruction
            modified_turns.append(
                {
                    "inputs": instruction_text,
                    "assistant": "",
                    "tool_call": None,
                    "tool_output": "",
                    "turn_type": "final_turn_instruction",  # Mark as system-generated instruction
                    "_retry_count": 0,  # System instructions don't have retries
                }
            )
            LOG.info("Added new turn with final turn instruction")

        return modified_turns

    @staticmethod
    def should_inject_final_turn(
        cur_step: int, total_steps: int, status: Optional[str], enable_final_turn_prompt: bool = True
    ) -> bool:
        """
        Determine if we should inject the final turn instruction.

        Args:
            cur_step: Current step number (0-indexed)
            total_steps: Total allowed steps
            status: Current status (None if still investigating)
            enable_final_turn_prompt: Whether the feature is enabled

        Returns:
            True if we should inject the final turn instruction
        """
        if not enable_final_turn_prompt or status is not None:
            return False

        # Trigger only on the last allowed step.
        steps_completed = cur_step + 1
        return steps_completed >= total_steps

    # Loop detection and prevention utilities
    @staticmethod
    def detect_repetitive_tool_calls(generations: List[Dict], threshold: int = 3) -> Tuple[bool, Optional[Dict]]:
        """
        Detect if the agent is stuck in a loop generating the same tool call repeatedly.

        Args:
            generations: List of generation dictionaries from the agent
            threshold: Number of identical calls to consider it a loop (default: 3)

        Returns:
            Tuple of (is_loop_detected, loop_info)
            where loop_info contains details about the repeated call if a loop is detected
        """
        if len(generations) < threshold:
            return False, None

        # Extract tool calls from generations
        tool_calls = []
        for gen in generations:
            gen_text = gen.get("generation", "")
            if "<tool_call>" in gen_text and "</tool_call>" in gen_text:
                # Extract the JSON between tool_call tags
                start = gen_text.find("<tool_call>") + len("<tool_call>")
                end = gen_text.find("</tool_call>")
                if start < end:
                    try:
                        tool_call_json = gen_text[start:end].strip()
                        # Normalize the JSON to handle formatting differences
                        tool_call_obj = json.loads(tool_call_json)
                        tool_call_normalized = json.dumps(tool_call_obj, sort_keys=True)
                        tool_calls.append(tool_call_normalized)
                    except json.JSONDecodeError as e:
                        LOG.debug(f"Failed to parse tool call JSON: {e}")
                        tool_calls.append(gen_text[start:end].strip())

        if not tool_calls:
            return False, None

        # Count occurrences of each tool call
        call_counts = Counter(tool_calls)

        # Check if any call appears more than threshold times
        for call, count in call_counts.items():
            if count >= threshold:
                # Calculate what percentage of recent calls are this repeated call
                recent_window = min(10, len(tool_calls))  # Look at last 10 calls
                recent_calls = tool_calls[-recent_window:]
                recent_repetitions = recent_calls.count(call)

                loop_info = {
                    "repeated_call": call,
                    "total_repetitions": count,
                    "recent_repetitions": recent_repetitions,
                    "recent_window": recent_window,
                    "loop_percentage": recent_repetitions / recent_window * 100,
                }

                # Consider it a loop if more than 70% of recent calls are the same
                if loop_info["loop_percentage"] >= 70:
                    return True, loop_info

        return False, None

    @staticmethod
    def inject_loop_intervention(turns: List[Dict], loop_info: Dict) -> List[Dict]:
        """
        Inject a system message to help the agent break out of a loop.

        Args:
            turns: Current conversation turns
            loop_info: Information about the detected loop

        Returns:
            Modified turns with intervention message
        """
        # Parse the repeated call to provide specific guidance
        try:
            repeated_call = json.loads(loop_info["repeated_call"])
            if "tool" in repeated_call:
                tool_name = repeated_call["tool"]
                tool_params = {key: value for key, value in repeated_call.items() if key != "tool"}
            elif len(repeated_call) == 1:
                tool_name, tool_params = next(iter(repeated_call.items()))
            else:
                raise ValueError("Unrecognized tool-call format")

            if not isinstance(tool_params, dict):
                raise TypeError("Tool parameters must be a dictionary")

            # Create specific guidance based on the tool
            if tool_name == "view_file":
                file_path = tool_params.get("path", "unknown")

                intervention = f"""SYSTEM INTERVENTION: Loop detected! You have attempted to view '{file_path}' {loop_info["total_repetitions"]} times with the same parameters.

The file appears to be too large or the output is being truncated. Please try a different approach:
1. View a specific section using line numbers (e.g., view_range: [1000, 1200])
2. Search for specific content with codebase_search
3. Inspect the repository structure with repo_tree
4. Inspect related imports with connected_tree

DO NOT repeat the same view_file command. Think of an alternative strategy."""
            else:
                intervention = f"""SYSTEM INTERVENTION: Loop detected! You have repeated the same {tool_name} command {loop_info["total_repetitions"]} times.

This suggests the current approach isn't working. Please:
1. Analyze why the previous attempts didn't provide useful information
2. Try a completely different tool or approach
3. Break down the problem into smaller steps
4. Consider if you're looking in the wrong place

DO NOT repeat the same command. Think of an alternative strategy."""
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            LOG.debug(f"Failed to parse repeated call for specific guidance: {e}")
            intervention = f"""SYSTEM INTERVENTION: Loop detected! You have repeated the same command {loop_info["total_repetitions"]} times.

Please try a different approach. The current strategy is not working."""

        # Add the intervention as a system turn
        intervention_turn = {
            "inputs": intervention,
            "tool_call": None,
            "tool_output": "",
            "assistant": "",  # Agent will respond to this
            "turn_type": "loop_intervention",  # Mark as system-generated intervention
            "_retry_count": 0,  # System interventions don't have retries
        }

        # Insert the intervention before the last turn
        modified_turns = copy.deepcopy(turns)
        if len(modified_turns) > 0:
            modified_turns.insert(-1, intervention_turn)
        else:
            modified_turns.append(intervention_turn)

        return modified_turns

    @staticmethod
    def analyze_loop_patterns(generations: List[Dict]) -> Dict:
        """
        Analyze patterns in tool call generations to identify potential issues.

        Args:
            generations: List of generation dictionaries

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "total_generations": len(generations),
            "unique_calls": 0,
            "most_common_call": None,
            "repetition_ratio": 0.0,
            "potential_issues": [],
        }

        if not generations:
            return analysis

        # Extract all tool calls
        tool_calls = []
        for gen in generations:
            gen_text = gen.get("generation", "")
            if "<tool_call>" in gen_text and "</tool_call>" in gen_text:
                start = gen_text.find("<tool_call>") + len("<tool_call>")
                end = gen_text.find("</tool_call>")
                if start < end:
                    tool_calls.append(gen_text[start:end].strip())

        if not tool_calls:
            analysis["potential_issues"].append("No tool calls found in generations")
            return analysis

        # Analyze patterns
        call_counts = Counter(tool_calls)
        analysis["unique_calls"] = len(call_counts)

        if call_counts:
            most_common = call_counts.most_common(1)[0]
            analysis["most_common_call"] = most_common[0]
            analysis["repetition_ratio"] = most_common[1] / len(tool_calls)

            # Identify issues
            if analysis["repetition_ratio"] > 0.5:
                analysis["potential_issues"].append(
                    f"High repetition: {most_common[1]}/{len(tool_calls)} calls are identical"
                )

            if analysis["unique_calls"] == 1 and len(tool_calls) > 3:
                analysis["potential_issues"].append("All tool calls are identical - severe loop detected")

            # Check for truncation indicators
            for call in tool_calls:
                if "view_file" in call and "[1, -1]" in call:
                    analysis["potential_issues"].append("Attempting to view entire files - may cause truncation")

        return analysis

    # Response length management
    @staticmethod
    def inject_length_warning(turns: List[Dict], warning_message: str) -> List[Dict]:
        """
        Inject a warning about response length into the conversation.

        Args:
            turns: The conversation turns
            warning_message: Warning to inject

        Returns:
            Updated turns with warning
        """
        warning_turn = {
            "inputs": f"[SYSTEM WARNING: {warning_message}]",
            "assistant": "",
            "tool_call": None,
            "tool_output": "",
            "turn_type": "length_warning",  # Mark as system-generated warning
            "_retry_count": 0,  # System warnings don't have retries
        }

        return turns + [warning_turn]

    @staticmethod
    def analyze_response_failure(
        response: str, turns: List[Dict], max_context_length: int, actual_total_tokens: Optional[int] = None
    ) -> Dict:
        """
        Analyze why a response generation failed due to length.

        This function analyzes the character distribution in the response
        to understand what parts are consuming the most space.

        Args:
            response: The generated response text
            turns: The dialogue turns
            max_context_length: Maximum context length
            actual_total_tokens: Actual total token count from LLM

        Returns:
            Detailed diagnostics about the failure
        """
        # Count characters in different parts
        thinking_pattern = r"<think>(.*?)</think>"
        tool_pattern = r"<tool_call>(.*?)</tool_call>"

        thinking_matches = re.findall(thinking_pattern, response, re.DOTALL)
        tool_matches = re.findall(tool_pattern, response, re.DOTALL)

        thinking_chars = sum(len(m) for m in thinking_matches)
        tool_chars = sum(len(m) for m in tool_matches)
        other_chars = len(response) - thinking_chars - tool_chars

        analysis = {
            # Use actual token count from LLM
            "total_response_tokens": actual_total_tokens if actual_total_tokens is not None else -1,
            "token_source": "actual" if actual_total_tokens is not None else "not_provided",
            # Character-based analysis only
            "thinking_chars": thinking_chars,
            "tool_call_chars": tool_chars,
            "other_chars": other_chars,
            "thinking_percentage": (thinking_chars / len(response) * 100) if response else 0,
            "tool_call_percentage": (tool_chars / len(response) * 100) if response else 0,
            "other_percentage": (other_chars / len(response) * 100) if response else 0,
            "total_chars": len(response),
            "num_turns": len(turns),
        }

        # Identify the main contributor based on character percentages
        if analysis["thinking_percentage"] > 70:
            analysis["main_issue"] = "excessive_thinking"
        elif analysis["tool_call_percentage"] > 50:
            analysis["main_issue"] = "excessive_tool_calls"
        else:
            analysis["main_issue"] = "general_verbosity"

        return analysis

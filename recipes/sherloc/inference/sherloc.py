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

"""SHERLOC: structured diagnostic localization for code repair agents.

This module implements the SHERLOC localization agent: a multi-turn, read-only exploration loop
that reads a bug report and a snapshot of the repository it belongs to, and returns the set of
source locations that must be edited, together with a structured diagnostic explanation of why.
SHERLOC never edits code. Its output is meant to be handed to a downstream repair agent.

What it consumes
    Each data point is a SWE-bench-style instance with:
      * ``instance_id`` - identifier used to locate the repository snapshot;
      * ``problem_statement`` - the natural-language bug report shown to the model;
      * ``patch`` (optional) - the reference patch. It is never shown to the model. It is used
        only to record how much of the reference edit surface survived repository filtering,
        which makes localization recall interpretable.
      * ``turns`` (optional) - a partially populated dialogue, allowing a run to be resumed.
    The repository itself is read from a pickled snapshot at
    ``<mount_directory>/<instance_id>.pkl``: a nested directory dictionary carrying file contents
    at the buggy commit. Working from a snapshot keeps every step hermetic and reproducible, and
    lets the tools answer without touching a live checkout.

The loop (one iteration per step, up to ``total_steps``)
    1. Truncate the dialogue with a first-and-recent policy so that the prompt plus the reserved
       generation budget stays inside ``max_seq_length``. The first turn holds the problem
       statement and repository tree, so it is always retained.
    2. Detect repeated identical tool calls and inject an intervention turn that tells the model
       it is repeating itself.
    3. Re-check that the assembled prompt fits before contacting the server, so an oversized
       context fails fast with a diagnosable reason instead of a server-side error.
    4. On the final step, inject an instruction requiring the model to commit to an answer.
    5. Generate. If the reply hits the generation cap it was probably cut off mid-answer, so a
       request for brevity is appended and the step is retried up to ``max_retries`` times.
    6. Parse the reply. It is either a ``<tool_call>`` block or a ``<findings>`` +
       ``<locations>`` block:
         * a tool call is executed read-only against the snapshot (view a file range, print the
           repository tree, print the import-connected tree around a file, or search the
           codebase) and its output becomes the next user turn;
         * a locations block ends the episode successfully. A sibling ``<findings>``
           block, when present, is stored as the free-form diagnosis.

What it emits
    One record per instance containing the predicted ``locations`` and ``findings``, the
    full turn-by-turn transcript (model reply, tool call, tool output, per-turn token counts,
    retry count, and which turns were in context), aggregate token usage, and a terminal
    ``status`` / ``reason`` pair. Every exit path is labelled, so failures can be separated by
    cause (context exhausted, answer truncated, no parsable action, step budget spent) rather
    than lumped together with genuine localization misses.
"""

import copy
import logging
import pickle
import sys
from dataclasses import field
from pathlib import Path

import hydra
import openai

from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, parse_reasoning, setup_logging
from recipes.sherloc.inference.sherloc_utils.context_manager import ContextManager
from recipes.sherloc.inference.sherloc_utils.defaults import (
    DEFAULT_COMMON_WORDS_FILTER,
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_FILE_EXTENSIONS,
)
from recipes.sherloc.inference.sherloc_utils.dialog_processor import DialogProcessor
from recipes.sherloc.inference.sherloc_utils.patch_processor import PatchProcessor
from recipes.sherloc.inference.sherloc_utils.repo_manager import RepoManager
from recipes.sherloc.inference.sherloc_utils.tool_executor import ToolExecutor

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class SherlocGenerationConfig(GenerationTaskConfig):
    """Configuration for the SHERLOC localization run.

    Inherits every field of :class:`GenerationTaskConfig` and adds the knobs that control the
    exploration loop, the repository view offered to the model, and the context/response budgets.

    Attributes:
        inference: Sampling parameters. The defaults are the ones used for the reported results:
            high-temperature nucleus sampling with a large generation budget, since a step may
            contain a long chain of thought followed by a tool call.
        server: Server connection settings, forwarded to the inference backend.
        prompt_template: Accepted for compatibility with launch commands that still pass this key.
            The conversation is assembled by :meth:`SherlocGenerationTask.fill_prompt` instead.
        multi_turn_key: Accepted for compatibility, as above.
        mount_directory: Directory holding the per-instance pickled repository snapshots.
        remove_thinking: Strip the reasoning span from a reply before it is written back into the
            dialogue, so that earlier chains of thought do not consume context on later steps.
        total_steps: Maximum number of exploration steps before the episode is declared failed.
        file_extensions: File suffixes kept in the repository view. Everything else is hidden from
            both the tree and the tools.
        exclude_dirs: Directory names pruned from the repository view. Tests, build artefacts,
            vendored code and documentation are dropped so the model spends its context on the
            source that a patch would plausibly touch.
        enable_implicit_tool_detection: Also accept a bare JSON tool call that is not wrapped in
            ``<tool_call>`` tags. Recovers steps that would otherwise be discarded as unparsable.
        common_words_filter: Stop words used when recovering an implicit tool call, to avoid
            treating ordinary prose as a search query.
        max_seq_length: Context window assumed for the served model. Drives truncation and the
            pre-generation fit check.
        show_line_counts: Annotate each file in the repository tree with its line count.
        max_view_lines: Upper bound on the number of lines a single file view may return.
        enable_loop_detection: Watch for identical consecutive tool calls.
        loop_detection_threshold: Number of identical consecutive calls that counts as a loop.
        enable_enhanced_context: Verify that the prompt fits before each generation call.
        context_safety_margin: Fraction of the context window usable by prompt plus generation;
            the remainder absorbs tokenizer disagreement between client and server.
        enable_response_length_management: Track reply length and retry replies that look cut off.
        max_retries: Retries allowed per step when a reply hits the generation cap.
        inject_length_warnings: On such a retry, add a turn asking for a shorter reply.
        response_warning_threshold: Fraction of the generation budget that triggers a log warning.
        response_critical_threshold: Fraction that triggers a stronger warning.
        enable_final_turn_prompt: On the final step, instruct the model to emit its findings and
            locations instead of attempting another tool call.
    """

    inference: InferenceConfig = field(
        default_factory=lambda: InferenceConfig(
            endpoint_type="text",
            temperature=0.99,
            top_k=0,
            top_p=0.95,
            min_p=0.0,
            random_seed=0,
            tokens_to_generate=81920,
            repetition_penalty=1.0,
            top_logprobs=None,
            extra_body={},
        )
    )
    server: dict = field(default_factory=dict)

    # Accepted for compatibility with existing launch commands; the conversation is built below.
    prompt_template: str | None = None
    multi_turn_key: str | None = None

    # Agent behavior settings
    mount_directory: str = "/repos/"
    remove_thinking: bool = False
    total_steps: int = 20

    # Repository filtering settings
    file_extensions: list = field(default_factory=lambda: list(DEFAULT_FILE_EXTENSIONS))
    exclude_dirs: list = field(default_factory=lambda: list(DEFAULT_EXCLUDE_DIRS))

    # Tool detection settings
    enable_implicit_tool_detection: bool = True
    common_words_filter: list = field(default_factory=lambda: list(DEFAULT_COMMON_WORDS_FILTER))

    max_seq_length: int = 262144
    show_line_counts: bool = False
    max_view_lines: int = 1000

    # Loop detection settings
    enable_loop_detection: bool = True
    loop_detection_threshold: int = 3

    # Enhanced context management settings
    enable_enhanced_context: bool = True
    context_safety_margin: float = 0.9

    # Response length management
    enable_response_length_management: bool = True
    max_retries: int = 2
    inject_length_warnings: bool = True
    response_warning_threshold: float = 0.75
    response_critical_threshold: float = 0.9

    # Final turn prompt settings. The instruction is injected on the last step only
    # (see ContextManager.should_inject_final_turn), and only while the episode is
    # still unresolved; there is no earlier-trigger threshold.
    enable_final_turn_prompt: bool = True


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_sherloc_generation_config", node=SherlocGenerationConfig)


class SherlocGenerationTask(GenerationTask):
    """Runs the SHERLOC exploration loop for one data point at a time."""

    def __init__(self, cfg: SherlocGenerationConfig):
        super().__init__(cfg)
        tokenizer = self.prompt.tokenizer if self.prompt else None
        self.tool_executor = ToolExecutor(cfg, tokenizer)

    def log_example_prompt(self, data):
        # The first prompt embeds an entire repository tree, which is not useful as a log sample.
        return

    def fill_prompt(self, data_point, data=None, prompt_format=None):
        """Build a multi-turn conversation from the accumulated turns of this episode.

        Each SHERLOC turn holds the user side (problem statement, or the previous tool output) and
        the assistant side (the model reply for that step), so the dialogue has to be replayed
        explicitly rather than rendered from a single-shot prompt template. The last turn carries
        only the user side, which is what the model is being asked to continue.

        Returns a chat-style message list, or, when the tokenizer defines a chat template and a
        text endpoint is in use, the template-rendered string for that message list.
        """
        turns = data_point.get("turns", [])
        if not turns:
            dp = data_point.copy()
            if "inputs" not in dp:
                dp["inputs"] = ""
            return super().fill_prompt(data_point=dp, data=data, prompt_format=prompt_format)

        messages = []

        if self.prompt and self.prompt.config.system:
            messages.append({"role": "system", "content": self.prompt.config.system})

        for turn in turns[:-1]:
            user_content = turn.get("inputs", "")
            if user_content:
                messages.append({"role": "user", "content": user_content})
            assistant_content = turn.get("assistant", "")
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})

        last_turn = turns[-1]
        messages.append({"role": "user", "content": last_turn.get("inputs", "")})

        # A text endpoint expects a single string, so apply the tokenizer's own chat template
        # rather than a hand-written one: the served model then sees exactly the format it was
        # trained on.
        if (
            self.prompt
            and self.prompt.tokenizer
            and hasattr(self.prompt.tokenizer, "chat_template")
            and self.prompt.tokenizer.chat_template
        ):
            chat_kwargs = dict(self.cfg.chat_template_kwargs) if self.cfg.chat_template_kwargs else {}
            return self.prompt.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **chat_kwargs
            )

        return messages

    async def process_single_datapoint(self, data_point, all_data):
        """Run the full localization episode for one instance and return its result record."""

        instance_id = data_point.get("instance_id", "unknown")

        # Instances are processed concurrently, so every line is tagged with its instance id.
        log_prefix = f"[{instance_id}] "

        def log_info(msg, indent=0):
            LOG.info(f"{log_prefix}{' ' * indent}{msg}")

        def log_debug(msg, indent=0):
            LOG.debug(f"{log_prefix}{' ' * indent}{msg}")

        def log_error(msg, indent=0):
            LOG.error(f"{log_prefix}{' ' * indent}{msg}")

        def log_warning(msg, indent=0):
            LOG.warning(f"{log_prefix}{' ' * indent}{msg}")

        log_info(f"Processing: {instance_id}")

        log_debug(
            f"Initial data_point keys: {list(data_point.keys()) if isinstance(data_point, dict) else 'not a dict'}"
        )
        if "turns" in data_point:
            log_debug(f"Initial turns structure: {data_point['turns']}")

        if not data_point.get("problem_statement", "").strip():
            log_warning("Skipping data point due to empty problem statement")

            log_info("\n" + "=" * 100)
            log_info(f"{'=' * 15} SKIPPED SAMPLE: {instance_id} (empty problem statement) {'=' * 15}")
            log_info("=" * 100 + "\n")

            return {
                "generation": [],
                "total_generated_tokens": 0,
                "num_turns": 0,
                "status": "skipped",
                "reason": "empty_problem_statement",
                "turns": [],
            }

        total_steps = self.cfg.total_steps
        chat_history = []
        total_generated_tokens = 0

        # Normalize the turn structure up front: a resumed run may carry partial turns, and every
        # later stage assumes the full set of keys is present.
        if "turns" in data_point and isinstance(data_point["turns"], list) and len(data_point["turns"]) > 0:
            for i, turn in enumerate(data_point["turns"]):
                if isinstance(turn, dict):
                    turn.setdefault("turn_id", i)
                    turn.setdefault("inputs", "")
                    turn.setdefault("assistant", "")
                    turn.setdefault("tool_call", None)
                    turn.setdefault("tool_output", "")
                    log_debug(
                        f"Turn {i} after setdefault - keys: {list(turn.keys())}, has assistant: {'assistant' in turn}",
                        indent=4,
                    )
                else:
                    log_warning(
                        f"Found non-dict turn at index {i}: {type(turn)}, replacing with empty structure", indent=4
                    )
                    data_point["turns"][i] = {
                        "turn_id": i,
                        "inputs": "",
                        "assistant": "",
                        "tool_call": None,
                        "tool_output": "",
                        "_retry_count": 0,
                    }
        else:
            data_point["turns"] = [
                {
                    "turn_id": 0,
                    "inputs": "",
                    "assistant": "",
                    "tool_call": None,
                    "tool_output": "",
                    "_retry_count": 0,
                }
            ]

        try:
            instance_filepath = Path(self.cfg.mount_directory).joinpath(f"{data_point['instance_id']}.pkl")

            with open(instance_filepath, "rb") as f:
                repo_dict = pickle.load(f)
            repo_dict = RepoManager.filter_repo_dict(repo_dict, self.cfg.exclude_dirs, self.cfg.file_extensions)
            tree_structure = RepoManager.tree_repo_dict(repo_dict, self.cfg.show_line_counts)

            # Filtering can hide part of the reference edit surface. Record how much of it is still
            # reachable so that recall numbers can be read against an attainable ceiling. The
            # reference patch is only inspected here and never enters the prompt.
            ground_truth_in_repo_percentage = 0.0
            if "patch" in data_point and data_point["patch"]:
                try:
                    locations = PatchProcessor.extract_locations_from_patch(data_point["patch"])
                    ground_truth_in_repo_percentage, debug_info = RepoManager.calculate_ground_truth_percentage(
                        repo_dict, locations, self.cfg.exclude_dirs, self.cfg.file_extensions
                    )
                    log_debug(f"Ground truth check debug info: {debug_info}", indent=4)
                    data_point["_missing_ground_truth_files"] = debug_info.get("missing_files_details", [])
                except Exception as e:
                    log_warning(f"Error checking ground truth files: {e}", indent=4)

            data_point["_ground_truth_in_repo_percentage"] = ground_truth_in_repo_percentage

            inputs = f"""
### Problem Description
{data_point["problem_statement"]}

### Repository Structure
{tree_structure}
"""

            data_point["turns"][0]["inputs"] = inputs
            data_point["turns"][0]["turn_id"] = 0
            # Cheap four-characters-per-token estimate; the exact count is unknown until the
            # server tokenizes the prompt, and this value is only used for context bookkeeping.
            data_point["turns"][0]["_input_tokens"] = len(inputs) // 4
            data_point["turns"][0]["_retry_count"] = 0
            log_debug(f"Initialized turns with problem statement, turn count: {len(data_point['turns'])}", indent=4)

        except Exception as e:
            log_error(f"Error loading repository: {e}")

            log_info("\n" + "=" * 100)
            log_info(f"{'=' * 15} FAILED SAMPLE: {instance_id} (repository loading error) {'=' * 15}")
            log_info("=" * 100 + "\n")

            return {
                "generation": [],
                "total_generated_tokens": 0,
                "num_turns": 0,
                "status": "failed",
                "reason": f"repository_loading_error: {str(e)}",
                "turns": data_point["turns"],
            }

        reason = None
        status = None
        try:
            for cur_step in range(total_steps):
                log_info(f"Step {cur_step + 1}/{total_steps}", indent=4)

                if (
                    "turns" not in data_point
                    or not isinstance(data_point["turns"], list)
                    or len(data_point["turns"]) == 0
                ):
                    log_error(
                        f"Invalid turns structure at step {cur_step}: {data_point.get('turns', 'missing')}", indent=8
                    )
                    status = "failed"
                    reason = "invalid_turns_structure"
                    break

                if (
                    hasattr(self.cfg, "max_seq_length")
                    and self.cfg.max_seq_length is not None
                    and self.cfg.max_seq_length > 0
                ):
                    original_turns_count = len(data_point["turns"])

                    # Keep the opening turn (problem statement plus repository tree) and the most
                    # recent turns, dropping the middle of the dialogue when the budget is tight.
                    log_debug("Using first-and-recent truncation strategy", indent=8)
                    safe_max = int(self.cfg.max_seq_length * self.cfg.context_safety_margin)

                    preview = ContextManager.get_truncation_preview(
                        data_point["turns"], safe_max, self.cfg.inference.tokens_to_generate
                    )
                    if "No truncation needed" not in preview:
                        log_debug(f"Truncation preview:\n{preview}", indent=12)

                    data_point["turns"], truncation_stats = ContextManager.first_and_recent_truncate(
                        data_point["turns"], safe_max, self.cfg.inference.tokens_to_generate
                    )

                    if truncation_stats["removed_turns"] > 0:
                        log_info(
                            f"Truncated dialogue from {original_turns_count} to {len(data_point['turns'])} turns",
                            indent=12,
                        )
                        log_info(f"Truncation stats: {truncation_stats}", indent=12)

                # The prompt for this step is assembled on a copy. The interventions below are
                # applied to that copy and are explicitly synced back whenever they should also
                # persist in the recorded dialogue.
                prepared_data_point = copy.deepcopy(data_point)

                context_turn_ids = [turn.get("turn_id", i) for i, turn in enumerate(prepared_data_point["turns"])]
                log_debug(f"Context includes turn IDs: {context_turn_ids}", indent=8)

                if self.cfg.enable_loop_detection and len(chat_history) >= self.cfg.loop_detection_threshold - 1:
                    is_loop, loop_info = ContextManager.detect_repetitive_tool_calls(
                        chat_history, self.cfg.loop_detection_threshold - 1
                    )

                    if is_loop:
                        log_warning(
                            f"Potential loop detected before generation! "
                            f"Previous {loop_info['total_repetitions']} calls were identical",
                            indent=12,
                        )
                        prepared_data_point["turns"] = ContextManager.inject_loop_intervention(
                            prepared_data_point["turns"], loop_info
                        )
                        # Persist the intervention so it stays in the transcript and in later steps.
                        data_point["turns"] = copy.deepcopy(prepared_data_point["turns"])
                        for i, turn in enumerate(data_point["turns"]):
                            turn["turn_id"] = i
                        log_debug(f"Added loop intervention as turn {len(data_point['turns']) - 1}", indent=12)

                if getattr(self.cfg, "enable_enhanced_context", True):
                    will_fit, error_msg, context_stats = ContextManager.check_context_before_generation(
                        prepared_data_point, self.cfg
                    )
                    if not will_fit:
                        log_error(f"Context length check failed: {error_msg}", indent=12)
                        log_error(f"Context stats: {context_stats}", indent=12)
                        status = "failed"
                        reason = "context_length_exceeded_proactive"
                        break

                if ContextManager.should_inject_final_turn(
                    cur_step,
                    total_steps,
                    status,
                    enable_final_turn_prompt=getattr(self.cfg, "enable_final_turn_prompt", True),
                ):
                    log_info(f"Injecting final turn instruction at step {cur_step + 1}/{total_steps}", indent=8)
                    prepared_data_point["turns"] = ContextManager.inject_final_turn_instruction(
                        prepared_data_point["turns"], is_final_turn=True
                    )
                    # The instruction may extend the last turn or open a new one; persist either way.
                    if len(prepared_data_point["turns"]) > len(data_point["turns"]):
                        data_point["turns"] = copy.deepcopy(prepared_data_point["turns"])
                        for i, turn in enumerate(data_point["turns"]):
                            turn["turn_id"] = i
                        log_debug(f"Added final turn instruction as new turn {len(data_point['turns']) - 1}", indent=8)
                    else:
                        data_point["turns"] = copy.deepcopy(prepared_data_point["turns"])
                        log_debug("Appended final turn instruction to existing turn", indent=8)
                    context_turn_ids = [turn.get("turn_id", i) for i, turn in enumerate(prepared_data_point["turns"])]
                    log_debug(f"Updated context after final turn instruction: {context_turn_ids}", indent=8)

                response_type = "normal"
                if cur_step == total_steps - 1:
                    response_type = "final_turn"

                retry_count = 0
                actual_generated_tokens = 0
                while retry_count <= self.cfg.max_retries:
                    try:
                        log_info(
                            f"Sending {len(prepared_data_point['turns'])} turns to LLM (attempt {retry_count + 1})",
                            indent=12,
                        )

                        llm_output = await super().process_single_datapoint(prepared_data_point, all_data)

                        llm_output["_context_turn_ids"] = context_turn_ids

                        actual_generated_tokens = llm_output.get("num_generated_tokens", 0)

                        # The reasoning span is needed to diagnose why a reply ran long.
                        full_gen = llm_output.get("_full_generation", llm_output.get("generation", ""))

                        if self.cfg.enable_response_length_management:
                            max_tokens = self.cfg.inference.tokens_to_generate

                            # A reply that spends its entire budget was almost certainly cut off
                            # before the closing tag, so it is not accepted as a complete answer.
                            is_acceptable = actual_generated_tokens < max_tokens
                            warning_msg = ""

                            if actual_generated_tokens > max_tokens:
                                warning_msg = f"Response exceeds token limit: {actual_generated_tokens} > {max_tokens}"
                            elif actual_generated_tokens >= max_tokens:
                                warning_msg = (
                                    f"Response at token limit: {actual_generated_tokens} = {max_tokens} "
                                    f"(likely truncated)"
                                )
                            elif actual_generated_tokens > max_tokens * self.cfg.response_critical_threshold:
                                warning_msg = (
                                    f"Response approaching token limit: {actual_generated_tokens}/{max_tokens} "
                                    f"({(actual_generated_tokens / max_tokens) * 100:.1f}%)"
                                )
                            elif actual_generated_tokens > max_tokens * self.cfg.response_warning_threshold:
                                warning_msg = (
                                    f"Response length warning: {actual_generated_tokens}/{max_tokens} "
                                    f"({(actual_generated_tokens / max_tokens) * 100:.1f}%)"
                                )

                            stats = {
                                "actual_tokens": actual_generated_tokens,
                                "max_tokens": max_tokens,
                                "percentage": (actual_generated_tokens / max_tokens) * 100 if max_tokens > 0 else 0,
                                "response_type": response_type,
                            }

                            if warning_msg:
                                log_warning(f"Response length check: {warning_msg}", indent=16)
                                log_debug(f"Response stats: {stats}", indent=16)

                            if not is_acceptable and retry_count < self.cfg.max_retries:
                                log_error(f"Response too long: {stats['actual_tokens']} tokens", indent=16)

                                failure_analysis = ContextManager.analyze_response_failure(
                                    full_gen,
                                    prepared_data_point["turns"],
                                    self.cfg.max_seq_length or 128000,
                                    actual_total_tokens=actual_generated_tokens,
                                )
                                log_info(f"Failure analysis: {failure_analysis}", indent=16)

                                if self.cfg.inject_length_warnings:
                                    # Ask for brevity where it is affordable: on a final turn the
                                    # answer itself must shrink, otherwise the reasoning should.
                                    if response_type == "final_turn":
                                        warning_msg = (
                                            "Please provide a shorter, more focused answer with concise findings "
                                            "and locations. Include only the most essential bullet points in "
                                            "<findings> without excessive explanation, and directly state the bug "
                                            "locations in <locations>."
                                        )
                                    else:
                                        warning_msg = (
                                            "Please be more concise: reduce your thinking/reasoning to only the most "
                                            "essential analysis steps. Skip redundant explanations and focus on the "
                                            "critical path to finding the bug."
                                        )

                                    prepared_data_point["turns"] = ContextManager.inject_length_warning(
                                        prepared_data_point["turns"], warning_msg
                                    )
                                    data_point["turns"] = copy.deepcopy(prepared_data_point["turns"])
                                    for i, turn in enumerate(data_point["turns"]):
                                        turn["turn_id"] = i
                                    context_turn_ids = [
                                        turn.get("turn_id", i) for i, turn in enumerate(prepared_data_point["turns"])
                                    ]
                                    log_debug(
                                        f"Added length warning as turn {len(data_point['turns']) - 1}", indent=20
                                    )

                                retry_count += 1
                                continue

                        break

                    except openai.BadRequestError as e:
                        if "Please reduce the length of the messages or completion" in str(
                            e
                        ) or "is longer than the model's context length" in str(e):
                            log_warning(
                                "SHERLOC generation failed due to running out of context. "
                                "Failing for subsequent subtasks automatically.",
                                indent=16,
                            )
                            status = "failed"
                            reason = "context_length_exceeded"
                            break
                        log_warning(f"SHERLOC generation failed with BadRequestError: {e}", indent=12)
                        status = "failed"
                        reason = f"bad_request_error: {str(e)}"
                        break

                if status == "failed":
                    break

                total_generated_tokens += actual_generated_tokens

                if actual_generated_tokens >= self.cfg.inference.tokens_to_generate:
                    log_warning(
                        f"Model generated {actual_generated_tokens} tokens "
                        f"(configured limit: {self.cfg.inference.tokens_to_generate}). "
                        f"Response was likely truncated. Consider the response incomplete.",
                        indent=8,
                    )
                    llm_output["_likely_truncated"] = True

                chat_history.append(llm_output)

                # Second loop check, now including the reply that was just produced.
                if self.cfg.enable_loop_detection and len(chat_history) >= self.cfg.loop_detection_threshold:
                    is_loop, loop_info = ContextManager.detect_repetitive_tool_calls(
                        chat_history, self.cfg.loop_detection_threshold
                    )

                    if is_loop:
                        log_warning(
                            f"Loop detected! Agent has repeated the same tool call "
                            f"{loop_info['total_repetitions']} times",
                            indent=12,
                        )
                        log_debug(f"Loop details: {loop_info}", indent=12)

                        data_point["turns"] = ContextManager.inject_loop_intervention(data_point["turns"], loop_info)
                        for i, turn in enumerate(data_point["turns"]):
                            turn["turn_id"] = i
                        log_debug(
                            f"Added post-generation loop intervention as turn {len(data_point['turns']) - 1}",
                            indent=12,
                        )

                        loop_warning = {"_loop_detected": True, "_loop_info": loop_info, "_intervention_added": True}
                        chat_history[-1].update(loop_warning)

                        pattern_analysis = ContextManager.analyze_loop_patterns(chat_history)
                        log_debug(f"Pattern analysis: {pattern_analysis}", indent=12)

                if self.cfg.remove_thinking:
                    parse_reasoning(
                        llm_output, "generation", getattr(self.cfg, "thinking_end", self.cfg.end_reasoning_string)
                    )

                try:
                    extracted_block = DialogProcessor.extract_response(llm_output["generation"], self.cfg)
                except Exception as e:
                    log_error(f"Error extracting response from LLM output: {e}", indent=8)
                    log_debug(f"LLM output was: {llm_output.get('generation', 'None')[:500]}...", indent=8)
                    status = "failed"
                    reason = f"response_extraction_error: {str(e)}"
                    break

                if not extracted_block:
                    log_warning("Model failed to generate a tool use or location. Ending generation.", indent=8)
                    # Distinguish "ran out of room mid-answer" from "produced nothing parsable",
                    # since only the first is a budget problem.
                    if llm_output.get("_likely_truncated", False):
                        status = "failed"
                        reason = "response_truncated_at_token_limit"
                        log_error(
                            f"Response was truncated at token limit ({actual_generated_tokens} tokens) "
                            f"and no valid tool/location was extracted. The model needs more tokens "
                            f"to complete its response, but a buffer should have been reserved.",
                            indent=12,
                        )
                    else:
                        status = "failed"
                        reason = "no_tool_or_location_generated"
                    break

                try:
                    if data_point["turns"] and len(data_point["turns"]) > 0:
                        current_turn = data_point["turns"][-1]
                        if isinstance(current_turn, dict):
                            current_turn["assistant"] = llm_output["generation"]
                            current_turn["assistant_raw"] = llm_output.get("raw_generation", llm_output["generation"])
                            current_turn["assistant_raw_w_think"] = llm_output.get(
                                "_full_generation", llm_output["generation"]
                            )
                            current_turn["_llm_tokens"] = actual_generated_tokens
                            current_turn["_context_turn_ids"] = context_turn_ids
                            current_turn["_retry_count"] = retry_count

                            if extracted_block:
                                if extracted_block.get("type") == "tool_calls":
                                    current_turn["tool_call"] = extracted_block.get("tool_call", None)
                                elif extracted_block.get("type") == "locations":
                                    current_turn["locations"] = extracted_block.get("locations", [])
                                    current_turn["findings"] = extracted_block.get("findings")
                        else:
                            log_error(f"Current turn is not a dict: {type(current_turn)}", indent=16)
                            status = "failed"
                            reason = "invalid_turn_structure"
                            break
                    else:
                        log_error("No turns available to add assistant response", indent=12)
                        status = "failed"
                        reason = "no_turns_available"
                        break
                except Exception as e:
                    log_error(f"Error adding assistant response to turn: {e}", indent=8)
                    status = "failed"
                    reason = f"turn_update_error: {str(e)}"
                    break

                if extracted_block.get("type") == "tool_calls":
                    if "tool_call" not in extracted_block:
                        log_error(f"Missing 'tool_call' in extracted block: {extracted_block}", indent=12)
                        status = "failed"
                        reason = "missing_tool_call_in_extracted_block"
                        break

                    tool_name = extracted_block.get("tool_call", {}).get("tool", "unknown")
                    log_info(f"Executing tool: {tool_name}", indent=8)

                    # Tools are read-only and answer from the in-memory snapshot.
                    tool_output_content, tool_output_tokens = self.tool_executor.execute_tool(
                        extracted_block["tool_call"], repo_dict
                    )

                    tool_output_to_store = tool_output_content

                    if self.cfg.enable_response_length_management:
                        log_info(
                            f"Tool output: {tool_output_tokens} tokens, {len(tool_output_content)} chars", indent=12
                        )

                    if data_point["turns"] and len(data_point["turns"]) > 0:
                        current_turn = data_point["turns"][-1]
                        if isinstance(current_turn, dict):
                            current_turn["tool_output"] = tool_output_to_store
                            current_turn["_tool_tokens"] = tool_output_tokens
                            log_debug(f"Added tool output to current turn {len(data_point['turns']) - 1}", indent=16)

                            # The tool output opens the next turn: it is what the model reads next.
                            new_turn = {
                                "turn_id": len(data_point["turns"]),
                                "inputs": tool_output_to_store,
                                "assistant": "",
                                "tool_call": None,
                                "tool_output": "",
                                "_retry_count": 0,
                            }
                            data_point["turns"].append(new_turn)
                            log_debug(
                                f"Added new turn for next iteration, total turns: {len(data_point['turns'])}, "
                                f"turn_id: {new_turn['turn_id']}",
                                indent=16,
                            )
                        else:
                            log_error(f"Current turn is not a dict: {type(current_turn)}", indent=16)
                            status = "failed"
                            reason = "invalid_turn_structure_for_tool_output"
                            break
                    else:
                        log_error("No turns available to add tool output", indent=12)
                        status = "failed"
                        reason = "no_turns_for_tool_output"
                        break
                elif extracted_block.get("type") == "locations":
                    if "locations" not in extracted_block:
                        log_error(f"Missing 'locations' in extracted block: {extracted_block}", indent=12)
                        status = "failed"
                        reason = "missing_locations_in_extracted_block"
                        break
                    # The model committed to an answer, which ends the episode.
                    data_point["locations"] = extracted_block["locations"]
                    data_point["findings"] = extracted_block.get("findings")
                    status = "success"
                    reason = None
                    break

                if data_point.get("turns") and len(data_point["turns"]) > 0:
                    last_turn = data_point["turns"][-1]
                    has_assistant = isinstance(last_turn, dict) and "assistant" in last_turn
                    log_debug(
                        f"Current turn count: {len(data_point['turns'])}, last turn has assistant: {has_assistant}",
                        indent=8,
                    )
                else:
                    log_debug("No turns available to check for assistant field", indent=8)

                if cur_step == total_steps - 1 and status is None:
                    status = "failed"
                    reason = "max_steps_exceeded"
                    break

            if status is None:
                status = "failed"
                if reason is None:
                    reason = "unknown_failure"
        except Exception as e:
            import traceback

            log_error(f"Error in {instance_id}: {type(e).__name__}: {e}")
            log_error(f"Traceback:\n{traceback.format_exc()}")

            status = "failed"
            reason = f"exception: {str(e)}"

        # A failed episode still has to return a well-formed transcript, since downstream analysis
        # reads the turns of failures as well as successes.
        if "turns" not in data_point:
            log_warning("Missing 'turns' in data_point at return time, initializing empty structure")
            data_point["turns"] = []

        for i, turn in enumerate(data_point.get("turns", [])):
            if not isinstance(turn, dict):
                log_error(f"Turn {i} is not a dictionary: {type(turn)}", indent=4)
                data_point["turns"][i] = {
                    "turn_id": i,
                    "inputs": str(turn) if turn else "",
                    "assistant": "",
                    "tool_call": None,
                    "tool_output": "",
                    "_retry_count": 0,
                }
            else:
                if "inputs" not in turn:
                    turn["inputs"] = ""
                if "assistant" not in turn:
                    turn["assistant"] = ""
                if "tool_call" not in turn:
                    turn["tool_call"] = None
                if "tool_output" not in turn:
                    turn["tool_output"] = ""
                if "turn_id" not in turn:
                    turn["turn_id"] = i

                log_debug(
                    f"Final turn {i} validation - has assistant: {'assistant' in turn}, keys: {list(turn.keys())}",
                    indent=4,
                )

        if "turns" in data_point:
            log_debug(f"Returning {len(data_point['turns'])} turns")

        # Promote the reference-coverage bookkeeping from scratch keys to output fields.
        ground_truth_in_repo_percentage = data_point.get("_ground_truth_in_repo_percentage", 0.0)
        missing_ground_truth_files = data_point.get("_missing_ground_truth_files", [])

        data_point.pop("_ground_truth_in_repo_percentage", None)
        data_point.pop("_missing_ground_truth_files", None)

        log_info(f"Completed: {instance_id} (status={status})")

        return {
            "generation": chat_history,
            "total_generated_tokens": total_generated_tokens,
            "num_turns": len(chat_history),
            "status": status,
            "reason": reason,
            "turns": data_point.get("turns", []),
            "locations": data_point.get("locations"),
            "findings": data_point.get("findings"),
            "ground_truth_in_repo_percentage": ground_truth_in_repo_percentage,
            "missing_ground_truth_files": missing_ground_truth_files,
        }


GENERATION_TASK_CLASS = SherlocGenerationTask


@hydra.main(version_base=None, config_name="base_sherloc_generation_config")
def sherloc_generation(cfg: SherlocGenerationConfig):
    cfg = SherlocGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = SherlocGenerationTask(cfg)
    task.generate()


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        HELP_MESSAGE = get_help_message(
            SherlocGenerationConfig,
            server_params=server_params(),
        )
        print(HELP_MESSAGE)
    else:
        setup_logging()
        sherloc_generation()

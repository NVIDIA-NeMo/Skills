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
import json
import logging
import sys
from dataclasses import asdict, field, is_dataclass
from pathlib import Path

import hydra

from nemo_skills.inference.generate import (
    GenerationTask,
    GenerationTaskConfig,
    InferenceConfig,
)
from nemo_skills.inference.model import server_params
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    setup_logging,
)

LOG = logging.getLogger(get_logger_name(__file__))


# ---------------------------------------------------------------------------
# Evaluation response parsers (ported from google-facts reference codebase)
# ---------------------------------------------------------------------------


def parse_grounding_json(response: str) -> bool:
    """Parse JSON grounding evaluation response.

    The judge decomposes the model response into sentences and labels each as
    'supported', 'unsupported', 'contradictory', or 'no_rad'. Returns True
    only if every parsed object has label in ('supported', 'no_rad').
    """
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0]
    response = response.strip()
    # Split concatenated JSON objects
    response = response.replace("}\n", "}\n@\n@\n")
    parsed_answers = []
    for chunk in response.split("\n@\n@\n"):
        try:
            chunk = chunk.replace("\n", " ").replace("\\'", "'")
            parsed = json.loads(chunk)
            parsed_answers.append(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    if len(parsed_answers) > 0:
        return all(d.get("label") in ("supported", "no_rad") for d in parsed_answers)
    return False


def parse_grounding_implicit_span(response: str) -> bool:
    """Parse implicit-span-level grounding evaluation response.

    The judge produces a chain-of-thought analysis ending with 'Final Answer:'
    followed by 'Accurate' or 'Inaccurate'.
    """
    splits = response.split("Final Answer:")
    if len(splits) <= 1:
        return False
    final_ans = splits[1].strip().lower()
    if "inaccurate" in final_ans or "false" in final_ans:
        return False
    if "accurate" in final_ans or "true" in final_ans:
        return True
    return False


def parse_quality_json(response: str) -> bool:
    """Parse quality evaluation response.

    The judge returns a JSON object with an 'Instruction Following' field.
    Returns True unless the response has 'Major Issue(s)'.
    """
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0]
    response = response.replace("\n", " ")
    try:
        parsed = json.loads(response)
    except (json.JSONDecodeError, ValueError):
        parsed = {}

    valid_labels = ["No Issues", "Minor Issue(s)", "Major Issue(s)", "Invalid"]
    instruction_following = parsed.get("Instruction Following", "Invalid")
    if instruction_following not in valid_labels:
        instruction_following = "Invalid"
    return instruction_following != "Major Issue(s)"


def select_grounding_method(model_name: str):
    """Pick the best grounding evaluation method based on the judge model.

    Per the FACTS paper, implicit_span_level works best for Claude/Anthropic
    models, while JSON decomposition works best for Gemini and GPT.
    """
    model_lower = model_name.lower()
    if "anthropic" in model_lower or "claude" in model_lower:
        return "implicit_span_level", parse_grounding_implicit_span
    return "json", parse_grounding_json


# ---------------------------------------------------------------------------
# Evaluation prompt loader
# ---------------------------------------------------------------------------


def _load_eval_prompts() -> dict[str, str]:
    """Load FACTS evaluation prompt templates.

    Tries local file first (created by prepare.py), falls back to HuggingFace.
    """
    try:
        import nemo_skills.dataset.facts_grounding as facts_module

        path = Path(facts_module.__file__).parent / "eval_prompts.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except (ImportError, FileNotFoundError):
        pass

    LOG.info("Loading evaluation prompts from HuggingFace dataset")
    from datasets import load_dataset

    ds = load_dataset("google/FACTS-grounding-public", "evaluation_prompts", split="prompts")
    return {item["evaluation_method"]: item["evaluation_prompt"] for item in ds}


# ---------------------------------------------------------------------------
# Judge config and task
# ---------------------------------------------------------------------------


@nested_dataclass(kw_only=True)
class FactsGroundingJudgeConfig(GenerationTaskConfig):
    """FACTS Grounding judge parameters."""

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)

    prompt_config: str = "judge/facts_grounding"
    generation_key: str = "judgement"

    # Whether to skip quality evaluation (saves 2 LLM calls per sample)
    skip_quality: bool = False


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_facts_grounding_judge_config", node=FactsGroundingJudgeConfig)


class FactsGroundingJudgeTask(GenerationTask):
    def __init__(self, cfg: FactsGroundingJudgeConfig):
        super().__init__(cfg)
        self.eval_prompts = _load_eval_prompts()
        model_name = cfg.server.get("model", "")
        self.grounding_method, self.grounding_parser = select_grounding_method(model_name)
        LOG.info("Using grounding method: %s (model: %s)", self.grounding_method, model_name)

    def log_example_prompt(self, all_data):
        LOG.info(
            "FACTS Grounding Judge - evaluation prompts loaded at runtime. Grounding method: %s. Data fields: %s",
            self.grounding_method,
            list(all_data[0].keys()) if all_data else "[]",
        )

    async def _judge_call(self, prompt_text: str) -> str:
        """Make a single LLM judge call with a custom prompt."""
        if is_dataclass(self.cfg.inference):
            inference_params = asdict(self.cfg.inference)
        else:
            inference_params = dict(self.cfg.inference)

        result = await self.generate_with_semaphore(
            prompt=[{"role": "user", "content": prompt_text}],
            **inference_params,
        )
        return result.get("generation", "")

    async def process_single_datapoint(self, data_point, all_data, prompt_format=None):
        generation = data_point.get("generation", "")
        user_request = data_point.get("user_request", "")
        context_document = data_point.get("context_document", "")
        full_prompt = data_point.get("full_prompt", "")

        # Build grounding evaluation prompt
        grounding_prompt = (
            self.eval_prompts[self.grounding_method]
            .replace("{{user_request}}", user_request)
            .replace("{{context_document}}", context_document)
            .replace("{{response}}", generation)
        )

        if self.cfg.skip_quality:
            grounding_response = await self._judge_call(grounding_prompt)
            grounding_passed = self.grounding_parser(grounding_response)
            return {
                "judgement_grounding": grounding_passed,
                "judgement_grounding_raw": grounding_response,
                "judgement_quality": True,
                "judgement_quality_raw": "",
                "generation": "",
            }

        # Run grounding evaluation and reference generation in parallel
        grounding_response, reference_response = await asyncio.gather(
            self._judge_call(grounding_prompt),
            self._judge_call(full_prompt),
        )
        grounding_passed = self.grounding_parser(grounding_response)

        # Quality evaluation: compare target response against judge's reference
        quality_prompt = (
            self.eval_prompts["ineligible_responses_filter_no_context"]
            .replace("{{user_request}}", user_request)
            .replace("{{response_a}}", generation)
            .replace("{{response_b}}", reference_response)
        )
        quality_response = await self._judge_call(quality_prompt)
        quality_passed = parse_quality_json(quality_response)

        return {
            "judgement_grounding": grounding_passed,
            "judgement_grounding_raw": grounding_response,
            "judgement_quality": quality_passed,
            "judgement_quality_raw": quality_response,
            "generation": "",
        }


GENERATION_TASK_CLASS = FactsGroundingJudgeTask


@hydra.main(version_base=None, config_name="base_facts_grounding_judge_config")
def generate(cfg: FactsGroundingJudgeConfig):
    cfg = FactsGroundingJudgeConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = FactsGroundingJudgeTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    FactsGroundingJudgeConfig,
    server_params=server_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()

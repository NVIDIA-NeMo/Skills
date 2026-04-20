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
from nemo_skills.inference.model import get_model, server_params
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


GROUNDING_LABELS = ("supported", "unsupported", "contradictory", "no_rad")


def _empty_sentence_stats() -> dict:
    return {
        "sentences_total": 0,
        "sentences_supported": 0,
        "sentences_unsupported": 0,
        "sentences_contradictory": 0,
        "sentences_no_rad": 0,
        "sentence_stats_available": False,
    }


def parse_grounding_json(response: str) -> tuple[bool, dict]:
    """Parse JSON grounding evaluation response.

    The judge decomposes the model response into sentences and labels each as
    'supported', 'unsupported', 'contradictory', or 'no_rad'. Returns a tuple
    ``(accurate, sentence_stats)`` where ``accurate`` is True only if every
    parsed object has label in ('supported', 'no_rad'), and ``sentence_stats``
    counts per-label sentences (all zeros when parsing fails).
    """
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0]
    response = response.strip()
    response = response.replace("}\n", "}\n@\n@\n")
    parsed_answers: list[dict] = []
    for chunk in response.split("\n@\n@\n"):
        try:
            chunk = chunk.replace("\n", " ").replace("\\'", "'")
            parsed = json.loads(chunk)
        except (json.JSONDecodeError, ValueError):
            continue
        # Judges sometimes emit a single JSON array ``[{...}, {...}]`` instead
        # of newline-delimited objects. Flatten either shape into a list of dicts.
        if isinstance(parsed, list):
            parsed_answers.extend(item for item in parsed if isinstance(item, dict))
        elif isinstance(parsed, dict):
            parsed_answers.append(parsed)

    stats = _empty_sentence_stats()
    if not parsed_answers:
        return False, stats

    stats["sentence_stats_available"] = True
    stats["sentences_total"] = len(parsed_answers)
    for d in parsed_answers:
        label = d.get("label")
        if label in GROUNDING_LABELS:
            stats[f"sentences_{label}"] += 1

    accurate = all(d.get("label") in ("supported", "no_rad") for d in parsed_answers)
    return accurate, stats


def parse_grounding_implicit_span(response: str) -> tuple[bool, dict]:
    """Parse implicit-span-level grounding evaluation response.

    The judge produces a chain-of-thought analysis ending with 'Final Answer:'
    followed by 'Accurate' or 'Inaccurate'. Sentence-level label stats are
    unavailable for this format.
    """
    stats = _empty_sentence_stats()
    splits = response.split("Final Answer:")
    if len(splits) <= 1:
        return False, stats
    final_ans = splits[1].strip().lower()
    if "inaccurate" in final_ans or "false" in final_ans:
        return False, stats
    if "accurate" in final_ans or "true" in final_ans:
        return True, stats
    return False, stats


def parse_quality_json(response: str) -> bool:
    """Parse quality evaluation response.

    The judge returns a JSON object with an 'Instruction Following' field.
    Returns True (eligible) unless the response has 'Major Issue(s)'.
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
    model_lower = (model_name or "").lower()
    if "anthropic" in model_lower or "claude" in model_lower:
        return "implicit_span_level", parse_grounding_implicit_span
    return "json", parse_grounding_json


def make_judge_tag(model_id: str) -> str:
    """Reference-compatible judge tag: last path segment, '-'/'.' → '_'."""
    return model_id.split("/")[-1].replace("-", "_").replace(".", "_")


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

# Default 3-judge ensemble (closest available on inference-api.nvidia.com to
# the google-facts reference's gemini-3-pro / gpt-5.2 / claude-opus-4-5).
DEFAULT_JUDGE_MODELS = (
    "gcp/google/gemini-3.1-pro-preview",
    "azure/openai/gpt-5.2",
    "aws/anthropic/claude-opus-4-5",
)


@nested_dataclass(kw_only=True)
class FactsGroundingJudgeConfig(GenerationTaskConfig):
    """FACTS Grounding judge parameters."""

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)

    prompt_config: str = "judge/facts_grounding"
    generation_key: str = "judgement"

    # List of judge models to run as an ensemble. All must be reachable via
    # the endpoint configured in ``server`` (one base_url shared across judges).
    # Leave empty / None to fall back to single-judge using ``server.model``.
    judge_models: list[str] = field(default_factory=lambda: list(DEFAULT_JUDGE_MODELS))

    # When True, skip the eligibility (quality) check — grounding only.
    skip_quality: bool = False


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_facts_grounding_judge_config", node=FactsGroundingJudgeConfig)


class FactsGroundingJudgeTask(GenerationTask):
    def __init__(self, cfg: FactsGroundingJudgeConfig):
        # ``GenerationTask.__init__`` calls ``self.setup_llm()`` — make sure
        # ``self.judges`` exists before that happens.
        self.judges: list[dict] = []
        super().__init__(cfg)
        self.eval_prompts = _load_eval_prompts()

    def setup_llm(self):
        # Build one LLM client per judge, all sharing the same server config
        # except for the ``model`` field. Returns the first judge's LLM so
        # anything in the base class that touches ``self.llm`` still works.
        base_server_config = dict(self.cfg.server)
        judge_ids = list(self.cfg.judge_models) if self.cfg.judge_models else []
        if not judge_ids:
            default_id = base_server_config.get("model")
            if not default_id:
                raise ValueError(
                    "No judge_models set and server.model is empty; set cfg.judge_models or cfg.server.model."
                )
            judge_ids = [default_id]

        output_dir = str(Path(self.cfg.output_file).parent) if self.cfg.output_file else ""

        for model_id in judge_ids:
            cfg_copy = dict(base_server_config)
            cfg_copy["model"] = model_id
            try:
                llm = get_model(
                    **cfg_copy,
                    tokenizer=None,
                    data_dir="",
                    output_dir=output_dir,
                )
            except TypeError:
                # Some model backends don't accept tokenizer kwarg.
                llm = get_model(**cfg_copy, data_dir="", output_dir=output_dir)

            method, parser = select_grounding_method(model_id)
            self.judges.append(
                {
                    "tag": make_judge_tag(model_id),
                    "model_id": model_id,
                    "llm": llm,
                    "grounding_method": method,
                    "grounding_parser": parser,
                }
            )
            LOG.info(
                "FACTS judge registered: tag=%s model=%s grounding_method=%s",
                self.judges[-1]["tag"],
                model_id,
                method,
            )

        return self.judges[0]["llm"]

    def log_example_prompt(self, all_data):
        LOG.info(
            "FACTS Grounding judges: %s. Data fields: %s",
            [j["tag"] for j in self.judges],
            list(all_data[0].keys()) if all_data else "[]",
        )

    def _inference_params(self) -> dict:
        if is_dataclass(self.cfg.inference):
            return asdict(self.cfg.inference)
        return dict(self.cfg.inference)

    @staticmethod
    def _is_reasoning_judge(model_id: str) -> bool:
        """Match openai.OpenAIModel._is_reasoning_model semantics.

        GPT-5 and o-series reject non-zero ``temperature``; strip it for them
        so the ensemble can run with the default ``inference.temperature=0.5``
        that other judges (Claude, Gemini) expect.
        """
        name = (model_id or "").lower()
        if "gpt-5" in name:
            return True
        # azure/openai/o1, .../o3, .../o4-mini, ...
        tail = name.rsplit("/", 1)[-1]
        return bool(tail) and tail[0] == "o" and len(tail) > 1 and tail[1].isdigit()

    def _inference_params_for(self, judge: dict) -> dict:
        params = self._inference_params()
        model_name = (judge.get("model_id") or "").lower()
        if self._is_reasoning_judge(model_name):
            # GPT-5 / o-series: explicitly unset temperature (None lets the
            # base model wrapper skip it entirely).
            params["temperature"] = None
            params["top_p"] = None
        elif "gemini" not in model_name:
            # Match google-facts reference: only Gemini accepts top_p alongside
            # temperature; Claude/Bedrock rejects both being set together.
            params["top_p"] = None
        return params

    async def _judge_call(self, judge: dict, prompt_text: str) -> str:
        """Single LLM judge call with a custom prompt, throttled by the shared semaphore."""
        params = self._inference_params_for(judge)
        async with self.semaphore:
            result = await judge["llm"].generate_async(
                prompt=[{"role": "user", "content": prompt_text}],
                **params,
            )
        return result.get("generation", "") if isinstance(result, dict) else ""

    async def _grounding_one(self, judge: dict, user_request: str, context_document: str, response: str):
        template = self.eval_prompts[judge["grounding_method"]]
        prompt = (
            template.replace("{{user_request}}", user_request)
            .replace("{{context_document}}", context_document)
            .replace("{{response}}", response)
        )
        raw = await self._judge_call(judge, prompt)
        passed, stats = judge["grounding_parser"](raw)
        return passed, stats, raw

    async def _reference_one(self, judge: dict, full_prompt: str) -> str:
        return await self._judge_call(judge, full_prompt)

    async def _quality_one(
        self, judge: dict, user_request: str, test_response: str, reference_response: str
    ) -> tuple[bool, str]:
        template = self.eval_prompts["ineligible_responses_filter_no_context"]
        prompt = (
            template.replace("{{user_request}}", user_request)
            .replace("{{response_a}}", test_response)
            .replace("{{response_b}}", reference_response)
        )
        raw = await self._judge_call(judge, prompt)
        return parse_quality_json(raw), raw

    async def process_single_datapoint(self, data_point, all_data, prompt_format=None):
        generation = data_point.get("generation", "")
        user_request = data_point.get("user_request", "")
        context_document = data_point.get("context_document", "")
        full_prompt = data_point.get("full_prompt", "")

        grounding_tasks = [self._grounding_one(j, user_request, context_document, generation) for j in self.judges]
        if self.cfg.skip_quality:
            grounding_results = await asyncio.gather(*grounding_tasks)
            reference_results = [""] * len(self.judges)
        else:
            reference_tasks = [self._reference_one(j, full_prompt) for j in self.judges]
            grounding_results, reference_results = await asyncio.gather(
                asyncio.gather(*grounding_tasks),
                asyncio.gather(*reference_tasks),
            )

        if self.cfg.skip_quality:
            quality_results = [(True, "")] * len(self.judges)
        else:
            quality_tasks = [
                self._quality_one(j, user_request, generation, ref) for j, ref in zip(self.judges, reference_results)
            ]
            quality_results = await asyncio.gather(*quality_tasks)

        grounding_per_judge: dict[str, bool] = {}
        quality_per_judge: dict[str, bool] = {}
        grounding_raw_per_judge: dict[str, str] = {}
        quality_raw_per_judge: dict[str, str] = {}
        sentence_stats_per_judge: dict[str, dict] = {}

        for judge, (g_passed, g_stats, g_raw), (q_passed, q_raw) in zip(
            self.judges, grounding_results, quality_results
        ):
            tag = judge["tag"]
            grounding_per_judge[tag] = bool(g_passed)
            quality_per_judge[tag] = bool(q_passed)
            grounding_raw_per_judge[tag] = g_raw
            quality_raw_per_judge[tag] = q_raw
            sentence_stats_per_judge[tag] = g_stats

        # Reference-style aggregates.
        grounding_mean = sum(grounding_per_judge.values()) / max(len(grounding_per_judge), 1)
        quality_consensus = bool(quality_per_judge and all(quality_per_judge.values()))

        # Backward-compat top-level fields (used by legacy consumers).
        legacy_grounding = grounding_mean >= 1.0  # unanimous
        legacy_quality = quality_consensus

        return {
            "judgement_grounding_per_judge": grounding_per_judge,
            "judgement_quality_per_judge": quality_per_judge,
            "grounding_raw_per_judge": grounding_raw_per_judge,
            "quality_raw_per_judge": quality_raw_per_judge,
            "sentence_stats_per_judge": sentence_stats_per_judge,
            "grounding_score_mean": grounding_mean,
            "quality_check_passed": quality_consensus,
            "judgement_grounding": legacy_grounding,
            "judgement_quality": legacy_quality,
            "judge_tags": [j["tag"] for j in self.judges],
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

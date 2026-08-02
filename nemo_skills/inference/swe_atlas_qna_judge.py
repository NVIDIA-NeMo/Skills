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

import asyncio
import json
import logging
import sys
from dataclasses import field

import hydra

from nemo_skills.code_execution.sandbox import sandbox_params
from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class SweAtlasQnAJudgeConfig(GenerationTaskConfig):
    """SWE-Atlas-QnA per-criterion judge configuration."""

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)
    prompt_config: str = "judge/swe-atlas-qna"
    generation_key: str = "judgement"
    add_generation_stats: bool = False

    max_judgement_attempts: int = 8
    judgement_retry_delay: float = 1.0


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_swe_atlas_qna_judge_config", node=SweAtlasQnAJudgeConfig)


def _remove_trailing_commas(text: str) -> str:
    """Remove commas before closing braces/brackets without modifying strings."""
    output = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue

        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                index += 1
                continue

        output.append(char)
        index += 1
    return "".join(output)


def _decode_first_json_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _extract_json_object(text: str) -> dict:
    """Extract the first JSON object from a possibly fenced judge response."""
    value = _decode_first_json_object(text)
    if value is not None:
        return value

    value = _decode_first_json_object(_remove_trailing_commas(text))
    if value is not None:
        return value
    raise ValueError("Judge response does not contain a valid JSON object")


def _extract_rating(text: str, criterion: dict) -> dict:
    parsed = _extract_json_object(text)
    ratings = parsed["ratings"]
    if not isinstance(ratings, list) or len(ratings) != 1 or not isinstance(ratings[0], dict):
        raise ValueError("Judge response must contain exactly one item in 'ratings'")

    rating = ratings[0]
    if rating["criterion_id"] != criterion["id"]:
        raise ValueError("Judge response criterion_id does not match the requested criterion")
    if rating["status"] not in ("YES", "NO") or str(rating["score"]) not in ("0", "1"):
        raise ValueError("Judge response has an invalid status or score")
    if (rating["status"] == "YES") != (str(rating["score"]) == "1"):
        raise ValueError("Judge response status and score disagree")

    rating["score"] = str(rating["score"])
    rating["rubric_statement"] = criterion["title"]
    return rating


class SweAtlasQnAJudgeTask(GenerationTask):
    """Judge every SWE-Atlas-QnA rubric criterion with an independent request."""

    @staticmethod
    def _parse_rubric(data_point):
        rubric = data_point.get("rubric")
        if isinstance(rubric, str):
            rubric = json.loads(rubric)
        if not isinstance(rubric, list) or not rubric:
            raise ValueError("SWE-Atlas-QnA rubric must be a non-empty JSON list")
        return rubric

    @staticmethod
    def _add_criterion_fields(data_point, criterion):
        return {
            **data_point,
            "criterion_id": criterion["id"],
            "rubric_statement": criterion["title"],
            "rubric_type": criterion.get("annotations", {}).get("type", ""),
        }

    def log_example_prompt(self, data):
        """Render the example prompt with one expanded rubric criterion."""
        if not data:
            return
        data_point = data[0]
        criterion = self._parse_rubric(data_point)[0]
        criterion_data = self._add_criterion_fields(data_point, criterion)
        LOG.info(
            "Example prompt:\nData dictionary: %s\nPrompt: %s",
            criterion_data,
            self.fill_prompt(criterion_data, data),
        )

    async def _judge_criterion(self, data_point, criterion, all_data, prompt_format):
        criterion_data = self._add_criterion_fields(data_point, criterion)
        max_attempts = max(1, self.cfg.max_judgement_attempts)
        retry_delay = max(0.0, self.cfg.judgement_retry_delay)

        for attempt in range(1, max_attempts + 1):
            result = await super().process_single_datapoint(criterion_data, all_data, prompt_format)
            raw_judgement = result["generation"] or ""
            try:
                rating = _extract_rating(raw_judgement, criterion)
                return rating, raw_judgement, result
            except (KeyError, TypeError, ValueError) as error:
                if attempt < max_attempts:
                    LOG.warning(
                        "Could not parse criterion %s judgement on attempt %d/%d: %s. Retrying.",
                        criterion["id"],
                        attempt,
                        max_attempts,
                        error,
                    )
                    if retry_delay:
                        await asyncio.sleep(retry_delay)
                    continue

                LOG.warning(
                    "Could not parse criterion %s judgement after %d attempts: %s",
                    criterion["id"],
                    max_attempts,
                    error,
                )
                rating = {
                    "criterion_id": criterion["id"],
                    "rubric_statement": criterion["title"],
                    "parse_error": str(error),
                }
                return rating, raw_judgement, result

    async def process_single_datapoint(self, data_point, all_data, prompt_format=None):
        rubric = self._parse_rubric(data_point)

        results = await asyncio.gather(
            *[self._judge_criterion(data_point, criterion, all_data, prompt_format) for criterion in rubric]
        )
        ratings, raw_judgements, generation_results = zip(*results)

        output = {
            "generation": json.dumps(
                {"ratings": list(ratings), "raw_judgements": list(raw_judgements)},
                ensure_ascii=False,
            )
        }
        for stats_key in ("num_generated_tokens", "num_input_tokens"):
            stats = [result.get(stats_key) for result in generation_results]
            if all(stat is not None for stat in stats):
                output[stats_key] = sum(stats)
        return output


GENERATION_TASK_CLASS = SweAtlasQnAJudgeTask


@hydra.main(version_base=None, config_name="base_swe_atlas_qna_judge_config")
def generate(cfg: SweAtlasQnAJudgeConfig):
    cfg = SweAtlasQnAJudgeConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    SweAtlasQnAJudgeTask(cfg).generate()


HELP_MESSAGE = get_help_message(
    SweAtlasQnAJudgeConfig,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()

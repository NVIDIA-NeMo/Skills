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

import hydra

from nemo_skills.code_execution.sandbox import sandbox_params
from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


def _extract_json_object(text: str) -> dict:
    """Extract the first JSON object from a possibly fenced judge response."""
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
    raise ValueError("Judge response does not contain a valid JSON object")


def _extract_rating(text: str, criterion: dict) -> dict:
    parsed = _extract_json_object(text)
    ratings = parsed.get("ratings")
    if not isinstance(ratings, list) or len(ratings) != 1 or not isinstance(ratings[0], dict):
        raise ValueError("Judge response must contain exactly one item in 'ratings'")

    rating = ratings[0]
    if rating.get("criterion_id") != criterion["id"]:
        raise ValueError("Judge response criterion_id does not match the requested criterion")
    rubric_statement = rating.get("rubric_statement")
    if not isinstance(rubric_statement, str) or " ".join(rubric_statement.split()) != " ".join(
        criterion["title"].split()
    ):
        raise ValueError("Judge response rubric_statement does not match the requested criterion")
    if rating.get("status") not in ("YES", "NO") or str(rating.get("score")) not in ("0", "1"):
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
        result = await super().process_single_datapoint(criterion_data, all_data, prompt_format)
        raw_judgement = result.get("generation") or ""
        try:
            rating = _extract_rating(raw_judgement, criterion)
        except (KeyError, TypeError, ValueError) as error:
            LOG.warning("Could not parse criterion %s judgement: %s", criterion.get("id"), error)
            rating = {
                "criterion_id": criterion.get("id"),
                "rubric_statement": criterion.get("title"),
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


@hydra.main(version_base=None, config_name="base_generation_config")
def generate(cfg: GenerationTaskConfig):
    cfg = GenerationTaskConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    SweAtlasQnAJudgeTask(cfg).generate()


HELP_MESSAGE = get_help_message(
    GenerationTaskConfig,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()

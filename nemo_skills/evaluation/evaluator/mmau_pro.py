# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import json
import logging
import re
from typing import Any

from tqdm import tqdm

from nemo_skills.evaluation.evaluator.base import BaseEvaluatorConfig
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def eval_mmau_pro(cfg):
    """Evaluate MMAU-Pro instruction following questions using AIF (Audio Instruction Following) format.

    This evaluator handles instruction following evaluation for MMAU-Pro benchmark.
    Other question types are handled by different evaluation methods:
    - Closed-form questions: Evaluated by nvembed_judge.py using NVEmbed similarity matching
    - Open-ended questions: Evaluated by LLM judge (Qwen) using judge/mmau-pro prompt config
    """
    eval_config = BaseEvaluatorConfig(**cfg)

    jsonl_file = eval_config.input_file
    LOG.info(f"Evaluating instruction following questions in {jsonl_file}")

    with open(jsonl_file, "rt", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]

    LOG.info(f"Processing {len(data)} instruction following samples")

    for idx, sample in enumerate(tqdm(data, desc="Evaluating samples")):
        data[idx] = evaluate_instruction_following_sample(sample)

    # Write all results at once
    with open(jsonl_file, "wt", encoding="utf-8") as fout:
        for sample in data:
            fout.write(json.dumps(sample) + "\n")

    LOG.info(f"Evaluation completed for {jsonl_file}")


def evaluate_instruction_following_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a single instruction following sample."""
    sample = sample.copy()
    generation = sample.get("generation", "").strip()

    if not generation:
        LOG.info("Empty generation detected for instruction following question")
        sample["is_correct"] = False
        sample["error"] = "empty_generation"
        return sample

    success = evaluate_aif_constraints(generation, sample["task_identifier"], sample["kwargs"], sample)
    sample["is_correct"] = success
    return sample


def evaluate_aif_constraints(
    response: str, task_identifier: str, kwargs: dict[str, Any], sample_data: dict[str, Any]
) -> bool:
    def count_words(text):
        return len(text.split())

    def count_sentences(text):
        return len([s for s in re.split(r"[.!?]+", text.strip()) if s.strip()])

    def count_paragraphs(text):
        return len([p for p in text.split("***") if p.strip()])

    def count_bullet_points(text):
        return len(re.findall(r"(?:^|\n)\s*\*\s+", text))

    def count_highlighted_sections(text):
        return len(re.findall(r"\*([^*]+)\*", text))

    def count_placeholders(text):
        return len(re.findall(r"\[[^\]]+\]", text))

    def count_capital_words(text):
        return len([word for word in text.split() if word.isupper()])

    def count_keyword_frequency(text, keyword):
        return len(re.findall(r"\b" + re.escape(keyword.lower()) + r"\b", text.lower()))

    def has_title(text):
        return bool(re.search(r"<<[^>]+>>", text))

    def has_postscript(text, marker):
        return re.sub(r"[^a-zA-Z]", "", marker).lower() in re.sub(r"[^a-zA-Z]", "", text).lower()

    def starts_with_phrase(text, phrase):
        return re.sub(r"[^a-zA-Z ]", "", text).lower().startswith(re.sub(r"[^a-zA-Z ]", "", phrase).lower())

    def ends_with_phrase(text, phrase):
        return re.sub(r"[^a-zA-Z ]", "", text).lower().endswith(re.sub(r"[^a-zA-Z ]", "", phrase).lower())

    def is_wrapped_in_quotes(text):
        return text.strip().startswith('"') and text.strip().endswith('"')

    def has_no_commas(text):
        return "," not in text

    def check_sections(text, num_sections, splitter):
        sections = [s for s in re.split(rf"\s*{re.escape(splitter)}\s*", text.strip()) if s.strip()]
        return len(sections) == num_sections

    checks = {
        "Include Keywords": lambda: all(k.lower() in response.lower() for k in kwargs.get("keywords", "").split(", ")),
        "Keyword Frequency": lambda: count_keyword_frequency(response, kwargs.get("keyword", ""))
        == kwargs.get("N", 0),
        "Forbidden Words": lambda: not any(
            w.lower() in response.lower() for w in kwargs.get("forbidden_words", "").split(", ")
        ),
        "Number Paragraphs": lambda: count_paragraphs(response) == kwargs.get("N", 0),
        "Number Words (at least)": lambda: count_words(response) >= kwargs.get("N", 0),
        "Number Words (at most)": lambda: count_words(response) <= kwargs.get("N", 0),
        "Number Words (range)": lambda: kwargs.get("N1", 0) <= count_words(response) <= kwargs.get("N2", 999),
        "Number Sentences (at least)": lambda: count_sentences(response) >= kwargs.get("N", 0),
        "Number Sentences (at most)": lambda: count_sentences(response) <= kwargs.get("N", 0),
        "Number Sentences (range)": lambda: kwargs.get("N1", 0) <= count_sentences(response) <= kwargs.get("N2", 999),
        "Postscript": lambda: has_postscript(response, kwargs.get("postscript_marker", "")),
        "Number Placeholder": lambda: count_placeholders(response) >= kwargs.get("N", 0),
        "Number Bullets": lambda: count_bullet_points(response) == kwargs.get("N", 0),
        "Title": lambda: has_title(response),
        "Minimum Number Highlighted Section": lambda: count_highlighted_sections(response) >= kwargs.get("N", 0),
        "Multiple Sections": lambda: check_sections(response, kwargs.get("N", 0), kwargs.get("section_splitter", "")),
        "Repeat Prompt": lambda: response.strip()
        .lower()
        .startswith(sample_data.get("prompt_transcription", "").strip().lower()),
        "Two Responses": lambda: len(response.split("******")) == 2
        and response.split("******")[0].lower().strip() != response.split("******")[1].lower().strip(),
        "All Uppercase": lambda: response.isupper(),
        "All Lowercase": lambda: response.islower(),
        "All-capital Words (at least)": lambda: count_capital_words(response) >= kwargs.get("N", 0),
        "All-capital Words (at most)": lambda: count_capital_words(response) <= kwargs.get("N", 0),
        "All-capital Words (range)": lambda: kwargs.get("N1", 0)
        <= count_capital_words(response)
        <= kwargs.get("N2", 999),
        "Start Checker": lambda: starts_with_phrase(response, kwargs.get("start_phrase", "")),
        "End Checker": lambda: ends_with_phrase(response, kwargs.get("end_phrase", "")),
        "Quotation": lambda: is_wrapped_in_quotes(response),
        "No Commas": lambda: has_no_commas(response),
    }

    return checks.get(task_identifier, lambda: False)()

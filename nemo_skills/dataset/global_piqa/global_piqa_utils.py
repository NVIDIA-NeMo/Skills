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

from datasets import get_dataset_config_names, load_dataset


def supported_languages() -> list[str]:
    return get_dataset_config_names("mrlbenchmarks/global-piqa-nonparallel")


def load_global_piqa_datasets(languages: list[str], split: str = "test") -> dict[str, list[dict]]:
    return {lang: load_dataset("mrlbenchmarks/global-piqa-nonparallel", lang)[split] for lang in languages}


def digit_to_letter(digit: int) -> str:
    return chr(ord("A") + digit)


class Schema:
    PROMPT = "prompt"
    SOLUTION0 = "solution0"
    SOLUTION1 = "solution1"
    LABEL = "label"


# Prompt and Regex are adapted from:
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/global_piqa/prompted/_template

QUERY_TEMPLATE = """
Given the following situation, which option is more likely to be correct?

Situation:
{prompt} ...

Option A: {solution0}

Option B: {solution1}

Your response should end with "The best answer is: [answer_letter]" where [answer_letter] is one of A or B.
""".strip()


ANSWER_REGEX1 = r"(?i)[Tt]he (?:[Bb]est [Aa]nswer|[Ff]inal [Aa]nswer|[Aa]nswer)[^A-B]*([A-B])"
ANSWER_REGEX2 = r"(?i)[Aa]nswer\s*:[^A-B]*([A-B])"
ANSWER_REGEX3 = r"(?i)\\boxed\{([A-B])\}"

# Fallback: greedily match everything, then capture the last A/B in the response.
# This is not in lm-evaluation-harness and is where we diverge from the original benchmark.
LETTER_REGEX = r"\b\(?\s*([A-B])\s*\)?\.?\b"
GREEDY_REGEX = r"[\s\S]*" + LETTER_REGEX
EXTRACT_REGEX = [ANSWER_REGEX1, ANSWER_REGEX2, ANSWER_REGEX3, GREEDY_REGEX]


def get_mcq_fields(entry: dict) -> dict:
    options_dict = {digit_to_letter(i): entry[f"solution{i}"] for i in range(2)}
    options_text = "\n".join(f"Option {letter}: {option}" for letter, option in options_dict.items())
    question = QUERY_TEMPLATE.format(**entry)
    return {"question": question, "options": options_text, **options_dict}

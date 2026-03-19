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


import os
import urllib.request
from pathlib import Path

import pandas

SUPPORTED_LANGUAGES = [
    "AR-XY",  # Arabic
    "BN-BD",  # Bengali
    "DE-DE",  # German
    "ES-LA",  # Spanish
    "FR-FR",  # French
    "HI-IN",  # Hindi
    "ID-ID",  # Indonesian
    "IT-IT",  # Italian
    "JA-JP",  # Japanese
    "KO-KR",  # Korean
    "PT-BR",  # Portuguese
    "ZH-CN",  # Chinese
    "SW-KE",  # Swahili
    "YO-NG",  # Yoruba
]

subject2category = {
    "abstract_algebra": "stem",
    "anatomy": "other",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = "(?i){}[ \t]*([A-D]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])"
# All the different ways "Answer" is written in different languages
MULTILINGUAL_ANSWER_REGEXES = [
    "Answer\s*:",
    "Answer\s*:​​​​​​",  # Korean invisible character
    "উত্তর\s*:",
    "उत्तर\s*:",
    "উত্তরঃ",
    "উত্তর\s*:",
    "Antwort\s*:",
    "답변\s*:",
    "정답\s*:",
    "답\s*:",
    "答案\s*：",
    "答案\s*:",
    "答\s*：",
    "答\s*:",
    "答复\s*：",
    "答曰\s*：",
    "الإجابة:",
    "الجواب:",
    "إجابة:",
    "الإجابة النهائية:",
    "الإجابة الصحيحة:",
    "الإجابة الصحيحة هي:",
    "الإجابة هي:",
    "الجواب النهائي:",
    "Respuesta\s*:",
    "Risposta\s*:",
    "答え\s*:",
    "答え\s*：",
    "回答\s*:",
    "回答\s*：",
    "解答\s*:",
    "Jawaban\s*:",
    "Réponse\s*:",
    "Resposta\s*:",
    "Jibu\s*:",
    "Idahun\s*:",
    "Ìdáhùn\s*:",
    "Idáhùn\s*:",
    "Àmọ̀nà\s*:",
    "Àdáhùn\s*:",
    "Ànúgọ\s*:",
    "Àṣàyàn\s*:",
]


class Schema:
    ANSWER: str = "Answer"
    QUESTION: str = "Question"
    SUBJECT: str = "Subject"
    OPTIONS: list[str] = ["A", "B", "C", "D"]


def download_mmmlu_datasets(languages: list[str]) -> dict[str, list[dict]]:
    OPENAI_PUBLIC_URL = "https://openaipublic.blob.core.windows.net/simple-evals/{}"
    data_dir = Path(__file__).absolute().parent
    mmmlu_datasets = {}
    for language in languages:
        suffix = "mmlu.csv" if language == "EN-US" else f"mmlu_{language}.csv"
        download_dst_path = data_dir / suffix
        if os.path.exists(download_dst_path):
            print(f"Skipping download of {suffix} because it already exists")
        else:
            url = OPENAI_PUBLIC_URL.format(suffix)
            urllib.request.urlretrieve(url, download_dst_path)
            if not os.path.exists(download_dst_path):
                raise RuntimeError(f"Failed to download {suffix}")

        df = pandas.read_csv(download_dst_path, index_col=0)
        examples = [row.to_dict() for _, row in df.iterrows()]
        mmmlu_datasets[language] = examples
    return mmmlu_datasets


def format_multichoice_question(row):
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)


def get_mcq_fields(entry: dict):
    options_dict = {letter: entry[letter] for letter in Schema.OPTIONS}
    options_text = "\n".join(f"{letter}) {option}" for letter, option in options_dict.items())
    prompt = format_multichoice_question(entry)
    return {"question": prompt, "options": options_text, **options_dict}

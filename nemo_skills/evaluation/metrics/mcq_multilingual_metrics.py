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

import logging

from langdetect import detect, DetectorFactory, LangDetectException

from nemo_skills.evaluation.metrics.base import as_percentage
from nemo_skills.evaluation.metrics.math_metrics import MathMetrics
from nemo_skills.utils import get_logger_name

# Set seed for consistent results
DetectorFactory.seed = 42


LOG = logging.getLogger(get_logger_name(__file__))


class MCQMultilingualMetrics(MathMetrics):
    def __init__(
        self,
        compute_no_answer: bool = True,
        question_key: str = "problem",
        answer_key: str = "predicted_answer",
    ):
        super().__init__(compute_no_answer=compute_no_answer, question_key=question_key, answer_key=answer_key)

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        correctness_dict = super()._get_score_dict(prediction)

        if "target_language" in prediction:
            language_correct = self._detect_language(prediction["generation"]) == prediction["target_language"]
        else:
            language_correct = False

        if "symbolic_correct" in correctness_dict:
            correctness_dict["symbolic_and_language_correct"] = language_correct and correctness_dict["symbolic_correct"]
        if "judge_correct" in correctness_dict:
            correctness_dict["judge_and_language_correct"] = language_correct and correctness_dict["judge_correct"]
        if "judge_correct" in correctness_dict and "symbolic_correct" in correctness_dict:
            correctness_dict["both_language_correct"] = (
                correctness_dict["symbolic_and_language_correct"] and correctness_dict["judge_and_language_correct"]
            )
            correctness_dict["any_language_correct"] = (
                correctness_dict["symbolic_and_language_correct"] or correctness_dict["judge_and_language_correct"]
            )
        return correctness_dict

    def metrics_to_print(self):
        metrics = super().metrics_to_print()
        metrics["symbolic_and_language_correct"] = as_percentage
        return metrics

    def _detect_language(self, text):
        """
        Detect the language of a given text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            str: The detected language code (e.g., 'en', 'es', 'fr') or 'unknown'
        """
        if not text or not text.strip():
            return 'unknown'

        try:
            # Clean the text by removing excessive whitespace
            cleaned_text = ' '.join(text.split())
            if len(cleaned_text) < 3:
                return 'unknown'

            detected_lang = detect(cleaned_text)
            return detected_lang
        except LangDetectException:
            return 'unknown'

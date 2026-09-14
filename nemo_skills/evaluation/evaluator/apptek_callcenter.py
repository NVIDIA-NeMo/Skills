# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""AppTek Call-Center Dialogues normalization helpers."""

from __future__ import annotations

from functools import lru_cache

import jiwer

from nemo_skills.evaluation.evaluator.apptek_callcenter_word_mappings import word_dict_to_map


@lru_cache
def _english_text_normalizer():
    """Return the Whisper English text normalizer used by the AppTek scorer."""
    from whisper_normalizer.english import EnglishTextNormalizer

    return EnglishTextNormalizer()


@lru_cache
def _ref_cleaner():
    """Return the reference-only pre-cleaner from the official scorer."""
    return jiwer.SubstituteRegexes({r"\b(\w+)~(?=\W|$)": ""})


@lru_cache
def _pred_cleaner():
    """Return the prediction-only pre-cleaner from the official scorer."""
    return jiwer.SubstituteWords({"oh": ""})


@lru_cache
def _common_string_transform():
    """Return the shared post-Whisper transform before tokenization."""
    return jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.SubstituteWords(word_dict_to_map),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ]
    )


def normalize_apptek_callcenter_text(text: str, is_prediction: bool) -> str:
    """Normalize text according to the AppTek Call-Center scoring protocol."""
    if is_prediction:
        text = _pred_cleaner().process_string(text)
    else:
        text = _ref_cleaner().process_string(text)

    text = _english_text_normalizer()(text)
    return _common_string_transform()(text)

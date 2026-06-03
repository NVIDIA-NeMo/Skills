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

"""Static CS-FLEURS language metadata.

CS-FLEURS metadata identifies languages by ISO 639-3 codes (e.g. ``ara``,
``cmn``, ``deu``) inside a ``"<matrix>-<embedded>"`` ``language`` field, which
differs from the BCP-47 locales used by the plain ``fleurs`` benchmark. This
table maps each code to a display name and an ISO 639-1 code; the latter is
passed to the audio evaluator as ``src_lang`` so that its number-normalization
(``num2words``) and multilingual normalizer behave correctly. Codes without a
ISO 639-1 form map to ``None`` (number normalization is then skipped, which the
evaluator handles gracefully).
"""

from __future__ import annotations

# ISO 639-3 code -> (display name, ISO 639-1 code or None for num2words/normalization)
CS_FLEURS_LANGUAGES: dict[str, tuple[str, str | None]] = {
    "ara": ("Arabic", "ar"),
    "ben": ("Bengali", "bn"),
    "bul": ("Bulgarian", "bg"),
    "cat": ("Catalan", "ca"),
    "ceb": ("Cebuano", None),
    "ces": ("Czech", "cs"),
    "cmn": ("Mandarin Chinese", "zh"),
    "cym": ("Welsh", "cy"),
    "deu": ("German", "de"),
    "ell": ("Greek", "el"),
    "eng": ("English", "en"),
    "fin": ("Finnish", "fi"),
    "fra": ("French", "fr"),
    "guj": ("Gujarati", "gu"),
    "heb": ("Hebrew", "he"),
    "hin": ("Hindi", "hi"),
    "hun": ("Hungarian", "hu"),
    "ind": ("Indonesian", "id"),
    "isl": ("Icelandic", "is"),
    "ita": ("Italian", "it"),
    "jav": ("Javanese", "jv"),
    "jpn": ("Japanese", "ja"),
    "kan": ("Kannada", "kn"),
    "kaz": ("Kazakh", "kk"),
    "khm": ("Khmer", "km"),
    "kir": ("Kyrgyz", "ky"),
    "kor": ("Korean", "ko"),
    "lao": ("Lao", "lo"),
    "lav": ("Latvian", "lv"),
    "lug": ("Ganda", "lg"),
    "mal": ("Malayalam", "ml"),
    "mar": ("Marathi", "mr"),
    "mya": ("Burmese", "my"),
    "nld": ("Dutch", "nl"),
    "pan": ("Punjabi", "pa"),
    "pol": ("Polish", "pl"),
    "por": ("Portuguese", "pt"),
    "ron": ("Romanian", "ro"),
    "rus": ("Russian", "ru"),
    "slk": ("Slovak", "sk"),
    "spa": ("Spanish", "es"),
    "swe": ("Swedish", "sv"),
    "tam": ("Tamil", "ta"),
    "tel": ("Telugu", "te"),
    "tgk": ("Tajik", "tg"),
    "tgl": ("Tagalog", "tl"),
    "tha": ("Thai", "th"),
    "tur": ("Turkish", "tr"),
    "ukr": ("Ukrainian", "uk"),
    "urd": ("Urdu", "ur"),
    "uzb": ("Uzbek", "uz"),
    "vie": ("Vietnamese", "vi"),
    "yor": ("Yoruba", "yo"),
    "yue": ("Cantonese Chinese", None),
    "zlm": ("Malay", "ms"),
}

# Matrix languages scored with Character Error Rate instead of Word Error Rate
# (scriptio-continua: no explicit word boundaries). Mirrors the fleurs benchmark
# CER_LOCALES, expressed in ISO 639-3.
CER_LANGS: frozenset[str] = frozenset(
    {
        "cmn",  # Mandarin Chinese
        "yue",  # Cantonese Chinese
        "jpn",  # Japanese
        "kor",  # Korean
        "tha",  # Thai
        "lao",  # Lao
        "mya",  # Burmese
        "khm",  # Khmer
        "vie",  # Vietnamese
    }
)


def split_pair(language: str) -> tuple[str, str]:
    """Split a CS-FLEURS ``language`` field (``"<matrix>-<embedded>"``).

    Returns ``(matrix_code, embedded_code)``. If no separator is present the
    whole string is treated as the matrix language with an empty embedded code.
    """
    parts = language.replace("_", "-").split("-")
    matrix = parts[0]
    embedded = parts[1] if len(parts) > 1 else ""
    return matrix, embedded


def get_lang_name(code: str) -> str:
    """Display name for an ISO 639-3 code, falling back to the code itself."""
    entry = CS_FLEURS_LANGUAGES.get(code)
    return entry[0] if entry else code


def get_iso1(code: str) -> str | None:
    """ISO 639-1 code for an ISO 639-3 code, or None when unavailable."""
    entry = CS_FLEURS_LANGUAGES.get(code)
    return entry[1] if entry else None


def uses_cer(matrix_code: str) -> bool:
    """Whether a code-switched pair with this matrix language is scored with CER."""
    return matrix_code in CER_LANGS

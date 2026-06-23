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

"""Unit tests for the Mixed Error Rate (MER) tokenizer used by code-switched ASR.

``mixed_segment`` is pure ``re``/``unicodedata`` and runs without optional deps.
The ``evaluate_mer`` tests need ``jiwer`` (and the normalization stack), so they
``importorskip`` and run in the GPU/CI image rather than the bare sandbox.
"""

from nemo_skills.evaluation.evaluator.audio import mixed_segment


class TestMixedSegmentCoverage:
    """Each scriptio-continua script must split into individual units, while
    Latin runs stay as whitespace-delimited words."""

    def test_latin_only_unchanged(self):
        assert mixed_segment("hello world") == ["hello", "world"]

    def test_empty(self):
        assert mixed_segment("") == []

    def test_mandarin_chars_split_english_words_kept(self):
        # The core MER property for cmn-eng.
        assert mixed_segment("我 like 猫") == ["我", "like", "猫"]

    def test_mandarin_without_spaces_still_splits(self):
        assert mixed_segment("报道称") == ["报", "道", "称"]

    def test_japanese_kana_split(self):
        # Naive CJK-only regexes miss kana; these must each be their own token.
        assert mixed_segment("サイエンス") == ["サ", "イ", "エ", "ン", "ス"]
        assert mixed_segment("もの") == ["も", "の"]

    def test_korean_hangul_split(self):
        assert mixed_segment("알려진") == ["알", "려", "진"]

    def test_thai_codepoint_mode_splits_every_codepoint(self):
        # Thai is not CJK and has no spaces; codepoint mode yields one token per
        # codepoint (base + each combining mark), consistent with jiwer CER.
        assert mixed_segment("ที่") == ["ท", "ี", "่"]

    def test_myanmar_codepoint_mode_splits_every_codepoint(self):
        assert mixed_segment("ကျော်") == ["က", "ျ", "ေ", "ာ", "်"]

    def test_mixed_scripts_in_one_utterance(self):
        assert mixed_segment("AI は computer の科目") == ["AI", "は", "computer", "の", "科", "目"]


class TestMixedSegmentGrapheme:
    """Grapheme mode keeps combining marks attached to their base; it matters
    only for the combining-mark scripts (Thai/Myanmar)."""

    def test_thai_grapheme_keeps_marks_with_base(self):
        # base ท + vowel ี + tone ่ -> a single perceived character.
        assert mixed_segment("ที่", grapheme=True) == ["ที่"]

    def test_myanmar_grapheme_keeps_stack_together(self):
        assert mixed_segment("ကျော်", grapheme=True) == ["ကျော်"]

    def test_grapheme_is_noop_for_han_kana_hangul(self):
        # No combining marks -> grapheme and codepoint modes agree.
        for text in ("报道称", "サイエンス", "알려진"):
            assert mixed_segment(text, grapheme=True) == mixed_segment(text, grapheme=False)

    def test_grapheme_keeps_latin_accents_on_their_word(self):
        # Decomposed e + U+0301 (combining acute) must stay attached to the
        # Latin word, not split it. Escapes pin the normalization form so the
        # test does not depend on the source file's encoding of accented chars.
        text = "cafe\u0301 \u6211"  # 'cafe' + combining acute, then Han '\u6211'
        assert mixed_segment(text, grapheme=True) == ["cafe\u0301", "\u6211"]


class TestEvaluateMer:
    """End-to-end MER scoring. Needs jiwer + the normalization stack."""

    def _evaluate_mer(self):
        import pytest

        pytest.importorskip("jiwer")
        pytest.importorskip("num2words")
        from nemo_skills.evaluation.evaluator.audio import evaluate_mer

        return evaluate_mer

    def test_perfect_match_is_zero(self):
        evaluate_mer = self._evaluate_mer()
        r = evaluate_mer("我 like 猫", "我 like 猫", normalization_mode="none")
        assert r["wer"] == 0.0
        assert r["wer_ref_words"] == 3  # 我 + like + 猫

    def test_one_english_word_substitution(self):
        evaluate_mer = self._evaluate_mer()
        # 3 mixed tokens, one English word wrong -> 1/3.
        r = evaluate_mer("我 like 猫", "我 hate 猫", normalization_mode="none")
        assert r["wer_ref_words"] == 3
        assert r["wer_substitutions"] == 1
        assert abs(r["wer"] - 1 / 3) < 1e-9

    def test_empty_reference_is_dropped(self):
        evaluate_mer = self._evaluate_mer()
        r = evaluate_mer("", "anything", normalization_mode="none")
        assert r["wer"] is None


class TestLowerNopunctNormalization:
    """Mark-preserving normalization (lowercase + unpunctuated) for paper-comparable
    CER. Dependency-free: this mode returns before any optional imports."""

    def _norm(self):
        from nemo_skills.evaluation.evaluator.audio import preprocess_asr_text

        return preprocess_asr_text

    def test_lowercases_and_strips_punctuation(self):
        assert self._norm()("Hello, World!", mode="lower_nopunct") == "hello world"

    def test_preserves_thai_combining_marks(self):
        # base ท + vowel ี + tone ่ all kept; trailing punctuation dropped.
        assert self._norm()("ที่!", mode="lower_nopunct") == "ที่"

    def test_preserves_myanmar_marks(self):
        assert self._norm()("ကျော่.", mode="lower_nopunct") == "ကျော่"

    def test_differs_from_no_tn_itn_which_strips_marks(self):
        # no_tn_itn drops the Thai marks (\w doesn't match them); lower_nopunct keeps them.
        norm = self._norm()
        assert norm("ที่", mode="lower_nopunct") != norm("ที่", mode="no_tn_itn")


class TestMultilingualPreserveMarks:
    """`preserve_marks` opt-in on multilingual normalization keeps abugida vowel/tone
    marks (so MER counts them) while retaining the rest of the multilingual pipeline.
    Dependency-free here: non-English lang + no digits avoids whisper/num2words imports."""

    def _norm(self):
        import pytest

        # multilingual mode imports whisper_normalizer unconditionally (for the
        # English-normalizer branch), so skip if it's unavailable.
        pytest.importorskip("whisper_normalizer")
        from nemo_skills.evaluation.evaluator.audio import preprocess_asr_text

        return preprocess_asr_text

    def test_preserve_marks_keeps_thai_vowel_and_tone(self):
        out = self._norm()("ที่!", mode="multilingual", lang="th", preserve_marks=True)
        assert out == "ที่"  # base + vowel + tone kept, punctuation dropped

    def test_default_multilingual_strips_marks(self):
        out = self._norm()("ที่", mode="multilingual", lang="th", preserve_marks=False)
        assert "ี" not in out and "่" not in out  # vowel ี and tone ่ removed

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

import math
import os
import re
import urllib.request
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from iso639.exceptions import InvalidLanguageValue
from iso639.iso639 import Lang

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float, default_formatting

# MediaPipe language detector. The ~315KB ``language_detector.tflite`` model is
# fetched on first use to ``~/.cache/nemo_skills/`` (override via NEMO_SKILLS_CACHE).
_LANG_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/language_detector/"
    "language_detector/float32/latest/language_detector.tflite"
)


def _lang_model_path() -> Path:
    cache_root = os.environ.get("NEMO_SKILLS_CACHE")
    base = Path(cache_root).resolve() if cache_root else Path.home() / ".cache" / "nemo_skills"
    return base / "language_detector.tflite"


@lru_cache(maxsize=1)
def _lang_detector():
    from mediapipe.tasks import python
    from mediapipe.tasks.python import text

    model_path = _lang_model_path()
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # Download to a temp path then atomic-rename so parallel processes
        # don't read a half-written file.
        tmp_path = model_path.with_suffix(model_path.suffix + ".tmp")
        urllib.request.urlretrieve(_LANG_MODEL_URL, tmp_path)
        os.replace(tmp_path, model_path)
    return text.LanguageDetector.create_from_options(
        text.LanguageDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            max_results=1,
        )
    )


def _detect_top_language(answer_text: str) -> str:
    """Return the top-1 ISO 639-1 code for ``answer_text``; ``"unknown"`` if undetectable."""
    if not answer_text or not answer_text.strip():
        return "unknown"
    result = _lang_detector().detect(answer_text)
    if not result.detections:
        return "unknown"
    return result.detections[0].language_code


class MTBenchMetrics(BaseMetrics):
    """MT-Bench scoring with optional per-entry target-language penalty.

    Multilingual variants (e.g. KoMT-Bench) set ``target_language`` (an ISO 639-1
    2-letter code) on each dataset entry. When set and the response is not in
    that language, raw judge scores are sqrt-penalized to discourage
    code-switching. Plain MT-Bench leaves ``target_language`` unset and reports
    raw scores only.
    """

    def __init__(self, compute_no_answer: bool = True):
        super().__init__(compute_no_answer=compute_no_answer)

    def _parse_rating(self, judgement):
        """Extract [[rating]] from judge output. Returns float 1-10 or None."""
        if not judgement:
            return None
        pattern = re.compile(r"\[\[(\d+\.?\d*)\]\]")
        matches = pattern.findall(judgement)
        if not matches:
            return None
        try:
            score = float(matches[-1])
            if 1.0 <= score <= 10.0:
                return score
        except ValueError:
            pass
        return None

    @staticmethod
    def _validate_target_language(lang):
        try:
            Lang(pt1=lang)
        except InvalidLanguageValue as exc:
            raise ValueError(f"target_language must be a valid ISO 639-1 2-letter code, got: {lang!r}") from exc

    def _score_one(self, pred):
        """Parse turn1/turn2 ratings for a single prediction (multi-judge aware),
        then apply target-language penalty if requested."""
        # all_judgements_turn* is only emitted by mt_bench_judge.py when num_judges>1.
        # Absence is the standard single-judge path, so .get() here is intentional.
        all_j1 = pred.get("all_judgements_turn1")
        all_j2 = pred.get("all_judgements_turn2")
        if all_j1 and all_j2:
            scores_t1 = [s for s in (self._parse_rating(j) for j in all_j1) if s is not None]
            scores_t2 = [s for s in (self._parse_rating(j) for j in all_j2) if s is not None]
            score_t1 = sum(scores_t1) / len(scores_t1) if scores_t1 else None
            score_t2 = sum(scores_t2) / len(scores_t2) if scores_t2 else None
        else:
            score_t1 = self._parse_rating(pred["judgement_turn1"])
            score_t2 = self._parse_rating(pred["judgement_turn2"])

        # Downstream pipeline fills missing string fields with "", so a truthy
        # check covers both unset and empty-string as "no language penalty".
        target_language = pred.get("target_language")
        pen_t1, pen_t2 = score_t1, score_t2
        if target_language:
            self._validate_target_language(target_language)
            answer_1 = pred["answer_1"]
            answer_2 = pred["answer_2"]
            if score_t1 is not None and _detect_top_language(answer_1) != target_language:
                pen_t1 = math.sqrt(score_t1)
            if score_t2 is not None and _detect_top_language(answer_2) != target_language:
                pen_t2 = math.sqrt(score_t2)
        return {
            "category": pred["category"],
            "score_turn1": score_t1,
            "score_turn2": score_t2,
            "penalized_turn1": pen_t1,
            "penalized_turn2": pen_t2,
            "has_target_language": bool(target_language),
        }

    def update(self, predictions):
        """Each (row, sample) contributes its own scored entry, so the aggregated
        average matches pass@1[avg-of-N] semantics for task:N runs."""
        super().update(predictions)
        for pred in predictions:
            self.entries.append(self._score_one(pred))

    def _build_overall(self, entries):
        overall = {"num_entries": self.total}
        self.update_common_metrics(overall)

        pen_t1 = [e["penalized_turn1"] for e in entries if e["penalized_turn1"] is not None]
        pen_t2 = [e["penalized_turn2"] for e in entries if e["penalized_turn2"] is not None]
        all_pen = pen_t1 + pen_t2
        if all_pen:
            overall["average_score"] = sum(all_pen) / len(all_pen)
        if pen_t1:
            overall["turn1_average"] = sum(pen_t1) / len(pen_t1)
        if pen_t2:
            overall["turn2_average"] = sum(pen_t2) / len(pen_t2)

        # Only emit raw_* when target_language was set on at least one entry — for
        # plain MT-Bench raw_* would just duplicate the primary averages.
        if any(e["has_target_language"] for e in entries):
            raw_t1 = [e["score_turn1"] for e in entries if e["score_turn1"] is not None]
            raw_t2 = [e["score_turn2"] for e in entries if e["score_turn2"] is not None]
            all_raw = raw_t1 + raw_t2
            if all_raw:
                overall["raw_average_score"] = sum(all_raw) / len(all_raw)
            if raw_t1:
                overall["raw_turn1_average"] = sum(raw_t1) / len(raw_t1)
            if raw_t2:
                overall["raw_turn2_average"] = sum(raw_t2) / len(raw_t2)

        cat_scores = defaultdict(lambda: {"turn1": [], "turn2": []})
        for entry in entries:
            cat = entry["category"]
            if entry["penalized_turn1"] is not None:
                cat_scores[cat]["turn1"].append(entry["penalized_turn1"])
            if entry["penalized_turn2"] is not None:
                cat_scores[cat]["turn2"].append(entry["penalized_turn2"])
        for cat, scores in sorted(cat_scores.items()):
            all_cat = scores["turn1"] + scores["turn2"]
            if all_cat:
                overall[f"category_{cat}"] = sum(all_cat) / len(all_cat)
        return overall

    def get_metrics(self):
        # pass@1 already aggregates every (row, sample) entry under task:N>1,
        # so it doubles as pass@1[avg-of-N]; we don't emit a redundant alias.
        return {"pass@1": self._build_overall(self.entries)}

    def evaluations_to_print(self):
        return ["pass@1"]

    def metrics_to_print(self):
        # MT-Bench scores are 1-10, not percentages
        metrics = self.get_metrics()
        all_keys = set()
        for eval_metrics in metrics.values():
            all_keys.update(eval_metrics.keys())
        return {
            key: as_float
            if isinstance(next((m[key] for m in metrics.values() if key in m), None), float)
            else default_formatting
            for key in all_keys
        }

    def reset(self):
        super().reset()
        self.entries = []

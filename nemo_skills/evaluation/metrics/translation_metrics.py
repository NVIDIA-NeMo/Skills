# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
from collections import defaultdict

import numpy as np
from sacrebleu import corpus_bleu

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float


def install_packages(lang):
    """Korean and Japanese tokenizations require extra dependencies."""
    subprocess.run(
        ["pip", "install", "-q", f"sacrebleu[{lang}]"],
        check=True,
        capture_output=True,
        text=True,
    )


SEGALE_FIELDS = [
    "segale_comet",
    "segale_comet_qe",
    "segale_metricx",
    "segale_metricx_qe",
    "segale_bleu",
    "segale_lang_fidelity",
]
# Count fields are summed across documents (not averaged) so that
# misalignment rate = segale_misaligned_seg / segale_total_seg is meaningful.
SEGALE_COUNT_FIELDS = ["segale_total_seg", "segale_misaligned_seg"]


class TranslationMetrics(BaseMetrics):
    def get_metrics(self):
        metrics_dict = {}
        num_seeds = 0
        for lang_pair in self.translation_dict:
            src_lang, tgt_lang = lang_pair.split("->")
            preds = self.translation_dict[lang_pair]["preds"]
            gts = self.translation_dict[lang_pair]["gts"]

            num_seeds = len(preds[0]) if preds else 0

            tokenize = "13a"
            if tgt_lang[:2] == "ja":
                install_packages(tgt_lang[:2])
                tokenize = "ja-mecab"
            if tgt_lang[:2] == "zh":
                tokenize = "zh"
            if tgt_lang[:2] == "ko":
                install_packages(tgt_lang[:2])
                tokenize = "ko-mecab"

            bleu_scores = []
            for i in range(num_seeds):
                predictions = [pred[i] for pred in preds]
                ground_truths = [gt[i] for gt in gts]
                bleu_scores.append(corpus_bleu(predictions, [ground_truths], tokenize=tokenize).score)

            metrics_dict[lang_pair] = {"bleu": bleu_scores}
            self.bleu_aggregation_dict["xx->xx"].append(bleu_scores)
            self.bleu_aggregation_dict[f"{src_lang}->xx"].append(bleu_scores)
            self.bleu_aggregation_dict[f"xx->{tgt_lang}"].append(bleu_scores)

            if "comets" in self.translation_dict[lang_pair]:
                comets = list(zip(*self.translation_dict[lang_pair]["comets"]))
                comet_scores = [np.mean(comets[i]) for i in range(num_seeds)]
                metrics_dict[lang_pair]["comet"] = comet_scores
                self.comet_aggregation_dict["xx->xx"].append(comet_scores)
                self.comet_aggregation_dict[f"{src_lang}->xx"].append(comet_scores)
                self.comet_aggregation_dict[f"xx->{tgt_lang}"].append(comet_scores)

            for field in SEGALE_FIELDS:
                storage_key = field + "s"
                if storage_key in self.translation_dict[lang_pair]:
                    scores = list(zip(*self.translation_dict[lang_pair][storage_key]))
                    field_scores = [np.mean(scores[i]) for i in range(num_seeds)]
                    metrics_dict[lang_pair][field] = field_scores
                    self.segale_aggregation_dicts[field]["xx->xx"].append(field_scores)
                    self.segale_aggregation_dicts[field][f"{src_lang}->xx"].append(field_scores)
                    self.segale_aggregation_dicts[field][f"xx->{tgt_lang}"].append(field_scores)

            for field in SEGALE_COUNT_FIELDS:
                storage_key = field + "s"
                if storage_key in self.translation_dict[lang_pair]:
                    counts = list(zip(*self.translation_dict[lang_pair][storage_key]))
                    field_counts = [np.sum(counts[i]) for i in range(num_seeds)]
                    metrics_dict[lang_pair][field] = field_counts
                    self.segale_count_aggregation_dicts[field]["xx->xx"].append(field_counts)
                    self.segale_count_aggregation_dicts[field][f"{src_lang}->xx"].append(field_counts)
                    self.segale_count_aggregation_dicts[field][f"xx->{tgt_lang}"].append(field_counts)

        for lang_pair in self.bleu_aggregation_dict:
            bleus = list(zip(*self.bleu_aggregation_dict[lang_pair]))
            bleu_scores = [np.mean(bleus[i]) for i in range(num_seeds)]
            metrics_dict[lang_pair] = {"bleu": bleu_scores}

            if self.comet_aggregation_dict.get(lang_pair):
                comets = list(zip(*self.comet_aggregation_dict[lang_pair]))
                comet_scores = [np.mean(comets[i]) for i in range(num_seeds)]
                metrics_dict[lang_pair]["comet"] = comet_scores

            for field in SEGALE_FIELDS:
                if self.segale_aggregation_dicts[field].get(lang_pair):
                    scores = list(zip(*self.segale_aggregation_dicts[field][lang_pair]))
                    metrics_dict[lang_pair][field] = [np.mean(scores[i]) for i in range(num_seeds)]

            for field in SEGALE_COUNT_FIELDS:
                if self.segale_count_aggregation_dicts[field].get(lang_pair):
                    counts = list(zip(*self.segale_count_aggregation_dicts[field][lang_pair]))
                    metrics_dict[lang_pair][field] = [np.sum(counts[i]) for i in range(num_seeds)]

        self._add_std_metrics(metrics_dict)

        return metrics_dict

    def _add_std_metrics(self, metrics_dict):
        for key in metrics_dict:
            metrics_list = ["bleu"]
            if "comet" in metrics_dict[key]:
                metrics_list.append("comet")
            for field in SEGALE_FIELDS:
                if field in metrics_dict[key]:
                    metrics_list.append(field)

            for metric in metrics_list:
                avg = np.mean(metrics_dict[key][metric])
                std = np.std(metrics_dict[key][metric])
                metrics_dict[key].update({metric: avg, f"{metric}_statistics": {"std_dev_across_runs": std}})

            # Count fields: collapse multi-seed list to a single value (sum).
            # std_dev is not meaningful for segment counts.
            for field in SEGALE_COUNT_FIELDS:
                if field in metrics_dict[key]:
                    metrics_dict[key][field] = int(np.sum(metrics_dict[key][field]))

            # Misalignment rate: fraction of spans flagged as misaligned.
            total = metrics_dict[key].get("segale_total_seg", 0)
            misaligned = metrics_dict[key].get("segale_misaligned_seg", 0)
            if total > 0:
                metrics_dict[key]["segale_misalignment_rate"] = misaligned / total
            else:
                metrics_dict[key]["segale_misalignment_rate"] = None

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        super().update(predictions)

        generations, ground_truths, comets = [], [], []
        segale_scores = {field: [] for field in SEGALE_FIELDS}
        segale_counts = {field: [] for field in SEGALE_COUNT_FIELDS}
        for pred in predictions:
            src_lang = pred["source_language"]
            tgt_lang = pred["target_language"]

            generation = pred["generation"]
            if generation is None:
                generation = ""

            generations.append(generation)
            if "translation" in pred:
                ground_truth = pred["translation"]
            elif "reference_sentences" in pred:
                ground_truth = " ".join(pred["reference_sentences"])
            else:
                raise KeyError("Expected 'translation' or 'reference_sentences' in prediction record")
            ground_truths.append(ground_truth)
            if "comet" in pred:
                comets.append(pred["comet"] * 100)
            # SEGALE writes the same doc-level score to every sentence record in
            # a document. Collect once per doc_id to avoid inflating averages.
            # Records without doc_id are sentence-level data — SEGALE does not
            # apply to them.
            doc_id = pred.get("doc_id")
            seen_key = (src_lang, tgt_lang, doc_id)
            if doc_id and seen_key not in self.seen_doc_ids:
                self.seen_doc_ids.add(seen_key)
                for field in SEGALE_FIELDS:
                    if field in pred:
                        segale_scores[field].append(pred[field])
                for field in SEGALE_COUNT_FIELDS:
                    if field in pred:
                        segale_counts[field].append(pred[field])

        lang_pair = f"{src_lang}->{tgt_lang}"
        self.translation_dict[lang_pair]["preds"].append(generations)
        self.translation_dict[lang_pair]["gts"].append(ground_truths)

        if len(comets) > 0:
            self.translation_dict[lang_pair]["comets"].append(comets)

        for field, values in segale_scores.items():
            if values:
                self.translation_dict[lang_pair][field + "s"].append(values)

        for field, values in segale_counts.items():
            if values:
                self.translation_dict[lang_pair][field + "s"].append(values)

    def reset(self):
        super().reset()
        self.translation_dict = defaultdict(lambda: defaultdict(list))
        self.bleu_aggregation_dict = defaultdict(list)
        self.comet_aggregation_dict = defaultdict(list)
        self.seen_doc_ids = set()
        self.segale_aggregation_dicts = {field: defaultdict(list) for field in SEGALE_FIELDS}
        self.segale_count_aggregation_dicts = {field: defaultdict(list) for field in SEGALE_COUNT_FIELDS}

    def evaluations_to_print(self):
        """Returns all translation pairs and aggregated multilingual dictionaries."""
        return list(self.translation_dict.keys()) + list(self.bleu_aggregation_dict.keys())

    def metrics_to_print(self):
        metrics_to_print = {"bleu": as_float, "comet": as_float}
        metrics_to_print.update({field: as_float for field in SEGALE_FIELDS})
        metrics_to_print.update({field: as_float for field in SEGALE_COUNT_FIELDS})

        def as_float_or_none(key, value, all_metrics):
            return None if value is None else as_float(key, value, all_metrics)

        metrics_to_print["segale_misalignment_rate"] = as_float_or_none
        return metrics_to_print

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

from collections import defaultdict

from sacrebleu import corpus_bleu

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float


class TranslationMetrics(BaseMetrics):
    # TODO: refactor BLEU computation so it reuses parent method functions from pass@k
    # TODO: add support for other translation metrics, such as COMET and MetricX

    def get_metrics(self):
        metrics_dict = {}
        for key in self.translation_dict:
            src_lang, tgt_lang = key.split("->")
            preds = self.translation_dict[key]["preds"]
            gts = self.translation_dict[key]["gts"]

            tokenize = "13a"
            if tgt_lang[:2] == "ja":
                tokenize = "ja-mecab"
            if tgt_lang[:2] == "zh":
                tokenize = "zh"
            if tgt_lang[:2] == "ko":
                tokenize = "ko-mecab"

            bleu_score = corpus_bleu(preds, [gts], tokenize=tokenize).score
            metrics_dict[key] = {"bleu": bleu_score}
            self.bleu_aggregation_dict["xx->xx"].append(bleu_score)
            self.bleu_aggregation_dict[f"{src_lang}->xx"].append(bleu_score)
            self.bleu_aggregation_dict[f"xx->{tgt_lang}"].append(bleu_score)

            if "comets" in self.translation_dict[key]:
                comet_score = sum(self.translation_dict[key]["comets"]) / len(self.translation_dict[key]["comets"])
                metrics_dict[key]["comet"] = comet_score
                self.comet_aggregation_dict["xx->xx"].append(comet_score)
                self.comet_aggregation_dict[f"{src_lang}->xx"].append(comet_score)
                self.comet_aggregation_dict[f"xx->{tgt_lang}"].append(comet_score)

        for key in self.bleu_aggregation_dict:
            metrics_dict[key] = {"bleu": sum(self.bleu_aggregation_dict[key]) / len(self.bleu_aggregation_dict[key])}
            if self.comet_aggregation_dict:
                metrics_dict[key]["comet"] = sum(self.comet_aggregation_dict[key]) / len(
                    self.comet_aggregation_dict[key]
                )

        return metrics_dict

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        super().update(predictions)

        for pred in predictions:
            src_lang = pred["source_language"]
            tgt_lang = pred["target_language"]
            generation = pred["generation"]
            ground_truth = pred["translation"]

            if generation is None:
                generation = ""

            self.translation_dict[f"{src_lang}->{tgt_lang}"]["preds"].append(generation)
            self.translation_dict[f"{src_lang}->{tgt_lang}"]["gts"].append(ground_truth)

            if "comet" in pred:
                self.translation_dict[f"{src_lang}->{tgt_lang}"]["comets"].append(pred["comet"] * 100)

    def reset(self):
        super().reset()
        self.translation_dict = defaultdict(lambda: defaultdict(list))
        self.bleu_aggregation_dict = defaultdict(list)
        self.comet_aggregation_dict = defaultdict(list)

    def evaluations_to_print(self):
        """Returns all translation pairs and aggregated multilingual dictionaries."""
        return list(self.translation_dict.keys()) + list(self.bleu_aggregation_dict.keys())

    def metrics_to_print(self):
        metrics_to_print = {"bleu": as_float, "comet": as_float}
        return metrics_to_print

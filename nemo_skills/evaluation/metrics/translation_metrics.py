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


class TranslationMetrics(BaseMetrics):
    # TODO: add support for other translation metrics, such as MetricX

    def get_metrics(self):
        metrics_dict = {}
        for key in self.translation_dict:
            src_lang, tgt_lang = key.split("->")
            preds = self.translation_dict[key]["preds"]
            gts = self.translation_dict[key]["gts"]

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

            metrics_dict[key] = {"bleu": bleu_scores}
            self.bleu_aggregation_dict["xx->xx"].append(bleu_scores)
            self.bleu_aggregation_dict[f"{src_lang}->xx"].append(bleu_scores)
            self.bleu_aggregation_dict[f"xx->{tgt_lang}"].append(bleu_scores)

            if "comets" in self.translation_dict[key]:
                comets = list(zip(*self.translation_dict[key]["comets"]))
                comet_scores = [np.mean(comets[i]) for i in range(num_seeds)]
                metrics_dict[key]["comet"] = comet_scores
                self.comet_aggregation_dict["xx->xx"].append(comet_scores)
                self.comet_aggregation_dict[f"{src_lang}->xx"].append(comet_scores)
                self.comet_aggregation_dict[f"xx->{tgt_lang}"].append(comet_scores)

        for key in self.bleu_aggregation_dict:
            bleus = list(zip(*self.bleu_aggregation_dict[key]))
            bleu_scores = [np.mean(bleus[i]) for i in range(num_seeds)]
            metrics_dict[key] = {"bleu": bleu_scores}

            if self.comet_aggregation_dict:
                comets = list(zip(*self.comet_aggregation_dict[key]))
                comet_scores = [np.mean(comets[i]) for i in range(num_seeds)]
                metrics_dict[key]["comet"] = comet_scores

        self._add_std_metrics(metrics_dict)

        return metrics_dict

    def _add_std_metrics(self, metrics_dict):
        for key in metrics_dict:
            metrics_list = ["bleu"]
            if "comet" in metrics_dict[key]:
                metrics_list.append("comet")

            for metric in metrics_list:
                avg = np.mean(metrics_dict[key][metric])
                std = np.std(metrics_dict[key][metric])

                metrics_dict[key].update({metric: avg, f"{metric}_statistics": {"std_dev_across_runs": std}})

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        super().update(predictions)

        generations, ground_truths, comets = [], [], []
        for pred in predictions:
            src_lang = pred["source_language"]
            tgt_lang = pred["target_language"]

            generation = pred["generation"]
            if generation is None:
                generation = ""

            generations.append(generation)
            ground_truths.append(pred["translation"])
            if "comet" in pred:
                comets.append(pred["comet"] * 100)

        self.translation_dict[f"{src_lang}->{tgt_lang}"]["preds"].append(generations)
        self.translation_dict[f"{src_lang}->{tgt_lang}"]["gts"].append(ground_truths)

        if "comet" in pred:
            self.translation_dict[f"{src_lang}->{tgt_lang}"]["comets"].append(comets)

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

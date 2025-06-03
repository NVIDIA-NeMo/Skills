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

import json
import logging
from pathlib import Path

from nemo_skills.evaluation.constants import JUDGE_MODEL, JUDGE_SERVER
from nemo_skills.evaluation.metrics.base import BaseMetrics
from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from nemo_skills.inference.server.model import get_model
from nemo_skills.utils import get_logger_name, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))


class MathMetrics(BaseMetrics):

    def setup(self, input_files):
        # checking if judgements are ready and fusing them with predictions
        # might get permission errors when running locally, since original file
        # is generated inside docker. Is there any way around that?
        for jsonl_file in unroll_files(input_files):
            if Path(jsonl_file + '-batch-request-id').exists():
                with open(jsonl_file + '-batch-request-id', 'rt', encoding='utf-8') as fin:
                    request_id = json.load(fin)['request_id']

                llm = get_model(server_type=JUDGE_SERVER, model=JUDGE_MODEL)
                metadata, outputs = llm.get_batch_results(request_id)

                if outputs is None:
                    raise RuntimeError(f"Judgements are not ready yet! Current status: {metadata}")

                with open(jsonl_file, 'rt', encoding='utf-8') as fin:
                    predictions = [json.loads(line) for line in fin]

                with open(jsonl_file, 'wt', encoding='utf-8') as fout:
                    for prediction, output in zip(predictions, outputs):
                        prediction['judgement'] = output['generation']
                        fout.write(json.dumps(prediction) + '\n')

                Path(jsonl_file + '-batch-request-id').unlink()

    def get_prediction_results(self, prediction):
        result = {}
        # TODO: rename is_correct since it's only for sympy now?
        if 'is_correct' in prediction:
            self.has_sympy = True
            result["correct_sympy"] = prediction['is_correct']
        if 'judgement' in prediction:
            self.has_judge = True
            result["correct_judge"] = is_correct_judgement(prediction['judgement'])
        if self.has_sympy and self.has_judge:
            result["both_correct"] = result["correct_sympy"] and result["correct_judge"]
            result["any_correct"] = result["correct_sympy"] or result["correct_judge"]

        return result

    def get_prediction_statistics(self, prediction):
        stats = super().get_prediction_statistics(prediction)
        stats["no_answer"] = prediction['predicted_answer'] is None
        return stats

    def get_reward_at_k(self, agg_mode_dict, pred_keys, predicted_answers, prediction_results):
        for k in range(len(prediction_results), 1, -1):
            for pred_field in pred_keys:
                valid_answers_and_results = [
                    (pred, result[pred_field])
                    for pred, result in zip(predicted_answers[:k], prediction_results[:k])
                    if pred is not None
                ]

                # If no valid answers, mark as incorrect
                if not valid_answers_and_results:
                    agg_mode_dict[f"rm_best@{k}"][pred_field] += False
                    continue

                # Answer is the top-scoring reward
                rm_result = sorted(valid_answers_and_results, key=lambda x: x[2], reverse=True)[0][1]

                # Update the metric
                agg_mode_dict[f"rm_best@{k}"][pred_field] += rm_result

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        super().update(predictions)

        self.get_pass_at_k(
            self.agg_mode_dict,
            predictions=predictions,
        )
        self.get_majority_at_k(
            predictions=predictions,
            predicted_answers=[pred['predicted_answer'] for pred in predictions],
        )

        if 'reward_model_score' in predictions[0]:
            self.get_reward_at_k(
                self.agg_mode_dict,
                predicted_answers=[pred['predicted_answer'] for pred in predictions],
                prediction_results=prediction_results,
            )

        # Log discrepancies between the two judgements
        if self.has_sympy and self.has_judge:
            prediction_results = [self.get_prediction_results(pred) for pred in predictions]
            if prediction_results[0]["correct_sympy"] != prediction_results[0]["correct_judge"]:
                LOG.debug(
                    "Discrepancy between symbolic (%s) and LLM checkers (%s).\n"
                    "Question: %s\nPredicted answer: %s\nExpected answer: %s\nLLM reasoning: %s\n",
                    bool(prediction_results[0]["correct_sympy"]),
                    bool(prediction_results[0]["correct_judge"]),
                    predictions[0]['problem'],
                    predictions[0]['predicted_answer'],
                    predictions[0]['expected_answer'],
                    predictions[0]['judgement'],
                )

    def get_metrics(self):
        metrics_dict = {}
        for agg_mode, agg_metric_dict in self.agg_mode_dict.items():
            metrics_dict[agg_mode] = {"num_entries": self.total}
            if self.has_sympy:
                metrics_dict[agg_mode]["symbolic_correct"] = (
                    agg_metric_dict["correct_sympy"]['correct'] / self.total
                ) * 100.0
            if self.has_judge:
                metrics_dict[agg_mode]["judge_correct"] = (
                    agg_metric_dict["correct_judge"]['correct'] / self.total
                ) * 100.0
            if self.has_sympy and self.has_judge:
                metrics_dict[agg_mode]["both_correct"] = (
                    agg_metric_dict["both_correct"]['correct'] / self.total
                ) * 100.0
                metrics_dict[agg_mode]["any_correct"] = (
                    agg_metric_dict["any_correct"]['correct'] / self.total
                ) * 100.0

            # metrics_dict[agg_mode]["no_answer"] = (agg_metric_dict["no_answer"] / self.total) * 100.0

        return metrics_dict

    def reset(self):
        super().reset()
        self.has_sympy = False
        self.has_judge = False
        self.has_reward = False

    def aggregations_to_print(self):
        """We will log all majority/rm/pass/pass@1[k] up to k, but only report the kth one."""
        return [f'pass@1[{self.max_k}]', f'majority@{self.max_k}', f'pass@{self.max_k}', f'rm_best@{self.max_k}']

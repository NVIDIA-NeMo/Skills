# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import re
import editdistance

from tqdm import tqdm

from nemo_skills.utils import get_logger_name, nested_dataclass, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class RulerEvaluatorConfig:
    parse_func: str = "default"
    match_type: str


def eval_ruler(cfg):
    def default_parse(prediction):
        prediction = prediction.strip()
        # Remove all non-printable characters
        np_pattern = re.compile(r"[\x00-\x1f]")
        pp_predict = np_pattern.sub("\n", prediction).strip()
        return pp_predict

    def string_match_all_single(preds, refs):
        """the metric function with input (predictions: [str], references: [[str]]) to compute score."""
        preds = [preds]
        refs = [refs]
        score = [
            sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)
        ][0]
        return score

    def string_match_part_single(preds, refs):
        preds = [preds]
        refs = [refs]
        score = [
            sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)])
        ][0]
        return score

    eval_config = RulerEvaluatorConfig(**cfg.eval_config)

    parse_funcs = {
        "default": default_parse,
    }
    match_type_funcs = {
        "all": string_match_all_single,
        "part": string_match_part_single,
    }

    for file in unroll_files(cfg.input_files):
        with open(file, "rt", encoding="utf-8") as fin:
            data = [json.loads(line) for line in fin]
        with open(file, "wt", encoding="utf-8") as fout:
            for sample in tqdm(data):
                parse_result = parse_funcs[eval_config.parse_func](sample["generation"])
                sample["is_correct"] = match_type_funcs[eval_config.match_type](
                    sample["generation"], sample["expected_answer"]
                )
                sample["predicted_answer"] = parse_result
                fout.write(json.dumps(sample) + "\n")


def eval_ruler2(cfg):
    def default_parse(prediction):
        prediction = prediction.strip()
        # Remove all non-printable characters
        np_pattern = re.compile(r"[\x00-\x1f]")
        pp_predict = np_pattern.sub("\n", prediction).strip()
        return pp_predict

    def post_process_preds(preds):
        if "</think>" in preds:
            preds = preds.split("</think>")[-1]

        if "Answer:" in preds:
            preds = preds.split("Answer:")[-1]
        return preds


    def wer(hypotheses: list[str], references: list[str]) -> float:
        scores = 0
        words = 0
        if len(hypotheses) != len(references):
            raise ValueError(
                "In word error rate calculation, hypotheses and reference"
                " lists must have the same number of elements. But I got:"
                "{0} and {1} correspondingly".format(len(hypotheses), len(references))
            )
        for h, r in zip(hypotheses, references):
            h_list = h.split()
            r_list = r.split()
            words += len(r_list)
            scores += editdistance.eval(h_list, r_list)
        if words != 0:
            wer = 1.0 * scores / words
        else:
            wer = float('inf')
        return wer

    def string_match_all_single(preds, refs):
        """the metric function with input (predictions: [str], references: [[str]]) to compute score."""
        preds = post_process_preds(preds)
        preds = [preds]
        refs = [refs]
        score = [
            sum([max(1.0 if r.lower() in pred.lower() else 0.0, 1 - wer([pred.lower()], [r.lower()])) for r in ref]) / len(ref) for pred, ref in zip(preds, refs)
        ][0]
        return score

    def string_match_2steps_single(preds, refs):
        preds = post_process_preds(preds)
        preds = preds.split("\n\n")[-1]
        preds = [preds]
        refs = [refs]
        score = [
            sum([max(1.0 if r.lower() in pred.lower() else 0.0, 1 - wer([pred.lower()], [r.lower()])) for r in ref]) / len(ref) for pred, ref in zip(preds, refs)
        ][0]
        return score

    def string_match_part_single(preds, refs):
        preds = post_process_preds(preds)
        preds = re.sub(r'Document \d+:(?:.*\n)+?\n', '', preds)

        preds = [preds]
        refs = [refs]
        score = [
            sum([max([max(1.0 if r.lower() in pred.lower() else 0.0, 1 - wer([pred.lower()], [r.lower()])) for r in ref]) for pred, ref in zip(preds, refs)])
        ][0]
        return score

    eval_config = RulerEvaluatorConfig(**cfg.eval_config)

    parse_funcs = {
        "default": default_parse,
    }
    match_type_funcs = {
        "all": string_match_all_single,
        "part": string_match_part_single,
        "2steps": string_match_2steps_single,
    }

    for file in unroll_files(cfg.input_files):
        with open(file, "rt", encoding="utf-8") as fin:
            data = [json.loads(line) for line in fin]
        with open(file, "wt", encoding="utf-8") as fout:
            for sample in tqdm(data):
                parse_result = parse_funcs[eval_config.parse_func](sample["generation"])
                sample["is_correct"] = match_type_funcs[eval_config.match_type](
                    sample["generation"], sample["expected_answer"]
                )
                sample["predicted_answer"] = parse_result
                fout.write(json.dumps(sample) + "\n")
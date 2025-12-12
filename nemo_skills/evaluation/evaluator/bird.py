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

import asyncio
import concurrent.futures
import json
import logging
from pathlib import Path
import os
import re
import sys
import time

from nemo_skills.dataset.birdbench.evaluation import execute_sql
from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.file_utils import jdump
from nemo_skills.utils import nested_dataclass


@nested_dataclass(kw_only=True)
class BirdEvaluatorConfig(BaseEvaluatorConfig):
    timeout = 30.0

    # Answer format can be "BOXED", "CODEBLOCK", or "USE_REGEX", the last of
    # which uses the given regex in the extraction_regex arg.
    answer_format = "CODEBLOCK"
    extraction_regex = None
    regex_dotall = False

    # Paths needed for ground truth SQL file and database directory
    gt_filepath = ""
    db_path = ""


class BirdEvaluator(BaseEvaluator):
    def __init__(self, config: dict, num_parallel_requests=10):
        super().__init__(config, num_parallel_requests)
        self.eval_config = BirdEvaluatorConfig(**self.config)

        self.db_ids = self._setup_db_ids()
        self.gts = self._get_ground_truths()


    """
    def _setup_db_ids(self):
        if not self.eval_config.gt_filepath or not os.path.exists(self.eval_config.gt_filepath):
            logging.error(f"BIRD eval ground truth file path not valid: {self.eval_config.gt_filepath}")

        db_ids = []
        with open(self.eval_config.gt_filepath, 'r') as f_in:
            for line in f_in:
                db_ids.append(line.split()[-1])

        return db_ids
    """


    def _get_ground_truths(self):
        """
        Retrieves ground truth entries in order, in the form (SQL query, db path).
        """
        gts = []

        with open(self.eval_config.gt_filepath, 'r') as f_in:
            for line in f_in:
                sql, db_name = line.strip().split('\t')
                db_loc = Path(self.eval_config.db_path, db_name, db_name + ".sqlite")

                gt_queries.append((sql, db_loc))

        return gts

        
    def _extract_answer(self, text):
        regex = ""
        dotall = False

        if answer_format == "CODEBLOCK":
            regex = r"(?:```sql)(.*?[a-zA-Z].*?)(?:```)"
            dotall = True
        elif answer_format == "BOXED":
            regex = r"(?:boxed\{\{)(.*?[a-zA-Z].*?)(?:\}\})"
            dotall = True
        elif answer_format == "USE_REGEX":
            regex = self.eval_config.extraction_regex
            regex_dotall = self.eval_config.regex_dotall

        if not regex:
            logging.error(
                "Answer format underspecified for BIRD evaluation; should be one of " +
                "{CODEBLOCK, BOXED, USE_REGEX (provide extraction_regex)}.\n" + 
                f"Got {answer_format} instead."
            )

        # Use regex to extract answer from text
        if dotall:
            code_matches = re.findall(regex, text, flags=re.DOTALL)
        else:
            code_matches = re.findall(regex, text)

        if not code_matches:
            return "SELECT 1"

        # Remove comments first
        ans = re.sub(r"--.*", "", code_matches[-1])  # Use last match
        # Collapse whitespace
        ans = re.sub(r"\s+", " ", ans)
        # Remove miscellaneous headers that snuck in
        ans = re.sub(r"^\*\*.*\*\*", "", ans).strip()

        return ans


    async def eval_full(self):  # type: ignore[override]
        infile = self.eval_cfg.input_file

        lines = []
        i = 0
        with open(infile, 'w') as f_out:
            for line in f_in:
                line = json.loads(line)

                # Attach ground truth and table data
                line["gt_sql"] = self.gts[i][0]
                line["db_path"] = self.gts[i][1]
                line["sql_index"] = i

                lines.append(line)
                i+= 1

        tasks = [self.eval_single(line) for line in lines]
        outputs = await asyncio.gather(*tasks)

        for line, output in zip(lines, outputs):
            line["res"] = output["res"]

        jdump(lines, infile, mode="wt")


    async def eval_single(self, data_point: dict):
        predicted_sql = extract_answer(data_point["generation"])
        ground_truth = data_point["gt_sql"]
        db_place = data_point["db_path"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(execute_sql, predicted_sql, ground_truth, db_place)
            try:
                # Wait for result with timeout as set
                res = future.result(timeout=self.eval_config.timeout)
            except concurrent.futures.TimeoutError:
                result = [(f'timeout',)]
                res = 0
            except Exception as e:
                result = [(f'error',)]  # possibly len(query) > 512 or not executable
                res = 0
        result = {"res": res}
        return result

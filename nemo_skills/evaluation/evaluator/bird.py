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

import logging
import re
import sqlite3
from pathlib import Path

from func_timeout import FunctionTimedOut, func_timeout

from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.utils import nested_dataclass

# The following code was modified from:
# https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/evaluation.py

# Original license as follows:

# MIT License
#
# Copyright (c) 2022 Alibaba Research
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def execute_sql(predicted_sql, ground_truth, db_path):
    # Connect to the database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
        res = 0
        if set(predicted_res) == set(ground_truth_res):
            res = 1
    return res


# ===== End of copied and modified code. =====


@nested_dataclass(kw_only=True)
class BirdEvaluatorConfig(BaseEvaluatorConfig):
    timeout: int = 30

    # Answer format can be "BOXED", "CODEBLOCK", or "USE_REGEX", the last of
    # which uses the given regex in the extraction_regex arg.
    answer_format: str = "CODEBLOCK"
    extraction_regex: str | None = None
    regex_dotall: bool = False


class BirdEvaluator(BaseEvaluator):
    def __init__(self, config: dict, num_parallel_requests=10):
        super().__init__(config, num_parallel_requests)
        self.eval_config = BirdEvaluatorConfig(**self.config)

        self.db_path = Path(self.eval_config.data_dir, "birdbench", "dev_20240627", "dev_databases")

    def _extract_answer(self, text):
        """Uses the specified format/regex to get the answer from the output text."""
        regex = ""
        dotall = False
        answer_format = self.eval_config.answer_format

        if answer_format == "CODEBLOCK":
            regex = r"(?:```sql)(.*?[a-zA-Z].*?)(?:```)"
            dotall = True
        elif answer_format == "BOXED":
            regex = r"(?:boxed\{\{)(.*?[a-zA-Z].*?)(?:\}\})"
            dotall = True
        elif answer_format == "USE_REGEX":
            regex = self.eval_config.extraction_regex
            dotall = self.eval_config.regex_dotall

        if not regex:
            logging.error(
                "Answer format underspecified for BIRD evaluation; should be one of "
                + "{CODEBLOCK, BOXED, USE_REGEX (provide extraction_regex)}.\n"
                + f"Got {answer_format} instead."
            )

        # Use regex to extract answer from text
        if dotall:
            code_matches = re.findall(regex, text, flags=re.DOTALL)
        else:
            code_matches = re.findall(regex, text)

        if not code_matches:
            return "SELECT 1"  # No-op filler

        # Remove comments first
        ans = re.sub(r"--.*?$|/\*.*?\*/", "", code_matches[-1], flags=re.DOTALL)  # Use last match
        # Collapse whitespace
        ans = re.sub(r"\s+", " ", ans)
        # Remove miscellaneous headers that snuck in
        ans = re.sub(r"^\*\*.*\*\*", "", ans).strip()

        return ans

    async def eval_single(self, data_point: dict):
        i = data_point["id"]
        db_id = data_point["db_id"]

        # Retrieve pred and gt
        predicted_sql = self._extract_answer(data_point["generation"])
        ground_truth = data_point["gt_sql"]
        db_place = str(Path(self.db_path, db_id, db_id + ".sqlite"))

        try:
            # Wait for result with timeout as set
            res = func_timeout(self.eval_config.timeout, execute_sql, args=(predicted_sql, ground_truth, db_place))
        except FunctionTimedOut:
            logging.info(f"SQL execution timed out for entry {i}")
            res = 0
        except Exception as e:
            logging.info(f"SQL execution failed for entry {i}:\n{e}")
            res = 0

        data_point["res"] = res
        return data_point

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
import shutil
import subprocess
from pathlib import Path

from nemo_skills.evaluation.evaluator.base import BaseEvaluatorConfig
from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class BFCLEvaluatorConfig(BaseEvaluatorConfig):
    model: str = "o3-mini-2025-01-31-FC"  # Uses the same eval as Llama-Nemotron
    timeout: int = 300


def eval_bfcl(cfg):
    """BFCL (Berkeley Function Calling Leaderboard) evaluation wrapper.

    This function wraps the external BFCL evaluation tool, converting between
    Nemo-Skills format and BFCL format, then merging results back.
    """
    eval_config = BFCLEvaluatorConfig(**cfg)
    model_name = eval_config.model.replace("/", "_")
    jsonl_file = eval_config.input_file
    # Output files are structures as bfcl_v3/TEST_CATEGORY/jsonl_file
    test_category = str(Path(jsonl_file).absolute().parent.name).removeprefix("bfcl_v3.")

    # Convert Nemo-Skills output file to BFCL format
    output_dir = Path("/opt/gorilla/berkeley-function-call-leaderboard") / f"result/{model_name}"
    score_file = (
        Path("/opt/gorilla/berkeley-function-call-leaderboard")
        / f"score/{model_name}"
        / f"BFCL_v3_{test_category}_score.json"
    )

    bfcl_input_file = _convert_to_bfcl_format(jsonl_file, output_dir=output_dir, test_category=test_category)

    try:
        # Run BFCL evaluation using the CLI
        # We need the OpenAI model class decoding functions for evaluation but not really the actual API key for evaluation
        # So we set the API key to a dummy value
        cmd = f"OPENAI_API_KEY=dummy bfcl evaluate --model {eval_config.model} --test-category {test_category}"

        LOG.info(f"Running BFCL evaluation: {cmd}")
        subprocess.run(cmd, shell=True, check=True, timeout=eval_config.timeout)

        # Merge the bfcl_input_file with the score_file, and write to the original file
        _merge_bfcl_results(jsonl_file, bfcl_input_file, score_file)

    except subprocess.TimeoutExpired:
        LOG.error(f"BFCL evaluation timed out after {eval_config.timeout} seconds")
        raise


def _convert_to_bfcl_format(jsonl_file, output_dir, test_category):
    """Convert Nemo-Skills JSONL format to BFCL expected format."""

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    bfcl_file = Path(output_dir, f"BFCL_v3_{test_category}_result.json")
    with open(jsonl_file, "rt", encoding="utf-8") as fin, open(bfcl_file, "wt", encoding="utf-8") as fout:
        for line in fin:
            sample = json.loads(line)
            if sample.get("result", None) is None:
                sample["result"] = sample["generation"]

            fout.write(json.dumps(sample) + "\n")

    return bfcl_file


def _merge_bfcl_results(generation_file, bfcl_fmted_file, score_file):
    """Merge BFCL evaluation results back into the original Nemo-Skills file."""

    # Load the score file has the format that it stores the aggregate scores at the top,
    # and the wrong instances after that
    wrong_instance_ids = set()
    with open(score_file, "rt", encoding="utf-8") as fin:
        first_line = True
        for line in fin:
            if first_line:
                first_line = False
                continue
            wrong_instance = json.loads(line)
            wrong_instance_ids.add(wrong_instance["id"])

    # Write to a temporary file first, then move it to the original file
    temp_file = Path(generation_file).with_suffix(".tmp")
    if temp_file.exists():
        temp_file.unlink()

    with (
        open(generation_file, "rt", encoding="utf-8") as gen_f,
        open(bfcl_fmted_file, "rt", encoding="utf-8") as bfcl_f,
        open(temp_file, "wt", encoding="utf-8") as fout,
    ):
        for gen_line, bfcl_line in zip(gen_f, bfcl_f):
            gen_instance = json.loads(gen_line)
            # Add the bfcl result to the generation instance
            gen_instance.update(json.loads(bfcl_line))

            if gen_instance["id"] in wrong_instance_ids:
                gen_instance["is_correct"] = False
            else:
                gen_instance["is_correct"] = True

            fout.write(json.dumps(gen_instance) + "\n")

    shutil.move(temp_file, generation_file)

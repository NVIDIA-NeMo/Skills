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

import json
import logging
import os
from typing import Any

from nemo_skills.evaluation.evaluator.base import BaseEvaluatorConfig
from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class SpecdecEvaluatorConfig(BaseEvaluatorConfig):
    """Config for the speculative-decoding evaluator.

    Attributes:
        specdec_stats: Pre-computed speculative decoding delta statistics
            (acceptance_length, acceptance_rate, per_position_acceptance_rates,
            etc.).  Injected by :class:`SpecdecGenerationTask` after computing
            the before/after delta from the server's ``/metrics`` endpoint.
    """

    specdec_stats: dict[str, Any]

    def __post_init__(self):
        if not isinstance(self.specdec_stats, dict):
            raise TypeError(f"specdec_stats must be a dictionary, got {type(self.specdec_stats)}")


def eval_specdec(cfg: dict[str, Any]) -> None:
    """Evaluate speculative decoding performance using pre-computed delta stats.

    This evaluator receives speculative decoding statistics that were computed
    by :class:`SpecdecGenerationTask` using a before/after delta of the
    server's Prometheus ``/metrics`` counters.

    It stamps each data point in the output JSONL with the computed metrics so
    that the downstream :class:`SpecdecMetrics` class can aggregate them.

    Metrics stamped onto each data point:

    * ``acceptance_length`` — ``1 + delta_accepted / delta_drafts``
    * ``acceptance_rate`` — ``(delta_accepted / delta_draft_tokens) * 100``
    * ``num_drafts`` — number of draft rounds during this run
    * ``draft_tokens`` — total draft tokens proposed during this run
    * ``accepted_tokens`` — total tokens accepted during this run
    * ``per_position_acceptance_rates`` — list of per-position acceptance
      probabilities

    Args:
        cfg: Evaluator configuration dict.  Must contain ``input_file`` and
            ``specdec_stats`` keys.
    """
    eval_config = SpecdecEvaluatorConfig(**cfg)

    # ------------------------------------------------------------------
    # 1. Read output file
    # ------------------------------------------------------------------
    jsonl_file = eval_config.input_file
    with open(jsonl_file, "rt", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]

    # ------------------------------------------------------------------
    # 2. Inject pre-computed spec-decode stats into each data point
    # ------------------------------------------------------------------
    stats = eval_config.specdec_stats
    LOG.info(
        "Stamping spec-decode stats onto %d data points: "
        "acceptance_length=%.4f, acceptance_rate=%.2f%%, num_drafts=%d",
        len(data),
        stats["acceptance_length"],
        stats["acceptance_rate"],
        stats["num_drafts"],
    )
    for sample in data:
        for key, value in stats.items():
            if key not in sample:
                sample[key] = value

    # ------------------------------------------------------------------
    # 3. Write back
    # ------------------------------------------------------------------
    tmp_file = jsonl_file + "-tmp"
    with open(tmp_file, "wt", encoding="utf-8") as fout:
        for sample in data:
            fout.write(json.dumps(sample) + "\n")

    os.replace(tmp_file, jsonl_file)
    LOG.info("Speculative decoding evaluation complete. Updated %d entries.", len(data))

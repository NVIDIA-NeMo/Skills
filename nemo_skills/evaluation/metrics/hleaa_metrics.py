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

from nemo_skills.evaluation.metrics.math_metrics import MathMetrics
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class HLEAAMetrics(MathMetrics):
    """Metrics for HLE with judge structured output for AA-compatibility."""

    def _postprocess_judgement(self, prediction: dict) -> dict:
        prediction = prediction.copy()
        try:
            judgement = json.loads(prediction["judgement"])
            prediction["judgement"] = "Judgement: {}".format(judgement["correct"])
        except (json.JSONDecodeError, KeyError) as e:
            LOG.debug(f"Failed to parse structured output judgement: {e}")
            prediction["judgement"] = "Judgement: FAILED_TO_POSTPROCESS"
        return prediction

    def update(self, predictions):
        preprocessed_predictions = [self._postprocess_judgement(pred) for pred in predictions]
        super().update(preprocessed_predictions)

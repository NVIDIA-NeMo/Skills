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

"""
IMO GradingBench Dataset

Task: Given a math problem and a student's solution, predict the grade.
- Model predicts one of 4 categories: 7 (Correct), 6 (Almost), 1 (Partial), 0 (Incorrect)
- Accuracy: Predicted vs Reward category (mapped to 7,6,1,0)
- MAE: |Predicted - actual Points (0-7)|

Source: https://github.com/google-deepmind/superhuman/blob/main/imobench/gradingbench.csv
"""

# Settings that define how evaluation should be done by default (all can be changed from cmdline)
DATASET_GROUP = "math"
METRICS_TYPE = "imo-grading"
EVAL_ARGS = "++eval_type=math"
GENERATION_ARGS = "++prompt_config=imobench/gradingbench"

# No judge needed - this is a classification task with exact match evaluation
# The model outputs a grade (0, 1, 6, or 7) and we compare with expected_answer
# Metrics: grading_correct (accuracy) and mae (mean absolute error vs actual points)

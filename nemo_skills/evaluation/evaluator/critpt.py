# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""CritPt evaluator for submitting predictions to Artificial Analysis API."""

import json
import logging
import os
import re

import requests

from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class CritPtEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for CritPt evaluator.

    This evaluator collects all predictions and submits them to the
    Artificial Analysis API for evaluation in batch mode.
    """

    api_url: str = "https://artificialanalysis.ai/api/v2/critpt/evaluate"
    api_key_env: str = "ARTIFICIAL_ANALYSIS_API_KEY"
    request_timeout: int = 600  # Timeout for API request in seconds


class CritPtEvaluator(BaseEvaluator):
    """CritPt evaluator that submits predictions to Artificial Analysis API.

    This evaluator processes CritPt benchmark results by:
    1. Extracting code from generation field
    2. Formatting submissions according to API requirements
    3. Submitting to Artificial Analysis API for evaluation
    4. Adding evaluation results back to data points
    """

    def __init__(self, config: dict, num_parallel_requests: int = 10):
        super().__init__(config, num_parallel_requests)
        self.eval_config = CritPtEvaluatorConfig(**self.config)

        # Get API key from environment
        self.api_key = os.getenv(self.eval_config.api_key_env)
        if not self.api_key:
            raise ValueError(
                f"{self.eval_config.api_key_env} environment variable not found. "
                "Please set it with your API key to use CritPt evaluation."
            )

    def _extract_code_from_generation(self, generation: str) -> str:
        """Extract Python code from generation text.

        Handles both fenced code blocks and plain text.
        """
        # Try to extract from code fence (```python...``` or ```...```)
        code_fence_pattern = r"```(?:python)?\s*\n(.*?)\n```"
        matches = re.findall(code_fence_pattern, generation, re.DOTALL)

        if matches:
            # Return the last code block if multiple exist
            return matches[-1].strip()

        # If no code fence, return the generation as-is
        return generation.strip()

    def _format_submission(self, data_point: dict) -> dict:
        """Format a single data point into CritPt submission format.

        Expected fields in data_point:
        - problem_id: unique identifier for the problem
        - generation: model's generated solution containing code
        - Other optional fields based on CritPt requirements
        """
        code = self._extract_code_from_generation(data_point["generation"])

        if not code.startswith("```"):
            code = "```python\n" + code + "\n```"

        # Submission Object Fields from https://artificialanalysis.ai/documentation#critpt-api

        submission = {
            "problem_id": data_point["problem_id"],
            "generated_code": code,
            "model": "unknown",
            "generation_config": {},
        }

        return submission

    async def eval_full(self) -> None:
        """Evaluate full dataset by submitting all predictions to API."""
        LOG.info(f"Loading predictions from {self.eval_config.input_file}")

        # Check if response file already exists
        response_file = self.eval_config.input_file.replace(".jsonl", "_critpt_response.json")
        if os.path.exists(response_file):
            LOG.info(
                f"Response file {response_file} already exists. Loading cached response and skipping API submission."
            )
            with open(response_file, "r") as f:
                response_data = json.load(f)
        else:
            # Load all data points
            data_points = []
            with open(self.eval_config.input_file, "rt", encoding="utf-8") as f:
                for line in f:
                    data_points.append(json.loads(line))

            LOG.info(f"Found {len(data_points)} data points")

            assert len(data_points) == 70, (
                f"CritPt API only supports 70 submissions at a time, but got {len(data_points)}"
            )

            # Format submissions
            submissions = []
            for data_point in data_points:
                submission = self._format_submission(data_point)
                submissions.append(submission)
            # Save submissions to file
            submission_file = self.eval_config.input_file.replace(".jsonl", "_submissions.jsonl")
            with open(submission_file, "wt", encoding="utf-8") as f:
                for submission in submissions:
                    f.write(json.dumps(submission) + "\n")
            LOG.info(f"Saved {len(submissions)} submissions to {submission_file}")

            # Submit to API
            LOG.info(f"Submitting {len(submissions)} predictions to CritPt API...")
            response_data = self._submit_to_api(submissions)

            # Save API response
            with open(response_file, "w") as f:
                json.dump(response_data, f, indent=2)
            LOG.info(f"Saved API response to {response_file}")

        # Load all data points for processing results
        data_points = []
        with open(self.eval_config.input_file, "rt", encoding="utf-8") as f:
            for line in f:
                data_points.append(json.loads(line))

        # Process aggregate statistics from API response
        # CritPt API returns aggregate accuracy, not per-example results
        accuracy = response_data["accuracy"]
        timeout_rate = response_data["timeout_rate"]
        server_timeout_count = response_data["server_timeout_count"]
        judge_error_count = response_data["judge_error_count"]

        LOG.info(
            f"API Results - Accuracy: {accuracy:.2%}, Timeout Rate: {timeout_rate:.2%}, "
            f"Server Timeouts: {server_timeout_count}, Judge Errors: {judge_error_count}"
        )

        for data_point in data_points:
            # Add evaluation results to data point
            data_point["evaluation_metadata"] = {
                "aggregate_accuracy": accuracy,
                "timeout_rate": timeout_rate,
                "server_timeout_count": server_timeout_count,
                "judge_error_count": judge_error_count,
            }
            data_point["full_dataset_accuracy"] = accuracy

        # Write to temp file then replace original
        temp_file = self.eval_config.input_file + "-tmp"
        LOG.info("Writing results to temporary file")
        with open(temp_file, "wt", encoding="utf-8") as f:
            for data_point in data_points:
                f.write(json.dumps(data_point) + "\n")

        # Atomically replace original with temp file
        os.replace(temp_file, self.eval_config.input_file)
        LOG.info("CritPt evaluation completed successfully")

    def _submit_to_api(self, submissions: list[dict]) -> dict:
        """Submit predictions to the Artificial Analysis API.

        Args:
            submissions: List of submission dictionaries

        Returns:
            API response as a dictionary

        Raises:
            requests.HTTPError: If the API request fails
        """
        payload = {
            "submissions": submissions,
            "batch_metadata": {},
        }

        LOG.debug(f"Sending {len(submissions)} submissions to {self.eval_config.api_url}")

        response = requests.post(
            self.eval_config.api_url,
            json=payload,
            headers={"x-api-key": self.api_key},
            timeout=self.eval_config.request_timeout,
        )

        response.raise_for_status()
        response_data = response.json()
        # add status code to response data
        response_data["status_code"] = response.status_code

        LOG.info(f"API response received with status {response.status_code}")
        LOG.debug(f"Response data: {json.dumps(response_data, indent=2)}")

        return response_data

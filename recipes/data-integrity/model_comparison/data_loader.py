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

"""Data loading and preprocessing module"""

import json
import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


def load_json_data(json_file_path):
    """Load and parse the JSON data"""
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def json_to_dataframe(data):
    """Convert JSON data to DataFrame for easier analysis"""
    rows = []
    for item in data:
        input_text = item["input"][0]["content"] if isinstance(item["input"], list) else str(item.get("input", ""))

        # Extract responses from different generators
        for key, value in item.items():
            if key.startswith("output_"):
                generator = key.replace("output_", "")
                rows.append(
                    {
                        "input": input_text,
                        "generator": generator,
                        "response": value,
                        "split": item.get("split", "unknown"),
                    }
                )

    df = pd.DataFrame(rows)
    return df


def load_and_prepare_data(json_file_path):
    """Load data and prepare it for analysis"""
    logger.info(f"📂 Loading data from: {json_file_path}")

    # Load JSON data
    data = load_json_data(json_file_path)

    # Convert to DataFrame
    df = json_to_dataframe(data)

    # Create summary info
    summary_info = {
        "total_responses": len(df),
        "generators": list(df["generator"].unique()),
        "splits": list(df["split"].unique()),
        "avg_responses_per_generator": len(df) / df["generator"].nunique(),
        "loaded_at": datetime.now().isoformat(),
    }

    logger.info(f"📊 Loaded {len(df)} responses from {df['generator'].nunique()} generators")
    logger.info(f"🤖 Comparing: {', '.join(df['generator'].unique())}")

    return data, df, summary_info

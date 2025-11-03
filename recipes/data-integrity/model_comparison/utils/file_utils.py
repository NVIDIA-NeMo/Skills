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

"""File handling utilities for model comparison analysis"""

import json
import logging
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def get_model_comparison_name(df):
    """Generate descriptive name including all models being compared"""
    if df is not None:
        models = sorted(df["generator"].unique())
        # Clean model names for filename
        clean_models = []
        for model in models:
            # Shorten long model names
            if len(model) > 20:
                # Try to extract key parts
                parts = model.split("-")
                if len(parts) > 2:
                    clean_name = f"{parts[0]}-{parts[1]}"
                else:
                    clean_name = model[:20]
            else:
                clean_name = model
            # Remove problematic characters for filenames
            clean_name = re.sub(r"[^\w\-_\.]", "_", clean_name)
            clean_models.append(clean_name)

        return "_vs_".join(clean_models)
    return "models"


def save_plot(subdirs, df, filename_suffix, title=""):
    """Save plot with organized naming and location"""
    model_names = get_model_comparison_name(df)
    filename = f"{model_names}_{filename_suffix}.png"
    filepath = os.path.join(subdirs["visualizations"], filename)

    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    logger.info(f"💾 Saved: {filename}")
    return filepath


def save_data(subdirs, df, data, filename_suffix, format="csv"):
    """Save data outputs with organized naming"""
    model_names = get_model_comparison_name(df)
    filename = f"{model_names}_{filename_suffix}.{format}"
    filepath = os.path.join(subdirs["data_outputs"], filename)

    if isinstance(data, pd.DataFrame):
        if format == "csv":
            data.to_csv(filepath, index=False)
        elif format == "excel":
            data.to_excel(filepath, index=False)
    elif isinstance(data, dict):
        with open(filepath.replace(f".{format}", ".json"), "w") as f:
            json.dump(data, f, indent=2, default=str)

    logger.info(f"💾 Saved data: {filename}")
    return filepath

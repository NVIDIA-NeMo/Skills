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

"""Response length analysis module"""

import logging

import matplotlib.pyplot as plt
import nltk
import seaborn as sns

from ..utils.file_utils import save_data, save_plot

logger = logging.getLogger(__name__)


def analyze_response_lengths(df, subdirs):
    """Analyze response length distributions with organized output"""
    logger.info(f"\n{'=' * 60}")
    logger.info("📏 RESPONSE LENGTH ANALYSIS")
    line = "=" * 60
    logger.info(line)

    # Calculate various length metrics
    df["char_count"] = df["response"].str.len()
    df["word_count"] = df["response"].apply(lambda x: len(str(x).split()))
    df["sentence_count"] = df["response"].apply(lambda x: len(nltk.sent_tokenize(str(x))) if x else 0)

    # Summary statistics
    length_stats = (
        df.groupby("generator")[["char_count", "word_count", "sentence_count"]]
        .agg(["mean", "median", "std", "min", "max", "count"])
        .round(2)
    )

    logger.info("Length Statistics by Generator:")
    logger.info(length_stats)

    # Save statistics
    save_data(subdirs, df, length_stats, "length_statistics", "csv")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Response Length Analysis", fontsize=16, y=0.98)

    # Word count distribution
    for generator in df["generator"].unique():
        data = df[df["generator"] == generator]["word_count"]
        axes[0, 0].hist(data, alpha=0.7, label=generator, bins=20)
    axes[0, 0].set_xlabel("Word Count")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Word Count Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Box plot comparison
    sns.boxplot(data=df, x="generator", y="word_count", ax=axes[0, 1])
    axes[0, 1].set_title("Word Count Comparison")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].grid(True, alpha=0.3)

    # Character vs word count
    for generator in df["generator"].unique():
        data = df[df["generator"] == generator]
        axes[1, 0].scatter(data["word_count"], data["char_count"], alpha=0.7, label=generator, s=30)
    axes[1, 0].set_xlabel("Word Count")
    axes[1, 0].set_ylabel("Character Count")
    axes[1, 0].set_title("Character vs Word Count Correlation")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Sentence count distribution
    sns.boxplot(data=df, x="generator", y="sentence_count", ax=axes[1, 1])
    axes[1, 1].set_title("Sentence Count Comparison")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(subdirs, df, "response_length_analysis")
    plt.close()

    return length_stats, df

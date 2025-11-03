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

"""Vocabulary diversity analysis module"""

import logging
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd

from ..utils.file_utils import save_data, save_plot

logger = logging.getLogger(__name__)


def analyze_vocabulary_diversity(df, subdirs):
    """Analyze vocabulary diversity metrics with organized output"""
    logger.info(f"\n{'=' * 60}")
    logger.info("📚 VOCABULARY DIVERSITY ANALYSIS")
    logger.info(f"\n{'=' * 60}")

    diversity_metrics = []

    for generator in df["generator"].unique():
        responses = df[df["generator"] == generator]["response"].tolist()
        responses = [str(res) for res in responses]
        all_text = " ".join(responses)

        # Tokenize and clean
        words = nltk.word_tokenize(all_text.lower())
        words = [w for w in words if w.isalpha()]

        # Calculate metrics
        total_words = len(words)
        unique_words = len(set(words))
        ttr = unique_words / total_words if total_words > 0 else 0  # Type-Token Ratio

        # Most common words
        word_freq = Counter(words)
        top_10_words = word_freq.most_common(10)

        # Hapax legomena (words that appear only once)
        hapax = sum(1 for count in word_freq.values() if count == 1)
        hapax_ratio = hapax / unique_words if unique_words > 0 else 0

        # Average word length
        avg_word_length = np.mean([len(w) for w in words]) if words else 0

        diversity_metrics.append(
            {
                "generator": generator,
                "total_words": total_words,
                "unique_words": unique_words,
                "type_token_ratio": round(ttr, 4),
                "hapax_legomena": hapax,
                "hapax_ratio": round(hapax_ratio, 4),
                "avg_word_length": round(avg_word_length, 2),
                "vocabulary_richness": round(unique_words / np.sqrt(total_words) if total_words > 0 else 0, 2),
                "top_10_words": [f"{word}({count})" for word, count in top_10_words],
            }
        )

    diversity_df = pd.DataFrame(diversity_metrics)
    logger.info("Vocabulary Diversity Metrics:")
    display_cols = ["generator", "total_words", "unique_words", "type_token_ratio", "hapax_ratio", "avg_word_length"]
    logger.info(diversity_df[display_cols])

    # Save detailed metrics
    save_data(subdirs, df, diversity_df, "vocabulary_diversity_metrics", "csv")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Vocabulary Diversity Analysis", fontsize=16, y=0.98)
    axes = axes.ravel()

    metrics = ["type_token_ratio", "unique_words", "hapax_ratio", "avg_word_length", "vocabulary_richness"]
    titles = ["Type-Token Ratio", "Unique Words", "Hapax Legomena Ratio", "Average Word Length", "Vocabulary Richness"]

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        axes[i].bar(
            diversity_df["generator"], diversity_df[metric], color=plt.cm.Set3(np.linspace(0, 1, len(diversity_df)))
        )
        axes[i].set_title(title)
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].grid(True, alpha=0.3)

        # Add value labels on bars
        for j, v in enumerate(diversity_df[metric]):
            axes[i].text(j, v + max(diversity_df[metric]) * 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # Hide unused subplot
    axes[5].set_visible(False)

    plt.tight_layout()
    save_plot(subdirs, df, "vocabulary_diversity_analysis")
    plt.close()

    return diversity_df

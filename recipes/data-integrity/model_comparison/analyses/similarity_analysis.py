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

"""Semantic similarity analysis module"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.file_utils import save_data, save_plot
from ..utils.text_utils import calculate_rouge_l

logger = logging.getLogger(__name__)


def analyze_semantic_similarity(df, subdirs, sentence_model=None):
    """Analyze semantic similarity between responses to identical prompts"""
    logger.info(f"\n{'=' * 60}")
    logger.info("🔄 SEMANTIC SIMILARITY ANALYSIS")
    logger.info(f"\n{'=' * 60}")

    if sentence_model is None:
        logger.info("⚠️  Warning: SentenceTransformer model not available. Using TF-IDF fallback...")
        return _fallback_similarity_analysis(df, subdirs)

    # Group by input prompt
    similarity_results = []

    cnt = 0
    for input_text in df["input"].unique():
        prompt_responses = df[df["input"] == input_text]
        cnt += 1

        if (cnt % 100) == 0:
            logger.info(f"Processed {cnt}")

        if len(prompt_responses) < 2:
            continue

        generators = prompt_responses["generator"].tolist()
        responses = prompt_responses["response"].tolist()

        # Compute embeddings
        embeddings = sentence_model.encode(responses)

        # Compute pairwise similarities
        similarity_matrix = cosine_similarity(embeddings)

        # Store results
        for i in range(len(generators)):
            for j in range(i + 1, len(generators)):
                # Calculate ROUGE-L score
                rouge_l_score = calculate_rouge_l(responses[i], responses[j])

                similarity_results.append(
                    {
                        "generator_1": generators[i],
                        "generator_2": generators[j],
                        "semantic_similarity": similarity_matrix[i, j],
                        "rouge_l": rouge_l_score,
                        "input_prompt": input_text,
                        "response_1": responses[i],
                        "response_2": responses[j],
                    }
                )

    similarity_df = pd.DataFrame(similarity_results)
    similarity_df = similarity_df.sort_values("semantic_similarity", ascending=True)

    # Summary statistics
    similarity_summary = (
        similarity_df.groupby(["generator_1", "generator_2"])
        .agg(
            {
                "semantic_similarity": ["mean", "std", "min", "max", "count"],
                "rouge_l": ["mean", "std", "min", "max", "count"],
            }
        )
        .round(4)
    )

    logger.info("Semantic Similarity and ROUGE-L Summary:")
    logger.info(similarity_summary)

    # Save detailed results
    save_data(subdirs, df, similarity_df, "semantic_similarity_detailed", "csv")
    save_data(subdirs, df, similarity_summary, "semantic_similarity_summary", "csv")

    # Create similarity matrices for visualization
    generators = df["generator"].unique()
    n_gen = len(generators)
    avg_semantic_matrix = np.zeros((n_gen, n_gen))
    avg_rouge_matrix = np.zeros((n_gen, n_gen))

    for i, gen1 in enumerate(generators):
        for j, gen2 in enumerate(generators):
            if i == j:
                avg_semantic_matrix[i, j] = 1.0
                avg_rouge_matrix[i, j] = 1.0
            else:
                subset = similarity_df[
                    ((similarity_df["generator_1"] == gen1) & (similarity_df["generator_2"] == gen2))
                    | ((similarity_df["generator_1"] == gen2) & (similarity_df["generator_2"] == gen1))
                ]
                if len(subset) > 0:
                    avg_semantic_matrix[i, j] = subset["semantic_similarity"].mean()
                    avg_rouge_matrix[i, j] = subset["rouge_l"].mean()

    # Create combined visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Semantic similarity heatmap
    mask = np.triu(np.ones_like(avg_semantic_matrix, dtype=bool), k=1)
    sns.heatmap(
        avg_semantic_matrix,
        xticklabels=generators,
        yticklabels=generators,
        annot=True,
        cmap="RdYlBu_r",
        center=0.5,
        square=True,
        fmt=".3f",
        mask=mask,
        cbar_kws={"label": "Cosine Similarity"},
        ax=axes[0],
    )
    axes[0].set_title("Average Semantic Similarity")

    # ROUGE-L heatmap
    sns.heatmap(
        avg_rouge_matrix,
        xticklabels=generators,
        yticklabels=generators,
        annot=True,
        cmap="RdYlBu_r",
        center=0.5,
        square=True,
        fmt=".3f",
        mask=mask,
        cbar_kws={"label": "ROUGE-L F1"},
        ax=axes[1],
    )
    axes[1].set_title("Average ROUGE-L Similarity")

    plt.suptitle("Similarity Comparison", fontsize=16, y=1.02)
    plt.tight_layout()
    save_plot(subdirs, df, "similarity_comparison_heatmaps")
    plt.close()

    # Create dual histogram
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Semantic similarity histogram
    axes[0].hist(similarity_df["semantic_similarity"], bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    axes[0].set_xlabel("Semantic Similarity Score")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Semantic Similarity Scores")
    axes[0].grid(True, alpha=0.3)

    # Add statistics to semantic plot
    mean_sem = similarity_df["semantic_similarity"].mean()
    median_sem = similarity_df["semantic_similarity"].median()
    axes[0].axvline(mean_sem, color="red", linestyle="--", alpha=0.8, label=f"Mean: {mean_sem:.3f}")
    axes[0].axvline(median_sem, color="orange", linestyle="--", alpha=0.8, label=f"Median: {median_sem:.3f}")
    axes[0].legend()

    # ROUGE-L histogram
    axes[1].hist(similarity_df["rouge_l"], bins=30, alpha=0.7, color="lightcoral", edgecolor="black")
    axes[1].set_xlabel("ROUGE-L Score")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of ROUGE-L Scores")
    axes[1].grid(True, alpha=0.3)

    # Add statistics to ROUGE-L plot
    mean_rouge = similarity_df["rouge_l"].mean()
    median_rouge = similarity_df["rouge_l"].median()
    axes[1].axvline(mean_rouge, color="red", linestyle="--", alpha=0.8, label=f"Mean: {mean_rouge:.3f}")
    axes[1].axvline(median_rouge, color="orange", linestyle="--", alpha=0.8, label=f"Median: {median_rouge:.3f}")
    axes[1].legend()

    plt.suptitle("Similarity Score Distributions", fontsize=16, y=1.02)
    plt.tight_layout()
    save_plot(subdirs, df, "similarity_scores_histograms")
    plt.close()

    return similarity_df, similarity_summary


def _fallback_similarity_analysis(df, subdirs):
    """Fallback similarity analysis using TF-IDF"""
    similarity_results = []

    for input_text in df["input"].unique():
        responses = df[df["input"] == input_text]

        if len(responses) < 2:
            continue

        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(responses["response"])
            similarity_matrix = cosine_similarity(tfidf_matrix)

            generators = responses["generator"].tolist()
            response_texts = responses["response"].tolist()

            for i in range(len(generators)):
                for j in range(i + 1, len(generators)):
                    rouge_l_score = calculate_rouge_l(response_texts[i], response_texts[j])
                    similarity_results.append(
                        {
                            "generator_1": generators[i],
                            "generator_2": generators[j],
                            "semantic_similarity": similarity_matrix[i, j],
                            "rouge_l": rouge_l_score,
                        }
                    )
        except Exception:
            continue

    if similarity_results:
        similarity_df = pd.DataFrame(similarity_results)
        save_data(subdirs, df, similarity_df, "semantic_similarity_tfidf", "csv")
        return similarity_df
    return None

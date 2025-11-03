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

"""UMAP-based analysis module"""

import logging
import os

import numpy as np
import pandas as pd
import textstat
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob

from ..utils.file_utils import save_data
from ..utils.model_utils import shorten_model_name

logger = logging.getLogger(__name__)

# UMAP and interactive plotting
try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.info("⚠️  Warning: UMAP not available. Install with: pip install umap-learn")

try:
    import plotly.express as px
    import plotly.graph_objects as go  # noqa: F401

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.info("⚠️  Warning: Plotly not available. Install with: pip install plotly")

write_image = False  # Global flag for saving static images


def analyze_response_embeddings_umap(df, subdirs, sentence_model):
    """Interactive UMAP visualization of response embeddings landscape"""
    logger.info(f"\n{'=' * 60}")
    logger.info("🗺️  RESPONSE EMBEDDING LANDSCAPE (UMAP)")
    logger.info(f"\n{'=' * 60}")

    if not UMAP_AVAILABLE:
        logger.info("⚠️  UMAP not available. Skipping embedding landscape analysis.")
        return None

    if not PLOTLY_AVAILABLE:
        logger.info("⚠️  Plotly not available. Skipping interactive embedding landscape analysis.")
        return None

    if sentence_model is None:
        logger.info("⚠️  SentenceTransformer model not available. Skipping embedding landscape analysis.")
        return None

    # Get embeddings for all responses
    logger.info("🔮 Computing embeddings...")
    embeddings = sentence_model.encode(df["response"].tolist())

    # Apply UMAP
    logger.info("🗺️  Applying UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    # Create DataFrame for plotting with shortened model names
    plot_df = df.copy()
    plot_df["umap_x"] = embedding_2d[:, 0]
    plot_df["umap_y"] = embedding_2d[:, 1]
    plot_df["response_preview"] = plot_df["response"].apply(
        lambda x: str(x)[:150] + "..." if len(str(x)) > 150 else str(x)
    )
    plot_df["input_preview"] = plot_df["input"].apply(lambda x: str(x)[:80] + "..." if len(str(x)) > 80 else x)
    plot_df["model_display"] = plot_df["generator"].apply(shorten_model_name)
    plot_df["original_generator"] = plot_df["generator"]  # Keep original for hover data

    # Create interactive plot
    fig = px.scatter(
        plot_df,
        x="umap_x",
        y="umap_y",
        color="model_display",
        hover_data={
            "response_preview": True,
            "input_preview": True,
            "original_generator": True,
            "word_count": True,
            "umap_x": ":.3f",
            "umap_y": ":.3f",
        },
        title="Interactive Response Embedding Landscape",
        labels={"umap_x": "UMAP Dimension 1", "umap_y": "UMAP Dimension 2", "model_display": "Model"},
    )

    # Customize the plot
    fig.update_layout(
        width=1400,  # Wider to accommodate legend
        height=700,
        showlegend=True,
        title_x=0.5,
        hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, font=dict(size=12), itemwidth=30),
    )

    # Update hover template
    fig.update_traces(
        hovertemplate="<b>Model: %{customdata[2]}</b><br>"
        + "Response: %{customdata[0]}<br>"
        + "Input: %{customdata[1]}<br>"
        + "Word Count: %{customdata[3]}<br>"
        + "Position: (%{x:.3f}, %{y:.3f})<br>"
        + "<extra></extra>"
    )

    logger.info("🎮 Interactive embedding landscape created! Displaying...")

    # Show the interactive plot
    fig.show()

    # Save interactive plot as HTML
    from ..utils.file_utils import get_model_comparison_name

    model_names = get_model_comparison_name(df)
    html_path = os.path.join(subdirs["visualizations"], f"{model_names}_response_embeddings_umap.html")
    fig.write_html(html_path)
    logger.info(f"💾 UMap Interactive plot saved: {html_path}")

    if write_image:
        for ext in ["png", "svg", "pdf"]:
            img_path = os.path.join(subdirs["visualizations"], f"{model_names}_response_embeddings_umap.{ext}")
            fig.write_image(img_path, scale=5 if ext == "png" else 1)
            logger.info(f"💾 UMap Image saved: {img_path}")

    # Save embedding coordinates
    save_data(subdirs, df, plot_df[["generator", "umap_x", "umap_y", "word_count"]], "embedding_coordinates", "csv")

    logger.info("✅ Response embedding landscape analysis complete")
    return plot_df


def analyze_input_response_mapping_umap(df, subdirs, sentence_model):
    """Interactive UMAP visualization of input-response relationship mapping"""
    logger.info(f"\n{'=' * 60}")
    logger.info("🎯 INPUT-RESPONSE MAPPING (UMAP)")
    logger.info(f"\n{'=' * 60}")

    if not UMAP_AVAILABLE:
        logger.info("⚠️  UMAP not available. Skipping input-response mapping analysis.")
        return None

    if not PLOTLY_AVAILABLE:
        logger.info("⚠️  Plotly not available. Skipping interactive input-response mapping analysis.")
        return None

    if sentence_model is None:
        logger.info("⚠️  SentenceTransformer model not available. Skipping input-response mapping analysis.")
        return None

    # Get unique inputs and all responses
    unique_inputs = df["input"].unique()
    all_responses = df["response"].tolist()

    # Embed both inputs and responses
    logger.info("🔮 Computing embeddings for inputs and responses...")
    all_texts = list(unique_inputs) + all_responses
    embeddings = sentence_model.encode(all_texts)

    # Apply UMAP on combined space
    logger.info("🗺️  Applying UMAP to combined input-response space...")
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.05, metric="cosine", random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    n_inputs = len(unique_inputs)

    # Create DataFrame for plotting
    # First, create input data
    input_df = pd.DataFrame(
        {
            "text_preview": [text[:100] + "..." if len(text) > 100 else text for text in unique_inputs],
            "type": "Input Prompt",
            "generator": "Input",
            "original_generator": "Input",
            "word_count": [len(text.split()) for text in unique_inputs],
            "umap_x": embedding_2d[:n_inputs, 0],
            "umap_y": embedding_2d[:n_inputs, 1],
        }
    )

    # Then, create response data with shortened model names
    response_df = df.copy()
    response_df["text_preview"] = response_df["response"].apply(
        lambda x: str(x)[:150] + "..." if len(str(x)) > 150 else x
    )
    response_df["type"] = "Response"
    response_df["generator"] = response_df["generator"].apply(shorten_model_name)
    response_df["original_generator"] = df["generator"]  # Keep original for hover
    response_df["umap_x"] = embedding_2d[n_inputs:, 0]
    response_df["umap_y"] = embedding_2d[n_inputs:, 1]

    # Combine for plotting - ensure same columns
    plot_df = pd.concat(
        [
            input_df[["text_preview", "type", "generator", "original_generator", "word_count", "umap_x", "umap_y"]],
            response_df[["text_preview", "type", "generator", "original_generator", "word_count", "umap_x", "umap_y"]],
        ],
        ignore_index=True,
    )

    # Create interactive plot
    fig = px.scatter(
        plot_df,
        x="umap_x",
        y="umap_y",
        color="generator",
        symbol="type",
        size="word_count",
        size_max=15,
        hover_data={
            "text_preview": True,
            "type": True,
            "original_generator": True,
            "word_count": True,
            "umap_x": ":.3f",
            "umap_y": ":.3f",
        },
        title="Interactive Input-Response Semantic Mapping",
        labels={"umap_x": "UMAP Dimension 1", "umap_y": "UMAP Dimension 2", "generator": "Model", "type": "Text Type"},
    )

    # Customize the plot
    fig.update_layout(
        width=1400,  # Wider to accommodate legend
        height=800,
        showlegend=True,
        title_x=0.5,
        hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, font=dict(size=12), itemwidth=30),
    )

    # Update hover template
    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>"
        + "Type: %{customdata[1]}<br>"
        + "Text: %{customdata[0]}<br>"
        + "Original Model: %{customdata[2]}<br>"
        + "Word Count: %{customdata[3]}<br>"
        + "Position: (%{x:.3f}, %{y:.3f})<br>"
        + "<extra></extra>"
    )

    logger.info("🎮 Interactive input-response mapping created! Displaying...")

    # Show the interactive plot
    fig.show()

    # Save interactive plot as HTML
    from ..utils.file_utils import get_model_comparison_name

    model_names = get_model_comparison_name(df)
    html_path = os.path.join(subdirs["visualizations"], f"{model_names}_input_response_mapping.html")
    fig.write_html(html_path)
    logger.info(f"💾 Interactive plot saved: {html_path}")

    if write_image:
        for ext in ["png", "svg", "pdf"]:
            img_path = os.path.join(subdirs["visualizations"], f"{model_names}_input_response_mapping.{ext}")
            fig.write_image(img_path, scale=5 if ext == "png" else 1)
            logger.info(f"💾 UMap Image saved: {img_path}")

    # Save mapping data
    mapping_df = pd.DataFrame(
        {
            "text": all_texts,
            "type": ["input"] * n_inputs + ["response"] * len(all_responses),
            "generator": [None] * n_inputs + df["generator"].tolist(),
            "umap_x": embedding_2d[:, 0],
            "umap_y": embedding_2d[:, 1],
        }
    )
    save_data(subdirs, df, mapping_df, "input_response_mapping", "csv")

    logger.info("✅ Input-response mapping analysis complete")
    return mapping_df


def analyze_multimodal_space_umap(df, subdirs, sentence_model):
    """Interactive UMAP visualization combining semantic and stylistic features"""
    logger.info(f"\n{'=' * 60}")
    logger.info("🎨 MULTIMODAL SPACE ANALYSIS (UMAP)")
    logger.info(f"\n{'=' * 60}")

    if not UMAP_AVAILABLE:
        logger.info("⚠️  UMAP not available. Skipping multimodal space analysis.")
        return None

    if not PLOTLY_AVAILABLE:
        logger.info("⚠️  Plotly not available. Skipping interactive multimodal space analysis.")
        return None

    if sentence_model is None:
        logger.info("⚠️  SentenceTransformer model not available. Skipping multimodal space analysis.")
        return None

    # Get semantic embeddings
    logger.info("🔮 Computing semantic embeddings...")
    semantic_embeddings = sentence_model.encode(df["response"].tolist())

    # Calculate stylistic features
    logger.info("📏 Computing stylistic features...")
    features = []
    for response in df["response"]:
        try:
            blob = TextBlob(response)
            features.append(
                [
                    len(response.split()),  # word count
                    textstat.flesch_reading_ease(response),  # readability
                    blob.sentiment.polarity,  # sentiment
                    blob.sentiment.subjectivity,  # subjectivity
                    len(set(response.lower().split())),  # unique words
                ]
            )
        except Exception:
            # Handle any errors in feature computation
            features.append([0, 0, 0, 0, 0])

    # Combine features
    stylistic_features = np.array(features)

    # Normalize stylistic features
    scaler = StandardScaler()
    stylistic_features_scaled = scaler.fit_transform(stylistic_features)

    # Concatenate semantic and stylistic features
    combined_features = np.hstack([semantic_embeddings, stylistic_features_scaled])

    # Apply UMAP
    logger.info("🗺️  Applying UMAP to combined semantic-stylistic space...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
    embedding_2d = reducer.fit_transform(combined_features)

    # Create DataFrame for plotting
    plot_df = df.copy()
    plot_df["umap_x"] = embedding_2d[:, 0]
    plot_df["umap_y"] = embedding_2d[:, 1]
    plot_df["readability"] = [f[1] for f in features]
    plot_df["sentiment_polarity"] = [f[2] for f in features]
    plot_df["sentiment_subjectivity"] = [f[3] for f in features]
    plot_df["unique_words"] = [f[4] for f in features]
    plot_df["response_preview"] = plot_df["response"].apply(lambda x: str(x)[:150] + "..." if len(str(x)) > 150 else x)
    plot_df["input_preview"] = plot_df["input"].apply(lambda x: str(x)[:80] + "..." if len(str(x)) > 80 else x)
    plot_df["model_display"] = plot_df["generator"].apply(shorten_model_name)
    plot_df["original_generator"] = plot_df["generator"]  # Keep original for hover data

    # Create interactive plot with color based on readability
    fig = px.scatter(
        plot_df,
        x="umap_x",
        y="umap_y",
        color="readability",
        symbol="model_display",
        size="word_count",
        size_max=20,
        hover_data={
            "response_preview": True,
            "input_preview": True,
            "original_generator": True,
            "readability": ":.1f",
            "sentiment_polarity": ":.2f",
            "sentiment_subjectivity": ":.2f",
            "unique_words": True,
            "word_count": True,
            "umap_x": ":.3f",
            "umap_y": ":.3f",
        },
        color_continuous_scale="RdYlBu_r",
        title="Interactive Multimodal Space (Semantic + Stylistic)",
        labels={
            "umap_x": "UMAP Dimension 1",
            "umap_y": "UMAP Dimension 2",
            "readability": "Readability Score",
            "model_display": "Model",
        },
    )

    # Customize the plot
    fig.update_layout(
        width=1400,  # Wider to accommodate legend
        height=800,
        showlegend=True,
        title_x=0.5,
        hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, font=dict(size=12), itemwidth=30),
    )

    # Update hover template
    fig.update_traces(
        hovertemplate="<b>Model: %{customdata[2]}</b><br>"
        + "Response: %{customdata[0]}<br>"
        + "Input: %{customdata[1]}<br>"
        + "Readability: %{customdata[3]}<br>"
        + "Sentiment: %{customdata[4]}<br>"
        + "Subjectivity: %{customdata[5]}<br>"
        + "Unique Words: %{customdata[6]}<br>"
        + "Total Words: %{customdata[7]}<br>"
        + "Position: (%{x:.3f}, %{y:.3f})<br>"
        + "<extra></extra>"
    )

    logger.info("🎮 Interactive multimodal space visualization created! Displaying...")

    # Show the interactive plot
    fig.show()

    # Save interactive plot as HTML
    from ..utils.file_utils import get_model_comparison_name

    model_names = get_model_comparison_name(df)
    html_path = os.path.join(subdirs["visualizations"], f"{model_names}_multimodal_space.html")
    fig.write_html(html_path)
    logger.info(f"💾 Interactive plot saved: {html_path}")

    if write_image:
        for ext in ["png", "svg", "pdf"]:
            img_path = os.path.join(subdirs["visualizations"], f"{model_names}_multimodal_space.{ext}")
            fig.write_image(img_path, scale=5 if ext == "png" else 1)
            logger.info(f"💾 UMap Image saved: {img_path}")

    # Create a second plot colored by model instead of readability
    # Fix negative readability scores for size parameter
    plot_df["readability_normalized"] = (
        plot_df["readability"] - plot_df["readability"].min() + 1
    )  # Shift to positive values

    fig2 = px.scatter(
        plot_df,
        x="umap_x",
        y="umap_y",
        color="model_display",
        size="readability_normalized",
        size_max=20,
        hover_data={
            "response_preview": True,
            "input_preview": True,
            "original_generator": True,
            "readability": ":.1f",
            "sentiment_polarity": ":.2f",
            "sentiment_subjectivity": ":.2f",
            "unique_words": True,
            "word_count": True,
            "umap_x": ":.3f",
            "umap_y": ":.3f",
        },
        title="Interactive Multimodal Space by Model (Size = Readability)",
        labels={
            "umap_x": "UMAP Dimension 1",
            "umap_y": "UMAP Dimension 2",
            "model_display": "Model",
            "readability_normalized": "Readability Score (Normalized)",
        },
    )

    fig2.update_layout(
        width=1400,  # Wider to accommodate legend
        height=800,
        showlegend=True,
        title_x=0.5,
        hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, font=dict(size=12), itemwidth=30),
    )

    fig2.update_traces(
        hovertemplate="<b>Model: %{customdata[2]}</b><br>"
        + "Response: %{customdata[0]}<br>"
        + "Input: %{customdata[1]}<br>"
        + "Readability: %{customdata[3]}<br>"
        + "Sentiment: %{customdata[4]}<br>"
        + "Subjectivity: %{customdata[5]}<br>"
        + "Unique Words: %{customdata[6]}<br>"
        + "Total Words: %{customdata[7]}<br>"
        + "Position: (%{x:.3f}, %{y:.3f})<br>"
        + "<extra></extra>"
    )

    logger.info("🎮 Second multimodal visualization (by model) created! Displaying...")
    fig2.show()

    # Save second plot
    html_path2 = os.path.join(subdirs["visualizations"], f"{model_names}_multimodal_by_model.html")
    fig2.write_html(html_path2)
    logger.info(f"💾 Second interactive plot saved: {html_path2}")

    if write_image:
        for ext in ["png", "svg", "pdf"]:
            img_path2 = os.path.join(subdirs["visualizations"], f"{model_names}_multimodal_by_model.{ext}")
            fig2.write_image(img_path2, scale=5 if ext == "png" else 1)
            logger.info(f"💾 UMap Image saved: {img_path2}")

    # Save multimodal data
    save_data(
        subdirs,
        df,
        plot_df[
            [
                "generator",
                "umap_x",
                "umap_y",
                "readability",
                "sentiment_polarity",
                "sentiment_subjectivity",
                "unique_words",
            ]
        ],
        "multimodal_features",
        "csv",
    )

    logger.info("✅ Multimodal space analysis complete")
    return plot_df

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

"""Interactive plot creation functions for UMAP visualizations"""

import logging
import os

from ..utils.file_utils import get_model_comparison_name
from ..utils.model_utils import shorten_model_name

logger = logging.getLogger(__name__)

# Interactive plotting
try:
    import plotly.express as px
    import plotly.graph_objects as go  # noqa: F401

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.info("⚠️  Warning: Plotly not available. Install with: pip install plotly")

# UMAP
try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.info("⚠️  Warning: UMAP not available. Install with: pip install umap-learn")

write_image = False  # Global flag for saving static images


def create_response_embeddings_umap(df, subdirs, sentence_model):
    """Create interactive UMAP visualization of response embeddings

    This is a wrapper that calls the analysis function from umap_analysis module
    """
    from ..analyses.umap_analysis import analyze_response_embeddings_umap

    return analyze_response_embeddings_umap(df, subdirs, sentence_model)


def create_input_response_mapping_umap(df, subdirs, sentence_model):
    """Create interactive UMAP visualization of input-response mapping

    This is a wrapper that calls the analysis function from umap_analysis module
    """
    from ..analyses.umap_analysis import analyze_input_response_mapping_umap

    return analyze_input_response_mapping_umap(df, subdirs, sentence_model)


def create_multimodal_space_umap(df, subdirs, sentence_model):
    """Create interactive UMAP visualization of multimodal space

    This is a wrapper that calls the analysis function from umap_analysis module
    """
    from ..analyses.umap_analysis import analyze_multimodal_space_umap

    return analyze_multimodal_space_umap(df, subdirs, sentence_model)


def create_interactive_explorer(df, subdirs, sentence_model):
    """Create interactive UMAP plot with response previews"""
    logger.info(f"\n{'=' * 60}")
    logger.info("🎮 INTERACTIVE RESPONSE EXPLORER")
    logger.info(f"\n{'=' * 60}")

    if not PLOTLY_AVAILABLE:
        logger.info("⚠️  Plotly not available. Skipping interactive explorer.")
        return None

    if not UMAP_AVAILABLE:
        logger.info("⚠️  UMAP not available. Skipping interactive explorer.")
        return None

    if sentence_model is None:
        logger.info("⚠️  SentenceTransformer model not available. Skipping interactive explorer.")
        return None

    # Get embeddings and apply UMAP
    logger.info("🔮 Computing embeddings for interactive visualization...")
    embeddings = sentence_model.encode(df["response"].tolist())

    logger.info("🗺️  Applying UMAP for interactive exploration...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    # Create DataFrame for plotting with shortened model names
    plot_df = df.copy()
    plot_df["umap_x"] = embedding_2d[:, 0]
    plot_df["umap_y"] = embedding_2d[:, 1]
    plot_df["response_preview"] = plot_df["response"].apply(lambda x: str(x)[:200] + "..." if len(str(x)) > 200 else x)
    plot_df["input_preview"] = plot_df["input"].apply(lambda x: str(x)[:100] + "..." if len(str(x)) > 100 else x)
    plot_df["model_display"] = plot_df["generator"].apply(shorten_model_name)
    plot_df["original_generator"] = plot_df["generator"]  # Keep original for hover data

    # Create interactive plot
    fig = px.scatter(
        plot_df,
        x="umap_x",
        y="umap_y",
        color="model_display",
        size="word_count",
        hover_data={
            "response_preview": True,
            "input_preview": True,
            "original_generator": True,
            "word_count": True,
            "umap_x": ":.3f",
            "umap_y": ":.3f",
        },
        title="Interactive Response Embedding Explorer",
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

    logger.info("🎮 Interactive explorer created! Displaying...")

    # Show the interactive plot
    fig.show()

    # Save interactive plot as HTML
    model_names = get_model_comparison_name(df)
    html_path = os.path.join(subdirs["visualizations"], f"{model_names}_interactive_explorer.html")
    fig.write_html(html_path)
    logger.info(f"💾 Interactive plot saved: {html_path}")

    if write_image:
        for ext in ["png", "svg", "pdf"]:
            img_path = os.path.join(subdirs["visualizations"], f"{model_names}_interactive_explorer.{ext}")
            fig.write_image(img_path, scale=5 if ext == "png" else 1)
            logger.info(f"💾 UMap Image saved: {img_path}")

    return fig

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

"""Report generation module"""

import logging
import os
from datetime import datetime

from .utils.file_utils import get_model_comparison_name

logger = logging.getLogger(__name__)


def generate_analysis_report(df, results_dir, subdirs, length_stats, diversity_stats, elapsed_times):
    """Generate comprehensive analysis report"""
    models = list(df["generator"].unique())
    model_comparison_name = get_model_comparison_name(df)

    report_content = f"""
# Model Comparison Analysis Report with UMAP Visualizations
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Models Compared: {", ".join(models)}

## Summary Statistics

### Response Lengths
{length_stats.to_string()}

### Vocabulary Diversity
{diversity_stats.to_string()}

## Key Findings

### Length Analysis
- Longest responses: {df.groupby("generator")["word_count"].mean().idxmax()}
- Shortest responses: {df.groupby("generator")["word_count"].mean().idxmin()}

### Vocabulary Analysis
- Most diverse vocabulary: {diversity_stats.loc[diversity_stats["type_token_ratio"].idxmax(), "generator"]}
- Least diverse vocabulary: {diversity_stats.loc[diversity_stats["type_token_ratio"].idxmin(), "generator"]}

### Similarity Analysis
- Semantic similarity measures content meaning overlap using sentence embeddings
- ROUGE-L measures surface-level lexical overlap using longest common subsequence
- Both metrics provide complementary views of response similarity

### UMAP Spatial Analysis
The UMAP visualizations reveal:
- **Embedding Landscape**: Shows clustering patterns and semantic territories of different models
- **Input-Response Mapping**: Reveals how models interpret and transform input prompts
- **Multimodal Space**: Combines semantic meaning with stylistic features
- **Interactive Explorer**: Allows detailed exploration of individual responses

## Performance Metrics
- Response lengths analysis: {elapsed_times["response_lengths"]:.2f} minutes
- Vocabulary diversity analysis: {elapsed_times["vocabulary_diversity"]:.2f} minutes
- Semantic similarity analysis: {elapsed_times["semantic_similarity"]:.2f} minutes
- Response embeddings UMAP: {elapsed_times["response_embeddings_umap"]:.2f} minutes
- Input-response mapping UMAP: {elapsed_times["input_response_mapping_umap"]:.2f} minutes
- Multimodal space UMAP: {elapsed_times["multimodal_space_umap"]:.2f} minutes
- Interactive UMAP explorer: {elapsed_times["interactive_umap_explorer"]:.2f} minutes

## Files Generated
This analysis generated the following files in: {results_dir}

### Traditional Visualizations:
- Response length analysis
- Vocabulary diversity analysis
- Semantic similarity and ROUGE-L heatmaps
- Similarity score distributions (dual histogram)

### UMAP Visualizations:
- Response embeddings landscape (HTML)
- Input-response mapping (HTML)
- Multimodal space analysis (HTML) - 2 versions
- Interactive explorer (HTML)

### Data Outputs:
- Detailed statistics (CSV format)
- Summary metrics (CSV format)
- UMAP coordinates (CSV format)
- Raw analysis data (JSON format)
"""

    # Save report
    report_path = os.path.join(subdirs["reports"], f"{model_comparison_name}_analysis_report.md")
    with open(report_path, "w") as f:
        f.write(report_content)

    logger.info(f"📋 Comprehensive report saved: {report_path}")
    return report_path


def generate_index_file(results_dir, subdirs, df):
    """Generate index file for easy navigation"""
    models = list(df["generator"].unique())
    model_comparison_name = get_model_comparison_name(df)

    index_content = f"""
# Model Comparison Results Index
Analysis completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Models Compared
{chr(10).join([f"- {model}" for model in models])}

## Directory Structure
```
{results_dir}/
├── visualizations/     # All charts and plots (including UMAP)
├── data_outputs/       # CSV and JSON data files
├── reports/           # Summary reports
└── README.md          # This file
```

## Key Files

### Traditional Analysis
- **Main Report**: reports/{model_comparison_name}_analysis_report.md
- **Length Analysis**: visualizations/{model_comparison_name}_response_length_analysis.png
- **Vocabulary Analysis**: visualizations/{model_comparison_name}_vocabulary_diversity_analysis.png
- **Similarity Heatmaps**: visualizations/{model_comparison_name}_similarity_comparison_heatmaps.png
- **Similarity Histograms**: visualizations/{model_comparison_name}_similarity_scores_histograms.png

### UMAP Analysis
- **Response Embeddings**: visualizations/{model_comparison_name}_response_embeddings_umap.html
- **Input-Response Mapping**: visualizations/{model_comparison_name}_input_response_mapping.html
- **Multimodal Space**: visualizations/{model_comparison_name}_multimodal_space.html
- **Multimodal by Model**: visualizations/{model_comparison_name}_multimodal_by_model.html
- **Interactive Explorer**: visualizations/{model_comparison_name}_interactive_explorer.html

## Data Files
- All CSV files contain detailed numerical results including both semantic similarity and ROUGE-L metrics
- UMAP coordinates are saved for further analysis
- JSON files contain structured analysis metadata
- Similarity analysis includes both semantic (meaning-based) and ROUGE-L (lexical overlap) metrics

## Usage Notes
- Open any HTML file in a web browser for interactive exploration
- All UMAP visualizations are interactive with hover-to-see-details functionality
- UMAP visualizations reveal clustering patterns and semantic relationships
- Point sizes in multimodal plots represent response lengths or readability scores
- Use the interactive features to validate clustering and explore individual responses
- Similarity analysis provides both semantic (meaning-based) and ROUGE-L (lexical overlap) metrics
- Compare the dual heatmaps to understand both semantic and surface-level similarities
"""

    index_path = os.path.join(results_dir, "README.md")
    with open(index_path, "w") as f:
        f.write(index_content)

    logger.info(f"📑 Index file created: {index_path}")
    return index_path

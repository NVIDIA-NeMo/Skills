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

"""Visualization modules for model comparison analysis"""

from .interactive_plots import (
    create_input_response_mapping_umap,
    create_interactive_explorer,
    create_multimodal_space_umap,
    create_response_embeddings_umap,
)
from .static_plots import (
    plot_response_lengths,
    plot_similarity_heatmaps,
    plot_similarity_histograms,
    plot_vocabulary_diversity,
)

__all__ = [
    "plot_response_lengths",
    "plot_vocabulary_diversity",
    "plot_similarity_heatmaps",
    "plot_similarity_histograms",
    "create_response_embeddings_umap",
    "create_input_response_mapping_umap",
    "create_multimodal_space_umap",
    "create_interactive_explorer",
]

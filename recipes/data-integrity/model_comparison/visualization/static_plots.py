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

"""Static plot creation functions"""

import logging

logger = logging.getLogger(__name__)


def plot_response_lengths(df, subdirs):
    """Create response length visualizations

    This is a wrapper that calls the analysis function from length_analysis module
    """
    from ..analyses.length_analysis import analyze_response_lengths

    return analyze_response_lengths(df, subdirs)


def plot_vocabulary_diversity(df, subdirs):
    """Create vocabulary diversity visualizations

    This is a wrapper that calls the analysis function from vocabulary_analysis module
    """
    from ..analyses.vocabulary_analysis import analyze_vocabulary_diversity

    return analyze_vocabulary_diversity(df, subdirs)


def plot_similarity_heatmaps(df, subdirs, sentence_model=None):
    """Create similarity heatmap visualizations

    This is a wrapper that calls the analysis function from similarity_analysis module
    """
    from ..analyses.similarity_analysis import analyze_semantic_similarity

    return analyze_semantic_similarity(df, subdirs, sentence_model)


def plot_similarity_histograms(df, subdirs, sentence_model=None):
    """Create similarity histogram visualizations

    This is handled within the analyze_semantic_similarity function
    """
    # The histograms are created as part of the similarity analysis
    # This function is here for API consistency
    pass

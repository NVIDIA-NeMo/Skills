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

"""Text processing utilities for model comparison analysis"""

import logging

# Configure logging
logger = logging.getLogger(__name__)

# ROUGE metric
try:
    from rouge import Rouge

    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.info("⚠️  Warning: ROUGE not available. Install with: pip install rouge")


def calculate_rouge_l(text1, text2):
    """Calculate ROUGE-L score between two texts"""
    if ROUGE_AVAILABLE:
        try:
            rouge = Rouge()
            scores = rouge.get_scores(text1, text2)
            return scores[0]["rouge-l"]["f"]
        except Exception:
            # Fallback to basic implementation if rouge fails
            return basic_rouge_l(text1, text2)
    else:
        return basic_rouge_l(text1, text2)


def basic_rouge_l(text1, text2):
    """Basic ROUGE-L implementation using LCS"""

    def lcs_length(seq1, seq2):
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        if m == 0 or n == 0:
            return 0

        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    # Tokenize texts
    tokens1 = str(text1).lower().split()
    tokens2 = str(text2).lower().split()

    if len(tokens1) == 0 or len(tokens2) == 0:
        return 0.0

    # Calculate LCS length
    lcs_len = lcs_length(tokens1, tokens2)

    # Calculate ROUGE-L F1 score
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / len(tokens1)
    recall = lcs_len / len(tokens2)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

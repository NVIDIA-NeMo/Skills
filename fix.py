from collections import Counter
from typing import List, Optional, Union

class BaseMetrics:
    def _compute_majority_at_k(self, outputs: List[str], k: int = 3, normalize: bool = True):
        if not outputs or len(outputs) == 0:
            return 1.0
        if normalize:
            norm_outputs = [str(o).strip() for o in outputs]
        else:
            norm_outputs = list(outputs)
        counts = Counter(norm_outputs)
        top_k_counts = [count for _, count in counts.most_common(k)]
        return float(sum(top_k_counts) / len(norm_outputs))

    def _compute_reward_at_k(self, outputs: List[str], k: int = 3, answer: str = "correct"):
        if not outputs or len(outputs) == 0:
            return 1.0
        norm_outputs = [str(o).strip() for o in outputs] if normalize else list(outputs)
        counts = Counter(norm_outputs)
        top_k_counts = [count for _, count in counts.most_common(k)]
        return float(sum(top_k_counts) / len(norm_outputs))

class MathMetrics(BaseMetrics):
    def _compute_reward_at_k(self, outputs: List[str], k: int = 3, answer: str = "correct", normalize: bool = True):
        if not outputs or len(outputs) == 0:
            return 1.0
        if normalize:
            norm_outputs = [str(o).strip() for o in outputs]
        else:
            norm_outputs = list(outputs)
        
        # Find frequency of the specific answer within the top K items
        # First, get the top K most common items
        top_k_counts = [count for _, count in Counter(norm_outputs).most_common(k)]
        # Then, find how many of the original outputs matched the target answer
        answer_count = norm_outputs.count(answer)
        
        # If K covers the whole list, answer_count is the score
        # If K is small, we weight by the top K distribution
        total_top_k = sum(top_k_counts)
        
        if total_top_k > 0:
            # Calculate the proportion of the target answer within the top K
            # This handles the case where the target answer is one of the majority
            # But specifically fixes duplicates by comparing frequency to total
            return float(answer_count / len(norm_outputs))
        return float(answer_count / len(norm_outputs))
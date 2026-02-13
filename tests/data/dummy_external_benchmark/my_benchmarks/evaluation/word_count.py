import re

from nemo_skills.evaluation.evaluator.base import BaseEvaluator


class WordCountEvaluator(BaseEvaluator):
    async def eval_single(self, data_point):
        """Extract predicted answer and compare to expected."""
        match = re.search(r"\\boxed\{(\d+)\}", data_point["generation"])
        predicted = int(match.group(1)) if match else None

        return {
            "predicted_answer": predicted,
            "is_correct": predicted == data_point["expected_answer"],
        }

from nemo_skills.evaluation.metrics.base import BaseMetrics


class WordCountMetrics(BaseMetrics):
    def _get_score_dict(self, prediction):
        return {"is_correct": prediction.get("is_correct", False)}

    def get_incorrect_sample(self, prediction):
        # used for automatic filtering data based on length
        # (we mark too long examples as incorrect using this method)
        prediction = prediction.copy()
        prediction["is_correct"] = False
        prediction["predicted_answer"] = None
        return prediction

    def update(self, predictions):
        # base class provides convenient helpers for calculating
        # common metrics like majority / pass
        super().update(predictions)
        predicted_answers = [pred["predicted_answer"] for pred in predictions]
        self._compute_pass_at_k(
            predictions=predictions,
            predicted_answers=predicted_answers,
        )
        self._compute_majority_at_k(
            predictions=predictions,
            predicted_answers=predicted_answers,
        )

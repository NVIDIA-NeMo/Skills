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

import json
import logging
import re
from collections import defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class SysBenchMetrics(BaseMetrics):
    """Metrics for SysBench dataset.

    This dataset uses an LLM-based equality checker (Officially, GPT-4o)
    to evaluate whether candidate answers are consistent with official answers.
    
    Implements official SysBench metrics:
    - CSR (Constraint Satisfaction Rate): Per constraint type accuracy
    - ISR (Instruction Satisfaction Rate): Per alignment type accuracy  
    - SSR (Session Satisfaction Rate): Per turn with accumulated session success
    """

    # Constraint type mapping (Chinese to English)
    CONSTRAINT_TYPES = {
        "动作约束": "Action",
        "内容约束": "Content",
        "背景约束": "Background",
        "角色约束": "Role",
        "格式约束": "Format",
        "风格约束": "Style",
    }
    
    NUM_TURNS = 5  # SysBench has 5 turns per dialogue

    def __init__(self):
        super().__init__()
        # Track metrics by domain/scene category for detailed analysis
        self.category_metrics = defaultdict(lambda: defaultdict(float))
        self.category_totals = defaultdict(int)
        self.token_stats = defaultdict(list)  # Track input token statistics
        
        # Store all predictions for computing official SysBench metrics
        self.all_predictions = []

    def reset(self):
        super().reset()
        self.category_metrics = defaultdict(lambda: defaultdict(float))
        self.category_totals = defaultdict(int)
        self.token_stats = defaultdict(list)
        self.all_predictions = []

    @staticmethod
    def parse_sysbench_judgement(judgement: str) -> dict | None:
        """Parse SysBench JSON judgement.
        
        Expected format:
        {
          "评判理由": "reason",
          "评判结果": {
            "1": "是",
            "2": "否",
            ...
          }
        }
        
        Note: The official SysBench judge prompt uses unquoted integer keys (1: instead of "1":)
        which is invalid JSON. We handle this by converting them to quoted strings.
        
        Returns:
            Dict with judgement results, or None if parsing fails
        """
        if judgement is None:
            return None
        
        try:
            # Try to extract JSON from code blocks if present
            judgement = judgement.strip()
            if "```json" in judgement:
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', judgement, re.DOTALL)
                if json_match:
                    judgement = json_match.group(1)
            elif "```" in judgement:
                json_match = re.search(r'```\s*(\{.*?\})\s*```', judgement, re.DOTALL)
                if json_match:
                    judgement = json_match.group(1)
            
            # Fix unquoted integer keys (e.g., {1: "是" -> {"1": "是")
            # This is needed because the official SysBench judge prompt format uses integer keys
            judgement = re.sub(r'([\s{,])(\d+)(\s*:\s*")', r'\1"\2"\3', judgement)
            
            # Parse JSON
            result = json.loads(judgement)
            
            # Validate required fields
            if "评判结果" in result and isinstance(result["评判结果"], dict):
                return result
            
            return None
        except (json.JSONDecodeError, AttributeError, ValueError):
            # ValueError can occur if JSON contains extremely large integers
            # that exceed Python's integer string conversion limit
            return None
    
    @staticmethod
    def is_sysbench_correct(judgement: str, prompt_infos: dict = None, current_user_content: str = None) -> bool:
        """Check if SysBench judgement indicates all criteria are met.
        
        A response is correct only if ALL criteria have "是" (yes) as the result.
        
        Args:
            judgement: The LLM judge's response
            prompt_infos: Full prompt_infos dict containing criteria
            current_user_content: Current user prompt (to lookup criteria in prompt_infos)
        
        Returns:
            True if all criteria are met, False otherwise
        """
        parsed = SysBenchMetrics.parse_sysbench_judgement(judgement)
        if parsed is None:
            return False
        
        results = parsed.get("评判结果", {})
        if not results:
            return False
        
        # Validate that judge output keys match expected criteria IDs
        if prompt_infos and current_user_content:
            criteria = prompt_infos[current_user_content]["criteria"]
            expected_keys = set(str(k) for k in criteria.keys())
            actual_keys = set(str(k) for k in results.keys())
            if expected_keys != actual_keys:
                LOG.warning(
                    f"Judge output keys {actual_keys} don't match expected criteria IDs {expected_keys}. "
                    f"Missing: {expected_keys - actual_keys}, Extra: {actual_keys - expected_keys}"
                )
                # Return False if keys don't match - judge output is unreliable
                return False
        
        # All criteria must be "是" for the turn to be correct
        return all(value == "是" for value in results.values())

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """Get correctness scores for a prediction using LLM-based equality checker."""
        correctness_dict = {}

        # Primary evaluation method: LLM-based equality checker
        if "judgement" in prediction:
            # Invalid generation: reasoning is not finished or non-reasoning generation is empty
            generation = prediction.get("current_assistant_content", prediction.get("generation", ""))
            correctness_dict["generation_valid"] = len(generation.strip()) > 0
            
            # Check if all criteria passed according to the judge
            correctness_dict["judge_correct"] = (
                self.is_sysbench_correct(
                    prediction["judgement"],
                    prompt_infos=prediction.get("prompt_infos"),
                    current_user_content=prediction.get("current_user_content")
                ) if correctness_dict["generation_valid"] else False
            )

        return correctness_dict

    @classmethod
    def get_incorrect_sample(cls, prediction: dict) -> dict:
        """Return a prediction that evaluates as incorrect."""
        prediction = prediction.copy()
        prediction["judgement"] = "INCORRECT"
        prediction["predicted_answer"] = None
        return prediction

    def _update_category_metrics(self, prediction: dict, score_dict: dict):
        """Update per-category metrics if domain/scene is available."""
        # Use domain as the primary category for SysBench
        domain = prediction.get("domain", "unknown")
        scene = prediction.get("scene", "unknown")
        category = f"{domain}/{scene}" if domain != "unknown" and scene != "unknown" else domain
        
        self.category_totals[category] += 1

        for score_method, is_correct in score_dict.items():
            if is_correct:
                self.category_metrics[category][score_method] += 1

    def _update_token_stats(self, prediction: dict):
        """Track input token statistics by category."""
        domain = prediction.get("domain", "unknown")
        scene = prediction.get("scene", "unknown")
        category = f"{domain}/{scene}" if domain != "unknown" and scene != "unknown" else domain
        
        input_tokens = prediction.get("num_input_tokens") or prediction.get("input_tokens")
        if input_tokens is not None:
            self.token_stats[category].append(int(input_tokens))

    def update(self, predictions):
        """Update the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                Each prediction should contain 'judgement' from LLM equality checker.
        """
        super().update(predictions)

        # Update category metrics and token stats using the first prediction
        # (they should all have the same expected answer and metadata)
        if predictions:
            pred = predictions[0]
            score_dict = self._get_score_dict(pred)
            self._update_category_metrics(pred, score_dict)
            self._update_token_stats(pred)
            
            # Store prediction with computed correctness for official metrics
            pred_copy = pred.copy()
            pred_copy["_is_correct"] = score_dict.get("judge_correct", False)
            self.all_predictions.append(pred_copy)

        # Compute standard pass@k and majority@k metrics
        # Here we use 'judgement' and 'current_assistant_content' (or 'generation') to calculate score and no_answer metric
        predicted_answers = [
            pred.get("current_assistant_content", pred.get("generation", ""))
            for pred in predictions
        ]

        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

    def _compute_csr(self):
        """Compute Constraint Satisfaction Rate (CSR) per constraint type.
        
        CSR measures what percentage of constraints of each type are satisfied.
        """
        # Count correct/total for each constraint type
        constraint_correct = defaultdict(int)
        constraint_total = defaultdict(int)
        
        for pred in self.all_predictions:
            is_correct = pred.get("_is_correct", False)
            prompt_infos = pred["prompt_infos"]
            current_user_content = pred["current_user_content"]
            
            # Get the criteria for this turn
            criteria = prompt_infos[current_user_content]["criteria"]
            
            # Parse judgement to get per-criterion results
            judgement = pred.get("judgement", "")
            parsed = self.parse_sysbench_judgement(judgement)
            results = parsed.get("评判结果", {}) if parsed else {}
            
            for crit_id, crit_info in criteria.items():
                crit_type = crit_info["criteria_type"]
                crit_type_en = self.CONSTRAINT_TYPES.get(crit_type, crit_type)
                
                # Check if this criterion passed
                crit_passed = results.get(str(crit_id), "否") == "是"
                
                constraint_total[crit_type_en] += 1
                if crit_passed:
                    constraint_correct[crit_type_en] += 1
        
        # Compute rates
        csr = {}
        total_correct = 0
        total_count = 0
        for crit_type in self.CONSTRAINT_TYPES.values():
            if constraint_total[crit_type] > 0:
                csr[crit_type] = 100.0 * constraint_correct[crit_type] / constraint_total[crit_type]
                total_correct += constraint_correct[crit_type]
                total_count += constraint_total[crit_type]
            else:
                csr[crit_type] = 0.0
        
        csr["Total"] = 100.0 * total_correct / total_count if total_count > 0 else 0.0
        return csr
    
    def _compute_isr(self):
        """Compute Instruction Satisfaction Rate (ISR) per alignment type.
        
        ISR measures accuracy separately for aligned vs misaligned instructions.
        """
        alignment_correct = defaultdict(int)
        alignment_total = defaultdict(int)
        
        for pred in self.all_predictions:
            is_correct = pred.get("_is_correct", False)
            alignment = pred.get("alignment", "unknown")
            
            alignment_total[alignment] += 1
            alignment_total["Total"] += 1
            if is_correct:
                alignment_correct[alignment] += 1
                alignment_correct["Total"] += 1
        
        isr = {}
        for align_type in ["align", "misalign", "Total"]:
            if alignment_total[align_type] > 0:
                isr[align_type] = 100.0 * alignment_correct[align_type] / alignment_total[align_type]
            else:
                isr[align_type] = 0.0
        
        return isr
    
    def _compute_ssr(self):
        """Compute Session Satisfaction Rate (SSR) with accumulated session success.
        
        SSR tracks per-turn success with the constraint that if turn N fails,
        all subsequent turns are counted as failed (accumulated session success).
        Separates by rounds_related (dependent vs parallel/independent).
        """
        # Group predictions by system_id
        dialogues = defaultdict(list)
        for pred in self.all_predictions:
            system_id = pred.get("system_id")
            if system_id is not None:
                dialogues[system_id].append(pred)
        
        # Sort turns within each dialogue
        for system_id in dialogues:
            dialogues[system_id].sort(key=lambda x: x.get("turn_idx", 0))
        
        # Track per-turn success with accumulated logic
        # dependent[turn] = (correct, total), parallel[turn] = (correct, total)
        dependent_correct = [0] * self.NUM_TURNS
        dependent_total = [0] * self.NUM_TURNS
        parallel_correct = [0] * self.NUM_TURNS
        parallel_total = [0] * self.NUM_TURNS
        
        # Also track accumulated session scores (all turns passed up to turn N)
        dependent_session_scores = []
        parallel_session_scores = []
        
        for system_id, turns in dialogues.items():
            if not turns:
                continue
                
            rounds_related = turns[0].get("rounds_related", False)
            accumulated_success = 0
            
            for turn_idx, turn in enumerate(turns):
                if turn_idx >= self.NUM_TURNS:
                    break
                    
                is_correct = turn.get("_is_correct", False)
                
                # Update accumulated success
                if is_correct and accumulated_success == turn_idx:
                    accumulated_success += 1
                
                # Count this turn as successful only if all previous turns succeeded
                turn_success = (accumulated_success == turn_idx + 1)
                
                if rounds_related:
                    dependent_total[turn_idx] += 1
                    if turn_success:
                        dependent_correct[turn_idx] += 1
                else:
                    parallel_total[turn_idx] += 1
                    if turn_success:
                        parallel_correct[turn_idx] += 1
            
            # Record session score (how many turns succeeded in a row)
            if rounds_related:
                dependent_session_scores.append(accumulated_success)
            else:
                parallel_session_scores.append(accumulated_success)
        
        # Compute SSR metrics
        ssr = {
            "dependent": {},
            "parallel": {},
        }
        
        # Per-turn rates for dependent (multi-turn related)
        for i in range(self.NUM_TURNS):
            if dependent_total[i] > 0:
                ssr["dependent"][f"R{i+1}"] = 100.0 * dependent_correct[i] / dependent_total[i]
            else:
                ssr["dependent"][f"R{i+1}"] = 0.0
        
        # Average session score for dependent
        if dependent_session_scores:
            ssr["dependent"]["SSR"] = 100.0 * sum(dependent_session_scores) / (len(dependent_session_scores) * self.NUM_TURNS)
        else:
            ssr["dependent"]["SSR"] = 0.0
        
        # Per-turn rates for parallel (independent)
        for i in range(self.NUM_TURNS):
            if parallel_total[i] > 0:
                ssr["parallel"][f"R{i+1}"] = 100.0 * parallel_correct[i] / parallel_total[i]
            else:
                ssr["parallel"][f"R{i+1}"] = 0.0
        
        # Average session score for parallel
        if parallel_session_scores:
            ssr["parallel"]["SSR"] = 100.0 * sum(parallel_session_scores) / (len(parallel_session_scores) * self.NUM_TURNS)
        else:
            ssr["parallel"]["SSR"] = 0.0
        
        # Total SSR
        all_session_scores = dependent_session_scores + parallel_session_scores
        if all_session_scores:
            ssr["Total"] = 100.0 * sum(all_session_scores) / (len(all_session_scores) * self.NUM_TURNS)
        else:
            ssr["Total"] = 0.0
        
        return ssr

    def get_metrics(self):
        """Get all computed metrics including official SysBench metrics (CSR, ISR, SSR)."""
        metrics_dict = super().get_metrics()

        # Compute official SysBench metrics
        if self.all_predictions:
            csr = self._compute_csr()
            isr = self._compute_isr()
            ssr = self._compute_ssr()
            
            # Add to metrics dict for the main evaluation mode
            for eval_mode in metrics_dict:
                # Add headline metrics
                metrics_dict[eval_mode]["CSR"] = csr.get("Total", 0.0)
                metrics_dict[eval_mode]["ISR"] = isr.get("Total", 0.0)
                metrics_dict[eval_mode]["SSR"] = ssr.get("Total", 0.0)
                
                # Add detailed breakdowns
                metrics_dict[eval_mode]["csr_breakdown"] = csr
                metrics_dict[eval_mode]["isr_breakdown"] = isr
                metrics_dict[eval_mode]["ssr_breakdown"] = ssr
            
            # Print official SysBench metrics table
            self._print_sysbench_metrics(csr, isr, ssr)

        # Add per-category metrics (domain/scene breakdown)
        if self.category_totals:
            category_results = {}
            for category, total in self.category_totals.items():
                category_results[category] = {}
                for score_method, correct_count in self.category_metrics[category].items():
                    accuracy = 100.0 * correct_count / total
                    category_results[category][score_method] = accuracy

                category_results[category]["total_samples"] = total

                # Add token statistics
                if category in self.token_stats and self.token_stats[category]:
                    tokens = self.token_stats[category]
                    category_results[category]["avg_input_tokens"] = int(sum(tokens) / len(tokens))
                    category_results[category]["max_input_tokens"] = max(tokens)
                    category_results[category]["min_input_tokens"] = min(tokens)

            # Add category breakdown to the main evaluation mode
            for eval_mode in metrics_dict:
                if eval_mode == f"pass@1[avg-of-{self.max_k}]":  # Only add to the main evaluation mode
                    metrics_dict[eval_mode]["category_breakdown"] = category_results

        return metrics_dict
    
    def _print_sysbench_metrics(self, csr, isr, ssr):
        """Print official SysBench metrics in a formatted table."""
        width = 70
        
        print("\n" + "=" * width)
        print(" Official SysBench Metrics ".center(width))
        print("=" * width)
        
        # Main metrics
        print(f"\n{'Metric':<20} {'Score':>10}")
        print("-" * 32)
        print(f"{'CSR (Total):':<20} {csr.get('Total', 0):.2f}%")
        print(f"{'ISR (Total):':<20} {isr.get('Total', 0):.2f}%")
        print(f"{'SSR (Total):':<20} {ssr.get('Total', 0):.2f}%")
        
        # CSR breakdown
        print(f"\n{'-' * width}")
        print(" CSR (Constraint Satisfaction Rate) by Type ".center(width))
        print("-" * width)
        print(f"{'Constraint Type':<20} {'Score':>10}")
        print("-" * 32)
        for ctype in ["Action", "Content", "Background", "Role", "Format", "Style"]:
            if ctype in csr:
                print(f"{ctype:<20} {csr[ctype]:.2f}%")
        
        # ISR breakdown
        print(f"\n{'-' * width}")
        print(" ISR (Instruction Satisfaction Rate) by Alignment ".center(width))
        print("-" * width)
        print(f"{'Alignment':<20} {'Score':>10}")
        print("-" * 32)
        print(f"{'Aligned:':<20} {isr.get('align', 0):.2f}%")
        print(f"{'Misaligned:':<20} {isr.get('misalign', 0):.2f}%")
        
        # SSR breakdown
        print(f"\n{'-' * width}")
        print(" SSR (Session Satisfaction Rate) by Turn ".center(width))
        print("-" * width)
        
        # Dependent (multi-turn related)
        print("\nMulti-turn Dependent:")
        dep = ssr.get("dependent", {})
        for i in range(1, self.NUM_TURNS + 1):
            print(f"  R{i}: {dep.get(f'R{i}', 0):.2f}%")
        print(f"  SSR: {dep.get('SSR', 0):.2f}%")
        
        # Parallel (independent)
        print("\nMulti-turn Parallel:")
        par = ssr.get("parallel", {})
        for i in range(1, self.NUM_TURNS + 1):
            print(f"  R{i}: {par.get(f'R{i}', 0):.2f}%")
        print(f"  SSR: {par.get('SSR', 0):.2f}%")
        
        print("\n" + "=" * width + "\n")

    def evaluations_to_print(self):
        """Return which evaluations should be printed in the summary."""
        if self.max_k > 1:
            return [f"pass@1[avg-of-{self.max_k}]", f"majority@{self.max_k}", f"pass@{self.max_k}"]
        else:
            return ["pass@1"]

    def metrics_to_print(self):
        """Control which metrics are displayed in the summary table."""
        from nemo_skills.evaluation.metrics.base import default_formatting

        # Show official SysBench metrics (CSR, ISR, SSR) as main metrics
        return {
            "CSR": default_formatting,
            "ISR": default_formatting,
            "SSR": default_formatting,
            "judge_correct": default_formatting,
            "num_entries": default_formatting,
            "no_answer": default_formatting,
        }

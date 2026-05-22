# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import json
import re
from collections import defaultdict
from pathlib import Path

from nemo_skills.evaluation.metrics.base import BaseMetrics


class CCCMetrics(BaseMetrics):
    """Metrics aggregator for CCC submissions and pass@k-style reports."""

    def __init__(self, **kwargs):
        """Initialize CCC metrics state and base metric options."""
        super().__init__(**kwargs)
        self.eval_results_dir = None
        self.random_seeds_by_index = []
        self.reset()

    def reset(self):
        """Reset counters and per-problem accumulation state."""
        super().reset()
        self.predictions_by_problem = defaultdict(list)
        self._solutions_written = False

    def setup(self, input_files):
        """Capture run metadata and random-seed indices from input filenames."""
        sorted_files = sorted(str(path) for path in input_files)
        if sorted_files:
            self.eval_results_dir = str(Path(sorted_files[0]).resolve().parent)
        pattern = re.compile(r"output-rs(\d+)\.jsonl$")
        self.random_seeds_by_index = []
        for path in sorted_files:
            match = pattern.search(path)
            self.random_seeds_by_index.append(int(match.group(1)) if match else None)

    def update(self, predictions):
        """Update aggregate metrics and index predictions by problem."""
        super().update(predictions)
        self._compute_pass_at_k(predictions)
        if not predictions:
            return

        annotated_predictions = []
        for idx, prediction in enumerate(predictions):
            annotated_prediction = dict(prediction)
            if idx < len(self.random_seeds_by_index):
                annotated_prediction["_rs"] = self.random_seeds_by_index[idx]
            annotated_predictions.append(annotated_prediction)

        problem_id = annotated_predictions[0].get("problem_id", annotated_predictions[0]["name"])
        self.predictions_by_problem[problem_id].extend(annotated_predictions)

    def _get_score_dict(self, submission):
        """Compute normalized score/correctness for a labeled subtask submission."""
        subtask = submission.get("subtask")
        subtask_result = submission.get("test_case_results", {}).get(subtask, {})
        score = float(subtask_result.get("score", 0.0))
        max_score = float(submission.get("subtask_score", 0.0))
        normalized = score / max_score if max_score > 0 else 0.0
        return {"correct": 1 if max_score > 0 and score >= max_score else 0, "score": normalized}

    def _aggregate_row_group(self, submissions, mode: str, subtask_name: str, declared_max_score: float | None = None):
        """Aggregate repeated row submissions for one subtask under avg/best mode."""
        scores = []
        sample_passed = []
        sample_total = []
        secret_passed = []
        secret_total = []
        compile_successes = []
        compile_attempts = []
        max_score = 0.0
        for submission in submissions[: self.max_k]:
            subtask_result = submission.get("test_case_results", {}).get(subtask_name, {})
            score = float(subtask_result.get("score", 0.0))
            outputs = subtask_result.get("outputs", [])
            # Use the declared subtask score as the scoring maximum.
            # Counting outputs only works for per-test ICPC tasks and breaks weighted IOI subtasks.
            if declared_max_score is not None:
                output_max_score = float(declared_max_score)
            elif "max_score" in subtask_result:
                output_max_score = float(subtask_result.get("max_score", 0.0))
            else:
                raise ValueError(
                    f"Max score is undefined for subtask '{subtask_name}'. "
                    "Expected declared subtask_score or test_case_results[*].max_score."
                )
            max_score = max(max_score, output_max_score)
            sample_tests = [out for out in outputs if out.get("test_group") == "sample"]
            secret_tests = [out for out in outputs if out.get("test_group") == "secret"]
            scores.append(score)
            sample_passed.append(sum(1 for out in sample_tests if float(out.get("score", 0.0)) > 0.0))
            sample_total.append(len(sample_tests))
            secret_passed.append(sum(1 for out in secret_tests if float(out.get("score", 0.0)) > 0.0))
            secret_total.append(len(secret_tests))
            compile_successes.append(1 if outputs and any(out.get("compile_success") for out in outputs) else 0)
            compile_attempts.append(1 if outputs else 0)

        if not scores:
            return {
                "score": 0.0,
                "max_score": max_score,
                "sample_tests_passed": 0,
                "sample_tests_total": 0,
                "secret_tests_passed": 0,
                "secret_tests_total": 0,
                "compile_successes": 0,
                "compile_attempts": 0,
                "compile_success_rate": 0.0,
                "submission_stats": {
                    "num_submissions": 0,
                    "min_score": 0.0,
                    "avg_score": 0.0,
                    "max_score": 0.0,
                    "nonzero_submissions": 0,
                    "full_score_submissions": 0,
                    "avg_sample_tests_passed": 0.0,
                    "avg_secret_tests_passed": 0.0,
                    "avg_compile_successes": 0.0,
                    "avg_compile_attempts": 0.0,
                    "avg_compile_success_rate": 0.0,
                },
            }

        agg_score = max(scores) if mode == "best" else sum(scores) / len(scores)
        agg_sample_passed = max(sample_passed) if mode == "best" else sum(sample_passed) / len(sample_passed)
        agg_secret_passed = max(secret_passed) if mode == "best" else sum(secret_passed) / len(secret_passed)
        # Compile counters should reflect raw execution volume across submissions,
        # not a best/avg row score. This keeps totals aligned with the number of
        # evaluated generations that produced outputs for this row.
        agg_compile_successes = sum(compile_successes)
        agg_compile_attempts = sum(compile_attempts)
        agg_compile_success_rate = (
            (100.0 * agg_compile_successes / agg_compile_attempts) if agg_compile_attempts else 0.0
        )

        return {
            "score": agg_score,
            "max_score": max_score,
            "sample_tests_passed": agg_sample_passed,
            "sample_tests_total": max(sample_total) if sample_total else 0,
            "secret_tests_passed": agg_secret_passed,
            "secret_tests_total": max(secret_total) if secret_total else 0,
            "compile_successes": agg_compile_successes,
            "compile_attempts": agg_compile_attempts,
            "compile_success_rate": agg_compile_success_rate,
            "submission_stats": {
                "num_submissions": len(scores),
                "min_score": min(scores),
                "avg_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "nonzero_submissions": sum(1 for s in scores if s > 0.0),
                "full_score_submissions": sum(1 for s in scores if max_score > 0 and s >= max_score),
                "avg_sample_tests_passed": sum(sample_passed) / len(sample_passed),
                "avg_secret_tests_passed": sum(secret_passed) / len(secret_passed),
                "avg_compile_successes": sum(compile_successes) / len(compile_successes),
                "avg_compile_attempts": sum(compile_attempts) / len(compile_attempts),
                "avg_compile_success_rate": (
                    100.0 * sum(compile_successes) / sum(compile_attempts) if sum(compile_attempts) else 0.0
                ),
            },
        }

    def _build_problem_reports(self, mode: str):
        """Build detailed per-problem and global metric reports for a mode."""
        total_score = 0.0
        total_max_score = 0.0
        per_problem_report = []
        total_sample_passed = 0.0
        total_sample_tests = 0
        total_secret_passed = 0.0
        total_secret_tests = 0
        total_submission_rows = 0
        total_nonzero_submission_rows = 0
        total_full_score_submission_rows = 0
        total_compile_successes = 0.0
        total_compile_attempts = 0.0
        problems_fully_solved = 0
        problems_sample_fully_solved = 0

        for problem_id, submissions in sorted(self.predictions_by_problem.items()):
            problem_name = submissions[0]["name"]
            grouped_rows = defaultdict(list)
            for idx, submission in enumerate(submissions):
                row_key = submission.get("id", f"__row_{idx}")
                grouped_rows[row_key].append(submission)
            declared_max_by_subtask = {}
            for submission in submissions:
                st = submission.get("subtask")
                if st is None:
                    continue
                declared = float(submission.get("subtask_score", 0.0))
                declared_max_by_subtask[st] = max(declared_max_by_subtask.get(st, 0.0), declared)
            all_subtasks = set()
            for submission in submissions:
                all_subtasks.update(submission.get("test_case_results", {}).keys())
            missing_max_subtasks = sorted(st for st in all_subtasks if st not in declared_max_by_subtask)
            if missing_max_subtasks:
                raise ValueError(
                    f"Problem '{problem_id}' has subtasks without defined max score: {missing_max_subtasks}. "
                    "Each subtask must have a declared subtask_score."
                )
            subtasks = {}
            labeled_row_reports = []
            for row_submissions in grouped_rows.values():
                labeled_subtask = row_submissions[0].get("subtask")
                if labeled_subtask is not None:
                    labeled_row_reports.append(
                        self._aggregate_row_group(
                            row_submissions,
                            mode,
                            subtask_name=labeled_subtask,
                            declared_max_score=declared_max_by_subtask.get(labeled_subtask),
                        )
                    )
                # Each row has full subtask results in test_case_results.
                # Score this row against all subtasks, then keep the best row per subtask.
                row_subtasks = row_submissions[0].get("test_case_results", {}).keys()
                for subtask in row_subtasks:
                    row_report = self._aggregate_row_group(
                        row_submissions,
                        mode,
                        subtask_name=subtask,
                        declared_max_score=declared_max_by_subtask.get(subtask),
                    )
                    prev = subtasks.get(subtask)
                    if prev is None or row_report["score"] > prev["score"]:
                        subtasks[subtask] = row_report

            problem_score = 0.0
            problem_max_score = 0.0
            correct_subtasks = 0
            problem_sample_passed = 0.0
            problem_sample_tests = 0
            problem_secret_passed = 0.0
            problem_secret_tests = 0
            problem_compile_successes = 0.0
            problem_compile_attempts = 0.0
            # Row/test counters should be counted once per labeled input row.
            # Summing per-subtask reports inflates IOI-style problems because the same row
            # carries test_case_results for many subtasks.
            problem_submission_rows = len(labeled_row_reports)
            problem_nonzero_submission_rows = sum(1 for report in labeled_row_reports if report["score"] > 0.0)
            problem_full_score_submission_rows = sum(
                1
                for report in labeled_row_reports
                if report["max_score"] > 0 and report["score"] >= report["max_score"]
            )
            for row_report in labeled_row_reports:
                problem_sample_passed += row_report["sample_tests_passed"]
                problem_sample_tests += row_report["sample_tests_total"]
                problem_secret_passed += row_report["secret_tests_passed"]
                problem_secret_tests += row_report["secret_tests_total"]
                problem_compile_successes += row_report["compile_successes"]
                problem_compile_attempts += row_report["compile_attempts"]

            for subtask_report in subtasks.values():
                correct = (
                    subtask_report["score"] >= subtask_report["max_score"]
                    if subtask_report["max_score"] > 0
                    else False
                )
                subtask_report["correct"] = correct
                problem_score += subtask_report["score"]
                problem_max_score += subtask_report["max_score"]
                correct_subtasks += int(correct)

            total_score += problem_score
            total_max_score += problem_max_score
            total_sample_passed += problem_sample_passed
            total_sample_tests += problem_sample_tests
            total_secret_passed += problem_secret_passed
            total_secret_tests += problem_secret_tests
            total_compile_successes += problem_compile_successes
            total_compile_attempts += problem_compile_attempts
            total_submission_rows += problem_submission_rows
            total_nonzero_submission_rows += problem_nonzero_submission_rows
            total_full_score_submission_rows += problem_full_score_submission_rows
            problems_fully_solved += int(problem_score >= problem_max_score and problem_max_score > 0)

            problem_report = {
                "problem_id": problem_id,
                "name": problem_name,
                "score": problem_score,
                "max_score": problem_max_score,
                "correct_subtasks": correct_subtasks,
                "num_subtasks": len(subtasks),
                "num_submission_rows": problem_submission_rows,
                "nonzero_submission_rows": problem_nonzero_submission_rows,
                "full_score_submission_rows": problem_full_score_submission_rows,
                "compile_successes": problem_compile_successes,
                "compile_attempts": problem_compile_attempts,
                "compile_success_rate": (
                    100.0 * problem_compile_successes / problem_compile_attempts if problem_compile_attempts else 0.0
                ),
                "subtasks": subtasks,
            }
            if mode == "best":
                solution_selection = self._select_minimal_solutions(problem_id, problem_name, submissions, subtasks)
                problem_report.update(solution_selection)
            if problem_sample_tests or problem_secret_tests:
                problem_report["sample_tests_passed"] = problem_sample_passed
                problem_report["sample_tests_total"] = problem_sample_tests
                problem_report["secret_tests_passed"] = problem_secret_passed
                problem_report["secret_tests_total"] = problem_secret_tests
                problem_report["sample_fully_solved"] = (
                    problem_sample_tests > 0 and problem_sample_passed >= problem_sample_tests
                )
                problems_sample_fully_solved += int(problem_report["sample_fully_solved"])
            per_problem_report.append(problem_report)

        return {
            "total_score": total_score,
            "total_max_score": total_max_score,
            "problems": per_problem_report,
            "num_problems": len(per_problem_report),
            "num_subtasks": sum(problem["num_subtasks"] for problem in per_problem_report),
            "num_submission_rows": total_submission_rows,
            "nonzero_submission_rows": total_nonzero_submission_rows,
            "full_score_submission_rows": total_full_score_submission_rows,
            "compile_successes": total_compile_successes,
            "compile_attempts": total_compile_attempts,
            "compile_success_rate": (100.0 * total_compile_successes / total_compile_attempts)
            if total_compile_attempts
            else 0.0,
            "problems_fully_solved": problems_fully_solved,
            "problems_sample_fully_solved": problems_sample_fully_solved,
            "problem_solve_rate": (100.0 * problems_fully_solved / len(per_problem_report))
            if per_problem_report
            else 0.0,
            "sample_tests_passed": total_sample_passed,
            "sample_tests_total": total_sample_tests,
            "secret_tests_passed": total_secret_passed,
            "secret_tests_total": total_secret_tests,
        }

    def _select_minimal_solutions(self, problem_id: str, problem_name: str, submissions: list[dict], subtasks: dict):
        """Select a minimal solution set that covers maximum achieved subtask scores."""
        ordered_subtasks = list(subtasks.keys())
        max_achieved_by_subtask = {subtask: float(report["score"]) for subtask, report in subtasks.items()}
        active_subtasks = [subtask for subtask, score in max_achieved_by_subtask.items() if score > 0.0]
        subtask_to_bit = {subtask: idx for idx, subtask in enumerate(active_subtasks)}
        full_mask = (1 << len(active_subtasks)) - 1

        best_row_by_mask = {}
        for submission in submissions:
            if not active_subtasks:
                break

            test_case_results = submission.get("test_case_results", {})
            if not isinstance(test_case_results, dict):
                continue

            achieved_subtask_scores = {}
            achieved_total_score = 0.0
            mask = 0
            for subtask in ordered_subtasks:
                subtask_result = test_case_results.get(subtask, {})
                score = 0.0
                if isinstance(subtask_result, dict):
                    try:
                        score = float(subtask_result.get("score", 0.0))
                    except Exception:
                        score = 0.0
                achieved_subtask_scores[subtask] = score
                if subtask in subtask_to_bit:
                    achieved_total_score += score
                    if score >= max_achieved_by_subtask[subtask]:
                        mask |= 1 << subtask_to_bit[subtask]

            if mask == 0:
                continue

            rs_value = submission.get("_rs")
            rs_sort = int(rs_value) if isinstance(rs_value, int) else 10**9
            row_id = submission.get("id")
            row_id_sort = row_id if isinstance(row_id, int) else 10**9
            candidate = {
                "problem_id": problem_id,
                "name": problem_name,
                "rs": rs_value,
                "id": row_id,
                "row_subtask": submission.get("subtask") if isinstance(submission.get("subtask"), str) else None,
                "achieved_total_score": round(achieved_total_score, 6),
                "achieved_subtask_scores": achieved_subtask_scores,
                "covers_subtasks": [subtask for subtask in active_subtasks if mask & (1 << subtask_to_bit[subtask])],
                "solution": submission.get("generation") if isinstance(submission.get("generation"), str) else "",
            }

            previous = best_row_by_mask.get(mask)
            candidate_key = (candidate["achieved_total_score"], -rs_sort, -row_id_sort)
            if previous is None:
                best_row_by_mask[mask] = candidate
                continue
            previous_rs_sort = int(previous["rs"]) if isinstance(previous.get("rs"), int) else 10**9
            previous_row_id_sort = previous["id"] if isinstance(previous.get("id"), int) else 10**9
            previous_key = (float(previous["achieved_total_score"]), -previous_rs_sort, -previous_row_id_sort)
            if candidate_key > previous_key:
                best_row_by_mask[mask] = candidate

        selected = []
        candidates = list(best_row_by_mask.items())
        if full_mask and candidates:
            dp = {0: (0, 0, tuple())}
            for idx, (candidate_mask, candidate) in enumerate(candidates):
                current = dict(dp)
                candidate_rs = int(candidate["rs"]) if isinstance(candidate.get("rs"), int) else 10**9
                for current_mask, (count, sum_rs, chosen_indices) in dp.items():
                    new_mask = current_mask | candidate_mask
                    state = (count + 1, sum_rs + candidate_rs, chosen_indices + (idx,))
                    previous = current.get(new_mask)
                    if previous is None or state < previous:
                        current[new_mask] = state
                dp = current
            if full_mask in dp:
                selected = [dict(candidates[idx][1]) for idx in dp[full_mask][2]]

        max_achieved_problem_score = round(sum(max_achieved_by_subtask[subtask] for subtask in active_subtasks), 6)
        return {
            "max_achieved_problem_score": max_achieved_problem_score,
            "max_achieved_subtask_scores": max_achieved_by_subtask,
            "selected_solution_count": len(selected),
            "combined_selected_score": max_achieved_problem_score,
            "selected": selected,
        }

    @staticmethod
    def _sanitize_filename_component(value):
        """Normalize path component text for safe filesystem use."""
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
        return sanitized.strip("._") or "unknown"

    @staticmethod
    def _extract_solution_code(solution_text: str) -> str:
        """Extract final code block from solution text for writing to disk."""
        matches = re.findall(r"```(?:cpp|c\+\+|cc)?\s*\n(.*?)```", solution_text or "", re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip() + "\n"
        return (solution_text or "").rstrip() + "\n"

    def _write_selected_solutions(self, report: dict):
        """Write selected representative solutions and score metadata artifacts."""
        if self._solutions_written or not self.eval_results_dir:
            return

        solutions_dir = Path(self.eval_results_dir) / "solutions"
        solutions_dir.mkdir(parents=True, exist_ok=True)

        score_report = {
            "base_path": self.eval_results_dir,
            "metric_source": f"ccc.pass@{self.max_k}",
            "problems": [],
        }

        for problem in report["problems"]:
            problem_dir = solutions_dir / self._sanitize_filename_component(problem["problem_id"])
            problem_dir.mkdir(parents=True, exist_ok=True)

            selected_entries = []
            for solution in problem.get("selected", []):
                rs_label = f"rs{solution['rs']}" if isinstance(solution.get("rs"), int) else "rs_unknown"
                row_id = solution.get("id")
                row_id_label = f"id{row_id}" if row_id is not None else "id_unknown"
                filename = f"{rs_label}_{row_id_label}.cpp"
                solution_path = problem_dir / filename
                solution_path.write_text(self._extract_solution_code(solution.get("solution", "")), encoding="utf-8")

                selected_entry = {
                    "filename": str(Path(problem["problem_id"]) / filename),
                    "rs": solution.get("rs"),
                    "id": solution.get("id"),
                    "row_subtask": solution.get("row_subtask"),
                    "achieved_total_score": solution.get("achieved_total_score"),
                    "achieved_subtask_scores": solution.get("achieved_subtask_scores"),
                    "covers_subtasks": solution.get("covers_subtasks"),
                }
                selected_entries.append(selected_entry)

            score_report["problems"].append(
                {
                    "problem_id": problem["problem_id"],
                    "name": problem["name"],
                    "max_problem_score": problem["max_score"],
                    "max_achieved_problem_score": problem.get("max_achieved_problem_score", 0.0),
                    "max_achieved_subtask_scores": problem.get("max_achieved_subtask_scores", {}),
                    "selected_solution_count": problem.get("selected_solution_count", 0),
                    "combined_selected_score": problem.get("combined_selected_score", 0.0),
                    "selected": selected_entries,
                }
            )

        score_report_path = solutions_dir / "solution_scores.json"
        with open(score_report_path, "w", encoding="utf-8") as fout:
            json.dump(score_report, fout, indent=2)

        self._solutions_written = True

    def get_metrics(self):
        """Return filtered metrics enriched with CCC-specific summary/report fields."""
        metrics_dict = super().get_metrics()
        keep_keys = [f"pass@1[avg-of-{self.max_k}]", f"pass@{self.max_k}"]
        metrics_dict = {k: v for k, v in metrics_dict.items() if k in keep_keys}

        avg_report = self._build_problem_reports(mode="avg")
        best_report = self._build_problem_reports(mode="best")
        self._write_selected_solutions(best_report)
        report_by_key = {
            f"pass@1[avg-of-{self.max_k}]": avg_report,
            f"pass@{self.max_k}": best_report,
        }

        for key, metric in metrics_dict.items():
            report = report_by_key[key]
            total_score = (
                int(report["total_score"]) if float(report["total_score"]).is_integer() else report["total_score"]
            )
            total_max_score = (
                int(report["total_max_score"])
                if float(report["total_max_score"]).is_integer()
                else report["total_max_score"]
            )
            problem_table = []
            for problem in report["problems"]:
                ordered_subtasks = list(problem["subtasks"].items())
                score_array = [
                    (
                        int(subtask_info["score"])
                        if float(subtask_info["score"]).is_integer()
                        else subtask_info["score"]
                    )
                    for _, subtask_info in ordered_subtasks
                ]
                problem_table.append(
                    {
                        "problem_id": problem["problem_id"],
                        "name": problem["name"],
                        "status": "passed"
                        if problem["score"] >= problem["max_score"] and problem["max_score"] > 0
                        else "failed",
                        "score": int(problem["score"]) if float(problem["score"]).is_integer() else problem["score"],
                        "max_score": int(problem["max_score"])
                        if float(problem["max_score"]).is_integer()
                        else problem["max_score"],
                        "compile_success_rate": problem["compile_success_rate"],
                        "score_array": score_array,
                    }
                )
            summary = {
                "total_score": total_score,
                "total_max_score": total_max_score,
                "problems_fully_solved": report["problems_fully_solved"],
                "problems_sample_fully_solved": report["problems_sample_fully_solved"],
                "problem_solve_rate": report["problem_solve_rate"],
                "num_problems": report["num_problems"],
                "num_subtasks": report["num_subtasks"],
                "num_submission_rows": report["num_submission_rows"],
                "nonzero_submission_rows": report["nonzero_submission_rows"],
                "full_score_submission_rows": report["full_score_submission_rows"],
                "compile_successes": report["compile_successes"],
                "compile_attempts": report["compile_attempts"],
                "compile_success_rate": report["compile_success_rate"],
            }
            if report["sample_tests_total"] or report["secret_tests_total"]:
                summary["sample_tests_passed"] = report["sample_tests_passed"]
                summary["sample_tests_total"] = report["sample_tests_total"]
                summary["secret_tests_passed"] = report["secret_tests_passed"]
                summary["secret_tests_total"] = report["secret_tests_total"]
            summary["problem_table"] = problem_table

            metric.clear()
            metric["summary"] = summary
            metric["total_score"] = total_score
            metric["total_max_score"] = total_max_score
            metric["problems_fully_solved"] = report["problems_fully_solved"]
            metric["problems_sample_fully_solved"] = report["problems_sample_fully_solved"]
            metric["problem_solve_rate"] = report["problem_solve_rate"]
            metric["num_problems"] = report["num_problems"]
            metric["num_subtasks"] = report["num_subtasks"]
            metric["num_submission_rows"] = report["num_submission_rows"]
            metric["nonzero_submission_rows"] = report["nonzero_submission_rows"]
            metric["full_score_submission_rows"] = report["full_score_submission_rows"]
            metric["compile_successes"] = report["compile_successes"]
            metric["compile_attempts"] = report["compile_attempts"]
            metric["compile_success_rate"] = report["compile_success_rate"]
            if report["sample_tests_total"] or report["secret_tests_total"]:
                metric["sample_tests_passed"] = report["sample_tests_passed"]
                metric["sample_tests_total"] = report["sample_tests_total"]
                metric["secret_tests_passed"] = report["secret_tests_passed"]
                metric["secret_tests_total"] = report["secret_tests_total"]
            metric["problems"] = report["problems"]
        return metrics_dict

    def evaluations_to_print(self):
        """Return metric keys to print in evaluation summaries."""
        return [f"pass@1[avg-of-{self.max_k}]", f"pass@{self.max_k}"]

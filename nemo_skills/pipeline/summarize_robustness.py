# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import glob
import json
import logging
import os
import tempfile
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import List, Optional

import numpy as np
import typer

from nemo_skills.evaluation.metrics import ComputeMetrics
from nemo_skills.evaluation.metrics.utils import read_predictions
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils import (
    check_if_mounted,
    cluster_download_dir,
    cluster_upload,
    get_cluster_config,
    get_env_variables,
    get_unmounted_path,
    resolve_mount_paths,
)
from nemo_skills.utils import get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


def get_metrics(prediction_files: List[str]) -> List[float] | List[float]:
    """Calculate accuracy and % of no-answer datapoints for each prediction file.
       If input_files contains 3 files, both returned lists will have 3 elements,
       where index i corresponds to the metrics from input_files[i].

    Args:
        input_files: List of file paths containing predictions to evaluate

    Returns:
        Tuple[List[float], List[float]]: A tuple containing:
            - per_file_metrics: List of accuracy scores for each file
            - no_answer: List of no-answer rates for each file
    """
    per_file_metrics = []
    no_answer = []
    for pred_file in prediction_files:
        metrics_calculator = ComputeMetrics(benchmark="custom", metric_type="math", max_samples=-1)
        metrics_calculator.calculator = metrics_calculator.get_metrics_calculator()

        with open(pred_file, "rt", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = read_predictions([line], idx, [f])
                metrics_calculator.calculator.update(data)
        metrics = metrics_calculator.calculator.get_metrics()
        per_file_metrics.append(metrics["pass@1"]["symbolic_correct"])
        no_answer.append(metrics["pass@1"]["no_answer"])

    return per_file_metrics, no_answer


def calculate_similarity(answer1: str | None, answer2: str | None) -> float:
    if answer1 is None and answer2 is None:
        return 0
    return 1 if answer1 == answer2 else 0


def calculate_consistency_rate(input_files: List[str]) -> float:
    """Calculate the consistency rate across multiple input files.
    Metric proposed in https://arxiv.org/abs/2403.14221

    Args:
        input_files: List of file paths containing predictions

    Returns:
        float: Average consistency rate as a percentage (0-100)

    For each datapoint, collect all predictions, and
    calculate similarity between all possible pairs of predictions.
    The consistency rate is the number of pairs of equivalent prediction pairs
    divided by the total number of prediction pairs (N choose 2).

    Example:
        If datapoint i has predictions [A, A, C] across 3 files, it will
        compare pairs (A,A), (A, C) and (A, C) and consistency rate will be 1/3 = 33.33%.

    """
    per_idx_preds = defaultdict(list)
    for inp_f in input_files:
        with open(inp_f, "rt", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = read_predictions([line], idx, [f])
                per_idx_preds[idx].append(data[0]["predicted_answer"])
    responses = per_idx_preds.values()
    total_similarity = 0
    total_combinations = 0

    for response_set in responses:
        if len(response_set) < 2:
            continue
        for answer1, answer2 in combinations(response_set, 2):
            total_similarity += calculate_similarity(answer1, answer2)
            total_combinations += 1

    if total_combinations == 0:
        return 100.0
    return round(total_similarity / total_combinations * 100, 2)


@app.command()
@typer_unpacker
def summarize_robustness(
    results_dir: str = typer.Argument(
        ...,
        help="Path to the dir with results. Needs to contain <benchmark> dirs inside. "
        "If cluster is specified, will fetch the results from there.",
    ),
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument. "
        "If not specified, will assume the results are in the local filesystem.",
    ),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    data_dir: str = typer.Option(
        None,
        help="Path to the data directory. If not specified, will use the default nemo_skills/dataset path. "
        "Can also specify through NEMO_SKILLS_DATA_DIR environment variable.",
    ),
    remote_tar_dir: str = typer.Option(None, help="Directory where remote tar files are created on clusters"),
    debug: bool = typer.Option(False, help="Print debug information"),
    mount_paths: str = typer.Option(None, help="Comma separated list of paths to mount on the remote machine"),
    save_metrics_path: Optional[str] = typer.Option(
        None,
        help="Path to save the metrics.json file. If not specified, will save to results_dir/metrics.json.",
    ),
    verbose: bool = typer.Option(True, help="Print download/upload progress"),
):
    """Summarize robustness evaluation results across multiple benchmarks and prompts.
       Calculate statistical metrics and consistency rates for model predictions.
       The results can be used to analyze model prediction stability and robustness against prompt variations.
       Works for any Math and MCQ benchmark (such as AIME, comp-math-24-25, GPQA, MMLU-Pro, etc).

    Expected Directory Structure:
        results_dir/
        ├── benchmark1/
        │   ├── prompt1/
        │   │   ├── output-rs0.jsonl
        │   │   ├── output-rs1.jsonl
        │   │   └── ...
        │   └── prompt2/
        │       └── ...
        └── benchmark2/
            └── ...

    Calculates the following both per benchmark across prompts and per prompt across random seeds:
        - Statistical metrics: min, max, average, standard deviation
        - Consistency Rate (CR): Agreement between different model runs
        - No-answer rate: Proportion of questions without answers
        - Cross-prompt standard deviation of averages

    Output:
        - Saves formatted tables with aggregated benchmark statistics in results_dir/summarize_robustness
        - Saves detailed per-prompt statistics for each benchmark in results_dir/summarize_robustness
        - Saves all metrics to a JSON file (metrics.json) in results_dir
    """

    setup_logging(disable_hydra_logs=False, log_level=logging.WARNING if not debug else logging.DEBUG)

    cluster = cluster or os.environ.get("NEMO_SKILLS_CONFIG")
    # copying results from the cluster if necessary
    upload_path = None
    if cluster is not None:
        cluster_config = get_cluster_config(cluster, config_dir)
        cluster_config = resolve_mount_paths(cluster_config, mount_paths)
        check_if_mounted(cluster_config, results_dir)
        if cluster_config.get("executor", "") == "local":
            results_dir = get_unmounted_path(cluster_config, results_dir)
        else:
            upload_path = results_dir
            temp_dir = tempfile.mkdtemp()
            print(f"Copying results from {results_dir} on cluster {cluster} to {temp_dir}")
            os.makedirs(temp_dir, exist_ok=True)
            cluster_download_dir(
                cluster_config,
                get_unmounted_path(cluster_config, results_dir),
                temp_dir,
                remote_tar_dir=get_unmounted_path(cluster_config, remote_tar_dir),
                verbose=verbose,
            )
            results_dir = Path(temp_dir) / Path(results_dir).name
        env_vars = get_env_variables(cluster_config)
        data_dir = data_dir or env_vars.get("NEMO_SKILLS_DATA_DIR") or os.environ.get("NEMO_SKILLS_DATA_DIR")
    else:
        cluster_config = None

    benchmarks_paths = [
        cand_path
        for cand_path in glob.glob(f"{results_dir}/*")
        if "summarize" not in os.path.basename(cand_path) and Path(cand_path).is_dir()
    ]

    if benchmarks_paths:
        # Ascertain that the benchmarks_paths are valid
        for benchmark_path in benchmarks_paths:
            # Valid benchmark_path should contain output*jsonl files
            if len(glob.glob(f"{benchmark_path}/**/output*jsonl", recursive=True)) == 0:
                raise ValueError(f"The benchmark directory {benchmark_path} lacks output*jsonl files.")
    else:
        print(f"No benchmarks found in {results_dir}")
        return

    metrics_to_print = {}
    print("Calculating robustness metrics for benchmarks found:", benchmarks_paths)
    for benchmark_path in sorted(benchmarks_paths):  # sorting to ensure consistent order
        benchmark = str(Path(benchmark_path).name)
        if not Path(benchmark_path).is_dir():
            continue

        metrics_to_print[benchmark] = dict()
        # calculate metrics per prompt
        all_eval_metrics = []
        for prompt_dir in sorted(glob.glob(f"{benchmark_path}/*")):
            prompt_name = str(Path(prompt_dir).name)
            input_files = glob.glob(f"{prompt_dir}/**/output-rs*.jsonl", recursive=True)
            if not input_files:
                print("No input files found for prompt", prompt_dir)
                continue
            per_file_metrics, no_answer = get_metrics(input_files)
            metrics_to_print[benchmark][prompt_name] = {
                "min": np.min(per_file_metrics),
                "max": np.max(per_file_metrics),
                "avg": np.mean(per_file_metrics),
                "std": np.std(per_file_metrics),
                "no_answer": np.mean(no_answer),
                "num_seeds": len(per_file_metrics),
            }
            all_eval_metrics.extend(per_file_metrics)
            # calculate consistency rate per prompt
            consistency_rate = calculate_consistency_rate(input_files)
            metrics_to_print[benchmark][prompt_name]["CR"] = consistency_rate

        # calculate metrics across all prompts and seeds
        metrics_to_print[benchmark]["aggregated"] = {
            "min": np.min(all_eval_metrics),
            "max": np.max(all_eval_metrics),
            "avg": np.mean(all_eval_metrics),
            "std": np.std(all_eval_metrics),
            "num_runs": len(all_eval_metrics),
        }

        input_files = glob.glob(f"{benchmark_path}/**/output-rs*.jsonl", recursive=True)
        consistency_rate = calculate_consistency_rate(input_files)
        metrics_to_print[benchmark]["aggregated"]["CR"] = consistency_rate

    # calculate the std of prompt averages
    for benchmark, metrics in metrics_to_print.items():
        prompt_avgs = [m["avg"] for k, m in metrics.items() if k != "aggregated"]
        metrics_to_print[benchmark]["aggregated"]["prompt_sensitivity"] = np.std(prompt_avgs)

    header_fields = ["min", "max", "avg", "std", "CR", "prompt_sensitivity"]
    header = f"{'dataset':<20} | "
    header += " | ".join(f"{stat}".center(7) for stat in header_fields)
    print(header)
    print("-" * len(header))
    # Print aggregated stats
    for benchmark in metrics_to_print.keys():
        bench_runs = f"{benchmark}@{metrics_to_print[benchmark]['aggregated']['num_runs']}"
        row = f"{bench_runs:<20} | " + " | ".join(
            f"{metrics_to_print[benchmark]['aggregated'][stat]:.2f}".center(7) for stat in header_fields
        )
        print(row)
    print("\n")

    # Print stats per prompt
    header_fields = ["min", "max", "avg", "std", "CR", "no_answer"]
    for benchmark, metrics in metrics_to_print.items():
        print(f" {benchmark} ".center(len(header), "-"))
        num_seeds = metrics["aggregated"]["num_runs"] // (len(metrics) - 1)  # excluding aggregated
        header = f"{'prompt@' + str(num_seeds):<20} | "
        header += " | ".join(f"{stat}".center(7) for stat in header_fields)
        print(header)
        print("-" * len(header))
        sorted_prompts = sorted(metrics.items(), key=lambda x: x[1]["avg"])
        for prompt_name, prompt_metrics in sorted_prompts:
            if prompt_name == "aggregated":
                continue
            row = f"{prompt_name:<20} | " + " | ".join(
                f"{prompt_metrics[stat]:.2f}".center(7) for stat in header_fields
            )
            print(row)
        print("\n")

    try:
        save_metrics_path = save_metrics_path or str(Path(results_dir) / "metrics.json")
        Path(save_metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_metrics_path, "wt", encoding="utf-8") as fout:
            json.dump(metrics_to_print, fout, indent=2)
        if upload_path is not None:
            cluster_upload(
                cluster_config,
                save_metrics_path,
                Path(get_unmounted_path(cluster_config, upload_path)) / "metrics.json",
                verbose=verbose,
            )
            print("Metrics are saved to", str(Path(get_unmounted_path(cluster_config, upload_path)) / "metrics.json"))
        else:
            print("Metrics are saved to", save_metrics_path)
    except PermissionError:
        print(f"Could not save metrics.json to {save_metrics_path}. Please check the permissions.")


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()

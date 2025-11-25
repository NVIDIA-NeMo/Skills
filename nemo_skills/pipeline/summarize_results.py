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

import glob
import json
import logging
import os
import re
import tempfile
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Optional

import typer

from nemo_skills.dataset.utils import ExtraDatasetType
from nemo_skills.evaluation.metrics import ComputeMetrics, default_formatting
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils import (
    check_if_mounted,
    cluster_download_dir,
    cluster_upload,
    get_cluster_config,
    get_env_variables,
    get_unmounted_path,
    parse_kwargs,
    resolve_mount_paths,
)
from nemo_skills.utils import get_logger_name, setup_logging, validate_wandb_project_name

LOG = logging.getLogger(get_logger_name(__file__))


def get_subset_name(benchmark: str, subset: str) -> str:
    """Construct a subset name based on the benchmark and subset."""
    if subset == "_all_":
        return benchmark
    return f"{benchmark}-{subset}"


def add_benchmark_groups(results, metrics_to_print, evaluations_to_print):
    # Average results for benchmarks with dot notation (e.g., ruler.niah_single_1, ruler.niah_single_2)
    benchmark_groups = defaultdict(list)
    for benchmark in results.keys():
        if "." in benchmark:
            prefix = benchmark.rsplit(".", 1)[0]
            benchmark_groups[prefix].append(benchmark)

    # Create a new ordered dictionary to ensure prefix benchmarks appear first
    new_results = OrderedDict()

    # Process each group with the same prefix and add to new dictionary first
    for prefix, benchmarks in benchmark_groups.items():
        if len(benchmarks) <= 1:  # Skip if there's only one benchmark with this prefix
            continue

        # Create a new entry for the average results
        new_results[prefix] = defaultdict(dict)

        # Use metrics_to_print and evaluations_to_print from the first benchmark in the group
        metrics_to_print[prefix] = metrics_to_print[benchmarks[0]]
        evaluations_to_print[prefix] = evaluations_to_print[benchmarks[0]]

        # Verify that all benchmarks have the same evaluation modes
        reference_benchmark = benchmarks[0]

        # Instead of relying on evaluations_to_print, get all evaluation modes directly from results
        all_eval_modes = set()
        for benchmark in benchmarks:
            all_eval_modes.update(results[benchmark].keys())

        # Average the metrics for each evaluation mode
        for eval_mode in all_eval_modes:
            # Check if this evaluation mode exists in all benchmarks
            missing_benchmarks = [b for b in benchmarks if eval_mode not in results[b]]
            if missing_benchmarks:
                raise ValueError(f"Evaluation mode '{eval_mode}' missing in benchmarks: {missing_benchmarks}")

            # Get reference metrics from first benchmark to validate others against
            reference_metrics = set(results[reference_benchmark][eval_mode].keys())

            # Verify all benchmarks have the same metrics
            for benchmark in benchmarks[1:]:
                current_metrics = set(results[benchmark][eval_mode].keys())
                if current_metrics != reference_metrics:
                    raise ValueError(
                        f"Metrics mismatch for benchmark '{benchmark}' in mode '{eval_mode}': "
                        f"Expected {reference_metrics}, got {current_metrics}"
                    )

            # Calculate averages for each metric
            for metric_key in results[reference_benchmark][eval_mode].keys():
                values = []
                for benchmark in benchmarks:
                    metric_value = results[benchmark][eval_mode][metric_key]
                    if metric_key == "num_entries":
                        continue  # Skip averaging num_entries as we'll replace it with num_benchmarks_in_group
                    if not isinstance(metric_value, (int, float)):
                        raise TypeError(
                            f"Cannot average non-numeric metric: '{metric_key}' in benchmark '{benchmark}', "
                            f"evaluation mode '{eval_mode}'. Got type: {type(metric_value)}"
                        )
                    values.append(metric_value)

                if metric_key != "num_entries":
                    new_results[prefix][eval_mode][metric_key] = sum(values) / len(values)
                    # keeping the original float/int types
                    if isinstance(results[reference_benchmark][eval_mode][metric_key], int):
                        new_results[prefix][eval_mode][metric_key] = int(new_results[prefix][eval_mode][metric_key])

            # Add num_benchmarks_in_group instead of num_entries
            new_results[prefix][eval_mode]["num_benchmarks_in_group"] = len(benchmarks)

        LOG.info(f"Created averaged results for benchmark group: {prefix}")

    # Now add all the original benchmarks to the new ordered dictionary
    for benchmark, data in results.items():
        if benchmark not in new_results:  # Skip if already added as a prefix
            new_results[benchmark] = data

    # Replace the original results with our new ordered one
    results.clear()
    results.update(new_results)


@app.command()
@typer_unpacker
def summarize_results(
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
    benchmarks: Optional[str] = typer.Option(
        None,
        help="Specify benchmarks to run (comma separated). "
        "If not specified, all benchmarks in the results_dir will be used.",
    ),
    data_dir: str = typer.Option(
        None,
        help="Path to the data directory. If not specified, will use the default nemo_skills/dataset path. "
        "Can also specify through NEMO_SKILLS_DATA_DIR environment variable.",
    ),
    remote_tar_dir: str = typer.Option(None, help="Directory where remote tar files are created on clusters"),
    debug: bool = typer.Option(False, help="Print debug information"),
    mount_paths: str = typer.Option(None, help="Comma separated list of paths to mount on the remote machine"),
    max_samples: int = typer.Option(-1, help="Limit metric computation only to first `max_samples`"),
    extra_datasets: str = typer.Option(
        None,
        help="Path to a custom dataset folder that will be searched in addition to the main one. "
        "Can also specify through NEMO_SKILLS_EXTRA_DATASETS.",
    ),
    extra_datasets_type: ExtraDatasetType = typer.Option(
        "local",
        envvar="NEMO_SKILLS_EXTRA_DATASETS_TYPE",
        help="If you have extra datasets locally, set to 'local', if on cluster, set to 'cluster'."
        "Can also specify through NEMO_SKILLS_EXTRA_DATASETS_TYPE environment variable.",
    ),
    metric_type: Optional[str] = typer.Option(
        None,
        help="Specify metric type to use a specific metric calculator.",
    ),
    max_seq_len: Optional[int] = typer.Option(
        None,
        help="Specify max_seq_len for computing metrics. Will consider anything longer as incorrect.",
    ),
    save_metrics_path: Optional[str] = typer.Option(
        None,
        help="Path to save the metrics.json file. If not specified, will save to results_dir/metrics.json.",
    ),
    verbose: bool = typer.Option(True, help="Print download/upload progress"),
    wandb_name: Optional[str] = typer.Option(None, help="Name of the wandb experiment to sync these results to"),
    wandb_group: str = typer.Option(None, help="Name of the wandb group to sync results to."),
    wandb_project: str = typer.Option(
        "nemo-skills",
        help="Name of the wandb project to sync results to.",
    ),
    metrics_kwargs: str = typer.Option(
        "",
        help="Additional kwargs to pass to the metrics calculator. Values should be provided as a JSON string or as a `dict` if invoking from code.",
    ),
):
    """Summarize results of an evaluation job."""
    setup_logging(disable_hydra_logs=False, log_level=logging.WARNING if not debug else logging.DEBUG)

    metrics_kwargs = parse_kwargs(metrics_kwargs)
    if " " in str(benchmarks):
        raise ValueError("benchmarks should be separated with commas")

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

    # Check for all possible directory structures
    # 1. {results_dir}/eval-results/{benchmark}/output*jsonl
    # 2. {results_dir}/{benchmark}/output*jsonl
    # 3. {results_dir}/output*jsonl

    # List to store all valid benchmarks results paths
    benchmarks_paths = []

    # Check for Option 3 - Root directory corresponds to a benchmark
    if Path(results_dir).is_dir() and len(glob.glob(f"{results_dir}/output*jsonl")) > 0:
        benchmarks_paths = [results_dir]
    else:
        cand_results_dir = Path(results_dir) / "eval-results"
        # Check for Option 1
        if cand_results_dir.exists() and cand_results_dir.is_dir():
            results_dir = cand_results_dir
        else:
            # Assume by default it's Option 2.
            # Verify if it indeed has this structure: {results_dir}/{benchmark}/output*jsonl
            if len(glob.glob(f"{results_dir}/*/output*jsonl")) == 0:
                raise ValueError(
                    f"The results directory {results_dir} does not contain any valid eval-results or output*jsonl files."
                )

        benchmarks_paths = [
            cand_path
            for cand_path in glob.glob(f"{results_dir}/*")
            if "-logs" not in os.path.basename(cand_path) and Path(cand_path).is_dir()
        ]

    if benchmarks:
        # Filter benchmarks_paths to only include the specified benchmarks
        benchmarks_paths = [b for b in benchmarks_paths if Path(b).name in benchmarks.split(",")]

    if benchmarks_paths:
        # Ascertain that the benchmarks_paths are valid
        for benchmark_path in benchmarks_paths:
            # Valid benchmark_path should contain output*jsonl files
            if len(glob.glob(f"{benchmark_path}/output*jsonl")) == 0:
                raise ValueError(f"The benchmark directory {benchmark_path} lacks output*jsonl files.")
    else:
        print(f"No benchmarks found in {results_dir}")
        return

    # TODO: this needs some clean up and refactoring into functions

    results = defaultdict(lambda: defaultdict(dict))
    metrics_to_print = {}
    evaluations_to_print = {}
    for benchmark_path in sorted(benchmarks_paths):  # sorting to ensure consistent order
        benchmark = str(Path(benchmark_path).name)
        if not Path(benchmark_path).is_dir():
            continue

        if metric_type is not None:
            metrics_calculator = ComputeMetrics(
                benchmark,
                metric_type=metric_type,
                max_samples=max_samples,
                metrics_kwargs=metrics_kwargs,
            )
        else:
            metrics_calculator = ComputeMetrics(
                benchmark,
                data_dir=data_dir,
                cluster_config=cluster_config,
                extra_datasets=extra_datasets,
                extra_datasets_type=extra_datasets_type,
                max_samples=max_samples,
                max_seq_len=max_seq_len,
                metrics_kwargs=metrics_kwargs,
            )

        metrics = {}

        has_greedy = Path(f"{benchmark_path}/output.jsonl").exists()
        input_files = sorted(
            [
                jsonl_file
                for jsonl_file in glob.glob(f"{benchmark_path}/*.jsonl")
                if Path(jsonl_file).name != "output.jsonl" and "_chunk_" not in Path(jsonl_file).name
            ]
        )
        has_sampling = len(input_files) > 0

        if has_greedy and has_sampling:
            raise ValueError(
                f"Both output.jsonl and output-rs*.jsonl found for benchmark {benchmark}. "
                "This indicates that the evaluation was done multiple times with different sampling parameters. "
                "It's not clear how to process this! Please remove output.jsonl or output-rs*.jsonl files and rerun."
            )

        if not has_greedy and not has_sampling:
            raise ValueError(
                f"No .jsonl files found for benchmark {benchmark} in {benchmark_path} after excluding chunked files."
            )

        if has_greedy:
            input_files = [f"{benchmark_path}/output.jsonl"]

        metrics = metrics_calculator.compute_metrics(input_files=input_files)
        if len(metrics) > 1:  # has subsets
            for subset, subset_metrics in metrics.items():
                results[get_subset_name(benchmark, subset)].update(subset_metrics)
        else:
            results[benchmark].update(metrics["_all_"])
        if len(metrics) > 1:
            for subset, subset_metrics in metrics.items():
                metrics_to_print[get_subset_name(benchmark, subset)] = metrics_calculator.metrics_to_print()
                evaluations_to_print[get_subset_name(benchmark, subset)] = metrics_calculator.evaluations_to_print()
        else:
            metrics_to_print[benchmark] = metrics_calculator.metrics_to_print()
            evaluations_to_print[benchmark] = metrics_calculator.evaluations_to_print()

    # grouping benchmarks that have a "." e.g ruler.niah_single_1, ruler.niah_single_2 -> ruler
    # to report average numbers
    add_benchmark_groups(results, metrics_to_print, evaluations_to_print)
    printed_max_seq_len = False
    for benchmark, benchmark_results in results.items():
        if not benchmark_results:
            continue
        max_widths = {}
        max_widths["evaluation_mode"] = len("evaluation_mode")
        for eval_mode in evaluations_to_print[benchmark]:
            if eval_mode not in benchmark_results:
                continue
            metrics = benchmark_results[eval_mode]
            if metrics_to_print[benchmark] is None:
                metrics_to_print[benchmark] = {metric: default_formatting for metric in metrics}

            metrics_to_print[benchmark] = {
                metric: format_fn for metric, format_fn in metrics_to_print[benchmark].items() if metric in metrics
            }

            for metric_key, format_fn in metrics_to_print[benchmark].items():
                metric_value = metrics[metric_key]
                formatted_value = format_fn(metric_key, metric_value, metrics)
                if formatted_value is not None:
                    max_widths[metric_key] = max(
                        max_widths.get(metric_key, len(metric_key)),
                        len(str(formatted_value)),
                    )
            max_widths["evaluation_mode"] = max(max_widths["evaluation_mode"], len(eval_mode))

        # Filter out metrics that would return None for any evaluation mode
        valid_metrics = {}
        for metric_key, format_fn in metrics_to_print[benchmark].items():
            is_valid = True
            for eval_mode in evaluations_to_print[benchmark]:
                if eval_mode not in benchmark_results:
                    continue
                metrics = benchmark_results[eval_mode]
                if metric_key in metrics and format_fn(metric_key, metrics[metric_key], metrics) is None:
                    is_valid = False
                    break
            if is_valid:
                valid_metrics[metric_key] = format_fn
        metrics_to_print[benchmark] = valid_metrics

        total_width = sum(max_widths.values()) + (len(max_widths) - 1) * 3
        if max_seq_len is not None and not printed_max_seq_len:
            print(f" Metrics for Max Sequence Length {max_seq_len} ".center(total_width, "-"))
        printed_max_seq_len = True
        print(f" {benchmark} ".center(total_width, "-"))
        headers = ["evaluation_mode"] + list(metrics_to_print[benchmark].keys())
        print(" | ".join([f"{header:<{max_widths[header]}}" for header in headers]))

        for eval_mode in evaluations_to_print[benchmark]:
            if eval_mode not in benchmark_results:
                continue
            metrics = benchmark_results[eval_mode]
            values = [f"{eval_mode:<{max_widths['evaluation_mode']}}"]
            for metric_key, format_fn in metrics_to_print[benchmark].items():
                metric_value = metrics[metric_key]
                formatted_value = format_fn(metric_key, metric_value, metrics)
                if formatted_value is not None:
                    values.append(f"{str(formatted_value):<{max_widths[metric_key]}}")
            print(" | ".join(values))

        print("\n")

    try:
        save_metrics_path = save_metrics_path or str(Path(results_dir) / "metrics.json")
        Path(save_metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_metrics_path, "wt", encoding="utf-8") as fout:
            json.dump(results, fout, indent=2)
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

    # syncing to wandb if asked
    if wandb_name is not None:
        import wandb

        validate_wandb_project_name(
            wandb_project=wandb_project,
            wandb_name=wandb_name,
            wandb_group=wandb_group,
            wandb_id=wandb_name + ("-" + wandb_group if wandb_group else "") + "-" + wandb_project,
        )
        run = wandb.init(
            project=wandb_project,
            name=wandb_name,
            id=wandb_name + ("-" + wandb_group if wandb_group else "") + "-" + wandb_project,
            resume="allow",
            group=wandb_group,
            settings=wandb.Settings(silent=True),
        )
        plots = {}

        for benchmark, benchmark_results in results.items():
            if not benchmark_results:
                continue

            # Store @k metrics separately for plotting
            k_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

            for eval_mode, metrics in benchmark_results.items():
                # Check if this is a @k metric
                k_match = re.search(r"@(\d+)$", eval_mode)
                if k_match:
                    k = int(k_match.group(1))
                    base_name = eval_mode.rsplit("@", 1)[0]

                # Store k and corresponding values for each metric, but log everything
                for metric_key, metric_value in metrics.items():
                    if k_match and metric_key != "num_entries":
                        k_metrics[metric_key][base_name]["k"].append(k)
                        k_metrics[metric_key][base_name]["value"].append(metric_value)

                    run.log({f"{benchmark}/{eval_mode}/{metric_key}": metric_value})

            # Create combined plot per metric key (line series)
            for metric_key, eval_modes in k_metrics.items():
                metric_xs = []
                metric_ys = []
                mode_keys = []

                # Sort by k values and get all evaluation modes for this metric
                for mode_name, values in eval_modes.items():
                    k_value_pairs = sorted(zip(values["k"], values["value"]))
                    k_values, metric_values = zip(*k_value_pairs)
                    metric_xs.append(k_values)
                    metric_ys.append(metric_values)
                    mode_keys.append(mode_name)

                # a few hardcoded metrics to ignore
                to_ignore = ["no_answer", "any_correct", "both_correct"]
                if metric_key in to_ignore:
                    continue

                plot_key = f"{benchmark}/{metric_key}"
                plots[plot_key] = wandb.plot.line_series(
                    xs=metric_xs,
                    ys=metric_ys,
                    keys=mode_keys,
                    title=f"{benchmark} - {metric_key}",
                    xname="number of samples",
                )

                # Create individual plots for each evaluation mode
                for mode_name, values in eval_modes.items():
                    k_value_pairs = sorted(zip(values["k"], values["value"]))
                    k_values, metric_values = zip(*k_value_pairs)

                    plot_data = [[x, y] for x, y in zip(k_values, metric_values)]
                    table = wandb.Table(data=plot_data, columns=["k", "value"])

                    plot_key = f"{benchmark}/{metric_key}/{mode_name}"
                    plots[plot_key] = wandb.plot.line(
                        table,
                        "k",
                        "value",
                        title=f"{benchmark} - {metric_key} - {mode_name}",
                    )

        # Log all plots
        try:
            run.log({**plots})
        except ValueError as e:
            print("Couldn't upload plots to wandb due to error:", str(e))
        run.finish()
        print(
            f"Results are synced to wandb project {wandb_project} under the name {wandb_name} and group {wandb_group}"
        )


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()

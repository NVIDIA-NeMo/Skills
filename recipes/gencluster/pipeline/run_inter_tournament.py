# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
from pathlib import Path
from typing import Dict, Union

from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments
from nemo_skills.pipeline.utils.cluster import cluster_path_exists, get_cluster_config
from nemo_skills.pipeline.utils.mounts import get_unmounted_path


def tournament_schedule_file_exists(
    tournament_schedule_file: Union[str, Path], cluster: Union[str, Dict, None] = None
) -> bool:
    """
    Check if the tournament schedule file exists for the given input file.
    """
    cluster_config = get_cluster_config(cluster)
    tournament_schedule_file_path = str(Path(tournament_schedule_file).absolute())
    print(f"Checking if {tournament_schedule_file_path} exists on remote cluster")
    unmounted_path = get_unmounted_path(cluster_config, tournament_schedule_file_path)
    unmounted_path = str(unmounted_path)
    if cluster_config.get("executor") == "slurm":
        print(f"Checking if {unmounted_path} exists on remote cluster")
        return cluster_path_exists(cluster_config, unmounted_path)
    return Path(unmounted_path).exists()


def main():
    """Main function to run solution generation with benchmark and run number from command line arguments."""
    parser = argparse.ArgumentParser(description="Run solution generation for ICPC benchmark")
    parser.add_argument(
        "--cluster_folder",
        type=str,
        default="/workspace/cluster_folder/",
        help="Cluster folder to use for the tournament (default: /workspace/cluster_folder/)",
    )
    parser.add_argument(
        "--num_games_per_cluster",
        type=int,
        default=30,
        help="Number of games per cluster (default: 30)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/workspace/llmcoding/eval_dataset/ioi25/test.jsonl",
        help="Path to the dataset jsonl file (default: /workspace/llmcoding/eval_dataset/ioi25/test.jsonl)",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=10,
        help="Number of chunks to use for the tournament (default: 10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss-120b",
        help="Model to use for solution generation (default: gpt-oss-120b)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/hf_models/",
        help="Path to the model (default: /hf_models/",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="iad",
        help="Cluster to use for solution generation (default: iad)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="schedule",
        choices=["schedule", "play_tournament", "submission"],
        help="Stage to run (default: schedule)",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Filter clusters (default: False)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="ioi",
        choices=["ioi", "icpc"],
        help="Benchmark to use for the tournament (default: ioi)",
    )
    parser.add_argument(
        "--inter-strategy",
        type=str,
        default="wins",
        help="Inter-cluster strategy passed to submission_ICPC.py (default: wins)",
    )
    parser.add_argument(
        "--intra-strategy",
        type=str,
        default="longest",
        help="Intra-cluster strategy passed to submission_ICPC.py (default: longest)",
    )
    parser.add_argument(
        "--selection-strategy",
        type=str,
        default="longest",
        help="Selection strategy passed to run_tournament_all.py (default: longest)",
    )
    args = parser.parse_args()
    cluster_folder = args.cluster_folder
    num_chunks = args.num_chunks
    model = args.model
    model_path = args.model_path
    cluster = args.cluster
    dataset_path = args.dataset_path
    stage = args.stage
    filter_enabled = args.filter
    benchmark = args.benchmark
    inter_strategy = args.inter_strategy
    intra_strategy = args.intra_strategy
    selection_strategy = args.selection_strategy
    if args.num_games_per_cluster % 2 != 0:
        raise ValueError("Number of games per cluster must be even")
    num_games_per_cluster = args.num_games_per_cluster
    # Build inference and eval arguments, conditionally adding options for gpt-oss-120b
    args_str = (
        "++skip_filled=True "
        "++prompt_config=/nemo_run/code/gencluster/prompts/selector.yaml "
        "++inference.temperature=1.0 "
        "++inference.top_p=1.0 "
        "++inference.tokens_to_generate=120000 "
        "++max_concurrent_requests=1024 "
    )
    server_type = "vllm"
    server_gpus = 4 if cluster == "hsg" else 8
    server_nodes = 1
    server_args = "--async-scheduling --max-num-seqs=1024"
    dependent_jobs = 0
    if model == "gpt-oss-120b":
        args_str += "++inference.extra_body.reasoning_effort=high "
    elif model == "DeepSeek-V3.2-Speciale":
        server_type = "sglang"
        args_str += "++inference.endpoint_type=chat ++chat_template_kwargs.thinking=true "
        server_nodes = 2 if cluster in ["oci-hsg", "hsg", "cw-dfw", "dfw"] else 1
        server_args = f"--ep-size {server_gpus * server_nodes} --dp {server_gpus * server_nodes} --enable-dp-attention --mem-fraction-static=0.8"
        dependent_jobs = 2

    schedule_dir = cluster_folder + f"/schedule_{selection_strategy}/"
    play_dir = cluster_folder + f"/play_tournament_{selection_strategy}/"
    initial_cluster_folder = cluster_folder
    if filter_enabled:
        cluster_folder = cluster_folder + f"/filtered_{selection_strategy}/"
    if stage == "schedule":
        if tournament_schedule_file_exists(schedule_dir + "/tournament_schedule.jsonl", cluster):
            print("Tournament schedule file already exists continuing...")
        else:
            print("Generating tournament schedule...")
            generate_schedule_command = ""
            if filter_enabled:
                generate_schedule_command = f"python /nemo_run/code/gencluster/scripts/filter_clusters.py --input-dir {initial_cluster_folder} --output-dir {cluster_folder};"
            genrate_schedule_command = (
                generate_schedule_command
                + f"python /nemo_run/code/gencluster/scripts/run_tournament_all.py {cluster_folder} {num_games_per_cluster} --remove-empty --selection-strategy {selection_strategy} --progress --simple --dataset_path {dataset_path} --output-dir {schedule_dir}; cat {schedule_dir}/*schedule.jsonl > {schedule_dir}/tournament_schedule.jsonl"
            )

            run_cmd(
                ctx=wrap_arguments(""),
                cluster=cluster,
                command=genrate_schedule_command,
                expname=f"play_tournament_schedule_{model}",
                log_dir=str(schedule_dir + "/schedule_logs"),
                num_nodes=1,
                num_gpus=0,
                with_sandbox=True,
                get_random_port=True,
                keep_mounts_for_sandbox=True,
                exclusive=True,
                time_min="01:00:00",
            )
        stage = "play_tournament"

    if stage == "play_tournament":
        print("Playing tournament...")
        generate(
            ctx=wrap_arguments(args_str),
            cluster=cluster,
            input_file=schedule_dir + "/tournament_schedule.jsonl",
            output_dir=play_dir,
            expname=f"play_tournament_{model}",
            model=f"{model_path}/{model}",
            server_type=server_type,
            server_gpus=server_gpus,
            server_nodes=server_nodes,
            server_args=server_args,
            num_chunks=num_chunks,
            num_random_seeds=1,
            dependent_jobs=dependent_jobs,
            time_min="04:00:00",
            run_after=[f"play_tournament_schedule_{model}"],
            with_sandbox=True,
        )
        stage = "submission"

    if stage == "submission":
        print("Generating submission results...")
        genrate_submission_command = f"if [ -f {play_dir}/output-rs0.jsonl ]; then python /nemo_run/code/gencluster/scripts/compute_tournament_score.py {play_dir}/output-rs0.jsonl; python /nemo_run/code/gencluster/scripts/merge_tournament_scores.py {cluster_folder} {play_dir}/output-rs0.results.csv {initial_cluster_folder}/cluster_with_scores_{selection_strategy}_reps;"
        if benchmark == "ioi":
            genrate_submission_command += f" python /nemo_run/code/gencluster/scripts/submission_IOI.py --ioi25     --intra-strategy={intra_strategy} --inter-strategy={inter_strategy} --data-subdir={cluster_folder}/cluster_with_scores_{selection_strategy}_reps     --all    --remove-empty &> {initial_cluster_folder}/submission_report.txt;"
        elif benchmark == "icpc":
            genrate_submission_command += f" python /nemo_run/code/gencluster/scripts/submission_ICPC.py --intra-strategy={intra_strategy} --inter-strategy={inter_strategy} --input-dir {initial_cluster_folder}/cluster_with_scores_{selection_strategy}_reps &> {initial_cluster_folder}/submission_selection_{selection_strategy}_intra_{intra_strategy}_inter_{inter_strategy}_report.txt;"
        else:
            raise ValueError(f"Invalid benchmark: {benchmark}. Expected ioi or icpc")
        genrate_submission_command += (
            ' else echo "Tournament results file does not exist, please run play_tournament stage first"; exit 1; fi'
        )
        run_cmd(
            ctx=wrap_arguments(""),
            cluster=cluster,
            command=genrate_submission_command,
            expname=f"tournament_submission_{model}",
            log_dir=str(schedule_dir + "/submission_logs"),
            num_nodes=1,
            num_gpus=0,
            with_sandbox=True,
            get_random_port=True,
            keep_mounts_for_sandbox=True,
            exclusive=True,
            time_min="01:00:00",
            run_after=[f"play_tournament_{model}"],
        )


if __name__ == "__main__":
    main()

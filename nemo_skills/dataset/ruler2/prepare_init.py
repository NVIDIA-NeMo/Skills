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

from pathlib import Path

try:
    from nemo_skills.dataset.ruler2.prepare_common import build_prepare_parser, parse_known_args
except ModuleNotFoundError:
    from prepare_common import build_prepare_parser, parse_known_args

DEFAULT_SETTINGS = """
DATASET_GROUP = "long-context"
METRICS_TYPE = "{metrics_type}"
GENERATION_ARGS = (
    "++prompt_config=generic/default "
    "{eval_args} "
)
"""


def prepare_task_init_for_ns(output_folder, task):
    """Add task-specific __init__.py settings."""
    output_folder = Path(output_folder) / task
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_folder / "__init__.py", "w", encoding="utf-8") as init_file:
        if task in ["mk_niah_medium", "mk_niah_hard"]:
            metrics_type = "multichoice"
            eval_args = "++eval_type=multichoice"
        elif task in ["mv_niah_medium"]:
            metrics_type = "ruler2"
            eval_args = "++eval_type=ruler2 ++eval_config.match_type=2steps"
        elif "qa" in task:
            metrics_type = "ruler2"
            eval_args = "++eval_type=ruler2 ++eval_config.match_type=part"
        else:
            metrics_type = "ruler2"
            eval_args = "++eval_type=ruler2 ++eval_config.match_type=all"

        init_file.write(DEFAULT_SETTINGS.format(metrics_type=metrics_type, eval_args=eval_args))


def prepare_setup_init_for_ns(output_folder, setup, tasks):
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_folder / "__init__.py", "w", encoding="utf-8") as init_file:
        init_file.write("IS_BENCHMARK_GROUP = True\n")
        init_file.write("SCORE_MODULE = 'nemo_skills.dataset.ruler2.ruler2_score'\n")
        benchmarks = ", ".join(f"'ruler2.{setup}.{task}': {{}}" for task in tasks)
        init_file.write(f"BENCHMARKS = {{{benchmarks}}}\n")


if __name__ == "__main__":
    parser = build_prepare_parser(description="Prepare RULER2 dataset init files.")
    args, unknown = parse_known_args(parser)
    _ = unknown
    output_folder = Path(__file__).parent / args.setup
    for task in args.tasks:
        prepare_task_init_for_ns(output_folder, task)
    prepare_setup_init_for_ns(output_folder, args.setup, args.tasks)
    print("RULER2 init preparation completed.")

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
    from nemo_skills.dataset.ruler.prepare_common import build_prepare_parser, parse_args_and_prepare_args
except ModuleNotFoundError:
    from prepare_common import build_prepare_parser, parse_args_and_prepare_args

DEFAULT_SETTINGS = """
DATASET_GROUP = "long-context"
METRICS_TYPE = "ruler"
GENERATION_ARGS = (
    "++prompt_config=generic/default "
    "++eval_type=ruler ++eval_config.match_type={match_type} "
{extra_generation_args})
"""

TEXT_COMPLETIONS_EXTRA_ARGS = """\
    "++inference.tokens_to_generate={tokens_to_generate} "
    "++start_assistant_response_key=generation "
    "++inference.endpoint_type=text "
"""


TOKENS_TO_GENERATE = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}
MATCH_TYPE = {"niah": "all", "vt": "all", "cwe": "all", "fwe": "all", "qa": "part"}


def prepare_task_init_for_ns(task, setup, data_format):
    """Create task-specific __init__.py with scoring settings."""
    task_dir = Path(__file__).parent / setup / task
    task_dir.mkdir(parents=True, exist_ok=True)

    with open(task_dir / "__init__.py", "w", encoding="utf-8") as init_file:
        short_name = task.split("_")[0]
        if data_format == "chat":
            extra_generation_args = ""
        else:
            extra_generation_args = TEXT_COMPLETIONS_EXTRA_ARGS.format(
                tokens_to_generate=TOKENS_TO_GENERATE[short_name]
            )

        init_file.write(
            DEFAULT_SETTINGS.format(match_type=MATCH_TYPE[short_name], extra_generation_args=extra_generation_args)
        )


def prepare_setup_init_for_ns(tasks, setup):
    """Create setup-level __init__.py that registers all generated tasks."""
    setup_dir = Path(__file__).parent / setup
    setup_dir.mkdir(parents=True, exist_ok=True)

    with open(setup_dir / "__init__.py", "w", encoding="utf-8") as init_file:
        init_file.write("IS_BENCHMARK_GROUP = True\n")
        init_file.write("SCORE_MODULE = 'nemo_skills.dataset.ruler.ruler_score'\n")
        benchmarks = ", ".join(f"'ruler.{setup}.{task}': {{}}" for task in tasks)
        init_file.write(f"BENCHMARKS = {{{benchmarks}}}\n")


def main():
    parser = build_prepare_parser(description="Prepare RULER dataset init files.")
    args, _ = parse_args_and_prepare_args(parser)

    for task in args.tasks:
        prepare_task_init_for_ns(task, args.setup, args.data_format)
    prepare_setup_init_for_ns(args.tasks, args.setup)
    print("RULER init preparation completed.")


if __name__ == "__main__":
    main()

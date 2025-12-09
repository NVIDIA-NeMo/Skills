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
import inspect
import logging
import shlex
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import typer

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.eval import eval as _eval
from nemo_skills.prompt.utils import load_config
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


@dataclass
class PromptConfig:
    prompt_config: str  # path to prompt config
    extract_regex: str = None  # optional regex to extract answer from model output


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def robust_eval(
    ctx: typer.Context,
    prompt_set_config: str = typer.Option(..., help="Yaml file containing list of prompts per benchmark"),
    output_dir: str = typer.Option(..., help="Where to store evaluation results"),
    benchmarks: str = typer.Option(
        ...,
        help="Need to be in a format <benchmark>:<number of repeats (to average scores or compute majority voting)>. "
        "Using <benchmark> or <benchmark>:0 will default to greedy decoding "
        "(can override with ++inference.temperature=X), but otherwise is equivalent to "
        "<benchmark>:1 (which defaults to T=0.7). "
        "If you want to use multiple benchmarks, separate them with comma. E.g. gsm8k:4,human-eval",
    ),
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    expname: str = typer.Option("robust_eval", help="Name of the experiment"),
    dry_run: bool = typer.Option(False, help="If True, will not run the job, but will validate all arguments."),
    reuse_code_exp: str = typer.Option(
        None,
        help="If specified, will reuse the code from this experiment. "
        "Can provide an experiment name or an experiment object if running from code.",
    ),
    reuse_code: bool = typer.Option(
        True,
        help="If True, will reuse the code from the provided experiment. "
        "If you use it from Python, by default the code will be re-used from "
        "the last submitted experiment in the current Python session, so set to False to disable "
        "(or provide reuse_code_exp to override).",
    ),
    _reuse_exp: str = typer.Option(None, help="Internal option to reuse an experiment object.", hidden=True),
    **ns_eval_kwargs,
):
    """\b
    Run evaluation on multiple prompts and benchmarks to measure LLM robustness against changes in prompt.
    robust_eval runs "ns eval" for each prompt and benchmark combination, creates folders with benchmark names containing every prompt result in a separate folder.
    Afterwards, runs summarize_robustness to aggregate the metrics across prompts for each benchmark and save in summarize_robustness folder in main output_dir.
    Usage is the same as "ns eval" with the addition of the --prompt_set_config argument, a yaml containing the list of prompts to use for each benchmark.
    \b
    Note: prompt_set_config should be a yaml file with the following structure: (example in nemo_skills/prompt/config/robustness/prompt_set_config.yaml)
    benchmark_name:
        - prompt: prompt_config_path_1
        - prompt: prompt_config_path_2
          extract_regex: regex_to_extract_answer_from_model_output
        ...
    another_benchmark_name:
        - prompt: prompt_config_path_1
        - prompt: prompt_config_path_2
        ...
    All other arguments are "ns eval" arguments.
    """

    prompt_set_config = load_config(
        prompt_set_config, config_dir=Path(__file__).parents[1].absolute() / "prompt" / "config"
    )
    benchmarks = benchmarks.split(",")
    benchmark_names = [b.split(":")[0] for b in benchmarks]
    if set(benchmark_names).difference(set(prompt_set_config.keys())):
        raise ValueError(
            f"prompt_set_config ({prompt_set_config.keys()}) must contain benchmark names ({benchmark_names})"
        )
    for arg in ctx.args:
        if "++prompt_config" in arg:
            raise ValueError("only prompt_set_config should be used to specify prompts, please unset ++prompt_config")

    dependent_tasks = []
    cluster_config = pipeline_utils.get_cluster_config(cluster, config_dir)
    with pipeline_utils.get_exp(expname, cluster_config, _reuse_exp) as exp:
        for benchmark in benchmarks:
            benchmark_name = benchmark.split(":")[0]  # Remove any :N suffix for output dir naming
            LOG.info(f"Running {len(prompt_set_config[benchmark_name])} prompts on {benchmark_name}")
            for prompt in prompt_set_config[benchmark_name]:
                LOG.info(f"Running prompt: {prompt}")
                # deepcopy ctx and ns_eval_kwargs in case smth is changes in _eval
                prompt_context = deepcopy(ctx)
                prompt = PromptConfig(**prompt)
                if prompt.extract_regex:
                    hydra_arg = f"++eval_config.extract_regex='{prompt.extract_regex}'"
                    # Quote properly for it to be correct passed to ns eval in terminal
                    shell_safe_arg = shlex.quote(shlex.quote(hydra_arg))
                    prompt_context.args.append(shell_safe_arg)

                prompt_context.args.append(f"++prompt_config={prompt.prompt_config}")

                prompt_kwargs = deepcopy(ns_eval_kwargs)
                prompt_name = Path(prompt.prompt_config).stem
                prompt_kwargs["expname"] = f"{expname}_{prompt_name}"
                _eval(
                    ctx=prompt_context,
                    output_dir=f"{output_dir}/{benchmark_name}/{prompt_name}",
                    benchmarks=benchmark,
                    cluster=cluster,
                    config_dir=config_dir,
                    dry_run=dry_run,
                    reuse_code=reuse_code,
                    **prompt_kwargs,
                )
                dependent_tasks.append(prompt_kwargs["expname"])

        sum_rob_command = f"python -m nemo_skills.pipeline.summarize_robustness {output_dir}"
        _ = pipeline_utils.add_task(
            exp,
            cmd=sum_rob_command,
            task_name=f"{expname}-sum_robustness",
            log_dir=f"{output_dir}/summarize_robustness",
            container=cluster_config["containers"]["nemo-skills"],
            cluster_config=cluster_config,
            reuse_code_exp=reuse_code_exp,
            reuse_code=reuse_code,
            run_after=dependent_tasks,
        )

        pipeline_utils.run_exp(exp, cluster_config, dry_run=dry_run)


# Copy the signature from eval and add prompt_set_config argument
original_sig = inspect.signature(_eval)
new_param = inspect.Parameter(
    "prompt_set_config",
    inspect.Parameter.KEYWORD_ONLY,
    default=typer.Option(..., help="Yaml file containing list of prompts per benchmark"),
    annotation=str,
)
new_params = list(original_sig.parameters.values()) + [new_param]
robust_eval.__signature__ = original_sig.replace(parameters=new_params)

if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name

    app()

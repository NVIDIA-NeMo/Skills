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


import typer

# isort: off
from nemo_skills.pipeline.app import app

# isort: on

# need the imports to make sure the commands are registered
from nemo_skills.pipeline.convert import convert
from nemo_skills.pipeline.eval import eval
from nemo_skills.pipeline.generate import generate
from nemo_skills.pipeline.megatron_lm.train import train_megatron_lm
from nemo_skills.pipeline.nemo_evaluator import nemo_evaluator
from nemo_skills.pipeline.nemo_rl.grpo import grpo_nemo_rl
from nemo_skills.pipeline.nemo_rl.sft import sft_nemo_rl
from nemo_skills.pipeline.prepare_data import prepare_data
from nemo_skills.pipeline.robust_eval import robust_eval
from nemo_skills.pipeline.run_cmd import run_cmd
from nemo_skills.pipeline.setup import setup
from nemo_skills.pipeline.start_server import start_server
from nemo_skills.pipeline.summarize_results import summarize_results
from nemo_skills.pipeline.summarize_robustness import summarize_robustness
from nemo_skills.pipeline.verl.ppo import ppo_verl

typer.main.get_command_name = lambda name: name


def wrap_arguments(arguments: str):
    """Returns a mock context object to allow using the cli entrypoints as functions."""

    class MockContext:
        def __init__(self, args):
            self.args = args
            self.obj = None

    # Keep simple whitespace splitting semantics, but drop empty tokens
    # so wrap_arguments("") does not produce a spurious empty argument.
    return MockContext(args=[arg for arg in arguments.split(" ") if arg])


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    app()

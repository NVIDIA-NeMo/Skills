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

import hydra

from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig
from nemo_skills.utils import get_help_message

SOLUTION_HEADER = "## Solution"
SELF_EVAL_HEADER = "## Self Evaluation"


def strip_self_eval(text: str) -> str:
    """Extract the solution portion, stripping thinking tokens and self-evaluation."""
    if not text:
        return text
    after_solution = text.split(SOLUTION_HEADER)[-1].rstrip()
    return after_solution.split(SELF_EVAL_HEADER)[0].rstrip()


class ProofbenchGenerationTask(GenerationTask):
    async def postprocess_single_output(self, output, original_data_point):
        generation = output.get("generation", "")
        if generation:
            output["generation"] = strip_self_eval(generation)
        await super().postprocess_single_output(output, original_data_point)


GENERATION_TASK_CLASS = ProofbenchGenerationTask


@hydra.main(version_base=None, config_name="base_generation_config")
def generate(cfg: GenerationTaskConfig):
    cfg = GenerationTaskConfig(_init_nested=True, **cfg)
    task = ProofbenchGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(GenerationTaskConfig)

if __name__ == "__main__":
    generate()

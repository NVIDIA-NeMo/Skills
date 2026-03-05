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

# VLMEvalKit integration module.
# Benchmarks are referenced as eval_kit.<VLMEvalKit_dataset_name>, e.g. eval_kit.MMBench_DEV_EN
# The sub-benchmark name after eval_kit. is dynamically resolved and passed to VLMEvalKit.

GENERATION_MODULE = "nemo_skills.inference.eval.eval_kit"
METRICS_TYPE = "eval_kit"
GENERATION_ARGS = ""
NUM_SAMPLES = 0  # VLMEvalKit inference is deterministic; no random seeds

# No JSONL input file; VLMEvalKit manages its own data via build_dataset()
SKIP_INPUT_FILE = True

# Note: SELF_CONTAINED_TASK is NOT set here because it depends on model_type.
# For mcore mode (Megatron in-process), the pipeline sets self_contained_task=True
# at runtime based on ++model_type=mcore in extra_arguments.
# For vllm mode, the standard NeMo Skills server/client flow is used.


def get_extra_generation_args(benchmark):
    """Return extra generation args for the given benchmark name.

    Extracts the VLMEvalKit dataset name from the dotted benchmark name
    (e.g. eval_kit.MMBench_DEV_EN -> ++vlm_dataset=MMBench_DEV_EN).
    """
    if "." not in benchmark:
        raise ValueError(
            f"eval_kit benchmark must be in 'eval_kit.<dataset_name>' format, got '{benchmark}'. "
            f"Example: eval_kit.MMBench_DEV_EN, eval_kit.LibriSpeech_test_clean"
        )
    sub = benchmark.split(".", 1)[1]
    return f" ++vlm_dataset={sub} "

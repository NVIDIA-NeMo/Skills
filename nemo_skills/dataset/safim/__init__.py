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

# settings that define how evaluation should be done by default (all can be changed from cmdline)
DATASET_GROUP = "code"
METRICS_TYPE = "infilling"
# Pipeline default input: nemo_skills/dataset/safim/{EVAL_SPLIT}.jsonl (see prepare.py).
# Use split=api|block|control when launching eval for each subset.
EVAL_SPLIT = "api"
# ``eval_config.subset`` must match the HuggingFace subset passed to ``safim.evaluate``.
GENERATION_ARGS = f"++prompt_config=generic/fim ++eval_type=safim ++eval_config.subset={EVAL_SPLIT}"
REQUIRES_SANDBOX = True
# With ``eval_config.postprocess=advanced``, truncation runs inside installed ``safim`` (``post_process=True``).
# For ``block``, safim needs a combined grammar .so: default ``/opt/safim-tree-sitter/safim-languages.so``
# (set as ``SAFIM_TREE_SITTER_SO`` in execeval Docker) or ``++eval_config.tree_sitter_so=...`` for custom images.
# Mounts must be shared so the sandbox can read the JSONL path from the main job.
KEEP_MOUNTS_FOR_SANDBOX = True
# Slurm+Pyxis: default sandbox srun flags so ExecEval can call prlimit(RLIMIT_RSS). If your site uses
# different Pyxis syntax, set ``sandbox_extra_srun_args`` in cluster YAML (then this list is ignored).
SANDBOX_EXTRA_SRUN_ARGS = [
    "--container-options=--cap-add=SYS_RESOURCE",
]
# Local Docker: pipeline sets NEMO_SKILLS_SANDBOX_CAP_SYS_RESOURCE=1 before starting the sandbox
# (Docker --cap-add=SYS_RESOURCE). For a manually started sandbox, run:
#   NEMO_SKILLS_SANDBOX_CAP_SYS_RESOURCE=1 ./nemo_skills/code_execution/local_sandbox/start_local_sandbox.sh
# Opt out (if your site forbids the capability): export NEMO_SKILLS_SANDBOX_CAP_SYS_RESOURCE=0 before launch.

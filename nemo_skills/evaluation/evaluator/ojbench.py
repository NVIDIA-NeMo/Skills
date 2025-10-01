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


import asyncio
import json
import logging
import shlex
import textwrap
from contextlib import asynccontextmanager
from dataclasses import field
from pathlib import Path

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.evaluation.evaluator.code import preprocess_code
from nemo_skills.utils import get_logger_name, nested_dataclass, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class OJBenchConfig:
    sandbox: dict = field(default_factory=lambda: {"sandbox_type": "local"})
    timeout: int = 6


@asynccontextmanager
async def sandbox_context(config: dict):
    sandbox = get_sandbox(**config)
    try:
        yield sandbox
    finally:
        LOG.info("Closing sandbox...")
        await sandbox.close()


async def install_packages(eval_config: OJBenchConfig) -> bool:
    """Helper to install packages inside the sandbox."""

    async with sandbox_context(eval_config.sandbox) as sandbox:
        LOG.info("Installing required packages for ojbench evaluation...")

        clone_cmd = "git clone https://github.com/He-Ren/OJBench.git"
        result, _ = await sandbox.execute_code(clone_cmd, language="shell", timeout=300)
        if result["process_status"] != "completed":
            stderr = result.get("stderr", "Unknown error")
            raise RuntimeError(f"Failed to clone OJBench repo: {stderr}")

        install_cmd = "pip install -e OJBench"
        result, _ = await sandbox.execute_code(install_cmd, language="shell", timeout=300)
        if result["process_status"] != "completed":
            stderr = result.get("stderr", "Unknown error")
            raise RuntimeError(f"Failed to install ojbench. Stderr: {stderr}")

        LOG.info("Successfully installed ojbench.")


async def eval_ojbench_async(cfg):
    eval_config = OJBenchConfig(**cfg.eval_config)
    problem_dirs = [
        Path(cfg.data_dir, "ojbench/OJBench_testdata/NOI"),
        Path(cfg.data_dir, "ojbench/OJBench_testdata/ICPC"),
    ]

    await install_packages(eval_config)

    async with sandbox_context(eval_config.sandbox) as sandbox:
        for jsonl_file_str in unroll_files(cfg.input_files):
            jsonl_file = Path(jsonl_file_str)
            with open(jsonl_file, encoding="utf-8") as f_in:
                samples = []
                for line in f_in:
                    sample = json.loads(line)
                    sample = preprocess_code(sample, sample["language"], strip_whitespace=True)
                    sample["prompt"] = sample.pop("question")
                    sample["content"] = f"```{sample['language']}\n{sample['completion']}\n```"
                    sample.pop("completion")
                    samples.append(sample)

            input_filename = jsonl_file.name.replace("output-", "eval-input-", 1)
            eval_input_file = jsonl_file.with_name(input_filename)
            results_filename = jsonl_file.name.replace("output-", "eval-results-", 1)
            eval_results_file = jsonl_file.with_name(results_filename)

            with open(eval_input_file, "w", encoding="utf-8") as f_out:
                f_out.writelines(json.dumps(sample) + "\n" for sample in samples)

            eval_code = textwrap.dedent(f"""
                import ojbench
                ojbench.init(problem_dirs={repr([str(p) for p in problem_dirs])})
                ojbench.judge_jsonl(
                    input_path={repr(str(eval_input_file))},
                    output_path={repr(str(eval_results_file))},
                    num_workers=16
                )
            """)

            cmd = f'env -i PATH="/usr/local/bin:/usr/bin:/bin" python3 -c {shlex.quote(eval_code)}'
            output, _ = await sandbox.execute_code(
                cmd,
                language="shell",
                timeout=eval_config.timeout * len(samples) + 60,
                max_output_characters=100_000,
            )

            if output.get("process_status") != "completed":
                raise RuntimeError(f"Evaluation failed for {jsonl_file}. Stderr: {output.get('stderr')}")

            with open(eval_results_file, "rt", encoding="utf-8") as fin:
                results = [json.loads(line) for line in fin]

            if len(results) != len(samples):
                LOG.error(f"Result count mismatch for {jsonl_file}: {len(results)} results vs {len(samples)} samples")
                continue

            for sample, result in zip(samples, results, strict=True):
                sample["verdict"] = result["verdict"]
                sample["is_passed"] = result["is_passed"]

            with open(jsonl_file, "w", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")


def eval_ojbench(cfg):
    """Synchronous wrapper to run the async evaluation."""
    asyncio.run(eval_ojbench_async(cfg))

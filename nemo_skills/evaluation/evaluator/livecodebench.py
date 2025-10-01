import asyncio
import json
import logging
import shlex
import shutil
import textwrap
from contextlib import asynccontextmanager
from dataclasses import field

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.evaluation.evaluator.code import preprocess_code
from nemo_skills.utils import get_logger_name, nested_dataclass, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))

LIVECODEBENCH_PYTHON_GIT_URL = "git+https://github.com/wasiahmad/livecodebench.git@livecodebench"
LIVECODEBENCH_PYPY3_GIT_URL = "git+https://github.com/wasiahmad/livecodebench.git"


@nested_dataclass(kw_only=True)
class LiveCodeBenchEvaluatorConfig:
    sandbox: dict = field(default_factory=lambda: {"sandbox_type": "local"})
    language: str = "python"  # "cpp" is another option now
    test_file: str = None
    interpreter: str = "python"  # use either "python" or pypy3
    timeout: int = 6
    num_processes: int = 12


@asynccontextmanager
async def sandbox_context(config: dict):
    sandbox = get_sandbox(**config)
    try:
        yield sandbox
    finally:
        LOG.info("Closing sandbox...")
        await sandbox.close()


async def install_packages(eval_config: LiveCodeBenchEvaluatorConfig) -> bool:
    """
    Installs required packages in a temporary sandbox.
    Returns True on success, False on failure.
    """
    async with sandbox_context(eval_config.sandbox) as sandbox:
        LOG.info(f"Installing livecodebench with {eval_config.interpreter}...")
        pip_cmd = "pip" if eval_config.interpreter == "python" else "pypy3 -m pip"
        git_url = LIVECODEBENCH_PYTHON_GIT_URL if eval_config.interpreter == "python" else LIVECODEBENCH_PYPY3_GIT_URL
        cmd = f"{pip_cmd} install {git_url}"

        result, _ = await sandbox.execute_code(cmd, language="shell", timeout=300)
        if result.get("process_status") != "completed":
            LOG.warning(f"Failed to install livecodebench: {result.get('stderr', 'Unknown error')}")
            return False

        LOG.info("Successfully installed livecodebench.")
        return True


async def eval_livecodebench_async(cfg):
    eval_config = LiveCodeBenchEvaluatorConfig(_init_nested=True, **cfg.eval_config)

    if eval_config.language == "python" and eval_config.interpreter not in ["python", "pypy3"]:
        raise ValueError("Python interpreter must be 'python' or 'pypy3'.")
    if eval_config.language == "cpp" and eval_config.test_file is None:
        raise ValueError("C++ evaluation requires a test_file.")

    if not await install_packages(eval_config):
        return

    async with sandbox_context(eval_config.sandbox) as sandbox:
        for jsonl_file in unroll_files(cfg.input_files):
            LOG.info(f"Processing file: {jsonl_file}")

            with open(jsonl_file, encoding="utf-8") as f_in:
                samples = [preprocess_code(json.loads(line), eval_config.language) for line in f_in]

            versions = {s["release_version"] for s in samples}
            if len(versions) > 1:
                raise ValueError(f"All samples should have the same release version. Found: {versions}")
            release_version = versions.pop()

            for s in samples:
                s["code_list"] = [s["completion"]]

            with open(jsonl_file, "w", encoding="utf-8") as f_out:
                f_out.writelines(json.dumps(sample) + "\n" for sample in samples)

            test_file_arg = repr(eval_config.test_file) if eval_config.test_file else "None"
            eval_code = textwrap.dedent(f"""
                from livecodebench.evaluate import evaluate
                evaluate(
                    custom_output_file='{jsonl_file}',
                    release_version='release_{release_version}',
                    test_file={test_file_arg},
                    k_list=[1],
                    language='{eval_config.language}',
                    num_process_evaluate={eval_config.num_processes},
                    timeout={eval_config.timeout}
                )
            """)

            cmd = f"{eval_config.interpreter} -c {shlex.quote(eval_code)}"
            output, _ = await sandbox.execute_code(
                cmd,
                language="shell",
                timeout=eval_config.timeout * len(samples) + 60,
                max_output_characters=100_000,
            )

            if output.get("process_status") != "completed":
                LOG.error(f"Evaluation failed for {jsonl_file}. Stderr: {output.get('stderr')}")
                continue

            with open(jsonl_file[:-6] + "_eval_results.json", "rt", encoding="utf-8") as fin:
                eval_grades = json.load(fin)

            with open(jsonl_file, "wt", encoding="utf-8") as f_out:
                for s in samples:
                    s["graded_list"] = eval_grades["eval"][s["task_id"]]["graded_list"]
                    f_out.write(json.dumps(s) + "\n")

            shutil.move(jsonl_file[:-6] + "_eval_results.json", jsonl_file[:-6] + "_eval_results-saved.json")
            LOG.info(f"Finished processing {jsonl_file}, results saved.")


def eval_livecodebench(cfg):
    """Synchronous wrapper to run the async evaluation."""
    asyncio.run(eval_livecodebench_async(cfg))

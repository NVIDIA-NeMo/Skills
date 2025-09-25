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
import re
import shutil
import subprocess
import sys
import textwrap
from argparse import Namespace
from dataclasses import field
from pathlib import Path

from omegaconf import OmegaConf

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.utils import get_logger_name, nested_dataclass, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))

BIGCODEBENCH_REQUIREMENTS_URL = (
    "https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt"
)


def preprocess_code(generation_dict: dict, language="python", strip_whitespace=True):
    completion = generation_dict["generation"]
    if strip_whitespace:
        completion = completion.strip()
    completion = completion.replace("\r", "")

    ##### To handle code generation by reasoning models
    # check for <think> and </think> tags
    if "<think>" in completion:
        if "</think>" in completion:
            # thinking trace completed, solution in after the trace
            match = re.search(r"</think>\s*(.*)", completion, re.DOTALL)
            completion = match.group(1).strip() if match else None
        else:
            completion = None

    if completion is None:
        generation_dict["completion"] = ""  # no valid solution generated
        return generation_dict
    #####

    start_with_lang_tag = f"```{language}"
    generic_start_end_tag = "```"

    if start_with_lang_tag in completion:
        def_line = completion.index(start_with_lang_tag) + len(start_with_lang_tag)
        completion = completion[def_line:]
        if strip_whitespace:
            completion = completion.strip()
        try:
            next_line = completion.index(generic_start_end_tag)
            completion = completion[:next_line]
            if strip_whitespace:
                completion = completion.strip()
        except Exception:
            print(completion)
            print("================\n")

    elif generic_start_end_tag in completion:
        def_line = completion.index(generic_start_end_tag) + len(generic_start_end_tag)
        completion = completion[def_line:]
        if strip_whitespace:
            completion = completion.strip()
        try:
            next_line = completion.index(generic_start_end_tag)
            completion = completion[:next_line]
            if strip_whitespace:
                completion = completion.strip()
        except Exception:
            print(completion)
            print("================\n")

    if completion.startswith(" ") and strip_whitespace:
        completion = completion.strip()

    generation_dict["completion"] = completion
    return generation_dict


def install_from_git(git_url):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", git_url])
        print("Package installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")


# TODO: use sandbox
@nested_dataclass(kw_only=True)
class LiveCodeBenchEvaluatorConfig:
    language: str = "python"  # "cpp" is another option now
    test_file: str = None


def eval_livecodebench(cfg):
    try:
        from livecodebench.evaluate import evaluate
    except ImportError:
        LOG.info("Package 'livecodebench' not found. Attempting to install...")
        # install_from_git("git+https://github.com/wasiahmad/livecodebench.git")
        install_from_git("git+https://github.com/wasiahmad/livecodebench.git@f285640c20aaf18df1ee5917621a596af4630b5e")
        try:
            from livecodebench.evaluate import evaluate
        except ImportError:
            LOG.info("Failed to install 'livecodebench'. Please install it manually.")
            raise

    eval_config = LiveCodeBenchEvaluatorConfig(_init_nested=True, **cfg.eval_config)
    assert eval_config.language in ["python", "cpp"]
    if eval_config.language == "cpp":
        assert eval_config.test_file is not None

    release_version = None
    for jsonl_file in unroll_files(cfg.input_files):
        with open(jsonl_file) as f:
            samples = [preprocess_code(json.loads(line), eval_config.language) for line in f]
            for sample in samples:
                sample["question_id"] = sample["task_id"]
                sample["code_list"] = [sample["completion"]]
                if release_version is None:
                    release_version = sample["release_version"]
                if release_version != sample["release_version"]:
                    raise ValueError(
                        f"All samples should have the same release version, "
                        f"but got {release_version} and {sample['release_version']}"
                    )

        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        # https://github.com/wasiahmad/livecodebench/blob/main/livecodebench/evaluate.py#L10
        evaluate(
            custom_output_file=jsonl_file,
            release_version=f"release_{release_version}",
            k_list=[1],
            language=eval_config.language,
            test_file=None if eval_config.language == "python" else eval_config.test_file,
            num_process_evaluate=12,
            timeout=6 if eval_config.language == "python" else 30,
        )

        with open(jsonl_file[:-6] + "_eval_results.json", "rt", encoding="utf-8") as fin:
            eval_grades = json.load(fin)
        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                sample["graded_list"] = eval_grades["eval"][sample["task_id"]]["graded_list"]
                f.write(json.dumps(sample) + "\n")

        # moving eval file to ensure metrics are recomputed
        shutil.move(jsonl_file[:-6] + "_eval_results.json", jsonl_file[:-6] + "_eval_results-saved.json")


def eval_livecodebench_pro(cfg):
    for jsonl_file in unroll_files(cfg.input_files):
        with open(jsonl_file) as f:
            samples = [preprocess_code(json.loads(line), "python") for line in f]
            for sample in samples:
                sample["problem_id"] = sample.pop("task_id")
                sample["text_response"] = sample.pop("completion")
                sample["response_meta"] = None

        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")


def eval_evalplus(cfg):
    # TODO: need to move it to a separate docker (either our sandbox or separate srun)
    from evalplus.evaluate import evaluate

    # processing each generation separately (TODO: evalplus can do it together, but need to figure out the format)
    for jsonl_file in unroll_files(cfg.input_files):
        with open(jsonl_file) as f:
            samples = [preprocess_code(json.loads(line)) for line in f]
        # all changes will be done with a new key "completion", so it's ok to write to the same file
        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        eval_config = {
            "samples": jsonl_file,
            "base_only": False,
            "parallel": None,
            "i_just_wanna_run": False,
            "test_details": False,
            "min_time_limit": 1,
            "gt_time_limit_factor": 4.0,
            "mini": False,
            "noextreme": False,
            "version": "default",
        }
        eval_config.update(OmegaConf.to_container(cfg.eval_config))
        evaluate(Namespace(**eval_config))
        with open(jsonl_file[:-6] + "_eval_results.json", "rt", encoding="utf-8") as fin:
            evalplus_grades = json.load(fin)
        # adding is_correct key to allow compute_metrics to work
        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                sample["is_correct"] = evalplus_grades["eval"][sample["task_id"]][0]["base_status"] == "pass"
                sample["is_correct-plus"] = (
                    sample["is_correct"] and evalplus_grades["eval"][sample["task_id"]][0]["plus_status"] == "pass"
                )
                f.write(json.dumps(sample) + "\n")

        # moving eval file as otherwise evalplus does not want to recompute metrics if it's present..
        shutil.move(jsonl_file[:-6] + "_eval_results.json", jsonl_file[:-6] + "_eval_results-saved.json")


def install_requirements(url):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", url])
        print("Requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")


def eval_livebench_coding(cfg):
    try:
        from livecodebench.evaluate import evaluate
    except ImportError:
        LOG.info("Package 'livecodebench' not found. Attempting to install...")
        install_from_git("git+https://github.com/wasiahmad/livecodebench.git@livebench")
        try:
            from livecodebench.evaluate import evaluate
        except ImportError:
            LOG.info("Failed to install 'livecodebench'. Please install it manually.")
            raise

    for jsonl_file in unroll_files(cfg.input_files):
        samples = []
        with open(jsonl_file) as f:
            for line in f:
                sample = json.loads(line)
                if sample["task"] == "coding_completion":
                    assert len(sample["partial_solution"]) > 0
                    sample = preprocess_code(sample, strip_whitespace=False)
                    sample["completion"] = sample["completion"].replace("\t", "    ")
                    full_solution = sample["partial_solution"] + "\n" + sample["completion"]
                    sample["code_list"] = [full_solution]
                else:
                    sample = preprocess_code(sample, strip_whitespace=True)
                    sample["code_list"] = [sample["completion"]]

                samples.append(sample)

        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        evaluate(
            custom_output_file=jsonl_file,
            k_list=[1],
            num_process_evaluate=12,
            timeout=6,
        )

        with open(jsonl_file[:-6] + "_eval_results.json", "rt", encoding="utf-8") as fin:
            eval_grades = json.load(fin)
        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                sample["graded_list"] = eval_grades["eval"][sample["question_id"]]["graded_list"]
                f.write(json.dumps(sample) + "\n")

        # moving eval file to ensure metrics are recomputed
        shutil.move(jsonl_file[:-6] + "_eval_results.json", jsonl_file[:-6] + "_eval_results-saved.json")


def install_or_upgrade_package(package_name):
    try:
        # Run the pip command to install or upgrade the package
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
        print(f"{package_name} has been successfully installed or upgraded.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing/upgrading {package_name}: {e}")


def eval_bigcodebench(cfg):
    try:
        from bigcodebench.evaluate import evaluate
    except ImportError:
        LOG.info("Package 'bigcodebench' not found. Attempting to install...")
        install_requirements(BIGCODEBENCH_REQUIREMENTS_URL)
        install_or_upgrade_package("bigcodebench")
        install_or_upgrade_package("numpy==1.26.4")  # <= needed to work with scikit-learn version 1.3.1
        try:
            from bigcodebench.evaluate import evaluate
        except ImportError:
            LOG.error("Failed to install 'bigcodebench'. Please install it manually.")
            raise

    data_split = None
    for jsonl_file in unroll_files(cfg.input_files):
        samples = []
        with open(jsonl_file) as f:
            for line in f:
                generation_dict = preprocess_code(json.loads(line))
                generation_dict["solution"] = generation_dict.pop("completion")
                samples.append(generation_dict)
        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
                if data_split is None:
                    data_split = sample["split"]
                elif data_split != sample["split"]:
                    raise ValueError(
                        f"All samples should have the same split, but got {data_split} and {sample['split']}"
                    )

        # https://github.com/bigcode-project/bigcodebench/blob/main/bigcodebench/evaluate.py#L117
        # if the input filename is "output.jsonl"
        # then there will be two output files (generated) after evaluation:
        # "output_eval_results-saved.json"
        # "output_pass_at_k.json"
        evaluate(
            "instruct",
            data_split,  # full, hard
            samples=jsonl_file,
            execution="local",
            pass_k="1",
            calibrated=True,
            save_pass_rate=True,  # saves pass_at_k results in file: "output_pass_at_k.json"
        )

        with open(jsonl_file[:-6] + "_eval_results.json", "rt", encoding="utf-8") as fin:
            eval_grades = json.load(fin)
        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                sample["status"] = eval_grades["eval"][sample["task_id"]][0]["status"]
                f.write(json.dumps(sample) + "\n")

        # moving eval file to ensure metrics are recomputed
        shutil.move(jsonl_file[:-6] + "_eval_results.json", jsonl_file[:-6] + "_eval_results-saved.json")


@nested_dataclass(kw_only=True)
class OJBenchConfig:
    sandbox: dict = field(default_factory=lambda: {"sandbox_type": "local"})


def eval_ojbench(cfg):
    eval_config = OJBenchConfig(**cfg.eval_config)
    LOG.info("Installing required packages for ojbench evaluation...")

    async def install_packages():
        sandbox = get_sandbox(**eval_config.sandbox)
        cmd = "pip install git+https://github.com/He-Ren/OJBench/tree/main"
        result, _ = await sandbox.execute_code(cmd, language="shell", timeout=300)
        if result["process_status"] != "completed":
            LOG.warning(f"Failed to install ojbench: {result.get('stderr', 'Unknown error')}")
        else:
            LOG.info("Successfully installed ojbench")

        await sandbox.close()

    asyncio.run(install_packages())

    for jsonl_file in unroll_files(cfg.input_files):
        # Read and preprocess all samples in one go
        with open(jsonl_file, encoding="utf-8") as f:
            samples = []
            for line in f:
                sample = json.loads(line)
                sample = preprocess_code(sample, sample["language"], strip_whitespace=True)
                sample["prompt"] = sample.pop("question")
                sample["content"] = f"```{sample['language']}\n{sample['completion']}\n```"
                sample.pop("completion")
                samples.append(sample)

        # Overwrite the file with preprocessed samples
        with open(jsonl_file, "wt", encoding="utf-8") as f:
            f.writelines(json.dumps(sample) + "\n" for sample in samples)

        # Judge all samples at once
        # results = ojbench.judge_jsonl_data(samples, num_workers=4)

        problem_dirs = [
            Path(cfg.data_dir, "ojbench/NOI"),
            Path(cfg.data_dir, "ojbench/ICPC"),
        ]
        eval_results_path = jsonl_file[:-6] + "_eval_results.jsonl"
        eval_code = textwrap.dedent(f"""
            import ojbench

            ojbench.init(problem_dirs={problem_dirs})

            ojbench.judge_jsonl(
                input_path={jsonl_file},
                output_path={eval_results_path},
                num_workers=8
            )
        """)

        sandbox = get_sandbox(**eval_config.sandbox)
        output, _ = await sandbox.execute_code(
            eval_code,
            timeout=eval_config.timeout * len(samples) + 60,
            max_output_characters=100_000,
        )

        if output.get("process_status") != "completed":
            LOG.error(f"Evaluation failed for {jsonl_file}. Stderr: {output.get('stderr')}")
            continue

        with open(eval_results_path, "rt", encoding="utf-8") as fin:
            results = []
            for line in fin:
                results.append(json.loads(line))

        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample, result in zip(samples, results):
                sample["verdict"] = result["verdict"]
                sample["is_passed"] = result["is_passed"]
                f.write(json.dumps(sample) + "\n")

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
from argparse import Namespace
from dataclasses import field

from omegaconf import OmegaConf

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.evaluation.evaluator import BaseEvaluator
from nemo_skills.evaluation.evaluator.base import BaseEvaluatorConfig
from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))

BIGCODEBENCH_REQUIREMENTS_URL = (
    "https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt"
)


@nested_dataclass(kw_only=True)
class CodeExecEvaluatorConfig:
    input_file: str
    sandbox: dict
    language: str = "python3"
    timeout: int = 10
    max_output_characters: int = 1000


class CodeExecEvaluator(BaseEvaluator):
    def __init__(self, config: dict, num_parallel_requests: int = 12):
        super().__init__(config, num_parallel_requests)
        self.eval_config = CodeExecEvaluatorConfig(**self.config)
        LOG.info(
            f"Evaluation config: language={self.eval_config.language}, timeout={self.eval_config.timeout}, "
            f"max_output_characters={self.eval_config.max_output_characters}"
        )
        self.sandbox = get_sandbox(self.eval_config.sandbox)
        self.sandbox.wait_for_sandbox(50)

    async def eval_single(self, data: dict):
        """Evaluate single code during generation."""
        task_id = data.get("task_id", "unknown")
        LOG.debug(f"Evaluating single sample: task_id={task_id}")

        output_dict = {
            "process_status": [],
            "correct_tests": [],
            "average_test_score": 0.0,
            "stdouts": [],
            "stderrs": [],
        }

        num_test_cases = len(data.get("test_cases", []))
        LOG.debug(f"Running {num_test_cases} test cases for task_id={task_id}")

        for test_case in data["test_cases"]:
            output, _ = await self.sandbox.execute_code(
                generated_code=data["code"],
                std_input=test_case["input"],
                language=self.eval_config.language,
                timeout=self.eval_config.timeout,
                max_output_characters=self.eval_config.max_output_characters,
            )
            output_dict["process_status"].append(output["process_status"])
            output_dict["stdouts"].append(output["stdout"])
            output_dict["stderrs"].append(output["stderr"])
            output_dict["correct_tests"].append(output["stdout"].strip() == test_case["output"].strip())

        output_dict["average_test_score"] = (
            0.0
            if len(output_dict["correct_tests"]) == 0
            else (sum(output_dict["correct_tests"]) / len(output_dict["correct_tests"]))
        )

        return {"code_execution": output_dict}

    async def eval_full(self):  # type: ignore[override]
        jsonl_file = self.eval_config.input_file
        LOG.info(f"Starting full evaluation on file: {jsonl_file}")

        with open(jsonl_file, "r", encoding="utf-8") as f:
            all_samples = [json.loads(line) for line in f]

        num_samples = len(all_samples)
        LOG.info(f"Loaded {num_samples} samples for evaluation")

        tasks = [self.eval_single(s) for s in all_samples]
        outputs = await asyncio.gather(*tasks)

        for s, o in zip(all_samples, outputs):
            s["code_execution"] = o["code_execution"]

        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in all_samples:
                f.write(json.dumps(sample) + "\n")

        LOG.info("Full evaluation completed successfully")


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


@nested_dataclass(kw_only=True)
class EvalPlusEvaluatorConfig(BaseEvaluatorConfig):
    # evalplus specific configurations
    evalplus: dict = field(default_factory=dict)


def eval_evalplus(cfg):
    cfg = EvalPlusEvaluatorConfig(**cfg)
    # TODO: need to move it to a separate docker (either our sandbox or separate srun)
    from evalplus.evaluate import evaluate

    jsonl_file = cfg.input_file
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
    eval_config.update(OmegaConf.to_container(cfg.evalplus))
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


def eval_livecodebench_pro(cfg):
    cfg = BaseEvaluatorConfig(**cfg)
    jsonl_file = cfg.input_file
    with open(jsonl_file) as f:
        samples = [preprocess_code(json.loads(line), "python") for line in f]
        for sample in samples:
            sample["problem_id"] = sample.pop("task_id")
            sample["text_response"] = sample.pop("completion")
            sample["response_meta"] = None

    with open(jsonl_file, "wt", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def eval_livebench_coding(cfg):
    cfg = BaseEvaluatorConfig(**cfg)
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

    jsonl_file = cfg.input_file
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
    cfg = BaseEvaluatorConfig(**cfg)
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
    jsonl_file = cfg.input_file
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
                raise ValueError(f"All samples should have the same split, but got {data_split} and {sample['split']}")

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


def eval_human_eval_infilling(cfg):
    cfg = BaseEvaluatorConfig(**cfg)
    try:
        from human_eval_infilling.evaluate import evaluate
    except ImportError:
        LOG.info("Package 'human_eval_infilling' not found. Attempting to install...")
        install_from_git("git+https://github.com/wasiahmad/human-eval-infilling.git")
        try:
            from human_eval_infilling.evaluate import evaluate
        except ImportError:
            LOG.error("Failed to install 'human_eval_infilling'. Please install it manually.")
            raise

    def remove_overlap(preceding_text, following_text, truncate_from="following"):
        assert truncate_from in ["preceding", "following"]
        preceding_len = len(preceding_text)
        following_len = len(following_text)
        for i in range(min(preceding_len, following_len), 0, -1):
            if truncate_from == "following":
                overlap = preceding_text[-i:]
                if overlap.strip() == "" and "\n" not in overlap:
                    continue
                if following_text.startswith(overlap):
                    return following_text[i:]
            elif truncate_from == "preceding":
                overlap = following_text[:i]
                if overlap.strip() == "" and "\n" not in overlap:
                    continue
                if preceding_text.endswith(overlap):
                    return preceding_text[:-i]
        return following_text if truncate_from == "following" else preceding_text

    def postprocess_code(sample):
        sample["completion"] = remove_overlap(sample["prefix"], sample["completion"], truncate_from="following")
        sample["completion"] = remove_overlap(sample["completion"], sample["suffix"], truncate_from="preceding")
        return sample

    data_split = None
    jsonl_file = cfg.input_file
    samples = []
    with open(jsonl_file) as f:
        for line in f:
            sample = json.loads(line)
            if data_split is None:
                data_split = sample["split"]
            elif data_split != sample["split"]:
                raise ValueError(f"All samples should have the same split, but got {data_split} and {sample['split']}")

            sample = preprocess_code(sample, strip_whitespace=False)
            sample["original_completion"] = sample["completion"]
            sample = postprocess_code(sample)
            samples.append(sample)

    # all changes will be done with a new key "completion", so it's ok to write to the same file
    with open(jsonl_file, "wt", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    evaluate(data_split, jsonl_file, k=[1], n_workers=4, timeout=3.0)

    with open(jsonl_file[:-6] + "_eval_results.json", "rt", encoding="utf-8") as fin:
        eval_grades = json.load(fin)

    with open(jsonl_file, "wt", encoding="utf-8") as f_out:
        for s in samples:
            s["passed"] = eval_grades["eval"][s["task_id"]][0]["passed"]
            f_out.write(json.dumps(s) + "\n")

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
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import threading
import time

from nemo_skills.code_execution.sandbox import LocalSandbox
from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.file_utils import jdump
from nemo_skills.utils import nested_dataclass, unroll_files


@nested_dataclass(kw_only=True)
class IOIEvaluatorConfig(BaseEvaluatorConfig):
    test_file: str = "test_metadata.json"
    input_file: str | None = None
    num_workers: int = 16  # number of test workers
    test_batch_size: int = 16  # number of tests to run concurrently
    overwrite: bool = False


_precompile_loop_tls = threading.local()
worker_sandbox = None  # type: ignore
worker_loop = asyncio.new_event_loop()
asyncio.set_event_loop(worker_loop)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _sandbox_exec_sync(sandbox: LocalSandbox, cmd: str, *, language: str = "shell", timeout: int = 120):
    """Run sandbox.execute_code synchronously with a persistent event loop.

    Re-creating and immediately closing a loop for every call can leave background
    tasks (e.g., httpx/anyio socket reads) unfinished, causing "Event loop is
    closed" errors.  We therefore maintain a single loop for all such
    pre-compile operations.
    """
    loop = getattr(_precompile_loop_tls, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _precompile_loop_tls.loop = loop

    # Use the loop within this thread exclusively.
    return loop.run_until_complete(sandbox.execute_code(cmd, language=language, timeout=timeout))[0]


def wait_for_sandbox(sandbox, timeout: int = 240, poll: float = 1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = _sandbox_exec_sync(sandbox, "echo hello world", language="shell", timeout=10)
            if resp.get("stdout", "").strip() == "hello world":
                return
        except Exception:
            pass
        time.sleep(poll)
    raise RuntimeError(f"Sandbox not ready after waiting {timeout}s")


def init_worker():
    """Per-process initializer: set up an event loop for httpx/asyncio calls."""
    global worker_sandbox, worker_loop
    worker_sandbox = None  # lazily initialised when first used
    worker_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(worker_loop)


def _precompile_grader(
    problem_name: str, grader_files, compile_code: str, run_code: str, sandbox: LocalSandbox
) -> str:
    """Precompile checker/grader for a problem once and return the directory path."""
    # Ensure sandbox belongs to this thread; if not, create a local one.
    if getattr(sandbox, "_owner_tid", None) != threading.get_ident():
        sandbox = LocalSandbox()
        wait_for_sandbox(sandbox)
        sandbox._owner_tid = threading.get_ident()

    pre_dir = f"/nemo_run/ioi_pre_{problem_name}_{os.getpid()}"
    # Create directories and files locally; sandbox shares the same filesystem
    os.makedirs(os.path.join(pre_dir, "graders"), exist_ok=True)

    # Dump grader related files locally
    for filepath, content in grader_files:
        target_path = os.path.join(pre_dir, filepath)
        target_dir = os.path.dirname(target_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)

    # Write compile.sh and run.sh locally and make them executable
    compile_path = os.path.join(pre_dir, "compile.sh")
    with open(compile_path, "w", encoding="utf-8") as f:
        f.write(compile_code)
    os.chmod(compile_path, 0o755)

    run_path = os.path.join(pre_dir, "run.sh")
    with open(run_path, "w", encoding="utf-8") as f:
        f.write(run_code)
    os.chmod(run_path, 0o755)

    # Run compile.sh inside the sandbox (same filesystem)
    _sandbox_exec_sync(sandbox, f"cd {pre_dir} && ./compile.sh || true", language="shell", timeout=120)

    return pre_dir


def run_test_case(task_args: dict, worker_id: int) -> dict:
    # Use high-resolution timestamp to guarantee uniqueness across parallel calls.
    unique_dir = f"/nemo_run/ioi_run_{worker_id}_{os.getpid()}_{time.time_ns()}"

    try:
        # 1. Create all necessary files locally (sandbox shares filesystem)
        precompiled_dir = task_args.get("precompiled_dir")
        os.makedirs(unique_dir, exist_ok=True)
        os.makedirs(os.path.join(unique_dir, "graders"), exist_ok=True)
        # Copy precompiled assets into unique run directory
        if precompiled_dir and os.path.isdir(precompiled_dir):
            shutil.copytree(precompiled_dir, unique_dir, dirs_exist_ok=True)
        # Write contestant solution
        with open(os.path.join(unique_dir, "graders", f"{task_args['problem_id']}.cpp"), "w", encoding="utf-8") as f:
            f.write(task_args["generated_code"])
        # Prepare input and expected output files
        with open(os.path.join(unique_dir, "input.txt"), "w", encoding="utf-8") as f:
            f.write(task_args["test_input"])
        with open(os.path.join(unique_dir, "correct_output.txt"), "w", encoding="utf-8") as f:
            f.write(task_args["test_output"])

        # 2. Compile only the problem solution (skip checker/grader recompilation)
        compile_command = f"cd {unique_dir} && ./compile.sh"
        sandbox = LocalSandbox()
        compile_result, _ = worker_loop.run_until_complete(
            sandbox.execute_code(compile_command, language="shell", timeout=120)
        )

        result = {
            "compile_success": not compile_result.get("stderr"),
            "compile_stdout": compile_result.get("stdout", ""),
            "compile_stderr": compile_result.get("stderr", ""),
            "run_stdout": "",
            "run_stderr": "",
            "error": "",
            "score": 0.0,
        }

        if not result["compile_success"]:
            return result

        # 3. Run the code
        run_command = f"cd {unique_dir} && ./run.sh"
        run_result, _ = worker_loop.run_until_complete(
            sandbox.execute_code(run_command, language="shell", timeout=120)
        )

        run_stdout = run_result.get("stdout", "")
        run_stderr = run_result.get("stderr", "")

        result.update(
            {
                "run_stdout": run_stdout,
                "run_stderr": run_stderr,
            }
        )

        try:
            result["score"] = float(result["run_stdout"].strip())
        except (ValueError, TypeError):
            result["score"] = 0.0

        return result

    except Exception as e:
        return {"score": 0.0, "output": "", "error": str(e)}

    finally:
        # 4. Clean up the directory locally
        try:
            shutil.rmtree(unique_dir, ignore_errors=True)
        except Exception:
            pass


def run_input_case(task_args: dict, worker_id: int) -> dict:
    # Use high-resolution timestamp to guarantee uniqueness across parallel calls.
    unique_dir = f"/nemo_run/ioi_run_{worker_id}_{os.getpid()}_{time.time_ns()}"

    try:
        # 1. Create all necessary files locally (sandbox shares filesystem)
        os.makedirs(unique_dir, exist_ok=True)
        for filepath, content in task_args.get("run_files", []):
            target_path = os.path.join(unique_dir, os.path.basename(filepath))
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
        for fname in ("compile", "run"):
            fpath = os.path.join(unique_dir, fname)
            if os.path.exists(fpath):
                os.chmod(fpath, 0o755)
        # Write contestant solution into problem solution file
        solution_path = os.path.join(unique_dir, f"{task_args['problem_id']}.cpp")
        with open(solution_path, "w", encoding="utf-8") as f:
            f.write(task_args["generated_code"])
        # Prepare only input file (no ground-truth for input-only runs)
        with open(os.path.join(unique_dir, "input.txt"), "w", encoding="utf-8") as f:
            f.write(task_args["test_input"])

        # 2. Compile using run_files toolchain
        compile_command = f"cd {unique_dir} && ./compile"
        sandbox = LocalSandbox()
        compile_result, _ = worker_loop.run_until_complete(
            sandbox.execute_code(compile_command, language="shell", timeout=120)
        )

        result = {
            "compile_success": not compile_result.get("stderr"),
            "compile_stdout": compile_result.get("stdout", ""),
            "compile_stderr": compile_result.get("stderr", ""),
            "run_stdout": "",
            "run_stderr": "",
            "error": "",
        }

        if not result["compile_success"]:
            return result

        # 3. Run the code using run_files runner
        run_command = f"cd {unique_dir} && ./run < input.txt"
        run_result, _ = worker_loop.run_until_complete(
            sandbox.execute_code(run_command, language="shell", timeout=120, max_output_characters=1000000)
        )

        run_stdout = sha256_hex(run_result.get("stdout", ""))
        run_stderr = run_result.get("stderr", "")

        result.update(
            {
                "run_stdout": run_stdout,
                "run_stderr": run_stderr,
            }
        )

        return result

    except Exception as e:
        return {"run_stdout": "", "run_stderr": "", "error": str(e)}

    finally:
        # 4. Clean up the directory locally
        try:
            shutil.rmtree(unique_dir, ignore_errors=True)
        except Exception:
            pass


def extract_final_cpp_block(text):
    pattern = r"```(?:cpp|Cpp)\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1] if matches else ""


def add_includes(code: str, problem_id: str) -> str:
    """
    Fix common compilation errors for IOI problems.
    """
    if not code:
        return code
    # has most of the useful functions
    code_header = "#include <bits/stdc++.h>\n"
    # include the problem header
    problem_header_include = f'#include "{problem_id}.h"'
    if problem_header_include not in code:
        code_header += problem_header_include + "\n"
    # use namespace std since models forget std:: often
    if "using namespace std;" not in code and "std::" not in code:
        code_header += "\nusing namespace std;\n\n"
    # add missing dummy implementations for IOI 25 triples problem
    dummy = ""
    if problem_id == "triples":
        has_count = re.search(r"\bcount_triples\s*\(", code) is not None
        has_construct = re.search(r"\bconstruct_range\s*\(", code) is not None
        if has_construct and not has_count:
            dummy += "long long count_triples(std::vector<int> H){return 0LL;}\n"
        elif has_count and not has_construct:
            dummy += "std::vector<int> construct_range(int M,int K){return {};}\n"
    return code_header + code + ("\n" + dummy if dummy else "")


class IOIEvaluator(BaseEvaluator):
    def __init__(self, config: dict, num_parallel_requests: int = 10):
        super().__init__(config, num_parallel_requests)
        self.eval_cfg = IOIEvaluatorConfig(_init_nested=True, **config)

        # Heavy runtime resources are lazily initialized within _evaluate_entry.
        self.sandbox = None
        self.metadata = None
        self.inputdata = None
        self.precompiled_cache = {}
        self.pool = None

    async def _initialize_runtime(self):
        """Asynchronously create sandbox and related runtime state on first use."""
        if self.sandbox is not None:
            return  # Already initialized

        # Run blocking setup in a background thread to avoid nested event‐loop issues.
        def _setup():
            sbox = LocalSandbox()
            wait_for_sandbox(sbox)
            # Remember the thread id that owns this sandbox instance.
            sbox._owner_tid = threading.get_ident()

            if not os.path.exists(self.eval_cfg.test_file):
                raise FileNotFoundError(
                    f"Metadata file {self.eval_cfg.test_file} does not exist."
                    " This file is generated when preparing the IOI dataset, and found in the dataset directory. "
                    " Please provide a valid parameter for ++eval_config.test_file=x when running IOI Evaluation."
                )
            with open(self.eval_cfg.test_file, "r") as f:
                metadata_local = json.load(f)
            input_local = None
            if self.eval_cfg.input_file:
                if not os.path.exists(self.eval_cfg.input_file):
                    raise FileNotFoundError(
                        f"Input file {self.eval_cfg.input_file} does not exist."
                        " Please provide a valid parameter for ++eval_config.input_file=x when running IOI Evaluation."
                    )
                with open(self.eval_cfg.input_file, "r") as f:
                    input_local = json.load(f)
            pool_local = multiprocessing.Pool(
                processes=self.eval_cfg.test_batch_size,
                initializer=init_worker,
            )

            return sbox, metadata_local, input_local, pool_local

        self.sandbox, self.metadata, self.inputdata, self.pool = await asyncio.to_thread(_setup)

    # Internal helper
    async def _evaluate_entry(self, entry: dict) -> dict:
        # Ensure runtime (sandbox, metadata, pool, etc.) is ready for evaluation.
        await self._initialize_runtime()
        completion = add_includes(extract_final_cpp_block(entry["generation"]), entry["ioi_id"])

        pid = entry["ioi_id"]

        # Retrieve helper scripts and grader resources from metadata instead of the dataset entry.
        problem_metadata = self.metadata[entry["name"]]
        subtask_meta = problem_metadata[entry["subtask"]]
        compile_code = subtask_meta["compile"]
        run_code = subtask_meta["run"]
        grader_files = subtask_meta["grader_files"]
        run_files = subtask_meta.get("run_files", [])

        if pid not in self.precompiled_cache:
            grader_dir = await asyncio.to_thread(
                _precompile_grader,
                pid,
                grader_files,
                compile_code,
                run_code,
                self.sandbox,
            )
            self.precompiled_cache[pid] = {"grader": grader_dir}
        pre_dir = self.precompiled_cache[pid]["grader"]

        subtask_state = {
            st: {
                "score": data["subtask_score"],
                "precision": data["score_precision"],
                "outputs": [],
                "scores": [],
                "passed": True,
            }
            for st, data in problem_metadata.items()
        }

        all_tests = [(st, tname, t) for st, data in problem_metadata.items() for tname, t in data["tests"].items()]

        batch_size = self.eval_cfg.test_batch_size

        for i in range(0, len(all_tests), batch_size):
            batch = [t for t in all_tests[i : i + batch_size] if subtask_state[t[0]]["passed"]]
            if not batch:
                continue

            tasks = []
            for _, _, test_data in batch:
                tasks.append(
                    {
                        "generated_code": completion,
                        "problem_id": pid,
                        "precompiled_dir": pre_dir,
                        "test_input": test_data["input"],
                        "test_output": test_data["output"],
                    }
                )

            # map with unique worker id argument
            results = await asyncio.to_thread(
                self.pool.starmap, run_test_case, [(ta, idx) for idx, ta in enumerate(tasks)]
            )

            for (subtask, test_name, _), result in zip(batch, results):
                st = subtask_state[subtask]
                result["test_name"] = test_name
                st["outputs"].append(result)
                st["scores"].append(float(result.get("score", 0)))
                if float(result.get("score", 0)) == 0.0:
                    st["passed"] = False

                # Debug prints similar to original implementation
                if not result.get("compile_success", True):
                    print(
                        f"Compile failed for problem '{entry['name']}', test '{test_name}':\n"
                        f"--- STDOUT ---\n{result.get('compile_stdout', '').strip()}\n"
                        f"--- STDERR ---\n{result.get('compile_stderr', '').strip()}\n"
                    )

        test_case_results = {}
        for st, data in subtask_state.items():
            score = round(min(data["scores"]) * data["score"], data["precision"]) if data["scores"] else 0.0
            test_case_results[st] = {"score": score, "outputs": data["outputs"]}

        # Optionally run custom input cases
        input_outputs = []
        if self.inputdata is not None:
            problem_inputs = self.inputdata[str(entry["id"])]
            for i in range(0, len(problem_inputs), batch_size):
                batch = problem_inputs[i : i + batch_size]
                tasks = []
                for test_data in batch:
                    tasks.append(
                        {
                            "generated_code": completion,
                            "problem_id": pid,
                            "run_files": run_files,
                            "test_input": test_data["content"],
                        }
                    )
                # map with unique worker id argument
                results = await asyncio.to_thread(
                    self.pool.starmap, run_input_case, [(ta, idx) for idx, ta in enumerate(tasks)]
                )
                for test_data, result in zip(batch, results):
                    test_name = test_data["file_name"]
                    test_type = "input"
                    result["test_name"] = test_name
                    result["test_type"] = test_type
                    input_outputs.append(result)

        return {
            "name": entry["name"],
            "subtask": entry["subtask"],
            "test_case_results": test_case_results,
            "input_case_results": input_outputs,
        }

    async def eval_full(self, input_files):  # type: ignore[override]
        for jsonl_file in unroll_files(input_files):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                all_samples = [json.loads(line) for line in f]

            tasks = [self._evaluate_entry(s) for s in all_samples]
            outputs = await asyncio.gather(*tasks)

            for s, o in zip(all_samples, outputs):
                s["test_case_results"] = o["test_case_results"]
                s["input_case_results"] = o["input_case_results"]

            jdump(all_samples, jsonl_file, mode="wt")

        if self.pool is not None:
            self.pool.close()
            self.pool.join()

    async def eval_single(self, data_point: dict):
        return await self._evaluate_entry(data_point)

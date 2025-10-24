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
import multiprocessing
import os
import re
import threading
import time
from typing import Dict

from nemo_skills.code_execution.sandbox import LocalSandbox
from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.file_utils import jdump
from nemo_skills.utils import nested_dataclass


@nested_dataclass(kw_only=True)
class IOIEvaluatorConfig(BaseEvaluatorConfig):
    test_file: str = "test_metadata.json"
    num_workers: int = 16  # number of test workers
    test_batch_size: int = 16  # number of tests to run concurrently
    overwrite: bool = False


_precompile_loop_tls = threading.local()
worker_sandbox = None  # type: ignore
worker_loop = asyncio.new_event_loop()
asyncio.set_event_loop(worker_loop)


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

    pre_dir = f"/tmp/ioi_pre_{problem_name}_{os.getpid()}"
    # Build shell script to create files and invoke compile.sh.
    creation_cmds = [
        f"mkdir -p {pre_dir}/graders",
    ]
    # Dump grader related files
    for filepath, content in grader_files:
        dir_name = os.path.dirname(filepath)
        if dir_name:
            creation_cmds.append(f"mkdir -p {pre_dir}/{dir_name}")
        creation_cmds.append(f"cat <<'_EOT_' > {pre_dir}/{filepath}\n{content}\n_EOT_\n")

    # Write compile.sh and run.sh as provided (needed later in workers)
    creation_cmds.append(
        f"cat <<'_EOT_' > {pre_dir}/compile.sh\n{compile_code}\n_EOT_\nchmod +x {pre_dir}/compile.sh\n"
    )
    creation_cmds.append(f"cat <<'_EOT_' > {pre_dir}/run.sh\n{run_code}\n_EOT_\nchmod +x {pre_dir}/run.sh\n")

    setup_script = "\n".join(creation_cmds)
    # 1. create files
    _sandbox_exec_sync(sandbox, setup_script, language="shell", timeout=120)

    # 2. run compile.sh but ignore final failure when problem cpp missing
    _sandbox_exec_sync(sandbox, f"cd {pre_dir} && ./compile.sh || true", language="shell", timeout=120)

    return pre_dir


def run_test_case(task_args: dict, worker_id: int) -> dict:
    # Use high-resolution timestamp to guarantee uniqueness across parallel calls.
    unique_dir = f"/tmp/ioi_run_{worker_id}_{os.getpid()}_{time.time_ns()}"

    try:
        # 1. Create all necessary files in one batch command
        precompiled_dir = task_args.get("precompiled_dir")
        # Step 1: prepare the working directory and copy shared pre-compiled artifacts first
        file_creation_commands = [
            # Create the unique run directory itself
            f"mkdir -p {unique_dir}",
            # Ensure `graders/` directory exists
            f"mkdir -p {unique_dir}/graders",
            f"cp -r {precompiled_dir}/* {unique_dir}/",
            # Next write the contestant's generated solution into the graders folder so it is not overwritten
            f"cat <<'_EOT_' > {unique_dir}/graders/{task_args['problem_id']}.cpp\n{task_args['generated_code']}\n_EOT_\n",
        ]

        # Prepare input and expected output files
        file_creation_commands.append(f"cat <<'_EOT_' > {unique_dir}/input.txt\n{task_args['test_input']}\n_EOT_\n")
        file_creation_commands.append(
            f"cat <<'_EOT_' > {unique_dir}/correct_output.txt\n{task_args['test_output']}\n_EOT_\n"
        )

        setup_script = "\n".join(file_creation_commands)
        sandbox = LocalSandbox()
        setup_result, _ = worker_loop.run_until_complete(
            sandbox.execute_code(setup_script, language="shell", timeout=120)
        )
        if setup_result.get("stderr"):
            raise Exception(f"File setup failed: {setup_result['stderr']}")

        # 2. Compile only the problem solution (skip checker/grader recompilation)
        # Compile the solution together with optional grader/stub sources without
        # recompiling the checker/manager again.
        compile_command = (
            f"cd {unique_dir} && "
            f'SRC="graders/{task_args["problem_id"]}.cpp"; '
            f'[ -e graders/grader.cpp ] && SRC="$SRC graders/grader.cpp"; '
            f'[ -e graders/stub.cpp ] && SRC="$SRC graders/stub.cpp"; '
            f"g++ -DEVAL -std=gnu++17 -O2 -pipe -s -o graders/{task_args['problem_id']} $SRC"
        )
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
        # 4. Clean up the directory
        # Fire and forget; ignore return values
        try:
            sandbox = LocalSandbox()
            worker_loop.run_until_complete(sandbox.execute_code(f"rm -rf {unique_dir}", language="shell", timeout=120))
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
        self.sandbox = None  # type: ignore
        self.metadata = None  # type: ignore
        self.precompiled_cache: Dict[str, str] = {}
        self.pool = None  # type: ignore

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
            pool_local = multiprocessing.Pool(
                processes=self.eval_cfg.test_batch_size,
                initializer=init_worker,
            )

            return sbox, metadata_local, pool_local

        self.sandbox, self.metadata, self.pool = await asyncio.to_thread(_setup)

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

        if pid not in self.precompiled_cache:
            self.precompiled_cache[pid] = await asyncio.to_thread(
                _precompile_grader,
                pid,
                grader_files,
                compile_code,
                run_code,
                self.sandbox,
            )
        pre_dir = self.precompiled_cache[pid]

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

        return {
            "name": entry["name"],
            "subtask": entry["subtask"],
            "test_case_results": test_case_results,
        }

    async def eval_full(self):  # type: ignore[override]
        jsonl_file = self.eval_cfg.input_file
        with open(jsonl_file, "r", encoding="utf-8") as f:
            all_samples = [json.loads(line) for line in f]

        tasks = [self._evaluate_entry(s) for s in all_samples]
        outputs = await asyncio.gather(*tasks)

        for s, o in zip(all_samples, outputs):
            s["test_case_results"] = o["test_case_results"]
            s["eval_status"] = o["eval_status"]

        jdump(all_samples, jsonl_file, mode="wt")

        if self.pool is not None:
            self.pool.close()
            self.pool.join()

    async def eval_single(self, data_point: dict):
        return await self._evaluate_entry(data_point)

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import asyncio
import json
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from nemo_skills.code_execution.sandbox import LocalSandbox
from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.file_utils import jdump
from nemo_skills.utils import nested_dataclass, unroll_files


@nested_dataclass(kw_only=True)
class CCCEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for CCC code-evaluation execution and batching."""

    test_file: str = "test_metadata.json"
    num_workers: int = 16
    test_batch_size: int = 16
    time_scale: float = 1.0
    overwrite: bool = False


_precompile_loop_tls = threading.local()
_test_loop_tls = threading.local()
_test_sandbox_tls = threading.local()
worker_sandbox = None  # type: ignore


def _sandbox_exec_sync(sandbox: LocalSandbox, cmd: str, *, language: str = "shell", timeout: int = 120):
    """Execute a sandbox command synchronously on a thread-local event loop."""
    loop = getattr(_precompile_loop_tls, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _precompile_loop_tls.loop = loop
    return loop.run_until_complete(sandbox.execute_code(cmd, language=language, timeout=timeout))[0]


def _test_exec_sync(sandbox: LocalSandbox, cmd: str, *, language: str = "shell", timeout: int = 120):
    """Execute a test command synchronously on a thread-local test loop."""
    loop = getattr(_test_loop_tls, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _test_loop_tls.loop = loop
    return loop.run_until_complete(sandbox.execute_code(cmd, language=language, timeout=timeout))[0]


def _get_thread_test_sandbox() -> LocalSandbox:
    """Get or create the worker-local sandbox instance used for test execution."""
    sandbox = getattr(_test_sandbox_tls, "sandbox", None)
    if sandbox is None:
        sandbox = LocalSandbox()
        _test_sandbox_tls.sandbox = sandbox
    return sandbox


def wait_for_sandbox(sandbox, timeout: int = 240, poll: float = 1.0):
    """Wait until sandbox responds to a trivial shell command."""
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


def _precompile_problem(problem_id: str, grader_files, compile_code: str, run_code: str, sandbox: LocalSandbox) -> str:
    """Materialize grader assets and precompile a problem once for reuse."""
    if getattr(sandbox, "_owner_tid", None) != threading.get_ident():
        sandbox = LocalSandbox()
        wait_for_sandbox(sandbox)
        sandbox._owner_tid = threading.get_ident()

    pre_dir = f"/nemo_run/ccc_pre_{problem_id}_{os.getpid()}"
    os.makedirs(os.path.join(pre_dir, "graders"), exist_ok=True)

    for filepath, content in grader_files:
        target_path = os.path.join(pre_dir, filepath)
        target_dir = os.path.dirname(target_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)

    for script_name, script_content in (("compile.sh", compile_code), ("run.sh", run_code)):
        script_path = os.path.join(pre_dir, script_name)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)

    _sandbox_exec_sync(sandbox, f"cd {pre_dir} && ./compile.sh || true", language="shell", timeout=120)
    return pre_dir


def run_test_case(task_args: dict, worker_id: int) -> dict:
    """Compile and run one generated solution against one CCC test case."""
    unique_dir = f"/nemo_run/ccc_run_{worker_id}_{os.getpid()}_{time.time_ns()}"
    try:
        precompiled_dir = task_args.get("precompiled_dir")
        os.makedirs(unique_dir, exist_ok=True)
        os.makedirs(os.path.join(unique_dir, "graders"), exist_ok=True)
        os.makedirs(os.path.join(unique_dir, "tmp"), exist_ok=True)
        if precompiled_dir and os.path.isdir(precompiled_dir):
            shutil.copytree(precompiled_dir, unique_dir, dirs_exist_ok=True)
        if task_args.get("task_type") == "SIMULATION":
            with open(os.path.join(unique_dir, "solution.odo"), "w", encoding="utf-8") as f:
                f.write(task_args["generated_code"])
        else:
            with open(
                os.path.join(unique_dir, "graders", f"{task_args['problem_id']}.cpp"), "w", encoding="utf-8"
            ) as f:
                f.write(task_args["generated_code"])
        with open(os.path.join(unique_dir, "input.txt"), "w", encoding="latin1") as f:
            f.write(task_args["test_input"])
        with open(os.path.join(unique_dir, "correct_output.txt"), "w", encoding="latin1") as f:
            f.write(task_args["test_output"])

        sandbox = _get_thread_test_sandbox()
        compile_result = _test_exec_sync(sandbox, f"cd {unique_dir} && ./compile.sh", language="shell", timeout=120)
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

        run_timeout = max(1, int(120 * float(task_args.get("time_scale", 1.0))))
        run_result = _test_exec_sync(
            sandbox,
            f"cd {unique_dir} && export TMPDIR={unique_dir}/tmp && TIME_LIMIT_SCALE={task_args.get('time_scale', 1.0)} ./run.sh",
            language="shell",
            timeout=run_timeout,
        )
        result["run_stdout"] = run_result.get("stdout", "")
        result["run_stderr"] = run_result.get("stderr", "")
        try:
            result["score"] = float(result["run_stdout"].strip())
        except (ValueError, TypeError):
            result["score"] = 0.0
        return result
    except Exception as e:
        return {"score": 0.0, "output": "", "error": str(e)}
    finally:
        try:
            shutil.rmtree(unique_dir, ignore_errors=True)
        except Exception:
            pass


def extract_final_cpp_block(text):
    """Extract the last fenced C++ block from model output text."""
    pattern = r"```(?:cpp|Cpp)\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1] if matches else (text or "")


def extract_final_text_block(text):
    """Extract the last fenced plain-text block from model output text."""
    pattern = r"```(?:txt|text|plain)\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return matches[-1] if matches else (text or "")


def extract_task_config(problem_metadata: dict) -> dict:
    """Read optional grader config JSON embedded in problem metadata files."""
    for relpath, content in problem_metadata.get("grader_files", []):
        if relpath == "graders/grader_config.json":
            try:
                return json.loads(content)
            except Exception:
                return {}
    return {}


def add_includes(code: str, problem_header_include: str | None = None, problem_id: str | None = None) -> str:
    """Inject standard includes/namespace and problem-specific compatibility shims."""
    if not code:
        return code
    code_header = "#include <bits/stdc++.h>\n"
    if problem_header_include:
        header_include = f'#include "{problem_header_include}"'
        if header_include not in code:
            code_header += header_include + "\n"
    if "using namespace std;" not in code and "std::" not in code:
        code_header += "\nusing namespace std;\n\n"
    dummy = ""
    if problem_id == "triples":
        has_count = re.search(r"\bcount_triples\s*\(", code) is not None
        has_construct = re.search(r"\bconstruct_range\s*\(", code) is not None
        if has_construct and not has_count:
            dummy += "long long count_triples(std::vector<int> H){return 0LL;}\n"
        elif has_count and not has_construct:
            dummy += "std::vector<int> construct_range(int M,int K){return {};}\n"
    return code_header + code + (("\n" + dummy) if dummy else ("\n" if not code.endswith("\n") else ""))


class CCCEvaluator(BaseEvaluator):
    """Evaluator that compiles and executes CCC submissions against grader tests."""

    def __init__(self, config: dict, num_parallel_requests: int = 10):
        """Initialize evaluator state and deferred runtime resources."""
        super().__init__(config, num_parallel_requests)
        self.eval_cfg = CCCEvaluatorConfig(_init_nested=True, **config)
        self.sandbox = None
        self.metadata = None
        self.precompiled_cache = {}
        self.pool = None

    async def _initialize_runtime(self):
        """Lazily initialize sandbox, metadata, and worker pool."""
        if self.sandbox is not None:
            return

        def _setup():
            """Create runtime dependencies on a worker thread."""
            sbox = LocalSandbox()
            wait_for_sandbox(sbox)
            sbox._owner_tid = threading.get_ident()
            if not os.path.exists(self.eval_cfg.test_file):
                raise FileNotFoundError(f"Metadata file {self.eval_cfg.test_file} does not exist.")
            with open(self.eval_cfg.test_file, "r", encoding="utf-8") as f:
                metadata_local = json.load(f)
            pool_local = ThreadPoolExecutor(max_workers=self.eval_cfg.test_batch_size)
            return sbox, metadata_local, pool_local

        self.sandbox, self.metadata, self.pool = await asyncio.to_thread(_setup)

    def _get_precompiled_dir(self, problem_id: str, problem_metadata: dict):
        """Return cached precompiled grader dir for a problem, compiling if needed."""
        if problem_id in self.precompiled_cache:
            cached = self.precompiled_cache[problem_id]
            return cached["grader"] if isinstance(cached, dict) else cached

        grader_dir = _precompile_problem(
            problem_id,
            problem_metadata["grader_files"],
            problem_metadata["compile"],
            problem_metadata["run"],
            self.sandbox,
        )
        self.precompiled_cache[problem_id] = {"grader": grader_dir}
        return grader_dir

    def _build_test_task(
        self, problem_id: str, pre_dir: str, completion: str, test_data: dict, task_type: str = "Batch"
    ):
        """Build a run_test_case task payload for one test input/output pair."""
        return {
            "generated_code": completion,
            "task_type": task_type,
            "problem_id": problem_id,
            "precompiled_dir": pre_dir,
            "test_input": test_data["input"],
            "test_output": test_data["output"],
            "time_scale": self.eval_cfg.time_scale,
        }

    def _aggregate_subtask_score(self, subtask_meta: dict, outputs: list[dict], failed: bool = False) -> float:
        """Aggregate per-test results into a subtask score using configured policy."""
        aggregation = subtask_meta["aggregation"]
        if aggregation == "min":
            if failed:
                return 0.0
            scores = [float(out.get("score", 0.0)) for out in outputs]
            precision = max(0, int(subtask_meta.get("score_precision", 0)))
            return round(
                (min(scores) if scores else 0.0) * float(subtask_meta["score"]),
                precision,
            )
        if aggregation == "sum_tests":
            return float(sum(1 for out in outputs if float(out.get("score", 0.0)) > 0.0))
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    async def _evaluate_entry(self, entry: dict) -> dict:
        """Evaluate one generation entry and return computed test-case results."""
        await self._initialize_runtime()

        problem_id = entry["problem_id"]
        problem_metadata = self.metadata[problem_id]
        task_config = extract_task_config(problem_metadata)
        task_type = str(task_config.get("task_type", "Batch"))
        if task_type == "SIMULATION":
            completion = extract_final_text_block((entry.get("generation") or ""))
        elif task_type == "MULTIFILE":
            completion = extract_final_cpp_block((entry.get("generation") or ""))
        else:
            completion = add_includes(
                extract_final_cpp_block((entry.get("generation") or "")),
                problem_metadata.get("problem_header_include"),
                problem_id,
            )
        pre_dir = await asyncio.to_thread(self._get_precompiled_dir, problem_id, problem_metadata)

        subtask_state = {
            subtask_name: {
                "aggregation": subtask_meta["aggregation"],
                "outputs": [],
                "failed": False,
            }
            for subtask_name, subtask_meta in problem_metadata["subtasks"].items()
        }
        test_to_subtasks = {}
        for subtask_name, subtask_meta in problem_metadata["subtasks"].items():
            for test_name in subtask_meta["test_names"]:
                test_to_subtasks.setdefault(test_name, []).append(subtask_name)

        all_test_items = list(problem_metadata["all_tests"].items())
        batch_size = self.eval_cfg.test_batch_size
        for i in range(0, len(all_test_items), batch_size):
            candidate_batch = all_test_items[i : i + batch_size]
            batch = []
            tasks = []
            for test_name, test_data in candidate_batch:
                subtasks = test_to_subtasks.get(test_name, [])
                should_run = False
                for subtask_name in subtasks:
                    state = subtask_state[subtask_name]
                    if state["aggregation"] == "sum_tests" or not state["failed"]:
                        should_run = True
                        break
                if not should_run:
                    continue
                batch.append((test_name, test_data))
                tasks.append(self._build_test_task(problem_id, pre_dir, completion, test_data, task_type=task_type))
            if not batch:
                continue
            loop = asyncio.get_running_loop()
            futures = [loop.run_in_executor(self.pool, run_test_case, task, idx) for idx, task in enumerate(tasks)]
            results = await asyncio.gather(*futures)
            for (test_name, _), result in zip(batch, results):
                result["test_name"] = test_name
                test_group = problem_metadata["all_tests"][test_name].get("group")
                if test_group is not None:
                    result["test_group"] = test_group
                for subtask_name in test_to_subtasks.get(test_name, []):
                    state = subtask_state[subtask_name]
                    if state["aggregation"] == "min" and state["failed"]:
                        continue
                    state["outputs"].append(dict(result))
                    if state["aggregation"] == "min" and float(result.get("score", 0.0)) == 0.0:
                        state["failed"] = True

        test_case_results = {}
        for subtask_name, subtask_meta in problem_metadata["subtasks"].items():
            state = subtask_state[subtask_name]
            test_case_results[subtask_name] = {
                "score": self._aggregate_subtask_score(subtask_meta, state["outputs"], failed=state["failed"]),
                "outputs": state["outputs"],
            }

        return {
            "name": entry["name"],
            "subtask": entry["subtask"],
            "test_case_results": test_case_results,
        }

    async def eval_full(self):  # type: ignore[override]
        """Evaluate all configured input files and write results in place."""
        await self._initialize_runtime()

        for jsonl_file in unroll_files(self.eval_cfg.input_file):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                all_samples = [json.loads(line) for line in f]

            # Precompile each unique problem once before row-level concurrency starts.
            unique_problem_ids = []
            seen_problem_ids = set()
            for sample in all_samples:
                problem_id = sample["problem_id"]
                if problem_id not in seen_problem_ids:
                    seen_problem_ids.add(problem_id)
                    unique_problem_ids.append(problem_id)

            for problem_id in unique_problem_ids:
                await asyncio.to_thread(self._get_precompiled_dir, problem_id, self.metadata[problem_id])

            tasks = [self._evaluate_entry(sample) for sample in all_samples]
            outputs = await asyncio.gather(*tasks)
            for sample, output in zip(all_samples, outputs):
                sample["test_case_results"] = output["test_case_results"]
            jdump(all_samples, jsonl_file, mode="wt")

        if self.pool is not None:
            self.pool.shutdown(wait=True)

    async def eval_single(self, data_point: dict):
        """Evaluate a single data point entry."""
        return await self._evaluate_entry(data_point)

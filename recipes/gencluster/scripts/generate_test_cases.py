#!/usr/bin/env python3
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


import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nemo_skills.code_execution.sandbox import LocalSandbox

# Thread-local sandbox/loop for parallel workers
_thread_local = threading.local()


def _get_thread_context():
    loop = getattr(_thread_local, "loop", None)
    sb = getattr(_thread_local, "sandbox", None)
    if loop is None or sb is None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        sb = LocalSandbox()
        _thread_local.loop = loop
        _thread_local.sandbox = sb
    return loop, sb


def wait_for_sandbox(sandbox, loop, timeout: int = 240, poll: float = 1.0):
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            result, _ = loop.run_until_complete(sandbox.execute_code("echo hello world", language="shell", timeout=10))
            if result.get("stdout", "").strip() == "hello world":
                return
        except Exception:
            pass
        loop.run_until_complete(asyncio.sleep(poll))
    raise RuntimeError(f"Sandbox not ready after waiting {timeout}s")


def run_generator(gen_binary_path, timeout=10, *, loop=None, sandbox: LocalSandbox = None):
    """Run a generator binary and return its stdout output"""
    # Prefer sandbox if provided
    if sandbox is not None and loop is not None:
        try:
            cmd = shlex.quote(str(gen_binary_path))
            result, _ = loop.run_until_complete(sandbox.execute_code(cmd, language="shell", timeout=timeout))
            if result.get("process_status") == "completed":
                return True, result.get("stdout", "")
            elif result.get("process_status") == "timeout":
                return False, "Generator timed out"
            else:
                return False, f"Generator failed: {result.get('stderr', '')}"
        except Exception as e:
            return False, f"Generator error: {str(e)}"
    # Fallback local execution
    try:
        result = subprocess.run([str(gen_binary_path)], capture_output=True, text=True, timeout=timeout)

        if result.returncode == 0:
            return True, result.stdout
        else:
            return (
                False,
                f"Generator failed with return code {result.returncode}: {result.stderr}",
            )

    except subprocess.TimeoutExpired:
        return False, "Generator timed out"
    except Exception as e:
        return False, f"Generator error: {str(e)}"


def run_generator_to_sandbox_file(gen_binary_path, timeout=10, *, loop=None, sandbox: LocalSandbox = None):
    """Run generator inside sandbox and write its stdout to a sandbox temp file. Returns (success, sandbox_tmp_path_or_error)."""
    if sandbox is None or loop is None:
        return False, "Sandbox not available"
    try:
        quoted_bin = shlex.quote(str(gen_binary_path))
        script = f'tmp_file=$(mktemp)\n{quoted_bin} > "$tmp_file"\necho "$tmp_file"\n'
        result, _ = loop.run_until_complete(sandbox.execute_code(script, language="shell", timeout=timeout))
        if result.get("process_status") == "timeout":
            return False, "Generator timed out"
        if result.get("process_status") != "completed":
            return False, result.get("stderr", "Generator failed")
        # The last line of stdout is the temp path
        stdout = result.get("stdout", "").strip()
        tmp_path = stdout.splitlines()[-1] if stdout else ""
        if not tmp_path:
            return False, "Failed to build sandbox temp file path"
        return True, tmp_path
    except Exception as e:
        return False, f"Generator error: {str(e)}"


def run_validator(
    val_binary_path,
    test_data,
    timeout=10,
    *,
    loop=None,
    sandbox: LocalSandbox = None,
    input_path_in_sandbox: str = None,
):
    """Run a validator binary with test data as stdin"""
    # Prefer sandbox if provided
    if sandbox is not None and loop is not None:
        try:
            quoted_bin = shlex.quote(str(val_binary_path))
            if not input_path_in_sandbox:
                return "error"
            quoted_in = shlex.quote(input_path_in_sandbox)
            script = f"{quoted_bin} < {quoted_in}\n"
            result, _ = loop.run_until_complete(sandbox.execute_code(script, language="shell", timeout=timeout))
            if result.get("process_status") == "timeout":
                return "timeout"
            output_lower = result.get("stdout", "").lower().strip()
            if "passed" in output_lower:
                return "passed"
            if "failed" in output_lower:
                return "failed"
            return "passed" if result.get("process_status") == "completed" else "failed"
        except Exception:
            return "error"
    # Fallback local execution
    try:
        result = subprocess.run(
            [str(val_binary_path)],
            input=test_data,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Check if validator output contains "passed" or "failed" (case insensitive)
        output_lower = result.stdout.lower().strip()

        if "passed" in output_lower:
            return "passed"
        elif "failed" in output_lower:
            return "failed"
        else:
            # If no clear passed/failed, check return code
            return "passed" if result.returncode == 0 else "failed"

    except subprocess.TimeoutExpired:
        return "timeout"
    except Exception:
        return "error"


def validate_dataset(
    test_data,
    val_binaries,
    min_validators=7,
    *,
    loop=None,
    sandbox: LocalSandbox = None,
    sandbox_input_path: str = None,
):
    """Validate test data against all validators and return if it passes threshold"""
    validation_results = []
    passed_count = 0

    for val_binary in val_binaries:
        result = run_validator(
            val_binary,
            test_data,
            loop=loop,
            sandbox=sandbox,
            input_path_in_sandbox=sandbox_input_path,
        )
        validation_results.append({"validator": val_binary.name, "result": result})

        if result == "passed":
            passed_count += 1

    if (len(val_binaries) * 0.8) < min_validators:
        is_valid = passed_count >= (len(val_binaries) * 0.8)
    else:
        is_valid = passed_count >= min_validators

    return is_valid, passed_count, len(val_binaries), validation_results


def generate_datasets_for_problem(
    problem_dir,
    binary_dir,
    output_dir,
    n_datasets,
    min_validators=7,
    generator_workers=4,
):
    """Generate N validated datasets for a single problem"""
    problem_name = problem_dir.name

    # Get generator and validator binaries
    gen_dir = binary_dir / problem_name / "gen"
    val_dir = binary_dir / problem_name / "val"

    if not gen_dir.exists() or not val_dir.exists():
        print(f"⚠️  Skipping {problem_name}: missing gen or val directory")
        return 0, []

    gen_binaries = [f for f in gen_dir.iterdir() if f.is_file() and os.access(f, os.X_OK)]
    val_binaries = [f for f in val_dir.iterdir() if f.is_file() and os.access(f, os.X_OK)]

    if len(gen_binaries) == 0:
        print(f"⚠️  Skipping {problem_name}: no generator binaries found")
        return 0, []

    if len(val_binaries) == 0:
        print(f"⚠️  Skipping {problem_name}: no validator binaries found")
        return 0, []

    print(f"\n=== Problem {problem_name} ===")
    print(f"Generators: {len(gen_binaries)}, Validators: {len(val_binaries)}")
    print(f"Target: {n_datasets} datasets (need ≥{min_validators}/{len(val_binaries)} validator approval)")

    # Create output directory for this problem (just the number, not "problem_X")
    problem_number = problem_name.replace("problem_", "")
    problem_output_dir = output_dir / problem_number
    problem_output_dir.mkdir(parents=True, exist_ok=True)

    # Parallel generation across generators with coordination
    generated_datasets = []
    # Detect already saved datasets and pick up numbering
    existing_indices = []
    try:
        for p in problem_output_dir.iterdir():
            if p.is_file() and p.name.startswith("dataset_") and p.suffix == ".txt":
                try:
                    idx_part = p.stem.split("_", 1)[1]
                    existing_indices.append(int(idx_part))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    existing_indices.sort()
    saved_count = len(existing_indices)
    next_index = (existing_indices[-1] + 1) if existing_indices else 1
    remaining_needed = max(0, n_datasets - saved_count)
    attempts = 0
    max_attempts = max(1, remaining_needed * 10)  # Safety limit based on remaining
    lock = threading.Lock()

    active_gens = list(gen_binaries)

    if saved_count >= n_datasets:
        print(f"  ✅ Already have {saved_count}/{n_datasets} datasets – skipping generation for this problem")
        return saved_count, generated_datasets

    def attempt_generation(gen_path):
        nonlocal attempts, saved_count, next_index, active_gens
        loop, sandbox = _get_thread_context()
        # Early stop check
        with lock:
            if saved_count >= n_datasets or attempts >= max_attempts or len(active_gens) == 0:
                return None
            attempts += 1
            attempt_no = attempts

        print(f"Attempt {attempt_no}: Using generator {gen_path.name}...")

        gen_ok = False
        sandbox_tmp_path = None
        # Use sandbox temp file to avoid moving data around
        gen_ok, gen_out = run_generator_to_sandbox_file(gen_path, loop=loop, sandbox=sandbox)
        if not gen_ok:
            print(f"  ❌ Generator failed: {gen_out}")
            return {
                "status": "gen_failed",
                "gen": gen_path,
            }
        sandbox_tmp_path = gen_out

        is_valid, passed_count, total_validators, validation_results = validate_dataset(
            None,
            val_binaries,
            min_validators,
            loop=loop,
            sandbox=sandbox,
            sandbox_input_path=sandbox_tmp_path,
        )

        if is_valid:
            with lock:
                if saved_count >= n_datasets:
                    # cleanup sandbox temp
                    try:
                        loop.run_until_complete(
                            sandbox.execute_code(
                                f"rm -f {shlex.quote(sandbox_tmp_path)}",
                                language="shell",
                                timeout=5,
                            )
                        )
                    except Exception:
                        pass
                    return {"status": "skipped_full", "gen": gen_path}
                dataset_idx = next_index
                next_index += 1
                saved_count += 1

            dataset_filename = f"dataset_{dataset_idx:03d}.txt"
            dataset_path = problem_output_dir / dataset_filename
            # Move file from sandbox temp to final location
            try:
                mv_res, _ = loop.run_until_complete(
                    sandbox.execute_code(
                        f"mv {shlex.quote(sandbox_tmp_path)} {shlex.quote(str(dataset_path))}",
                        language="shell",
                        timeout=30,
                    )
                )
                if mv_res.get("process_status") != "completed":
                    raise RuntimeError(mv_res.get("stderr", "Failed to move sandbox file"))
            except Exception as e:
                print(f"  ❌ Failed to move dataset from sandbox: {e}")
                try:
                    loop.run_until_complete(
                        sandbox.execute_code(
                            f"rm -f {shlex.quote(sandbox_tmp_path)}",
                            language="shell",
                            timeout=5,
                        )
                    )
                except Exception:
                    pass
                return {
                    "status": "gen_failed",
                    "gen": gen_path,
                }

            report_filename = f"dataset_{dataset_idx:03d}_validation.json"
            report_path = problem_output_dir / report_filename
            validation_report = {
                "dataset_file": dataset_filename,
                "generator": gen_path.name,
                "passed_validators": passed_count,
                "total_validators": total_validators,
                "validation_results": validation_results,
                "attempt_number": attempt_no,
            }
            with open(report_path, "w") as f:
                json.dump(validation_report, f, indent=2)

            print(
                f"  ✅ Dataset {saved_count}/{n_datasets} saved: {passed_count}/{total_validators} validators passed"
            )
            # Cleanup sandbox temp file
            try:
                loop.run_until_complete(
                    sandbox.execute_code(
                        f"rm -f {shlex.quote(sandbox_tmp_path)}",
                        language="shell",
                        timeout=5,
                    )
                )
            except Exception:
                pass
            return {
                "status": "saved",
                "gen": gen_path,
                "dataset_path": dataset_path,
                "validation_report": validation_report,
            }
        else:
            print(f"  ❌ Validation failed: only {passed_count}/{total_validators} validators passed")
            print(
                f"  ⛔ Dropping generator {gen_path.name} due to failed validation, validation_results: {validation_results}"
            )
            with lock:
                active_gens = [g for g in active_gens if g != gen_path]
            # Cleanup sandbox temp file
            try:
                loop.run_until_complete(
                    sandbox.execute_code(
                        f"rm -f {shlex.quote(sandbox_tmp_path)}",
                        language="shell",
                        timeout=5,
                    )
                )
            except Exception:
                pass
            return {
                "status": "validation_failed",
                "gen": gen_path,
            }

    # Thread pool for generators
    with ThreadPoolExecutor(max_workers=max(1, generator_workers)) as executor:
        futures = set()
        # Seed initial submissions (one per active generator up to capacity)
        gen_it_index = 0
        while saved_count < n_datasets and attempts < max_attempts and len(active_gens) > 0:
            # Top up futures pool
            while (
                len(futures) < generator_workers
                and len(active_gens) > 0
                and saved_count < n_datasets
                and attempts < max_attempts
            ):
                gen_path = active_gens[gen_it_index % len(active_gens)]
                gen_it_index += 1
                futures.add(executor.submit(attempt_generation, gen_path))

            if not futures:
                break

            # Process first completed
            done, futures = {next(iter(as_completed(futures)))}, futures
            future = next(iter(done))
            futures.discard(future)
            result = future.result()
            if result and result.get("status") == "saved":
                generated_datasets.append(
                    {
                        "dataset_path": result["dataset_path"],
                        "validation_report": result["validation_report"],
                    }
                )
            # Loop continues topping up and processing

    if saved_count < n_datasets:
        print(
            f"⚠️  Warning: Only generated {saved_count}/{n_datasets} datasets after {max_attempts} attempts, len(active_gens): {len(active_gens)}"
        )

    return saved_count, generated_datasets


def main():
    parser = argparse.ArgumentParser(description="Generate N validated datasets per problem")
    parser.add_argument("n_datasets", type=int, help="Number of datasets to generate per problem")
    parser.add_argument(
        "--min-validators",
        type=int,
        default=7,
        help="Minimum number of validators that must pass (default: 7)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory (default: current directory)",
    )
    parser.add_argument(
        "--workers-problems",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Number of parallel problem workers",
    )
    parser.add_argument(
        "--workers-generators",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Number of parallel generator workers per problem",
    )

    args = parser.parse_args()

    # Ensure sandbox server is up before proceeding (explicit execute check)
    print("Waiting for sandbox to be ready...")
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _sb = LocalSandbox()
    try:
        wait_for_sandbox(_sb, _loop, timeout=60, poll=1.0)
        print("✓ Sandbox is ready")
    finally:
        try:
            _loop.run_until_complete(_sb.close())
        except Exception:
            pass
        try:
            _loop.close()
        except Exception:
            pass

    if args.n_datasets <= 0:
        print("Error: N must be a positive integer")
        sys.exit(1)

    # Set up directories
    base_dir = Path(args.base_dir) if args.base_dir else Path.cwd()
    binary_dir = base_dir / "compiled_binaries"
    output_dir = base_dir / "generated_datasets"

    if not binary_dir.exists():
        print(f"Error: Binary directory {binary_dir} does not exist")
        print("Please run the extraction and compilation script first")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Get all problem directories
    problem_dirs = []
    for item in binary_dir.iterdir():
        if item.is_dir() and item.name.startswith("problem_"):
            problem_dirs.append(item)

    problem_dirs.sort(key=lambda x: int(x.name.split("_")[1]))

    if not problem_dirs:
        print(f"Error: No problem directories found in {binary_dir}")
        sys.exit(1)

    print("🚀 Starting dataset generation")
    print(f"Target: {args.n_datasets} datasets per problem")
    print(f"Minimum validators: {args.min_validators}")
    print(f"Problems found: {len(problem_dirs)}")
    print(f"Output directory: {output_dir}")

    # Process problems in parallel
    total_datasets = 0
    successful_problems = 0

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers_problems) as executor:
        futures = {
            executor.submit(
                generate_datasets_for_problem,
                problem_dir,
                binary_dir,
                output_dir,
                args.n_datasets,
                args.min_validators,
                args.workers_generators,
            ): problem_dir
            for problem_dir in problem_dirs
        }
        for future in as_completed(futures):
            datasets_generated, _ = future.result()
            total_datasets += datasets_generated
            if datasets_generated > 0:
                successful_problems += 1

    end_time = time.time()
    duration = end_time - start_time

    # Final summary
    print("\n🎯 FINAL SUMMARY")
    print(f"{'=' * 50}")
    print(f"Problems processed: {len(problem_dirs)}")
    print(f"Problems with datasets: {successful_problems}")
    print(f"Total datasets generated: {total_datasets}")
    print(f"Target datasets: {len(problem_dirs) * args.n_datasets}")
    print(f"Success rate: {total_datasets / (len(problem_dirs) * args.n_datasets) * 100:.1f}%")
    print(f"Time taken: {duration:.1f} seconds")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

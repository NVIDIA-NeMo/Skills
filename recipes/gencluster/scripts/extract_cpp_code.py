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
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nemo_skills.code_execution.sandbox import LocalSandbox


def extract_final_cpp_block(text):
    """Extract the final C++ code block from text using the provided pattern"""
    pattern = r"```(?:cpp|Cpp)\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1] if matches else ""


def wait_for_sandbox(sandbox, loop, timeout: int = 240, poll: float = 1.0):
    """Block until the sandbox responds to a simple shell command."""
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            result, _ = loop.run_until_complete(sandbox.execute_code("echo hello world", language="shell", timeout=10))
            if result.get("stdout", "").strip() == "hello world":
                return
        except Exception:
            pass
        # simple sleep
        loop.run_until_complete(asyncio.sleep(poll))
    raise RuntimeError(f"Sandbox not ready after waiting {timeout}s")


def compile_cpp_file(cpp_file_path, binary_dir, sandbox, loop):
    """Compile a C++ file inside the sandbox and return compilation status"""
    cpp_file = Path(cpp_file_path)
    binary_dir = Path(binary_dir)
    binary_name = cpp_file.stem
    binary_path = binary_dir / binary_name

    # Ensure binary directory exists (use Python, not sandbox)
    binary_dir.mkdir(parents=True, exist_ok=True)

    # Compile using gnu++17 similar to evaluators
    compile_cmd = f"g++ -std=gnu++17 -O2 -pipe -s -o {binary_path} {cpp_file}"
    try:
        result, _ = loop.run_until_complete(sandbox.execute_code(compile_cmd, language="shell", timeout=120))
        stderr = result.get("stderr", "")
        if stderr.strip():
            return False, "Compilation failed", stderr
        return True, "Success", ""
    except Exception as e:
        return False, "Error", str(e)


def process_jsonl_file(jsonl_path, output_dir, binary_dir, folder_name, source_id, sandbox, loop):
    """Process a single JSONL file; use per-line 'id' to organize outputs."""
    extracted_count = 0
    compiled_count = 0
    compilation_results = []
    split_dir = "gen" if folder_name == "generators" else "val"

    try:
        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, 1):
                try:
                    # Parse JSON line
                    data = json.loads(line.strip())
                    generation = data.get("generation", "")
                    pid = str(data.get("id", "")).strip()

                    if generation and pid:
                        # Ensure per-problem directories exist using per-line id
                        problem_dir = output_dir / f"problem_{pid}"
                        problem_type_dir = problem_dir / split_dir
                        problem_type_dir.mkdir(parents=True, exist_ok=True)

                        binary_problem_dir = binary_dir / f"problem_{pid}"
                        binary_type_dir = binary_problem_dir / split_dir
                        binary_type_dir.mkdir(parents=True, exist_ok=True)

                        # Create organized filename including source to avoid collisions across rs0/rs1
                        output_filename = f"{source_id}_line_{line_num}.cpp"
                        output_path = problem_type_dir / output_filename
                        binary_exe_path = binary_type_dir / Path(output_filename).stem

                        # Extract C++ code only if file doesn't already exist
                        cpp_code = extract_final_cpp_block(generation)
                        relative_path = f"problem_{pid}/{split_dir}/{output_filename}"

                        if output_path.exists():
                            print(f"Skip extract (exists): {relative_path}")
                        else:
                            if cpp_code.strip():
                                with open(output_path, "w", encoding="utf-8") as cpp_file:
                                    cpp_file.write(cpp_code.strip())
                                extracted_count += 1
                                print(f"Extracted C++ code to: {relative_path}")
                            else:
                                # No code and nothing to extract; skip compilation as well
                                print(f"No code found for: {relative_path}, skipping")
                                continue

                        # Compile only if binary does not already exist
                        if binary_exe_path.exists():
                            compilation_results.append(
                                {
                                    "file": relative_path,
                                    "success": True,
                                    "status": "Skipped (binary exists)",
                                    "error": "",
                                }
                            )
                            compiled_count += 1
                            print("  ✓ Skip compile (exists)")
                        else:
                            success, status, error_msg = compile_cpp_file(output_path, binary_type_dir, sandbox, loop)
                            compilation_results.append(
                                {
                                    "file": relative_path,
                                    "success": success,
                                    "status": status,
                                    "error": error_msg,
                                }
                            )

                            if success:
                                compiled_count += 1
                                print("  ✓ Compiled successfully")
                            else:
                                print(f"  ✗ Compilation failed: {status}")
                                if error_msg.strip():
                                    print(f"    Error: {error_msg.strip()[:100]}...")  # First 100 chars

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON in {jsonl_path} line {line_num}: {e}")
                except Exception as e:
                    print(f"Error processing line {line_num} in {jsonl_path}: {e}")

    except Exception as e:
        print(f"Error reading file {jsonl_path}: {e}")

    return extracted_count, compiled_count, compilation_results


def main():
    """Extract C++ blocks from generation outputs and compile them in parallel."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Extract and compile C++ code from JSONL under generators/ and validators/"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Input directory containing generators/ and validators/ folders",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=(os.cpu_count() or 4),
        help="Number of parallel worker threads",
    )
    args = parser.parse_args()

    # Base directory
    base_dir = Path(args.input_dir)

    # Output directories
    output_dir = base_dir / "extracted_cpp"
    binary_dir = base_dir / "compiled_binaries"
    output_dir.mkdir(exist_ok=True)
    binary_dir.mkdir(exist_ok=True)

    # Statistics tracking
    start_time = time.time()
    total_extracted = 0
    total_compiled = 0
    processed_files = 0
    all_compilation_results = []
    failed_compilations = []

    # Initialize sandbox and verify compiler availability inside sandbox
    worker_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(worker_loop)
    sandbox = LocalSandbox()
    wait_for_sandbox(sandbox, worker_loop)
    try:
        result, _ = worker_loop.run_until_complete(sandbox.execute_code("g++ --version", language="shell", timeout=30))
        if result.get("stderr", "").strip():
            print("✗ g++ not available inside sandbox")
            return
        print("✓ g++ compiler found in sandbox")
    except Exception:
        print("✗ Failed to verify g++ inside sandbox")
        return
    finally:
        try:
            worker_loop.run_until_complete(sandbox.close())
        except Exception:
            pass
        try:
            worker_loop.close()
        except Exception:
            pass

    # Build task list across both generators and validators
    tasks = []  # list of tuples (folder_name, Path)
    for folder_name in ["generators", "validators"]:
        folder_path = base_dir / folder_name

        if not folder_path.exists():
            print(f"Folder {folder_name} does not exist, skipping...")
            continue

        print(f"\n=== Scanning {folder_name} ===")

        jsonl_files = sorted(folder_path.rglob("*.jsonl"), key=lambda p: str(p))
        if not jsonl_files:
            print(f"No JSONL files found in: {folder_path}")
            continue
        for fpath in jsonl_files:
            tasks.append((folder_name, fpath))

    # Define per-file worker that owns its own loop and sandbox
    def _process_file(folder_name, fpath: Path):
        """Process one JSONL file using a worker-local sandbox instance."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        sb = LocalSandbox()
        try:
            # best-effort readiness
            try:
                wait_for_sandbox(sb, loop, timeout=60, poll=1.0)
            except Exception:
                pass

            rs = fpath.stem
            extracted, compiled, compilation_results = process_jsonl_file(
                fpath,
                output_dir,
                binary_dir,
                folder_name,
                rs,
                sb,
                loop,
            )
            rel = fpath.relative_to(base_dir)
            return str(rel), extracted, compiled, compilation_results
        finally:
            try:
                loop.run_until_complete(sb.close())
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass

    # Run tasks in a thread pool
    if tasks:
        print(f"\n=== Running {len(tasks)} files with {args.workers} workers ===")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {executor.submit(_process_file, folder, path): (folder, path) for folder, path in tasks}
            for future in as_completed(future_to_task):
                rel, extracted, compiled, compilation_results = future.result()
                print(f"\nProcessing: {rel}")

                total_extracted += extracted
                total_compiled += compiled
                processed_files += 1
                all_compilation_results.extend(compilation_results)

                failed_in_file = [r for r in compilation_results if not r["success"]]
                failed_compilations.extend(failed_in_file)

                print(f"  -> Extracted {extracted} C++ files, compiled {compiled}/{extracted} successfully")
                if failed_in_file:
                    print(f"  -> {len(failed_in_file)} compilation failures")

    # Final summary
    print("\n=== FINAL SUMMARY ===")
    print(f"Processed JSONL files: {processed_files}")
    print(f"Total C++ files extracted: {total_extracted}")
    print(f"Total C++ files compiled successfully: {total_compiled}")
    print(f"Compilation success rate: {total_compiled / total_extracted * 100:.1f}%" if total_extracted > 0 else "N/A")
    print("\nDirectory structure:")
    print(f"  {output_dir}/")
    print("    problem_*/")
    print("      gen/        (from generators)")
    print("      val/        (from validators)")
    print(f"  {binary_dir}/")
    print("    problem_*/")
    print("      gen/        (compiled binaries)")
    print("      val/        (compiled binaries)")

    # Detailed failure analysis
    if failed_compilations:
        print(f"\n=== COMPILATION FAILURES ({len(failed_compilations)}) ===")
        failure_types = {}
        for failure in failed_compilations:
            status = failure["status"]
            failure_types[status] = failure_types.get(status, 0) + 1

        for failure_type, count in sorted(failure_types.items()):
            print(f"  {failure_type}: {count} files")

        # Show first few detailed errors
        print("\nFirst 5 compilation errors:")
        for i, failure in enumerate(failed_compilations[:5]):
            print(f"  {i + 1}. {failure['file']}: {failure['status']}")
            if failure["error"].strip():
                error_lines = failure["error"].strip().split("\n")
                for line in error_lines[:2]:  # Show first 2 lines of error
                    print(f"     {line}")
                if len(error_lines) > 2:
                    print(f"     ... (and {len(error_lines) - 2} more lines)")
    else:
        print("\n🎉 All files compiled successfully!")

    # Runtime summary
    elapsed = time.time() - start_time
    mm, ss = divmod(int(elapsed), 60)
    hh, mm = divmod(mm, 60)
    human = (f"{hh}h {mm}m {ss}s" if hh else f"{mm}m {ss}s") if mm or hh else f"{int(elapsed)}s"
    print(f"\nTime spent: {human} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()

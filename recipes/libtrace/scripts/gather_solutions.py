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

"""
Analyze and sample solutions, including library-usage checks.

Inputs: output-rs*.jsonl from boxed inference.
Outputs: stats to stdout, and sampled JSONL files when mode=sample.

Example (stats):
  python /nemo_run/code/recipes/libtrace/scripts/gather_solutions.py \
    --mode stats --input_dir /workspace/libtrace-results/boxed-inference-chem/results \
    --dataset_name chem --require_boxed

Example (sample):
  python /nemo_run/code/recipes/libtrace/scripts/gather_solutions.py \
    --mode sample --input_dir /workspace/libtrace-results/boxed-inference-chem/results \
    --dataset_name chem --output_dir /workspace/libtrace-results/gather-solutions-chem/results/sampled \
    --target 240000 --require_boxed
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json as _json_std
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:  # orjson is significantly faster; fallback to std json
    import orjson as _orjson  # type: ignore

    def _json_loads(s: str):
        return _orjson.loads(s)

    def _json_dumps(obj) -> str:
        return _orjson.dumps(obj).decode("utf-8")

except Exception:  # pragma: no cover - best effort
    _orjson = None

    def _json_loads(s: str):
        return _json_std.loads(s)

    def _json_dumps(obj) -> str:
        return _json_std.dumps(obj)


try:
    _tqdm = importlib.import_module("tqdm").tqdm
except Exception:
    _tqdm = None


def _progress(iterable, **kwargs):
    return _tqdm(iterable, **kwargs) if _tqdm else iterable


DEFAULT_USER_PROMPT_PREFIX = (
    "Solve the following problem. Make sure to put the answer (and only answer) inside \\boxed{}.\n\n"
)

OUTPUT_PATTERN_RE = re.compile(
    r"<\|start\|>"  # start sentinel already prefixed for assistant output concat
    r"(?P<source_info>.*?)"
    r"<\|message\|>"
    r"(?P<content>.*?)"
    r"(?P<terminator><\|end\|>|<\|call\|>|<\|return\|>)",
    re.DOTALL,
)


@dataclass
class Statistics:
    total_solutions: int
    total_solutions_without_code_executions: int
    used_given_name: int
    used_any_name: int


def parse_messages(
    problem: str,
    generation: str,
    user_prompt_prefix: str,
    dump_json: bool,
) -> list[dict]:
    messages: list[dict] = []
    tool_call_counter = 0
    assistant_reasoning_buffer: dict | None = None

    full_interaction_str = (
        "<|start|>user<|message|>"
        + user_prompt_prefix
        + problem
        + "<|end|><|start|>assistant"
        + generation
        + "<|return|>"
    )
    matches = list(OUTPUT_PATTERN_RE.finditer(full_interaction_str))

    for i, match in enumerate(matches):
        parts = match.groupdict()
        source_info = parts["source_info"].strip()
        content = parts["content"].strip()
        terminator = parts["terminator"]

        if "system" in source_info:
            messages.append({"role": "system", "content": content})
            continue

        if "user" in source_info:
            messages.append({"role": "user", "content": content})
            continue

        if assistant_reasoning_buffer and "to=python code" not in source_info:
            messages.append(assistant_reasoning_buffer)
            assistant_reasoning_buffer = None

        if "to=assistant" in source_info:  # tool response
            tool_name = "stateful_python_code_exec"
            tool_call_id = f"call_{tool_call_counter - 1}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": content,
                }
            )
        else:  # assistant side
            if "to=python code" in source_info:  # tool call
                if not assistant_reasoning_buffer:
                    assistant_reasoning_buffer = {"role": "assistant"}
                tool_call_id = f"call_{tool_call_counter}"
                tool_name = "stateful_python_code_exec"
                assistant_reasoning_buffer.setdefault("tool_calls", []).append(
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": _json_dumps({"code": content}) if dump_json else {"code": content},
                        },
                    }
                )
                messages.append(assistant_reasoning_buffer)
                assistant_reasoning_buffer = None
                tool_call_counter += 1
            else:  # reasoning / final answer
                is_final_answer = (terminator == "<|return|>") or (i == len(matches) - 1 and terminator != "<|call|>")
                if is_final_answer:
                    messages.append({"role": "assistant", "content": content})
                else:
                    assistant_reasoning_buffer = {
                        "role": "assistant",
                        "reasoning_content": content,
                    }

    if assistant_reasoning_buffer:  # flush
        messages.append(assistant_reasoning_buffer)

    if (
        len(messages) >= 2
        and messages[-1]["role"] == "assistant"
        and "content" in messages[-1]
        and messages[-2]["role"] == "assistant"
        and "reasoning_content" in messages[-2]
        and "tool_calls" not in messages[-2]
    ):
        messages[-1]["reasoning_content"] = messages[-2]["reasoning_content"]
        messages.pop(-2)

    for msg in messages:
        if msg["role"] == "assistant" and "content" not in msg:
            msg["content"] = ""

    return messages


class UsedNamesExtractor(ast.NodeVisitor):
    """
    AST visitor that extracts ALL fully-qualified names used in the code.
    This allows efficient checking against multiple target names by parsing once.
    """

    def __init__(self):
        self.import_aliases: dict[str, str] = {}
        self.wildcard_imports: set[str] = set()
        self.used_names: set[str] = set()
        self.imported_from_modules: set[str] = set()

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            module_name = alias.name
            local_name = alias.asname if alias.asname else alias.name
            self.import_aliases[local_name] = module_name
            if "." in module_name and alias.asname is None:
                first_component = module_name.split(".")[0]
                if first_component not in self.import_aliases:
                    self.import_aliases[first_component] = first_component
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        if module:
            self.imported_from_modules.add(module)

        for alias in node.names:
            if alias.name == "*":
                self.wildcard_imports.add(module)
            else:
                imported_name = alias.name
                local_name = alias.asname if alias.asname else alias.name
                full_path = f"{module}.{imported_name}" if module else imported_name
                self.import_aliases[local_name] = full_path
        self.generic_visit(node)

    def _get_full_attribute_path(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            local_name = node.id
            if local_name in self.import_aliases:
                return self.import_aliases[local_name]
            return None
        if isinstance(node, ast.Attribute):
            base_path = self._get_full_attribute_path(node.value)
            if base_path is None:
                return None
            return f"{base_path}.{node.attr}"
        if isinstance(node, ast.Call):
            return self._get_full_attribute_path(node.func)
        return None

    def visit_Name(self, node: ast.Name):
        if node.id in self.import_aliases:
            self.used_names.add(self.import_aliases[node.id])
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        full_path = self._get_full_attribute_path(node)
        if full_path:
            self.used_names.add(full_path)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        full_path = self._get_full_attribute_path(node.func)
        if full_path:
            self.used_names.add(full_path)
        self.generic_visit(node)


def _extract_used_names_from_code(code: str) -> tuple[set[str], set[str]]:
    code = code.replace("\x00", "")
    try:
        tree = ast.parse(code)
        extractor = UsedNamesExtractor()
        extractor.visit(tree)
        return extractor.used_names, extractor.imported_from_modules
    except (SyntaxError, ValueError):
        return set(), set()


def _check_name_in_extracted(name: str, used_names: set[str], imported_from_modules: set[str]) -> bool:
    if name in used_names:
        return True
    name_prefix = name + "."
    for used in used_names:
        if used.startswith(name_prefix):
            return True
    parts = name.split(".")
    if len(parts) >= 2:
        module_prefix = ".".join(parts[:-1])
        if module_prefix in imported_from_modules:
            if name in used_names:
                return True
    return False


def _check_any_name_in_extracted(all_names: set[str], used_names: set[str], imported_from_modules: set[str]) -> bool:
    if all_names & used_names:
        return True
    for used in used_names:
        parts = used.split(".")
        for i in range(1, len(parts)):
            prefix = ".".join(parts[:i])
            if prefix in all_names:
                return True
    for name in all_names:
        parts = name.split(".")
        if len(parts) >= 2:
            module_prefix = ".".join(parts[:-1])
            if module_prefix in imported_from_modules and name in used_names:
                return True
    return False


def _find_matching_names(all_names: set[str], used_names: set[str], imported_from_modules: set[str]) -> set[str]:
    matched = set()
    matched.update(all_names & used_names)
    for used in used_names:
        parts = used.split(".")
        for i in range(1, len(parts)):
            prefix = ".".join(parts[:i])
            if prefix in all_names:
                matched.add(prefix)
    for name in all_names & used_names:
        parts = name.split(".")
        if len(parts) >= 2:
            module_prefix = ".".join(parts[:-1])
            if module_prefix in imported_from_modules:
                matched.add(name)
    return matched


def _extract_all_code_from_messages(messages: list, dump_json: bool) -> str:
    all_code_blocks = []
    for msg in messages:
        if msg["role"] == "assistant" and "tool_calls" in msg:
            for tool_call in msg["tool_calls"]:
                if tool_call["function"]["name"] == "stateful_python_code_exec":
                    args = tool_call["function"]["arguments"]
                    if dump_json:
                        code = _json_loads(args)["code"]
                    else:
                        code = args["code"]
                    all_code_blocks.append(code)
    return "\n".join(all_code_blocks)


def _has_boxed_in_last_assistant(messages: list) -> bool:
    for msg in reversed(messages):
        if msg["role"] == "assistant" and msg.get("content"):
            return r"\boxed{" in msg["content"] or "\\boxed{" in msg["content"]
    return False


def resolve_input_files(input_dir: str | None, input_files: list[str] | None, pattern: str) -> list[Path]:
    if input_dir is None and input_files is None:
        raise ValueError("Provide --input_dir or --input_files.")
    if input_dir is not None and input_files is not None:
        raise ValueError("Use only one of --input_dir or --input_files.")

    if input_dir is not None:
        input_path = Path(input_dir)
        files = sorted(input_path.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matched {input_path / pattern}")
        return files

    for path in input_files:
        if not Path(path).exists():
            raise FileNotFoundError(f"Input file not found: {path}")
    return [Path(path) for path in input_files]


def _get_code_rounds(data: dict, field: str, require_field: bool) -> int:
    if field not in data:
        if require_field:
            raise KeyError(f"Missing '{field}' in record.")
        return 0
    value = data[field]
    if not isinstance(value, int):
        raise TypeError(f"Expected '{field}' to be int, got {type(value)}")
    return value


def _weighted_sample_indices(weights: list[float], k: int, rng: random.Random) -> list[int]:
    if k <= 0:
        return []
    positive = [(idx, w) for idx, w in enumerate(weights) if w > 0]
    if not positive:
        return rng.sample(range(len(weights)), k)
    if k >= len(positive):
        return [idx for idx, _ in positive]

    scored = []
    for idx, w in positive:
        u = rng.random()
        score = u ** (1.0 / w)
        scored.append((score, idx))
    scored.sort(reverse=True)
    return [idx for _, idx in scored[:k]]


_ALL_NAMES: set[str] = set()
_USER_PROMPT_PREFIX = DEFAULT_USER_PROMPT_PREFIX
_DUMP_JSON = True
_CODE_ROUNDS_FIELD = "code_rounds_executed"
_REQUIRE_CODE_ROUNDS = True
_REQUIRE_BOXED = True


def _init_worker(
    all_names: set[str],
    user_prompt_prefix: str,
    dump_json: bool,
    code_rounds_field: str,
    require_code_rounds: bool,
    require_boxed: bool,
):
    global _ALL_NAMES
    global _USER_PROMPT_PREFIX
    global _DUMP_JSON
    global _CODE_ROUNDS_FIELD
    global _REQUIRE_CODE_ROUNDS
    global _REQUIRE_BOXED

    _ALL_NAMES = all_names
    _USER_PROMPT_PREFIX = user_prompt_prefix
    _DUMP_JSON = dump_json
    _CODE_ROUNDS_FIELD = code_rounds_field
    _REQUIRE_CODE_ROUNDS = require_code_rounds
    _REQUIRE_BOXED = require_boxed


def process_line(line: str) -> tuple[bool, bool, bool, bool]:
    data = _json_loads(line)
    code_rounds = _get_code_rounds(data, _CODE_ROUNDS_FIELD, _REQUIRE_CODE_ROUNDS)
    if code_rounds == 0:
        return (True, False, False, False)

    messages = parse_messages(data["problem"], data["generation"], _USER_PROMPT_PREFIX, _DUMP_JSON)
    if _REQUIRE_BOXED and not _has_boxed_in_last_assistant(messages):
        return (False, False, False, False)

    combined_code = _extract_all_code_from_messages(messages, dump_json=_DUMP_JSON)
    if not combined_code:
        return (True, True, False, False)

    used_names, imported_from_modules = _extract_used_names_from_code(combined_code)
    used_given_name = _check_name_in_extracted(data["name"], used_names, imported_from_modules)
    used_any_name = _check_any_name_in_extracted(_ALL_NAMES, used_names, imported_from_modules)

    return (True, True, used_given_name, used_any_name)


def process_line_for_sampling(args: tuple[int, str]) -> tuple[int, bool, bool, bool, bool, tuple[str, ...]]:
    line_idx, line = args
    data = _json_loads(line)
    code_rounds = _get_code_rounds(data, _CODE_ROUNDS_FIELD, _REQUIRE_CODE_ROUNDS)
    if code_rounds == 0:
        return (line_idx, True, False, False, False, ())

    messages = parse_messages(data["problem"], data["generation"], _USER_PROMPT_PREFIX, _DUMP_JSON)
    has_boxed = _has_boxed_in_last_assistant(messages)
    if _REQUIRE_BOXED and not has_boxed:
        return (line_idx, True, False, False, False, ())

    combined_code = _extract_all_code_from_messages(messages, dump_json=_DUMP_JSON)
    if not combined_code:
        return (line_idx, True, True, has_boxed, False, ())

    used_names, imported_from_modules = _extract_used_names_from_code(combined_code)
    used_given = _check_name_in_extracted(data["name"], used_names, imported_from_modules)
    matched_names = _find_matching_names(_ALL_NAMES, used_names, imported_from_modules)

    return (line_idx, True, True, has_boxed, used_given, tuple(matched_names))


def run_stats(
    all_lines: list[str],
    all_names: set[str],
    user_prompt_prefix: str,
    dump_json: bool,
    code_rounds_field: str,
    require_code_rounds: bool,
    require_boxed: bool,
    num_workers: int,
) -> Statistics:
    import multiprocessing as mp

    stats = Statistics(
        total_solutions=0,
        total_solutions_without_code_executions=0,
        used_given_name=0,
        used_any_name=0,
    )

    with mp.Pool(
        num_workers,
        initializer=_init_worker,
        initargs=(
            all_names,
            user_prompt_prefix,
            dump_json,
            code_rounds_field,
            require_code_rounds,
            require_boxed,
        ),
    ) as pool:
        results = pool.imap_unordered(process_line, all_lines, chunksize=100)
        for is_valid, has_code_execution, used_given, used_any in _progress(
            results, total=len(all_lines), desc="Processing", unit="line"
        ):
            if not is_valid:
                continue
            stats.total_solutions += 1
            if not has_code_execution:
                stats.total_solutions_without_code_executions += 1
                continue
            if used_given:
                stats.used_given_name += 1
            if used_any:
                stats.used_any_name += 1

    return stats


def save_sampled_messages(
    indices: list[int],
    all_lines: list[str],
    output_file: Path,
    description: str,
    user_prompt_prefix: str,
    dump_json: bool,
    given_name_set: set[int],
) -> None:
    print(f"Saving {description} (dump_json={dump_json}) to {output_file}...")
    indices_set = set(indices)
    with output_file.open("w", encoding="utf-8") as f:
        for idx in _progress(sorted(indices_set), desc=f"Writing {description}", unit="record"):
            line = all_lines[idx]
            data = _json_loads(line)
            messages = parse_messages(data["problem"], data["generation"], user_prompt_prefix, dump_json)
            record = {
                "problem": data["problem"],
                "name": data["name"],
                "messages": messages,
                "used_given_name": idx in given_name_set,
            }
            f.write(_json_dumps(record) + "\n")


def run_sample(
    all_lines: list[str],
    all_names: set[str],
    user_prompt_prefix: str,
    dump_json: bool,
    code_rounds_field: str,
    require_code_rounds: bool,
    require_boxed: bool,
    num_workers: int,
    target_total: int,
    output_dir: str,
    dataset_name: str,
    random_seed: int,
) -> None:
    import multiprocessing as mp

    print(f"\n{'=' * 60}")
    print(f"SAMPLING DATASET: {dataset_name}")
    print(f"Target total: {target_total:,}")
    print(f"{'=' * 60}")

    indexed_lines = list(enumerate(all_lines))
    given_name_indices: list[int] = []
    any_name_only_data: list[tuple[int, tuple[str, ...]]] = []
    name_usage_counts: Counter[str] = Counter()
    skipped_no_boxed = 0

    with mp.Pool(
        num_workers,
        initializer=_init_worker,
        initargs=(
            all_names,
            user_prompt_prefix,
            dump_json,
            code_rounds_field,
            require_code_rounds,
            require_boxed,
        ),
    ) as pool:
        results = pool.imap_unordered(process_line_for_sampling, indexed_lines, chunksize=100)
        for line_idx, is_valid, has_code, has_boxed, used_given, names_used in _progress(
            results, total=len(all_lines), desc="Categorizing", unit="line"
        ):
            if not is_valid or not has_code:
                continue
            if require_boxed and not has_boxed:
                skipped_no_boxed += 1
                continue
            if used_given:
                given_name_indices.append(line_idx)
            elif names_used:
                any_name_only_data.append((line_idx, names_used))
                for name in names_used:
                    name_usage_counts[name] += 1

    if require_boxed:
        print(f"\nSkipped {skipped_no_boxed:,} solutions without \\boxed{{}}")
    print(f"Found {len(given_name_indices):,} solutions using given name")
    print(f"Found {len(any_name_only_data):,} solutions using any name but not given")

    num_given = len(given_name_indices)
    num_to_sample = max(0, target_total - num_given)
    if num_to_sample > len(any_name_only_data):
        print(f"Warning: Not enough any_name records ({len(any_name_only_data):,}) to reach target. Using all.")
        num_to_sample = len(any_name_only_data)

    print(f"Will sample {num_to_sample:,} additional records")

    rng = random.Random(random_seed)
    random_sample_data = rng.sample(any_name_only_data, num_to_sample)
    random_sample_indices = [idx for idx, _ in random_sample_data]

    weights = []
    for _, names_used in any_name_only_data:
        weight = sum(1.0 / name_usage_counts[name] for name in names_used if name_usage_counts[name] > 0)
        weights.append(weight)

    inverse_freq_positions = _weighted_sample_indices(weights, num_to_sample, rng)
    inverse_freq_sample_indices = [any_name_only_data[i][0] for i in inverse_freq_positions]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    random_all_indices = given_name_indices + random_sample_indices
    inverse_all_indices = given_name_indices + inverse_freq_sample_indices
    given_name_set = set(given_name_indices)

    save_sampled_messages(
        random_all_indices,
        all_lines,
        output_path / f"{dataset_name}_random_json.jsonl",
        f"{dataset_name} random+json",
        user_prompt_prefix,
        dump_json=True,
        given_name_set=given_name_set,
    )
    save_sampled_messages(
        random_all_indices,
        all_lines,
        output_path / f"{dataset_name}_random_nojson.jsonl",
        f"{dataset_name} random+nojson",
        user_prompt_prefix,
        dump_json=False,
        given_name_set=given_name_set,
    )
    save_sampled_messages(
        inverse_all_indices,
        all_lines,
        output_path / f"{dataset_name}_inverse_freq_json.jsonl",
        f"{dataset_name} inverse_freq+json",
        user_prompt_prefix,
        dump_json=True,
        given_name_set=given_name_set,
    )
    save_sampled_messages(
        inverse_all_indices,
        all_lines,
        output_path / f"{dataset_name}_inverse_freq_nojson.jsonl",
        f"{dataset_name} inverse_freq+nojson",
        user_prompt_prefix,
        dump_json=False,
        given_name_set=given_name_set,
    )

    print(f"\n{'=' * 60}")
    print(f"COMPLETED: {dataset_name}")
    print(f"{'=' * 60}")
    print(f"Given name records: {len(given_name_indices):,}")
    print(f"Sampled records: {num_to_sample:,}")
    print(f"Total per file: {len(random_all_indices):,}")
    print("Output files:")
    print(f"  {output_path / f'{dataset_name}_random_json.jsonl'}")
    print(f"  {output_path / f'{dataset_name}_random_nojson.jsonl'}")
    print(f"  {output_path / f'{dataset_name}_inverse_freq_json.jsonl'}")
    print(f"  {output_path / f'{dataset_name}_inverse_freq_nojson.jsonl'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gather and analyze solutions.")
    parser.add_argument("--mode", choices=["stats", "sample"], default="stats")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--input_files", type=str, nargs="+", default=None)
    parser.add_argument("--pattern", type=str, default="output-rs*.jsonl")
    parser.add_argument("--dataset_name", type=str, default="dataset")
    parser.add_argument("--target", type=int, required=True, help="Target number of sampled records.")
    parser.add_argument("--output_dir", type=str, default="sampled_solutions")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=None, help="Default: os.cpu_count()//2")
    parser.add_argument(
        "--require_code_rounds",
        action="store_true",
        default=True,
        help="Skip records without code_rounds_executed field (default: True)",
    )
    parser.add_argument("--no_require_code_rounds", action="store_false", dest="require_code_rounds")
    parser.add_argument(
        "--require_boxed", action="store_true", default=True, help="Skip records without \\boxed{} (default: True)"
    )
    parser.add_argument("--no_require_boxed", action="store_false", dest="require_boxed")
    parser.add_argument(
        "--dump_json",
        action="store_true",
        default=True,
        help="Serialize tool_call arguments as JSON strings (default: True)",
    )
    parser.add_argument("--no_dump_json", action="store_false", dest="dump_json")
    parser.add_argument("--user_prompt_prefix", type=str, default=DEFAULT_USER_PROMPT_PREFIX)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_files = resolve_input_files(args.input_dir, args.input_files, args.pattern)
    import os

    num_workers = args.num_workers if args.num_workers is not None else max(1, (os.cpu_count() or 2) // 2)

    print("Reading all lines from files...")
    all_lines: list[str] = []
    for path in input_files:
        with path.open("r", encoding="utf-8") as f:
            all_lines.extend(f.readlines())
    if not all_lines:
        raise RuntimeError("No lines found in input files.")
    print(f"Total lines to process: {len(all_lines):,}")

    print("Gathering all unique names...")
    all_names: set[str] = set()
    for line in _progress(all_lines, desc="Extracting names", unit="line"):
        data = _json_loads(line)
        all_names.add(data["name"])
    if not all_names:
        raise RuntimeError("No names found in input data.")
    print(f"Found {len(all_names):,} unique names to check against")

    if args.mode == "stats":
        stats = run_stats(
            all_lines=all_lines,
            all_names=all_names,
            user_prompt_prefix=args.user_prompt_prefix,
            dump_json=args.dump_json,
            code_rounds_field="code_rounds_executed",
            require_code_rounds=args.require_code_rounds,
            require_boxed=args.require_boxed,
            num_workers=num_workers,
        )

        solutions_with_code = stats.total_solutions - stats.total_solutions_without_code_executions
        print(f"\n{'=' * 60}")
        print(f"RESULTS FOR: {args.dataset_name}")
        print(f"{'=' * 60}")
        print(f"Total solutions: {stats.total_solutions}")
        print(f"Total solutions without code executions: {stats.total_solutions_without_code_executions}")
        print(f"Total solutions with code executions: {solutions_with_code}")
        print(f"{'=' * 60}")
        print(f"Used given name (specific to problem): {stats.used_given_name}")
        print(f"Used any name (from entire dataset): {stats.used_any_name}")
        if solutions_with_code > 0:
            print(f"{'=' * 60}")
            print(f"Percentage used given name: {stats.used_given_name / solutions_with_code * 100:.2f}%")
            print(f"Percentage used any name: {stats.used_any_name / solutions_with_code * 100:.2f}%")
        print(f"{'=' * 60}\n")
        sys.exit(0)

    if args.mode == "sample":
        run_sample(
            all_lines=all_lines,
            all_names=all_names,
            user_prompt_prefix=args.user_prompt_prefix,
            dump_json=args.dump_json,
            code_rounds_field="code_rounds_executed",
            require_code_rounds=args.require_code_rounds,
            require_boxed=args.require_boxed,
            num_workers=num_workers,
            target_total=args.target,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            random_seed=args.random_seed,
        )

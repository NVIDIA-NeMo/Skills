# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Prepare IHEval splits from the zhihz0535/IHEval HF mirror.

Writes one test.jsonl per sub-benchmark; variants are discovered from the repo
file listing. Data is downloaded at prepare time and is not committed.
"""

import argparse
import json
from pathlib import Path

HF_REPO = "zhihz0535/IHEval"
HF_ROOT = "iheval"  # top-level dir inside the HF dataset repo

# split_name -> (category, task, task_id, category_id)
# `category` and `task` are the path segments in the HF layout
# iheval/<category>/<task>/<setting>/<variant>/; `category_id` and `task_id` are
# the normalized labels written to each row's "category" / "task" fields.
SUB_BENCHMARKS = {
    "rule_following_single": ("rule-following", "single-turn", "single", "rule_following"),
    "rule_following_multi": ("rule-following", "multi-turn", "multi", "rule_following"),
    "task_execution_verb_extract": ("task-execution", "verb-extract", "verb_extract", "task_execution"),
    "task_execution_translation": ("task-execution", "translation", "translation", "task_execution"),
    "task_execution_lang_detect": ("task-execution", "lang-detect", "lang_detect", "task_execution"),
    "safety_hijack": ("safety", "user-prompt-hijack", "hijack", "safety"),
    "safety_extract": ("safety", "system-prompt-extract", "extract", "safety"),
    "tool_use_webpage": ("tool-use", "get-webpage", "webpage", "tool_use"),
    "tool_use_slack_user": ("tool-use", "slack-user", "slack_user", "tool_use"),
}

SETTINGS = ("aligned", "conflict", "reference")


def list_variants(repo_files, category, task, setting):
    """Return variant names that have an ``input_data.json`` for this (category, task, setting)."""
    prefix = f"{HF_ROOT}/{category}/{task}/{setting}/"
    variants = set()
    for f in repo_files:
        if f.startswith(prefix) and f.endswith("/input_data.json"):
            variant = f[len(prefix) : -len("/input_data.json")]
            if "/" not in variant:  # exactly one level (the variant dir)
                variants.add(variant)
    return sorted(variants)


def load_input_data(category, task, setting, variant):
    """Download + parse one ``input_data.json`` from the HF dataset repo."""
    from huggingface_hub import hf_hub_download

    rel = f"{HF_ROOT}/{category}/{task}/{setting}/{variant}/input_data.json"
    local = hf_hub_download(repo_id=HF_REPO, filename=rel, repo_type="dataset")
    with open(local, "rt", encoding="utf-8") as fin:
        return json.load(fin)


def build_messages(row, category_id, task_id, setting):
    """Build an OpenAI-style ``messages`` list from one upstream row."""
    system = row.get("system")
    instruction = row.get("instruction", "")

    # Multi-turn rule-following with a real dialog history.
    conv_hist = row.get("conversation_history")
    if category_id == "rule_following" and task_id == "multi" and isinstance(conv_hist, list) and len(conv_hist) >= 2:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        # conversation_history is [user_turn_1, assistant_turn_1] (possibly more pairs).
        pairs = list(zip(conv_hist[0::2], conv_hist[1::2], strict=True))
        for user_msg, assistant_msg in pairs:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": instruction})
        return messages

    # Tool-use with an upstream-supplied tool call/return.
    tool = row.get("tool")
    if category_id == "tool_use" and isinstance(tool, dict) and tool:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": instruction})

        call = tool.get("call") or {}
        ret = tool.get("return") or {}
        call_id = call.get("id") or ret.get("id") or "call_0"
        call_name = call.get("name") or (tool.get("definition") or {}).get("name") or ""
        call_args = call.get("arguments", {})

        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": call_name, "arguments": json.dumps(call_args, ensure_ascii=False)},
                    }
                ],
            }
        )
        messages.append(
            {"role": "tool", "tool_call_id": call_id, "name": call_name, "content": ret.get("content", "")}
        )
        return messages

    # Reference setting with no system prompt — single user message.
    if setting == "reference" or system in (None, ""):
        return [{"role": "user", "content": instruction}]

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
    ]


def _last_user_text(messages):
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
    return ""


def process_sub_benchmark(out_root, repo_files, split_name, spec):
    category, task, task_id, category_id = spec
    out_dir = out_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / "test.jsonl"

    rows_written = 0
    with output_file.open("w", encoding="utf-8") as fout:
        for setting in SETTINGS:
            for variant in list_variants(repo_files, category, task, setting):
                data = load_input_data(category, task, setting, variant)
                for i, row in enumerate(data):
                    messages = build_messages(row, category_id, task_id, setting)
                    answer = row.get("answer")
                    upstream_id = row.get("id", i)
                    record = {
                        "id": f"iheval-{task_id}-{setting}-{variant}-{upstream_id}",
                        "messages": messages,
                        "question": _last_user_text(messages),
                        "setting": setting,
                        "variant": variant,
                        "category": category_id,
                        "task": task_id,
                        "answer": answer,
                        "expected_answer": answer,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    rows_written += 1

    return output_file, rows_written


def main(args):
    from huggingface_hub import HfApi

    out_root = Path(__file__).absolute().parent
    repo_files = HfApi().list_repo_files(HF_REPO, repo_type="dataset")

    selected = dict(SUB_BENCHMARKS)
    if args.only:
        wanted = set(args.only)
        missing = wanted.difference(SUB_BENCHMARKS)
        if missing:
            raise SystemExit(f"Unknown sub-benchmarks: {sorted(missing)}")
        selected = {name: SUB_BENCHMARKS[name] for name in wanted}

    for split_name, spec in selected.items():
        output_file, n = process_sub_benchmark(out_root, repo_files, split_name, spec)
        print(f"[{split_name}] wrote {n} rows -> {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare IHEval test splits from the HuggingFace mirror.")
    parser.add_argument("--split", default="test", choices=("test",), help="Local split name.")
    parser.add_argument(
        "--only", nargs="*", default=None, help="Restrict to these sub-benchmark output dirs (e.g. safety_hijack)."
    )
    args = parser.parse_args()
    main(args)

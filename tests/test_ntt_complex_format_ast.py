# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import importlib.util
import json
from pathlib import Path


def _load_module(name: str, rel_path: str):
    module_path = Path(__file__).parents[1] / rel_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")


def _ast_row(row_id: str, dataset: str, src_lang: str, tgt_lang: str):
    return {
        "id": row_id,
        "task_type": "Multilingual-AST",
        "expected_answer": "hola mundo",
        "source": "hello world",
        "reference": "hola mundo",
        "audio_path": f"/dataset/{dataset}/audio/{row_id}.wav",
        "duration": 1.0,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. /no_think"},
            {
                "role": "user",
                "content": "Please translate the given speech to Spanish.",
                "audio": {"path": f"/dataset/{dataset}/audio/{row_id}.wav", "duration": 1.0},
            },
        ],
        "subset_for_metrics": f"{src_lang}->{tgt_lang}",
        "extra_fields": {
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "src_lang_name": "English",
            "tgt_lang_name": "Spanish",
        },
    }


def test_prepare_format_ast_manifest(tmp_path):
    prepare = _load_module("ntt_complex_prepare", "nemo_skills/dataset/ntt-complex/prepare.py")
    source_root = tmp_path / "source"
    output_root = tmp_path / "out"
    _write_jsonl(source_root / "fleurs" / "st" / "test.jsonl", [_ast_row("fleurs-1", "fleurs", "en_us", "es_419")])
    _write_jsonl(source_root / "covost2" / "ast" / "test.jsonl", [_ast_row("covost2-1", "covost2", "en", "es")])

    prepare.prepare_ntt_complex(
        data_dir=output_root,
        source_data_dir=source_root,
        samples_per_source=1,
        sources=["fleurs", "covost2"],
    )

    rows = [
        json.loads(line)
        for line in (output_root / "format_ast" / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 6
    assert {row["format_id"] for row in rows} == {"json_object", "srt_single_cue", "markdown_table"}
    assert {row["origin_dataset"] for row in rows} == {"fleurs", "covost2"}
    assert all(row["task_type"] == "Format-AST" for row in rows)
    assert all(row["ntt_complex_subtest"] == "format_ast" for row in rows)
    assert all(row["messages"][-1]["audio"]["path"].startswith("/dataset/") for row in rows)


def test_format_ast_evaluator_extracts_and_scores():
    eval_mod = _load_module("ntt_complex_eval", "nemo_skills/dataset/ntt-complex/ntt_complex_eval.py")
    evaluator = eval_mod.NTTComplexEvaluator({"strip_helpful_prefixes": True}, num_parallel_requests=1)
    sample = _ast_row("sample-1", "fleurs", "en_us", "es_419")
    sample.update(
        {
            "task_type": "Format-AST",
            "format_id": "json_object",
            "generation": json.dumps(
                {
                    "source_language": "English",
                    "target_language": "Spanish",
                    "translation": "hola mundo",
                }
            ),
        }
    )

    result = asyncio.run(evaluator.eval_single(sample))

    assert result["format_valid"] is True
    assert result["format_ast_is_correct"] is True
    assert result["extracted_translation"] == "hola mundo"
    assert result["pred_text"] == "hola mundo"


def test_format_ast_evaluator_flags_invalid_format():
    eval_mod = _load_module("ntt_complex_eval_invalid", "nemo_skills/dataset/ntt-complex/ntt_complex_eval.py")
    evaluator = eval_mod.NTTComplexEvaluator({}, num_parallel_requests=1)
    sample = _ast_row("sample-2", "covost2", "en", "es")
    sample.update(
        {
            "task_type": "Format-AST",
            "format_id": "markdown_table",
            "generation": "hola mundo",
        }
    )

    result = asyncio.run(evaluator.eval_single(sample))

    assert result["format_valid"] is False
    assert result["format_ast_is_correct"] is False
    assert result["format_error"] == "too_few_markdown_rows"

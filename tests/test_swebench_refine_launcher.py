import importlib.util
import json
from pathlib import Path

import pytest

LAUNCHER_PATH = (
    Path(__file__).parents[1]
    / "nemo_skills"
    / "inference"
    / "eval"
    / "launchers"
    / "swebench_refine"
    / "merge_results.py"
)


def _load_merge_module():
    spec = importlib.util.spec_from_file_location("swebench_refine_merge_results", LAUNCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_merge_results_validates_chunks_and_preserves_input_order(tmp_path, monkeypatch):
    merge_results = _load_merge_module()
    input_file = tmp_path / "input.jsonl"
    run_dir = tmp_path / "run"
    output_dir = run_dir / "eval-results" / "swe-bench"
    output_dir.mkdir(parents=True)
    _write_jsonl(input_file, [{"instance_id": "a"}, {"instance_id": "b"}, {"instance_id": "c"}])
    _write_jsonl(
        output_dir / "output_chunk_0.jsonl",
        [
            {"instance_id": "a", "swe-bench-metrics": {"resolved": True}},
            {"instance_id": "b", "swe-bench-metrics": {"resolved": False}},
        ],
    )
    _write_jsonl(
        output_dir / "output_chunk_1.jsonl",
        [{"instance_id": "c", "swe-bench-metrics": {"resolved": True}}],
    )
    (output_dir / "output_chunk_0.jsonl.done").touch()
    (output_dir / "output_chunk_1.jsonl.done").touch()
    monkeypatch.setenv("RUN_DIR", str(run_dir))
    monkeypatch.setenv("INPUT_FILE", str(input_file))
    monkeypatch.setenv("NUM_CHUNKS", "2")

    merge_results.main()

    merged = [json.loads(line) for line in (output_dir / "output.jsonl").read_text().splitlines()]
    assert [row["instance_id"] for row in merged] == ["a", "b", "c"]


def test_merge_results_refuses_missing_done_marker(tmp_path, monkeypatch):
    merge_results = _load_merge_module()
    input_file = tmp_path / "input.jsonl"
    run_dir = tmp_path / "run"
    output_dir = run_dir / "eval-results" / "swe-bench"
    output_dir.mkdir(parents=True)
    _write_jsonl(input_file, [{"instance_id": "a"}])
    _write_jsonl(output_dir / "output_chunk_0.jsonl", [{"instance_id": "a"}])
    monkeypatch.setenv("RUN_DIR", str(run_dir))
    monkeypatch.setenv("INPUT_FILE", str(input_file))
    monkeypatch.setenv("NUM_CHUNKS", "1")

    with pytest.raises(RuntimeError, match="missing done marker"):
        merge_results.main()

#!/usr/bin/env python3
import json
import os
from pathlib import Path


def get_instance_id(row: dict) -> str | None:
    return row.get("instance_id") or (row.get("swe-bench-outputs") or {}).get("instance_id")


def main() -> None:
    run_dir = Path(os.environ["RUN_DIR"])
    input_file = Path(os.environ["INPUT_FILE"])
    num_chunks = int(os.environ.get("NUM_CHUNKS", "8"))
    output_dir = run_dir / "eval-results/swe-bench"

    input_rows = [json.loads(line) for line in input_file.read_text().splitlines() if line.strip()]
    expected_ids = [get_instance_id(row) for row in input_rows]
    if not all(expected_ids):
        raise ValueError("input contains rows without instance_id")
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("input contains duplicate instance IDs")

    base_size, remainder = divmod(len(expected_ids), num_chunks)
    expected_counts = [base_size + (chunk_id < remainder) for chunk_id in range(num_chunks)]
    rows = []
    errors = []
    for chunk_id, expected_count in enumerate(expected_counts):
        chunk_path = output_dir / f"output_chunk_{chunk_id}.jsonl"
        done_path = output_dir / f"output_chunk_{chunk_id}.jsonl.done"
        if not done_path.exists():
            errors.append(f"chunk {chunk_id}: missing done marker")
        if not chunk_path.exists():
            errors.append(f"chunk {chunk_id}: missing output")
            continue
        chunk_rows = [json.loads(line) for line in chunk_path.read_text().splitlines() if line.strip()]
        if len(chunk_rows) != expected_count:
            errors.append(f"chunk {chunk_id}: {len(chunk_rows)} rows, expected {expected_count}")
        rows.extend(chunk_rows)

    actual_ids = [get_instance_id(row) for row in rows]
    if len(rows) != len(expected_ids):
        errors.append(f"merged rows: {len(rows)}, expected {len(expected_ids)}")
    if len(set(actual_ids)) != len(actual_ids):
        errors.append("duplicate instance IDs in merged rows")
    if set(actual_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(actual_ids))
        extra = sorted(set(actual_ids) - set(expected_ids))
        errors.append(f"instance ID mismatch: missing={missing}, extra={extra}")
    if errors:
        raise RuntimeError("\n".join(errors))

    by_id = {get_instance_id(row): row for row in rows}
    output_path = output_dir / "output.jsonl"
    temporary = output_path.with_suffix(f".jsonl.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as output:
        for instance_id in expected_ids:
            output.write(json.dumps(by_id[instance_id]) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(output_path)

    resolved = sum(bool(row.get("swe-bench-metrics", {}).get("resolved")) for row in rows)
    print(f"MERGED {len(rows)} rows -> {output_path}")
    print(f"PASS {resolved}/{len(rows)} = {resolved / len(rows):.1%}")


if __name__ == "__main__":
    main()

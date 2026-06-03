import json
import os

from utils import load_jsonl, write_jsonl


def _pool_path(pool_dir, source_name, problem_idx):
    return os.path.join(pool_dir, source_name, f"{problem_idx}.jsonl")


def _load_pool_records(path):
    if not os.path.exists(path):
        return []
    records, parse_stats = load_jsonl(path, return_stats=True, skip_bad_lines=True)
    if parse_stats["num_rows_bad"]:
        print(f"[_load_pool_records] Skipped {parse_stats['num_rows_bad']} malformed rows from {path}")
    return records


def _write_pool_records(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


class ProofPoolManager:
    def __init__(self, pool_dir, solved_threshold=0.99999):
        self.pool_dir = pool_dir
        self.solved_threshold = solved_threshold

    @staticmethod
    def best_record(records):
        if not records:
            return None

        def sort_key(record):
            self_eval = record.get("self_eval", {})
            if isinstance(self_eval, dict):
                self_eval_score = self_eval.get("self_eval_score", 0)
            else:
                self_eval_score = 0
            return (
                record.get("meanscore", 0),
                self_eval_score,
                record.get("round_idx", -1),
                record.get("proof_id", -1),
            )

        return max(records, key=sort_key)

    def load_pool(self, source_name, problem_idx):
        return _load_pool_records(_pool_path(self.pool_dir, source_name, problem_idx))

    def ingest_new_records(self, source_name, problem_idx, new_records, round_idx):
        pool_path = _pool_path(self.pool_dir, source_name, problem_idx)
        existing = _load_pool_records(pool_path)
        proof_dedup = set()
        proof_id_dedup = set()
        for record in existing:
            proof_dedup.add(record.get("proof"))
            if "proof_id" in record:
                proof_id_dedup.add(record["proof_id"])

        next_proof_id = max(proof_id_dedup) + 1 if proof_id_dedup else 1
        to_append = []
        for record in new_records:
            proof = record["proof"]
            if proof in proof_dedup:
                continue
            proof_dedup.add(proof)
            record = dict(record)
            record["proof_id"] = next_proof_id
            record["round_idx"] = round_idx
            next_proof_id += 1
            to_append.append(record)

        if to_append:
            _write_pool_records(pool_path, to_append)

        return existing + to_append

    def is_solved(self, records):
        for record in records:
            if record.get("meanscore", 0) > self.solved_threshold:
                return True
        return False

    def snapshot_pool(self, snapshot_dir):
        for root, _, files in os.walk(self.pool_dir):
            for name in files:
                if not name.endswith(".jsonl"):
                    continue
                source_name = os.path.basename(root)
                src_path = os.path.join(root, name)
                records = _load_pool_records(src_path)
                dest_path = os.path.join(snapshot_dir, source_name, name)
                write_jsonl(dest_path, records)

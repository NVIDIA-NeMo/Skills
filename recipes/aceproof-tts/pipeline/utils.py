import hashlib
import json
import os

import regex

_SYSTEM_PROMPT_CACHE = {}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_data(path):
    if path.endswith(".jsonl"):
        return load_jsonl(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _escaped_prefix(text, max_chars=200):
    return text[:max_chars].encode("unicode_escape", errors="ignore").decode("ascii")


def load_jsonl(path, return_stats=False, skip_bad_lines=True, max_bad_line_examples=5):
    items = []
    stats = {
        "path": path,
        "num_lines_total": 0,
        "num_lines_empty": 0,
        "num_rows_parsed": 0,
        "num_rows_bad": 0,
        "num_rows_skipped": 0,
        "bad_line_examples": [],
    }
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            stats["num_lines_total"] += 1
            line = line.strip()
            if not line:
                stats["num_lines_empty"] += 1
                continue
            # Fast reject on clearly non-JSON prefixes to avoid expensive parser work on corrupt blobs.
            if line[0] not in "{[":
                stats["num_rows_bad"] += 1
                stats["num_rows_skipped"] += 1
                if len(stats["bad_line_examples"]) < max_bad_line_examples:
                    stats["bad_line_examples"].append(
                        {
                            "line_num": line_num,
                            "error": "invalid_prefix",
                            "line_length": len(line),
                            "line_prefix_escaped": _escaped_prefix(line),
                        }
                    )
                if not skip_bad_lines:
                    raise ValueError(f"Invalid JSON prefix at {path}:{line_num}")
                continue
            try:
                items.append(json.loads(line))
                stats["num_rows_parsed"] += 1
            except Exception as exc:
                stats["num_rows_bad"] += 1
                stats["num_rows_skipped"] += 1
                if len(stats["bad_line_examples"]) < max_bad_line_examples:
                    stats["bad_line_examples"].append(
                        {
                            "line_num": line_num,
                            "error": f"{type(exc).__name__}: {exc}",
                            "line_length": len(line),
                            "line_prefix_escaped": _escaped_prefix(line),
                        }
                    )
                if not skip_bad_lines:
                    raise
    denom = stats["num_rows_parsed"] + stats["num_rows_bad"]
    stats["bad_row_ratio"] = (stats["num_rows_bad"] / denom) if denom else 0.0
    if return_stats:
        return items, stats
    return items


def write_json(path, data):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def write_jsonl(path, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def hash_problem_idx(question):
    return hashlib.sha256(question.encode()).hexdigest()


def extract_boxed_answers(text):
    answers = []
    for piece in text.split("boxed{")[1:]:
        n = 0
        for i, ch in enumerate(piece):
            if ch == "{":
                n += 1
            elif ch == "}":
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == "%":
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    return answers


def _normalize_prover_output(text):
    text = text.strip()
    text = regex.sub(r"(^|\n)\s*\*+\s*Solution\s*\*+\s*\n", "\n## Solution\n", text)
    text = regex.sub(r"\n\s*\*+\s*Self Evaluation\s*\*+\s*\n", "\n## Self Evaluation\n", text)
    text = regex.sub(r"(^|\n)## Solution\s*\n", "\n## Solution\n", text)
    text = regex.sub(r"\n## Self Evaluation\s*\n", "\n## Self Evaluation\n", text)
    return text.strip()


def extract_solution(text):
    text = _normalize_prover_output(text)
    return regex.split(r"## Solution\s*\n", regex.split(r"\n## Self Evaluation\s*\n", text)[0])[1].strip()


def extract_self_eval(text):
    text = _normalize_prover_output(text)
    return regex.split(r"\n## Self Evaluation\s*\n", text)[1].strip()


def strip_think(text):
    if text is None:
        return ""
    text = text.strip()
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


RESPONSE_METADATA_KEYS = (
    "reasoning_content",
    "num_generated_tokens",
    "num_reasoning_tokens",
    "num_answer_tokens",
)


def response_metadata(response, prefix):
    return {f"{prefix}_{key}": response[key] for key in RESPONSE_METADATA_KEYS if response.get(key) is not None}


def drop_response_metadata(row):
    row = dict(row)
    suffixes = tuple(f"_{key}" for key in RESPONSE_METADATA_KEYS)
    for key in list(row):
        if key in RESPONSE_METADATA_KEYS or key.endswith(suffixes):
            row.pop(key, None)
    return row


def parse_verification_score(text):
    if text is None:
        return None
    text = strip_think(text)
    scores = [s.strip() for s in extract_boxed_answers(text) if s.strip()]
    if not scores:
        return None
    try:
        score = float(scores[-1])
    except ValueError:
        return None
    if score in (0.0, 0.5, 1.0):
        return score
    return None


def is_complete_finish_reason(finish_reason):
    if finish_reason is None:
        return True
    reason = str(finish_reason).lower()
    return reason in ("stop", "eos", "eos_token")


def summarize_lengths(values):
    if not values:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p99": 0.0,
        }
    values = sorted(values)
    count = len(values)
    total = sum(values)
    mid = count // 2
    if count % 2 == 0:
        median = (values[mid - 1] + values[mid]) / 2
    else:
        median = values[mid]
    return {
        "count": count,
        "min": values[0],
        "max": values[-1],
        "mean": total / count,
        "median": median,
        "p90": values[int(0.9 * (count - 1))],
        "p99": values[int(0.99 * (count - 1))],
    }


def load_system_prompt(system_prompt=None, system_prompt_path=None):
    if system_prompt_path:
        cached = _SYSTEM_PROMPT_CACHE.get(system_prompt_path)
        if cached is None:
            with open(system_prompt_path, "r", encoding="utf-8") as f:
                cached = f.read().rstrip("\n")
            _SYSTEM_PROMPT_CACHE[system_prompt_path] = cached
        return cached
    if system_prompt is None:
        return None
    return str(system_prompt).rstrip("\n")

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

RECIPE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROMPTS_DIR = os.path.join(RECIPE_ROOT, "prompts")
PROMPT_PATHS = {
    "proof_gen": os.path.join(PROMPTS_DIR, "proof_generation.yaml"),
    "verify": os.path.join(PROMPTS_DIR, "proof_verification.yaml"),
    "refine": os.path.join(PROMPTS_DIR, "proof_refinement.yaml"),
}
_PROMPT_CACHE = {}
_PROMPT_PATHS_CACHE = {}


def strip_think(text):
    if text is None:
        return ""
    text = text.strip()
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


def parse_thinking(text):
    """Parse text to separate thinking content from response.
    Returns dict with thinking, response, thinking_words, has_thinking."""
    if text is None:
        return {
            "raw": "",
            "thinking": None,
            "response": "",
            "thinking_words": 0,
            "has_thinking": False,
        }

    text = text.strip()
    if "</think>" in text:
        parts = text.split("</think>", 1)
        thinking_part = parts[0]
        response_part = parts[1].strip() if len(parts) > 1 else ""

        # Remove <think> tag if present
        if thinking_part.startswith("<think>"):
            thinking_part = thinking_part[7:]
        thinking_part = thinking_part.strip()

        # Count words in thinking
        thinking_words = len(thinking_part.split())

        return {
            "raw": text,
            "thinking": thinking_part,
            "response": response_part,
            "thinking_words": thinking_words,
            "has_thinking": True,
        }

    return {
        "raw": text,
        "thinking": None,
        "response": text,
        "thinking_words": 0,
        "has_thinking": False,
    }


def load_prompt_template(path):
    if not path:
        return None
    if path in _PROMPT_CACHE:
        return _PROMPT_CACHE[path]
    if not os.path.exists(path):
        _PROMPT_CACHE[path] = None
        return None

    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    template = None
    for idx, line in enumerate(lines):
        if line.startswith("user:"):
            rest = line.split("user:", 1)[1].strip()
            if rest.startswith("|"):
                content_lines = []
                indent = None
                for next_line in lines[idx + 1 :]:
                    if not next_line.strip():
                        content_lines.append("")
                        continue
                    leading = len(next_line) - len(next_line.lstrip(" "))
                    if indent is None:
                        indent = leading
                    if leading < (indent or 0):
                        break
                    content_lines.append(next_line[indent:])
                template = "\n".join(content_lines).rstrip()
            else:
                template = rest
            break

    if template is None:
        template = "\n".join(lines).rstrip()

    _PROMPT_CACHE[path] = template
    return template


def safe_format(template, **kwargs):
    class DefaultDict(dict):
        def __missing__(self, key):
            return ""

    return template.format_map(DefaultDict(**kwargs))


def load_prompt_paths(run_dir):
    if run_dir in _PROMPT_PATHS_CACHE:
        return _PROMPT_PATHS_CACHE[run_dir]

    paths = {
        "proof_gen": PROMPT_PATHS.get("proof_gen"),
        "verify": PROMPT_PATHS.get("verify"),
        "refine": PROMPT_PATHS.get("refine"),
        "refine_instruction": PROMPT_PATHS.get("proof_gen"),
    }

    manifest_path = os.path.join(run_dir, "prompt_paths.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            for stage, entry in manifest.items():
                if not isinstance(entry, dict):
                    continue
                prompt_path = entry.get("prompt")
                if prompt_path:
                    paths[stage] = prompt_path
                if stage == "refine":
                    refine_prompt = entry.get("proof_generation_prompt")
                    if refine_prompt:
                        paths["refine_instruction"] = refine_prompt
        except Exception:
            pass

    _PROMPT_PATHS_CACHE[run_dir] = paths
    return paths


def build_stage_prompt(stage, data, prompt_paths=None):
    if not data:
        return None
    prompt_paths = prompt_paths or {}

    if stage == "proof_gen":
        template = load_prompt_template(prompt_paths.get("proof_gen") or PROMPT_PATHS.get("proof_gen"))
        question = data.get("question") or data.get("statement")
        if not template or not question:
            return None
        return safe_format(template, question=question)

    if stage == "verify":
        template = load_prompt_template(prompt_paths.get("verify") or PROMPT_PATHS.get("verify"))
        statement = data.get("question") or data.get("statement")
        proof = data.get("proof", "")
        if not template or not statement:
            return None
        return safe_format(template, statement=statement, proof=proof)

    if stage == "refine":
        refine_template = load_prompt_template(prompt_paths.get("refine") or PROMPT_PATHS.get("refine"))
        gen_template = load_prompt_template(
            prompt_paths.get("refine_instruction") or prompt_paths.get("proof_gen") or PROMPT_PATHS.get("proof_gen")
        )
        question = data.get("question") or data.get("statement")
        proofs_to_refine = data.get("proofs_to_refine", "")
        if not refine_template or not gen_template or not question:
            return None
        instruction = safe_format(gen_template, question=question).strip()
        return safe_format(
            refine_template,
            instruction=instruction,
            proofs_to_refine=proofs_to_refine,
        )

    return None


def extract_display_fields(data, stage, prompt_paths=None):
    """Extract key fields for display based on stage type."""
    if not data:
        return None

    result = {
        "raw": data,
        "display": {},
        "model_input": None,
        "model_output": None,
    }

    # Common fields to extract for display
    display_fields = [
        "problem_idx",
        "source_name",
        "row_id",
        "verify_row_id",
        "proof_id",
        "round_idx",
        "meanscore",
        "verification_score",
        "self_eval_score",
        "verification_seed",
    ]

    for field in display_fields:
        if field in data:
            result["display"][field] = data[field]

    # Extract model input (prompt/messages)
    if "prompt" in data:
        result["model_input"] = {
            "type": "prompt",
            "content": data["prompt"],
            "char_count": len(data["prompt"]) if data["prompt"] else 0,
        }
    elif "messages" in data:
        messages = data["messages"]
        # Format messages for display
        formatted = []
        total_chars = 0
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            total_chars += len(content) if content else 0
            formatted.append({"role": role, "content": content})
        result["model_input"] = {
            "type": "messages",
            "content": formatted,
            "char_count": total_chars,
            "message_count": len(messages),
        }
    else:
        prompt = build_stage_prompt(stage, data, prompt_paths=prompt_paths)
        if prompt:
            result["model_input"] = {
                "type": "messages",
                "content": [{"role": "user", "content": prompt}],
                "char_count": len(prompt),
                "message_count": 1,
            }

    # Extract model output
    output_fields = ["generation", "proof", "rating_text", "response", "output", "completion"]
    for field in output_fields:
        if field in data:
            parsed = parse_thinking(data[field])
            result["model_output"] = {
                "field_name": field,
                **parsed,
            }
            break

    # Also check for self_eval nested structure
    if "self_eval" in data and isinstance(data["self_eval"], dict):
        result["display"]["self_eval_score"] = data["self_eval"].get("self_eval_score")
        if "self_eval_text" in data["self_eval"]:
            parsed = parse_thinking(data["self_eval"]["self_eval_text"])
            result["self_eval_output"] = {
                "field_name": "self_eval_text",
                **parsed,
            }

    return result


def process_stage_item(item, stage, prompt_paths=None):
    """Process a stage item to extract display-friendly data."""
    processed = {
        "key": item["key"],
        "key_field": item["key_field"],
        "has_input": item["input"] is not None,
        "has_output": item["output"] is not None,
    }

    if item["input"]:
        processed["input"] = extract_display_fields(item["input"], stage, prompt_paths=prompt_paths)
    else:
        processed["input"] = None

    if item["output"]:
        processed["output"] = extract_display_fields(item["output"], stage, prompt_paths=prompt_paths)
    else:
        processed["output"] = None

    if processed["input"] and processed["output"]:
        if not processed["input"].get("model_input") and processed["output"].get("model_input"):
            processed["input"]["model_input"] = processed["output"]["model_input"]

    return processed


def sha1_text(text):
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def file_mtime(path):
    if not path or not os.path.exists(path):
        return None
    return os.path.getmtime(path)


def stage_status(stage_dir):
    input_file = os.path.join(stage_dir, "input.jsonl")
    output_file = os.path.join(stage_dir, "output.jsonl")
    done_file = output_file + ".done"
    stats_file = os.path.join(stage_dir, "stats.json")

    status = {
        "input_exists": os.path.exists(input_file),
        "output_exists": os.path.exists(output_file),
        "done_exists": os.path.exists(done_file),
        "input_file": input_file,
        "output_file": output_file,
        "stats_file": stats_file if os.path.exists(stats_file) else None,
    }

    if os.path.exists(stats_file):
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                status["stats"] = json.load(f)
        except Exception:
            status["stats"] = None

    start = file_mtime(input_file)
    end = file_mtime(done_file) or file_mtime(output_file)
    status["start_time"] = start
    status["end_time"] = end
    if start and end and end >= start:
        status["duration_sec"] = end - start
    else:
        status["duration_sec"] = None

    return status


def list_rounds(run_dir):
    rounds_dir = os.path.join(run_dir, "rounds")
    rounds = []
    if not os.path.isdir(rounds_dir):
        return rounds
    for name in os.listdir(rounds_dir):
        if not name.startswith("R"):
            continue
        try:
            idx = int(name[1:])
        except ValueError:
            continue
        rounds.append(idx)
    return sorted(rounds)


def _parse_int(value):
    try:
        return int(value)
    except Exception:
        return None


def _read_pipeline_config(run_dir):
    config_path = os.path.join(run_dir, "run_config.yaml")
    if not os.path.exists(config_path):
        return {}

    start_round = None
    max_rounds = None
    in_pipeline = False
    pipeline_indent = None

    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()
            if not in_pipeline:
                if stripped.startswith("pipeline:"):
                    in_pipeline = True
                    pipeline_indent = indent
                continue
            if indent <= (pipeline_indent or 0):
                break
            if stripped.startswith("start_round:"):
                start_round = _parse_int(stripped.split(":", 1)[1].strip())
            elif stripped.startswith("max_rounds:"):
                max_rounds = _parse_int(stripped.split(":", 1)[1].strip())

    result = {}
    if start_round is not None:
        result["start_round"] = start_round
    if max_rounds is not None:
        result["max_rounds"] = max_rounds
    return result


def load_questions(run_dir):
    input_path = os.path.join(run_dir, "rounds", "R1", "proof_gen", "input.jsonl")
    if not os.path.exists(input_path):
        return {}
    rows = load_jsonl(input_path)
    questions = {}
    for row in rows:
        problem_idx = row.get("problem_idx")
        source_name = row.get("source_name", "unknown")
        if problem_idx:
            key = (source_name, problem_idx)
            if key not in questions:
                questions[key] = {
                    "question": row.get("question", ""),
                    "source_name": source_name,
                }
    return questions


def load_proof_pool(run_dir):
    pool_dir = os.path.join(run_dir, "proof_pool")
    if not os.path.isdir(pool_dir):
        return {}
    problems = {}
    for root, _, files in os.walk(pool_dir):
        for name in files:
            if not name.endswith(".jsonl"):
                continue
            source_name = os.path.basename(root)
            problem_idx = os.path.splitext(name)[0]
            path = os.path.join(root, name)
            rows = load_jsonl(path)
            for row in rows:
                proof = row.get("proof", "")
                row["proof_hash"] = sha1_text(proof)
                row["proof_preview"] = strip_think(proof)[:400]
            problems[(source_name, problem_idx)] = rows
    return problems


def build_score_distribution(run_dir, bin_count=10):
    pool_dir = os.path.join(run_dir, "proof_pool")
    if not os.path.isdir(pool_dir):
        return {"bins": [], "total": 0, "perfect": 0}

    counts = [0] * bin_count
    total = 0
    perfect = 0
    for root, _, files in os.walk(pool_dir):
        for name in files:
            if not name.endswith(".jsonl"):
                continue
            path = os.path.join(root, name)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    score = row.get("meanscore")
                    if not isinstance(score, (int, float)):
                        continue
                    score = max(0.0, min(1.0, float(score)))
                    if score >= 1.0:
                        perfect += 1
                    total += 1
                    idx = min(int(score * bin_count), bin_count - 1)
                    counts[idx] += 1

    bins = []
    for idx, count in enumerate(counts):
        min_v = idx / bin_count
        max_v = (idx + 1) / bin_count
        bins.append(
            {
                "min": min_v,
                "max": max_v,
                "label": f"{min_v:.1f}-{max_v:.1f}",
                "count": count,
            }
        )

    return {"bins": bins, "total": total, "perfect": perfect}


def _collect_stage_counts(run_dir, stage, rounds):
    counts = defaultdict(int)
    latest = {}
    for round_idx in rounds:
        stage_dir = os.path.join(run_dir, "rounds", f"R{round_idx}", stage)
        output_file = os.path.join(stage_dir, "output.jsonl")
        if not os.path.exists(output_file):
            continue
        rows = load_jsonl(output_file)
        for row in rows:
            problem_idx = row.get("problem_idx")
            source_name = row.get("source_name", "unknown")
            if not problem_idx:
                continue
            key = (source_name, problem_idx)
            counts[key] += 1
            latest[key] = max(latest.get(key, 0), round_idx)
    return counts, latest


def build_problem_index(run_dir):
    questions = load_questions(run_dir)
    pool = load_proof_pool(run_dir)

    index = []
    rounds = list_rounds(run_dir)
    proof_counts = {}
    latest_rounds = {}

    if not pool:
        proof_gen_counts, proof_gen_latest = _collect_stage_counts(run_dir, "proof_gen", rounds)
        refine_counts, refine_latest = _collect_stage_counts(run_dir, "refine", rounds)
        for key, value in proof_gen_counts.items():
            proof_counts[key] = proof_counts.get(key, 0) + value
        for key, value in refine_counts.items():
            proof_counts[key] = proof_counts.get(key, 0) + value
        latest_rounds.update(proof_gen_latest)
        for key, value in refine_latest.items():
            latest_rounds[key] = max(latest_rounds.get(key, 0), value)

    all_keys = set(questions.keys()) | set(pool.keys()) | set(proof_counts.keys())
    for source_name, problem_idx in sorted(all_keys):
        rows = pool.get((source_name, problem_idx))
        question = questions.get((source_name, problem_idx), {}).get("question", "")
        if rows:
            best = max(
                rows,
                key=lambda r: (r.get("meanscore", 0), r.get("self_eval", {}).get("self_eval_score", 0)),
            )
            avg_score = sum(r.get("meanscore", 0) for r in rows) / len(rows)
            latest_round = max(r.get("round_idx", 0) for r in rows)
            best_meanscore = best.get("meanscore", 0)
            best_self_eval_score = best.get("self_eval", {}).get("self_eval_score", 0)
            num_proofs = len(rows)
        else:
            avg_score = 0
            latest_round = latest_rounds.get((source_name, problem_idx), 0)
            best_meanscore = 0
            best_self_eval_score = 0
            num_proofs = proof_counts.get((source_name, problem_idx), 0)

        index.append(
            {
                "source_name": source_name,
                "problem_idx": problem_idx,
                "question": question,
                "best_meanscore": best_meanscore,
                "best_self_eval_score": best_self_eval_score,
                "avg_meanscore": avg_score,
                "num_proofs": num_proofs,
                "latest_round": latest_round,
            }
        )

    index.sort(key=lambda r: (r["source_name"], r["problem_idx"]))
    return index


def build_problem_detail(run_dir, source_name, problem_idx):
    questions = load_questions(run_dir)
    pool = load_proof_pool(run_dir)
    rows = pool.get((source_name, problem_idx))
    question = questions.get((source_name, problem_idx), {}).get("question", "")
    if not rows:
        if question:
            return {
                "source_name": source_name,
                "problem_idx": problem_idx,
                "question": question,
                "trend": [],
                "proofs": [],
            }
        return None

    rows = sorted(
        rows,
        key=lambda r: (
            r.get("meanscore", 0),
            r.get("self_eval", {}).get("self_eval_score", 0),
            r.get("round_idx", -1),
            r.get("proof_id", -1),
        ),
        reverse=True,
    )

    trend = defaultdict(list)
    for row in rows:
        trend[row.get("round_idx", 0)].append(row.get("meanscore", 0))

    trend_rows = []
    running_best = None
    for round_idx in sorted(trend.keys()):
        scores = trend[round_idx]
        round_best = max(scores)
        if running_best is None or round_best > running_best:
            running_best = round_best
        trend_rows.append(
            {
                "round_idx": round_idx,
                "best_meanscore": running_best,
                "avg_meanscore": sum(scores) / len(scores),
                "num_proofs": len(scores),
            }
        )

    proofs = []
    for row in rows:
        proof = row.get("proof", "")
        proofs.append(
            {
                "proof_id": row.get("proof_id"),
                "proof_hash": row.get("proof_hash"),
                "meanscore": row.get("meanscore", 0),
                "self_eval_score": row.get("self_eval", {}).get("self_eval_score", 0),
                "round_idx": row.get("round_idx", 0),
                "dep_proof_ids": row.get("dep_proof_ids", []),
                "proof_preview": row.get("proof_preview"),
                "proof": proof,
            }
        )

    return {
        "source_name": source_name,
        "problem_idx": problem_idx,
        "question": question,
        "trend": trend_rows,
        "proofs": proofs,
    }


def build_verification_cache(run_dir, round_idx):
    verify_path = os.path.join(run_dir, "rounds", f"R{round_idx}", "verify", "output.jsonl")
    if not os.path.exists(verify_path):
        return None

    cache = defaultdict(list)
    rows = load_jsonl(verify_path)
    for row in rows:
        proof = row.get("proof", "")
        proof_hash = sha1_text(proof)
        key = (row.get("problem_idx"), proof_hash)
        cache[key].append(
            {
                "verification_score": row.get("verification_score"),
                "rating_text": strip_think(row.get("rating_text", "")),
                "verification_seed": row.get("verification_seed"),
            }
        )
    return cache


def stage_paths(run_dir, round_idx, stage):
    stage_dir = os.path.join(run_dir, "rounds", f"R{round_idx}", stage)
    return (
        os.path.join(stage_dir, "input.jsonl"),
        os.path.join(stage_dir, "output.jsonl"),
        os.path.join(stage_dir, "output.jsonl.done"),
    )


def select_key_field(stage, rows):
    if stage == "verify":
        for row in rows:
            if "verify_row_id" in row:
                return "verify_row_id"
    return "row_id"


def filter_rows(rows, problem_idx=None, source_name=None):
    filtered = []
    for row in rows:
        if problem_idx and row.get("problem_idx") != problem_idx:
            continue
        if source_name and row.get("source_name") != source_name:
            continue
        filtered.append(row)
    return filtered


def sort_key(value):
    if isinstance(value, int):
        return (0, value)
    if isinstance(value, str):
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)
    return (2, str(value))


def load_stage_items(run_dir, round_idx, stage, problem_idx=None, source_name=None):
    input_path, output_path, done_path = stage_paths(run_dir, round_idx, stage)
    input_rows = load_jsonl(input_path) if os.path.exists(input_path) else []
    output_rows = load_jsonl(output_path) if os.path.exists(output_path) else []

    input_rows = filter_rows(input_rows, problem_idx=problem_idx, source_name=source_name)
    output_rows = filter_rows(output_rows, problem_idx=problem_idx, source_name=source_name)

    key_field = select_key_field(stage, output_rows or input_rows)
    input_map = {}
    for row in input_rows:
        key = row.get(key_field)
        if key is not None and key not in input_map:
            input_map[key] = row
    output_map = {}
    for row in output_rows:
        key = row.get(key_field)
        if key is not None and key not in output_map:
            output_map[key] = row

    keys = sorted(set(input_map) | set(output_map), key=sort_key)
    items = []
    for key in keys:
        items.append(
            {
                "key": key,
                "key_field": key_field,
                "input": input_map.get(key),
                "output": output_map.get(key),
            }
        )

    return {
        "input_path": input_path,
        "output_path": output_path,
        "done_path": done_path,
        "input_total": len(input_rows),
        "output_total": len(output_rows),
        "items": items,
    }


class RunCache:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.problem_index = None
        self.problem_detail = {}
        self.verification_cache = {}
        self.prompt_paths = None
        self.last_refresh = 0

    def refresh(self):
        self.problem_index = build_problem_index(self.run_dir)
        self.problem_detail = {}
        self.verification_cache = {}
        self.prompt_paths = load_prompt_paths(self.run_dir)
        self.last_refresh = time.time()


app = FastAPI()


@app.on_event("startup")
def _startup():
    if not hasattr(app.state, "cache"):
        app.state.cache = None


def get_cache():
    cache = getattr(app.state, "cache", None)
    if cache is None:
        run_dir = (
            getattr(app.state, "run_dir", None) or os.environ.get("ACEPROOF_TTS_RUN_DIR") or os.environ.get("RUN_DIR")
        )
        if not run_dir:
            raise HTTPException(
                status_code=500,
                detail="Run directory not configured. Start with --run-dir or set ACEPROOF_TTS_RUN_DIR.",
            )
        cache = RunCache(run_dir)
        app.state.run_dir = run_dir
        app.state.cache = cache
    if cache.prompt_paths is None:
        cache.prompt_paths = load_prompt_paths(cache.run_dir)
    return cache


@app.get("/api/summary")
def api_summary():
    cache = get_cache()
    run_dir = cache.run_dir
    rounds_present = list_rounds(run_dir)
    config = _read_pipeline_config(run_dir)
    start_round = config.get("start_round", 1)
    max_rounds = config.get("max_rounds")
    if max_rounds:
        rounds = list(range(start_round, max_rounds + 1))
    else:
        rounds = rounds_present
    stages = []
    for round_idx in rounds:
        round_dir = os.path.join(run_dir, "rounds", f"R{round_idx}")
        round_stages = {"round_idx": round_idx}
        if round_idx == 1:
            round_stages["proof_gen"] = stage_status(os.path.join(round_dir, "proof_gen"))
        else:
            round_stages["refine"] = stage_status(os.path.join(round_dir, "refine"))
        round_stages["verify"] = stage_status(os.path.join(round_dir, "verify"))
        stages.append(round_stages)

    score_distribution = build_score_distribution(run_dir)

    return {
        "run_dir": run_dir,
        "rounds": rounds,
        "rounds_present": rounds_present,
        "pipeline": config,
        "stages": stages,
        "score_distribution": score_distribution,
    }


@app.get("/api/problems")
def api_problems():
    cache = get_cache()
    if cache.problem_index is None:
        cache.refresh()
    return {"problems": cache.problem_index}


@app.get("/api/problem/{source_name}/{problem_idx}")
def api_problem(source_name: str, problem_idx: str):
    cache = get_cache()
    key = (source_name, problem_idx)
    if key not in cache.problem_detail:
        detail = build_problem_detail(cache.run_dir, source_name, problem_idx)
        if detail is None:
            raise HTTPException(status_code=404, detail="Problem not found")
        cache.problem_detail[key] = detail
    return cache.problem_detail[key]


@app.get("/api/verifications")
def api_verifications(round_idx: int, problem_idx: str, proof_hash: str, offset: int = 0, limit: int = 64):
    cache = get_cache()
    if round_idx not in cache.verification_cache:
        cache.verification_cache[round_idx] = build_verification_cache(cache.run_dir, round_idx) or {}
    round_cache = cache.verification_cache.get(round_idx, {})
    rows = round_cache.get((problem_idx, proof_hash))
    if rows is None:
        return {"total": 0, "items": []}
    total = len(rows)
    sliced = rows[offset : offset + limit]
    avg = sum(r.get("verification_score", 0) for r in rows) / total if total else 0
    return {"total": total, "avg_score": avg, "items": sliced}


@app.get("/api/stage")
def api_stage(
    round_idx: int,
    stage: str,
    problem_idx: str,
    source_name: str | None = None,
    offset: int = 0,
    limit: int = 50,
    processed: bool = True,
):
    cache = get_cache()
    stage = stage.strip()
    if stage not in {"proof_gen", "refine", "verify"}:
        raise HTTPException(status_code=400, detail="Unknown stage")
    payload = load_stage_items(
        cache.run_dir,
        round_idx,
        stage,
        problem_idx=problem_idx,
        source_name=source_name,
    )
    items = payload["items"]
    total = len(items)
    sliced = items[offset : offset + limit]

    # Process items for better display if requested
    if processed:
        sliced = [process_stage_item(item, stage, prompt_paths=cache.prompt_paths) for item in sliced]

    return {
        "round_idx": round_idx,
        "stage": stage,
        "problem_idx": problem_idx,
        "source_name": source_name,
        "input_path": payload["input_path"],
        "output_path": payload["output_path"],
        "done_path": payload["done_path"],
        "input_total": payload["input_total"],
        "output_total": payload["output_total"],
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": sliced,
    }


@app.get("/api/stage/item")
def api_stage_item(
    round_idx: int,
    stage: str,
    problem_idx: str,
    key: str,
    source_name: str | None = None,
):
    """Fetch a single stage item by key for detailed viewing."""
    cache = get_cache()
    stage = stage.strip()
    if stage not in {"proof_gen", "refine", "verify"}:
        raise HTTPException(status_code=400, detail="Unknown stage")

    payload = load_stage_items(
        cache.run_dir,
        round_idx,
        stage,
        problem_idx=problem_idx,
        source_name=source_name,
    )

    # Find the item by key
    for item in payload["items"]:
        if str(item["key"]) == str(key):
            return {
                "round_idx": round_idx,
                "stage": stage,
                "key": key,
                "item": process_stage_item(item, stage, prompt_paths=cache.prompt_paths),
                "raw_item": item,
            }

    raise HTTPException(status_code=404, detail=f"Item with key {key} not found")


@app.get("/")
def index():
    return FileResponse(os.path.join(app.state.static_dir, "index.html"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    static_dir = os.path.join(os.path.dirname(__file__), "static")

    app.state.run_dir = run_dir
    app.state.static_dir = static_dir
    app.state.cache = RunCache(run_dir)

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

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

"""The run driver: one task per problem (search, then final selection), results, and clean shutdown."""

import asyncio
import contextlib
import signal
from pathlib import Path

from nemotron_imo_tts import selection
from nemotron_imo_tts.client import FatalRequestError
from nemotron_imo_tts.orchestrator import ProblemOrchestrator
from nemotron_imo_tts.resume import Ledger
from nemotron_imo_tts.text import append_jsonl, load_jsonl, write_jsonl

RESULTS_FILE = "results.jsonl"
SUBMISSIONS_FILE = "submissions.jsonl"
STOP_FILE = "STOP"
ROUND_STAGES = {"gen", "verify", "refine", "errors"}
SELECTION_STAGES = {selection.STAGE_JUDGE: "judge.jsonl", selection.STAGE_JUDGE_ERRORS: "errors.jsonl"}


def problem_dir(run_dir, problem):
    return Path(run_dir) / "problems" / problem["problem_idx"]


def make_sink(pdir):
    lock = asyncio.Lock()

    async def sink(stage, round_idx, row):
        if stage in ROUND_STAGES:
            path = pdir / "rounds" / f"R{round_idx}" / f"{stage}.jsonl"
        elif stage in SELECTION_STAGES:
            path = pdir / "selection" / SELECTION_STAGES[stage]
        else:
            raise ValueError(f"unknown sink stage {stage!r}")
        async with lock:
            await asyncio.to_thread(append_jsonl, path, row)

    return sink


def write_pool(run_dir, problem, pool_records):
    path = Path(run_dir) / "proof_pool" / problem.get("source_name", "unknown") / f"{problem['problem_idx']}.jsonl"
    write_jsonl(path, pool_records)


def _finalist_entry(record, judgment):
    entry = {
        "generation_model_id": record.generation_model_id,
        "proof": record.proof,
        "proof_sha256": record.proof_sha256,
        "proof_id": record.proof_id,
        "round_idx": record.round_idx,
        "accepted": record.accepted,
        "meanscore": record.meanscore,
        "verifier_means": dict(record.verdict.verifier_means) if record.verdict else {},
        "self_eval_score": record.self_eval_score,
    }
    if judgment is not None:
        entry.update(
            {
                "num_judgments": judgment.num_judgments,
                "num_valid_judgments": judgment.num_valid_judgments,
                "num_resampled_judgments": judgment.num_resampled_judgments,
                "mean_imo_score": judgment.mean_imo_score,
                "per_judge_means": judgment.per_judge_means,
            }
        )
    return entry


def result_rows(problem, result, judgments, cfg):
    """(results row, submissions row) for one finished problem."""
    by_record = {id(j.record): j for j in judgments}
    finalists = [_finalist_entry(record, by_record.get(id(record))) for record in result.finalists]
    selected_record = None
    selected_judgment = None
    if judgments:
        selected_judgment = selection.select(judgments, cfg)
        selected_record = selected_judgment.record
    elif result.finalists:
        selected_record = selection.select_unjudged(result.finalists, cfg)
    base = {
        "problem_idx": problem["problem_idx"],
        "source_name": problem.get("source_name", "unknown"),
        "question": problem["question"],
    }
    results_row = {
        **base,
        "status": result.status,
        "rounds_run": result.rounds_run,
        "num_candidates": result.num_candidates,
        "num_parked_requests": result.num_parked,
        "num_prompt_too_long": result.num_prompt_too_long,
        "finalists": finalists,
        "selected_generation_model_id": selected_record.generation_model_id if selected_record else None,
        "selected_proof_sha256": selected_record.proof_sha256 if selected_record else None,
    }
    submission_row = {
        **base,
        "proof": selected_record.proof if selected_record else "",
        "proof_sha256": selected_record.proof_sha256 if selected_record else None,
        "generation_model_id": selected_record.generation_model_id if selected_record else None,
        "accepted": bool(selected_record.accepted) if selected_record else False,
        "meanscore": selected_record.meanscore if selected_record else None,
        "mean_imo_score": selected_judgment.mean_imo_score if selected_judgment else None,
    }
    return results_row, submission_row


async def run_problem(cfg, run_dir, prompts, layer, problem, log):
    pdir = problem_dir(run_dir, problem)
    sink = make_sink(pdir)
    ledger = Ledger.scan(pdir)
    log.info("[%s] starting search (%d persisted rows)", problem["problem_idx"], len(ledger))
    orchestrator = ProblemOrchestrator(cfg, prompts, layer.request, sink, ledger)
    result = await orchestrator.run(problem)
    log.info(
        "[%s] search finished: %s after %d round(s), %d candidates, %d finalist(s), %d parked",
        problem["problem_idx"],
        result.status,
        result.rounds_run,
        result.num_candidates,
        len(result.finalists),
        result.num_parked,
    )
    judgments = []
    if cfg.judges and result.finalists:
        judgments = await selection.judge_finalists(
            problem, result.finalists, cfg, prompts, layer.request, sink, ledger
        )
        for judgment in judgments:
            log.info(
                "[%s] finalist %s: mean IMO score %s over %d/%d valid judgments",
                problem["problem_idx"],
                judgment.record.generation_model_id,
                judgment.mean_imo_score,
                judgment.num_valid_judgments,
                judgment.num_judgments,
            )
    write_pool(run_dir, problem, result.pool)
    return result_rows(problem, result, judgments, cfg)


def finished_problem_ids(run_dir):
    path = Path(run_dir) / RESULTS_FILE
    if not path.exists():
        return set()
    return {row["problem_idx"] for row in load_jsonl(path)}


async def run_driver(cfg, run_dir, prompts, layer, problems, log, stop_poll_s=2.0):
    """Run every unfinished problem; returns the process exit code."""
    run_dir = Path(run_dir)
    done = finished_problem_ids(run_dir)
    todo = [p for p in problems if p["problem_idx"] not in done]
    log.info("%d problem(s) to run, %d already finished", len(todo), len(done))
    if not todo:
        return 0

    semaphore = asyncio.Semaphore(cfg.concurrency.problems)
    write_lock = asyncio.Lock()
    stop_event = asyncio.Event()
    fatal = []

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
            loop.add_signal_handler(sig, stop_event.set)

    async def watch_stop():
        while not stop_event.is_set():
            if (run_dir / STOP_FILE).exists():
                log.warning("STOP file found in %s; draining", run_dir)
                stop_event.set()
                return
            await asyncio.sleep(stop_poll_s)

    async def one(problem):
        async with semaphore:
            try:
                results_row, submission_row = await run_problem(cfg, run_dir, prompts, layer, problem, log)
            except FatalRequestError as exc:
                log.error("[%s] fatal request error: %s", problem["problem_idx"], exc)
                fatal.append(exc)
                stop_event.set()
                return
            async with write_lock:
                append_jsonl(run_dir / SUBMISSIONS_FILE, submission_row)
                append_jsonl(run_dir / RESULTS_FILE, results_row)
            log.info("[%s] result written (%s)", problem["problem_idx"], results_row["status"])

    tasks = [asyncio.create_task(one(problem)) for problem in todo]
    watcher = asyncio.create_task(watch_stop())
    stopper = asyncio.create_task(stop_event.wait())
    pending = set(tasks)
    exit_code = 0
    try:
        while pending:
            done_now, _ = await asyncio.wait(pending | {stopper}, return_when=asyncio.FIRST_COMPLETED)
            pending = {task for task in pending if not task.done()}
            if stopper in done_now:
                break
            for task in done_now - {stopper}:
                if task.exception() is not None:
                    log.error("problem task failed: %r", task.exception())
                    exit_code = 1
                    stop_event.set()
                    break
            if stop_event.is_set():
                break
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        watcher.cancel()
        stopper.cancel()
        await asyncio.gather(watcher, stopper, return_exceptions=True)
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.remove_signal_handler(sig)
    if fatal:
        return 1
    if stop_event.is_set() and exit_code == 0:
        log.warning("stopped before all problems finished; relaunch the same command to resume")
    return exit_code

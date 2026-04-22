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

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import shutil
import textwrap
from dataclasses import field, fields
from pathlib import Path
from typing import Any

from nemo_skills.evaluation.evaluator.base import BaseEvaluatorConfig
from nemo_skills.evaluation.evaluator.code import preprocess_code
from nemo_skills.evaluation.evaluator.livecodebench import (
    execute_in_sandbox_with_retries,
    is_sandbox_available,
    sandbox_context,
)
from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))

SAFIM_GIT_URL = "git+https://github.com/wasiahmad/safim.git"

# Combined grammars built in Dockerfile.sandbox.execeval; safim ``ast_utils`` uses SAFIM_TREE_SITTER_SO.
_SAFIM_DEFAULT_TREE_SITTER_SO = "/opt/safim-tree-sitter/safim-languages.so"

# ExecEval workers often have cwd=/root/execution_engine; a local unittest.py shadows stdlib and
# breaks huggingface datasets (ModuleNotFoundError: unittest.mock). Use a clean working directory.
_SAFIM_SANDBOX_CWD = "/tmp"


def _remove_overlap(preceding_text: str, following_text: str, *, truncate_from: str = "following") -> str:
    """Trim overlap between infill and prefix/suffix (same idea as human-eval-infilling)."""
    assert truncate_from in ("preceding", "following")
    preceding_len = len(preceding_text)
    following_len = len(following_text)
    for i in range(min(preceding_len, following_len), 0, -1):
        if truncate_from == "following":
            overlap = preceding_text[-i:]
            if overlap.strip() == "" and "\n" not in overlap:
                continue
            if following_text.startswith(overlap):
                return following_text[i:]
        else:
            overlap = following_text[:i]
            if overlap.strip() == "" and "\n" not in overlap:
                continue
            if preceding_text.endswith(overlap):
                return preceding_text[:-i]
    return following_text if truncate_from == "following" else preceding_text


def _fim_postprocess(sample: dict) -> dict:
    sample["completion"] = _remove_overlap(sample["prefix"], sample["completion"], truncate_from="following")
    sample["completion"] = _remove_overlap(sample["completion"], sample["suffix"], truncate_from="preceding")
    return sample


def _fence_language(dataset_language: str) -> str:
    lang = str(dataset_language).lower()
    if lang in ("cpp", "c++"):
        return "cpp"
    if lang in ("csharp", "c#"):
        return "csharp"
    return lang


@nested_dataclass(kw_only=True)
class SafimEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for SAFIM via ``safim.evaluate`` in the sandbox."""

    sandbox: dict = field(default_factory=lambda: {"sandbox_type": "local"})
    # HuggingFace subset: api | block | control (passed to ``safim.evaluate`` as ``completion_type``).
    subset: str = "api"
    # Optional filter; ``None`` evaluates all languages in the subset.
    language: str | None = None
    # ExecEval HTTP API port inside the sandbox (must match gunicorn LISTEN_PORT; pipeline sets NEMO_SKILLS_SANDBOX_PORT).
    execeval_port: int = field(default_factory=lambda: int(os.environ.get("NEMO_SKILLS_SANDBOX_PORT", "6000")))
    # Passed to ``safim.evaluate`` (parallelism against ExecEval).
    max_workers: int = 16
    num_retries: int = 3
    eval_timeout_buffer_s: float = 120.0
    # ``None``: fence/thinking cleanup only. ``basic``: FIM overlap trim (NeMo-side). ``advanced``:
    # pass ``post_process=True`` into ``safim.evaluate`` (subset-specific rules in installed safim).
    postprocess: str | None = None
    # Path to combined tree-sitter ``.so`` inside the sandbox (``block`` + ``post_process``). Default matches
    # NeMo execeval image (``Dockerfile.sandbox.execeval``); image ENV ``SAFIM_TREE_SITTER_SO`` also sets this.
    tree_sitter_so: str | None = None
    safim_git_url: str = SAFIM_GIT_URL


def _safim_eval_subprocess_env(eval_config: SafimEvaluatorConfig) -> str:
    """``env`` prefix for the sandbox ``python -c`` that runs ``safim.evaluate`` (tree-sitter + BLAS caps)."""
    # Paths are inside the sandbox container (not the Slurm driver); override via ``eval_config.tree_sitter_so``.
    so = eval_config.tree_sitter_so or _SAFIM_DEFAULT_TREE_SITTER_SO
    return (
        f"env SAFIM_TREE_SITTER_SO={shlex.quote(so)} "
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 GOTO_NUM_THREADS=1 BLIS_NUM_THREADS=1 "
    )


def _safim_cfg_subset(cfg: dict) -> dict:
    allowed = {f.name for f in fields(SafimEvaluatorConfig)}
    return {k: v for k, v in cfg.items() if k in allowed}


def _normalize_safim_postprocess(mode: str | None) -> str | None:
    """Return ``None``, ``basic``, or ``advanced``; raise if invalid."""
    if mode is None:
        return None
    if not isinstance(mode, str):
        raise TypeError(f"eval_config.postprocess must be str or None, got {type(mode).__name__}")
    key = mode.strip().lower()
    if not key:
        return None
    if key not in ("basic", "advanced"):
        raise ValueError(f"eval_config.postprocess must be None, 'basic', or 'advanced' (got {mode!r})")
    return key


def _post_process_flag(pp: str | None, subset: str) -> bool:
    """Whether to pass ``post_process=True`` into ``safim.evaluate`` (``pp`` already normalized)."""
    if pp != "advanced":
        return False
    if subset not in ("api", "block", "control"):
        LOG.warning(
            "postprocess=advanced is only supported for subset api, block, or control (got %r); "
            "running evaluate with post_process=False.",
            subset,
        )
        return False
    return True


def _preprocess_safim_jsonl(
    jsonl_file: str,
    pp: str | None,
) -> list[dict]:
    """Read JSONL, fence-extract completion, optional NeMo-only basic FIM trim; write back.

    ``pp`` must be the return value of :func:`_normalize_safim_postprocess` (``None``, ``basic``, or ``advanced``).
    """
    path = Path(jsonl_file)
    with open(path, encoding="utf-8") as fh:
        raw = [json.loads(line) for line in fh]

    if not raw:
        raise ValueError(f"No samples found in {jsonl_file}")

    seen: set[str] = set()
    duplicates: list[str] = []
    prepared: list[dict] = []
    for idx, row in enumerate(raw):
        row = dict(row)
        fence_lang = _fence_language(row["language"])
        row = preprocess_code(row, language=fence_lang, strip_whitespace=False)
        if pp == "basic":
            row = _fim_postprocess(row)
        tid = str(row.get("task_id", f"row-{idx}"))
        row["task_id"] = tid
        if tid in seen:
            duplicates.append(tid)
        seen.add(tid)
        prepared.append(row)

    if duplicates:
        raise ValueError(
            "safim.evaluate() keeps one completion per task_id (dict merge over JSONL). "
            f"Duplicate task_id(s) in {jsonl_file}: {sorted(set(duplicates))}. "
            "Use one row per task per file (e.g. separate output-rs*.jsonl seeds)."
        )

    with open(path, "w", encoding="utf-8") as fh:
        for row in prepared:
            fh.write(json.dumps(row) + "\n")

    return prepared


def _postprocess_safim_results(jsonl_file: str, samples: list[dict]) -> None:
    """Merge ``safim`` harness ``eval[task_id][0].passed`` into JSONL; rename harness file."""
    jsonl_path = Path(jsonl_file)
    results_path = jsonl_path.with_name(jsonl_path.stem + "_eval_results.json")
    saved_path = jsonl_path.with_name(jsonl_path.stem + "_eval_results-saved.json")

    with open(results_path, encoding="utf-8") as fh:
        eval_payload = json.load(fh)

    eval_block = eval_payload.get("eval") or {}

    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for s in samples:
            tid = str(s["task_id"])
            entry = eval_block.get(tid)
            passed = False
            if isinstance(entry, list) and entry:
                passed = bool(entry[0].get("passed", False))
            s["passed"] = passed
            fh.write(json.dumps(s) + "\n")

    shutil.move(str(results_path), str(saved_path))
    LOG.info("Finished processing %s, results saved.", jsonl_file)


async def _install_safim_in_sandbox(sandbox, eval_config: SafimEvaluatorConfig) -> bool:
    cmd = f"cd {_SAFIM_SANDBOX_CWD} && python -m pip install {eval_config.safim_git_url}"
    out, _ = await execute_in_sandbox_with_retries(
        sandbox, eval_config.num_retries, cmd, language="shell", timeout=300
    )
    if out.get("process_status") != "completed":
        LOG.warning("Failed to install safim: %s", out.get("stderr", "Unknown error"))
        return False
    LOG.info("Successfully installed safim.")
    return True


async def eval_safim_async(eval_config: SafimEvaluatorConfig) -> None:
    async with sandbox_context(eval_config.sandbox) as sandbox:
        if not await _install_safim_in_sandbox(sandbox, eval_config):
            return

        jsonl_path = Path(eval_config.input_file).resolve()
        jsonl_file = str(jsonl_path)
        try:
            pp = _normalize_safim_postprocess(eval_config.postprocess)
        except (TypeError, ValueError) as e:
            LOG.error("%s", e)
            return
        samples = _preprocess_safim_jsonl(jsonl_file, pp)
        output_json = str(jsonl_path.with_name(jsonl_path.stem + "_eval_results.json"))
        lang_repr = "None" if eval_config.language is None else repr(eval_config.language)
        post_process = _post_process_flag(pp, eval_config.subset)
        eval_code = textwrap.dedent(
            f"""
            from safim.evaluate import evaluate
            evaluate(
                {repr(eval_config.subset)},
                {repr(jsonl_file)},
                {repr(output_json)},
                post_process={repr(post_process)},
                language={lang_repr},
                port={int(eval_config.execeval_port)},
                max_workers={int(eval_config.max_workers)},
            )
            """
        )

        cmd = f"cd {_SAFIM_SANDBOX_CWD} && {_safim_eval_subprocess_env(eval_config)}python -c {shlex.quote(eval_code)}"
        timeout_s = max(300.0, len(samples) * 30.0 + eval_config.eval_timeout_buffer_s)
        output, _ = await execute_in_sandbox_with_retries(
            sandbox,
            eval_config.num_retries,
            cmd,
            language="shell",
            timeout=timeout_s,
            max_output_characters=100_000,
        )
        if output.get("process_status") != "completed":
            LOG.error("SAFIM evaluation failed for %s. Stderr: %s", jsonl_file, output.get("stderr"))
            return

        results_path = Path(output_json)
        if not results_path.is_file():
            LOG.error("Expected SAFIM output at %s but file is missing.", output_json)
            return

        _postprocess_safim_results(jsonl_file, samples)


def eval_safim(cfg: dict) -> dict[str, Any]:
    """Require a reachable sandbox, install ``safim`` there, and run evaluation."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass

    if not isinstance(cfg, dict):
        raise TypeError(f"eval_safim expected dict-like config, got {type(cfg)}")

    eval_cfg = SafimEvaluatorConfig(_init_nested=True, **_safim_cfg_subset(cfg))

    if not eval_cfg.input_file:
        raise ValueError("eval_safim requires input_file in eval config.")

    sandbox_is_ready = asyncio.run(is_sandbox_available(eval_cfg.sandbox))
    if not sandbox_is_ready:
        raise RuntimeError(
            "SAFIM evaluation requires a reachable NeMo code sandbox. "
            "Start the sandbox sidecar and set host/port (e.g. Hydra ++sandbox.host, ++sandbox.port, "
            "eval_config.sandbox, or NEMO_SKILLS_SANDBOX_HOST / NEMO_SKILLS_SANDBOX_PORT). "
            "On Slurm, the client step must use a host that reaches the sandbox container (not 127.0.0.1 from "
            "another container); generate merges SLURM_MASTER_NODE into eval_config.sandbox when appropriate. "
            "For eval_type=safim, generate.py merges eval_config.sandbox with top-level sandbox and those env vars."
        )

    asyncio.run(eval_safim_async(eval_cfg))

    return {}

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

"""Run the Nemotron-IMO-TTS pipeline: search, final selection, and resumable outputs.

    python recipes/nemotron-imo-tts/run.py --config recipes/nemotron-imo-tts/configs/imo2026-ensemble.yaml \\
        --output-dir /path/to/run [--dry-run]

Relaunching the same command resumes: completed work is replayed from the run directory and only missing
requests are issued. Create a file named STOP in the output directory to drain and exit cleanly.
"""

import argparse
import asyncio
import logging
import resource
import sys
from pathlib import Path

RECIPE_ROOT = Path(__file__).resolve().parent
if str(RECIPE_ROOT) not in sys.path:
    sys.path.insert(0, str(RECIPE_ROOT))

from nemotron_imo_tts import config as cfgmod  # noqa: E402
from nemotron_imo_tts import driver, manifest, prompting  # noqa: E402
from nemotron_imo_tts.client import ContextBudgeter, RequestLayer  # noqa: E402

LANE_FD_MARGIN = 64


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Run config yaml (see configs/).")
    parser.add_argument("--output-dir", required=True, help="Run directory: outputs, ledgers, manifests.")
    parser.add_argument("--dry-run", action="store_true", help="Preflight only: validate, freeze, and stop.")
    return parser.parse_args(argv)


def setup_logging(run_dir):
    log = logging.getLogger("nemotron_imo_tts")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    log.addHandler(stream)
    run_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(run_dir / "run.log")
    file_handler.setFormatter(fmt)
    log.addHandler(file_handler)
    log.propagate = False
    return log


def check_open_file_limit(cfg):
    lanes = cfg.concurrency
    needed = lanes.generation + lanes.verification + lanes.refinement + lanes.judging + LANE_FD_MARGIN
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != resource.RLIM_INFINITY and soft < needed:
        raise SystemExit(
            f"The open-file limit ({soft}) is below the configured concurrency lanes plus margin ({needed}). "
            f"Raise it with `ulimit -n {needed}` or lower the concurrency settings."
        )


async def preflight_models(layer, cfg, log):
    seen = set()
    for spec in cfgmod.all_model_specs(cfg):
        if spec.base_url in seen:
            continue
        seen.add(spec.base_url)
        served = await layer.list_models(spec)
        wanted = {s.model for s in cfgmod.all_model_specs(cfg) if s.base_url == spec.base_url}
        if served is None:
            log.warning("%s does not list its models; skipping the model-name check", spec.base_url)
            continue
        missing = sorted(wanted - set(served))
        if missing:
            log.warning("%s does not list configured model(s) %s; requests may fail", spec.base_url, missing)


async def async_main(cfg, run_dir, problems, prompts, log, dry_run, transport):
    budgeter = ContextBudgeter(
        cfg.context_budget.tokenizer if cfg.context_budget else None,
        cfg.context_budget.safety_margin_tokens if cfg.context_budget else 0,
    )
    layer = RequestLayer(cfg, budgeter, transport=transport)
    try:
        if cfg.context_budget:
            try:
                await asyncio.to_thread(budgeter.load)
            except Exception as exc:  # noqa: BLE001 - transformers raises a mix of OSError/ValueError/ImportError
                log.error("cannot load tokenizer %s: %s", cfg.context_budget.tokenizer, exc)
                return 1
            log.info("loaded tokenizer %s", cfg.context_budget.tokenizer)
        await preflight_models(layer, cfg, log)
        if dry_run:
            log.info("dry run: preflight passed for %s; nothing submitted", run_dir)
            return 0
        return await driver.run_driver(cfg, run_dir, prompts, layer, problems, log)
    finally:
        await layer.aclose()


def main(argv=None, transport=None):
    args = parse_args(argv)
    run_dir = Path(args.output_dir).resolve()
    log = setup_logging(run_dir)
    try:
        cfg = cfgmod.load_config(args.config)
        for spec in cfgmod.all_model_specs(cfg):
            cfgmod.resolve_api_key(spec)
        check_open_file_limit(cfg)
        counts = cfgmod.derived_counts(cfg)
        log.info(
            "counts: %d round-1 attempts, %d judgments per proof, %d refinements per round, %d judgments per finalist",
            counts["generations_per_problem"],
            counts["judgments_per_proof"],
            counts["refinements_per_round"],
            counts["judgments_per_finalist"],
        )
        problems, input_sha256 = manifest.freeze_input(cfg.input, run_dir)
        prompts_dir = RECIPE_ROOT / "prompts"
        prompts = prompting.load_prompt_set(
            prompts_dir, list(cfg.search.generation_prompts), with_judge=cfg.judges is not None
        )
        prompts_doc = manifest.prompt_manifest(prompts_dir, prompts)
        manifest.write_or_verify(run_dir, manifest.PROMPT_MANIFEST, prompts_doc)
        code_sha, code_dirty = manifest.code_state(RECIPE_ROOT)
        if code_dirty:
            log.warning("the code checkout has uncommitted changes; the run manifest records code_dirty=true")
        created = manifest.write_or_verify_run_manifest(
            run_dir, manifest.run_manifest_keys(cfg, input_sha256, prompts_doc), code_sha, code_dirty
        )
        log.info("%s run manifest in %s (%d problems)", "wrote" if created else "verified", run_dir, len(problems))
    except (cfgmod.ConfigError, manifest.ManifestError) as exc:
        log.error("%s", exc)
        return 2
    return asyncio.run(async_main(cfg, run_dir, problems, prompts, log, args.dry_run, transport))


if __name__ == "__main__":
    sys.exit(main())

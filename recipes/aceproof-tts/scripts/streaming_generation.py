# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Streaming per-problem generation module for the aceproof-tts recipe.

A GenerationTask subclass whose `process_single_datapoint` runs the entire
per-problem gen -> verify -> score -> (solved? / refine) loop via
`streaming_orchestrator.ProblemOrchestrator`. Every request goes through the
shared `self.semaphore`, so all proof_gen / verify / refine calls reuse skills'
single global concurrency bound + transport (no hand-rolled client).

Two model clients front the gateway by profile name (they share base_url and
differ only in the OpenAI `model`): `self.llm` for proof_gen + refine, and
`self.verify_llm` for verification.
"""

import asyncio
import json
import logging
import os
import sys
from dataclasses import field

import hydra
from omegaconf import OmegaConf

from nemo_skills.inference.generate import GenerationTask, InferenceConfig
from nemo_skills.inference.model import get_model, server_params
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    setup_logging,
)

try:  # config class name differs across NeMo-Skills branches
    from nemo_skills.inference.generate import GenerateSolutionsConfig as _BaseGenConfig
except ImportError:  # pragma: no cover
    from nemo_skills.inference.generate import GenerationTaskConfig as _BaseGenConfig

PIPELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pipeline"))
if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)

from proof_pool_manager import ProofPoolManager  # noqa: E402
from streaming_orchestrator import ProblemOrchestrator, StreamingConfig  # noqa: E402
from utils import hash_problem_idx, load_system_prompt  # noqa: E402

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class StreamingInferenceConfig(InferenceConfig):
    pass


@nested_dataclass(kw_only=True)
class StreamingGenerationConfig(_BaseGenConfig):
    inference: StreamingInferenceConfig = field(default_factory=StreamingInferenceConfig)
    prompt_format: str = "openai"

    # where rounds/R*/{proof_gen, verify, refine}/output.jsonl + proof_pool live
    streaming_output_dir: str | None = None

    # prompt templates (paths to the recipe prompt yamls)
    gen_prompt_config_path: str | None = None
    verify_prompt_config_path: str | None = None
    refine_prompt_config_path: str | None = None

    # optional per-stage system prompts
    gen_system_prompt_path: str | None = None
    verify_system_prompt_path: str | None = None
    refine_system_prompt_path: str | None = None

    # second profile (the OpenAI `model`) used for verification on the gateway
    verify_model: str | None = None
    # refine reuses the proof_gen profile by default (matches aceproof config)
    refine_model: str | None = None

    # Optional per-stage token budgets. Leave unset to let the gateway/server
    # use its available context window instead of sending max_completion_tokens.
    gen_tokens_to_generate: int | None = None
    verify_tokens_to_generate: int | None = None
    refine_tokens_to_generate: int | None = None

    # Optional per-stage sampling overrides. If unset, cfg.inference.* is used.
    gen_temperature: float | None = None
    verify_temperature: float | None = None
    refine_temperature: float | None = None
    gen_top_p: float | None = None
    verify_top_p: float | None = None
    refine_top_p: float | None = None

    # scaling knobs (mirror the aceproof scaling block)
    n_parallel_proof_gen: int = 128
    n_verification_per_proof: int = 64
    n_agg_trials: int = 32
    n_best_proofs_to_sample: int = 32
    n_proofs_to_refine: int = 1
    max_rating_per_score: int = 4
    solved_threshold: float = 0.99999
    max_rounds: int = 2

    # early-stop experiment knobs
    min_verifications_per_proof: int = 8
    early_stop_only_if_score_lt_1: bool = True
    cancel_remaining: bool = True

    # Per-role in-flight concurrency lanes. The base GenerationTask shares ONE
    # FIFO semaphore for everything; at full scale the 25k gen tasks (created
    # before any verify) monopolize it and verify never reaches the front of the
    # FIFO -> verification starves. Separate lanes let verify/refine acquire
    # independently of the gen backlog. Size to the fleet: proofgen ~384 slots
    # (48x8), verify ~1024 slots (16x64), plus a small pipelining buffer.
    gen_max_concurrent: int = 512
    verify_max_concurrent: int = 1280
    refine_max_concurrent: int = 512

    # write proof_pool on disk so finalize_results.py stays usable
    write_proof_pool: bool = True


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_streaming_generation_config", node=StreamingGenerationConfig)


class StreamingProofTask(GenerationTask):
    def log_example_prompt(self, data):
        return

    def setup_prompt(self):
        # prompts are built inside the orchestrator from the templates
        return None

    def setup_llm(self):
        llm = super().setup_llm()

        # second client for the verify profile (same gateway, different model)
        verify_model = self.cfg.verify_model or self.cfg.server.get("model")
        verify_server = dict(self.cfg.server)
        verify_server["model"] = verify_model
        self.verify_llm = get_model(
            **verify_server,
            tokenizer=self.tokenizer,
            data_dir=self.data_dir or "",
            output_dir=str(self.cfg.streaming_output_dir or "."),
        )
        # refine reuses the proof_gen profile by default
        if self.cfg.refine_model and self.cfg.refine_model != self.cfg.server.get("model"):
            refine_server = dict(self.cfg.server)
            refine_server["model"] = self.cfg.refine_model
            self.refine_llm = get_model(
                **refine_server,
                tokenizer=self.tokenizer,
                data_dir=self.data_dir or "",
                output_dir=str(self.cfg.streaming_output_dir or "."),
            )
        else:
            self.refine_llm = llm

        # prompt templates
        self.gen_template = OmegaConf.load(self.cfg.gen_prompt_config_path).user
        self.verify_template = OmegaConf.load(self.cfg.verify_prompt_config_path).user
        self.refine_template = OmegaConf.load(self.cfg.refine_prompt_config_path).user
        self.proof_gen_template = self.gen_template  # refine instruction reuses proof_gen prompt

        self.gen_system_prompt = load_system_prompt(None, self.cfg.gen_system_prompt_path)
        self.verify_system_prompt = load_system_prompt(None, self.cfg.verify_system_prompt_path)
        self.refine_system_prompt = load_system_prompt(None, self.cfg.refine_system_prompt_path)

        # streaming output dir + pool manager + per-file sink locks
        self.streaming_output_dir = self.cfg.streaming_output_dir or str(
            os.path.dirname(os.path.abspath(self.cfg.output_file))
        )
        self._sink_locks: dict[str, asyncio.Lock] = {}
        self.pool_manager = (
            ProofPoolManager(
                os.path.join(self.streaming_output_dir, "proof_pool"),
                solved_threshold=self.cfg.solved_threshold,
            )
            if self.cfg.write_proof_pool
            else None
        )

        if self.cfg.n_agg_trials > 0 and self.cfg.n_parallel_proof_gen % self.cfg.n_agg_trials == 0:
            self.n_samples_per_trial = self.cfg.n_parallel_proof_gen // self.cfg.n_agg_trials
        else:
            self.n_samples_per_trial = 1

        # Per-role concurrency lanes (see StreamingGenerationConfig). Created here
        # synchronously, exactly like the base self.semaphore in __init__ -- the
        # asyncio.Semaphore binds to the loop on first await inside async_loop.
        self.gen_semaphore = asyncio.Semaphore(self.cfg.gen_max_concurrent)
        self.verify_semaphore = asyncio.Semaphore(self.cfg.verify_max_concurrent)
        self.refine_semaphore = asyncio.Semaphore(self.cfg.refine_max_concurrent)
        return llm

    def _streaming_cfg(self):
        return StreamingConfig(
            n_parallel_proof_gen=self.cfg.n_parallel_proof_gen,
            n_verification_per_proof=self.cfg.n_verification_per_proof,
            n_agg_trials=self.cfg.n_agg_trials,
            n_best_proofs_to_sample=self.cfg.n_best_proofs_to_sample,
            n_proofs_to_refine=self.cfg.n_proofs_to_refine,
            max_rating_per_score=self.cfg.max_rating_per_score,
            n_samples_per_trial=self.n_samples_per_trial,
            solved_threshold=self.cfg.solved_threshold,
            max_rounds=self.cfg.max_rounds,
            min_verifications_per_proof=self.cfg.min_verifications_per_proof,
            early_stop_only_if_score_lt_1=self.cfg.early_stop_only_if_score_lt_1,
            cancel_remaining=self.cfg.cancel_remaining,
        )

    async def _request(self, role, messages, seed):
        if role == "verify":
            client, max_tokens, sem = self.verify_llm, self.cfg.verify_tokens_to_generate, self.verify_semaphore
            temperature = self.cfg.verify_temperature
            top_p = self.cfg.verify_top_p
        elif role == "refine":
            client, max_tokens, sem = self.refine_llm, self.cfg.refine_tokens_to_generate, self.refine_semaphore
            temperature = self.cfg.refine_temperature
            top_p = self.cfg.refine_top_p
        else:
            client, max_tokens, sem = self.llm, self.cfg.gen_tokens_to_generate, self.gen_semaphore
            temperature = self.cfg.gen_temperature
            top_p = self.cfg.gen_top_p
        request_kwargs = {
            "prompt": messages,
            "temperature": self.cfg.inference.temperature if temperature is None else temperature,
            "top_p": self.cfg.inference.top_p if top_p is None else top_p,
            "random_seed": seed,
            "timeout": self.cfg.inference.timeout,
        }
        if max_tokens is not None:
            request_kwargs["tokens_to_generate"] = max_tokens
        async with sem:
            return await client.generate_async(**request_kwargs)

    async def _sink(self, stage, round_idx, row):
        path = os.path.join(self.streaming_output_dir, "rounds", f"R{round_idx}", stage, "output.jsonl")
        lock = self._sink_locks.get(path)
        if lock is None:
            lock = asyncio.Lock()
            self._sink_locks[path] = lock
            os.makedirs(os.path.dirname(path), exist_ok=True)
        line = json.dumps(row, ensure_ascii=False)
        async with lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    async def process_single_datapoint(self, data_point, all_data):
        problem = dict(data_point)
        question = (problem.get("question") or problem.get("problem") or "").strip()
        problem["question"] = question
        if "problem_idx" not in problem or problem["problem_idx"] is None:
            if problem.get("id") is not None:
                problem["problem_idx"] = str(problem["id"])
            elif question:
                problem["problem_idx"] = hash_problem_idx(question)
        problem.setdefault("source_name", "unknown")

        orchestrator = ProblemOrchestrator(
            cfg=self._streaming_cfg(),
            request=self._request,
            sink=self._sink,
            gen_prompt_template=self.gen_template,
            verify_prompt_template=self.verify_template,
            refine_prompt_template=self.refine_template,
            proof_gen_prompt_template=self.proof_gen_template,
            gen_system_prompt=self.gen_system_prompt,
            verify_system_prompt=self.verify_system_prompt,
            refine_system_prompt=self.refine_system_prompt,
            pool_manager=self.pool_manager,
        )
        result = await orchestrator.run(problem)
        if "generation" not in result:
            result["generation"] = ""
        return result


GENERATION_TASK_CLASS = StreamingProofTask


@hydra.main(version_base=None, config_name="base_streaming_generation_config")
def streaming_generation(cfg: StreamingGenerationConfig):
    cfg = StreamingGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    task = StreamingProofTask(cfg)
    task.generate()


if __name__ == "__main__":
    HELP_MESSAGE = get_help_message(
        StreamingGenerationConfig,
        server_params=server_params(),
    )
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        streaming_generation()

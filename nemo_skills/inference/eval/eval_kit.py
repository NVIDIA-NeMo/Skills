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

"""VLMEvalKit integration for NeMo Skills.

This module implements a self-contained generation task that uses VLMEvalKit's
inference and evaluation pipeline. Two modes are supported:

1. Megatron in-process (model_type=mcore): VLMEvalKit's MultiModalMCore loads
   and runs the Megatron model directly. No NeMo Skills server is started.

2. vLLM client (model_type=vllm): NeMo Skills starts a vLLM server normally,
   and VLMEvalKit's VLLMLocal connects to it as a client.

Benchmarks are referenced as eval_kit.<VLMEvalKit_dataset_name> in NeMo Skills,
e.g. --benchmarks eval_kit.MMBench_DEV_EN
"""

import json
import logging
import os
import pickle
import threading
from dataclasses import field
from pathlib import Path

import hydra
from omegaconf import MISSING

try:
    from nemo_skills.inference.generate import GenerationTask
except ImportError:
    # On the cluster, GenerationTask may not be importable due to missing deps
    # (nemo_run, litellm, etc.). The inheritance is only needed for the pipeline's
    # __func__ check which runs locally. On the cluster we just need a base class.
    GenerationTask = object

from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))

# VLMEvalKit (vlmeval) is packaged alongside Skills code by nemo_run when
# NEMO_SKILLS_VLMEVALKIT_PATH is set (see eval.py extra_package_dirs logic).
# It lands at /nemo_run/code/vlmeval/ on the cluster, importable via PYTHONPATH.
# func-timeout is installed at job start via --installation_command in the run script.
# No venv-based requirements are needed (get_generation_requirements returns None).


@nested_dataclass(kw_only=True)
class EvalKitConfig:
    """Configuration for VLMEvalKit generation task."""

    # VLMEvalKit dataset name (injected by pipeline from benchmark name)
    vlm_dataset: str = MISSING

    # Model configuration
    model_type: str = "mcore"  # "mcore" or "vllm"
    model_config: str | None = None  # Path to YAML config for mcore
    load_dir: str | None = None  # Checkpoint directory for mcore
    load_ckpt: str | None = None  # Specific checkpoint for mcore
    server_url: str | None = None  # URL for vLLM server (vllm mode)
    model_name: str | None = None  # Model name for vLLM

    # Inference parameters
    reasoning: bool = False
    temperature: float = 1.0
    top_k: int = 1
    top_p: float = 0.95

    # Video dataset parameters
    nframe: int = 16
    fps: int = -1
    nframe_max: int = -1
    use_subtitle: bool = False
    media_dir: str = "./"

    # Evaluation parameters
    eval_mode: str = "all"  # "all", "infer", or "eval"
    judge: str | None = None
    judge_nproc: int = 4
    judge_retry: int = 3

    # Output configuration (populated by the pipeline)
    work_dir: str = "./outputs"
    output_file: str = ""
    skip_filled: bool = False  # Accepted from pipeline but unused (VLMEvalKit has its own resume)

    # Fields accepted from pipeline but unused by eval_kit (avoids Hydra errors from common_args)
    eval_config: dict = field(default_factory=dict)


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_eval_kit_config", node=EvalKitConfig)


class EvalKitGenerationTask(GenerationTask):
    """Generation task using VLMEvalKit.

    Supports two modes:
    - mcore: Self-contained, no external server. Pipeline sets
      self_contained_task=True so no server is started.
    - vllm: Pipeline starts a vLLM server normally. This task overrides
      ``configure_client_overrides`` to translate the server address into
      eval_kit's flat config fields (``++server_url``, ``++model_name``)
      instead of the standard nested ``++server.*`` overrides.
    """

    # --- Declarative pipeline attributes (read generically by pipeline/eval.py) ---
    CONTAINER_KEY = "eval_kit"
    USE_TORCHRUN = True

    @classmethod
    def is_self_contained(cls, extra_arguments: str = "") -> bool:
        """Self-contained only when user explicitly requests mcore mode.

        Note: EvalKitConfig.model_type defaults to "mcore" at runtime, but
        at submission time we check explicit user intent.  Without the flag
        the pipeline assumes vllm (server-based) mode.
        """
        return "++model_type=mcore" in extra_arguments

    @classmethod
    def configure_client_overrides(cls, *, host: str, port: int, model: str, server_type: str) -> str:
        """Return Hydra overrides for connecting to an already-running server.

        EvalKitConfig uses flat fields (server_url, model_name) rather than
        the standard nested ``server.*`` group, so we translate here.
        """
        return f"++server_url=http://{host}:{port} ++model_name={model} ++model_type=vllm "

    @classmethod
    def get_env_prefix(cls) -> str:
        """Shell env setup prepended before the main command (Megatron/VLMEvalKit needs)."""
        return (
            'export LMUData="${LMUData:-${LMUDATA:-}}" && '
            "export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:${LD_LIBRARY_PATH:-} && "
            "export MKL_THREADING_LAYER=GNU && "
            "export OMP_NUM_THREADS=1 && "
            "export MKL_NUM_THREADS=1 && "
            "ldconfig && "
            # Create empty .env so VLMEvalKit's load_env() doesn't emit ERROR logs.
            "touch /nemo_run/code/.env 2>/dev/null; "
        )

    @classmethod
    def get_extra_package_dirs(cls) -> list[str]:
        """Directories to package alongside nemo_run code (VLMEvalKit vlmeval/)."""
        vlmevalkit_path = os.environ.get("NEMO_SKILLS_VLMEVALKIT_PATH")
        if vlmevalkit_path:
            pkg = os.path.join(vlmevalkit_path, "vlmeval")
            if os.path.isdir(pkg):
                return [pkg]
        return []

    @classmethod
    def get_generation_default_args(cls):
        return ""

    @classmethod
    def get_generation_requirements(cls):
        # VLMEvalKit is installed via --installation_command (pip install from mounted source).
        # No additional venv-based requirements needed.
        return None

    def __init__(self, cfg: EvalKitConfig):
        self.cfg = cfg

        # Validate environment
        lmu_data = os.environ.get("LMUData")
        if not lmu_data:
            raise ValueError(
                "LMUData environment variable must be set for eval_kit benchmarks. "
                "Add LMUData=/mounted/path to your cluster config env_vars."
            )

        # Build model FIRST so that initialize_megatron() sets up the
        # distributed process group before we need dist.barrier() for
        # rank-0-first dataset download.
        if cfg.model_type == "mcore":
            from vlmeval.vlm.multimodal_mcore.model import MultiModalMCore

            if not cfg.model_config:
                raise ValueError("model_config is required for mcore model_type.")
            self.model = MultiModalMCore(
                model_config=cfg.model_config,
                load_dir=cfg.load_dir,
                load_ckpt=cfg.load_ckpt,
                reasoning=cfg.reasoning,
            )
            self.model_name = f"mcore_{Path(cfg.model_config).stem}"
        elif cfg.model_type == "vllm":
            from vlmeval.vlm.vllm_local import VLLMLocal

            if not cfg.server_url:
                raise ValueError("server_url is required for vllm model_type.")
            self.model = VLLMLocal(
                vllm_url=cfg.server_url,
                autospawn=False,
                model_name=cfg.model_name or "vllm_local",
                reasoning_mode=cfg.reasoning,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
            )
            self.model_name = cfg.model_name or "vllm_local"
        else:
            raise ValueError(f"Unknown model_type: {cfg.model_type}. Must be 'mcore' or 'vllm'.")

        # Build dataset after model so the distributed process group is available
        # for the rank-0-first download pattern (run.py:428-433).
        from vlmeval.dataset import build_dataset

        dataset_kwargs = self._build_dataset_kwargs()
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        if world_size > 1:
            import torch.distributed as dist

            if rank == 0:
                build_dataset(cfg.vlm_dataset, **dataset_kwargs)
            dist.barrier()

        self.dataset = build_dataset(cfg.vlm_dataset, **dataset_kwargs)
        if self.dataset is None:
            raise ValueError(f"VLMEvalKit dataset '{cfg.vlm_dataset}' is not valid.")

        self.work_dir = os.path.join(cfg.work_dir, "eval_kit_work", cfg.vlm_dataset)
        os.makedirs(self.work_dir, exist_ok=True)

        # Async JSONL writer state
        self._async_stop = threading.Event()
        self._async_written_indices = set()
        self._async_lock = threading.Lock()
        self._async_thread = None

    # ------------------------------------------------------------------
    # Incremental JSONL writer (mirrors NeMo Skills' -async pattern)
    # ------------------------------------------------------------------

    def _build_index_to_meta(self):
        """Build a lookup from dataset index -> {question, answer} for JSONL rows."""
        meta = {}
        df = self.dataset.data
        for _, row in df.iterrows():
            idx = row["index"]
            meta[idx] = {
                "question": str(row["question"]) if "question" in row.index else "",
                "expected_answer": str(row["answer"]) if "answer" in row.index else "",
            }
        return meta

    def _pkl_to_prediction(self, value):
        """Extract the prediction string from a pkl entry (str or dict)."""
        if isinstance(value, dict) and "prediction" in value:
            return str(value["prediction"])
        return str(value)

    def _async_writer_loop(self, pkl_path, index_meta, output_path, poll_interval=5):
        """Background thread: poll the pkl file and append new entries to JSONL."""
        while not self._async_stop.is_set():
            self._flush_pkl_to_jsonl(pkl_path, index_meta, output_path)
            self._async_stop.wait(timeout=poll_interval)
        # Final flush after inference signals stop
        self._flush_pkl_to_jsonl(pkl_path, index_meta, output_path)

    def _flush_pkl_to_jsonl(self, pkl_path, index_meta, output_path):
        """Read the pkl, find new entries, append them to the JSONL file."""
        if not os.path.exists(pkl_path):
            return
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            # pkl may be mid-write; skip this cycle
            return
        if not isinstance(data, dict):
            return

        new_entries = []
        with self._async_lock:
            for idx, value in data.items():
                if idx not in self._async_written_indices:
                    self._async_written_indices.add(idx)
                    meta = index_meta.get(idx, {})
                    new_entries.append(
                        {
                            "generation": self._pkl_to_prediction(value),
                            "expected_answer": meta.get("expected_answer", ""),
                            "question": meta.get("question", ""),
                        }
                    )

        if new_entries:
            with open(output_path, "a", encoding="utf-8") as f:
                for entry in new_entries:
                    f.write(json.dumps(entry) + "\n")
            LOG.info(
                "Async JSONL: flushed %d new entries (total %d)", len(new_entries), len(self._async_written_indices)
            )

    def _start_async_writer(self):
        """Start the background JSONL writer if output_file is configured."""
        if not self.cfg.output_file:
            return
        rank = int(os.environ.get("RANK", 0))
        if rank != 0:
            return

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ds_name = self.dataset.dataset_name
        pkl_path = os.path.join(self.work_dir, f"0{world_size}_{ds_name}.pkl")

        output_dir = Path(self.cfg.output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Clear any previous async file
        async_path = self.cfg.output_file
        if os.path.exists(async_path):
            os.remove(async_path)

        index_meta = self._build_index_to_meta()

        self._async_stop.clear()
        self._async_written_indices.clear()
        self._async_thread = threading.Thread(
            target=self._async_writer_loop,
            args=(pkl_path, index_meta, async_path),
            daemon=True,
        )
        self._async_thread.start()
        LOG.info("Started async JSONL writer, monitoring %s", pkl_path)

    def _stop_async_writer(self):
        """Stop the background writer and wait for final flush."""
        if self._async_thread is None:
            return
        self._async_stop.set()
        self._async_thread.join(timeout=30)
        self._async_thread = None
        LOG.info("Async JSONL writer stopped (%d entries written)", len(self._async_written_indices))

    def _build_dataset_kwargs(self):
        """Build dataset kwargs mirroring VLMEvalKit's run.py:390-425."""
        from vlmeval.smp import listinstr

        kwargs = {}
        ds = self.cfg.vlm_dataset

        if ds in ["MMLongBench_DOC", "DUDE", "DUDE_MINI", "SLIDEVQA", "SLIDEVQA_MINI"]:
            kwargs["model"] = self.cfg.model_name or self.cfg.model_config or ""

        if ds in (
            "Video-MME",
            "Video-MME-With-Audio",
            "WorldSense-AVLM",
            "MetropolisVideoDataset",
            "WorldSense",
            "avqa_val",
        ):
            kwargs["use_subtitle"] = self.cfg.use_subtitle
        if ds in (
            "Video-MME",
            "MetropolisVideoDataset",
            "MLVU",
            "LongVideoBench",
            "MMBench-Video",
            "MVBench",
            "MLVU_MCQ",
            "PAI-Bench-U",
        ):
            kwargs["nframe"] = self.cfg.nframe
        if ds in [
            "Video-MME",
            "MLVU",
            "LongVideoBench",
            "WorldSense",
            "avqa_val",
            "MMBench-Video",
            "MVBench",
            "MLVU_MCQ",
            "PAI-Bench-U",
        ]:
            kwargs["fps"] = self.cfg.fps
        if ds in [
            "Video-MME",
            "MLVU",
            "LongVideoBench",
            "WorldSense",
            "avqa_val",
            "MLVU_MCQ",
            "MMBench-Video",
            "PAI-Bench-U",
        ]:
            kwargs["nframe_max"] = self.cfg.nframe_max
        if ds in ["ANet-RTL", "Charades-STA"]:
            kwargs["nframe"] = self.cfg.nframe

        if listinstr(["Video-MME-With-Audio", "DailyOmni", "WorldSense-AVLM", "JensenKeyNote"], ds):
            kwargs["media_dir"] = self.cfg.media_dir

        return kwargs

    def generate(self):
        """Run VLMEvalKit inference and evaluation."""
        from vlmeval.inference import infer_data_job
        from vlmeval.inference_mt import infer_data_job_mt
        from vlmeval.inference_video import infer_data_job_video
        from vlmeval.smp import get_pred_file_format

        dataset = self.dataset
        ds_name = dataset.dataset_name
        pred_format = get_pred_file_format()
        result_file_base = f"{self.model_name}_{ds_name}.{pred_format}"

        rank = int(os.environ.get("RANK", 0))

        # Start incremental JSONL writer before inference begins
        self._start_async_writer()

        # Dispatch to correct inference function (mirrors run.py:453-488)
        try:
            if self.cfg.eval_mode != "eval":
                if dataset.MODALITY == "VIDEO":
                    self.model = infer_data_job_video(
                        model=self.model,
                        work_dir=self.work_dir,
                        model_name=self.model_name,
                        dataset=dataset,
                        result_file_name=result_file_base,
                        strip_think=not self.cfg.reasoning,
                        reasoning_flag=self.cfg.reasoning,
                    )
                elif dataset.TYPE == "MT":
                    self.model = infer_data_job_mt(
                        model=self.model,
                        work_dir=self.work_dir,
                        model_name=self.model_name,
                        dataset=dataset,
                    )
                else:
                    self.model = infer_data_job(
                        model=self.model,
                        work_dir=self.work_dir,
                        model_name=self.model_name,
                        dataset=dataset,
                        strip_think=not self.cfg.reasoning,
                        reasoning_flag=self.cfg.reasoning,
                    )
        finally:
            self._stop_async_writer()

        # Evaluate (mirrors run.py:490-548)
        eval_result = {}
        if self.cfg.eval_mode != "infer" and rank == 0:
            from vlmeval.smp import get_pred_file_path

            result_file = get_pred_file_path(self.work_dir, self.model_name, ds_name, use_env_format=True)
            judge_kwargs = {
                "nproc": self.cfg.judge_nproc,
                "verbose": False,
                "retry": self.cfg.judge_retry,
            }
            if self.cfg.judge:
                judge_kwargs["model"] = self.cfg.judge

            if os.path.exists(result_file):
                try:
                    eval_result = dataset.evaluate(result_file, **judge_kwargs)
                except KeyError as e:
                    if e.args and e.args[0] == "model":
                        LOG.warning(
                            "Dataset %s requires a judge model for evaluation (e.g. MathVista). "
                            "Skipping evaluation. Set ++judge=<model> (e.g. gpt-4o) to enable. "
                            "Inference output was still written.",
                            ds_name,
                        )
                        eval_result = {}
                    else:
                        raise
                if eval_result is None:
                    eval_result = {}

        # Convert to NeMo Skills format and write outputs (rank 0 only)
        if rank == 0:
            self._convert_to_nemo_skills_format(eval_result)

            # Write .done file for pipeline tracking
            if self.cfg.output_file:
                Path(f"{self.cfg.output_file}.done").touch()

    def _convert_to_nemo_skills_format(self, eval_result):
        """Rewrite the final ordered JSONL output and eval_kit_metrics.json.

        The async writer has already been producing incremental JSONL during
        inference.  Here we overwrite with the authoritative, properly-ordered
        result that VLMEvalKit merged from all ranks.
        """
        if not self.cfg.output_file:
            return

        from vlmeval.smp import get_pred_file_path
        from vlmeval.smp import load as vlm_load

        output_dir = Path(self.cfg.output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write JSONL (required by summarize_results to find output*jsonl files)
        result_file = get_pred_file_path(
            self.work_dir,
            self.model_name,
            self.dataset.dataset_name,
            use_env_format=True,
        )
        if os.path.exists(result_file):
            df = vlm_load(result_file)
            with open(self.cfg.output_file, "w", encoding="utf-8") as f:
                for _, row in df.iterrows():
                    entry = {
                        "generation": str(row["prediction"]) if "prediction" in row.index else "",
                        "expected_answer": str(row["answer"]) if "answer" in row.index else "",
                        "question": str(row["question"]) if "question" in row.index else "",
                    }
                    f.write(json.dumps(entry) + "\n")
            LOG.info("Wrote final ordered JSONL to %s (%d entries)", self.cfg.output_file, len(df))
        else:
            LOG.warning("VLMEvalKit result file not found: %s", result_file)

        # Write aggregate metrics for EvalKitMetrics to read
        # eval_result can be a dict or a pandas DataFrame (e.g. ASR); avoid "if eval_result" for DataFrame
        if eval_result is not None:
            metrics_data = eval_result if isinstance(eval_result, dict) else {"result": str(eval_result)}
            metrics_path = output_dir / "eval_kit_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, indent=2, default=str)
            LOG.info("Wrote eval_kit metrics to %s", metrics_path)


GENERATION_TASK_CLASS = EvalKitGenerationTask


@hydra.main(version_base=None, config_name="base_eval_kit_config")
def main(cfg: EvalKitConfig):
    cfg = EvalKitConfig(_init_nested=True, **cfg)
    task = EvalKitGenerationTask(cfg)
    task.generate()


if __name__ == "__main__":
    main()

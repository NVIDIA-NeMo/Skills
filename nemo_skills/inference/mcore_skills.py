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

"""NeMo Skills generation via VLMEvalKit MultiModalMCore in-process.

This module implements Option A from the plan: read NeMo Skills JSONL, fill
prompts with NeMo Skills prompt config, run inference through MultiModalMCore
synchronously, write NeMo Skills-format JSONL. No HTTP server; evaluation
remains NeMo Skills metrics on the output.

When run with torchrun (multi-GPU), all ranks participate in model.generate();
only rank 0 performs file I/O.
"""

import json
import logging
import os
import re
from dataclasses import field
from pathlib import Path

import hydra
from omegaconf import MISSING
from tqdm import tqdm

from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import chunk_data, get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))

try:
    from nemo_skills.inference.generate import GenerationTask
except ImportError:
    GenerationTask = None

if GenerationTask is not None:
    _get_server_command_fn = GenerationTask.get_server_command_fn
else:

    @classmethod
    def _get_server_command_fn(cls):
        from nemo_skills.pipeline.utils import get_server_command

        return get_server_command


@nested_dataclass(kw_only=True)
class MegatronMCoreConfig:
    """Configuration for MegatronMCore NeMo Skills generation."""

    input_file: str = MISSING
    output_file: str = MISSING

    # Prompt config for text-only data (used by fill_prompt). Not needed when the
    # input JSONL already contains OpenAI-format 'messages' (e.g. asr-leaderboard).
    prompt_config: str | None = None

    # Tokenizer for prompt filling (format_as_string=True). HF model name or path.
    # Required when prompt_config is set; optional for messages-only data.
    tokenizer: str | None = None

    # MultiModalMCore model
    model_config: str = MISSING
    load_dir: str | None = None
    load_ckpt: str | None = None
    reasoning: bool = False

    # Prompt options (mirror GenerationTaskConfig where needed)
    code_tags: str | None = None
    examples_type: str | None = None
    system_message: str | None = None
    start_assistant_response_key: str | None = None
    chat_template_kwargs: dict = field(default_factory=dict)

    # Base directory to resolve relative audio/image paths (e.g. NEMO_SKILLS_DATA_DIR).
    data_dir: str = ""

    # Generation limits and resume
    max_samples: int = -1
    skip_filled: bool = False
    num_chunks: int | None = None
    chunk_id: int | None = None

    # Output
    generation_key: str = "generation"
    add_generation_stats: bool = True
    async_position_key: str = "_async_position"

    dry_run: bool = False

    # Dataset name passed to MultiModalMCore.generate() — used by VLMEvalKit internally
    # for dataset-specific logic (e.g. video tile config). Defaults to "nemo_skills".
    dataset_name: str = "nemo_skills"

    # Accepted from pipeline/dataset modules but unused by mcore_skills (avoid Hydra errors).
    # These come via ++key=value overrides from dataset modules (e.g. asr-leaderboard).
    eval_config: dict = field(default_factory=dict)
    eval_type: str | None = None
    prompt_format: str | None = None
    enable_audio: bool = False


def _make_mcore_model(cfg: MegatronMCoreConfig):
    from vlmeval.vlm.multimodal_mcore.model import MultiModalMCore

    return MultiModalMCore(
        model_config=cfg.model_config,
        load_dir=cfg.load_dir,
        load_ckpt=cfg.load_ckpt,
        reasoning=cfg.reasoning,
    )


class MegatronMCoreGenerationTask:
    """Generation task using NeMo Skills data + prompts and VLMEvalKit MultiModalMCore in-process."""

    get_server_command_fn = _get_server_command_fn

    # --- Declarative pipeline attributes (read generically by pipeline/eval.py) ---
    CONTAINER_KEY = "eval_kit"
    USE_TORCHRUN = True
    # Metrics are computed by VLMEvalKit (asr_wer etc.) and saved as
    # eval_kit_metrics.json — tell the summarize step to use EvalKitMetrics.
    METRICS_TYPE_OVERRIDE = "eval_kit"

    @classmethod
    def is_self_contained(cls, extra_arguments: str = "") -> bool:
        """Always self-contained (in-process MultiModalMCore, no HTTP server)."""
        return True

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
        return None

    def __init__(self, cfg: MegatronMCoreConfig):
        self.cfg = cfg
        # Prompt is only needed for text-only data (no 'messages' field).
        # For multimodal data with OpenAI-format messages, _build_mcore_messages
        # extracts content directly — no prompt template required.
        if cfg.prompt_config:
            self.prompt = get_prompt(
                prompt_config=cfg.prompt_config,
                tokenizer=cfg.tokenizer,
                code_tags=cfg.code_tags,
                examples_type=cfg.examples_type,
                system_message=cfg.system_message,
            )
        else:
            self.prompt = None
        self.model = _make_mcore_model(cfg)

    def load_data(self):
        data = []
        with open(self.cfg.input_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                data.append(json.loads(line))
        if self.cfg.num_chunks is not None and self.cfg.chunk_id is not None:
            data, self.cfg.output_file = chunk_data(data, self.cfg.output_file, self.cfg.chunk_id, self.cfg.num_chunks)
            LOG.info(
                "Chunking: %d chunks, processing chunk %d; samples in chunk: %d",
                self.cfg.num_chunks,
                self.cfg.chunk_id,
                len(data),
            )
        if self.cfg.max_samples > 0:
            data = data[: self.cfg.max_samples]
        return data

    def skip_completed_samples(self, data: list) -> list:
        if not self.cfg.skip_filled or not Path(self.cfg.output_file).exists():
            return data
        filled = 0
        with open(self.cfg.output_file, "rt", encoding="utf-8") as fin:
            for _ in fin:
                filled += 1
        if filled >= len(data):
            return []
        return data[filled:]

    def fill_prompt(self, data_point: dict, data: list) -> str:
        from copy import deepcopy

        data_point = deepcopy(data_point)
        filled = self.prompt.fill(
            data_point,
            start_assistant_response_key=self.cfg.start_assistant_response_key,
            chat_template_kwargs=self.cfg.chat_template_kwargs or {},
            format_as_string=True,
        )
        return filled if isinstance(filled, str) else str(filled)

    def _get_data_dir(self) -> str:
        """Return the effective data_dir from cfg or eval_config."""
        data_dir = getattr(self.cfg, "data_dir", None) or ""
        if not data_dir and getattr(self.cfg, "eval_config", None):
            data_dir = self.cfg.eval_config.get("data_dir") or ""
        return data_dir

    def _resolve_path(self, path: str) -> str:
        """Resolve a media file path, handling relative paths and mount mismatches.

        1. Relative paths are joined with data_dir.
        2. Absolute paths that don't exist on disk are retried relative to data_dir
           (handles mount mismatches, e.g. JSONL has /dataset/... but data is at /data/...).
        """
        if not path:
            return path
        data_dir = self._get_data_dir()
        if not os.path.isabs(path):
            if data_dir:
                return os.path.join(data_dir, path)
            return path
        # Absolute path — use as-is if it exists
        if os.path.exists(path):
            return path
        # Absolute path doesn't exist — try stripping the first directory component
        # and re-rooting under data_dir (e.g. /dataset/asr-leaderboard/... → /data/asr-leaderboard/...)
        if data_dir:
            # Strip leading /mount_name/ to get the relative portion
            parts = path.strip("/").split("/", 1)
            if len(parts) == 2:
                relative = parts[1]
                candidate = os.path.join(data_dir, relative)
                if os.path.exists(candidate):
                    return candidate
        return path

    def _build_mcore_messages(self, data_point: dict) -> list | None:
        """Convert a NeMo Skills data point into MultiModalMCore message list.

        If the data point has a 'messages' field (OpenAI format), converts it to
        list[dict] with types: "text", "image", "sound".

        Only user/assistant message text is included — system messages are skipped
        because MultiModalMCore's generate_inner() builds its own prompt template
        with system/user roles internally.

        If no 'messages' field, returns None (caller should use fill_prompt for text-only).
        """
        messages = data_point.get("messages")
        if not messages:
            return None

        mcore: list[dict] = []
        text_parts: list[str] = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Skip system messages — generate_inner builds its own system prompt.
            if role == "system":
                continue

            # Audio: single or multiple
            if "audio" in msg:
                audio = msg["audio"]
                if isinstance(audio, dict) and "path" in audio:
                    path = self._resolve_path(audio["path"])
                    mcore.append({"type": "sound", "value": path, "sample_rate": 16000})
            if "audios" in msg:
                for audio in msg["audios"]:
                    if isinstance(audio, dict) and "path" in audio:
                        path = self._resolve_path(audio["path"])
                        mcore.append({"type": "sound", "value": path, "sample_rate": 16000})

            # Content: str or list of content items (text, image_url)
            if isinstance(content, str):
                if content.strip():
                    text_parts.append(content.strip())
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text" and "text" in item:
                            text_parts.append(item["text"].strip())
                        elif item.get("type") == "image_url":
                            image_url = item.get("image_url") or {}
                            url = image_url.get("url", "")
                            if url.startswith("file://"):
                                path = url[7:]
                            else:
                                path = url
                            if path:
                                path = self._resolve_path(path)
                                mcore.append({"type": "image", "value": path})

        combined_text = "\n".join(t for t in text_parts if t)
        if combined_text:
            mcore.append({"type": "text", "value": combined_text})

        if not mcore:
            return None
        return mcore

    def dump_outputs(self, outputs: list, fout):
        for out in outputs:
            fout.write(json.dumps(out) + "\n")

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        """Strip <think>...</think> tags (including empty ones) from model output."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _generate_for_sample(self, data_point: dict, data: list) -> str:
        """Run model inference for a single data point. Returns generated text."""
        message_list = self._build_mcore_messages(data_point)
        if message_list is not None:
            raw = self.model.generate(message_list, dataset=self.cfg.dataset_name)
            return self._strip_thinking_tags(raw)
        if self.prompt is None:
            raise ValueError(
                "Data point has no 'messages' field and prompt_config is not set. "
                "Either provide ++prompt_config for text-only data or ensure "
                "the input JSONL contains OpenAI-format 'messages'."
            )
        prompt_str = self.fill_prompt(data_point, data)
        raw = self.model.generate(
            [{"type": "text", "value": prompt_str}],
            dataset=self.cfg.dataset_name,
        )
        return self._strip_thinking_tags(raw)

    def generate(self):
        import sys

        # Use Megatron DP rank/size for data sharding (matches VLMEvalKit pattern).
        # With data_parallel=True in generate_and_post_process, each DP rank runs
        # generation independently on its shard while TP ranks synchronise internally.
        dp_rank = self.model.get_dp_rank()
        dp_size = self.model.get_dp_size()

        output_dir = Path(self.cfg.output_file).absolute().parent
        if dp_rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)

        data = self.load_data()
        data = self.skip_completed_samples(data)
        if not data:
            if dp_rank == 0:
                LOG.info("No data to process, skipping generation")
            return
        if self.cfg.dry_run:
            if dp_rank == 0:
                LOG.info("Dry run: would process %d samples", len(data))
            return

        # Round-robin shard by dp_rank (same strategy as VLMEvalKit infer_data).
        my_indices = list(range(dp_rank, len(data), dp_size))
        my_data = [data[i] for i in my_indices]

        if dp_rank == 0:
            LOG.info(
                "Data parallelism: dp_size=%d, total=%d, this rank=%d samples",
                dp_size,
                len(data),
                len(my_data),
            )

        # Per-rank output file — visible during the run so progress can be
        # monitored (e.g. ``wc -l output_rank*.jsonl``).  Contains a
        # ``_dp_global_idx`` field used for ordered merging at the end.
        rank_file = output_dir / f"output_rank{dp_rank}.jsonl"

        # Suppress VLMEvalKit's per-sample print() on non-primary DP ranks to
        # avoid 8x duplicate output in logs.
        _real_stdout = sys.stdout
        if dp_rank != 0:
            sys.stdout = open(os.devnull, "w")

        try:
            with open(rank_file, "w", encoding="utf-8") as fout:
                iterator = tqdm(my_data, desc=f"mcore_skills[dp{dp_rank}]") if dp_rank == 0 else my_data
                for local_idx, data_point in enumerate(iterator):
                    global_idx = my_indices[local_idx]
                    gen = self._generate_for_sample(data_point, data)
                    output = {
                        "_dp_global_idx": global_idx,
                        self.cfg.generation_key: gen,
                        **{k: v for k, v in data_point.items() if k != self.cfg.async_position_key},
                    }
                    fout.write(json.dumps(output) + "\n")
                    fout.flush()
        finally:
            if dp_rank != 0:
                sys.stdout.close()
                sys.stdout = _real_stdout

        # Barrier: wait for all DP ranks to finish writing.
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()

        # Rank 0 merges per-rank files into the final ordered output.
        if dp_rank == 0:
            all_results: dict[int, str] = {}
            for r in range(dp_size):
                rf = output_dir / f"output_rank{r}.jsonl"
                if rf.exists() and rf.stat().st_size > 0:
                    with open(rf, "rt", encoding="utf-8") as fin:
                        for line in fin:
                            entry = json.loads(line)
                            idx = entry.pop("_dp_global_idx")
                            all_results[idx] = json.dumps(entry)

            mode = "a" if self.cfg.skip_filled and Path(self.cfg.output_file).exists() else "w"
            merged_lines = [all_results[idx] + "\n" for idx in sorted(all_results.keys())]
            with open(self.cfg.output_file, mode, encoding="utf-8") as fout:
                fout.writelines(merged_lines)
            LOG.info(
                "Merged %d results from %d DP ranks into %s",
                len(all_results),
                dp_size,
                self.cfg.output_file,
            )

            # Clean up per-rank files after successful merge.
            for r in range(dp_size):
                rf = output_dir / f"output_rank{r}.jsonl"
                rf.unlink(missing_ok=True)

            # Evaluate using VLMEvalKit (same as eval_kit.py does).
            # Done BEFORE marking .done so failed metrics prevent false completion.
            self._evaluate_results()

            Path(f"{self.cfg.output_file}.done").touch()

    def _evaluate_results(self):
        """Compute metrics using VLMEvalKit's evaluation functions.

        Uses the same asr_wer() that eval_kit.py calls via dataset.evaluate(),
        so metrics are identical.  Saves eval_kit_metrics.json (consumed by
        EvalKitMetrics in the summarize step).
        """
        output_file = self.cfg.output_file
        if not output_file or not Path(output_file).exists():
            return

        output_path = Path(output_file)

        try:
            from vlmeval.dataset.avlm.utils import asr_wer

            # Read entries and build VLMEvalKit-format results list
            entries = []
            results = []
            with open(output_file, "rt", encoding="utf-8") as fin:
                for line in fin:
                    entry = json.loads(line)
                    # Strip leftover <think> tags (older runs may have them)
                    gen_key = self.cfg.generation_key
                    gen = entry.get(gen_key, "")
                    cleaned = self._strip_thinking_tags(gen)
                    if cleaned != gen:
                        entry[gen_key] = cleaned
                    entries.append(entry)
                    results.append(
                        {
                            "gt": entry.get("expected_answer", ""),
                            "pred": entry[gen_key],
                        }
                    )

            # Re-write output.jsonl with cleaned generations
            with open(output_file, "w", encoding="utf-8") as fout:
                for entry in entries:
                    fout.write(json.dumps(entry) + "\n")

            # Compute WER using VLMEvalKit (same function as eval_kit path)
            wer_score = asr_wer(results)
            LOG.info("ASR WER: %.2f%%", wer_score)

            # Save as eval_kit_metrics.json (same format eval_kit.py writes)
            metrics = {"wer": wer_score}
            metrics_file = output_path.parent / "eval_kit_metrics.json"
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            LOG.info("Metrics saved to %s", metrics_file)

        except ImportError:
            LOG.warning(
                "VLMEvalKit asr_wer not available — skipping eval-kit-style metrics. "
                "The summarize_results job will compute metrics separately."
            )
        except Exception:
            LOG.exception("Inline metrics computation failed")


GENERATION_TASK_CLASS = MegatronMCoreGenerationTask

cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_mcore_skills_config", node=MegatronMCoreConfig)


@hydra.main(version_base=None, config_name="base_mcore_skills_config")
def main(cfg: MegatronMCoreConfig):
    cfg = MegatronMCoreConfig(_init_nested=True, **cfg)
    task = MegatronMCoreGenerationTask(cfg)
    task.generate()


if __name__ == "__main__":
    import nemo_skills.utils as utils

    utils.setup_logging()
    main()

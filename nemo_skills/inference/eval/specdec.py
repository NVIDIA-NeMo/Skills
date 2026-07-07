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

import contextlib
import glob
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import hydra
import requests

from nemo_skills.inference.generate import (
    GenerationTask,
    GenerationTaskConfig,
)
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


class SpecDecodeMetricsError(Exception):
    """Exception raised when fetching speculative decoding metrics fails."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


# ---------------------------------------------------------------------------
# Speculative decoding metrics from the server's Prometheus endpoint
# ---------------------------------------------------------------------------


@dataclass
class SpecDecodeMetrics:
    """Unified speculative decoding snapshot scraped from ``/metrics``."""

    # VLLM counters
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    accepted_per_pos: dict[int, int] = field(default_factory=dict)

    # SGLang gauges and counters
    spec_accept_length: float = 0.0
    spec_accept_rate: float = 0.0
    num_requests: int = 0
    generation_tokens: int = 0


def _fetch_metrics_text(base_url: str) -> str | None:
    """Fetch raw Prometheus text from ``/metrics`` endpoint."""
    metrics_url = f"{base_url.rstrip('/')}/metrics"
    try:
        response = requests.get(metrics_url, timeout=30)
        if response.status_code != 200:
            LOG.warning("Metrics endpoint returned status %d", response.status_code)
            return None
        return response.text
    except requests.RequestException as exc:
        LOG.warning("Failed to fetch metrics from %s: %s", metrics_url, exc)
        return None


def fetch_vllm_spec_decode_metrics(base_url: str) -> SpecDecodeMetrics:
    """Fetch speculative decoding metrics from the server's ``/metrics`` endpoint.

    Parses Prometheus text-format exposition looking for VLLM
    ``vllm:spec_decode_*`` counters including per-position acceptance counts.

    Args:
        base_url: Server root URL, e.g. ``http://127.0.0.1:5000``.

    Returns:
        :class:`SpecDecodeMetrics` with the scraped counters.
    """
    text = _fetch_metrics_text(base_url)
    if text is None:
        message = "Failed to fetch metrics from the server"
        LOG.error(message)
        raise SpecDecodeMetricsError(message)

    metrics = SpecDecodeMetrics()
    found_spec_decode = False
    pos_label = 'position="'

    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("vllm:spec_decode"):
            continue

        found_spec_decode = True
        if "_created" in line:
            continue

        parts = line.split()
        if not parts:
            continue

        with contextlib.suppress(ValueError):
            metric_value = int(float(parts[-1]))
            if "num_drafts" in line:
                metrics.num_drafts += metric_value
            elif "num_draft_tokens" in line:
                metrics.num_draft_tokens += metric_value
            elif "num_accepted_tokens_per_pos" in line:
                if pos_label in line:
                    start = line.index(pos_label) + len(pos_label)
                    end = line.index('"', start)
                    pos = int(line[start:end])
                    metrics.accepted_per_pos[pos] = metrics.accepted_per_pos.get(pos, 0) + metric_value
            elif "num_accepted_tokens" in line:
                metrics.num_accepted_tokens += metric_value

    if not found_spec_decode:
        message = "No vllm:spec_decode_* metrics found on the server (speculative decoding may not be enabled)."
        LOG.error(message)
        raise SpecDecodeMetricsError(message)

    return metrics


def find_sglang_metrics_file(metrics_dir: str) -> str | None:
    """Find the most recent SGLang metrics file in the given directory.

    SGLang creates files like ``sglang-request-metrics-YYYYMMDD_HH.log``.

    Args:
        metrics_dir: Directory containing SGLang metrics files.

    Returns:
        Path to the most recent metrics file, or ``None`` if not found.
    """
    if not os.path.isdir(metrics_dir):
        LOG.warning("SGLang metrics directory does not exist: %s", metrics_dir)
        return None

    pattern = os.path.join(metrics_dir, "sglang-request-metrics-*.log")
    files = glob.glob(pattern)
    if not files:
        LOG.warning("No SGLang metrics files found matching pattern: %s", pattern)
        return None

    # Return the most recently modified file
    latest_file = max(files, key=os.path.getmtime)
    LOG.info("Using SGLang metrics file: %s", latest_file)
    return latest_file


def fetch_sglang_spec_decode_metrics(base_url: str) -> SpecDecodeMetrics:
    """Fetch speculative decoding metrics from an SGLang server's ``/metrics`` endpoint.

    Parses Prometheus text-format exposition looking for ``sglang:spec_accept_*``
    gauges and ``sglang:num_requests_total`` / ``sglang:generation_tokens_total``
    counters.

    Args:
        base_url: Server root URL, e.g. ``http://127.0.0.1:5000``.

    Returns:
        :class:`SpecDecodeMetrics` with the scraped values.
    """
    text = _fetch_metrics_text(base_url)
    if text is None:
        message = "Failed to fetch metrics from the server"
        LOG.error(message)
        raise SpecDecodeMetricsError(message)

    metrics = SpecDecodeMetrics()
    found_spec = False

    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        with contextlib.suppress(ValueError):
            if "sglang:spec_accept_length{" in line or line.startswith("sglang:spec_accept_length "):
                metrics.spec_accept_length = float(parts[-1])
                found_spec = True
            elif "sglang:spec_accept_rate{" in line or line.startswith("sglang:spec_accept_rate "):
                metrics.spec_accept_rate = float(parts[-1])
                found_spec = True
            elif "sglang:num_requests_total{" in line or line.startswith("sglang:num_requests_total "):
                metrics.num_requests = int(float(parts[-1]))
            elif "sglang:generation_tokens_total{" in line or line.startswith("sglang:generation_tokens_total "):
                metrics.generation_tokens = int(float(parts[-1]))

    if not found_spec:
        message = "No sglang:spec_accept_* metrics found on the server (speculative decoding may not be enabled)."
        LOG.error(message)
        raise SpecDecodeMetricsError(message)
    return metrics


def _build_specdec_stats(
    *,
    num_drafts: int,
    draft_tokens: int,
    accepted_tokens: int,
    acceptance_rate_fraction: float,
    acceptance_length: float,
    per_position_acceptance_rates: list[float] | None = None,
) -> dict[str, Any]:
    """Build a normalized spec-decode payload for evaluator injection."""
    return {
        "num_drafts": num_drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted_tokens,
        "acceptance_rate": acceptance_rate_fraction * 100,
        "acceptance_length": acceptance_length,
        "per_position_acceptance_rates": per_position_acceptance_rates or [],
    }


def _compute_weighted_delta(
    *,
    before_avg: float,
    after_avg: float,
    before_count: int,
    after_count: int,
) -> float | None:
    """Compute benchmark-only average from cumulative weighted averages."""
    delta_count = after_count - before_count
    if delta_count <= 0:
        return None
    if before_count == 0:
        return after_avg
    weighted_after = after_avg * after_count
    weighted_before = before_avg * before_count
    return (weighted_after - weighted_before) / delta_count


def compute_sglang_spec_decode_delta(
    before: SpecDecodeMetrics,
    after: SpecDecodeMetrics,
) -> dict[str, Any] | None:
    """Compute benchmark-specific acceptance metrics from two SGLang snapshots.

    SGLang exposes ``spec_accept_length`` and ``spec_accept_rate`` as **gauges**
    (running averages from the server's perspective).  Combined with the
    ``num_requests_total`` and ``generation_tokens_total`` **counters** we can
    back out the benchmark-specific averages::

        weighted_al_after  = al_after  x n_after
        weighted_al_before = al_before x n_before
        benchmark_al = (weighted_al_after - weighted_al_before)
                       / (n_after - n_before)

    If the before snapshot had 0 requests (fresh server), the after values are
    returned as-is.

    Args:
        before: Snapshot taken before generation.
        after: Snapshot taken after generation.

    Returns:
        Dictionary with ``acceptance_length``, ``acceptance_rate``, counters,
        etc., or ``None`` if no requests were generated.
    """
    delta_requests = after.num_requests - before.num_requests
    delta_gen_tokens = after.generation_tokens - before.generation_tokens

    if delta_requests <= 0:
        LOG.warning(
            "SGLang: no new requests between before (%d) and after (%d) snapshots.",
            before.num_requests,
            after.num_requests,
        )
        return None

    acceptance_length = _compute_weighted_delta(
        before_avg=before.spec_accept_length,
        after_avg=after.spec_accept_length,
        before_count=before.num_requests,
        after_count=after.num_requests,
    )
    acceptance_rate_fraction = _compute_weighted_delta(
        before_avg=before.spec_accept_rate,
        after_avg=after.spec_accept_rate,
        before_count=before.num_requests,
        after_count=after.num_requests,
    )
    if acceptance_length is None or acceptance_rate_fraction is None:
        LOG.warning("SGLang: failed to compute weighted delta from request counters.")
        return None

    LOG.info(
        "SGLang Prometheus delta: requests=%d, gen_tokens=%d, acceptance_length=%.4f, acceptance_rate=%.2f%%",
        delta_requests,
        delta_gen_tokens,
        acceptance_length,
        acceptance_rate_fraction * 100,
    )

    return _build_specdec_stats(
        num_drafts=delta_requests,
        draft_tokens=delta_gen_tokens,
        accepted_tokens=int(delta_gen_tokens * acceptance_rate_fraction) if delta_gen_tokens > 0 else 0,
        acceptance_rate_fraction=acceptance_rate_fraction,
        acceptance_length=acceptance_length,
    )


def compute_vllm_spec_decode_delta(
    before: SpecDecodeMetrics,
    after: SpecDecodeMetrics,
) -> dict[str, Any] | None:
    """Compute the delta of speculative decoding counters between two snapshots.

    Uses the before/after pattern to isolate metrics to only the generation
    run (since Prometheus counters are cumulative).

    Computes:
        * **acceptance_rate** — ``delta_accepted / delta_draft_tokens * 100``
        * **acceptance_length** — ``1 + delta_accepted / delta_drafts``
          (the "+1" accounts for the target model's verified token that
          is always emitted even on a full rejection)
        * **per_position_acceptance_rates** — per-position acceptance
          probability, computed as ``delta_per_pos[i] / delta_drafts``

    Args:
        before: Counters snapshot taken before generation.
        after: Counters snapshot taken after generation.

    Returns:
        Dictionary with computed spec decode statistics, or ``None`` if the
        delta contains no meaningful data.
    """
    delta_drafts = after.num_drafts - before.num_drafts
    delta_draft_tokens = after.num_draft_tokens - before.num_draft_tokens
    delta_accepted = after.num_accepted_tokens - before.num_accepted_tokens

    # Compute per-position acceptance rates
    per_pos_rates: list[float] = []
    if delta_drafts > 0:
        positions = sorted(set(before.accepted_per_pos.keys()) | set(after.accepted_per_pos.keys()))
        for pos in positions:
            before_val = before.accepted_per_pos.get(pos, 0)
            after_val = after.accepted_per_pos.get(pos, 0)
            delta_pos = after_val - before_val
            per_pos_rates.append(delta_pos / delta_drafts)

    if delta_draft_tokens <= 0:
        LOG.warning(
            "No speculative decoding activity detected during generation "
            "(delta_draft_tokens=%d). Metrics will be empty.",
            delta_draft_tokens,
        )
        return None

    acceptance_rate_fraction = delta_accepted / delta_draft_tokens
    acceptance_length = 1 + delta_accepted / delta_drafts if delta_drafts > 0 else 0.0

    return _build_specdec_stats(
        num_drafts=delta_drafts,
        draft_tokens=delta_draft_tokens,
        accepted_tokens=delta_accepted,
        acceptance_rate_fraction=acceptance_rate_fraction,
        acceptance_length=acceptance_length,
        per_position_acceptance_rates=per_pos_rates,
    )


# ---------------------------------------------------------------------------
# Generation config & task
# ---------------------------------------------------------------------------


@nested_dataclass(kw_only=True)
class SpecdecGenerationConfig(GenerationTaskConfig):
    """SPEED-Bench generation config for speculative decoding evaluation.

    Extends the standard generation config to inject server connection
    information into the evaluator so it can query the server's ``/metrics``
    Prometheus endpoint after generation completes.

    For the full list of supported parameters, use
    ``python -m nemo_skills.inference.generate --help``
    """

    # Directory to write SGLang metrics to.  If not specified, will use a
    # temporary directory created at launch.  This is the only reliable way
    # to share the tempdir between the pipeline process (which builds the
    # server command) and this generation worker process.
    metrics_file_dir: str | None = None
    max_concurrent_requests: int = 32

    def _post_init_validate_server(self):
        super()._post_init_validate_server()
        assert self.server["server_type"] in ["sglang", "vllm"], (
            f"server_type must be either 'sglang' or 'vllm' for specdec generation, got '{self.server['server_type']}'"
        )


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_specdec_generation_config", node=SpecdecGenerationConfig)


class SpecdecGenerationTask(GenerationTask):
    """Custom generation task for SPEED-Bench speculative decoding evaluation.

    Captures speculative decoding counters from the server's ``/metrics``
    Prometheus endpoint *before* and *after* generation, then computes the
    delta to derive accurate acceptance length (AL), acceptance rate, and
    per-position conditional acceptance rates.
    """

    _sglang_metrics_dir: str | None = None

    def __init__(self, cfg: SpecdecGenerationConfig):
        super().__init__(cfg)
        self._before_metrics: SpecDecodeMetrics | None = None

    @classmethod
    def _ensure_sglang_metrics_dir(cls) -> str:
        """Return (and lazily create) a unique temp directory for SGLang metrics."""
        if cls._sglang_metrics_dir is None:
            import tempfile

            cls._sglang_metrics_dir = tempfile.mkdtemp(prefix="sglang-metrics-")
            LOG.info("Created SGLang metrics temp directory: %s", cls._sglang_metrics_dir)
        return cls._sglang_metrics_dir

    @classmethod
    def get_generation_default_args(cls) -> str:
        """Pass the SGLang metrics temp-directory to the generation worker.

        Called in the **pipeline** process.  The returned hydra override is
        appended to the generation command, so the worker can read the same
        path from ``self.cfg.server.metrics_file_dir``.
        """
        metrics_dir = cls._ensure_sglang_metrics_dir()
        return f"++metrics_file_dir={metrics_dir}"

    @classmethod
    def get_server_command_fn(cls) -> callable:
        """Return a wrapper around the default server command builder.

        When the server type is ``sglang``, the wrapper automatically appends
        ``--enable-metrics --export-metrics-to-file --export-metrics-to-file-dir``
        so that SGLang writes per-request speculative decoding metrics that the
        evaluator can aggregate after generation.
        """
        from nemo_skills.pipeline.utils import get_server_command

        metrics_dir = cls._ensure_sglang_metrics_dir()
        sglang_metrics_args = f"--enable-metrics --export-metrics-to-file --export-metrics-to-file-dir {metrics_dir}"

        def specdec_server_command(
            server_type,
            num_gpus,
            num_nodes,
            model_path,
            cluster_config,
            server_port,
            server_args="",
            server_entrypoint=None,
        ):
            if server_type == "sglang":
                server_args = f"{server_args} {sglang_metrics_args}".strip()
            return get_server_command(
                server_type=server_type,
                num_gpus=num_gpus,
                num_nodes=num_nodes,
                model_path=model_path,
                cluster_config=cluster_config,
                server_port=server_port,
                server_args=server_args,
                server_entrypoint=server_entrypoint,
            )

        return specdec_server_command

    def inject_sglang_metrics(
        self,
        metrics_file_path: str,
    ) -> dict[str, Any] | None:
        """Inject SGLang metrics into the generation output.

        SGLang writes per-request metrics as JSON lines when launched with
        ``--export-metrics-to-file``.  Each line contains:

        - ``id``: Request ID (matches the ``rid`` in request_parameters)
        - ``spec_accept_length``: Acceptance length for this request
        - ``spec_accept_rate``: Acceptance rate for this request
        - ``spec_accept_token_num``: Number of accepted tokens
        - ``spec_draft_token_num``: Number of draft tokens
        - ``spec_verify_ct``: Number of verification steps (drafts)

        Args:
            metrics_file_path: Path to the SGLang metrics JSONL file.
            request_ids: Optional set of request IDs to filter by.  If ``None``,
                **all** entries in the file are aggregated (useful when the
                server is dedicated to the benchmark).

        Returns:
            Dictionary with aggregated spec decode statistics, or ``None`` if no
            matching requests were found or metrics are unavailable.
        """
        if not os.path.exists(metrics_file_path):
            LOG.warning("SGLang metrics file not found: %s, skipping metrics injection", metrics_file_path)
            return None

        metrics: dict[str, Any] = {}
        with open(metrics_file_path, "rt", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                try:
                    metric = json.loads(line)
                    metrics[metric["id"]] = {
                        "acceptance_length": metric["spec_accept_length"],
                        "acceptance_rate": metric["spec_accept_rate"] * 100,
                        "accepted_tokens": metric["spec_accept_token_num"],
                        "draft_tokens": metric["spec_draft_token_num"],
                        "num_drafts": metric["spec_verify_ct"],
                    }
                except json.JSONDecodeError:
                    LOG.warning(f"Failed to parse JSON line {i} for metrics injection, skipping")
                    continue

        data_points = []
        with open(self.cfg.output_file, "rt", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                try:
                    data_points.append(json.loads(line))
                except json.JSONDecodeError:
                    LOG.warning(f"Failed to parse JSON line {i} for metrics injection, skipping")
                    continue

        with open(self.cfg.output_file, "wt", encoding="utf-8") as fout:
            for data_point in data_points:
                if all(response_id in metrics for response_id in data_point["response_ids"]):
                    data_point.update(
                        {
                            "num_drafts": sum(
                                metrics[response_id]["num_drafts"] for response_id in data_point["response_ids"]
                            ),
                            "draft_tokens": sum(
                                metrics[response_id]["draft_tokens"] for response_id in data_point["response_ids"]
                            ),
                            "accepted_tokens": sum(
                                metrics[response_id]["accepted_tokens"] for response_id in data_point["response_ids"]
                            ),
                            "acceptance_rate": sum(
                                metrics[response_id]["acceptance_rate"] for response_id in data_point["response_ids"]
                            )
                            / len(data_point["response_ids"]),
                            "acceptance_length": sum(
                                metrics[response_id]["acceptance_length"] for response_id in data_point["response_ids"]
                            )
                            / len(data_point["response_ids"]),
                        }
                    )
                else:
                    LOG.warning(
                        "No metrics found for response_ids: %s, skipping data point", data_point["response_ids"]
                    )
                fout.write(json.dumps(data_point) + "\n")

        try:
            return_value = {
                "num_drafts": sum([data_point["num_drafts"] for data_point in data_points]),
                "draft_tokens": sum([data_point["draft_tokens"] for data_point in data_points]),
                "accepted_tokens": sum([data_point["accepted_tokens"] for data_point in data_points]),
                "acceptance_rate": sum([data_point["acceptance_rate"] for data_point in data_points])
                / len(data_points),
                "acceptance_length": sum([data_point["acceptance_length"] for data_point in data_points])
                / len(data_points),
                "per_position_acceptance_rates": [],
            }
        except KeyError:
            LOG.warning("Metrics injection failed for some data points, skipping")
            return None
        return return_value

    async def process_single_datapoint(self, data_point, all_data, prompt_format="openai"):
        """Handle single-turn and multi-turn generation.

        For single-turn data points (or when ``multiturn`` is ``False``), this
        falls through to the base-class implementation.

        For multi-turn data points, it loops through every turn, building up
        the conversation and accumulating token counts.

        Also tracks request IDs for SGLang metrics matching.
        """

        messages = []
        responses = []

        for message in data_point["messages"]:
            messages.append(message)
            new_data_point = {"messages": messages}
            current_response = await super().process_single_datapoint(
                new_data_point, all_data, prompt_format=prompt_format
            )
            if "response" in current_response:
                raw_response = current_response.pop("response")
                current_response["response_id"] = raw_response.id
            responses.append(current_response)
            messages.append({"role": "assistant", "content": current_response["generation"]})

        # Return aggregated results
        return {
            "generation": [response["generation"] for response in responses],
            "num_generated_tokens": sum([response["num_generated_tokens"] for response in responses]),
            "response_ids": [response["response_id"] for response in responses],
        }

    def _get_server_base_address(self) -> str:
        """Derive the server base address from the config.

        Returns:
            Server base address string, e.g. ``http://127.0.0.1:5000``.
        """
        return self.cfg.server.get("base_url") or f"http://{self.cfg.server['host']}:{self.cfg.server['port']}"

    def wait_for_server(self):
        """Wait for the server, then snapshot speculative decoding counters.

        For VLLM: Captures the "before" VLLM Prometheus counters
        (``vllm:spec_decode_*``) so that after generation we can compute a
        clean delta isolated to our generation run.

        For SGLang: Captures the "before" SGLang Prometheus gauges
        (``sglang:spec_accept_length``, ``sglang:spec_accept_rate``) and
        counters (``sglang:num_requests_total``, etc.) for the same purpose.
        If the metrics file (``--export-metrics-to-file``) is available after
        generation, it takes priority over the Prometheus delta.
        """
        super().wait_for_server()
        server_type = self.cfg.server["server_type"]
        base_url = self._get_server_base_address()

        if server_type == "sglang":
            # SGLang: Fetch "before" snapshot from Prometheus /metrics
            LOG.info("Fetching BEFORE SGLang spec-decode metrics from %s/metrics", base_url)
            self._before_metrics = fetch_sglang_spec_decode_metrics(base_url)
            LOG.info(
                "SGLang before snapshot: accept_length=%.4f, accept_rate=%.4f, num_requests=%d, gen_tokens=%d",
                self._before_metrics.spec_accept_length,
                self._before_metrics.spec_accept_rate,
                self._before_metrics.num_requests,
                self._before_metrics.generation_tokens,
            )
            return

        # VLLM: Fetch before snapshot
        LOG.info("Fetching BEFORE VLLM spec-decode metrics from %s/metrics", base_url)
        self._before_metrics = fetch_vllm_spec_decode_metrics(base_url)
        LOG.info(
            "Before snapshot: drafts=%d, draft_tokens=%d, accepted=%d, per_pos_keys=%s",
            self._before_metrics.num_drafts,
            self._before_metrics.num_draft_tokens,
            self._before_metrics.num_accepted_tokens,
            sorted(self._before_metrics.accepted_per_pos.keys()),
        )

    def run_batch_evaluation(self):
        """Fetch after-metrics, compute delta, then run the evaluator.

        For **VLLM**: Uses Prometheus ``/metrics`` endpoint with before/after
        delta on the ``vllm:spec_decode_*`` counters.

        For **SGLang**: Two strategies are attempted in order:

        1. **Metrics file (preferred)** — if the SGLang server was launched
           with ``--export-metrics-to-file``, read the per-request JSONL log
           and aggregate speculative decoding statistics.
        2. **Prometheus before/after fallback** — fetch
           ``sglang:spec_accept_length``, ``sglang:spec_accept_rate`` (gauges)
           and ``sglang:num_requests_total`` (counter) *before* and *after*
           generation, then back out the benchmark-specific averages from the
           weighted delta.

        The ``eval_specdec`` evaluator receives the pre-computed stats in
        ``eval_config["specdec_stats"]`` and stamps them onto each data point
        in the output JSONL.
        """
        server_address = self._get_server_base_address()
        server_type = self.cfg.server["server_type"]

        specdec_stats: dict[str, Any] | None = None

        if server_type == "sglang":
            # ----- Strategy 1: Metrics file (preferred) -----
            metrics_dir = getattr(self.cfg, "metrics_file_dir", None)
            if metrics_dir:
                metrics_file = find_sglang_metrics_file(metrics_dir)
                if metrics_file:
                    specdec_stats = self.inject_sglang_metrics(metrics_file)
                else:
                    LOG.warning("Could not find SGLang metrics file in %s", metrics_dir)

            # ----- Strategy 2: Prometheus before/after fallback -----
            if specdec_stats is None:
                LOG.info("Falling back to SGLang Prometheus before/after delta…")
                LOG.info("Fetching AFTER SGLang spec-decode metrics from %s/metrics", server_address)
                after_sglang = fetch_sglang_spec_decode_metrics(server_address)

                specdec_stats = compute_sglang_spec_decode_delta(self._before_metrics, after_sglang)
                LOG.info(
                    "SGLang Prometheus delta: acceptance_length=%.4f, "
                    "acceptance_rate=%.2f%%, requests=%d, gen_tokens=%d",
                    specdec_stats["acceptance_length"],
                    specdec_stats["acceptance_rate"],
                    specdec_stats["num_drafts"],
                    specdec_stats["draft_tokens"],
                )
        else:
            # VLLM: Fetch "after" snapshot and compute delta
            LOG.info("Fetching AFTER spec-decode metrics from %s/metrics", server_address)
            after_metrics = fetch_vllm_spec_decode_metrics(server_address)

            specdec_stats = compute_vllm_spec_decode_delta(self._before_metrics, after_metrics)
            LOG.info(
                "Spec-decode delta: drafts=%d, draft_tokens=%d, accepted=%d, "
                "acceptance_rate=%.2f%%, acceptance_length=%.4f",
                specdec_stats["num_drafts"],
                specdec_stats["draft_tokens"],
                specdec_stats["accepted_tokens"],
                specdec_stats["acceptance_rate"],
                specdec_stats["acceptance_length"],
            )
            if specdec_stats["per_position_acceptance_rates"]:
                LOG.info(
                    "Per-position acceptance rates: %s",
                    [f"{r:.4f}" for r in specdec_stats["per_position_acceptance_rates"]],
                )

        # Inject into eval_config for the evaluator
        self.cfg.eval_config["specdec_stats"] = specdec_stats

        super().run_batch_evaluation()


GENERATION_TASK_CLASS = SpecdecGenerationTask


@hydra.main(version_base=None, config_name="base_specdec_generation_config")
def specdec_generation(cfg: SpecdecGenerationConfig):
    cfg = SpecdecGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = SpecdecGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    SpecdecGenerationConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        specdec_generation()

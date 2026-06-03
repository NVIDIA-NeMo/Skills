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

"""Ray execution backend for NeMo Skills pipeline.

This module contains the RayBackend class responsible for submitting,
tracking, and stopping Ray Jobs via the Ray Jobs API
(dashboard_url / JobSubmissionClient).

Key design points
-----------------
* Jobs are submitted **concurrently** using a ThreadPoolExecutor so that
  paired workloads (e.g. GRPO training + vLLM judge server) can run at
  the same time.  The previous serial loop caused the judge to never
  start while training was blocking, making the judge host-file handshake
  time out.
* Each thread polls its own job until it reaches a terminal state, so
  the overall start_experiment() call returns only after every submitted
  job has finished (or one has failed).
* task_dependencies expressed as Ray Jobs are honoured by ordering
  submission: a job whose deps are not yet SUCCEEDED is not submitted
  until they are, using a ready-queue approach inside the concurrent
  executor.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
import shlex
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Dict

import nemo_run as run

from nemo_skills.pipeline.utils.backends import BackendRunOptions, ExecutionBackend
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))

# How long to sleep between Ray Jobs status polls (seconds).
_POLL_INTERVAL = 2
_STOP_VERIFY_TIMEOUT = 30

# Terminal states that mean the job is done (success or failure).
_TERMINAL_STATES = {"SUCCEEDED", "FAILED", "STOPPED"}
_SUCCESS_STATE = "SUCCEEDED"
_FAILURE_STATES = {"FAILED", "STOPPED"}


class RayBackend(ExecutionBackend):
    """Execution backend that submits tasks as Ray Jobs via the Jobs API.

    When ``dashboard_url`` is provided, jobs queued by ``add_task()``
    (stored on the experiment as ``_ns_ray_jobs_queue``) are submitted
    concurrently to the Ray cluster through ``JobSubmissionClient``.
    Pre-flight cluster checks are run before the first submission.

    Supports per-task ``entrypoint_label_selector`` so that tasks can be
    routed to specific node pools based on framework labels (e.g.
    ``framework=ray`` for CPU workers, ``framework=nemorl`` for GPU nodes).
    """

    name = "ray"

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        precreated_cluster: bool = False,
        control_plane: str | None = None,
        kubernetes_mode: str | None = None,
        dashboard_url: str | None = None,
        entrypoint_label_selector: Dict[str, str] | None = None,
        image_label_key: str | None = None,
        image_label_selectors: Dict[str, Dict[str, str]] | None = None,
    ):
        self.endpoint = endpoint.strip() if endpoint else None
        self.dashboard_url = self._normalize_dashboard_url(dashboard_url, self.endpoint)
        self.precreated_cluster = precreated_cluster
        self.control_plane = (control_plane or "").strip().lower() or None
        if entrypoint_label_selector is not None and not isinstance(entrypoint_label_selector, dict):
            raise ValueError("entrypoint_label_selector must be a dict[str, str] when provided")
        self.entrypoint_label_selector = {str(k): str(v) for k, v in (entrypoint_label_selector or {}).items()}
        self.image_label_key = (image_label_key or "").strip() or None
        self.image_label_selectors = self._normalize_image_label_selectors(image_label_selectors)
        self.kubernetes_mode = None
        if self.control_plane == "kubernetes":
            normalized_mode = (kubernetes_mode or "offline").strip().lower()
            if normalized_mode not in {"offline", "online"}:
                raise ValueError(
                    f"Unsupported kubernetes ray mode '{kubernetes_mode}'. "
                    "Supported values are 'offline' and 'online'."
                )
            self.kubernetes_mode = normalized_mode
        self._preflight_done = False
        self._manifest_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_dashboard_url(dashboard_url: str | None, endpoint: str | None) -> str | None:
        if dashboard_url:
            return str(dashboard_url).strip()
        if not endpoint:
            return None
        ep = str(endpoint).strip()
        if ep.startswith("http://") or ep.startswith("https://"):
            return ep
        if ep.startswith("ray://"):
            host_port = ep[len("ray://"):]
            host = host_port.split(":", 1)[0]
            return f"http://{host}:8265"
        return None

    @staticmethod
    def _sanitize_submission_id(value: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "-", value)
        return sanitized.strip("-._") or "job"

    @staticmethod
    def _to_status_str(status: Any) -> str:
        return str(getattr(status, "value", status)).upper()

    def _get_jobs_client(self):
        if not self.dashboard_url:
            raise RuntimeError(
                "Ray Jobs submission requires backend.dashboard_url "
                "(e.g. http://<head-host>:8265)."
            )
        try:
            from ray.job_submission import JobSubmissionClient
        except Exception as exc:
            raise RuntimeError(
                "Ray Jobs submission requires 'ray[jobs]' support in the submission environment."
            ) from exc
        return JobSubmissionClient(self.dashboard_url)

    def _stop_jobs_best_effort(
        self,
        client,
        job_ids: list[str],
        *,
        timeout_s: int = _STOP_VERIFY_TIMEOUT,
        reason: str = "cleanup",
    ) -> None:
        """Best-effort stop with bounded verification.

        Sends stop requests for all provided jobs, then polls status for up to
        ``timeout_s`` seconds to confirm jobs reached a terminal state. This is
        intentionally bounded to avoid perpetual waits during error handling.
        """
        pending = {str(jid) for jid in job_ids if jid}
        if not pending:
            return

        for job_id in list(pending):
            try:
                client.stop_job(job_id)
            except Exception as exc:
                LOG.warning("Failed to send stop for Ray job %s during %s: %s", job_id, reason, exc)

        deadline = time.monotonic() + max(timeout_s, 0)
        last_seen: Dict[str, str] = {}
        while pending and time.monotonic() < deadline:
            terminal_now: list[str] = []
            for job_id in list(pending):
                try:
                    status = self._to_status_str(client.get_job_status(job_id))
                    last_seen[job_id] = status
                except Exception:
                    # Status lookups can transiently fail; keep trying until timeout.
                    continue
                if status in _TERMINAL_STATES:
                    terminal_now.append(job_id)
            for job_id in terminal_now:
                pending.discard(job_id)
            if pending:
                time.sleep(_POLL_INTERVAL)

        if pending:
            summary = {jid: last_seen.get(jid, "UNKNOWN") for jid in sorted(pending)}
            LOG.warning(
                "Timed out waiting for Ray job cleanup during %s after %ss; "
                "jobs may still be running: %s",
                reason,
                timeout_s,
                summary,
            )

    def _write_final_job_log(
        self,
        client,
        *,
        job_id: str,
        task_name: str,
        status: str,
        job_log_file: str | None,
    ) -> None:
        """Persist final Ray job logs once the job reaches a terminal state.

        This is best-effort: failures to fetch/write logs are warned and ignored
        so they do not mask the primary job outcome.
        """
        if not job_log_file:
            return
        try:
            logs = client.get_job_logs(job_id) or ""
        except Exception as exc:
            LOG.warning(
                "Failed to fetch final logs for Ray job %s (%s): %s",
                job_id,
                task_name,
                exc,
            )
            return

        try:
            os.makedirs(os.path.dirname(job_log_file), exist_ok=True)
            with open(job_log_file, "w", encoding="utf-8") as f:
                if logs:
                    f.write(logs)
                    if not logs.endswith("\n"):
                        f.write("\n")
                f.write(
                    f"\n=== FINAL_STATUS task={task_name} job_id={job_id} "
                    f"status={status} ts={time.strftime('%Y-%m-%dT%H:%M:%S%z')} ===\n"
                )
        except Exception as exc:
            LOG.warning(
                "Failed to write final logs for Ray job %s (%s) to %s: %s",
                job_id,
                task_name,
                job_log_file,
                exc,
            )

    def _append_stage_job_record(
        self,
        *,
        stage_manifest_file: str | None,
        record: Dict[str, Any],
    ) -> None:
        """Append one stage-job mapping record as JSONL (best-effort)."""
        if not stage_manifest_file:
            return
        try:
            os.makedirs(os.path.dirname(stage_manifest_file), exist_ok=True)
            line = json.dumps(record, sort_keys=True)
            with self._manifest_lock:
                with open(stage_manifest_file, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.write("\n")
        except Exception as exc:
            LOG.warning(
                "Failed to append stage-job mapping record to %s: %s",
                stage_manifest_file,
                exc,
            )

    @staticmethod
    def _prepare_job_entrypoint_command(command: str) -> str:
        # When running *inside* a Ray Job, the driver is already on the cluster.
        # Replace any forwarded Ray Client endpoint with local cluster autodiscovery.
        stripped = re.sub(r"export\s+RAY_ADDRESS=[^&;]+&&\s*", "", command)
        return f"export RAY_ADDRESS=auto && {stripped.strip()}"

    @staticmethod
    def _normalize_image_label_selectors(
        selectors: Dict[str, Dict[str, str]] | None,
    ) -> Dict[str, Dict[str, str]]:
        if selectors is None:
            return {}
        if not isinstance(selectors, dict):
            raise ValueError(
                "image_label_selectors must be a dict[str, dict[str, str]] when provided"
            )
        normalized: Dict[str, Dict[str, str]] = {}
        for pattern, labels in selectors.items():
            if not isinstance(labels, dict):
                raise ValueError(
                    "image_label_selectors values must be dicts "
                    "(either label dict or {'key': ..., 'value': ...})"
                )
            # Convenience form: {"key": "...", "value": "..."}
            if set(labels.keys()) == {"key", "value"}:
                normalized[str(pattern)] = {str(labels["key"]): str(labels["value"])}
            else:
                normalized[str(pattern)] = {str(k): str(v) for k, v in labels.items()}
        return normalized

    def _labels_for_image(self, container_image: str) -> Dict[str, str]:
        labels: Dict[str, str] = {}
        for pattern, selector in self.image_label_selectors.items():
            if fnmatch.fnmatch(container_image, pattern):
                for key, value in selector.items():
                    labels.setdefault(key, value)
        return labels

    @staticmethod
    def _normalize_label_value(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9_.-]", "-", value.lower())
        normalized = re.sub(r"-+", "-", normalized).strip("-._")
        return normalized or "unknown"

    # ------------------------------------------------------------------
    # Stage metadata
    # ------------------------------------------------------------------

    def stage_metadata(
        self,
        *,
        use_with_ray_cluster: bool = False,
        container_image: str | None = None,
    ) -> Dict[str, Any] | None:
        if self.precreated_cluster and self.endpoint:
            should_use_embedded_ray_cluster = False
        else:
            should_use_embedded_ray_cluster = True

        metadata = super().stage_metadata(use_with_ray_cluster=should_use_embedded_ray_cluster)
        metadata = dict(metadata or {})
        metadata["execution_backend"] = "kubernetes-ray" if self.control_plane == "kubernetes" else "ray"
        if self.control_plane:
            metadata["ray_control_plane"] = self.control_plane
        if self.kubernetes_mode:
            metadata["kubernetes_mode"] = self.kubernetes_mode
        if self.endpoint:
            metadata["ray_address"] = self.endpoint
            metadata["ray_cluster_mode"] = "precreated" if self.precreated_cluster else "managed"
        if self.dashboard_url:
            metadata["ray_dashboard_url"] = self.dashboard_url
        selector = dict(self.entrypoint_label_selector)
        if container_image:
            image_specific_selector = self._labels_for_image(container_image)
            if image_specific_selector:
                # Image-derived selectors only fill gaps; explicit
                # entrypoint_label_selector keys win (like image_label_key below).
                for key, value in image_specific_selector.items():
                    selector.setdefault(key, value)
            elif self.image_label_key:
                selector.setdefault(
                    self.image_label_key, self._normalize_label_value(container_image)
                )
        if selector:
            metadata["ray_entrypoint_label_selector"] = selector
            metadata["entrypoint_label_selector"] = selector
        return metadata

    def get_env_overrides(self) -> Dict[str, str]:
        if not self.endpoint:
            return {}
        return {"RAY_ADDRESS": self.endpoint}

    # ------------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------------

    @staticmethod
    def _preflight_config(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        cfg = cluster_config.get("preflight") or {}
        if cfg is None:
            return {}
        if not isinstance(cfg, dict):
            raise ValueError("cluster_config.preflight must be a dict when provided")
        return cfg

    @staticmethod
    def _normalize_resource_key(key: str) -> str:
        k = str(key).strip().lower()
        if k in {"gpu", "gpus"}:
            return "GPU"
        if k in {"cpu", "cpus"}:
            return "CPU"
        if k in {"mem", "memory"}:
            return "memory"
        return str(key)

    @staticmethod
    def _extract_node_labels(nodes_detail: list[Any]) -> list[Dict[str, str]]:
        label_maps: list[Dict[str, str]] = []
        for node in nodes_detail:
            if hasattr(node, "model_dump"):
                node_data = node.model_dump()
            elif hasattr(node, "dict"):
                node_data = node.dict()
            elif isinstance(node, dict):
                node_data = node
            else:
                continue
            labels = node_data.get("labels") or node_data.get("node_labels") or {}
            if isinstance(labels, dict):
                label_maps.append({str(k): str(v) for k, v in labels.items()})
        return label_maps

    def _run_preflight(self, cluster_config: Dict[str, Any], options: BackendRunOptions) -> None:
        preflight_cfg = self._preflight_config(cluster_config)
        enabled = bool(preflight_cfg.get("enabled", True))
        if not enabled:
            return

        if self.precreated_cluster and not self.endpoint:
            raise RuntimeError(
                "Ray preflight failed: backend.precreated_cluster=true requires backend.endpoint."
            )

        if options.dry_run:
            return

        if self._preflight_done:
            return

        required_labels = preflight_cfg.get("required_node_labels") or []
        min_resources = preflight_cfg.get("min_cluster_resources") or {}
        require_reachable = bool(preflight_cfg.get("require_ray_endpoint", self.precreated_cluster))
        strict_label_check = bool(preflight_cfg.get("strict_label_check", True))

        if not require_reachable and not required_labels and not min_resources:
            return

        if require_reachable and not self.endpoint:
            raise RuntimeError(
                "Ray preflight failed: backend.endpoint is required for connectivity checks."
            )

        try:
            import ray
        except Exception as exc:
            raise RuntimeError(
                "Ray preflight failed: python package 'ray' is required on the submission host."
            ) from exc

        try:
            init_kwargs = {"ignore_reinit_error": True, "logging_level": logging.ERROR}
            if self.endpoint:
                init_kwargs["address"] = self.endpoint
            ray.init(**init_kwargs)

            live_nodes = [n for n in (ray.nodes() or []) if n.get("Alive", False)]
            if require_reachable and not live_nodes:
                raise RuntimeError(
                    "Ray preflight failed: endpoint reachable but no live nodes were found."
                )

            resources_total: Dict[str, float] = {}
            for node in live_nodes:
                for key, value in (node.get("Resources") or {}).items():
                    try:
                        resources_total[str(key)] = resources_total.get(str(key), 0.0) + float(value)
                    except Exception:
                        continue

            if min_resources:
                if not isinstance(min_resources, dict):
                    raise RuntimeError(
                        "Ray preflight failed: preflight.min_cluster_resources must be a dict."
                    )
                for key, required_value in min_resources.items():
                    normalized_key = self._normalize_resource_key(str(key))
                    try:
                        required_float = float(required_value)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Ray preflight failed: invalid min_cluster_resources value "
                            f"for '{key}': {required_value}"
                        ) from exc
                    available = float(resources_total.get(normalized_key, 0.0))
                    if available < required_float:
                        raise RuntimeError(
                            f"Ray preflight failed: resource '{normalized_key}' "
                            f"available={available} < required={required_float}."
                        )

            if required_labels:
                if not isinstance(required_labels, list):
                    raise RuntimeError(
                        "Ray preflight failed: preflight.required_node_labels must be a list."
                    )
                labels_by_node: list[Dict[str, str]] = []
                try:
                    from ray.util.state import list_nodes
                    labels_by_node = self._extract_node_labels(list_nodes(detail=True) or [])
                except Exception:
                    labels_by_node = []

                if not labels_by_node:
                    msg = (
                        "Ray preflight could not inspect node labels from ray state API. "
                        "Set preflight.strict_label_check=false to bypass label checks."
                    )
                    if strict_label_check:
                        raise RuntimeError(msg)
                    LOG.warning(msg)
                else:
                    for label_req in required_labels:
                        if not isinstance(label_req, dict):
                            raise RuntimeError(
                                "Ray preflight failed: each required_node_labels item "
                                "must be a dict with key/value."
                            )
                        key = str(label_req.get("key", "")).strip()
                        value = str(label_req.get("value", "")).strip()
                        if not key:
                            raise RuntimeError(
                                "Ray preflight failed: required_node_labels entries "
                                "must include non-empty key."
                            )
                        if not any(
                            node_labels.get(key) == value for node_labels in labels_by_node
                        ):
                            raise RuntimeError(
                                f"Ray preflight failed: no live node matched label '{key}={value}'."
                            )

            self._preflight_done = True
        finally:
            try:
                ray.shutdown()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Experiment lifecycle
    # ------------------------------------------------------------------

    def start_experiment(
        self, exp: run.Experiment, cluster_config: Dict[str, Any], options: BackendRunOptions
    ):
        self._run_preflight(cluster_config, options)
        pending_jobs = getattr(exp, "_ns_ray_jobs_queue", None)
        if not pending_jobs:
            return super().start_experiment(exp, cluster_config, options)

        if options.dry_run:
            LOG.info(
                "Dry run mode enabled; skipping Ray Jobs submission for %d task(s).",
                len(pending_jobs),
            )
            return

        client = self._get_jobs_client()
        self._submit_jobs_concurrently(client, exp, pending_jobs)

    def _submit_jobs_concurrently(self, client, exp: run.Experiment, pending_jobs: list) -> None:
        """Submit all pending jobs concurrently so paired workloads (e.g. training
        + judge) can run at the same time rather than serially.

        Dependency ordering
        -------------------
        Each queued job may carry a ``dep_task_names`` list.  A job is submitted
        only after all its named dependencies have reached SUCCEEDED state.  The
        executor uses a simple ready-queue loop so dependencies are resolved
        without blocking unrelated jobs.

        This fixes the finance GRPO training+judge pattern where:
        - Training polls a host-file written by the judge
        - The judge must therefore be running concurrently with training
        - The old serial loop submitted training, blocked until it finished,
          then submitted the judge — meaning the judge never came up while
          training needed it, causing the wait-for-host-file loop to time out.
        """
        # job_id -> status (populated as jobs complete)
        completed: Dict[str, str] = {}
        # job_id -> Future
        futures: Dict[str, Future] = {}
        # remaining jobs not yet submitted
        pending = list(pending_jobs)

        exp_title = getattr(exp, "_title", "exp")
        submitted_meta: list[Dict[str, Any]] = []

        def _poll_until_done(
            job_id: str,
            task_name: str,
            job_log_file: str | None,
            stage_run_id: str | None,
            stage_manifest_file: str | None,
            submission_id: str | None,
        ) -> Dict[str, Any]:
            """Poll a single Ray Job until it reaches a terminal state."""
            last_logs = ""
            while True:
                status_str = self._to_status_str(client.get_job_status(job_id))
                try:
                    logs = client.get_job_logs(job_id) or ""
                except Exception:
                    logs = ""
                if logs and logs != last_logs:
                    delta = logs[len(last_logs):] if logs.startswith(last_logs) else logs
                    if delta.strip():
                        for line in delta.rstrip().splitlines():
                            LOG.info("ray-job/%s %s", job_id, line)
                    last_logs = logs
                if status_str in _TERMINAL_STATES:
                    self._write_final_job_log(
                        client,
                        job_id=job_id,
                        task_name=task_name,
                        status=status_str,
                        job_log_file=job_log_file,
                    )
                    self._append_stage_job_record(
                        stage_manifest_file=stage_manifest_file,
                        record={
                            "event": "terminal",
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "stage_run_id": stage_run_id,
                            "task_name": task_name,
                            "job_id": job_id,
                            "submission_id": submission_id,
                            "status": status_str,
                            "dashboard_url": self.dashboard_url,
                            "job_log_file": job_log_file,
                        },
                    )
                    return {"job_id": job_id, "task_name": task_name, "status": status_str}
                time.sleep(_POLL_INTERVAL)

        def _deps_satisfied(job: Dict[str, Any]) -> bool:
            """Return True if all named dependencies have SUCCEEDED."""
            deps = job.get("dep_task_names") or []
            return all(completed.get(d) == _SUCCESS_STATE for d in deps)

        def _submit_one(job: Dict[str, Any], idx: int) -> Dict[str, Any]:
            """Prepare and submit a single job; return its metadata."""
            command = str(job.get("command", "")).strip()
            if not command:
                raise ValueError(f"Ray job at index {idx} has an empty command.")
            command = self._prepare_job_entrypoint_command(command)
            entrypoint = f"bash -lc {shlex.quote(command)}"
            selector = job.get("entrypoint_label_selector")
            task_name = str(job.get("task_name", "nemo-run"))
            metadata = {"nemo_task_name": task_name}
            base_id = self._sanitize_submission_id(
                str(job.get("submission_id") or f"{exp_title}-{idx}")
            )
            submission_id = f"{base_id}-{uuid.uuid4().hex[:8]}"
            LOG.info(
                "Submitting Ray Job %s to %s (selector=%s)",
                submission_id,
                self.dashboard_url,
                selector or {},
            )
            job_id = client.submit_job(
                entrypoint=entrypoint,
                submission_id=submission_id,
                metadata=metadata,
                entrypoint_label_selector=selector or None,
            )
            return {
                "task_name": task_name,
                "job_id": job_id,
                "submission_id": submission_id,
                "job_log_file": job.get("job_log_file"),
                "stage_run_id": job.get("stage_run_id"),
                "stage_manifest_file": job.get("stage_manifest_file"),
            }

        # Max workers = number of jobs so all can be in-flight simultaneously.
        max_workers = max(len(pending_jobs), 1)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ray-job") as pool:
            idx = 0
            while pending or futures:
                # Submit any jobs whose dependencies are now satisfied.
                still_pending = []
                for job in pending:
                    if _deps_satisfied(job):
                        try:
                            meta = _submit_one(job, idx)
                        except Exception as exc:
                            # Cancel all in-flight jobs on submission error.
                            self._stop_jobs_best_effort(
                                client,
                                list(futures.keys()),
                                reason="submission-failure",
                            )
                            raise RuntimeError(
                                f"Failed to submit Ray job '{job.get('task_name', idx)}': {exc}"
                            ) from exc
                        idx += 1
                        submitted_meta.append(meta)
                        self._append_stage_job_record(
                            stage_manifest_file=meta.get("stage_manifest_file"),
                            record={
                                "event": "submitted",
                                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                "stage_run_id": meta.get("stage_run_id"),
                                "task_name": meta.get("task_name"),
                                "job_id": meta.get("job_id"),
                                "submission_id": meta.get("submission_id"),
                                "status": "SUBMITTED",
                                "dashboard_url": self.dashboard_url,
                                "job_log_file": meta.get("job_log_file"),
                            },
                        )
                        future = pool.submit(
                            _poll_until_done,
                            meta["job_id"],
                            meta["task_name"],
                            meta.get("job_log_file"),
                            meta.get("stage_run_id"),
                            meta.get("stage_manifest_file"),
                            meta.get("submission_id"),
                        )
                        futures[meta["job_id"]] = future
                    else:
                        still_pending.append(job)
                pending = still_pending

                if not futures:
                    # No jobs in flight and no jobs ready — dependency deadlock.
                    if pending:
                        unresolved = [j.get("task_name", "?") for j in pending]
                        raise RuntimeError(
                            f"Ray Jobs dependency deadlock: tasks {unresolved} cannot be "
                            "satisfied (missing or failed dependency)."
                        )
                    break

                # Wait for at least one future to finish before checking pending again.
                done_iter = as_completed(futures.values(), timeout=None)
                try:
                    done_future = next(done_iter)
                except StopIteration:
                    break

                # Find the job_id for this future.
                done_job_id = next(
                    (jid for jid, f in futures.items() if f is done_future), None
                )
                if done_job_id is None:
                    continue

                result = done_future.result()  # raises if _poll_until_done raised
                status = result["status"]
                task_name = result["task_name"]
                completed[task_name] = status
                del futures[done_job_id]
                LOG.info("Ray job %s (%s) finished with status %s", done_job_id, task_name, status)

                if status in _FAILURE_STATES:
                    # Cancel all still-running jobs and fail fast.
                    self._stop_jobs_best_effort(
                        client,
                        list(futures.keys()),
                        reason=f"job-failure:{task_name}",
                    )
                    raise RuntimeError(
                        f"Ray job '{task_name}' (id={done_job_id}) ended with status {status}."
                    )

        setattr(exp, "_ns_ray_jobs_submitted", submitted_meta)

    def track_experiment(
        self, exp: run.Experiment, include_finished: bool = True
    ) -> Dict[str, Any]:
        submitted = getattr(exp, "_ns_ray_jobs_submitted", None)
        if not submitted or not self.dashboard_url:
            return super().track_experiment(exp, include_finished=include_finished)

        client = self._get_jobs_client()
        active_states = {"RUNNING", "PENDING", "SUBMITTED", "UNKNOWN"}
        tracked: Dict[str, Any] = {}
        for item in submitted:
            task_name = str(item.get("task_name", "nemo-run"))
            job_id = str(item.get("job_id", ""))
            if not job_id:
                continue
            status = self._to_status_str(client.get_job_status(job_id))
            if include_finished or status in active_states:
                tracked[task_name] = {
                    "status": status,
                    "handle": job_id,
                    "submission_id": item.get("submission_id"),
                    "stage_run_id": item.get("stage_run_id"),
                    "dashboard_url": self.dashboard_url,
                    "job_log_file": item.get("job_log_file"),
                    "stage_manifest_file": item.get("stage_manifest_file"),
                }
        return tracked

    def stop_experiment(
        self, exp: run.Experiment, only_active: bool = True
    ) -> list[str]:
        submitted = getattr(exp, "_ns_ray_jobs_submitted", None)
        if not submitted or not self.dashboard_url:
            return super().stop_experiment(exp, only_active=only_active)

        client = self._get_jobs_client()
        cancelled_jobs = []
        active_states = {"RUNNING", "PENDING", "SUBMITTED", "UNKNOWN"}
        for item in submitted:
            task_name = str(item.get("task_name", "nemo-run"))
            job_id = str(item.get("job_id", ""))
            if not job_id:
                continue
            status = self._to_status_str(client.get_job_status(job_id))
            if only_active and status not in active_states:
                continue
            self._stop_jobs_best_effort(
                client,
                [job_id],
                reason=f"stop-experiment:{task_name}",
            )
            cancelled_jobs.append(task_name)
        return cancelled_jobs

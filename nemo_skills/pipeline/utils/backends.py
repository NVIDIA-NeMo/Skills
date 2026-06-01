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

import logging
from dataclasses import dataclass
from typing import Any, Dict

import nemo_run as run

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def _normalize_backend_config(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize backend config from multiple compatible keys."""
    backend_config = cluster_config.get("backend") or cluster_config.get("execution_backend") or {}
    if isinstance(backend_config, str):
        return {"name": backend_config}
    if not isinstance(backend_config, dict):
        raise ValueError(
            f"cluster_config backend must be a dict or string, got {type(backend_config).__name__}"
        )
    return backend_config


def _resolve_selector_keys_with_container_map(
    selectors: Dict[str, Any] | None,
    containers: Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    """Resolve image selector keys from container names when possible.

    If a selector key matches a key in cluster_config["containers"], it is replaced
    by that container's configured image/path. Non-matching keys are preserved as-is
    to retain compatibility with explicit image/path or glob keys.
    """
    if selectors is None or not isinstance(selectors, dict):
        return selectors

    if not isinstance(containers, dict):
        containers = {}

    resolved: Dict[str, Any] = {}
    for key, value in selectors.items():
        key_str = str(key)
        resolved_key = str(containers.get(key_str, key_str))
        resolved[resolved_key] = value
    return resolved


@dataclass
class BackendRunOptions:
    sequential: bool = False
    dry_run: bool = False


class ExecutionBackend:
    """Execution backend hook interface.

    Backends may customize script metadata and lifecycle operations for experiments.
    """

    name = "default"

    def stage_metadata(
        self,
        *,
        use_with_ray_cluster: bool = False,
        container_image: str | None = None,
    ) -> Dict[str, Any] | None:
        if use_with_ray_cluster:
            return {"use_with_ray_cluster": True}
        return None

    def get_env_overrides(self) -> Dict[str, str]:
        return {}

    def start_experiment(self, exp: run.Experiment, cluster_config: Dict[str, Any], options: BackendRunOptions):
        if options.dry_run:
            LOG.info("Dry run mode is enabled, not running the experiment.")
            return

        if cluster_config["executor"] != "slurm":
            exp.run(detach=False, tail_logs=True, sequential=options.sequential)
        else:
            exp.run(detach=True, sequential=options.sequential)

    def track_experiment(self, exp: run.Experiment, include_finished: bool = True) -> Dict[str, Any]:
        status_dict = exp.status(return_dict=True)
        if include_finished:
            return status_dict

        active_states = {
            "RUNNING",
            "PENDING",
            "SUBMITTED",
            "UNKNOWN",
        }
        return {
            task_name: info
            for task_name, info in status_dict.items()
            if str(info.get("status", "")).split(".")[-1] in active_states
        }

    def stop_experiment(self, exp: run.Experiment, only_active: bool = True) -> list[str]:
        cancelled_jobs = []
        active_states = {
            "RUNNING",
            "PENDING",
            "SUBMITTED",
            "UNKNOWN",
        }
        status_map = exp.status(return_dict=True)
        for task_name, info in status_map.items():
            state = str(info.get("status", "")).split(".")[-1]
            handle = info.get("handle")
            if not handle:
                continue
            if only_active and state not in active_states:
                continue
            exp.cancel(handle)
            cancelled_jobs.append(task_name)

        return cancelled_jobs


# ---------------------------------------------------------------------------
# RayBackend lives in ray_backend.py – imported here for backwards compat.
# ---------------------------------------------------------------------------
from nemo_skills.pipeline.utils.ray_backend import RayBackend  # noqa: E402  # re-exported


class _RayBackendShim(RayBackend):
    """Shim so that isinstance checks against the old import path still work."""


# Keep the name RayBackend pointing at the canonical class.
del _RayBackendShim  # only the alias is needed


# Placeholder so linters don't complain about the import being "unused".
__all__ = [
    "BackendRunOptions",
    "ExecutionBackend",
    "RayBackend",
    "get_execution_backend",
    "track_stage_tasks",
    "stop_stage_tasks",
]

def get_execution_backend(cluster_config: Dict[str, Any], *, with_ray: bool = False) -> ExecutionBackend:
    """Resolve execution backend from cluster config and compatibility flags.

    Resolution priority:
    1. Explicit backend/ execution_backend in cluster config
    2. Legacy with_ray compatibility flag
    3. Default backend
    """
    backend_config = _normalize_backend_config(cluster_config)
    backend_name = str(backend_config.get("name") or "").strip().lower()
    legacy_ray_endpoint = cluster_config.get("ray_endpoint")

    if not backend_name:
        if with_ray:
            return RayBackend(endpoint=legacy_ray_endpoint, precreated_cluster=bool(legacy_ray_endpoint))
        return ExecutionBackend()

    if backend_name in {"default", "none"}:
        return ExecutionBackend()
    if backend_name == "ray":
        endpoint = (
            backend_config.get("endpoint")
            or backend_config.get("ray_endpoint")
            or (backend_config.get("kubernetes") or {}).get("endpoint")
            or legacy_ray_endpoint
        )
        dashboard_url = (
            backend_config.get("dashboard_url")
            or backend_config.get("jobs_api_url")
            or (backend_config.get("kubernetes") or {}).get("dashboard_url")
        )
        precreated_cluster = bool(backend_config.get("precreated_cluster", False)) or bool(endpoint)
        control_plane = str(backend_config.get("control_plane") or "").strip().lower()
        k8s_cfg = backend_config.get("kubernetes") or {}
        selector = backend_config.get("entrypoint_label_selector") or k8s_cfg.get("entrypoint_label_selector")
        image_label_key = backend_config.get("image_label_key") or k8s_cfg.get("image_label_key")
        image_label_selectors = backend_config.get("image_label_selectors") or k8s_cfg.get("image_label_selectors")
        image_label_selectors = _resolve_selector_keys_with_container_map(
            image_label_selectors, cluster_config.get("containers")
        )
        if control_plane == "kubernetes":
            mode = k8s_cfg.get("mode", "offline")
            endpoint = endpoint or k8s_cfg.get("endpoint")
            return RayBackend(
                endpoint=endpoint,
                precreated_cluster=bool(endpoint),
                control_plane="kubernetes",
                kubernetes_mode=mode,
                dashboard_url=dashboard_url,
                entrypoint_label_selector=selector,
                image_label_key=image_label_key,
                image_label_selectors=image_label_selectors,
            )
        return RayBackend(
            endpoint=endpoint,
            precreated_cluster=precreated_cluster,
            control_plane=control_plane or None,
            dashboard_url=dashboard_url,
            entrypoint_label_selector=selector,
            image_label_key=image_label_key,
            image_label_selectors=image_label_selectors,
        )
    if backend_name in {"kubernetes-ray", "ray-kubernetes", "ray_kubernetes"}:
        k8s_cfg = backend_config.get("kubernetes") or {}
        mode = k8s_cfg.get("mode", "offline")
        selector = backend_config.get("entrypoint_label_selector") or k8s_cfg.get("entrypoint_label_selector")
        image_label_key = backend_config.get("image_label_key") or k8s_cfg.get("image_label_key")
        image_label_selectors = backend_config.get("image_label_selectors") or k8s_cfg.get("image_label_selectors")
        image_label_selectors = _resolve_selector_keys_with_container_map(
            image_label_selectors, cluster_config.get("containers")
        )
        endpoint = (
            backend_config.get("endpoint")
            or k8s_cfg.get("endpoint")
            or backend_config.get("ray_endpoint")
            or legacy_ray_endpoint
        )
        dashboard_url = (
            backend_config.get("dashboard_url")
            or backend_config.get("jobs_api_url")
            or k8s_cfg.get("dashboard_url")
        )
        return RayBackend(
            endpoint=endpoint,
            precreated_cluster=bool(endpoint),
            control_plane="kubernetes",
            kubernetes_mode=mode,
            dashboard_url=dashboard_url,
            entrypoint_label_selector=selector,
            image_label_key=image_label_key,
            image_label_selectors=image_label_selectors,
        )

    raise ValueError(
        f"Unsupported execution backend '{backend_name}'. Supported backends: default, ray, kubernetes-ray"
    )


def _with_exp(exp_or_name: run.Experiment | str):
    if isinstance(exp_or_name, run.Experiment):
        class _ExperimentCtx:
            def __init__(self, exp):
                self.exp = exp

            def __enter__(self):
                return self.exp

            def __exit__(self, exc_type, exc, tb):
                return False

        return _ExperimentCtx(exp_or_name)

    try:
        return run.Experiment.from_title(exp_or_name)
    except Exception:
        return run.Experiment.from_id(exp_or_name)


def track_stage_tasks(
    exp_or_name: run.Experiment | str,
    cluster_config: Dict[str, Any],
    *,
    include_finished: bool = True,
) -> Dict[str, Any]:
    """Track stage/task states through the configured execution backend."""
    backend = get_execution_backend(cluster_config)
    with _with_exp(exp_or_name) as exp:
        return backend.track_experiment(exp, include_finished=include_finished)


def stop_stage_tasks(
    exp_or_name: run.Experiment | str,
    cluster_config: Dict[str, Any],
    *,
    only_active: bool = True,
) -> list[str]:
    """Stop stage tasks through the configured execution backend."""
    backend = get_execution_backend(cluster_config)
    with _with_exp(exp_or_name) as exp:
        return backend.stop_experiment(exp, only_active=only_active)

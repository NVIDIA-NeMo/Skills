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
from typing import TYPE_CHECKING, Any, Dict

import nemo_run as run

from nemo_skills.pipeline.utils.cluster import get_env_variables
from nemo_skills.utils import get_logger_name

if TYPE_CHECKING:
    # Re-exported via __getattr__ at runtime (kept lazy); declared here so type checkers
    # and ruff's __all__ resolution see RayBackend as a defined name without importing
    # the heavy ray_backend module on the default path.
    from nemo_skills.pipeline.utils.ray_backend import RayBackend

LOG = logging.getLogger(get_logger_name(__file__))


def _normalize_backend_config(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize backend config from multiple compatible keys."""
    backend_config = cluster_config.get("backend") or cluster_config.get("execution_backend") or {}
    if isinstance(backend_config, str):
        return {"name": backend_config}
    if not isinstance(backend_config, dict):
        raise ValueError(f"cluster_config backend must be a dict or string, got {type(backend_config).__name__}")
    return backend_config


# Backend names that select a Ray backend (canonical name plus accepted aliases).
_RAY_BACKEND_NAMES = frozenset({"ray", "kubernetes-ray", "ray-kubernetes", "ray_kubernetes"})


def get_backend_name(cluster_config: Dict[str, Any]) -> str:
    """Return the normalized (lowercased) backend name, or '' when none is set.

    Accepts the same string- or dict-form ``backend`` / ``execution_backend`` config as
    :func:`get_execution_backend`, so callers never re-implement backend-name parsing
    (and never crash on the supported bare-string form, e.g. ``backend: ray``).
    """
    return str(_normalize_backend_config(cluster_config).get("name") or "").strip().lower()


def is_ray_backend_name(cluster_config: Dict[str, Any]) -> bool:
    """True when the configured backend name selects a Ray backend (any alias)."""
    return get_backend_name(cluster_config) in _RAY_BACKEND_NAMES


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
        """Return per-stage script metadata, or None when no metadata applies."""
        if use_with_ray_cluster:
            return {"use_with_ray_cluster": True}
        return None

    def get_env_overrides(self) -> Dict[str, str]:
        """Return environment variable overrides to inject into stage commands."""
        return {}

    def start_experiment(self, exp: run.Experiment, cluster_config: Dict[str, Any], options: BackendRunOptions):
        """Run the experiment, detaching for Slurm and tailing logs otherwise."""
        if options.dry_run:
            LOG.info("Dry run mode is enabled, not running the experiment.")
            return

        if cluster_config["executor"] != "slurm":
            exp.run(detach=False, tail_logs=True, sequential=options.sequential)
        else:
            exp.run(detach=True, sequential=options.sequential)

    def track_experiment(self, exp: run.Experiment, include_finished: bool = True) -> Dict[str, Any]:
        """Return a task-name to status map, optionally filtering to active tasks."""
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
        """Cancel experiment tasks and return the names of the cancelled tasks."""
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
# RayBackend lives in ray_backend.py. It is imported lazily -- only when a Ray
# backend is actually selected (see get_execution_backend) -- so the default /
# Slurm path never pays to import the ~900-line ray_backend module. A module-level
# __getattr__ (PEP 562) keeps the historical `from ...backends import RayBackend`
# re-export working without eagerly importing the module at package load.
# ---------------------------------------------------------------------------


def __getattr__(name: str):
    if name == "RayBackend":
        from nemo_skills.pipeline.utils.ray_backend import RayBackend

        return RayBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# RayBackend is re-exported here (via __getattr__) for backwards-compatible imports.
__all__ = [
    "BackendRunOptions",
    "ExecutionBackend",
    "RayBackend",
    "get_backend_name",
    "is_ray_backend_name",
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

    # Import RayBackend lazily -- only when a Ray backend is actually selected -- so the
    # default / Slurm resolution path never imports the heavy ray_backend module.
    if with_ray or backend_name in _RAY_BACKEND_NAMES:
        from nemo_skills.pipeline.utils.ray_backend import RayBackend

    if not backend_name:
        if with_ray:
            return RayBackend(
                endpoint=legacy_ray_endpoint,
                precreated_cluster=bool(legacy_ray_endpoint),
                env_vars=get_env_variables(cluster_config),
            )
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
        # A dashboard_url (Jobs API) targets an existing cluster just like an
        # explicit precreated_cluster flag or a Ray Client endpoint. Treat any of
        # them as the precreated signal, and never downgrade an explicit
        # precreated_cluster: true.
        precreated_cluster = (
            bool(backend_config.get("precreated_cluster", False)) or bool(endpoint) or bool(dashboard_url)
        )
        control_plane = str(backend_config.get("control_plane") or "").strip().lower()
        # `backend.name: ray` with `executor: none` targets a pre-provisioned Ray Jobs
        # cluster, which needs a dashboard URL to submit to. If none resolved (and this
        # is not a Kubernetes control plane), fail fast with an actionable message rather
        # than fall through to the opaque nemo-run error
        # "use_with_ray_cluster is only supported for SlurmExecutor".
        if not precreated_cluster and control_plane != "kubernetes":
            executor_kind = str(cluster_config.get("executor") or "").strip().lower()
            if executor_kind == "none":
                raise ValueError(
                    "backend.name: ray with executor: none targets a pre-provisioned Ray Jobs "
                    "cluster and requires a dashboard URL — set "
                    "backend.dashboard_url: http://<head>:<port> (or backend.precreated_cluster: true). "
                    "To run Ray inside a Slurm allocation instead, set executor: slurm."
                )
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
                precreated_cluster=precreated_cluster,
                control_plane="kubernetes",
                kubernetes_mode=mode,
                dashboard_url=dashboard_url,
                entrypoint_label_selector=selector,
                image_label_key=image_label_key,
                image_label_selectors=image_label_selectors,
                env_vars=get_env_variables(cluster_config),
            )
        return RayBackend(
            endpoint=endpoint,
            precreated_cluster=precreated_cluster,
            control_plane=control_plane or None,
            dashboard_url=dashboard_url,
            entrypoint_label_selector=selector,
            image_label_key=image_label_key,
            image_label_selectors=image_label_selectors,
            env_vars=get_env_variables(cluster_config),
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
            backend_config.get("dashboard_url") or backend_config.get("jobs_api_url") or k8s_cfg.get("dashboard_url")
        )
        # endpoint, dashboard_url, or an explicit flag all signal a precreated cluster.
        precreated_cluster = (
            bool(backend_config.get("precreated_cluster", False)) or bool(endpoint) or bool(dashboard_url)
        )
        return RayBackend(
            endpoint=endpoint,
            precreated_cluster=precreated_cluster,
            control_plane="kubernetes",
            kubernetes_mode=mode,
            dashboard_url=dashboard_url,
            entrypoint_label_selector=selector,
            image_label_key=image_label_key,
            image_label_selectors=image_label_selectors,
            env_vars=get_env_variables(cluster_config),
        )

    raise ValueError(
        f"Unsupported execution backend '{backend_name}'. Supported backends: default, ray, kubernetes-ray"
    )


def _with_exp(exp_or_name: run.Experiment | str):
    """Return a context manager yielding an Experiment from an instance, title, or id."""
    if isinstance(exp_or_name, run.Experiment):

        class _ExperimentCtx:
            """Context manager that yields an already-instantiated experiment unchanged."""

            def __init__(self, exp):
                """Store the experiment to yield from the context."""
                self.exp = exp

            def __enter__(self):
                """Return the wrapped experiment."""
                return self.exp

            def __exit__(self, exc_type, exc, tb):
                """Leave the context without suppressing exceptions."""
                return False

        return _ExperimentCtx(exp_or_name)

    try:
        return run.Experiment.from_title(exp_or_name)
    except FileNotFoundError:
        # Title not found; fall back to treating the argument as an experiment id.
        # Any other error from from_title is a real failure and must propagate.
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

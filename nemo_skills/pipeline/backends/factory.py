# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Factory for creating compute backend instances.

This module provides a factory pattern for instantiating the appropriate
compute backend based on cluster configuration.

Usage
-----

    from nemo_skills.pipeline.backends import get_backend

    # Creates appropriate backend based on executor type in config
    backend = get_backend(cluster_config)

    # Backend selection:
    # - executor: "kubernetes" -> KubernetesBackend (native K8s Job API)
    # - executor: "slurm" -> SlurmBackend (wrapper around nemo-run, limited)
    # - executor: "local" -> LocalBackend (Docker containers)
    # - executor: "none" -> LocalBackend (direct execution, no container)

Note on nemo-run Integration
----------------------------

This factory is SEPARATE from the nemo-run based get_executor() in exp.py.

- get_backend() returns ComputeBackend instances (this module)
- get_executor() returns nemo-run executor instances (exp.py)

For Kubernetes: Use get_backend() - full native support
For Slurm: Use get_executor() for production - SlurmBackend here is limited
For Local: Either works, but get_executor() has better nemo-run integration

See nemo_skills.pipeline.backends.integration for detailed architecture notes.
"""

import logging
from typing import Dict, Type

from nemo_skills.pipeline.backends.base import ComputeBackend
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))

# Registry of available backends
_BACKEND_REGISTRY: Dict[str, Type[ComputeBackend]] = {}


def register_backend(name: str):
    """Decorator to register a backend class.

    Usage:
        @register_backend("kubernetes")
        class KubernetesBackend(ComputeBackend):
            ...
    """

    def decorator(cls: Type[ComputeBackend]):
        """Register the decorated backend class under ``name``."""
        _BACKEND_REGISTRY[name] = cls
        return cls

    return decorator


class BackendFactory:
    """Factory for creating compute backend instances.

    This factory supports automatic fallback to an alternative backend
    if the primary backend is unavailable.
    """

    @staticmethod
    def get_backend(
        cluster_config: Dict,
        fallback: bool = True,
    ) -> ComputeBackend:
        """Create a compute backend based on cluster configuration.

        Args:
            cluster_config: Cluster configuration dict. Must contain 'executor' key
                           with value 'slurm', 'kubernetes', 'local', or 'none'.
            fallback: If True and primary backend fails, try fallback_executor.

        Returns:
            Configured ComputeBackend instance.

        Raises:
            ValueError: If executor type is not supported.
            RuntimeError: If backend initialization fails and no fallback available.
        """
        if "executor" not in cluster_config:
            raise ValueError("cluster_config must include an 'executor' key")
        executor = cluster_config["executor"]
        if not isinstance(executor, str) or not executor.strip():
            raise ValueError("cluster_config['executor'] must be a non-empty string")

        # Normalize executor name
        executor = executor.strip().lower()
        primary_config = dict(cluster_config)
        primary_config["executor"] = executor

        primary_error = None

        # Try primary backend
        try:
            backend = BackendFactory._create_backend(executor, primary_config)
        except ValueError:
            # Preserve unknown/invalid executor errors as-is.
            raise
        except Exception as e:
            primary_error = e
        else:
            # Health check
            if backend.health_check():
                LOG.info(f"Successfully initialized {executor} backend")
                return backend
            primary_error = RuntimeError(f"{executor} backend health check failed")

        LOG.warning(f"Failed to initialize {executor} backend: {primary_error}")

        # Try fallback if configured
        fallback_executor = cluster_config.get("fallback_executor")
        if fallback and fallback_executor:
            if not isinstance(fallback_executor, str) or not fallback_executor.strip():
                LOG.error(f"Invalid fallback executor value: {fallback_executor!r}")
                raise RuntimeError(
                    f"Failed to initialize backend '{executor}': invalid fallback_executor value {fallback_executor!r}"
                ) from primary_error

            fallback_executor = fallback_executor.strip().lower()
            LOG.info(f"Attempting fallback to {fallback_executor} backend")
            try:
                fallback_config = dict(cluster_config)
                fallback_config["executor"] = fallback_executor
                fallback_backend = BackendFactory._create_backend(fallback_executor, fallback_config)
            except ValueError:
                # Preserve invalid fallback executor errors as-is.
                raise
            except Exception as fallback_error:
                LOG.error(f"Fallback backend {fallback_executor} also failed: {fallback_error}")
                raise RuntimeError(
                    f"Failed to initialize backend '{executor}' and fallback '{fallback_executor}': {fallback_error}"
                ) from primary_error

            if fallback_backend.health_check():
                LOG.info(f"Successfully initialized fallback {fallback_executor} backend")
                return fallback_backend

            LOG.error(f"Fallback backend {fallback_executor} health check failed")
            raise RuntimeError(
                f"Failed to initialize backend '{executor}': primary backend failed and "
                f"fallback backend '{fallback_executor}' failed health check"
            ) from primary_error

        raise RuntimeError(f"Failed to initialize backend '{executor}': {primary_error}") from primary_error

    @staticmethod
    def _create_backend(executor: str, cluster_config: Dict) -> ComputeBackend:
        """Create a backend instance by name.

        Args:
            executor: Backend name ('slurm', 'kubernetes', 'local', 'none').
            cluster_config: Cluster configuration dict.

        Returns:
            ComputeBackend instance.

        Raises:
            ValueError: If executor type is not supported.
        """
        # Check registry first (for dynamically registered backends)
        if executor in _BACKEND_REGISTRY:
            return _BACKEND_REGISTRY[executor](cluster_config)

        # Built-in backends with lazy imports
        if executor == "slurm":
            from nemo_skills.pipeline.backends.slurm import SlurmBackend

            return SlurmBackend(cluster_config)

        elif executor == "kubernetes":
            from nemo_skills.pipeline.backends.kubernetes import KubernetesBackend

            return KubernetesBackend(cluster_config)

        elif executor in ("local", "none"):
            from nemo_skills.pipeline.backends.local import LocalBackend

            return LocalBackend(cluster_config)

        else:
            available = list(_BACKEND_REGISTRY.keys()) + ["slurm", "kubernetes", "local", "none"]
            raise ValueError(f"Unknown executor '{executor}'. Available backends: {available}")

    @staticmethod
    def list_backends() -> list:
        """List all available backend names."""
        builtin = ["slurm", "kubernetes", "local", "none"]
        registered = list(_BACKEND_REGISTRY.keys())
        return builtin + [r for r in registered if r not in builtin]


def get_backend(cluster_config: Dict, fallback: bool = True) -> ComputeBackend:
    """Convenience function to get a compute backend.

    This is a shorthand for BackendFactory.get_backend().

    Args:
        cluster_config: Cluster configuration dict.
        fallback: If True, try fallback_executor if primary fails.

    Returns:
        Configured ComputeBackend instance.
    """
    return BackendFactory.get_backend(cluster_config, fallback=fallback)

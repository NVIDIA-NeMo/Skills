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

"""Compute backend abstractions for NeMo-Skills.

This module provides a unified interface for submitting and managing jobs
across different compute backends (Slurm, Kubernetes, Local/Docker).
"""

from nemo_skills.pipeline.backends.base import (
    ComputeBackend,
    ContainerSpec,
    JobHandle,
    JobSpec,
    JobStatus,
    ResourceSpec,
)
from nemo_skills.pipeline.backends.factory import BackendFactory, get_backend
from nemo_skills.pipeline.backends.integration import (
    create_data_processing_job_spec,
    create_inference_job_spec,
    create_training_job_spec,
    is_kubernetes_cluster,
    is_local_executor,
    is_slurm_cluster,
    run_job_and_wait,
)

__all__ = [
    # Base classes and types
    "ComputeBackend",
    "ContainerSpec",
    "JobHandle",
    "JobSpec",
    "JobStatus",
    "ResourceSpec",
    # Factory
    "BackendFactory",
    "get_backend",
    # Integration utilities
    "create_data_processing_job_spec",
    "create_inference_job_spec",
    "create_training_job_spec",
    "is_kubernetes_cluster",
    "is_local_executor",
    "is_slurm_cluster",
    "run_job_and_wait",
]

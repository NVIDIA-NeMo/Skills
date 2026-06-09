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

from nemo_skills.pipeline.utils.backends import get_execution_backend


def test_kubernetes_ray_selector_with_image_label_translation():
    cluster_config = {
        "executor": "slurm",
        "backend": {
            "name": "ray",
            "control_plane": "kubernetes",
            "precreated_cluster": True,
            "endpoint": "ray://ray-head.ray.svc.cluster.local:10001",
            "kubernetes": {
                "mode": "offline",
                "entrypoint_label_selector": {"type": "worker"},
                "image_label_key": "nemo/image",
            },
        },
    }

    backend = get_execution_backend(cluster_config)
    metadata = backend.stage_metadata(container_image="nvcr.io/nvidia/pytorch:25.02-py3")

    assert metadata["execution_backend"] == "kubernetes-ray"
    assert metadata["ray_control_plane"] == "kubernetes"
    assert metadata["ray_cluster_mode"] == "precreated"
    assert metadata["kubernetes_mode"] == "offline"
    assert metadata["entrypoint_label_selector"] == {
        "type": "worker",
        "nemo/image": "nvcr.io-nvidia-pytorch-25.02-py3",
    }
    assert metadata["ray_entrypoint_label_selector"] == metadata["entrypoint_label_selector"]


def test_kubernetes_ray_alias_uses_same_selector_logic():
    cluster_config = {
        "executor": "slurm",
        "backend": {
            "name": "kubernetes-ray",
            "endpoint": "ray://ray-head.ray.svc.cluster.local:10001",
            "entrypoint_label_selector": {"type": "worker"},
        },
    }

    backend = get_execution_backend(cluster_config)
    metadata = backend.stage_metadata(container_image="nvcr.io/nvidia/pytorch:25.02-py3")

    assert metadata["execution_backend"] == "kubernetes-ray"
    assert metadata["entrypoint_label_selector"] == {"type": "worker"}


def test_image_translation_does_not_override_explicit_selector_value():
    cluster_config = {
        "executor": "slurm",
        "backend": {
            "name": "ray",
            "control_plane": "kubernetes",
            "endpoint": "ray://ray-head.ray.svc.cluster.local:10001",
            "entrypoint_label_selector": {
                "type": "worker",
                "nemo/image": "preset-image-label",
            },
            "image_label_key": "nemo/image",
        },
    }

    backend = get_execution_backend(cluster_config)
    metadata = backend.stage_metadata(container_image="nvcr.io/nvidia/pytorch:25.02-py3")

    assert metadata["entrypoint_label_selector"]["nemo/image"] == "preset-image-label"
    assert metadata["entrypoint_label_selector"]["type"] == "worker"


def test_image_label_selectors_add_explicit_key_value_pairs():
    cluster_config = {
        "executor": "slurm",
        "containers": {
            "nemo-skills": "/containers/nemo-skills.sqsh",
            "nemo-rl": "/containers/nemo-rl.sqsh",
        },
        "backend": {
            "name": "ray",
            "control_plane": "kubernetes",
            "endpoint": "ray://ray-head.ray.svc.cluster.local:10001",
            "kubernetes": {
                "mode": "offline",
                "entrypoint_label_selector": {"type": "worker"},
                "image_label_selectors": {
                    "nemo-skills": {
                        "key": "nemo/workload",
                        "value": "skills",
                    },
                    "nemo-rl": {
                        "nemo/workload": "rl",
                    },
                },
            },
        },
    }

    backend = get_execution_backend(cluster_config)
    metadata = backend.stage_metadata(container_image="/containers/nemo-skills.sqsh")

    assert metadata["entrypoint_label_selector"]["type"] == "worker"
    assert metadata["entrypoint_label_selector"]["nemo/workload"] == "skills"


def test_image_label_selectors_prefer_static_selector_keys():
    cluster_config = {
        "executor": "slurm",
        "containers": {
            "nemo-skills": "/containers/nemo-skills.sqsh",
        },
        "backend": {
            "name": "ray",
            "control_plane": "kubernetes",
            "endpoint": "ray://ray-head.ray.svc.cluster.local:10001",
            "kubernetes": {
                "mode": "offline",
                "entrypoint_label_selector": {"nemo/workload": "static"},
                "image_label_selectors": {
                    "nemo-skills": {
                        "nemo/workload": "dynamic",
                    }
                },
            },
        },
    }

    backend = get_execution_backend(cluster_config)
    metadata = backend.stage_metadata(container_image="/containers/nemo-skills.sqsh")

    assert metadata["entrypoint_label_selector"]["nemo/workload"] == "static"


def test_ray_backend_forwards_required_env_vars_to_runtime_env():
    cluster_config = {
        "executor": "slurm",
        "required_env_vars": ["MY_JUDGE_KEY=secret-123"],
        "backend": {"name": "ray", "dashboard_url": "http://ray-head:8265"},
    }

    backend = get_execution_backend(cluster_config)
    runtime_env = backend._build_runtime_env()

    assert runtime_env is not None
    assert runtime_env["env_vars"]["MY_JUDGE_KEY"] == "secret-123"


def test_ray_backend_runtime_env_normalizes_and_filters_values():
    from nemo_skills.pipeline.utils.ray_backend import RayBackend

    backend = RayBackend(dashboard_url="http://ray-head:8265", env_vars={"A": "1", "B": None, "C": 2})

    assert backend._build_runtime_env() == {"env_vars": {"A": "1", "C": "2"}}


def test_ray_backend_runtime_env_none_when_no_env():
    from nemo_skills.pipeline.utils.ray_backend import RayBackend

    backend = RayBackend(dashboard_url="http://ray-head:8265", env_vars={})

    assert backend._build_runtime_env() is None

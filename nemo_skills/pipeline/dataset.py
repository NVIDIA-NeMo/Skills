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

"""Cluster-aware dataset loading.

Wraps core's get_dataset_module with cluster support (mount-path resolution,
SSH downloads). Pipeline code that needs cluster_config should import from here
instead of nemo_skills.dataset.utils.
"""

import importlib
import tempfile
from pathlib import Path

from nemo_skills.dataset.utils import (
    add_to_path,
    import_from_path,
)
from nemo_skills.dataset.utils import (
    get_dataset_module as _core_get_dataset_module,
)
from nemo_skills.pipeline.utils import cluster_download_file, get_unmounted_path


def _get_dataset_module_from_cluster(cluster_config, mounted_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = str(Path(tmpdir) / "init.py")
        cluster_dataset_path = get_unmounted_path(cluster_config, mounted_path)
        try:
            cluster_download_file(cluster_config, cluster_dataset_path, tmp_path)
        except FileNotFoundError:
            raise RuntimeError(
                f"Init file {mounted_path} not found on the cluster. "
                f"Please check the dataset name you're using. Did you forget to run prepare data commands?"
            )
        return import_from_path(tmp_path)


def get_dataset_module(dataset, data_dir=None, cluster_config=None, extra_benchmark_map=None):
    """Cluster-aware dataset module loading.

    Delegates to core's get_dataset_module for local resolution (path-based,
    built-in, extra benchmark map). When a cluster_config is provided and the
    dataset requires data_dir fallback, handles mount-path resolution and
    SSH downloads.
    """
    # No cluster config → delegate entirely to core
    if cluster_config is None or cluster_config.get("executor") == "none":
        return _core_get_dataset_module(dataset, data_dir=data_dir, extra_benchmark_map=extra_benchmark_map)

    # Try core resolution first (without data_dir — path, built-in, extra map)
    try:
        return _core_get_dataset_module(dataset, extra_benchmark_map=extra_benchmark_map)
    except RuntimeError:
        if not data_dir:
            raise

    # Core couldn't resolve it; use data_dir with cluster awareness
    dataset_as_path = dataset.replace(".", "/")
    if cluster_config["executor"] == "local":
        with add_to_path(get_unmounted_path(cluster_config, data_dir)):
            dataset_module = importlib.import_module(dataset)
    else:
        dataset_module = _get_dataset_module_from_cluster(cluster_config, f"{data_dir}/{dataset_as_path}/__init__.py")
    return dataset_module, data_dir

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

import hashlib
import logging
import re
import subprocess
from pathlib import Path

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))

_DOCKERFILE_PREFIX = "dockerfile:"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _sanitize_image_component(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return sanitized


def _resolve_dockerfile_path(dockerfile_path_str: str) -> Path:
    dockerfile_path = Path(dockerfile_path_str.strip())
    if dockerfile_path.is_absolute():
        raise ValueError("Dockerfile path must be specified relative to the repository root.")

    resolved = (_REPO_ROOT / dockerfile_path).resolve()
    try:
        resolved.relative_to(_REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"Dockerfile path '{dockerfile_path}' escapes the repository root.") from exc

    if not resolved.exists():
        raise FileNotFoundError(
            f"Dockerfile '{dockerfile_path}' not found relative to repository root '{_REPO_ROOT}'."
        )
    if not resolved.is_file():
        raise ValueError(f"Dockerfile path '{dockerfile_path}' does not resolve to a file.")

    return resolved


def _build_local_docker_image(dockerfile_spec: str) -> str:
    dockerfile_path = _resolve_dockerfile_path(dockerfile_spec)
    rel_identifier = dockerfile_path.relative_to(_REPO_ROOT).as_posix()
    image_name = f"locally-built-{_sanitize_image_component(rel_identifier)}"
    digest = hashlib.sha256(dockerfile_path.read_bytes()).hexdigest()[:12]
    image_ref = f"{image_name}:{digest}"
    context_dir = _REPO_ROOT

    LOG.info("Building Docker image %s from %s (context: %s)", image_ref, dockerfile_path, context_dir)
    try:
        subprocess.run(
            [
                "docker",
                "build",
                "-f",
                str(dockerfile_path),
                "-t",
                image_ref,
                str(context_dir),
            ],
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Docker is required to build images from dockerfile specifications, but it was not found in PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to build Docker image from {dockerfile_path}") from exc

    return image_ref


def resolve_container_image(container: str, cluster_config: dict) -> str:
    if not container.startswith(_DOCKERFILE_PREFIX):
        return container

    if cluster_config["executor"] != "local":
        raise ValueError("dockerfile container specifications are only supported for the local executor.")

    dockerfile_spec = container[len(_DOCKERFILE_PREFIX) :].strip()
    if not dockerfile_spec:
        raise ValueError("dockerfile container specification must include a path.")
    return _build_local_docker_image(dockerfile_spec)

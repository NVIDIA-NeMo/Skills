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

import os
from pathlib import Path

import docker
import yaml


def docker_run(command, image_name=None, volume_paths=None):
    test_config_path = Path(__file__).absolute().parent / "gpu-tests" / "test-local.yaml"
    with test_config_path.open("r") as f:
        config = yaml.safe_load(f.read())

    if image_name is None:
        image_name = "bash"
    if volume_paths is None:
        volume_paths = config["mounts"]

    client = docker.from_env()

    try:
        # Process volume paths
        volumes = {}
        for path in volume_paths:
            src, dst = path.split(":")
            volumes[os.path.abspath(src)] = {"bind": dst, "mode": "rw"}

        # Run the container
        full_command = f"bash -c '{command}'"
        result = client.containers.run(
            image_name,
            command=full_command,
            volumes=volumes,
            remove=True,
            detach=False,
        )
        logs = result.decode("utf-8")
        print("Operation completed.")
        print("Container logs:", logs)
    except docker.errors.ContainerError as e:
        print(f"Container exited with non-zero status code: {e.exit_status}")
        print(f"Container logs: {e.stderr.decode('utf-8')}")
        raise
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise  # Re-raise the exception after printing
    finally:
        client.close()


def docker_rm(paths):
    assert isinstance(paths, list), "paths should be a list of paths to remove"
    docker_run(command=f"rm -rf {' '.join([os.path.abspath(p) for p in paths])}")


def docker_rm_and_mkdir(file_):
    directory = Path(file_).absolute().parent
    rm_mkdir_cmd = f"rm -f {str(file_)} && mkdir -p {str(directory)}"
    docker_run(rm_mkdir_cmd)

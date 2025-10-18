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

import logging
import os
from typing import Dict, Optional

import nemo_run as run

from nemo_skills.pipeline.utils.cluster import get_tunnel
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def is_mounted_filepath(cluster_config: dict | None, path: str, mounts: Optional[list] = None) -> bool:
    """
    Check if the filepath is mounted using the cluster config. Does not raise an error if the filepath is not mounted.

    Args:
        cluster_config: cluster config dictionary
        path: path to the file to be mounted
        mounts: if specified, will check against that list and not use cluster_config

    Returns:
        bool: Whether the filepath is mounted.
    """
    mounts = mounts or (get_mounts_from_config(cluster_config) + ["/nemo_run/code:/nemo_run/code"])
    # Find which mount path matches the filepaths prefix
    for mount in mounts:
        mount_source, mount_dest = mount.split(":")
        if path.startswith(mount_dest):
            return True

    return False


def check_if_mounted(cluster_config, path_to_check):
    """Will check that path_to_check is referenced inside one of the mounts."""
    if cluster_config["executor"] == "none":
        # no mounts in local executor
        return

    if not is_mounted_filepath(cluster_config, path_to_check):
        raise ValueError(f"The path '{path_to_check}' is not mounted. Check cluster config.")


def _resolve_path_placeholders(path: str) -> str:
    """Resolve environment variable placeholders in a path."""
    if path is not None and "${" in path and "}" in path:
        path = os.path.expandvars(path)
        if "$" in path or "{" in path or "}" in path:
            raise ValueError(
                f"Path `{path}` contains env variable placeholders, but the required env var is "
                f"not provided in the environment to resolve."
            )
    return path


def check_mounts(
    cluster_config,
    log_dir: str,
    mount_map: Dict[str, Optional[str]] = None,
    create_remote_dir: bool = False,
    check_mounted_paths: bool = False,
):
    """
    Utility method to check if the paths are mounted, whether the remote directories exist, create them if they dont exist
    and finally resolve their mounted paths.

    Args:
        cluster_config: cluster config dictionary
        log_dir: optional str representing the log directory
        mount_map: a dictionary mapping the paths to their default mount locations.
            Keys must be the mount source, and values must be the mount destination.
            If the mount destination is None, the path will not be mounted but still checked and remote directory will still be created.
            Mount destinations must be absolute paths and begin with '/'.
        create_remote_dir: if True, will create the remote directories for the provided paths. Assumes the paths
            provided are directories.
        check_mounted_paths: if True, will perform remote calls to dynamically mount the provided paths, and create
            remote directories if required. Finally, will assert that all remote paths are valid files or directories.

    Returns:
        tuple: a tuple of the mounted paths for the provided paths in mount_map, and the log_dir
    """
    # Check if mount map is provided
    mount_map = mount_map or {}

    # Compute directory of all files
    remote_dir_list = []

    # Check paths and add to mount list if not mounted
    if check_mounted_paths:
        for path, default_mount in mount_map.items():
            # Check if path contains ${} env variables placeholders
            path = _resolve_path_placeholders(path)

            if not is_mounted_filepath(cluster_config, path):
                # check if the path is a file or a directory
                # so that the directory can be created
                if create_remote_dir:
                    # check if the path is a file or a directory
                    # so that the directory can be created
                    is_file = os.path.splitext(path) != ""
                    if is_file:
                        raise ValueError(
                            f"Path {path} provided is a file, but create_remote_dir is True. "
                            f"Please provide a directory instead."
                        )
                    remote_dir_list.append(path)

                if default_mount is not None:
                    # Check that path is not empty and is an absolute path
                    assert default_mount[0] == "/", (
                        f"Default mount path should be absolute path, given {default_mount}"
                    )

                    # Add mount path to the cluster config
                    add_mount_path(path, default_mount, cluster_config)

    else:
        # Just check if the paths are mounted
        for path in mount_map.keys():
            if path is not None:
                path = _resolve_path_placeholders(path)

                check_if_mounted(cluster_config, path)

    # check if the paths are mounted, get them if they arent but have mount sources
    # will error out if there are no corresponding mount sources
    new_paths = [get_mounted_path(cluster_config, path) for path in mount_map.keys()]

    if log_dir:
        if check_mounted_paths:
            # Create log dir in some location that will be mounted
            remote_dir_list.append(log_dir)
        else:
            check_if_mounted(cluster_config, log_dir)
        log_dir = get_mounted_path(cluster_config, log_dir)

    if check_mounted_paths:
        # Create missing remote directories
        if remote_dir_list:
            create_remote_directory(remote_dir_list, cluster_config)

        # Check that the file or dir exists at the remote location
        check_remote_mount_directories(list(mount_map.keys()) + [log_dir], cluster_config)

    if new_paths:
        return *new_paths, log_dir
    return log_dir


def get_mounted_path(cluster_config: dict, path: str):
    """
    Resolve the mounted filepath using the cluster config to merge the mount destination path to the filepath.

    Args:
        cluster_config: cluster config dictionary
        path: path to the file to be mounted

    Returns:
        str: mounted filepath

    Raises:
        ValueError: if the filepath is not mounted
    """
    if cluster_config["executor"] == "none":
        # no mounts in local executor
        return path
    if path is None:
        return None

    # Find which mount path matches the filepaths prefix
    mount_path = None
    for mount in get_mounts_from_config(cluster_config) + ["/nemo_run/code:/nemo_run/code"]:
        mount_source, mount_dest = mount.split(":")
        mount_source = mount_source.rstrip("/")
        if path.startswith(mount_source):
            mount_path = mount
            break

        elif path.startswith(mount_dest):
            # already mounted, return immediately
            return path

    if mount_path is None:
        raise ValueError(
            f"Could not find a mount path for the file path `{path}`. Check cluster config. Below paths are mounted: \n"
            f"{cluster_config['mounts']}"
        )

    # replace the mount destination inside the filepath with the mount source
    mount_source, mount_dest = mount_path.split(":")
    # append the rest of the path to the mount destination
    relative = path[len(mount_source) :].lstrip("/")
    filepath = os.path.join(mount_dest, relative)

    return filepath


def get_unmounted_path(cluster_config: dict, path: str):
    """
    Resolve the mounted filepath using the cluster config to merge the mount destination path to the filepath.
    If the filepath is already mounted, it will return the filepath as is.

    Args:
        cluster_config: cluster config dictionary
        path: path to the file to be mounted

    Returns:
        str: mounted filepath

    Raises:
        ValueError: if the filepath is not mounted
    """
    if cluster_config["executor"] == "none":
        # no mounts in local executor
        return path
    if path is None:
        return None

    # Find which mount path matches the filepaths prefix
    mount_path = None
    for mount in get_mounts_from_config(cluster_config) + ["/nemo_run/code:/nemo_run/code"]:
        mount_source, mount_dest = mount.split(":")
        if path.startswith(mount_dest):
            mount_path = mount
            break

        elif path.startswith(mount_source):
            # already mounted, return immediately
            return path

    if mount_path is None:
        raise ValueError(
            f"Could not find a mount path for the file path `{path}`. Check cluster config. Below paths are mounted: \n"
            f"{cluster_config['mounts'] + ['/nemo_run/code:/nemo_run/code']}"
        )

    # replace the mount destination inside the filepath with the mount source
    mount_source, mount_dest = mount_path.split(":")

    # append the rest of the path to the mount source
    filepath = mount_source + path[len(mount_dest) :]  # replace the mount destination with the mount source

    return filepath


def add_mount_path(mount_source: str, mount_dest: str, cluster_config):
    """Add a mount path to the cluster configuration."""

    if cluster_config is None:
        raise ValueError("Cluster config is not provided.")

    if "mounts" in cluster_config:
        original_mounts = get_mounts_from_config(cluster_config) + ["/nemo_run/code:/nemo_run/code"]
        added_mount = False
        for mount_path in original_mounts:
            source, destination = mount_path.split(":")

            if source == mount_source and destination == mount_dest:
                return

        if not added_mount:
            cluster_config["mounts"].append(f"{mount_source}:{mount_dest}")
            logging.info(f"Added mount path: `{mount_source}:{mount_dest}`")

    else:
        raise ValueError("No mounts found in cluster config, can only add to existing mount list.")


def create_remote_directory(directory: str | list, cluster_config: dict):
    """Create a remote directory on the cluster."""

    if cluster_config is None:
        raise ValueError("Cluster config is not provided.")

    if isinstance(directory, str):
        directory = [directory]

    # Get unmounted path of all directories
    directory = [
        get_unmounted_path(cluster_config, dir_path) if is_mounted_filepath(cluster_config, dir_path) else dir_path
        for dir_path in directory
    ]

    if cluster_config.get("executor") != "slurm":
        tunnel = run.LocalTunnel(job_dir=directory[0])
        for dir_path in directory:
            tunnel.run(f"mkdir -p {dir_path}", hide=False, warn=True)
            logging.info(f"Created directory: {dir_path} in local filesystem.")
        tunnel.cleanup()

    elif cluster_config.get("executor") == "slurm":
        tunnel = get_tunnel(cluster_config)
        for dir_path in directory:
            tunnel.run(f"mkdir -p {dir_path}", hide=False, warn=True)
            logging.info(f"Created directory: {dir_path} on remote cluster.")
        tunnel.cleanup()

    else:
        raise ValueError(f"Unsupported executor: {cluster_config.get('executor')}")


def resolve_mount_paths(cluster_config: dict, mount_paths: str | list | dict, create_remote_dir: bool = True):
    """
    Resolve the mount paths using the cluster config to merge the mount destination path to the filepath.
    Args:
        cluster_config: The cluster configuration dictionary to update.
        mount_paths: The mount paths to resolve - can be a string (comma separated), dict or a list of strings.
            Each mount path should be in the format `src:dest`.
        create_remote_dir: Whether to create the remote directories for the mount paths.
    """
    if mount_paths is not None:
        if isinstance(mount_paths, str):
            mount_paths_list = mount_paths.split(",")
        elif isinstance(mount_paths, dict):
            mount_paths_list = [f"{src}:{dest}" for src, dest in mount_paths.items()]
        else:
            mount_paths_list = mount_paths

        # remove empty strings from the list and strip whitespace
        mount_paths_list = [path.strip() for path in mount_paths_list]
        mount_paths_list = [path for path in mount_paths_list if path != ""]

        for idx, path in enumerate(mount_paths_list):
            assert ":" in path, f"Invalid mount path: {path}. Each path must be in the format `src:dest`"
            src, dest = path.split(":")
            src = src.strip().strip("\n")
            dest = dest.strip().strip("\n")

            LOG.info(f"Adding mount path:- {src}:{dest}")
            mount_paths_list[idx] = (src, dest)
            add_mount_path(src, dest, cluster_config)

        if create_remote_dir:
            LOG.info("Creating remote directories for mount paths:")
            all_src_dir = [src for src, _ in mount_paths_list]
            # Check if it is a file or a directory and only create the directory
            for idx in range(len(all_src_dir)):
                if os.path.splitext(all_src_dir[idx])[1] != "":
                    all_src_dir[idx] = os.path.dirname(all_src_dir[idx])
                LOG.info(f"Attempting to create remote directory: {all_src_dir[idx]}")

            create_remote_directory(all_src_dir, cluster_config)

    return cluster_config


def check_remote_mount_directories(directories: list, cluster_config: dict, exit_on_failure: bool = True):
    """Check if a directory exists on the cluster."""
    if cluster_config is None:
        raise ValueError("Cluster config is not provided.")
    if isinstance(directories, str):
        directories = [directories]

    # Get unmounted path of all directories
    directories = [
        get_unmounted_path(cluster_config, dir_path) if is_mounted_filepath(cluster_config, dir_path) else dir_path
        for dir_path in directories
    ]

    if cluster_config.get("executor") != "slurm":
        # there is no error locally if mounts aren't present, so we are skipping the check
        return
    elif cluster_config.get("executor") == "slurm":
        tunnel = get_tunnel(cluster_config)
        missing_source_locations = []
        for directory in directories:
            result = tunnel.run(f'test -e {directory} && echo "Directory Exists"', hide=True, warn=True)
            if "Directory Exists" not in result.stdout:
                missing_source_locations.append(directory)
        tunnel.cleanup()
        if len(missing_source_locations) > 0 and exit_on_failure:
            missing_source_locations = [
                f"{loc} DOES NOT exist at source destination" for loc in missing_source_locations
            ]
            missing_source_locations = "\n".join(missing_source_locations)
            raise FileNotFoundError(
                f"Some files or directories do not exist at the source location for mounting !!\n\n"
                f"{missing_source_locations}"
            )
    else:
        raise ValueError(f"Unsupported executor: {cluster_config.get('executor')}")


def get_mounts_from_config(cluster_config: dict):
    """
    Determines if there are mount paths that are being passed via environment variables.
    Selects the key in the cluster config called `mounts` which is a list of strings.
    Each string is in the format of `<str | {env_var}>:<str | {env_var}>` where `env_var`
    is the name of the environment variable.

    Args:
        cluster_config (dict): cluster config dictionary

    Returns:
        list: updated list of mounts
    """
    mounts = cluster_config.get("mounts", [])

    # if there are env_mounts, we will add the mounts from the env_mounts
    for mount_id in range(len(mounts)):
        mount = mounts[mount_id]

        if ":" not in mount:
            raise ValueError(f"Invalid mount format: {mount}. The mount path must be separated by a colon.")

        mount_source, mount_target = mount.split(":")

        if mount_source[0] == "{" and mount_source[-1] == "}":
            # Resolve the environment variable for the mount source
            mount_source = mount_source[1:-1]

            if mount_source not in os.environ:
                raise ValueError(
                    f"Required environment variable {mount_source} not found in env variables passed in cluster configs."
                )

            mount_source = os.environ[mount_source]

        if mount_target[0] == "{" and mount_target[-1] == "}":
            # Resolve the environment variable for the mount target
            mount_target = mount_target[1:-1]

            if mount_target not in os.environ:
                raise ValueError(
                    f"Required environment variable {mount_target} not found in env variables passed in cluster configs."
                )

            mount_target = os.environ[mount_target]

        # add the mount to the list of mounts
        resolved_mount = f"{mount_source}:{mount_target}"
        mounts[mount_id] = resolved_mount

    return mounts

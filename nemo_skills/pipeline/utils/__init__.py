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

# importing every utility function here to make them available in the pipeline.utils namespace

from nemo_skills.pipeline.utils.cluster import (
    _get_tunnel_cached,
    cluster_download_dir,
    cluster_download_file,
    cluster_path_exists,
    cluster_upload,
    get_cluster_config,
    get_env_variables,
    get_slurm_timeout_str,
    get_timeout_str,
    get_tunnel,
    kwargs_to_string,
    parse_kwargs,
    progress_callback,
    read_config,
    temporary_env_update,
    tunnel_hash,
    update_ssh_tunnel_config,
)
from nemo_skills.pipeline.utils.docker_images import resolve_container_image
from nemo_skills.pipeline.utils.exp import (
    CustomJobDetails,
    add_task,
    get_executor,
    get_exp,
    get_exp_handles,
    get_nsight_cmd,
    get_sandbox_command,
    run_exp,
)
from nemo_skills.pipeline.utils.generation import (
    configure_client,
    get_chunked_rs_filename,
    get_generation_cmd,
    get_remaining_jobs,
    normalize_models_config,
    normalize_parameter,
    wrap_cmd,
)
from nemo_skills.pipeline.utils.mounts import (
    add_mount_path,
    check_if_mounted,
    check_mounts,
    check_remote_mount_directories,
    create_remote_directory,
    get_mounted_path,
    get_mounts_from_config,
    get_unmounted_path,
    is_mounted_filepath,
    resolve_mount_paths,
)
from nemo_skills.pipeline.utils.packager import (
    get_git_repo_path,
    get_packager,
    get_registered_external_repo,
    register_external_repo,
)
from nemo_skills.pipeline.utils.server import (
    SupportedServers,
    SupportedServersSelfHosted,
    get_free_port,
    get_ray_server_cmd,
    get_server_command,
    get_server_wait_cmd,
    set_python_path_and_wait_for_server,
    should_get_random_port,
    wrap_python_path,
)

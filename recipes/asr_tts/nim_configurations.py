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

# This file contains image uris and corresponding server args for different NIM containers.

nim_configurations = {
    "magpie-tts-multilingual:1.3.0-34013444": {
        "image_uri": "nvcr.io/nvstaging/nim/magpie-tts-multilingual:1.3.0-34013444",
        "server_args": "--nim-tags-selector batch_size=32 --nim-disable-model-download false",
        "server_entrypoint": "DISABLE_RIVA_REALTIME_SERVER=True python3 -m nemo_skills.inference.server.serve_riva_nim",
        "installation_command": "pip install nvidia-riva-client==2.21.1",
        "local_images": {},  # items should be "cluster_name": "path_to_image.sqsh"
    },
    "parakeet-tdt-0.6b-v2:1.0.0": {
        "image_uri": "nvcr.io/nim/nvidia/parakeet-tdt-0.6b-v2:1.0.0",
        "server_args": "--CONTAINER_ID parakeet-tdt-0.6b-v2 --nim-tags-selector name=parakeet-tdt-0.6b-v2,mode=ofl  --nim-disable-model-download false",
        "server_entrypoint": "python3 -m nemo_skills.inference.server.serve_riva_nim",
        "installation_command": "pip install nvidia-riva-client==2.21.1",
        "local_images": {},  # items should be "cluster_name": "path_to_image.sqsh"
    },
}

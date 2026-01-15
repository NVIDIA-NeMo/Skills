#!/bin/bash

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

# NOTE: needs to run from the root of the repo!

SANDBOX_NAME=${1:-'local-sandbox'}
docker build --tag=${SANDBOX_NAME} --build-arg="NUM_WORKERS=$((`nproc --all`))" -f dockerfiles/Dockerfile.sandbox .

echo "Multi-worker mode: Starting $((`nproc --all`)) workers with session affinity"
docker run --network=host --rm \
    --memory=${NEMO_SKILLS_SANDBOX_MEM_LIMIT:-"16g"} \
    ${UWSGI_CPU_AFFINITY:+-e UWSGI_CPU_AFFINITY=${UWSGI_CPU_AFFINITY}} \
    ${UWSGI_PROCESSES:+-e UWSGI_PROCESSES=${UWSGI_PROCESSES}} \
    -v /nemo_run:/nemo_run \
    --name=local-sandbox ${SANDBOX_NAME}

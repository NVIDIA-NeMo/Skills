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

"""
Patch for litellm.litellm_core_utils.logging_worker.LoggingWorker
to disable its functionality and make all methods no-op.

Currently, the async_loop function in generate.py sometimes gets stuck forever because some of the coroutines in the litellm logging worker are not finished.
Debugging why the logger is not finished is non-trivial, so we just patch it to disable its functionality.
The behavior is that it keeps slurm jobs from existing and we waste gpus.
It always happens in docker containers, but does not happen locally.
"""

from typing import Coroutine


class NoOpLoggingWorker:
    """No-op implementation of LoggingWorker that disables all functionality."""

    def __init__(self, *args, **kwargs):
        pass

    def _ensure_queue(self) -> None:
        pass

    def start(self) -> None:
        pass

    async def _worker_loop(self) -> None:
        pass

    def enqueue(self, coroutine: Coroutine) -> None:
        if coroutine is not None:
            coroutine.close()

    def ensure_initialized_and_enqueue(self, async_coroutine: Coroutine):
        if async_coroutine is not None:
            async_coroutine.close()

    async def stop(self) -> None:
        pass

    async def flush(self) -> None:
        pass

    async def clear_queue(self):
        pass


def patch_litellm_logging_worker():
    """
    Patches the litellm LoggingWorker to disable its functionality.
    This prevents any logging worker from keeping the client alive.
    """
    try:
        import litellm.litellm_core_utils.logging_worker as logging_worker_module

        logging_worker_module.LoggingWorker = NoOpLoggingWorker
        logging_worker_module.GLOBAL_LOGGING_WORKER = NoOpLoggingWorker()
    except ModuleNotFoundError:
        # Ensure compatibility with different litellm versions
        pass

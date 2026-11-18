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

"""Retry transient Apptainer pulls without rerunning failed agent rollouts."""

import asyncio
import logging
import os
import random
import signal
from pathlib import Path

LOG = logging.getLogger(__name__)


def is_transient_image_pull_failure(output: str) -> bool:
    for line in output.splitlines():
        lower = line.lower()
        if (
            "fatal:" in lower
            and "unable to handle" in lower
            and "while building sif from layers: conveyor failed to get:" in lower
            and any(
                marker in lower
                for marker in (
                    "unexpected end of json input",
                    "protocol_error",
                    "unexpected eof",
                    "connection reset by peer",
                    "tls handshake timeout",
                    "502 bad gateway",
                    "503 service unavailable",
                    "504 gateway timeout",
                )
            )
        ):
            return True
    return False


async def run_apptainer_command(
    command: str,
    log_path: Path,
    timeout: float,
    image_pull_max_retries: int,
    min_retry_interval: int,
    max_retry_interval: int,
) -> int:
    if image_pull_max_retries < 1:
        raise ValueError("image_pull_max_retries must be at least 1.")
    for attempt in range(image_pull_max_retries):
        with log_path.open("w" if attempt == 0 else "a") as log:
            if attempt:
                log.write(f"\nRetrying transient image pull (attempt {attempt + 1}/{image_pull_max_retries})\n")
                log.flush()
            offset = log.tell()
            process = await asyncio.create_subprocess_shell(command, stdout=log, stderr=log, start_new_session=True)
            try:
                await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                # The shell launches Apptainer and the agent as descendants.
                # Kill their isolated process group so a hung agent cannot outlive the timeout.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
                raise
        if process.returncode == 0:
            return 0
        # Only inspect this attempt. A previous pull error must not cause a
        # later agent/runtime failure to be retried as another image pull.
        with log_path.open("rb") as log:
            log.seek(offset)
            output = log.read().decode("utf-8", errors="replace")
        if not is_transient_image_pull_failure(output) or attempt + 1 == image_pull_max_retries:
            return process.returncode
        delay = random.randint(min_retry_interval, max_retry_interval)
        LOG.warning("Transient Apptainer pull failure; retrying in %d seconds. Logs: %s", delay, log_path)
        if delay > 0:
            await asyncio.sleep(delay)
    raise AssertionError("Unreachable image pull retry state.")

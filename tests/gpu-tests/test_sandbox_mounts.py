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

import json
from pathlib import Path

import pytest

from nemo_skills.pipeline.cli import run_cmd, wrap_arguments
from tests.conftest import docker_rm


@pytest.mark.gpu
def test_sandbox_mounts_read_only_functional():
    base_dir = Path("/tmp/nemo-skills-tests/sandbox-mounts-read-only")
    sandbox_source = base_dir / "source"
    output_file = base_dir / "result.json"
    script_file = base_dir / "check_sandbox_mount.py"

    docker_rm([str(base_dir)])
    sandbox_source.mkdir(parents=True)
    sandbox_source.joinpath("input.txt").write_text("sandbox-mount-data\n")

    script_file.write_text(
        f"""
import asyncio
import json
from pathlib import Path

from nemo_skills.code_execution.sandbox import get_sandbox


async def main():
    sandbox = get_sandbox("local")
    try:
        read_result, _ = await sandbox.execute_code(
            "cat /sandbox-ro/input.txt",
            language="shell",
            timeout=10,
        )
        write_result, _ = await sandbox.execute_code(
            "touch /sandbox-ro/should-not-exist",
            language="shell",
            timeout=10,
        )
    finally:
        await sandbox.close()

    Path({str(output_file)!r}).write_text(
        json.dumps({{"read": read_result, "write": write_result}}, sort_keys=True)
    )


asyncio.run(main())
""".lstrip()
    )

    run_cmd(
        cluster="test-local",
        config_dir=Path(__file__).parent,
        log_dir=str(base_dir / "logs"),
        with_sandbox=True,
        sandbox_mounts=[f"{sandbox_source}:/sandbox-ro:ro"],
        ctx=wrap_arguments(f"python {script_file}"),
    )

    result = json.loads(output_file.read_text())

    assert result["read"]["process_status"] == "completed"
    assert result["read"]["stdout"] == "sandbox-mount-data\n"
    assert result["write"]["process_status"] == "error"
    assert not sandbox_source.joinpath("should-not-exist").exists()

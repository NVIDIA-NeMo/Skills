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

"""Tests that core modules can be imported without pipeline dependencies.

The core/pipeline boundary rule is: Pipeline can import Core, but Core cannot
import Pipeline. Concretely, this means that core modules must not have
top-level imports of nemo_run or anything under nemo_skills.pipeline.

These tests verify this by importing core modules in a subprocess where
nemo_run is blocked via a sys.modules override.
"""

import pathlib
import subprocess
import sys

import pytest


def _discover_core_modules():
    """Find all importable nemo_skills subpackages except pipeline."""
    import nemo_skills

    root = pathlib.Path(nemo_skills.__file__).parent
    modules = []
    for init in sorted(root.rglob("__init__.py")):
        rel = init.parent.relative_to(root)
        parts = rel.parts
        if not parts:  # root __init__.py
            continue
        if parts[0] == "pipeline" or "__pycache__" in parts:
            continue
        modules.append("nemo_skills." + ".".join(parts))
    return modules


CORE_MODULES = _discover_core_modules()


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_core_module_imports_without_nemo_run(module_name):
    """Each core module must import successfully when nemo_run is unavailable."""
    script = (
        "import sys, importlib; "
        # Block nemo_run from being importable
        "sys.modules['nemo_run'] = None; "
        "sys.modules['nemo_skills.pipeline'] = None; "
        "sys.modules['nemo_skills.pipeline.utils'] = None; "
        f"importlib.import_module('{module_name}'); "
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Importing {module_name} failed when nemo_run is unavailable.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


def test_pipeline_can_import_core():
    """Pipeline modules should be able to import core modules."""
    script = "from nemo_skills.pipeline.dataset import get_dataset_module; print('OK')"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Pipeline failed to import core modules.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

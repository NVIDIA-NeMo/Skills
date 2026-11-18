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

"""Prepare the staged OpenHands checkout for browser-free SWE evaluation on musl."""

import argparse
import os
from pathlib import Path

import tomlkit


def prepare_headless_openhands(root: Path) -> None:
    if os.environ.get("RUN_WITH_BROWSING", "false").lower() == "true":
        raise RuntimeError("OpenHands browser evaluation is not supported on Alpine.")
    project_path = root / "pyproject.toml"
    project = tomlkit.parse(project_path.read_text(encoding="utf-8"))
    browser_packages = {"browsergym-core", "playwright"}
    dependencies = project["tool"]["poetry"]["dependencies"]
    if not browser_packages.issubset(dependencies):
        raise RuntimeError("Unexpected OpenHands dependencies; Alpine setup requires the supported 1.2.1 layout.")
    project["project"]["dependencies"] = [
        dependency
        for dependency in project["project"]["dependencies"]
        if not dependency.startswith(("browsergym-core", "playwright"))
    ]
    for name in browser_packages:
        del dependencies[name]
    # Install the evaluation loader from the same lockfile instead of letting
    # a later pip install upgrade shared dependencies outside OpenHands' pins.
    lock = tomlkit.parse((root / "poetry.lock").read_text(encoding="utf-8"))
    datasets_version = next(package["version"] for package in lock["package"] if package["name"] == "datasets")
    dependencies["datasets"] = datasets_version
    project["project"]["dependencies"].append(f"datasets=={datasets_version}")
    # The SWE runner needs main/runtime, not the browser benchmark and developer groups.
    groups = project["tool"]["poetry"]["group"]
    for name in list(groups):
        if name != "runtime":
            del groups[name]
    project.pop("dependency-groups", None)
    changes = {project_path: tomlkit.dumps(project)}

    def replace(relative_path: str, old: str, new: str) -> None:
        path = root / relative_path
        text = changes.get(path, path.read_text(encoding="utf-8"))
        if text.count(old) != 1:
            raise RuntimeError(f"Unsupported OpenHands source in {relative_path}; expected one occurrence of {old!r}.")
        changes[path] = text.replace(old, new)

    # Register the code agents; browser agents import BrowserGym at module load time.
    path = "openhands/agenthub/__init__.py"
    for name in ("browsing_agent", "visualbrowsing_agent"):
        replace(path, f"    {name},\n", "")
        replace(path, f"    '{name}',\n", "")
    path = "openhands/agenthub/codeact_agent/tools/__init__.py"
    replace(path, "from .browser import BrowserTool\n", "")
    replace(path, "    'BrowserTool',\n", "")
    path = "openhands/agenthub/codeact_agent/codeact_agent.py"
    replace(path, "from openhands.agenthub.codeact_agent.tools.browser import BrowserTool\n", "")
    replace(
        path,
        "        if self.config.enable_browsing:\n",
        "        if self.config.enable_browsing:\n"
        "            from openhands.agenthub.codeact_agent.tools.browser import BrowserTool\n\n",
    )
    path = "openhands/agenthub/codeact_agent/function_calling.py"
    replace(path, "    BrowserTool,\n", "")
    replace(
        path,
        "from openhands.agenthub.codeact_agent.tools import (\n",
        "from openhands.llm.tool_names import BROWSER_TOOL_NAME\n\nfrom openhands.agenthub.codeact_agent.tools import (\n",
    )
    replace(
        path,
        "tool_call.function.name == BrowserTool['function']['name']",
        "tool_call.function.name == BROWSER_TOOL_NAME",
    )
    # Disabled browser actions already return an error before these lazy imports.
    path = "openhands/runtime/action_execution_server.py"
    replace(path, "from openhands.runtime.browser import browse\n", "")
    replace(path, "from openhands.runtime.browser.browser_env import BrowserEnv\n", "")
    replace(
        path,
        "        logger.debug('Initializing browser asynchronously')\n",
        "        from openhands.runtime.browser.browser_env import BrowserEnv\n\n"
        "        logger.debug('Initializing browser asynchronously')\n",
    )
    replace(
        path,
        "        return await browse(action, self.browser, self.initial_cwd)\n",
        "        from openhands.runtime.browser import browse\n\n"
        "        return await browse(action, self.browser, self.initial_cwd)\n",
    )
    replace(
        path,
        "        browser_observation = await browse(action, self.browser, self.initial_cwd)\n",
        "        from openhands.runtime.browser import browse\n\n"
        "        browser_observation = await browse(action, self.browser, self.initial_cwd)\n",
    )
    # Validate every source anchor before modifying the disposable framework clone.
    original = root / ".nemo-skills-original.lock"
    original.write_bytes((root / "poetry.lock").read_bytes())
    for path, text in changes.items():
        path.write_text(text, encoding="utf-8")


def verify_locked_versions(root: Path) -> None:
    def versions(filename: str) -> dict[str, str]:
        lock = tomlkit.parse((root / filename).read_text(encoding="utf-8"))
        return {package["name"]: package["version"] for package in lock["package"]}

    original = versions(".nemo-skills-original.lock")
    current = versions("poetry.lock")
    if {"playwright", "browsergym-core"} & current.keys():
        raise RuntimeError("Browser dependencies remain in the Alpine OpenHands lockfile.")
    changed = {name: version for name, version in current.items() if original.get(name) != version}
    if changed:
        raise RuntimeError(f"Alpine OpenHands setup changed locked package versions: {changed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--check-lockfile", action="store_true")
    args = parser.parse_args()
    if args.check_lockfile:
        verify_locked_versions(args.repo)
    else:
        prepare_headless_openhands(args.repo)

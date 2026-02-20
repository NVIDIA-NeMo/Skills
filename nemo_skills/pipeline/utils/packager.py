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
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import nemo_run as run

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


@dataclass
class RepoMetadata:
    """Metadata for a repo that is used in the experiment."""

    name: str
    path: Path

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)

        if not self.path.exists():
            raise ValueError(f"Repository path `{self.path}` does not exist.")


# Registry of external repos that should be packaged with the code in the experiment
EXTERNAL_REPOS = {
    "nemo_skills": RepoMetadata(
        name="nemo_skills", path=Path(__file__).absolute().parents[2]
    ),  # path to nemo_skills repo
}


def register_external_repo(metadata: RepoMetadata, ignore_if_registered: bool = False):
    """Register an external repo to be packaged with the code in the experiment.

    Args:
        metadata (RepoMetadata): Metadata for the external repo.
        ignore_if_registered (bool): If True, will not raise an error if the repo is already registered.
    """
    if metadata.name in EXTERNAL_REPOS:
        if ignore_if_registered:
            return
        raise ValueError(f"External repo {metadata.name} is already registered.")

    EXTERNAL_REPOS[metadata.name] = metadata


def get_registered_external_repo(name: str) -> Optional[RepoMetadata]:
    """Get the path to the registered external repo.

    Args:
        name (str): Name of the external repo.

    Returns:
        A path to the external repo if it is registered, otherwise None.
    """
    if name not in EXTERNAL_REPOS:
        return None

    return EXTERNAL_REPOS[name]


def resolve_external_data_path(local_data_path: str | Path) -> str:
    """Resolve a local external dataset path to its /nemo_run/code/ container path.

    External repos registered via register_external_repo() are packaged and extracted
    at /nemo_run/code/ relative to their git root. This function finds which registered
    external repo the local_data_path belongs to and computes the correct container path.

    Args:
        local_data_path: Local filesystem path to the dataset's parent directory.

    Returns:
        The container path starting with /nemo_run/code/.

    Raises:
        RuntimeError: If local_data_path doesn't belong to any registered external repo.
    """
    local_data_path = Path(local_data_path).resolve()

    for repo_name, repo_meta in EXTERNAL_REPOS.items():
        if repo_name == "nemo_skills":
            continue

        repo_path = repo_meta.path.resolve()
        try:
            local_data_path.relative_to(repo_path)
        except ValueError:
            continue

        # Found the matching repo. Compute path relative to the git root
        # since git archive produces paths relative to it.
        git_root = get_git_repo_path(repo_path)
        if git_root is None:
            raise RuntimeError(
                f"External repo '{repo_name}' at '{repo_path}' is not a git repository. "
                f"Only git repos can be registered for packaging."
            )
        effective_root = Path(git_root).resolve()
        relative = local_data_path.relative_to(effective_root)
        if str(relative) == ".":
            return "/nemo_run/code"
        return f"/nemo_run/code/{relative}"

    registered = ", ".join(f"'{k}' ({v.path})" for k, v in EXTERNAL_REPOS.items() if k != "nemo_skills")
    raise RuntimeError(
        f"External dataset path '{local_data_path}' does not belong to any registered external repo. "
        f"Registered external repos: {registered or 'none'}. "
        f"Make sure the external repo containing this dataset calls register_external_repo()."
    )


def get_git_repo_path(path: str | Path = None):
    """Check if the path is a git repo.

    Args:
        path: Path to the directory to check. If None, will check the current directory.

    Returns:
        Path to the repo if it is a git repo, otherwise None.
    """
    original_path = os.getcwd()
    try:
        if path:
            os.chdir(path)

        repo_path = (
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
        return Path(repo_path)

    except subprocess.CalledProcessError:
        return None

    finally:
        os.chdir(original_path)


def get_packager(extra_package_dirs: tuple[str] | None = None):
    """Will check if we are running from a git repo and use git packager or default packager otherwise."""
    nemo_skills_dir = get_registered_external_repo("nemo_skills").path

    if extra_package_dirs:
        include_patterns = [str(Path(d) / "*") for d in extra_package_dirs]
        include_pattern_relative_paths = [str(Path(d).parent) for d in extra_package_dirs]
    else:
        include_patterns = []
        include_pattern_relative_paths = []

    check_uncommited_changes = not bool(int(os.getenv("NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK", 0)))

    # are we in a git repo? If yes, we are uploading the current code
    repo_path = get_git_repo_path(path=None)  # check if we are in a git repo in pwd

    if repo_path:
        # Do we have nemo_skills package in this repo? If no, we need to pick it up from installed location
        if not (Path(repo_path) / "nemo_skills").is_dir():
            LOG.info(
                "Not running from Nemo-Skills repo, trying to upload installed package. "
                "Make sure there are no extra files in %s",
                str(nemo_skills_dir / "*"),
            )
            include_patterns.append(str(nemo_skills_dir / "*"))
            include_pattern_relative_paths.append(str(nemo_skills_dir.parent))
        else:
            # picking up local dataset files if we are in the right repo
            dataset_dir = nemo_skills_dir / "dataset"
            for f in dataset_dir.rglob("*.jsonl"):
                include_patterns.append(str(f))
                include_pattern_relative_paths.append(str(nemo_skills_dir.parent))

        root_package = run.GitArchivePackager(
            include_pattern=include_patterns,
            include_pattern_relative_path=include_pattern_relative_paths,
            check_uncommitted_changes=check_uncommited_changes,
        )
    else:
        LOG.info(
            "Not running from a git repo, trying to upload installed package. Make sure there are no extra files in %s",
            str(nemo_skills_dir / "*"),
        )
        include_patterns.append(str(nemo_skills_dir / "*"))
        include_pattern_relative_paths.append(str(nemo_skills_dir.parent))

        root_package = run.PatternPackager(
            include_pattern=include_patterns,
            relative_path=include_pattern_relative_paths,
        )

    extra_repos = {}
    if len(EXTERNAL_REPOS) > 1:
        # Insert root package as the first package
        extra_repos["nemo_run"] = root_package

        for repo_name, repo_meta in EXTERNAL_REPOS.items():
            if repo_name == "nemo_skills":
                continue

            repo_path = repo_meta.path
            if get_git_repo_path(repo_path):
                # Extra repos is a git repos, so we need to package only committed files
                # but also pick up generated jsonl files that may not be committed
                git_root = Path(get_git_repo_path(repo_path)).resolve()
                jsonl_files = list(git_root.rglob("*.jsonl"))
                include_pattern = [str(f) for f in jsonl_files]
                include_pattern_relative_path = [str(git_root)] * len(jsonl_files)
                extra_repos[repo_name] = run.GitArchivePackager(
                    basepath=str(repo_path),
                    include_pattern=include_pattern,
                    include_pattern_relative_path=include_pattern_relative_path,
                    check_uncommitted_changes=check_uncommited_changes,
                )
            else:
                # Extra repos is not a git repo, so we need to package all files in the directory
                repo_include_pattern = [str(Path(repo_path) / "*")]
                repo_include_pattern_relative_path = [str(Path(repo_path).parent)]
                extra_repos[repo_name] = run.PatternPackager(
                    include_pattern=repo_include_pattern,
                    relative_path=repo_include_pattern_relative_path,
                )

        # Return hybrid packager
        return run.HybridPackager(sub_packagers=extra_repos, extract_at_root=True)

    return root_package

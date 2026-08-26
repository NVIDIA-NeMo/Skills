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

"""Repository views that the SHERLOC agent navigates.

A repository snapshot is held in memory as a nested dictionary. Directories map
to sub-dictionaries and a file maps to a record with the keys ``classes``,
``functions`` and ``text``, where ``text`` is the list of source lines. This
module derives from that structure the views used by the agent and by the
evaluation harness:

* ``filter_repo_dict`` removes excluded directories and unwanted file types
* ``tree_repo_dict`` renders the directory tree served by the ``repo_tree`` tool
* ``connected_tree_repo_dict`` renders intra-project import dependencies for the
  ``connected_tree`` tool
* ``calculate_ground_truth_percentage`` reports how much of the reference patch
  survives filtering, which bounds the recall an agent can reach
"""

import logging
import re
from typing import Any, Dict, List


class RepoManager:
    """Repository management utilities."""

    @staticmethod
    def filter_repo_dict(repo_dict: dict, exclude_dirs: list, file_extensions: list) -> dict:
        """Remove excluded directories and unwanted file types from a repository dict.

        Directories left empty by filtering are dropped as well, so the agent is
        never shown a path it cannot read.

        Args:
            repo_dict: Repository snapshot with a 'structure' key
            exclude_dirs: Directory names to remove at any depth
            file_extensions: Extensions to keep, written without the leading dot

        Returns:
            A shallow copy of repo_dict whose 'structure' has been filtered.
        """

        def filter_level(d):
            filtered = {}
            for key, value in d.items():
                # Skip excluded directories
                if key in exclude_dirs:
                    continue

                # Check if it's a file (has extension) and if extension is allowed
                if "." in key and key.split(".")[-1] not in file_extensions:
                    continue

                # Determine if this is a folder
                is_folder = isinstance(value, dict) and set(value.keys()) != {"classes", "functions", "text"}

                if is_folder:
                    # Recursively filter the folder
                    filtered_subfolder = filter_level(value)
                    # Only include the folder if it has content after filtering
                    if filtered_subfolder:
                        filtered[key] = filtered_subfolder
                else:
                    # Include files and leaf nodes
                    filtered[key] = value

            return filtered

        # Create a new repo_dict with filtered structure
        filtered_repo_dict = repo_dict.copy()
        if "structure" in repo_dict:
            filtered_repo_dict["structure"] = filter_level(repo_dict["structure"])

        return filtered_repo_dict

    @staticmethod
    def tree_repo_dict(repo_dict: dict, show_line_counts: bool = True):
        """Render the repository structure as an indented directory tree.

        Args:
            repo_dict: Repository snapshot with a 'structure' key
            show_line_counts: Annotate every file with its line count, which
                lets the agent pick sensible ranges for the view_file tool

        Returns:
            The tree rendering, one entry per line.
        """

        def build_level(d, prefix=""):
            lines = []
            items = list(d.keys())

            for i, key in enumerate(items):
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "|-- "

                node = d[key]
                is_folder = isinstance(node, dict) and set(node.keys()) != {"classes", "functions", "text"}

                # Add line count for files if enabled
                if (
                    show_line_counts
                    and not is_folder
                    and isinstance(node, dict)
                    and "text" in node
                    and isinstance(node["text"], list)
                ):
                    line_count = len(node["text"])
                    lines.append(f"{prefix}{connector}{key} ({line_count} lines)")
                else:
                    lines.append(f"{prefix}{connector}{key}")

                if is_folder:
                    new_prefix = prefix + ("    " if is_last else "|   ")
                    lines.extend(build_level(node, new_prefix))
            return lines

        all_lines = build_level(repo_dict["structure"])
        return ".\n" + "\n".join(all_lines)

    @staticmethod
    def calculate_ground_truth_percentage(
        repo_dict: dict, locations: List[Dict[str, Any]], exclude_dirs: list, file_extensions: list
    ) -> tuple:
        """Report how many reference files survive repository filtering.

        Filtering hides directories and file types from the agent, so a
        reference file that filtering removed can never be predicted. The
        percentage therefore bounds the file level recall attainable on a
        sample, and the diagnostics record why each missing file is absent.

        Args:
            repo_dict: The filtered repository dictionary with a 'structure' key
            locations: Location dicts from extract_locations_from_patch
            exclude_dirs: Directory names removed during filtering
            file_extensions: File extensions kept during filtering

        Returns:
            Tuple of (percentage, debug_info_dict)
        """
        LOG = logging.getLogger(__name__)

        if not locations or "structure" not in repo_dict:
            return 0.0, {}

        # Get all files in the repo tree
        all_files = set()

        def collect_files(node, path=""):
            if isinstance(node, dict):
                for key, value in node.items():
                    # Files are recognised exactly as in filter_repo_dict: a node is a
                    # file when its keys are {'classes', 'functions', 'text'}, and any
                    # other dict is a directory.
                    is_file = False
                    if value is None:
                        # Some files might be represented as None
                        is_file = True
                    elif isinstance(value, dict):
                        # A file has exactly the keys {'classes', 'functions', 'text'}
                        # A folder is any other dict
                        is_folder = set(value.keys()) != {"classes", "functions", "text"}
                        is_file = not is_folder

                    if is_file:
                        file_path = f"{path}/{key}" if path else key
                        all_files.add(file_path)
                        LOG.debug(f"Found file in repo: {file_path}")
                    elif isinstance(value, dict):  # It's a directory
                        new_path = f"{path}/{key}" if path else key
                        collect_files(value, new_path)

        collect_files(repo_dict["structure"])
        LOG.info(f"Total files found in filtered repo: {len(all_files)}")

        # Check how many ground truth files exist
        ground_truth_files = {loc["file_path"] for loc in locations}
        LOG.info(f"Ground truth files from patch: {ground_truth_files}")

        # Check if ground truth files were filtered out by extension
        filtered_out_by_extension = []
        for gt_file in ground_truth_files:
            if "." in gt_file:
                ext = gt_file.split(".")[-1]
                if ext not in file_extensions:
                    filtered_out_by_extension.append((gt_file, ext))

        if filtered_out_by_extension:
            LOG.warning(f"Ground truth files filtered out by extension: {filtered_out_by_extension}")
            LOG.info(f"Allowed extensions: {file_extensions}")

        # Check if ground truth files are in excluded directories
        filtered_out_by_dir = []
        for gt_file in ground_truth_files:
            path_parts = gt_file.split("/")
            for part in path_parts[:-1]:  # Check all directory parts except filename
                if part in exclude_dirs:
                    filtered_out_by_dir.append((gt_file, part))
                    break

        if filtered_out_by_dir:
            LOG.warning(f"Ground truth files in excluded directories: {filtered_out_by_dir}")
            LOG.info(f"Excluded directories: {exclude_dirs[:10]}...")  # Show first 10

        # Check for exact matches
        existing_files = ground_truth_files.intersection(all_files)

        # Also check for potential path mismatches (e.g., leading slashes, different separators)
        normalized_existing = []
        if not existing_files and ground_truth_files:
            LOG.debug("No exact matches found, checking for path variations...")
            # Normalize paths for comparison
            normalized_repo_files = {f.strip("/").replace("//", "/") for f in all_files}
            normalized_gt_files = {f.strip("/").replace("//", "/") for f in ground_truth_files}
            normalized_existing = normalized_gt_files.intersection(normalized_repo_files)
            if normalized_existing:
                LOG.warning(
                    f"Found {len(normalized_existing)} files with normalized paths, but paths don't match exactly"
                )
                LOG.debug(f"Sample repo files: {list(all_files)[:5]}")
                LOG.debug(f"Sample GT files: {list(ground_truth_files)[:5]}")

        percentage = 0.0
        if ground_truth_files:
            percentage = (len(existing_files) / len(ground_truth_files)) * 100

        LOG.info(
            f"Ground truth files: {len(ground_truth_files)}, Existing in repo: {len(existing_files)} ({percentage:.1f}%)"
        )

        # Provide explanation if percentage is 0
        if ground_truth_files and percentage == 0:
            LOG.warning("0% ground truth files found in filtered repo structure!")
            if filtered_out_by_extension:
                LOG.warning(f"  - {len(filtered_out_by_extension)} files filtered by extension")
            if filtered_out_by_dir:
                LOG.warning(f"  - {len(filtered_out_by_dir)} files in excluded directories")
            if not filtered_out_by_extension and not filtered_out_by_dir:
                LOG.warning("  - Files may have been excluded during repo loading or have path mismatches")

        # Collect missing files for detailed reporting
        missing_files = ground_truth_files - existing_files
        missing_files_details = []

        for missing_file in missing_files:
            reason = "unknown"
            details = {"file": missing_file}

            # Check if filtered by extension
            if "." in missing_file:
                ext = missing_file.split(".")[-1]
                if ext not in file_extensions:
                    reason = "filtered_by_extension"
                    details["extension"] = ext

            # Check if filtered by directory
            path_parts = missing_file.split("/")
            for part in path_parts[:-1]:  # Check all directory parts except filename
                if part in exclude_dirs:
                    reason = "filtered_by_directory"
                    details["excluded_dir"] = part
                    break

            if reason == "unknown":
                # Not removed by the extension/directory filters, so it is
                # absent from the snapshot itself.
                reason = "not_in_repository"

            details["reason"] = reason
            missing_files_details.append(details)

        debug_info = {
            "total_repo_files": len(all_files),
            "total_ground_truth_files": len(ground_truth_files),
            "existing_files": len(existing_files),
            "filtered_by_extension": len(filtered_out_by_extension),
            "filtered_by_dir": len(filtered_out_by_dir),
            "normalized_matches": len(normalized_existing) if normalized_existing else 0,
            "missing_files_details": missing_files_details,
        }

        return percentage, debug_info

    @staticmethod
    def connected_tree_repo_dict(repo_dict: dict, target_file: str = None, show_line_counts: bool = False):
        """
        Generate a concise connected tree showing ONLY internal project dependencies.

        Key features:
        - Shows ONLY imports from detected top-level project packages and relative imports
        - Excludes ALL external libraries (no numpy, torch, hydra, etc.)
        - Skips test/docs/build/cache directories
        - Only analyzes Python files (.py)
        - Limits display to 10 imports/dependents per file
        - Summary mode shows top 10 most connected files

        Args:
            repo_dict: Repository dictionary structure
            target_file: Optional file path to focus on (if None, shows summary)
            show_line_counts: Whether to show line counts for files (default: False)

        Returns:
            Concise string showing internal project structure and dependencies

        Example output for specific file:
            ═══ src/service.py ═══

            → IMPORTS (2):
               • src.utils
               • .models

            ← IMPORTED BY (1):
               • src/app.py
        """

        def detect_internal_modules(structure):
            """Detect the base module names from the repository structure."""
            internal_modules = set()

            # Look at top-level directories that contain Python files
            for key, value in structure.items():
                if isinstance(value, dict):
                    # Check if this directory contains Python files
                    def check_for_python(d):
                        for k, v in d.items():
                            if isinstance(v, dict):
                                if "text" in v and k.endswith(".py"):
                                    return True
                                if check_for_python(v):
                                    return True
                        return False

                    if check_for_python(value) or (key + ".py" in structure):
                        internal_modules.add(key)

            return tuple(internal_modules)

        def extract_imports_from_file(file_node, internal_modules=None):
            """Extract ONLY internal import statements from a file node."""
            if not isinstance(file_node, dict) or "text" not in file_node:
                return []

            imports = set()
            lines = file_node["text"]

            # Use the detected internal modules.
            if internal_modules is None:
                internal_modules = ()

            for line in lines:
                line = line.strip()
                # Match various import patterns
                if line.startswith("import ") or line.startswith("from "):
                    # Handle "from module import ..." and "import module"
                    if line.startswith("from "):
                        # Handle relative imports
                        if line.startswith("from ."):
                            # Relative import - always internal
                            match = re.match(r"from\s+(\.[^\s]+)\s+import", line)
                            if match:
                                imports.add(match.group(1))
                        else:
                            # Absolute import
                            match = re.match(r"from\s+([^\s]+)\s+import", line)
                            if match:
                                module = match.group(1)
                                # Only keep if it's an internal module
                                if any(module.startswith(internal) for internal in internal_modules):
                                    imports.add(module)
                    elif line.startswith("import "):
                        # import module, module2
                        match = re.match(r"import\s+(.+)", line)
                        if match:
                            modules = match.group(1).split(",")
                            for module in modules:
                                module = module.strip().split(" as ")[0]  # Remove 'as alias'
                                # Only keep if it's an internal module
                                if any(module.startswith(internal) for internal in internal_modules):
                                    imports.add(module)

            return list(imports)

        def normalize_module_to_file(module, all_files):
            """Convert internal module name to actual file paths in the repo."""
            matches = []

            # Handle relative imports
            if module.startswith("."):
                # A relative import cannot be resolved to a path without knowing the
                # importing package. Such imports are internal by construction, so they
                # are reported as imports but are not linked to a specific file.
                return matches

            # Convert module path to file path patterns
            # e.g., 'nemo_skills.inference.eval' -> 'nemo_skills/inference/eval'
            module_path = module.replace(".", "/")

            for file_path in all_files:
                # Check for exact module match with .py extension
                if file_path == module_path + ".py":
                    matches.append(file_path)
                # Check for __init__.py in package
                elif file_path == module_path + "/__init__.py":
                    matches.append(file_path)
                # Check if the module path is part of the file path
                elif module_path in file_path:
                    # Make sure it's a proper path component match
                    # e.g., 'utils' shouldn't match 'myutils.py'
                    path_parts = file_path.split("/")
                    module_parts = module_path.split("/")
                    for i in range(len(path_parts) - len(module_parts) + 1):
                        if path_parts[i : i + len(module_parts)] == module_parts:
                            matches.append(file_path)
                            break

            return list(set(matches))  # Remove duplicates

        def collect_file_dependencies(structure, internal_modules):
            """Collect all files with their dependencies."""
            files_data = {}

            # Directories to skip for faster processing
            skip_dirs = {
                "__pycache__",
                ".git",
                "node_modules",
                "venv",
                ".env",
                "dist",
                "build",
                "tests",
                "test",
                "testing",
                "docs",
                "documentation",
                "migrations",
                ".pytest_cache",
                ".mypy_cache",
                ".tox",
                "htmlcov",
                "coverage",
            }

            def traverse(d, path=""):
                for key, value in d.items():
                    # Skip certain directories
                    if key in skip_dirs:
                        continue

                    current_file_path = f"{path}/{key}" if path else key

                    if isinstance(value, dict) and "text" in value:
                        # It's a file - only process Python files
                        if key.endswith(".py"):
                            imports = extract_imports_from_file(value, internal_modules)
                            if imports:  # Only include files that have internal imports
                                files_data[current_file_path] = {
                                    "imports": imports,
                                    "line_count": len(value["text"]) if show_line_counts else None,
                                    "dependencies": [],  # Will be filled later
                                    "dependents": [],  # Will be filled later
                                }
                    elif isinstance(value, dict):
                        # It's a directory
                        traverse(value, current_file_path)

            traverse(structure)

            # Now resolve imports to actual files
            all_files = list(files_data.keys())
            for file_path, file_info in files_data.items():
                for imported_module in file_info["imports"]:
                    matching_files = normalize_module_to_file(imported_module, all_files)
                    for match in matching_files:
                        if match != file_path:  # Don't self-reference
                            file_info["dependencies"].append(match)
                            files_data[match]["dependents"].append(file_path)

            # Remove duplicates
            for file_info in files_data.values():
                file_info["dependencies"] = list(set(file_info["dependencies"]))
                file_info["dependents"] = list(set(file_info["dependents"]))

            return files_data

        def build_dependency_tree(files_data, target_file=None):
            """Build a concise dependency tree representation."""
            if target_file:
                # Show dependency info for specific file
                if target_file not in files_data:
                    return f"ERROR: File '{target_file}' not found in repository."

                file_info = files_data[target_file]
                line_count = (
                    f" [{file_info['line_count']}L]"
                    if show_line_counts and file_info["line_count"] is not None
                    else ""
                )

                lines = [f"═══ {target_file}{line_count} ═══"]

                # Show imports (what this file depends on)
                if file_info["dependencies"]:
                    lines.append(f"\n→ IMPORTS ({len(file_info['dependencies'])}):")
                    for dep in sorted(file_info["dependencies"])[:10]:  # Limit to 10 most relevant
                        lines.append(f"   • {dep}")
                    if len(file_info["dependencies"]) > 10:
                        lines.append(f"   ... and {len(file_info['dependencies']) - 10} more")
                else:
                    lines.append("\n→ IMPORTS: None")

                # Show dependents (what depends on this file)
                if file_info["dependents"]:
                    lines.append(f"\n← IMPORTED BY ({len(file_info['dependents'])}):")
                    for dep in sorted(file_info["dependents"])[:10]:  # Limit to 10 most relevant
                        lines.append(f"   • {dep}")
                    if len(file_info["dependents"]) > 10:
                        lines.append(f"   ... and {len(file_info['dependents']) - 10} more")
                else:
                    lines.append("\n← IMPORTED BY: None")

                return "\n".join(lines)

            else:
                # Show summary of most connected files only
                lines = ["═══ REPOSITORY CONNECTION SUMMARY ═══\n"]

                # Find most connected files (by total connections)
                connection_scores = []
                for file_path, file_info in files_data.items():
                    total_connections = len(file_info["dependencies"]) + len(file_info["dependents"])
                    if total_connections > 0:  # Only show files with connections
                        connection_scores.append((total_connections, file_path, file_info))

                # Sort by connection count
                connection_scores.sort(reverse=True)

                # Show top 10 most connected files
                lines.append("TOP CONNECTED FILES:")
                for i, (score, file_path, file_info) in enumerate(connection_scores[:10]):
                    imports = len(file_info["dependencies"])
                    imported_by = len(file_info["dependents"])
                    lines.append(f"{i + 1:2d}. {file_path}")
                    lines.append(f"    → imports: {imports}, ← imported by: {imported_by}")

                if len(connection_scores) > 10:
                    lines.append(f"\n... and {len(connection_scores) - 10} more files with connections")

                # Summary statistics
                total_files = len(files_data)
                connected_files = len(connection_scores)
                lines.append(f"\nTOTAL: {connected_files}/{total_files} files have import connections")

                return "\n".join(lines)

        # Main logic
        # Auto-detect internal modules from the repository structure
        internal_modules = detect_internal_modules(repo_dict["structure"])

        # Log detected modules for debugging
        logger = logging.getLogger(__name__)
        logger.debug(f"Detected internal modules: {internal_modules}")

        # Collect dependencies using only internal modules
        files_data = collect_file_dependencies(repo_dict["structure"], internal_modules)

        return build_dependency_tree(files_data, target_file)

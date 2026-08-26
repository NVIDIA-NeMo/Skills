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

"""Execution of the repository inspection tools offered to the SHERLOC agent.

The agent never touches a checkout on disk. It reads the repository snapshot
held in memory as a nested dictionary, which keeps tool calls deterministic and
free of side effects. Four tools are exposed:

* ``view_file`` prints a line numbered slice of a file
* ``repo_tree`` prints the directory tree
* ``connected_tree`` prints intra-project import dependencies
* ``codebase_search`` prints context windows around literal matches

Every tool returns its rendered output together with a token count. The served
model's tokenizer is used when available; otherwise the same four-characters-
per-token estimate as the initial repository prompt is used. Failures are
returned as text for the model to read rather than raised, so a bad tool call
costs a turn instead of the trajectory.
"""

import logging
from typing import Tuple

from nemo_skills.utils import get_logger_name
from recipes.sherloc.inference.sherloc_utils.repo_manager import RepoManager

LOG = logging.getLogger(get_logger_name(__file__))


class ToolExecutor:
    """Handles tool execution for the assistant against a nested dictionary representation of a repository."""

    def __init__(self, cfg, tokenizer=None):
        """Create an executor bound to a generation config.

        Args:
            cfg: Generation config. ``max_view_lines`` caps the view_file output,
                and ``show_line_counts`` controls tree rendering.
            tokenizer: Tokenizer used to count tool output when available.
        """
        self.cfg = cfg
        self.tokenizer = tokenizer

    def _count_tokens(self, text: str) -> int:
        """Count tokens with the model tokenizer, or estimate when unavailable."""
        if not text:
            return 0
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(text) // 4

    def execute_tool(self, extracted_block: dict, repo_dict: dict) -> Tuple[str, int]:
        """Execute a parsed tool call against the repository snapshot.

        Args:
            extracted_block: Parsed tool call, such as
                ``{"tool": "view_file", "path": ..., "view_range": ...}``
            repo_dict: Repository snapshot with a 'structure' key

        Returns:
            Tuple of (output_content, token_count). Errors are returned as
            output text so the model can correct itself on the next turn.
        """
        tool_name = extracted_block.get("tool", "")

        # Some responses omit the "tool" field; infer the tool from the parameters
        if not tool_name:
            if "path" in extracted_block:
                tool_name = "view_file"
            elif "query" in extracted_block:
                tool_name = "codebase_search"
            elif "file" in extracted_block:
                tool_name = "connected_tree"
            else:
                LOG.error(f"Cannot determine tool from extracted_block: {extracted_block}")
                error_msg = "Error: Cannot determine which tool to execute. Please specify 'tool' field in your JSON."
                return error_msg, self._count_tokens(error_msg)

        LOG.info(f"Executing tool: {tool_name}")

        if tool_name == "view" or tool_name == "view_file":
            return self._execute_view_tool(extracted_block, repo_dict)
        elif tool_name == "repo_tree":
            return self._execute_repo_tree_tool(repo_dict)
        elif tool_name == "connected_tree":
            return self._execute_connected_tree_tool(extracted_block, repo_dict)
        elif tool_name == "codebase_search":
            return self._execute_codebase_search_tool(extracted_block, repo_dict)
        else:
            LOG.error(f"Unknown tool: {tool_name}")
            error_msg = f"Error: Unknown tool '{tool_name}'"
            return error_msg, self._count_tokens(error_msg)

    def _get_node_from_path(self, repo_dict: dict, path: str):
        """Helper function to navigate the nested dict using a file path."""
        # Normalize path to handle both empty and non-empty paths
        parts = [part for part in path.split("/") if part]
        current_level = repo_dict["structure"]
        try:
            for part in parts:
                current_level = current_level[part]
            return current_level
        except (KeyError, TypeError):
            return None

    def _is_file_node(self, node: dict) -> bool:
        """Checks if a node in the dictionary represents a file."""
        return isinstance(node, dict) and "text" in node and isinstance(node["text"], list)

    def _is_dir_node(self, node: dict) -> bool:
        """Checks if a node in the dictionary represents a directory."""
        return isinstance(node, dict) and "text" not in node

    def _find_similar_files(self, repo_dict: dict, target_filename: str, max_suggestions: int = 3) -> list:
        """Find files with similar names to the target filename."""
        similar_files = []
        target_name = target_filename.lower()

        def search_recursive(d, current_path=""):
            for key, value in d.items():
                current_file_path = f"{current_path}/{key}" if current_path else key

                if self._is_file_node(value):
                    # Check if filename matches (case-insensitive)
                    if key.lower() == target_name:
                        similar_files.append(current_file_path)
                    # Check if filename contains the target or target contains filename
                    elif target_name in key.lower() or key.lower() in target_name:
                        similar_files.append(current_file_path)
                elif self._is_dir_node(value):
                    search_recursive(value, current_file_path)

        search_recursive(repo_dict["structure"])

        # Sort by similarity (exact matches first, then by length difference)
        def similarity_score(filepath):
            filename = filepath.split("/")[-1].lower()
            if filename == target_name:
                return 0  # Exact match gets highest priority
            elif target_name in filename:
                return 1  # Target contained in filename
            elif filename in target_name:
                return 2  # Filename contained in target
            else:
                return 3  # Other matches

        similar_files.sort(key=similarity_score)
        return similar_files[:max_suggestions]

    def _execute_view_tool(self, extracted_block: dict, repo_dict: dict) -> Tuple[str, int]:
        """Execute the view tool to show file contents from the dictionary."""
        try:
            max_lines = self.cfg.max_view_lines
            file_path = extracted_block.get("path", "")
            if not file_path:
                error_msg = "Error: No file path provided for the 'view' tool."
                return error_msg, self._count_tokens(error_msg)
            view_range = extracted_block.get("view_range")

            LOG.info(f"Viewing file: {file_path}")
            if view_range:
                LOG.info(f"Range: lines {view_range[0]}-{view_range[1]}")

            file_node = self._get_node_from_path(repo_dict, file_path)

            if not self._is_file_node(file_node):
                LOG.error(f"File not found or is a directory: {file_path}")

                # Extract just the filename for searching similar files
                filename = file_path.split("/")[-1] if "/" in file_path else file_path
                similar_files = self._find_similar_files(repo_dict, filename)

                error_msg = f"Error: File not found at path '{file_path}'"

                if similar_files:
                    error_msg += "\n\n💡 Did you mean one of these files?"
                    for i, similar_file in enumerate(similar_files, 1):
                        error_msg += f"\n   {i}. {similar_file}"
                    error_msg += "\n\nTry using the exact path from the suggestions above."
                else:
                    error_msg += f"\n\n💡 No similar files found with name '{filename}'. Use the repo_tree tool to browse available files."

                return error_msg, self._count_tokens(error_msg)

            lines = file_node["text"]  # The lines are already clean, without trailing '\n'

            if view_range is None:
                start_line, end_line = 1, len(lines)
                LOG.info(f"Showing entire file ({len(lines)} lines)")
            else:
                if (
                    not isinstance(view_range, list)
                    or len(view_range) != 2
                    or not all(isinstance(line_number, int) for line_number in view_range)
                ):
                    LOG.error(f"Invalid view_range format: {view_range}. Expected [start, end] or [start, -1]")
                    error_msg = (
                        f"Error: Invalid view_range format. Expected [start, end] or [start, -1], got {view_range}"
                    )
                    return error_msg, self._count_tokens(error_msg)
                start_line, end_line = view_range[0], view_range[1]
                if end_line == -1:
                    end_line = len(lines)
                    LOG.info(f"Showing from line {start_line} to end of file ({len(lines)} lines)")
                else:
                    LOG.info(f"Showing lines {start_line}-{end_line}")

            if start_line < 1:
                start_line = 1
            if end_line > len(lines):
                end_line = len(lines)
            if start_line > end_line:
                LOG.error(f"Start line {start_line} is greater than end line {end_line}")
                # Provide the error message but also show the entire file for context
                error_msg = f"Error: Start line {start_line} is greater than end line {end_line}"
                error_msg += "\n\nShowing entire file to help you understand the file size:\n"
                error_msg += f"File: {file_path} (total lines: {len(lines)})\n"
                error_msg += "=" * 80 + "\n"

                # Show entire file with line numbers
                if max_lines > 0 and len(lines) > max_lines:
                    # Truncate if file is too large
                    result_lines = [f"{i + 1:4d}: {lines[i]}" for i in range(max_lines)]
                    error_msg += "\n".join(result_lines)
                    error_msg += f"\n\n... ({len(lines) - max_lines} more lines truncated)"
                else:
                    result_lines = [f"{i + 1:4d}: {lines[i]}" for i in range(len(lines))]
                    error_msg += "\n".join(result_lines)

                return error_msg, self._count_tokens(error_msg)

            # Check if file content needs truncation
            total_lines_to_show = end_line - start_line + 1
            truncated = False
            truncation_message = ""

            if max_lines > 0 and total_lines_to_show > max_lines:
                # Truncate to max_lines, but keep the requested start_line
                original_end_line = end_line
                end_line = start_line + max_lines - 1
                truncated = True
                truncation_message = f"\nWARNING: File content truncated. Showing {max_lines} lines out of {total_lines_to_show} requested lines (original range: {start_line}-{original_end_line}). Use smaller ranges to view specific sections.\n"
                LOG.warning(f"File {file_path} truncated from {total_lines_to_show} to {max_lines} lines")

            result_lines = [f"{i + 1:4d}: {lines[i]}" for i in range(start_line - 1, end_line)]
            LOG.info(f"Successfully read {len(result_lines)} lines from {file_path}")

            file_header = f"File: {file_path} (lines {start_line}-{end_line})"
            if truncated:
                file_header += f" [TRUNCATED - Original file has {len(lines)} total lines]"

            file_content = file_header + truncation_message + "\n" + "\n".join(result_lines)
            return file_content, self._count_tokens(file_content)

        except (IndexError, TypeError) as e:
            LOG.error(f"Error executing view tool: {str(e)}")
            error_msg = f"Error executing view tool: {str(e)}"
            return error_msg, self._count_tokens(error_msg)

    def _execute_repo_tree_tool(self, repo_dict: dict) -> Tuple[str, int]:
        """Generates a tree view of the repository from the nested dictionary."""
        try:
            show_line_counts = self.cfg.show_line_counts if self.cfg else True
            tree_output = RepoManager.tree_repo_dict(repo_dict, show_line_counts)
            return tree_output, self._count_tokens(tree_output)
        except Exception as e:
            LOG.error(f"Error executing repo_tree tool: {str(e)}")
            error_msg = f"Error executing repo_tree tool: {str(e)}"
            return error_msg, self._count_tokens(error_msg)

    def _execute_connected_tree_tool(self, extracted_block: dict, repo_dict: dict) -> Tuple[str, int]:
        """Generates a connected tree view showing import dependencies."""
        try:
            file_path = extracted_block.get("file", None)
            show_line_counts = self.cfg.show_line_counts if self.cfg else True

            LOG.info(f"Generating connected tree for file: {file_path or 'entire repository'}")
            tree_output = RepoManager.connected_tree_repo_dict(repo_dict, file_path, show_line_counts)
            return tree_output, self._count_tokens(tree_output)
        except Exception as e:
            LOG.error(f"Error executing connected_tree tool: {str(e)}")
            error_msg = f"Error executing connected_tree tool: {str(e)}"
            return error_msg, self._count_tokens(error_msg)

    def _execute_codebase_search_tool(self, extracted_block: dict, repo_dict: dict) -> Tuple[str, int]:
        """Executes a codebase search on the nested dictionary, showing context around matches."""
        query = extracted_block.get("query", "")
        if not query:
            error_msg = "Error: No search query provided for codebase_search tool."
            return error_msg, self._count_tokens(error_msg)

        LOG.info(f"Searching codebase for: {query}")

        try:
            results = []
            query_lower = query.lower()

            def search_recursive(current_dict: dict, current_path: str):
                for name, node in current_dict.items():
                    new_path = f"{current_path}/{name}" if current_path else name

                    if self._is_dir_node(node):
                        search_recursive(node, new_path)

                    elif self._is_file_node(node):
                        lines = node["text"]
                        # Find all lines containing the query (using 0-based indexing)
                        match_indices = [i for i, line in enumerate(lines) if query_lower in line.lower()]

                        # If no matches were found in this file, skip it
                        if not match_indices:
                            continue

                        file_result_parts = [f"File: {new_path}"]
                        total_matches_in_file = len(match_indices)

                        # Use a set to track lines already shown in a snippet to avoid overlaps
                        # for matches that are close to each other.
                        shown_indices = set()

                        MAX_SNIPPETS_PER_FILE = 3  # Limit the number of context blocks per file
                        snippets_created = 0

                        for match_idx in match_indices:
                            if snippets_created >= MAX_SNIPPETS_PER_FILE:
                                break

                            # If this match was already included in a previous snippet's context, skip it.
                            if match_idx in shown_indices:
                                continue

                            snippets_created += 1

                            # Define the context window: 20 lines before, the match, 20 lines after
                            start_idx = max(0, match_idx - 20)
                            end_idx = min(len(lines), match_idx + 21)

                            file_result_parts.append(
                                f"\n--- Snippet {snippets_created} (match on line {match_idx + 1}) ---"
                            )

                            for i in range(start_idx, end_idx):
                                line_num = i + 1
                                line_content = lines[i].rstrip()

                                # Highlight the specific matching line with a '>'
                                if i == match_idx:
                                    prefix = f"> {line_num:4d}"
                                else:
                                    prefix = f"  {line_num:4d}"

                                file_result_parts.append(f"{prefix}: {line_content}")
                                shown_indices.add(i)

                        # Add a summary if some matches were not shown in detail
                        if total_matches_in_file > snippets_created:
                            remaining_matches = total_matches_in_file - snippets_created
                            file_result_parts.append(f"\n... and {remaining_matches} more match(es) in this file.")

                        # The -total_matches_in_file is used to sort results by relevance (most matches first)
                        results.append((-total_matches_in_file, "\n".join(file_result_parts)))

            search_recursive(repo_dict["structure"], "")

            # Sort results by match count (descending), then by file path (ascending)
            results.sort()
            results = results[:5]

            # Extract just the formatted string part for the final output
            final_results = [res[1] for res in results]

            if not final_results:
                no_results_msg = f"No results found for query: '{query}'"
                return no_results_msg, self._count_tokens(no_results_msg)

            # Add a brief header to indicate these are search results
            header = f"Search results for: '{query}' (showing top {len(final_results)} files with most matches)\n"
            header += "=" * 50

            # Join the results from different files with a clear separator
            results_text = "\n\n==================================================\n\n".join(final_results)
            full_output = header + "\n\n" + results_text
            return full_output, self._count_tokens(full_output)

        except Exception as e:
            LOG.error(f"Error executing codebase_search tool: {str(e)}")
            error_msg = f"Error executing codebase_search tool: {str(e)}"
            return error_msg, self._count_tokens(error_msg)

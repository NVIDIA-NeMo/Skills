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

"""Shared utilities for proof processing and evaluation."""

import re
from dataclasses import dataclass
from typing import Any, Dict

from nemo_skills.code_execution.utils import clean_formal_generation
from nemo_skills.dataset.utils import get_lean4_header


@dataclass
class ProofBuildConfig:
    """Configuration for building proofs from generations."""

    final_answer_key: str | None = None
    extract_code_mode: str = "last"  # "first" or "last"
    restate_formal_statement: bool = True
    strip_theorem_from_proof: bool = True


def extract_proof_only(lean_code: str) -> str:
    """Extract only the proof part from a Lean theorem/example.

    This function removes the theorem/example header and returns just the proof body.

    Args:
        lean_code: The full Lean code including theorem statement

    Returns:
        The proof part only (everything after ':=')
    """
    lines = lean_code.strip().splitlines()
    if not lines:
        return ""

    header_start_pattern = re.compile(r"^\s*(theorem|example)\b")
    header_start_idx = None

    # 1. Find where the theorem starts
    for i, line in enumerate(lines):
        if header_start_pattern.match(line):
            header_start_idx = i
            break

    if header_start_idx is None:
        return lean_code.strip()

    # 2. Find where ':=' occurs, starting from the header
    header_end_idx = None
    for i in range(header_start_idx, len(lines)):
        if ":=" in lines[i]:
            header_end_idx = i
            break

    if header_end_idx is None:
        return lean_code.strip()

    # 3. Extract the line after ':='
    header_line, after = lines[header_end_idx].split(":=", 1)
    proof_first_line = after.strip()

    # 4. Collect proof lines
    if proof_first_line:
        proof_lines = [proof_first_line] + lines[header_end_idx + 1 :]
    else:
        proof_lines = lines[header_end_idx + 1 :]

    # 5. Remove leading 'by' (with or without indentation)
    if proof_lines:
        first = proof_lines[0].lstrip()
        if first == "by":
            proof_lines = proof_lines[1:]
        elif first.startswith("by "):
            proof_lines[0] = first[3:]  # Strip 'by '

    return "\n".join(proof_lines).rstrip()


def build_lean4_proof(
    generation: str, data_point: Dict[str, Any], config: ProofBuildConfig, answer_format: str = "lean4-proof"
) -> str:
    """Build a complete Lean4 proof from generation and data point.

    Args:
        generation: The raw generation from the model
        data_point: Dictionary containing header, formal_statement, etc.
        config: Configuration for proof building
        answer_format: Either "lean4-proof" or "lean4-statement"

    Returns:
        Complete Lean4 proof ready for execution
    """
    if answer_format == "lean4-proof":
        # Clean the generation and extract the formal proof
        cleaned_generation = clean_formal_generation(
            generation, final_answer_key=config.final_answer_key, extract_code_mode=config.extract_code_mode
        )

        # Combine header + formal_statement + proof
        header = data_point.get("header", "")
        formal_statement = data_point.get("formal_statement", "") if config.restate_formal_statement else ""

        if config.strip_theorem_from_proof:
            proof_part = extract_proof_only(cleaned_generation)
        else:
            proof_part = cleaned_generation

        predicted_proof = header + formal_statement + proof_part

    elif answer_format == "lean4-statement":
        # For statements, add header and append sorry
        cleaned_generation = clean_formal_generation(generation, extract_code_mode=config.extract_code_mode)
        header = get_lean4_header()
        predicted_proof = header + cleaned_generation + "\n sorry"

    else:
        raise ValueError(f"Unknown answer_format: {answer_format}")

    return predicted_proof


def determine_proof_status(compiler_output: Dict[str, Any]) -> str:
    """Determine proof status from compiler output.

    Args:
        compiler_output: Dictionary containing process_status, stdout, stderr

    Returns:
        Status string: "completed", "timeout", "has_sorry", or other process status
    """
    process_status = compiler_output.get("process_status", "unknown")

    if process_status == "timeout":
        return "timeout"
    elif process_status != "completed":
        return process_status

    # Check stdout and stderr for proof status indicators
    stdout = compiler_output.get("stdout", "").lower()
    stderr = compiler_output.get("stderr", "").lower()
    combined = stdout + "\n" + stderr

    # Check for sorry (incomplete proof)
    if re.search(r"\bsorry\b", combined) is not None:
        return "has_sorry"

    # If process completed without errors, consider it successful
    return "completed"


def prepare_predicted_proof_from_line_dict(
    line_dict: Dict[str, Any],
    config: ProofBuildConfig,
    answer_format: str = "lean4-proof",
    use_predicted_proof_key: bool = False,
) -> str:
    """Prepare predicted_proof from a line dictionary (for batch processing).

    Args:
        line_dict: Dictionary containing generation and other fields
        config: Configuration for proof building
        answer_format: Either "lean4-proof" or "lean4-statement"
        use_predicted_proof_key: If True, use existing predicted_proof key

    Returns:
        Complete Lean4 proof ready for execution

    Raises:
        ValueError: If use_predicted_proof_key is True but key is missing
    """
    if use_predicted_proof_key:
        if "predicted_proof" not in line_dict:
            raise ValueError(
                "predicted_proof key not found in the line_dict. Set use_predicted_proof_key=False to re-combine"
            )
        return line_dict["predicted_proof"]

    return build_lean4_proof(
        generation=line_dict["generation"], data_point=line_dict, config=config, answer_format=answer_format
    )

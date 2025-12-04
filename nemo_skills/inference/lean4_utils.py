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

import logging
import re

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))

# ------------------------------------------------------------------------------------------------
# The following code is adapted from https://github.com/Goedel-LM/Goedel-Prover-V2
# ------------------------------------------------------------------------------------------------


def remove_comments(text):  # remove comments
    # First remove all /- ... -/ blocks
    text = re.sub(r"/-.*?-/", "", text, flags=re.DOTALL)
    # Then remove -- comments from each line
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        cleaned_line = line.split("--", 1)[0]
        if cleaned_line.strip() == "":
            continue
        cleaned_lines.append(cleaned_line)
    # Join back together and remove excessive empty lines
    cleaned_text = "\n".join(cleaned_lines)
    return cleaned_text.strip()


def move_imports_to_beginning(input_string):
    lines = input_string.split("\n")
    import_lines = [line for line in lines if line.startswith("import")]
    other_lines = [line for line in lines if not line.startswith("import")]
    return "\n".join(import_lines + other_lines)


def return_theorem_to_prove(text):
    # Pattern that matches from 'theorem' or 'lemma' to ':= by sorry' with any content in between
    pattern = r"((?:theorem).*?:=\s*by\s*sorry)"
    match = re.search(pattern, text, re.DOTALL)
    return match.span() if match else None


def return_theorem_to_replace(text):
    # Pattern that matches from 'theorem' or 'lemma' to ':= by sorry' with any content in between
    pattern = r"((?:^|\s)theorem\s+.*?:=\s*by)"
    match = re.search(pattern, text, re.DOTALL)
    return match.span() if match else None


def replace_statement_in_proof(statement, proof):
    if ("apply?" in proof) or ("exact?" in proof):
        return "**Error**, 'apply?' or 'exact?' is used, which is not allowed."
    stats_re = remove_comments(statement)
    stats_span_ = return_theorem_to_prove(stats_re)
    if stats_span_ is None:
        error_app = "\n".join(["\n"] + ["-- " + x for x in statement.split("\n")])
        return f"**Error**, can not find 'theorem' and ':= sorry' in {error_app}"
    proof_str = remove_comments(proof)
    span = return_theorem_to_replace(proof_str)
    if span is None:
        error_app = "\n".join(["\n"] + ["-- " + x for x in proof.split("\n")])
        return f"**Error**, can not find 'theorem' and ':=' in {error_app}"
    return stats_re[: stats_span_[1]].replace("sorry", "") + proof_str[span[1] :]


def refine_by_sorry(text):
    # Define the regular expression pattern
    target_pattern = r":=\s*(?:by\s*)?(?:sorry\s*)?"
    replacement = ":= by sorry"  # The new text we want to insert
    # We construct the pattern with two capturing groups
    # (group 1: the part from 'theorem' to just before our target)
    # (group 2: the target pattern itself)
    combined_pattern = r"(theorem.*?)(" + target_pattern + r")"
    # Find the first match
    match = re.search(combined_pattern, text, re.DOTALL)
    if match:
        # The part of the string BEFORE the target we want to replace
        # We use match.start(2) which is the start of the second group (our target)
        prefix = text[: match.start(2)]
        # Concatenate the prefix with the replacement to get the final, truncated string
        final_text = prefix + replacement
    else:
        final_text = text
    return final_text


def extract_code(inputs):
    import_head = (
        "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"
    )
    pattern = r"```lean4\n(.*?)\n```"
    matches = re.findall(pattern, inputs, re.DOTALL)
    if matches:
        return import_head + matches[-1]
    pattern = r"```lean4\n(.*?)```"
    matches = re.findall(pattern, inputs, re.DOTALL)
    if matches:
        return import_head + matches[-1]
    pattern = r"```lean\n(.*?)```"
    matches = re.findall(pattern, inputs, re.DOTALL)
    if matches:
        return import_head + matches[-1]
    return "None"


def parse_error(log_string):
    # Pattern to match multiline warnings
    # warning_pattern = re.compile(
    #    r"(/lean4/my_project/.*?:\d+:\d+: warning:.*?)(?=\n/lean4/my_project|\Z)",
    #    re.DOTALL,
    # )
    # Pattern to match multiline errors
    error_pattern = re.compile(
        r"(/lean4/my_project/.*?:\d+:\d+: error:.*?)(?=\n/lean4/my_project|\Z)",
        re.DOTALL,
    )
    # Find all warnings and errors
    # warnings = warning_pattern.findall(log_string)
    errors = error_pattern.findall(log_string)
    pattern = re.compile(r":(\d+):(\d+):")
    error_list = []
    for error in errors:
        match = pattern.search(error)
        error_list.append(
            {
                "pos": {"line": int(match.group(1)), "column": int(match.group(2))},
                "endPos": None,
                "data": error.split("error:")[1],
            }
        )

    return error_list


def get_error_str(code, errors, error_thres=True):
    err_str = ""
    code_lines = code.split("\n")
    # token_lengths = [len(line) + 1 for line in code_lines]
    error_num_thres = 8 if error_thres else len(errors)

    for i, error in enumerate(errors[:error_num_thres]):
        start_line = error["pos"]["line"] - 1
        start_col = error["pos"]["column"]
        if start_line >= len(code_lines):
            LOG.warning(
                "Error line %d exceeds code length %d. Errors: %s, Code: %s", start_line, len(code_lines), errors, code
            )
            continue
        if error["endPos"] is None:
            end_line = start_line
            end_col = len(code_lines[start_line])
        else:
            end_line = error["endPos"]["line"] - 1
            end_col = error["endPos"]["column"]

        err_str += f"\nError {i + 1}:\n"
        err_str += "\nCorresponding Code:\n```lean4\n"
        error_code = ""
        for ii in range(-4, 0):
            if start_line + ii >= 0:
                error_code += f"{code_lines[start_line + ii]}\n"
        if start_line != end_line:
            error_code += code_lines[start_line][:start_col] + "<error>" + code_lines[start_line][start_col:] + "\n"
            if not error_thres:
                for j in range(start_line + 1, end_line):
                    error_code += f"{code_lines[j]}\n"
            else:
                show_line = 6
                for j in range(start_line + 1, min(end_line, start_line + show_line)):
                    error_code += f"{code_lines[j]}\n"
                if end_line > start_line + show_line:
                    leading_spaces = len(code_lines[j]) - len(code_lines[j].lstrip(" "))
                    error_code += "\n" + " " * leading_spaces + "... --[Truncated]-- ...\n"
            error_code += code_lines[end_line][:end_col] + "</error>" + code_lines[end_line][end_col:] + "\n"
        else:
            error_code += (
                code_lines[start_line][:start_col]
                + "<error>"
                + code_lines[start_line][start_col:end_col]
                + "</error>"
                + code_lines[start_line][end_col:]
                + "\n"
            )
        if end_line + 1 < len(code_lines):
            error_code += f"{code_lines[end_line + 1]}\n"
        err_str += error_code
        err_str += "\n```\n"
        err_str += f"\nError Message: {error['data']}\n"
    if len(errors) > error_num_thres:
        err_str += f"\n... [Omitted {len(errors) - error_num_thres} more errors] ...\n"
    return err_str

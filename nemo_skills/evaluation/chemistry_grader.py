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

"""Deterministic comparison helpers for chemistry answers (SMILES / SAFE / numeric).

Answer *extraction* is intentionally not handled here: callers extract the final
answer with :func:`nemo_skills.evaluation.math_grader.extract_answer` (the same
boxed/regex extractor used for math), then pass it to :func:`chemistry_equal`.
This module normalizes LaTeX/markdown wrappers and compares the expected and
predicted answers using RDKit canonicalization for molecular strings and a
tolerant numeric comparison otherwise. RDKit is imported lazily so the numeric/
string normalization stays importable in environments without RDKit.
"""

import logging
import math
import re

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


# LaTeX command wrappers that may surround a chemistry answer, e.g. \text{CCO}.
# The content inside the braces is kept; the wrapper is removed.
_LATEX_WRAPPERS = (
    "boxed",
    "fbox",
    "text",
    "mathrm",
    "mathbf",
    "mathsf",
    "mathtt",
    "textbf",
    "textrm",
    "mbox",
    "operatorname",
    "ce",  # mhchem
)
_LATEX_WRAPPER_RE = re.compile(r"\\(?:" + "|".join(_LATEX_WRAPPERS) + r")\s*\{([^{}]*)\}")
_SPACING_MACRO_RE = re.compile(r"\\[,;!:\s]")


def _strip_chem_formatting(value) -> str | None:
    """Remove LaTeX/markdown wrappers commonly emitted around a chemistry answer.

    Note: backslashes and forward slashes are preserved because they are valid
    SMILES bond/stereo characters (e.g. ``F/C=C/F``). Only known LaTeX command
    wrappers and math/markdown delimiters are stripped.
    """
    if value is None:
        return None

    text = str(value).strip()

    # Strip surrounding math/markdown delimiters (possibly a few layers deep).
    for _ in range(3):
        stripped = text.strip()
        if len(stripped) >= 2 and stripped[0] == "$" and stripped[-1] == "$":
            stripped = stripped[1:-1].strip()
        if stripped.startswith("\\(") and stripped.endswith("\\)"):
            stripped = stripped[2:-2].strip()
        if stripped.startswith("\\[") and stripped.endswith("\\]"):
            stripped = stripped[2:-2].strip()
        if len(stripped) >= 2 and stripped[0] == "`" and stripped[-1] == "`":
            stripped = stripped.strip("`").strip()
        if stripped == text:
            break
        text = stripped

    # Unwrap LaTeX command wrappers (iterate to handle nesting like \text{\mathrm{..}}).
    for _ in range(5):
        unwrapped = _LATEX_WRAPPER_RE.sub(r"\1", text)
        if unwrapped == text:
            break
        text = unwrapped

    text = _SPACING_MACRO_RE.sub("", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.strip().strip('"').strip("'").strip()
    if len(text) >= 2 and text[0] == "{" and text[-1] == "}":
        text = text[1:-1].strip()

    return text


def _to_number(value) -> float | None:
    """Parse a scalar number from a string, tolerating a trailing percent sign."""
    if value is None:
        return None
    text = str(value).strip().rstrip(".").strip()
    is_percent = False
    if text.endswith("%"):
        text = text[:-1].strip()
        is_percent = True
    try:
        number = float(text)
    except (ValueError, TypeError):
        return None
    return number / 100.0 if is_percent else number


def _get_rdkit_chem():
    """Import RDKit lazily and silence its native logger. Returns the Chem module or None."""
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        return Chem
    except ImportError:
        LOG.warning("RDKit is not installed; molecular comparison will fall back to string equality.")
        return None


def canonical_smiles(smiles, isomeric: bool = True) -> str | None:
    """Return RDKit canonical SMILES for a (possibly multi-component / SAFE) string.

    The whole string is parsed at once. RDKit's canonical form is independent of
    component ordering and atom numbering, and reconstructs SAFE fragment strings
    correctly, so no manual splitting or external ``safe`` dependency is needed.
    Returns None if RDKit is unavailable or the string does not parse.
    """
    if not smiles:
        return None
    chem = _get_rdkit_chem()
    if chem is None:
        return None
    candidate = str(smiles).replace(" ", "")
    try:
        mol = chem.MolFromSmiles(candidate)
    except Exception:  # RDKit can raise on malformed input
        return None
    if mol is None:
        return None
    try:
        return chem.MolToSmiles(mol, isomericSmiles=isomeric)
    except Exception:
        return None


def inchikey(smiles) -> str | None:
    """Return the InChIKey for a SMILES string (secondary equivalence check), or None."""
    if not smiles:
        return None
    chem = _get_rdkit_chem()
    if chem is None:
        return None
    candidate = str(smiles).replace(" ", "")
    try:
        mol = chem.MolFromSmiles(candidate)
        if mol is None:
            return None
        return chem.MolToInchiKey(mol)
    except Exception:
        return None


def chemistry_equal(
    expected_answer,
    predicted_answer,
    metadata: dict | None = None,
    ignore_stereo: bool = False,
    numeric_rel_tol: float = 1e-6,
    numeric_abs_tol: float = 1e-9,
) -> bool:
    """Deterministically decide whether a predicted chemistry answer matches the ground truth.

    Comparison strategy (first match wins):
      1. Exact normalized string match.
      2. Numeric comparison when both sides parse as numbers (handles trailing ``%``).
      3. RDKit canonical SMILES match (handles SMILES, SAFE, multi-component, atom
         ordering), with an InChIKey secondary check and an optional
         connectivity-only (stereo-insensitive) fallback.
      4. Case-insensitive whitespace-stripped string equality as a last resort.
    """
    if predicted_answer is None:
        return False

    expected = _strip_chem_formatting(expected_answer)
    predicted = _strip_chem_formatting(predicted_answer)
    if not expected or not predicted:
        return False

    if expected == predicted:
        return True

    expected_num = _to_number(expected)
    predicted_num = _to_number(predicted)
    if expected_num is not None and predicted_num is not None:
        return math.isclose(expected_num, predicted_num, rel_tol=numeric_rel_tol, abs_tol=numeric_abs_tol)
    if expected_num is not None or predicted_num is not None:
        # One side is numeric and the other is not; exact match already failed.
        return False

    canonical_expected = canonical_smiles(expected, isomeric=not ignore_stereo)
    canonical_predicted = canonical_smiles(predicted, isomeric=not ignore_stereo)
    if canonical_expected is not None and canonical_predicted is not None:
        if canonical_expected == canonical_predicted:
            return True
        expected_key = inchikey(expected)
        predicted_key = inchikey(predicted)
        if expected_key and predicted_key and expected_key == predicted_key:
            return True
        if ignore_stereo:
            flat_expected = canonical_smiles(expected, isomeric=False)
            flat_predicted = canonical_smiles(predicted, isomeric=False)
            if flat_expected and flat_predicted and flat_expected == flat_predicted:
                return True
        return False

    # Neither numeric nor parseable as molecules: tolerant string comparison.
    return expected.replace(" ", "").lower() == predicted.replace(" ", "").lower()

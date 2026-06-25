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

import pytest

from nemo_skills.evaluation.chemistry_grader import _strip_chem_formatting, chemistry_equal

# Molecular comparison needs RDKit; numeric/string normalization tests do not.
# Extraction is delegated to nemo_skills.evaluation.math_grader.extract_answer
# (shared with math) and is covered by the math grader tests.
rdkit = pytest.importorskip("rdkit", reason="RDKit is required for molecular comparison tests")


@pytest.mark.parametrize(
    "value, expected",
    [
        ("$CCO$", "CCO"),
        ("\\text{CCO}", "CCO"),
        ("\\mathrm{CC(=O)O}", "CC(=O)O"),
        ("`CCO`", "CCO"),
        ('"CCO"', "CCO"),
        ("\\ce{H2O}", "H2O"),
    ],
)
def test_strip_chem_formatting(value, expected):
    assert _strip_chem_formatting(value) == expected


@pytest.mark.parametrize(
    "expected_answer, predicted_answer",
    [
        ("3", "3"),
        ("3", "3.0"),
        ("3.0", "3"),
        ("10%", "0.1"),  # percent normalization
        ("5", "5 "),
        ("1.5", "\\boxed{1.5}"),
    ],
)
def test_chemistry_equal_numeric_correct(expected_answer, predicted_answer):
    assert chemistry_equal(expected_answer, predicted_answer) is True


@pytest.mark.parametrize(
    "expected_answer, predicted_answer",
    [
        ("3", "4"),
        ("3", "two"),
        ("3.0", "3.5"),
        ("5", None),
        ("5", ""),
    ],
)
def test_chemistry_equal_numeric_incorrect(expected_answer, predicted_answer):
    assert chemistry_equal(expected_answer, predicted_answer) is False


@pytest.mark.parametrize(
    "expected_answer, predicted_answer",
    [
        ("CCO", "OCC"),  # atom-order invariance (ethanol)
        ("c1ccccc1", "C1=CC=CC=C1"),  # aromatic vs Kekule (benzene)
        ("CCO.O", "O.CCO"),  # multi-component order invariance
        ("CC(=O)O", "\\boxed{\\text{OC(C)=O}}"),  # LaTeX-wrapped acetic acid
        ("CCO", " CCO "),  # surrounding whitespace
    ],
)
def test_chemistry_equal_smiles_correct(expected_answer, predicted_answer):
    assert chemistry_equal(expected_answer, predicted_answer) is True


@pytest.mark.parametrize(
    "expected_answer, predicted_answer",
    [
        ("CCO", "CCC"),  # ethanol vs propane
        ("CCO", "ethanol"),  # prose is not a valid molecule -> rejected (was an LLM-judge false positive)
        ("CC(=O)O", "this is acetic acid"),
        ("c1ccccc1", "not a molecule"),
    ],
)
def test_chemistry_equal_smiles_incorrect(expected_answer, predicted_answer):
    assert chemistry_equal(expected_answer, predicted_answer) is False


def test_chemistry_equal_stereo_sensitive_by_default():
    # L- vs D-alanine differ only in stereochemistry.
    l_alanine = "C[C@@H](N)C(=O)O"
    d_alanine = "C[C@H](N)C(=O)O"
    assert chemistry_equal(l_alanine, d_alanine) is False
    assert chemistry_equal(l_alanine, d_alanine, ignore_stereo=True) is True


def test_chemistry_equal_ez_stereo():
    trans = "F/C=C/F"
    cis = "F/C=C\\F"
    assert chemistry_equal(trans, cis) is False
    assert chemistry_equal(trans, cis, ignore_stereo=True) is True


def test_chemistry_equal_none_predicted():
    assert chemistry_equal("CCO", None) is False

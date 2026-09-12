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

import copy

import pytest

pytest.importorskip("bfcl_eval")

from nemo_skills.dataset.bfcl_v3.utils import func_doc_language_specific_pre_processing  # noqa: E402

JAVA_FUNCTION = [
    {
        "name": "MathUtils.sumList",
        "description": "Sum a list of integers.",
        "parameters": {
            "type": "dict",
            "properties": {
                "values": {"type": "ArrayList", "items": {"type": "Integer"}, "description": "Values to add."},
                "scale": {"type": "int", "description": "Scale factor."},
            },
            "required": ["values"],
        },
    }
]

JS_FUNCTION = [
    {
        "name": "sumList",
        "description": "Sum a list of numbers.",
        "parameters": {
            "type": "dict",
            "properties": {
                "values": {"type": "array", "items": {"type": "number"}, "description": "Values to add."},
                "options": {
                    "type": "dict",
                    "properties": {"round": {"type": "Boolean"}},
                    "description": "Options.",
                },
            },
            "required": ["values"],
        },
    }
]


@pytest.mark.parametrize("test_category", ["java", "simple_java"])
def test_java_categories_are_stringified(test_category):
    processed = func_doc_language_specific_pre_processing(copy.deepcopy(JAVA_FUNCTION), test_category)
    props = processed[0]["parameters"]["properties"]
    assert processed[0]["description"].endswith("Note that the provided function is in Java 8 SDK syntax.")
    assert props["values"]["type"] == "string"
    assert "items" not in props["values"]
    assert "Integer" in props["values"]["description"]
    assert props["scale"]["type"] == "string"


@pytest.mark.parametrize("test_category", ["javascript", "simple_javascript"])
def test_javascript_categories_are_stringified(test_category):
    processed = func_doc_language_specific_pre_processing(copy.deepcopy(JS_FUNCTION), test_category)
    props = processed[0]["parameters"]["properties"]
    assert processed[0]["description"].endswith("Note that the provided function is in JavaScript syntax.")
    assert props["values"]["type"] == "string"
    assert "items" not in props["values"]
    assert props["options"]["type"] == "string"
    assert "properties" not in props["options"]


@pytest.mark.parametrize("test_category", ["simple_python", "simple", "irrelevance", "multi_turn_base"])
def test_other_categories_are_left_unchanged(test_category):
    processed = func_doc_language_specific_pre_processing(copy.deepcopy(JAVA_FUNCTION), test_category)
    props = processed[0]["parameters"]["properties"]
    assert processed[0]["description"].endswith("Note that the provided function is in Python 3 syntax.")
    assert props["values"]["type"] == "ArrayList"
    assert props["values"]["items"] == {"type": "Integer"}

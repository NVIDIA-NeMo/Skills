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

from nemo_skills.dataset.utils import coerce_numeric_answer


def test_integer_past_float_mantissa_stays_exact():
    assert coerce_numeric_answer("9007199254740993") == 9007199254740993
    assert coerce_numeric_answer("-9007199254740993") == -9007199254740993
    assert coerce_numeric_answer("9007199254740993.0") == 9007199254740993


def test_ordinary_numeric_answers_are_unchanged():
    assert coerce_numeric_answer("42") == 42
    assert coerce_numeric_answer("-3") == -3
    assert coerce_numeric_answer("1.0") == 1
    assert coerce_numeric_answer("1.5") == 1.5
    assert coerce_numeric_answer("1.50") == 1.5
    assert coerce_numeric_answer("insufficient") == "insufficient"

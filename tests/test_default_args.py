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

import pytest

from nemo_skills.pipeline.cli import generate, wrap_arguments


def test_error_on_missing_default():
    with pytest.raises(TypeError):
        generate(ctx=wrap_arguments(""))


def test_wrap_arguments_empty_string():
    assert wrap_arguments("").args == []


def test_wrap_arguments_drops_extra_spaces():
    assert wrap_arguments("++a=1   ++b=2 ").args == ["++a=1", "++b=2"]

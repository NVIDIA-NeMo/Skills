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

import pytest

from nemo_skills.evaluation.utils import get_eval_group, load_config


def test_load_config_raises_on_empty_file(tmp_path):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("")

    with pytest.raises(ValueError, match="empty"):
        load_config(str(config_path))


def test_load_config_raises_on_comments_only_file(tmp_path):
    config_path = tmp_path / "comments_only.yaml"
    config_path.write_text("# just a comment\n# another one\n")

    with pytest.raises(ValueError, match="empty"):
        load_config(str(config_path))


def test_load_config_returns_dict_for_valid_file(tmp_path):
    config_path = tmp_path / "valid.yaml"
    config_path.write_text("key: value\n")

    assert load_config(str(config_path)) == {"key": "value"}


def test_get_eval_group_raises_on_empty_file(tmp_path):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("")

    with pytest.raises(ValueError, match="empty"):
        get_eval_group(str(config_path))


def test_get_eval_group_passes_through_dict_unchanged():
    config = {"already": "a dict"}
    assert get_eval_group(config) is config

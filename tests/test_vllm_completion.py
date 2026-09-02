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

"""Tests for VLLMModel text-completion request building."""

from unittest.mock import patch

from nemo_skills.inference.model.vllm import VLLMModel


def _completion_request(**extra_body):
    """Build a text-completion request dict without a live server."""
    with patch.object(VLLMModel, "__init__", lambda self, **kwargs: None):
        model = VLLMModel()
    model._tunnel = None  # avoid __del__ touching an unset attribute at GC time
    return model._build_completion_request_params(
        prompt="hello",
        tokens_to_generate=8,
        extra_body=extra_body or None,
    )


def test_completion_skip_special_tokens_defaults_to_false():
    """With no override, skip_special_tokens stays False and is absent from extra_body."""
    request = _completion_request()
    assert request["skip_special_tokens"] is False
    assert "skip_special_tokens" not in request["extra_body"]


def test_completion_skip_special_tokens_honored_from_extra_body():
    """A value passed via extra_body is lifted to the single top-level field, not duplicated."""
    request = _completion_request(skip_special_tokens=True)
    assert request["skip_special_tokens"] is True
    assert "skip_special_tokens" not in request["extra_body"]

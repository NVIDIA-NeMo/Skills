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

from nemo_skills.training.nemo_rl.convert_dcp_to_hf import copy_tokenizer_files, resolve_tokenizer_path


def test_resolve_tokenizer_path_prefers_runtime_tokenizer(tmp_path):
    step_dir = tmp_path / "checkpoints" / "step_10"
    runtime_tokenizer = tmp_path / "checkpoints" / "tokenizer"
    step_dir.mkdir(parents=True)
    runtime_tokenizer.mkdir()
    config_path = step_dir / "config.yaml"
    config_path.touch()
    config = {
        "policy": {
            "model_name": "model-tokenizer",
            "tokenizer": {"name": "configured-tokenizer"},
        }
    }

    assert resolve_tokenizer_path(config, config_path) == str(runtime_tokenizer)


def test_resolve_tokenizer_path_falls_back_to_configured_tokenizer(tmp_path):
    config_path = tmp_path / "step_10" / "config.yaml"
    config_path.parent.mkdir()
    config_path.touch()
    config = {
        "policy": {
            "model_name": "model-tokenizer",
            "tokenizer": {"name": "configured-tokenizer"},
        }
    }

    assert resolve_tokenizer_path(config, config_path) == "configured-tokenizer"


def test_copy_tokenizer_files_preserves_all_runtime_artifacts(tmp_path):
    tokenizer_path = tmp_path / "tokenizer"
    nested_path = tokenizer_path / "custom"
    hf_ckpt_path = tmp_path / "final_hf_model"
    nested_path.mkdir(parents=True)
    hf_ckpt_path.mkdir()
    (tokenizer_path / "chat_template.jinja").write_text("runtime template")
    (nested_path / "tokenizer.model").write_text("tokenizer data")

    copy_tokenizer_files(tokenizer_path, hf_ckpt_path)

    assert (hf_ckpt_path / "chat_template.jinja").read_text() == "runtime template"
    assert (hf_ckpt_path / "custom" / "tokenizer.model").read_text() == "tokenizer data"

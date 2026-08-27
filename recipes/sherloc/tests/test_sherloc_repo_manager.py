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

from recipes.sherloc.inference.sherloc_utils.repo_manager import RepoManager


def _file(*lines):
    return {"classes": [], "functions": [], "text": list(lines)}


def test_connected_tree_resolves_relative_imports_to_leaf_modules():
    repo = {
        "structure": {
            "pkg": {
                "__init__.py": _file(),
                "app.py": _file("from .models import Model"),
                "models.py": _file("class Model:", "    pass"),
            }
        }
    }

    app_tree = RepoManager.connected_tree_repo_dict(repo, target_file="pkg/app.py")
    models_tree = RepoManager.connected_tree_repo_dict(repo, target_file="pkg/models.py")

    assert "pkg/models.py" in app_tree
    assert "pkg/app.py" in models_tree
    assert "ERROR:" not in models_tree

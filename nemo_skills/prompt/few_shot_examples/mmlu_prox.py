# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
Few-shot examples for MMLU-ProX: loaded dynamically from the validation split
of li-lab/MMLU-ProX per language (first n examples). Keys: mmlu_prox_few_shot_{language}.
"""

import importlib.util
import types
from pathlib import Path
from typing import Any, Iterator, List

# Number of validation examples to use per language (change here if needed)
DEFAULT_N = 5

_loader_module = None


def _get_loader() -> types.ModuleType:
    global _loader_module
    if _loader_module is None:
        loader_path = Path(__file__).resolve().parent.parent.parent / "dataset" / "mmlu-prox" / "few_shot_loader.py"
        spec = importlib.util.spec_from_file_location("_mmlu_prox_few_shot_loader", loader_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load MMLU-ProX few-shot loader from {loader_path}")
        _loader_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_loader_module)
    return _loader_module


class _MMLUProXExamplesMap:
    """
    Mapping that resolves mmlu_prox_few_shot_{language} by loading the first n
    validation examples from li-lab/MMLU-ProX. No static keys; used via ChainMap in __init__.py.
    """

    def __getitem__(self, key: str) -> List[dict]:
        if not key.startswith("mmlu_prox_few_shot_"):
            raise KeyError(key)
        language = key.replace("mmlu_prox_few_shot_", "", 1)
        return _get_loader().load_mmlu_prox_validation_few_shot(language, n=DEFAULT_N)

    def __contains__(self, key: Any) -> bool:
        return isinstance(key, str) and key.startswith("mmlu_prox_few_shot_")

    def __iter__(self) -> Iterator:
        return iter(())

    def keys(self) -> Iterator:
        return iter(())

    def items(self) -> Iterator:
        return iter(())

    def __len__(self) -> int:
        return 0


examples_map = _MMLUProXExamplesMap()

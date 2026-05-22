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

"""
Registry of few-shot example sets, exposed as a single `examples_map` dict-like object.

Most example sets are **static**: they are plain dicts with hardcoded keys (e.g. "gsm8k",
"math_algebra") whose values are lists of {problem, solution} dicts. These are merged into
`_static_examples_map` at import time.

Some example sets are **dynamic**: their keys are only known at inference time (e.g.
mmlu_prox_few_shot_{language}, where `language` comes from each dataset row). A dynamic map
cannot be enumerated upfront — instead it implements __getitem__ and __contains__ to resolve
keys on demand, typically by downloading data from an external source (HuggingFace) and
caching the result. Dynamic maps have no static keys, so they are excluded from
`_static_example_sets` and its deduplication check.

`examples_map` is a ChainMap that places dynamic maps first. At lookup time, a key is
checked against each dynamic map before falling back to the merged static map.
"""

from collections import ChainMap

from nemo_skills.prompt.few_shot_examples.gsm8k import examples_map as examples_gsm8k
from nemo_skills.prompt.few_shot_examples.lean4 import examples_map as examples_lean4
from nemo_skills.prompt.few_shot_examples.math import examples_map as examples_math
from nemo_skills.prompt.few_shot_examples.mmlu import examples_map as examples_mmlu
from nemo_skills.prompt.few_shot_examples.mmlu_pro import examples_map as examples_mmlu_pro
from nemo_skills.prompt.few_shot_examples.mmlu_prox import examples_map as examples_mmlu_prox
from nemo_skills.prompt.few_shot_examples.open_science import examples_map as examples_open_science

# Static example sets only. Dynamic maps (e.g. mmlu_prox) are excluded here
# because they have no enumerable keys — they resolve keys on demand at lookup time.
_static_example_sets = [
    examples_gsm8k,
    examples_math,
    examples_lean4,
    examples_mmlu_pro,
    examples_mmlu,
    examples_open_science,
]

_static_examples_map = {k: v for d in _static_example_sets for k, v in d.items()}

# Verify no duplicate keys exist across static example sets.
expected_total_examples = sum(len(s) for s in _static_example_sets)
assert len(_static_examples_map) == expected_total_examples, "Duplicate keys in static examples!"

# Dynamic maps are consulted first so they take priority over any static keys.
# mmlu_prox resolves mmlu_prox_few_shot_{language} keys on demand from the HuggingFace validation split.
examples_map = ChainMap(examples_mmlu_prox, _static_examples_map)

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

import json


def load_json(path):
    """Load a JSON file from the given path."""
    with open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def get_nested_value(nested_dict, nested_keys):
    for k in nested_keys:
        if not isinstance(nested_dict, dict) or k not in nested_dict:
            return None
        nested_dict = nested_dict[k]
    # resolves to the value eventually
    return nested_dict


_soft_assert_failures = []


def soft_assert(condition: bool, message: str):
    """Record a failure instead of raising immediately.

    Use this in place of `assert` when you want to collect all failures
    and report them at the end of the script.
    """
    if not condition:
        _soft_assert_failures.append(str(message))


def assert_all():
    """If any soft assertions failed, print them and exit with non-zero status.

    Does nothing if there are no failures. Intended to be called once at the end
    of a check script, before printing a success message.
    """
    if not _soft_assert_failures:
        print("ALL TESTS PASSED")
        return
    print(f"\nTEST FAILURES ({len(_soft_assert_failures)})\n")
    for i, msg in enumerate(_soft_assert_failures, 1):
        print(f"{i:3d}. {msg}")
    raise SystemExit(1)

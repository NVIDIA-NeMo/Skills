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

"""Model-related utility functions"""


def shorten_model_name(model_name):
    """Shorten long model names for better legend readability"""
    if len(model_name) <= 20:
        return model_name

    # Common shortening patterns
    shortcuts = {
        "Instruct": "Inst",
        "Instruction": "Inst",
        "Chat": "Chat",
        "deepseek": "DS",
        "Mixtral": "Mixtral",
        "llama": "Llama",
        "nemotron": "Nemotron",
        "ultra": "Ultra",
    }

    # Apply shortcuts
    short_name = model_name
    for long_form, short_form in shortcuts.items():
        short_name = short_name.replace(long_form, short_form)

    # If still too long, take first part + version
    if len(short_name) > 25:
        parts = short_name.split("-")
        if len(parts) >= 2:
            # Take first part and any version numbers
            version_parts = [p for p in parts if any(c.isdigit() for c in p)]
            main_part = parts[0]
            if version_parts:
                short_name = f"{main_part}-{version_parts[0]}"
            else:
                short_name = main_part

    # Final fallback: truncate and add ellipsis
    if len(short_name) > 25:
        short_name = short_name[:22] + "..."

    return short_name

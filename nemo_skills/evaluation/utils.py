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
from pathlib import Path

import yaml


# TODO: DRY with prompt's load_config function
def load_config(config: str, config_dir: str | None = None) -> dict:
    """
    Reads the eval group configuration from the yaml file.

    Args:
        config (str): The location of the eval group config file.
            Can be the full path to a yaml file (if ends with .yaml) or one of the available configs.
            If configs starts with nemo_skills we will look relative to the repo root.
            If not, we will look relative to the config_dir parameter
        config_dir (str): The dir to look for the config file.

    Returns:
        The loaded dictionary.
    """
    if config_dir is None:
        config_dir = str(Path(__file__).parent.absolute() / "eval_group")

    if config.endswith(".yaml"):
        config_path = Path(config).absolute()
    elif config.startswith("nemo_skills"):
        config_path = Path(__file__).parents[2].absolute() / f"{config}.yaml"
    else:
        config_path = Path(config_dir) / f"{config}.yaml"

    with open(config_path, "rt", encoding="utf-8") as fin:
        config_data = yaml.safe_load(fin)
    if config_data is None:
        raise ValueError(f"Config file {config_path} is empty or contains only comments.")
    return config_data


def get_eval_group(eval_config: str | dict, eval_group_dir: str | None = None) -> dict:
    if isinstance(eval_config, dict):
        return eval_config

    return load_config(eval_config, eval_group_dir)

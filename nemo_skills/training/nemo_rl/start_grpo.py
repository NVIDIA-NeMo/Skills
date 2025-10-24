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

# copied and edited from https://github.com/NVIDIA/NeMo-RL/blob/ab1b638a499308caea022648daaf6994d390cbde/examples/run_grpo_math.py

import argparse
import copy
import importlib
import json
import os
import pprint
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

from datasets import Dataset, load_dataset
from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec, TaskDataProcessFnCallable, TaskDataSpec
from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY, get_actor_python_env
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import setup_make_sequence_length_divisible_by


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


# ===============================================================================
#                             Custom Math Dataset (@nemo-skills)
# ===============================================================================


def load_jsonl_as_dataset(
    filepath: str,
    force_string: bool = False,
    keep_fields: Optional[list[str]] = None,
) -> Dataset:
    """
    Load a JSONL file and convert it to a Hugging Face Dataset.

    Args:
        filepath (str): Path to the .jsonl file.

    Returns:
        Dataset: Hugging Face Dataset object.
    """
    records: list[dict[str, Any]] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(obj)

    return Dataset.from_list(records)


def extract_dataset(split, dataset_path):
    if dataset_path is None:
        return None
    if not dataset_path.startswith("/"):
        original_ds = load_dataset(dataset_path, split=split)
    else:
        original_ds = load_jsonl_as_dataset(dataset_path)
    return original_ds


def format_passthrough(data):
    return {
        **data,
        # For v0.1 release, nemo rl datasets require a task_name key such that user can map a task processor per unique task.
        "task_name": "math",
    }


def prepare_math_dataset(split_ds):
    # Format the examples, removing original columns
    train_formatted = split_ds["train"].map(format_passthrough)
    val_raw = split_ds.get("validation", None)
    val_formatted = None if val_raw is None else val_raw.map(format_passthrough)

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class NeMoSkillsDataset:
    """Custom dataset class for NeMo Skills Math Environment."""

    def __init__(self, training_data, validation_data):
        """Initialize the dataset with training and validation data."""
        self.training_data = training_data
        self.validation_data = validation_data

        # Load the datasets
        self.formatted_ds = prepare_math_dataset(
            {
                "train": extract_dataset("train", training_data),
                "validation": extract_dataset("validation", validation_data),
            }
        )


# ===============================================================================
#                             Math Data Processor
# ===============================================================================
TokenizerType = PreTrainedTokenizerBase


@dataclass
class NSTaskDataSpec(TaskDataSpec):
    prompt_spec: dict[str, Any] | None = None


# TaskDataProcessFnCallable
def ns_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: NSTaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    prompt_spec = task_data_spec.prompt_spec
    extra_env_info = copy.deepcopy(datum_dict)

    prompt = get_prompt(
        prompt_config=prompt_spec["prompt_config"],
        tokenizer=tokenizer,
        examples_type=prompt_spec["examples_type"],
        config_dir=prompt_spec["config_dir"],
    )
    # we need to include system message here as roles are only used for masking
    # so prompt.fill can return a combined system + user message
    # if we use separate, it will have double BOS in the tokens!
    user_message = prompt.fill(datum_dict, format_as_string=True)
    message_log = [
        {
            "role": "user",
            "content": user_message,
            "token_ids": tokenizer([user_message], return_tensors="pt", add_special_tokens=False)["input_ids"][0],
        }
    ]

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # make smaller and mask out
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][: min(4, max_seq_length // len(message_log))]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict["task_name"],
    }
    return output


def setup_data(
    tokenizer: TokenizerType,
    data_config: DataConfig,
    env_configs: dict[str, Any],
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset],
    dict[str, EnvironmentInterface],
    dict[str, EnvironmentInterface],
]:
    print("\n▶ Setting up data...")
    prompt_config = data_config["prompt"]
    math_task_spec = NSTaskDataSpec(
        task_name="math",
        prompt_spec=prompt_config,
    )

    data = NeMoSkillsDataset(
        data_config["train_data_path"],
        data_config["val_data_path"],
    )

    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = defaultdict(
        lambda: (math_task_spec, ns_data_processor)
    )
    task_data_processors["math"] = (math_task_spec, ns_data_processor)

    # Allow overriding the environment class via the Hydra/YAML config.
    # If `env_cls` is provided inside env_configs["math"], we dynamically
    # import and instantiate that environment instead of the default
    # `MathEnvironment`.  This lets users plug in custom reward functions
    # without modifying the rest of the code.

    env_cls_path = env_configs["math"].get(
        "env_cls",
        "nemo_skills.training.nemo_rl.environments.math_environment.MathEnvironment",
    )
    ACTOR_ENVIRONMENT_REGISTRY[env_cls_path] = PY_EXECUTABLES.SYSTEM

    module_name, class_name = env_cls_path.rsplit(".", 1)
    env_module = importlib.import_module(module_name)
    env_cls = getattr(env_module, class_name)

    math_env = env_cls.options(  # type: ignore  # ray.remote wrapper
        runtime_env={
            "py_executable": get_actor_python_env(env_cls_path),
            "env_vars": dict(os.environ),  # Pass through all env vars
        }
    ).remote(env_configs["math"])
    dataset = AllTaskProcessedDataset(
        data.formatted_ds["train"],
        tokenizer,
        math_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset: Optional[AllTaskProcessedDataset] = None
    if data.formatted_ds["validation"]:
        val_dataset = AllTaskProcessedDataset(
            data.formatted_ds["validation"],
            tokenizer,
            math_task_spec,
            task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )
    else:
        val_dataset = None

    task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: math_env)
    task_to_env["math"] = math_env
    return dataset, val_dataset, task_to_env, task_to_env


def main() -> None:
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "grpo.yaml")

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
    if config["policy"]["make_sequence_length_divisible_by"] is None:
        tp = config["policy"]["tensor_model_parallel_size"]
        cp = config["policy"]["context_parallel_size"]
        config["policy"]["make_sequence_length_divisible_by"] = setup_make_sequence_length_divisible_by(tp, cp)
    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}")

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, "A generation config is required for GRPO"
    config["policy"]["generation"] = configure_generation_config(config["policy"]["generation"], tokenizer)

    # setup data
    (
        dataset,
        val_dataset,
        task_to_env,
        val_task_to_env,
    ) = setup_data(tokenizer, config["data"], config["env"])

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()

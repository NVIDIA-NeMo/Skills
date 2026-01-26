# Copyright 2023 https://github.com/ShishirPatil/gorilla
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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

import copy
import importlib
import inspect
import json
import logging
import re
from pathlib import Path

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


# Copied from:
# - https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval/multi_turn_utils.py
# - https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl_eval/model_handler/utils.py


BACKEND_PATH_PREFIX = "bfcl_eval.eval_checker.multi_turn_eval.func_source_code"

CLASS_FILE_PATH_MAPPING = {
    "GorillaFileSystem": f"{BACKEND_PATH_PREFIX}.gorilla_file_system",
    "MathAPI": f"{BACKEND_PATH_PREFIX}.math_api",
    "MessageAPI": f"{BACKEND_PATH_PREFIX}.message_api",
    "TwitterAPI": f"{BACKEND_PATH_PREFIX}.posting_api",
    "TicketAPI": f"{BACKEND_PATH_PREFIX}.ticket_api",
    "TradingBot": f"{BACKEND_PATH_PREFIX}.trading_bot",
    "TravelAPI": f"{BACKEND_PATH_PREFIX}.travel_booking",
    "VehicleControlAPI": f"{BACKEND_PATH_PREFIX}.vehicle_control",
    # The following classes are not part of the multi-turn categories suite, but they share the same evaluation pipeline for simplicity
    # WebSearchAPI agent is altered to use raw DuckDuckGo instead of SerpAPI
    "WebSearchAPI": "nemo_skills.inference.eval.bfcl_web_search",
    "MemoryAPI_kv": f"{BACKEND_PATH_PREFIX}.memory_kv",
    "MemoryAPI_vector": f"{BACKEND_PATH_PREFIX}.memory_vector",
    "MemoryAPI_rec_sum": f"{BACKEND_PATH_PREFIX}.memory_rec_sum",
}

# These classes are stateless and do not require any initial configuration
STATELESS_CLASSES = [
    "MathAPI",
]

MAXIMUM_STEP_LIMIT = 20

DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC = (
    "I have updated some more functions you can choose from. What about now?"
)


def convert_to_function_call(function_call_list):
    if isinstance(function_call_list, dict):
        function_call_list = [function_call_list]
    # function_call_list is of type list[dict[str, str]] or list[dict[str, dict]]
    execution_list = []
    for function_call in function_call_list:
        for key, value in function_call.items():
            if isinstance(value, str):
                value = json.loads(value)
            execution_list.append(f"{key}({','.join([f'{k}={repr(v)}' for k, v in value.items()])})")

    return execution_list


def execute_multi_turn_func_call(
    func_call_list: list[str],  # a list of strings of func calls
    initial_config: dict,
    involved_classes: list,
    test_entry_id: str,
    long_context: bool = False,
) -> tuple[list[str], dict]:
    """
    Execute the function calls in the list.

    Args:
        func_call_list (list[str]): A list of strings of function calls.
        initial_config (dict): A dictionary of initial configurations for the classes.
        involved_classes (list): A list of class names that are involved in the function calls.
        test_entry_id (str): The id of the test entry.
        long_context (bool): Whether to use long context.
    """

    class_method_name_mapping = {}
    involved_instances = {}
    for class_name in involved_classes:
        module_name = CLASS_FILE_PATH_MAPPING[class_name]
        # TODO: Handler the model name issue from handler more elegantly
        instance_name = f"sample_model_{test_entry_id}_{class_name}_instance"
        instance_name = re.sub(r"[-./]", "_", instance_name)
        if instance_name not in globals():
            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
            class_instance = class_()
            if class_name not in STATELESS_CLASSES:
                class_initial_config = initial_config.get(class_name, {})
                class_initial_config_to_load = copy.deepcopy(class_initial_config)
                if "model_result_dir" in class_initial_config_to_load:
                    class_initial_config_to_load["model_result_dir"] = Path(
                        class_initial_config_to_load["model_result_dir"]
                    )
                # Deep copy the initial configuration to avoid mutation issues
                class_instance._load_scenario(class_initial_config_to_load, long_context=long_context)
            globals()[instance_name] = class_instance
        # This happens in subsequent turns
        else:
            class_instance = globals()[instance_name]

        involved_instances[class_name] = class_instance

        # Retrieve all method names and map them to the instance
        for method_name, method in inspect.getmembers(class_instance, predicate=inspect.ismethod):
            # Skip private methods
            if method_name.startswith("_"):
                continue
            class_method_name_mapping[method_name] = instance_name

    execution_results = []
    for func_call in func_call_list:
        # Add the instance name to the method calls
        func_call = _process_method_calls(func_call, class_method_name_mapping)

        # Evaluate the function call
        try:
            # We need to make a copy here because otherwise the `eval(func_call)` would error.
            func_call_copy = func_call
            # Before calling `eval`, we need to make sure that the function call is safe
            # We do so by checking if the function is `kill` or `exit`, etc.
            # Extract the function name first
            if "(" in func_call_copy:
                func_call_copy = func_call_copy.split("(")[0]
            # Situation where the function call is a method call
            if "." in func_call_copy:
                func_call_copy = func_call_copy.split(".")[1]
            if func_call_copy in ["kill", "exit", "quit", "remove", "unlink", "popen", "Popen", "run"]:
                raise Exception(f"Function call {func_call_copy} is not allowed.")

            func_call_result = eval(func_call)

            if isinstance(func_call_result, str):
                pass
            elif isinstance(func_call_result, dict):
                # Some function returns a object instance, which is not serializable
                try:
                    func_call_result = json.dumps(func_call_result)
                except Exception:
                    func_call_result = str(func_call_result)
            else:
                func_call_result = str(func_call_result)

            execution_results.append(func_call_result)
        except Exception as e:
            # Hard fail for missing/unsupported web-search backends.
            # BFCL would otherwise swallow the exception and keep going, which makes the failure silent.
            try:
                from nemo_skills.inference.eval.bfcl_web_search import WebSearchBackendUnavailable

                if isinstance(e, WebSearchBackendUnavailable):
                    raise
            except ImportError:
                # If bfcl_web_search isn't importable for some reason, fall back to existing behavior.
                pass
            execution_results.append(f"Error during execution: {str(e)}")

    return execution_results, involved_instances


def is_empty_execute_response(input_list: list):
    if len(input_list) == 0:
        return True
    if len(input_list) == 1 and len(input_list[0]) == 0:
        return True
    return False


def _process_method_calls(function_call_string: str, instance_mapping: dict) -> str:
    """
    Prepends the instance name to the function name for each of the function name represented in the string, you will
    also be provided with the mapping of method name to instance name.

    Example input:
    ```
    f(x = g((1, 2), h(3)), y = (4), z = (5, 6))
    ```

    Example return:
    ```
    a.f(x=a.g((1, 2), a.h(3)), y=(4), z=(5, 6))
    ```

    Args:
        function_call_string (str): The function call string to parse.
        class_mapping (dict): A dictionary mapping method names to instance names.

    Returns:
        str: The parsed function call string with instance names prepended to method names.
    """

    def replace_function(match):
        func_name = match.group(1)
        if func_name in instance_mapping:
            return f"{instance_mapping[func_name]}.{func_name}"
        return func_name

    # Regular expression to match function names
    pattern = r"\b([a-zA-Z_]\w*)\s*(?=\()"

    # Replace function names with their class-prepended versions
    processed_string = re.sub(pattern, replace_function, function_call_string)

    return processed_string

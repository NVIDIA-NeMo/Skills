#!/usr/bin/env python3
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

"""
CLI entrypoint for the Unified NeMo Inference Server.

Configuration is YAML-based: provide a config file with backend type and
all backend-specific parameters. The config is validated against the
backend's config class.

Usage:
    python -m nemo_skills.inference.server.serve_unified \\
        --config /path/to/backend_config.yaml \\
        --port 8000

    # Or with --model for nemo-skills pipeline compatibility:
    python -m nemo_skills.inference.server.serve_unified \\
        --model /path/to/model \\
        --backend magpie_tts \\
        --codec_model /path/to/codec \\
        --port 8000

Backend-specific options are passed as extra CLI flags and forwarded to the
backend's config dataclass automatically. For example:

    --server_args "--backend magpie_tts --codec_model /path --use_cfg --cfg_scale 2.5"

Any flag not recognized by the server itself is parsed generically:
    --flag           -> {"flag": True}
    --key value      -> {"key": <auto-typed value>}
    --key=value      -> {"key": <auto-typed value>}
    --no_flag        -> {"flag": False}

See each backend's config class for available options (e.g. MagpieTTSConfig).

Example YAML config (backend_config.yaml):
    backend: magpie_tts
    model_path: /path/to/model
    codec_model_path: /path/to/codec
    device: cuda
    dtype: bfloat16
    temperature: 0.6
    top_k: 80
    use_cfg: true
    cfg_scale: 2.5
"""

import argparse
import inspect
import os
import shutil
import sys
from typing import Optional


def setup_pythonpath(code_path: Optional[str] = None):
    """Set up PYTHONPATH for NeMo and the unified server.

    Args:
        code_path: Single path or colon-separated paths to add to PYTHONPATH
    """
    paths_to_add = []

    if code_path:
        for path in code_path.split(":"):
            if path and path not in paths_to_add:
                paths_to_add.append(path)

    # Add recipes path for unified server imports
    this_dir = os.path.dirname(os.path.abspath(__file__))
    ns_eval_root = os.path.dirname(os.path.dirname(os.path.dirname(this_dir)))
    if os.path.exists(os.path.join(ns_eval_root, "recipes")):
        paths_to_add.append(ns_eval_root)

    # Container pattern
    if os.path.exists("/nemo_run/code"):
        paths_to_add.append("/nemo_run/code")

    current_path = os.environ.get("PYTHONPATH", "")
    for path in paths_to_add:
        if path not in current_path.split(":"):
            current_path = f"{path}:{current_path}" if current_path else path

    os.environ["PYTHONPATH"] = current_path

    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)


def apply_safetensors_patch(hack_path: Optional[str]):
    """Apply safetensors patch if provided (for some NeMo models)."""
    if not hack_path or not os.path.exists(hack_path):
        return

    try:
        import safetensors.torch as st_torch

        dest_path = inspect.getfile(st_torch)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copyfile(hack_path, dest_path)
        print(f"[serve_unified] Applied safetensors patch: {hack_path} -> {dest_path}")
    except Exception as e:
        print(f"[serve_unified] Warning: Failed to apply safetensors patch: {e}")


def load_yaml_config(config_path: str) -> dict:
    """Load YAML config file."""
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


def _coerce_value(value: str):
    """Try to coerce a string value to int, float, or bool."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value


def parse_extra_args(extra_args: list) -> dict:
    """Convert unknown CLI args to a config dict.

    Handles these patterns:
        --flag           -> {"flag": True}
        --key value      -> {"key": <auto-typed value>}
        --key=value      -> {"key": <auto-typed value>}
        --no_flag        -> {"flag": False}  (strip no_ prefix)
    """
    result = {}
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if not arg.startswith("--"):
            i += 1
            continue

        # Handle --key=value
        if "=" in arg:
            key, value = arg[2:].split("=", 1)
            result[key] = _coerce_value(value)
            i += 1
            continue

        key = arg[2:]

        # Check if next token is a value (not another flag)
        if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
            result[key] = _coerce_value(extra_args[i + 1])
            i += 2
            continue

        # Bare flag: --no_X -> {X: False}, otherwise {key: True}
        if key.startswith("no_"):
            result[key[3:]] = False
        else:
            result[key] = True
        i += 1

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Unified NeMo Inference Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Primary: YAML config
    parser.add_argument("--config", default=None, help="Path to YAML config file")

    # Standard args for nemo-skills pipeline compatibility
    parser.add_argument("--model", default=None, help="Path to the model")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--backend", default="magpie_tts", help="Backend type")

    # Server configuration
    parser.add_argument("--batch_size", type=int, default=8, help="Maximum batch size")
    parser.add_argument("--batch_timeout", type=float, default=0.1, help="Batch timeout in seconds")

    # Generation defaults
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")

    # Model configuration
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")

    # Environment setup
    parser.add_argument("--code_path", default=None, help="Path to add to PYTHONPATH")
    parser.add_argument("--hack_path", default=None, help="Path to safetensors patch")

    # Debug
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Parse known args; everything else is backend-specific
    args, unknown = parser.parse_known_args()
    extra_config = parse_extra_args(unknown)

    # Setup environment
    setup_pythonpath(args.code_path)
    apply_safetensors_patch(args.hack_path)

    if args.code_path:
        os.environ["UNIFIED_SERVER_CODE_PATH"] = args.code_path

    if args.debug:
        os.environ["DEBUG"] = "1"

    # Set CUDA devices
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.num_gpus))

    # Build configuration
    if args.config:
        # YAML config mode
        config_dict = load_yaml_config(args.config)
        backend_type = config_dict.pop("backend", args.backend)
        # CLI overrides
        if args.model:
            config_dict["model_path"] = args.model
        # Merge any extra CLI args into YAML config (CLI wins)
        config_dict.update(extra_config)
    else:
        # CLI args mode (backward compatible)
        if not args.model:
            parser.error("--model is required when not using --config")
        backend_type = args.backend
        config_dict = {
            "model_path": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        # Merge backend-specific args from extra CLI flags
        config_dict.update(extra_config)

    # Print configuration
    print("=" * 60)
    print("[serve_unified] Starting Unified NeMo Inference Server")
    print("=" * 60)
    print(f"  Backend: {backend_type}")
    print(f"  Model: {config_dict.get('model_path', 'N/A')}")
    print(f"  Port: {args.port}")
    print(f"  GPUs: {args.num_gpus}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Batch Timeout: {args.batch_timeout}s")
    if args.config:
        print(f"  Config: {args.config}")
    if extra_config:
        print(f"  Extra CLI Config: {extra_config}")
    print("=" * 60)

    # Import and run
    try:
        import uvicorn

        from recipes.multimodal.server.unified_server import create_app

        app = create_app(
            backend_type=backend_type,
            config_dict=config_dict,
            batch_size=args.batch_size,
            batch_timeout=args.batch_timeout,
        )

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    except ImportError as e:
        print(f"[serve_unified] Error: Failed to import unified server: {e}")
        print("[serve_unified] Make sure the recipes.multimodal.server package is in PYTHONPATH")
        sys.exit(1)
    except Exception as e:
        print(f"[serve_unified] Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

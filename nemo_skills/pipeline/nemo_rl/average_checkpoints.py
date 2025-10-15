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

import argparse
import json
import logging
import os
import shutil
from enum import Enum

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO)


class SupportedBackends(str, Enum):
    fsdp = "fsdp"
    megatron = "megatron"


def list_candidate_model_dirs(checkpoint_dir, steps):
    """List subfolders whose names contain any of the specified step numbers."""
    out = []
    for name in os.listdir(checkpoint_dir):
        if not os.path.isdir(os.path.join(checkpoint_dir, name)):
            continue
        if any(f"step_{s}" in name for s in steps):
            out.append(name)
    out.sort()
    return out


def find_index_json(model_dir):
    """Find a *.safetensors.index.json file if it exists in the directory."""
    for file in os.listdir(model_dir):
        if file.endswith(".safetensors.index.json"):
            return os.path.join(model_dir, file)
    return None


def build_key_to_shard_map(model_dir):
    """
    Return a dict mapping parameter key -> shard filename.
    Preferred path: read from model.safetensors.index.json if present.
    Fallback path: when index is missing, scan all .safetensors files and
    build the mapping on the fly, then emit a lightweight index for consistency.
    """
    idx_path = find_index_json(model_dir)
    if idx_path and os.path.exists(idx_path):
        with open(idx_path, "r") as fr:
            idx = json.load(fr)
        weight_map = idx.get("weight_map", {})
        if not weight_map:
            raise ValueError(f"Empty weight_map in {idx_path}")
        return weight_map

    # No index present: scan all .safetensors shards and enumerate their keys.
    # Note: GRPO Megatron converted checkpoints typically do not include an index JSON file.
    logging.warning(
        f"[WARN] No .safetensors.index.json found in {model_dir}. Scanning .safetensors files to build weight_map."
    )
    weight_map = {}
    safetensor_files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]

    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    for sf in safetensor_files:
        sf_path = os.path.join(model_dir, sf)
        # safe_open allows listing tensor names without loading them into memory.
        with safe_open(sf_path, framework="pt") as f:
            for k in f.keys():
                # If duplicate keys exist across shards, the last one wins;
                # typical HF sharding ensures uniqueness per key.
                weight_map[k] = sf

    # Generate model.safetensors.index.json for each step hf model folder
    temp_idx = os.path.join(model_dir, "model.safetensors.index.json")
    try:
        with open(temp_idx, "w") as fw:
            json.dump({"weight_map": weight_map}, fw, indent=2)
        logging.info(f"[INFO] Generated temporary index at {temp_idx} ({len(weight_map)} tensors).")
    except Exception as e:
        # Non-fatal: averaging can proceed without writing the temp index.
        logging.warning(f"[WARN] Failed to write temporary index at {temp_idx}: {e}")

    return weight_map


def copy_side_files(src_model_dir, dst_dir):
    """Copy all non-weight auxiliary files from the first model directory to the output."""
    for fname in os.listdir(src_model_dir):
        # Skip weight shards and indexes for megatron
        if fname.endswith(".safetensors") or fname.endswith(".safetensors.index.json"):
            continue
        # Skip FSDP .bin files and their index
        if fname.startswith("pytorch_model") and (fname.endswith(".bin") or fname.endswith(".index.json")):
            continue
        src = os.path.join(src_model_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif os.path.isfile(src):
            shutil.copy2(src, dst)


# Convert FSDP bin checkpoints into safetensors format (before averaging)
def convert_fsdp_bin_to_safetensors(model_dir):
    """
    Convert FSDP .bin checkpoints in a directory into safetensors shards.
    If sharded, uses pytorch_model.bin.index.json as the source of truth.
    Also writes model.safetensors.index.json pointing to the new shards.
    """
    logging.info(f"[FSDP] Converting bin checkpoints in {model_dir} -> safetensors")

    # Find index for sharded FSDP checkpoints
    idx_path = os.path.join(model_dir, "pytorch_model.bin.index.json")
    if not os.path.exists(idx_path):
        # Single-file case
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.exists(bin_path):
            logging.warning(f"[FSDP] No .bin found in {model_dir}, skipping conversion.")
            return
        state_dict = torch.load(bin_path, map_location="cpu")
        save_file({k: v.contiguous() for k, v in state_dict.items()}, os.path.join(model_dir, "model.safetensors"))
        idx = {"weight_map": {k: "model.safetensors" for k in state_dict}}
        with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(idx, f, indent=2)
        logging.info("[FSDP] Converted single bin to model.safetensors")
        return

    # Sharded case
    with open(idx_path) as f:
        idx = json.load(f)
    wm = idx["weight_map"]

    shard_to_keys = {}
    for k, v in wm.items():
        shard_to_keys.setdefault(v, []).append(k)

    for shard, ks in shard_to_keys.items():
        shard_path = os.path.join(model_dir, shard)
        if not os.path.exists(shard_path):
            logging.warning(f"[FSDP] Missing shard: {shard}")
            continue
        sd = torch.load(shard_path, map_location="cpu")
        out_path = shard_path.replace(".bin", ".safetensors")
        save_file({k: sd[k].contiguous() for k in ks}, out_path)
        logging.info(f"[FSDP] Converted {shard} -> {os.path.basename(out_path)}")

    new_index = {"weight_map": {k: v.replace(".bin", ".safetensors") for k, v in wm.items()}}
    with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)
    logging.info("[FSDP] Wrote model.safetensors.index.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, help="Root directory containing multiple model subfolders.")
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        required=True,
        help="List of step numbers to include (e.g. --steps 100 200 300).",
    )
    parser.add_argument(
        "--remove_checkpoints_after_average",
        action="store_true",
        help="If set, will delete the original step directories after averaging.",
    )
    parser.add_argument(
        "--backend",
        type=SupportedBackends,
        required=True,
        help="Choose backend. Supported options: fsdp, megatron.",
    )
    args = parser.parse_args()

    model_dirs = list_candidate_model_dirs(args.checkpoint_dir, args.steps)
    if not model_dirs:
        raise SystemExit("No model subdirectories found")

    logging.info("Selected model dirs:")
    for dirs in model_dirs:
        logging.info("  - %s", dirs)

    # If backend = fsdp, first convert .bin checkpoints to safetensors
    if args.backend == SupportedBackends.fsdp:
        for dirname in model_dirs:
            convert_fsdp_bin_to_safetensors(os.path.join(args.checkpoint_dir, dirname))

    n = len(model_dirs)
    logging.info("Averaging %d checkpoints", n)

    first_dirname = model_dirs[0]
    first_dir = os.path.join(args.checkpoint_dir, first_dirname)

    # Build key->file map from the first directory.
    # This will read the index if present, or scan shards when missing.
    key2file_first = build_key_to_shard_map(first_dir)
    keys = sorted(key2file_first.keys())
    logging.info("Total parameter tensors: %d", len(keys))

    # For each other directory, ensure exact same key set (strict mode).
    per_dir_key2file = {first_dirname: key2file_first}
    for dirname in model_dirs[1:]:
        md = os.path.join(args.checkpoint_dir, dirname)
        k2f = build_key_to_shard_map(md)
        if set(k2f.keys()) != set(keys):
            raise SystemExit(f"[Strict] Key sets differ between {dirname} and first model.")
        per_dir_key2file[dirname] = k2f

    out_dir = os.path.join(args.checkpoint_dir, "final_hf_model")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Group keys by the shard filename defined in the first directory.
    shard_to_keys = {}
    for k in keys:
        shard_to_keys.setdefault(key2file_first[k], []).append(k)

    # Average shard-by-shard to keep memory usage low.
    for shard_name, shard_keys in shard_to_keys.items():
        logging.info("Averaging shard %s with %d tensors", shard_name, len(shard_keys))
        out_state = {}
        for i, k in enumerate(shard_keys):
            # Read the reference tensor from the first dir
            with safe_open(os.path.join(first_dir, shard_name), framework="pt") as f0:
                t0 = f0.get_tensor(k)
                ref_shape = tuple(t0.shape)
                ref_dtype = t0.dtype
                acc = t0.to(dtype=torch.float32)

            # Accumulate from the remaining dirs
            for dirname in model_dirs[1:]:
                md = os.path.join(args.checkpoint_dir, dirname)
                shard_k = per_dir_key2file[dirname][k]
                with safe_open(os.path.join(md, shard_k), framework="pt") as fh:
                    tk = fh.get_tensor(k)
                    if tuple(tk.shape) != ref_shape:
                        raise SystemExit(f"[Strict] Shape mismatch for {k} in {dirname}")
                    if tk.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                        raise SystemExit("Int tensor not supported for averaging: %s (%s)" % (k, tk.dtype))
                    acc.add_(tk.to(dtype=torch.float32))

            acc.div_(float(n))
            out_state[k] = acc.to(dtype=ref_dtype)
            if (i % 100) == 0:
                logging.info("  ... %d / %d tensors", i, len(shard_keys))

        # Save averaged shard using the same shard filename as in the first dir.
        out_path = os.path.join(out_dir, shard_name)
        save_file(out_state, out_path)
        logging.info("Saved shard: %s", out_path)
        out_state.clear()

    # Write/Copy index.json to the output directory:
    # If the first dir has an index, copy it and add metadata.
    # Otherwise, scan the output shards and create a fresh index.
    idx_path = find_index_json(first_dir)
    if idx_path and os.path.exists(idx_path):
        with open(idx_path, "r") as fr:
            idx = json.load(fr)
        meta = idx.setdefault("metadata", {})
        meta["averaged_from"] = model_dirs
        with open(os.path.join(out_dir, os.path.basename(idx_path)), "w") as fw:
            json.dump(idx, fw, indent=2)
        logging.info("Copied index json.")
    else:
        logging.warning(f"[WARN] No index.json found in {first_dir}. Generating one for {out_dir}")
        weight_map = {}
        safetensor_files = [f for f in os.listdir(out_dir) if f.endswith(".safetensors")]
        for sf in safetensor_files:
            sf_path = os.path.join(out_dir, sf)
            with safe_open(sf_path, framework="pt") as f:
                for k in f.keys():
                    weight_map[k] = sf
        idx = {"weight_map": weight_map, "metadata": {"averaged_from": model_dirs}}
        new_idx_path = os.path.join(out_dir, "model.safetensors.index.json")
        with open(new_idx_path, "w") as fw:
            json.dump(idx, fw, indent=2)
        logging.info(f"[INFO] Created new model.safetensors.index.json with {len(weight_map)} tensors")

    # Copy side (non-weight) files from the first directory.
    copy_side_files(first_dir, out_dir)
    logging.info("Averaged (sharded) checkpoint saved at: %s", out_dir)
    logging.info("Done.")

    if args.remove_checkpoints_after_average:
        logging.info("Cleaning up original step directories...")
        for model_dir in model_dirs:
            full_path = os.path.join(args.checkpoint_dir, model_dir)
            try:
                shutil.rmtree(full_path)
                logging.info("Deleted directory: %s", full_path)
            except Exception as e:
                logging.warning("Failed to delete %s: %s", full_path, e)


if __name__ == "__main__":
    main()

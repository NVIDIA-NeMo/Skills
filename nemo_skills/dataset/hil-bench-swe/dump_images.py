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

"""Materialize HIL-Bench SWE per-task images as Apptainer .sif files.

Unlike nemo_skills/dataset/swe-bench/dump_images.py (which runs ``apptainer build
docker://...``), HIL-Bench ships each task's environment as a compressed Docker image
tarball (``<uid>.tar.zst``) hosted on a HuggingFace bucket. For each task this script:

    1. downloads ``<uid>.tar.zst`` from the HuggingFace Storage Bucket holding the images
       (via ``download_bucket_files``; needs HF_TOKEN with access to the bucket),
    2. decompresses it to a ``docker save`` archive (``zstd``/python ``zstandard``),
    3. builds an Apptainer image directly from that archive
       (``apptainer build <instance_id>.sif docker-archive:<tar>``).

No Docker daemon is required (Apptainer reads the ``docker-archive:`` source directly),
which makes this runnable on a cluster compute node that has apptainer + (zstd or the
python ``zstandard`` package).

The resulting ``<instance_id>.sif`` files line up with the ``container_formatter`` used by
``prepare.py``. Run this AFTER ``ns prepare_data hil-bench-swe`` so the input jsonl exists.
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

DATASET_DIR = Path(__file__).parent


def parse_hf_bucket(hf_url: str) -> tuple[str, str]:
    """Parse ``hf://buckets/<org>/<bucket>/<path>`` into (bucket_id, remote_path).

    The HIL-Bench SWE images live in a HuggingFace Storage Bucket (an S3-like store, NOT a
    dataset repo), referenced via ``hf://buckets/...`` paths. E.g.
        hf://buckets/ScaleAI/hil-bench-swe-images/images/<uid>.tar.zst
        -> bucket_id="ScaleAI/hil-bench-swe-images", remote_path="images/<uid>.tar.zst"
    """
    prefix = "hf://buckets/"
    if not hf_url.startswith(prefix):
        raise ValueError(f"Unexpected image url (expected hf://buckets/...): {hf_url}")
    parts = hf_url[len(prefix) :].split("/")
    bucket_id = "/".join(parts[:2])
    remote_path = "/".join(parts[2:])
    return bucket_id, remote_path


def hf_download(hf_url: str, dest_dir: Path) -> Path:
    """Download an image tarball from a HuggingFace Storage Bucket.

    Uses HF_TOKEN (or HUGGING_FACE_HUB_TOKEN). The bucket is access-controlled, so the token
    must belong to an account granted access to the bucket. Requires a huggingface_hub
    version with bucket support (download_bucket_files).
    """
    from huggingface_hub import download_bucket_files

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    bucket_id, remote_path = parse_hf_bucket(hf_url)
    local = dest_dir / Path(remote_path).name
    print(f"  downloading bucket {bucket_id}::{remote_path}")
    download_bucket_files(
        bucket_id=bucket_id,
        files=[(remote_path, str(local))],
        raise_on_missing_files=True,
        token=token,
    )
    if not local.exists():
        raise RuntimeError(f"Bucket download did not produce {local}")
    return local


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("  $ " + " ".join(cmd))
    return subprocess.run(cmd, check=True, **kwargs)


def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


def decompress_zst(zst_path: Path, tar_path: Path):
    """Decompress a .tar.zst to .tar using the zstd binary, else python zstandard."""
    if _have("zstd"):
        run(["zstd", "-d", "-f", str(zst_path), "-o", str(tar_path)])
        return
    try:
        import zstandard  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Neither the `zstd` binary nor the python `zstandard` package is available to "
            "decompress images. Install one of them (`pip install zstandard`)."
        ) from e
    print("  decompressing with python zstandard")
    dctx = zstandard.ZstdDecompressor()
    with open(zst_path, "rb") as fin, open(tar_path, "wb") as fout:
        dctx.copy_stream(fin, fout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default=str(DATASET_DIR / "default.jsonl"),
        help="Prepared HIL-Bench SWE jsonl (output of prepare.py).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write <instance_id>.sif files (match prepare.py's container_formatter).",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Scratch directory for downloads/decompression (defaults to <output_dir>/_work).",
    )
    parser.add_argument(
        "--keep_tar", action="store_true", help="Keep downloaded/decompressed tarballs."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Rebuild .sif files even if they already exist."
    )
    # Parallelism: split the work across many jobs (chunks) and across workers within a job.
    parser.add_argument(
        "--num_chunks", type=int, default=None, help="Split records into this many chunks."
    )
    parser.add_argument(
        "--chunk_id", type=int, default=None, help="Process only this chunk id (0-indexed)."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of images to download+build concurrently within this job.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir) if args.work_dir else out_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # apptainer's build scratch (rootfs unpack + mksquashfs) -> work_dir on the large user Lustre, so big
    # images (e.g. an Electron app with huge node_modules) don't overflow node-local /tmp ("No space left
    # on device"). These are plain sequential writes, fine on Lustre.
    # The blob CACHE is deliberately LEFT on the node-local default: the conveyor copies layers with
    # copy_file_range, which Lustre handles incorrectly (short writes -> "expected blob size X, but only
    # wrote Y"). Keep --max_workers (STAGE_WORKERS) low so the node-local cache doesn't fill up.
    # subprocess inherits os.environ, so the `apptainer build` below picks these up.
    _apptainer_tmp = work_dir / "_apptainer_tmp"
    _apptainer_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["APPTAINER_TMPDIR"] = os.environ["SINGULARITY_TMPDIR"] = str(_apptainer_tmp)

    with open(args.input_file) as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Strided chunking spreads large images evenly across chunks.
    if args.num_chunks is not None and args.chunk_id is not None:
        records = records[args.chunk_id :: args.num_chunks]
        print(f"Processing chunk {args.chunk_id}/{args.num_chunks}: {len(records)} images")

    failures: list[str] = []

    def process_one(rec: dict):
        instance_id = rec["instance_id"]
        sif_path = out_dir / f"{instance_id}.sif"
        if sif_path.exists() and not args.overwrite:
            print(f"[skip] {instance_id}: {sif_path} already exists")
            return
        print(f"[build] {instance_id}")
        tar_path = work_dir / f"{instance_id}.tar"
        zst_path = None
        try:
            zst_path = hf_download(rec["image_url"], work_dir)
            decompress_zst(zst_path, tar_path)
            # Build the .sif straight from the docker save archive (no docker daemon).
            run(["apptainer", "build", str(sif_path), f"docker-archive:{tar_path}"])
            print(f"  -> {sif_path}")
        except Exception as e:
            print(f"[FAIL] {instance_id}: {e}")
            failures.append(instance_id)
            sif_path.unlink(missing_ok=True)  # avoid leaving a partial/corrupt .sif
        finally:
            if not args.keep_tar:
                if zst_path is not None:
                    zst_path.unlink(missing_ok=True)
                tar_path.unlink(missing_ok=True)

    if args.max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            list(ex.map(process_one, records))
    else:
        for rec in records:
            process_one(rec)

    if failures:
        print(f"Done with {len(failures)} failure(s): {', '.join(failures)}")
        raise SystemExit(1)
    print("Done.")


if __name__ == "__main__":
    main()

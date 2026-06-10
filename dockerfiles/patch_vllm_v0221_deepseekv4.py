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

"""Patch vLLM v0.22.1 CUDA-13 compatibility issues for DeepSeek-V4-Pro.

This is intentionally narrow and fails closed if the expected upstream code is
not present. It keeps the fast CUTE/DeepGEMM path enabled while working around
two known image-level compatibility issues seen with DeepSeek-V4-Pro:

1. vLLM FlashAttention CUTE calls ``nvvm.fmax`` without the explicit result
   type expected by the installed CUTLASS DSL binding.
2. CUTLASS DSL's MLIR verifier can reject generated modules with a CUDA-13
   ``llvm.mlir.global_dtors`` schema mismatch even though the JIT can consume
   the generated module.
"""

from __future__ import annotations

import py_compile
from pathlib import Path

VLLM_CUTE_UTILS = Path("/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/cute/utils.py")

CUTLASS_VERIFY_PATHS = [
    Path("/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cutlass_dsl/cutlass.py"),
    Path("/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/dsl.py"),
]


def patch_fmax() -> None:
    """Patch vLLM's CUTE helper to match the installed nvvm.fmax binding."""
    text = VLLM_CUTE_UTILS.read_text()
    old = """    # * NVVM call based on nvvm version
    if CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9:
        # Old API: requires explicit result type as first positional argument
        return Float32(
            nvvm.fmax(
                T.f32(),
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
                loc=loc,
                ip=ip,
            )
        )
    else:
        # New API: infers result type automatically
        return Float32(
            nvvm.fmax(
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
                loc=loc,
                ip=ip,
            )
        )
"""
    new = """    # CUTLASS DSL shipped in this vLLM image still expects the explicit result
    # type argument for nvvm.fmax, including on CUDA 13. Keep the fast CUTE path
    # enabled while matching the installed nvvm binding signature.
    return Float32(
        nvvm.fmax(
            T.f32(),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
            loc=loc,
            ip=ip,
        )
    )
"""
    if old not in text:
        if "type argument for nvvm.fmax" in text:
            print(f"fmax patch already present: {VLLM_CUTE_UTILS}")
            return
        raise SystemExit(f"Expected fmax block was not found in {VLLM_CUTE_UTILS}; refusing to patch blindly")
    VLLM_CUTE_UTILS.write_text(text.replace(old, new))
    print(f"patched {VLLM_CUTE_UTILS}")


def verifier_guard(indent: str) -> list[str]:
    """Return a guarded MLIR verifier call preserving all unexpected errors."""
    return [
        f"{indent}try:\n",
        f"{indent}    module.operation.verify()\n",
        f"{indent}except Exception as exc:\n",
        f"{indent}    msg = str(exc)\n",
        f"{indent}    if (\n",
        f'{indent}        "llvm.mlir.global_dtors" in msg\n',
        f"{indent}        and \"requires attribute 'data'\" in msg\n",
        f"{indent}    ):\n",
        f'{indent}        __import__("warnings").warn(\n',
        f'{indent}            "Skipping CUTLASS MLIR verifier for known CUDA 13 "\n',
        f'{indent}            "global_dtors schema mismatch; compiled fast CUTE "\n',
        f'{indent}            "kernel is still passed to the JIT engine.",\n',
        f"{indent}            RuntimeWarning,\n",
        f"{indent}        )\n",
        f"{indent}    else:\n",
        f"{indent}        raise\n",
    ]


def patch_global_dtors_verifier(path: Path) -> int:
    """Guard known CUDA 13 global_dtors verifier failures in one CUTLASS file."""
    lines = path.read_text().splitlines(keepends=True)
    out: list[str] = []
    count = 0
    for line in lines:
        if line.strip() == "module.operation.verify()":
            indent = line[: len(line) - len(line.lstrip())]
            out.extend(verifier_guard(indent))
            count += 1
        else:
            out.append(line)

    if count:
        path.write_text("".join(out))
        print(f"patched {path} verify calls {count}")
    else:
        print(f"no module.operation.verify() call found in {path}")
    return count


def main() -> None:
    """Apply all image patches and compile-check modified Python files."""
    patch_fmax()
    existing_cutlass_paths = [path for path in CUTLASS_VERIFY_PATHS if path.exists()]
    patched = sum(patch_global_dtors_verifier(path) for path in existing_cutlass_paths)
    if patched == 0:
        raise SystemExit("No CUTLASS verifier calls were patched; refusing to continue")
    for path in [VLLM_CUTE_UTILS, *existing_cutlass_paths]:
        py_compile.compile(str(path), doraise=True)


if __name__ == "__main__":
    main()

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

"""Select a portable cryptography wheel for OpenHands' copied environment."""

import argparse
import hashlib
import importlib.metadata
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    # NeMo-Skills also supports Python 3.10; tomlkit is already a project dependency.
    import tomlkit as tomllib


def install_compatible_cryptography(lockfile: Path) -> None:
    # Alpine uses musl and must retain its separate dependency setup.
    if Path("/etc/alpine-release").exists() or platform.libc_ver()[0] == "musl":
        print("Skipping glibc wheel replacement on Alpine/musl.")
        return

    architectures = {"x86_64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}
    architecture = architectures.get(platform.machine())
    if platform.system() != "Linux" or architecture is None:
        raise RuntimeError("OpenHands wheel compatibility requires Linux x86_64 or aarch64.")

    lock = tomllib.loads(lockfile.read_text(encoding="utf-8"))
    package = next((entry for entry in lock["package"] if entry["name"] == "cryptography"), None)
    if package is None:
        raise RuntimeError("OpenHands lockfile does not contain cryptography.")
    version = importlib.metadata.version("cryptography")
    if version != package["version"]:
        raise RuntimeError(f"Installed cryptography {version} differs from locked version {package['version']}.")

    # Debian 11 has glibc 2.31. Select a 2.28 wheel, regardless of the newer
    # NeMo-Skills installation container's glibc version.
    wheel_platform = f"manylinux_2_28_{architecture}"
    pip = [sys.executable, "-m", "pip"]
    with tempfile.TemporaryDirectory(prefix="openhands-compatible-wheel-") as directory:
        subprocess.run(
            pip
            + [
                "download",
                f"cryptography=={version}",
                "--platform",
                wheel_platform,
                "--python-version",
                f"{sys.version_info.major}.{sys.version_info.minor}",
                "--implementation",
                "cp",
                "--abi",
                "abi3",
                "--only-binary=:all:",
                "--no-deps",
                "--dest",
                directory,
            ],
            check=True,
        )
        wheels = list(Path(directory).glob("cryptography-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one cryptography wheel, found {len(wheels)}.")
        wheel = wheels[0]
        expected_hash = next((entry["hash"] for entry in package["files"] if entry["file"] == wheel.name), None)
        actual_hash = "sha256:" + hashlib.sha256(wheel.read_bytes()).hexdigest()
        if expected_hash is None or actual_hash != expected_hash:
            raise RuntimeError(f"Downloaded wheel {wheel.name} does not match the OpenHands lockfile hash.")
        subprocess.run(pip + ["install", "--force-reinstall", "--no-deps", str(wheel)], check=True)
        print(f"Installed lockfile-verified {wheel.name} for Debian 11 compatibility.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lockfile", type=Path, required=True)
    install_compatible_cryptography(parser.parse_args().lockfile)

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

import hashlib
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from nemo_skills.inference.eval import openhands_compat


class TestOpenHandsCryptographyCompatibility(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.lockfile = Path(self.directory.name) / "poetry.lock"
        self.content = b"compatible wheel fixture"
        self.wheel_name = "cryptography-45.0.3-cp311-abi3-manylinux_2_28_x86_64.whl"
        self.write_lockfile()
        self.context.enter_context(patch.object(openhands_compat.platform, "system", return_value="Linux"))
        self.context.enter_context(patch.object(openhands_compat.platform, "machine", return_value="x86_64"))
        self.context.enter_context(patch.object(openhands_compat.platform, "libc_ver", return_value=("glibc", "2.35")))
        self.context.enter_context(patch.object(openhands_compat.Path, "exists", return_value=False))
        self.context.enter_context(patch.object(openhands_compat.importlib.metadata, "version", return_value="45.0.3"))

    def write_lockfile(self, wheel_name=None, digest=None):
        self.lockfile.write_text(
            '[[package]]\nname = "cryptography"\nversion = "45.0.3"\n'
            f'files = [{{file = "{wheel_name or self.wheel_name}", '
            f'hash = "sha256:{digest or hashlib.sha256(self.content).hexdigest()}"}}]\n'
        )

    def download_wheel(self, command, check):
        self.assertTrue(check)
        if "download" in command:
            Path(command[command.index("--dest") + 1], self.wheel_name).write_bytes(self.content)

    def test_selects_portable_wheel_and_keeps_version_and_dependencies(self):
        with patch.object(openhands_compat.subprocess, "run", side_effect=self.download_wheel) as run:
            openhands_compat.install_compatible_cryptography(self.lockfile)
        download, install = [call.args[0] for call in run.call_args_list]
        self.assertEqual(download[download.index("--platform") + 1], "manylinux_2_28_x86_64")
        self.assertIn("cryptography==45.0.3", download)
        self.assertIn("--only-binary=:all:", download)
        self.assertIn("--no-deps", download)
        self.assertEqual(install[3:6], ["install", "--force-reinstall", "--no-deps"])
        self.assertEqual(Path(install[-1]).name, self.wheel_name)

    def test_selects_aarch64_wheel(self):
        self.wheel_name = self.wheel_name.replace("x86_64", "aarch64")
        self.write_lockfile()
        with (
            patch.object(openhands_compat.platform, "machine", return_value="aarch64"),
            patch.object(openhands_compat.subprocess, "run", side_effect=self.download_wheel) as run,
        ):
            openhands_compat.install_compatible_cryptography(self.lockfile)
        self.assertIn("manylinux_2_28_aarch64", run.call_args_list[0].args[0])

    def test_hash_mismatch_prevents_installation(self):
        self.write_lockfile(digest="0" * 64)
        with patch.object(openhands_compat.subprocess, "run", side_effect=self.download_wheel) as run:
            with self.assertRaisesRegex(RuntimeError, "lockfile hash"):
                openhands_compat.install_compatible_cryptography(self.lockfile)
        self.assertEqual(run.call_count, 1)

    def test_unlisted_wheel_prevents_installation(self):
        self.write_lockfile(wheel_name="another-wheel.whl")
        with patch.object(openhands_compat.subprocess, "run", side_effect=self.download_wheel) as run:
            with self.assertRaisesRegex(RuntimeError, "lockfile hash"):
                openhands_compat.install_compatible_cryptography(self.lockfile)
        self.assertEqual(run.call_count, 1)

    def test_version_mismatch_prevents_download(self):
        with (
            patch.object(openhands_compat.importlib.metadata, "version", return_value="46.0.0"),
            patch.object(openhands_compat.subprocess, "run") as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "differs from locked version"):
                openhands_compat.install_compatible_cryptography(self.lockfile)
        run.assert_not_called()

    def test_skips_alpine_and_musl_without_reading_lockfile(self):
        for alpine, libc in [(True, ""), (False, "musl")]:
            with (
                self.subTest(alpine=alpine, libc=libc),
                patch.object(openhands_compat.Path, "exists", return_value=alpine),
                patch.object(openhands_compat.platform, "libc_ver", return_value=(libc, "")),
                patch.object(openhands_compat.subprocess, "run") as run,
            ):
                openhands_compat.install_compatible_cryptography(Path("missing.lock"))
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

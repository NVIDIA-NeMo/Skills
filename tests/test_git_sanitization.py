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

"""Real-Git regressions; also runnable with Python's standard unittest runner."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from nemo_skills.inference.eval.git_sanitization import rewrite_legacy_git_cleanup

# Match the commands in existing prepared multilingual datasets.
LEGACY_CLEANUP = (
    "rm -f .git/packed-refs && "
    "find .git/refs/heads -type f ! -name 'main' -delete && "
    "rm -rf .git/refs/tags .git/refs/remotes && "
    "git reflog expire --expire=now --all && "
    "git gc --prune=now --aggressive"
)
LEGACY_PRE_COMMANDS = (
    "git checkout -B main && "
    "git config user.email 'user@example.com' && "
    "git config user.name 'user' && " + LEGACY_CLEANUP
)


class GitSanitizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name) / "repo"
        self.repo.mkdir()
        self.git("init", "-b", "main")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "Test")
        self.git("config", "commit.gpgsign", "false")

    def git(self, *args, check=True):
        return subprocess.run(["git", *args], cwd=self.repo, check=check, capture_output=True, text=True)

    def shell(self, command):
        return subprocess.run(["bash", "-c", command], cwd=self.repo, capture_output=True, text=True)

    def seed_repo(self):
        (self.repo / "source.py").write_text("base\n")
        self.git("add", ".")
        self.git("commit", "-m", "base")
        base = self.git("rev-parse", "HEAD").stdout.strip()
        (self.repo / "source.py").write_text("hidden solution\n")
        self.git("commit", "-am", "future solution")
        future = self.git("rev-parse", "HEAD").stdout.strip()
        self.git("branch", "solution", future)
        self.git("tag", "-a", "solution-tag", "-m", "hidden answer", future)
        self.git("update-ref", "refs/remotes/origin/main", future)
        self.git("update-ref", "refs/hidden/answer", future)
        self.git("symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main")
        self.git("symbolic-ref", "refs/hidden/current", "refs/heads/main")
        (self.repo / "source.py").write_text("stashed answer\n")
        self.git("stash", "push", "-m", "hidden stash")
        self.git("reset", "--hard", base)
        return base, future

    def assert_sanitized(self, base, future):
        self.assertEqual(self.git("rev-parse", "HEAD").stdout.strip(), base)
        self.assertEqual(self.git("symbolic-ref", "HEAD").stdout.strip(), "refs/heads/main")
        self.assertEqual(
            self.git("for-each-ref", "--format=%(refname)").stdout.splitlines(),
            ["refs/heads/main"],
        )
        self.assertNotEqual(self.git("cat-file", "-e", future, check=False).returncode, 0)
        self.assertEqual(self.git("reflog", "show", "--all").stdout, "")
        self.git("fsck", "--full")

    def test_legacy_commands_break_packed_main(self):
        self.seed_repo()
        self.git("pack-refs", "--all", "--prune")
        self.assertFalse((self.repo / ".git/refs/heads/main").exists())
        self.shell(LEGACY_PRE_COMMANDS)
        self.assertNotEqual(self.git("rev-parse", "--verify", "HEAD", check=False).returncode, 0)

    def test_packed_refs_preserve_head_and_prune_hidden_history(self):
        base, future = self.seed_repo()
        self.git("pack-refs", "--all", "--prune")
        self.assertFalse((self.repo / ".git/refs/heads/main").exists())
        result = self.shell(rewrite_legacy_git_cleanup(LEGACY_PRE_COMMANDS))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_sanitized(base, future)

    def test_loose_refs_preserve_setup_changes_and_untracked_files(self):
        base, future = self.seed_repo()
        (self.repo / "source.py").write_text("legitimate setup change\n")
        self.git("add", "source.py")
        (self.repo / "setup.lock").write_text("setup artifact\n")
        index_before = self.git("write-tree").stdout
        result = self.shell(rewrite_legacy_git_cleanup(LEGACY_PRE_COMMANDS))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_sanitized(base, future)
        self.assertEqual((self.repo / "source.py").read_text(), "legitimate setup change\n")
        self.assertEqual((self.repo / "setup.lock").read_text(), "setup artifact\n")
        self.assertEqual(self.git("write-tree").stdout, index_before)

    def test_detached_head_is_preserved_as_main(self):
        base, future = self.seed_repo()
        self.git("checkout", "--detach", base)
        result = self.shell(rewrite_legacy_git_cleanup(LEGACY_CLEANUP))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_sanitized(base, future)

    def test_invalid_head_stops_before_agent(self):
        result = self.shell(rewrite_legacy_git_cleanup(LEGACY_CLEANUP) + " && touch agent-started")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.repo / "agent-started").exists())

    def test_gc_failure_stops_before_agent(self):
        self.seed_repo()
        result = self.shell(
            'git() { if [ "$1" = gc ]; then return 37; fi; command git "$@"; }; '
            + rewrite_legacy_git_cleanup(LEGACY_PRE_COMMANDS)
            + " && touch agent-started"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.repo / "agent-started").exists())

    def test_ref_deletion_failure_stops_before_agent(self):
        self.seed_repo()
        result = self.shell(
            'git() { if [ "$1" = update-ref ] && [ "$2" = --no-deref ]; '
            'then return 37; fi; command git "$@"; }; '
            + rewrite_legacy_git_cleanup(LEGACY_PRE_COMMANDS)
            + " && touch agent-started"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.repo / "agent-started").exists())

    def test_unrelated_commands_and_rewritten_commands_are_unchanged(self):
        custom = "git reset --hard abc123 && echo 'custom setup'"
        self.assertEqual(rewrite_legacy_git_cleanup(custom), custom)
        fixed = rewrite_legacy_git_cleanup(LEGACY_PRE_COMMANDS)
        self.assertEqual(rewrite_legacy_git_cleanup(fixed), fixed)


if __name__ == "__main__":
    unittest.main()

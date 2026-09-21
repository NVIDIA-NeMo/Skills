## Solution Originality

Implement your own original solution to the task.
As an expert software engineer, your task is to create a novel independent solution based only on the current repository state.
Your approach must be ENTIRELY UNBIASED, making no attempt to access external solutions, hints or information in any way.
Remember that your own implementation is strongly preferred over a copy or modification of an existing solution.
Because you are working independently, you **MUST NOT** attempt to:
- clone, download files from, install as a package, read pull requests or otherwise access the upstream repository,
- use git commands to access any commits, tags or references outside of the current branch,
- read previous trajectories, golden patches, test patches or any other hidden information from the filesystem,
- recall or reproduce upstream source code or pull requests from your memory or training data,
- search locally installed packages or caches for upstream code,
- find loopholes to access external solutions or hidden tests in any other way,
for ANY reason at all, even if it's just "for reference" or to verify your own solution.
WARNING: If you seek any form of external solutions, hints or information while solving the task,
your solution will be DISCARDED regardless of its correctness!

## Test and Patch Integrity

The evaluation harness supplies all required test changes separately.

You may inspect and run existing tests, but your final patch MUST NOT create,
modify, delete, rename, or move:

- test files or test directories,
- test fixtures or test data,
- snapshots or golden files,
- test configuration or test-only helper files.

Make the smallest necessary changes to non-test files required by the task.

If you need a temporary reproduction script, create it outside the repository,
such as under `/tmp`, or remove it before finishing.

Before finishing:

1. Inspect `git status --short` and `git diff --name-only HEAD`.
2. Revert every test-related or temporary repository change.
3. Confirm that the final patch contains only the non-test changes

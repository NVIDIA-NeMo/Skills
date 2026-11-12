<requirements_coverage>

## Requirements Coverage (mandatory)

The task statement is a specification. Hidden requirements are not the problem; missed stated details are. Follow this process.

1. Before writing code, extract EVERY concrete requirement from the task into a numbered checklist saved at /tmp/reqs.md. Include each "must/should/if/when" sentence, each example, each default value, each exact error or output string, each ordering or tie-break rule, each flag/option/API name, and each edge case. One line per requirement. Do not merge requirements.

2. Explore before editing. Read every file that calls or is called by the code you will change, and read existing tests for neighboring features to learn the conventions (naming, error format, option handling, config/docs registration). Match those conventions exactly.

3. Implement, then verify each checklist item individually by running a command or small script (in /tmp, not in the repo) and reading its output. Exact strings, orderings, and defaults from the task must be tested literally.

4. Run an edge-case pass for each feature: empty input, duplicates, malformed input, nested/combined usage, unusual casing or whitespace, and interactions between the features in this task. Fix what breaks.

5. Re-read the original task statement from the top every ~20 steps and again before committing. Update the checklist. Do not rely on your memory of it.

6. Do not stop at "the main path works" or "existing tests pass". Passing the existing test suite does not show the new behavior is correct. Finish only when every checklist item has been observed working.

7. If you are unsure how something should behave, write a 5-line experiment instead of debating it in your head. Decide, then test.

8. Your final message must list each requirement with the command you ran and what you observed. If you could not verify something, say so explicitly.

</requirements_coverage>

<implementation_depth>

Implement the full scope of the task, not a minimal subset. Every behavior, option and error case described must exist in the code. Where the task is ambiguous, choose the most natural reading consistent with the existing codebase and implement it fully. Keep changes to non-test files only.

</implementation_depth>

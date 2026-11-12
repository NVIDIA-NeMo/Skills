<requirements_coverage>

## Requirements Coverage (mandatory)

The task statement is the complete specification. Every detail it states (names, defaults, exact messages, ordering, edge cases) is a requirement and must be implemented and verified. Implement the full scope, not a minimal subset. Where the task is ambiguous, choose the most natural reading consistent with the existing codebase and implement it fully.

1. Before writing code, extract every concrete requirement from the task into a numbered checklist saved at /tmp/reqs.md (outside the repo). Include each "must/should/if/when" sentence, example, default value, exact error or output string, ordering or tie-break rule, flag/option/API name, and edge case. One line per requirement; do not merge requirements.

2. Explore before editing. Read the code you will change, the files that call it or that it calls, and the existing tests for neighboring features. Learn the conventions (naming, error format, option handling, config/docs registration) and match them exactly.

3. After implementing, verify each checklist item individually by running a command or small script (in /tmp, not in the repo) and reading its output. Test exact strings, orderings and defaults literally.

4. For each feature, run an edge-case pass: empty input, duplicates, malformed input, nested or combined usage, unusual casing or whitespace, and interactions between the features in this task. Fix what breaks.

5. Re-read the original task statement from the top about every 20 steps and again before committing, and update the checklist. Do not rely on your memory of it.

6. Passing the existing test suite does not show the new behavior is correct. Finish only when every checklist item has been observed working.

7. If you are unsure how something should behave, write a short experiment instead of debating it. Decide, then test.

8. Your final message must list each requirement with the command you ran and what you observed, one line each. If you could not verify something, say so explicitly.

</requirements_coverage>

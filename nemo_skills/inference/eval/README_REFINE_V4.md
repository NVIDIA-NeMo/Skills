# SWE-bench Refine V4: Failure-Aware Raw Evidence

V4 is a conservative extension of refine V1. It keeps the raw verifier evidence and previous diff, but adds a short instruction selected by the verifier failure type.

## Motivation

The earlier evals suggest Qwen3.5-4B benefits from raw verifier output, while heavier structured prompts can over-constrain or distract it. V4 therefore avoids a long checklist. It only routes the next-round instruction by failure mode:

- `no_patch`: treat the previous run as weak evidence and start from the original issue.
- `target_tests_still_failing`: focus on failing `FAIL_TO_PASS` evidence.
- `regression_introduced`: preserve existing behavior first; narrow or revert risky prior changes.
- `mixed_failure`: remove regression first, then address target tests.
- `syntax_or_import_error`: fix syntax/import/setup failure before broader changes.
- `timeout`: look for excessive work or infinite loops and prefer a localized fix.
- `patch_apply_failed`: recover intent, but produce a clean patch against the fresh repo.

## Seed Shape

For an unresolved round, V4 appends to the original problem statement:

1. clean-repo reminder
2. verifier failure type
3. one short failure-type-specific instruction
4. previous changed files
5. temporary/debug artifact-file warning if detected
6. failing `FAIL_TO_PASS` tests
7. regressed `PASS_TO_PASS` tests
8. key verifier output plus deduplicated raw verifier tail
9. previous patch, middle-truncated by `carry_over_token_budget`
10. final instruction: previous patch is evidence only; keep, revise, or discard it

## Expected Signal

V4 tests whether small failure-aware routing improves rescue rate without losing V1's raw-evidence advantage. The most important slices to compare are:

- rescue from `no_patch`
- rescue from `target_tests_still_failing`
- regression recovery from `regression_introduced` / `mixed_failure`
- final `no_patch` rate
- artifact patch rate
- changed-file overlap from attempt0 to final

## Eval Config

Use:

```bash
++refine_strategy=failure_aware
++max_refine_rounds=2
++refine_attempt0_bank_file=/path/to/fixed_attempt0_bank.jsonl
```

The checked-in Slurm launcher accepts the same configuration:

```bash
launchers/swebench_refine/submit_refine_swebench_batch.sh \
  --context-k 64 \
  --turns 100 \
  --max-refine-rounds 2 \
  --refine-strategy failure_aware \
  --refine-attempt0-bank-file /path/to/fixed_attempt0_bank.jsonl
```

See [`launchers/swebench_refine/README.md`](launchers/swebench_refine/README.md) for the required model, input, image, runtime-cache, and cluster options.

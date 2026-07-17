# Refine V3dedup: Compact Raw Evidence

This document describes the eval-side **v3dedup** refine variant implemented in
`swebench.py`.

Use:

```text
agent_framework=swe_agent_refine
refine_strategy=compact_raw
```

## One-Line Summary

V3dedup runs a clean-repo multi-round SWE-agent chain. It differs from v1 only
in the seed format for round 1+:

```text
key verifier traceback/assertion first
optional non-overlapping verifier context
previous patch diff
neutral continuation instruction
```

It does not include v2's structured hypothesis checklist, artifact paths,
changed-file lists, or failed-test tables. It keeps raw verifier evidence, but
deduplicates overlapping key snippets and verifier-tail context.

## Eval Loop

For each instance:

1. Round 0 sees the original SWE-bench problem statement.
2. SWE-agent runs from a clean repository and produces `model_patch`.
3. SWE-bench harness verifies the patch.
4. If resolved, stop early.
5. If unresolved, build a new problem statement from:
   - original problem statement
   - previous patch
   - key verifier traceback/assertion
   - non-overlapping verifier output tail
6. The next round starts from a clean repository again.

No workspace state, full trajectory, or full conversation history is carried
across rounds. Cross-round information is text-only.

## Why This Variant Exists

V3dedup tests the following hypothesis:

```text
For small coding models, the useful verifier evidence can be drowned out by
long raw logs, large previous diffs, or repeated traceback text.
```

Compared with v1, v3dedup front-loads the highest-signal verifier failure.
Compared with v2, it avoids heavy structure and avoids telling the model to
preserve the previous patch.

| Field | V1 basic refine | V2 structured_hypothesis | V3dedup compact_raw |
|---|---|---|---|
| Previous patch | yes | yes | yes, middle-truncated |
| Raw verifier tail | yes | yes | yes, deduplicated against key snippet |
| Key traceback/assertion | no special position | yes | yes, first-class section |
| Structured hypothesis/checklist | no | yes | no |
| Artifact paths / changed-file lists | no | yes | no |
| Clean repo each round | yes | yes | yes |

## Suggested Eval Config

```text
agent_framework=swe_agent_refine
refine_strategy=compact_raw
max_refine_rounds=2
agent_max_turns=200
carry_over_token_budget=30000
refine_verify_feedback_chars=6000
refine_failure_snippet_chars=3000
```

For continuation/scaling evals, use `refine_resume_bank_file` so already
resolved samples remain frozen and only unresolved samples continue.

## Seed Shape

Conceptually, round `k > 0` receives the original problem plus:

````text
---
Your previous automated refine round did NOT resolve the issue.

You are starting again from a clean repository. Use the previous round only as
debugging evidence.

Key verifier output:
```text
<traceback/assertion/error-focused snippet>
```

Additional verifier context:
```text
<non-overlapping tail of test_output.txt/log, omitted if redundant>
```

Previous patch:
```diff
<previous patch, middle-truncated if needed>
```

Use the previous patch only as evidence. You may keep, revise, or discard it.
Produce a complete minimal patch from the clean repository.
````

## Budget Semantics

| Config | Meaning |
|---|---|
| `carry_over_token_budget` | Approximate budget for previous patch text. Estimation uses about 4 chars per token. Long patches keep head and tail and drop the middle. |
| `refine_verify_feedback_chars` | Maximum raw verifier feedback tail to carry into the next seed. |
| `refine_failure_snippet_chars` | Maximum key traceback/assertion snippet extracted from verifier output. |

## Controlled Eval100 Result

Fixed attempt0 bank comparison:

| Variant | Strategy | Resolved | Accuracy | Delta vs attempt0 |
|---|---|---:|---:|---:|
| Baseline attempt0 only | no refine | 29/100 | 29% | - |
| V1 basic refine | `baseline` | 37/100 | 37% | +8 |
| V2 test-evidence refine | `structured_hypothesis` | 35/100 | 35% | +6 |
| V3 compact raw legacy | `compact_raw_legacy` | 37/100 | 37% | +8 |
| V3dedup compact raw | `compact_raw` | 38/100 | 38% | +9 |
| V4 failure-aware | `failure_aware` | 37/100 | 37% | +8 |

Multi-round v3dedup scaling on eval100:

| Stage | Newly solved | Conditional rescue rate | Cumulative solved | Cumulative pass |
|---|---:|---:|---:|---:|
| attempt0 | 29 | 29/100 = 29.0% | 29 | 29% |
| refine round 1 | 9 | 9/71 = 12.7% | 38 | 38% |
| refine round 2 | 5 | 5/62 = 8.1% | 43 | 43% |
| refine round 3 | 2 | 2/57 = 3.5% | 45 | 45% |
| r4 continuation | 3 | 3/55 = 5.5% | 48 | 48% |
| r5 continuation | 4 | 4/52 = 7.7% | 52 | 52% |
| r6 continuation | 3 | 3/48 = 6.2% | 55 | 55% |
| r7 continuation | 1 | 1/45 = 2.2% | 56 | 56% |

## Metrics To Inspect

| Metric | Question |
|---|---|
| `chain_resolved` | Did any round solve the task? |
| `resolved_at_refine_round` | Which round first solved it? |
| `refine_rescued` | Did refine rescue an unresolved attempt0? |
| `rescued_from_failure_type` | Which failure type was rescued? |
| `num_repeat_failure_attempts` | Does the model repeat the same failed tests? |
| `num_artifact_patch_attempts` | Did the patch include debug/reproduce artifacts? |
| `patch_size_delta_attempt0_to_final` | Did refine shrink or bloat the patch? |

Important: raw resolved can overstate patch quality. In the r7 eval100 run, raw
resolved was 56/100, but artifact-free resolved was only 42/100.


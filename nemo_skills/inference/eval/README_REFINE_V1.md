# Refine V1: Basic Patch + Verifier Feedback

This document describes the eval-side refine v1 mechanism in `swebench.py` and how to port the same idea into multi-turn SWE training.

## One-line Summary

Refine v1 runs SWE-agent for the same instance up to `max_refine_rounds` times. Round 0 sees the original problem only. If it fails, round 1 starts from a clean repository and receives a text seed containing the previous patch diff plus the tail of the real SWE-bench verifier output.

The chain is successful if any attempt resolves the instance.

## Current Eval Config

Typical eval config:

```text
agent_framework=swe_agent_refine
refine_strategy=baseline
max_refine_rounds=2
agent_max_turns=200
carry_over_token_budget=40000
refine_verify_feedback_chars=8000
```

Important: `refine_strategy=baseline` here means "basic refine v1", not "no refine". The non-refine baseline is `agent_framework=swe_agent`.

## Attempt Flow

```text
Original SWE-bench problem_statement
  |
  v
Attempt 0: SWE-agent on clean repo
  |
  v
Extract model_patch and trajectory
  |
  v
Run SWE-bench verifier
  |-- resolved=True  -> stop, chain success
  |
  |-- unresolved
       |
       v
Build v1 seed from previous patch + verifier log tail
       |
       v
Attempt 1: SWE-agent on clean repo with seed appended to problem_statement
       |
       v
Run SWE-bench verifier again
       |-- resolved=True  -> chain success
       |-- unresolved     -> chain failure
```

Each attempt gets a clean checkout. Workspace state is not preserved across attempts. The only carry-over is text appended to the problem statement.

## V1 Seed

The v1 seed is intentionally simple. It contains:

1. A notice that the previous automated attempt did not resolve the issue.
2. The previous attempt's patch diff.
3. The tail of `test_output.log` from SWE-bench verification.
4. A short instruction to refine the patch.

Conceptually:

````text
---
Your previous automated attempt did NOT resolve the issue.

Here is the diff you produced so far:
```diff
<previous model_patch, middle-truncated by carry_over_token_budget>
```

Running the tests on that patch produced:
```text
<last refine_verify_feedback_chars chars of test_output.log>
```

Continue refining the patch...
````

Only the immediately previous attempt is passed forward. V1 does not include the full trajectory, full conversation history, or live workspace state.

## Token Budgets

`carry_over_token_budget` only controls the maximum size of the previous patch diff that is included in the next attempt seed.

The implementation uses a rough estimate:

```text
1 token ~= 4 chars
```

So:

```text
carry_over_token_budget=40000 ~= 160000 chars of previous diff
```

If the patch is too large, the diff is middle-truncated: keep head and tail, drop the middle. This does not cap the full prompt. The full prompt also includes the original problem, SWE-agent wrapper prompt, verifier output, and other agent context.

## Eval Artifacts

Per attempt artifacts are separated by suffix:

```text
rs2/trajectories_r0/
rs2/eval-outputs_r0/
rs2/trajectories_r1/
rs2/eval-outputs_r1/
```

The final output row contains `swe-bench-refine`, for example:

```json
{
  "refine_strategy": "baseline",
  "num_refine_rounds": 2,
  "max_refine_rounds": 2,
  "chain_resolved": true,
  "resolved_at_refine_round": 1,
  "attempt0_resolved": false,
  "attempt0_failure_type": "target_tests_still_failing",
  "final_failure_type": "resolved",
  "refine_attempted": true,
  "refine_rescued": true,
  "rescued_from_failure_type": "target_tests_still_failing",
  "per_attempt": [...]
}
```

## Eval100 Result

Run:

```text
qwen35_4b_swebench_full_eval100_qwen3coder_qwen3reason_refinev1_64k_200step_2att_8n
```

Result:

```text
chain pass@1 = 45/100 = 45.0%
```

Breakdown:

| Metric | Value |
|---|---:|
| Attempt 0 resolved | 33 |
| Attempt 1 rescued | 12 |
| Final resolved | 45 |
| Final unresolved | 55 |
| Refine attempted | 67 |
| Rescue rate among refined samples | 12/67 = 17.9% |

Failure type transition summary:

| Field | Distribution |
|---|---|
| `attempt0_failure_type` | resolved 33, no_patch 29, target_tests_still_failing 29, mixed_failure 6, regression_introduced 3 |
| `final_failure_type` | resolved 45, no_patch 29, target_tests_still_failing 19, mixed_failure 6, regression_introduced 1 |
| `rescued_from_failure_type` | no_patch 5, target_tests_still_failing 5, mixed_failure 1, regression_introduced 1 |

Caveat: this 64k run logged many context length validation errors. The result is still useful for comparing the current run, but a cleaner comparison should reduce `carry_over_token_budget`, for example to `30000`.

## Porting V1 to Multi-turn Training

The closest training analogue is a two-episode chain:

```text
turn group 0:
  model attempts the original SWE task
  environment extracts patch
  verifier runs tests
  reward/metrics recorded

turn group 1:
  model receives original task + previous patch + verifier feedback
  model starts from clean repo or reset repo
  verifier runs tests again
  final chain reward/metrics recorded
```

Recommended training-side observation for attempt 1:

````text
Original problem statement

Previous patch:
```diff
<previous patch, truncated>
```

Verifier feedback:
```text
<test_output.log tail>
```

Continue refining the patch.
````

The key choice is whether training should reset the repo between attempts:

| Choice | Meaning |
|---|---|
| Clean repo, text carry-over only | Matches current eval v1 exactly |
| Persistent workspace | More like an IDE repair loop, but not directly comparable to current eval |

For initial multi-turn training, use clean repo plus text carry-over to stay aligned with eval.

## Training Metrics to Add

Core chain metrics:

| Metric | Meaning |
|---|---|
| `chain_resolved` | Any attempt solved the instance |
| `resolved_at_attempt` | First successful attempt index |
| `attempt0_resolved` | Whether the first attempt solved it |
| `refine_attempted` | Whether a later attempt was run |
| `refine_rescued` | Attempt 0 failed but a later attempt solved |
| `rescued_from_failure_type` | Failure category before rescue |

Verifier metrics:

| Metric | Meaning |
|---|---|
| `fail_to_pass_failed_count` | Target tests still failing |
| `pass_to_pass_failed_count` | Previously passing tests now failing; regression count |
| `unknown_failed_tests_count` | Failing tests not matched to known FAIL_TO_PASS/PASS_TO_PASS lists |
| `failure_type` | no_patch, patch_apply_failed, target_tests_still_failing, regression_introduced, mixed_failure, timeout, etc. |
| `failure_type_transition` | Attempt k failure type -> attempt k+1 failure type |

Patch metrics:

| Metric | Meaning |
|---|---|
| `patch_exists` | Model produced a patch |
| `patch_successfully_applied` | Patch can be applied by verifier |
| `num_changed_files` | Patch breadth |
| `num_added_lines`, `num_removed_lines` | Patch size |
| `num_hunks` | Patch structural complexity |
| `patch_size_delta` | Change in patch size across attempts |
| `changed_file_overlap_with_previous` | Whether the model revisits the same files |

Repeat-failure metrics:

| Metric | Meaning |
|---|---|
| `failed_test_overlap_with_previous` | Same tests failed again after refine |
| `num_repeat_failure_attempts` | Count of attempts that repeated failures |
| `target_failure_delta` | Change in FAIL_TO_PASS failures after refine |
| `regression_delta` | Change in PASS_TO_PASS failures after refine |

Efficiency and context metrics:

| Metric | Meaning |
|---|---|
| `turns_used_per_attempt` | Agent turns consumed |
| `tool_calls_per_attempt` | Shell/editor/test calls |
| `tokens_in`, `tokens_out` | Model cost/length |
| `carry_over_patch_chars` | Size of previous patch included in seed |
| `verify_feedback_chars` | Size of verifier feedback included |
| `patch_truncated` | Whether previous diff was truncated |
| `context_length_error` | Whether server rejected prompt due to context |

For reward modeling, a useful scalar is:

```text
refine_delta_reward =
  final_resolved_reward
  - attempt0_resolved_reward
  - regression_penalty
  - repeat_failure_penalty
```

This separates "refine actually repaired the chain" from "the first attempt was already solved."

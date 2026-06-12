# Repeat10 Multiplexer/New-Checkpoint Report

Status as of 2026-06-10 21:35 PDT, with live continuation notes added after 22:40 PDT. This report summarizes the completed local repeat10 workstream that reran previously successful AceProof harness variants against the `ultra` cloud multiplexer endpoint/new checkpoint, with no explicit token cap in requests.

## Live Continuation After Initial Report

Active follow-up work after the 21:35 PDT report focuses on the six still-unconfirmed rows:

```text
N1, proofbench133_028, proofbench133_109, C50, proofbench133_030, G25
```

New active runs:

- `outputs/repeat10-ultra-mp-remaining6-r2/`: two-round streaming solve/verify/refine batch over the six remaining rows, 14 prompt/temperature arms, 64 first-round proof attempts per problem. This has so far produced many normal-verifier G25 promotions, all matching the known invalid Pitot-converse pattern, plus invalid `proofbench133_109` candidates with Gaussian-integer exponent/sign or modular-residue flaws.
- `outputs/repeat10-ultra-mp-remaining6-genonly-wide/`: generation-only wide batch, 8 prompt families at temperature 1.0, 128 attempts per problem and prompt family. This avoids normal-verifier early stopping and is intended to create a larger candidate pool for fixed promotion sidecars.
- `outputs/repeat10-ultra-mp-remaining6-repairseed-genonly/`: generation-only repair-seed batch, temperatures 1.0 and 0.8, 128 attempts per repair seed. It uses `recipes/aceproof-tts/dataset/repeat10-remaining6-repairseeds-20260610.jsonl`, which contains prior candidate proofs plus non-authoritative audit notes for `N1`, `proofbench133_109`, `proofbench133_030`, `G25`, and `proofbench133_028`. This produced a new audited `N1` solution route: repairseed seed 89 proves block equality between consecutive prime indices using the prime-divisor-of-differences lemma plus Sylvester's theorem, then uses prime gaps/PNT for the limit. Independent audits judged the argument sound, with only minor wording cleanup around using set equality rather than pointwise `a_i=i` and explicitly establishing positivity before applying Sylvester.
- `outputs/repeat10-ultra-mp-remaining5-nog25-genonly-wide/`: no-G25 generation-only wide batch over `N1`, `proofbench133_028`, `proofbench133_109`, `C50`, and `proofbench133_030`, launched after G25 dominated fast completions with static-rejected false proofs.

Verifier/promotion experiments:

- `proof_verification_strict_audit.yaml` and `proof_verification_theorem_gate.yaml` were not reliable for G25; they continued to give score 1 to the false Pitot-converse candidates.
- `proof_verification_false_converse_guard.yaml` partially reduced G25 false positives but remained inconsistent: it sometimes rejected the false side-sum sufficiency argument and sometimes still asserted the false Pitot iff.
- `proof_verification_countermodel_audit.yaml` did not fix G25 in early rows; it still accepted the same false candidates.
- `run_static_promotion_filters.py` is currently the most reliable fixed promotion gate for this specific failure mode. On the calibration set it rejected exactly the G25 false/uncertain Pitot-converse candidates and did not reject the known-valid C39/N14/N24/N39 examples. Applied to current live G25 final rows, it rejected all normal-verifier G25 promotions seen so far.

Live continuation update after 23:00 PDT:

- `N1` now has a confirmed audited repeat10 route from `outputs/repeat10-ultra-mp-remaining6-repairseed-genonly/temp10/repairseed/rounds/R1/proof_gen/output_chunk_0.jsonl-async`, candidate uid `N1:89:repairseed/rounds:ebcd6c24a6fc47cc`. The proof uses the prime-divisor-of-differences lemma, induction over consecutive primes, Sylvester's theorem for blocks of `q`-smooth consecutive integers, and PNT for prime gaps. Independent audits found it valid assuming Sylvester's theorem is allowed. Fixed verifier sidecars also support this family: theorem/congruence/Gaussian guards accepted seed 89, and strict/Gaussian guards accepted related N1 variants.
- `proofbench133_109` remains unsolved. Four additional high-value candidates, including a fresh no-G25 `formalsanity` candidate and three repairseed candidates, were independently audited invalid. The recurring fatal flaw is the same: after Gaussian-integer cube reduction, the proofs silently choose a unit/sign/quadrant branch, or treat factors such as `c` and `c^2-3d^2` as positive when both may be negative. The new `proof_verification_gaussian_sign_branch_guard.yaml` prompt was added to make this a fixed verifier-sidecar check.
- Current fair promptwide G25 rows are not useful solves. The deterministic Pitot-converse filter rejects 93 of 94 completed G25 rows from `outputs/repeat10-ultra-mp-current5-fair-promptwide/`; the single survivor had an empty proof string, so effectively all completed G25 fair-prompt candidates are rejected.
- New coordinated/decomposition infrastructure was added in commit `bf641ef86` and pushed to `igitman/nemo-skills-aceproof` branch `aceproof-repeat10-mux-share-20260610`. It includes model-generated attempt-memory prompts/builders, decomposition planner/subproblem/assembly builders, a stricter Gaussian sign verifier, and launchers for current-five promptwide/decomposition/verifier batches.
- Active coordinated runs at this point include:
  - `outputs/repeat10-ultra-mp-current5-fair-promptwide/`
  - `outputs/repeat10-ultra-mp-current5-decomp-planner/`
  - `outputs/repeat10-decomp-subproblem-solver/v1_solver/`
  - `outputs/repeat10-ultra-mp-current5-branch-exhaustive/`
  - `outputs/repeat10-genonly-candidate-verification/v4_*`

The current strategy is to exclude already solved `N1` from future large candidate-generation batches where practical, keep using static G25 filtering for promotion, and push `proofbench133_109` through either a genuinely branch-exhaustive Gaussian proof or a non-Gaussian route.

## Executive Summary

We targeted 10 rows that the new checkpoint still needed help with under the reference setup:

- `C39`, `N1`, `N14`, `N24`, `N39`, `proofbench133_028`, `proofbench133_109`, `C50`, `proofbench133_030`, `G25`.

Confirmed audited solves from this repeat10 workstream:

| Problem | Source | Method / arm | Verifier signal | Final status |
| --- | --- | --- | --- | --- |
| `N14` | `gems-remarkable` | `temp10/formalsanity`, R1 direct generation | `1/1`, mean `1.0` | Confirmed valid by audit |
| `N24` | `gems-remarkable` | `temp08/proofonly`, R1 direct generation | no verifier rows for the valid candidate | Confirmed valid out-of-band by audit |
| `C39` | `gems-remarkable` | `temp10/proofonly`, reduced remaining-8 rerun | original confirmed row `1/4`, mean `0.625`; later C39 candidates reached `11/12` | Confirmed valid by audit, assuming Vosper equality theorem is allowed |
| `N39` | `gems-remarkable` | N39 generic repair-seed stream, `temp08/repairseed` seed 9 and `temp10/repairseed` seed 51 | seed 9: `2/4`, mean `0.75`; seed 51: `3/4`, mean `0.875` | Confirmed valid by audit |

The most important result from the completed tail is that `N39` is now a repeat10 success. It was first solved by explicit audit-feedback variants, but more importantly it was also solved by the more generic repair-seed stream that seeded from the previous candidate plus a coarse audit note, without the explicit targeted modular fixes used in the later audit-feedback prompt.

The most important operational finding is unchanged: final pipeline promotion alone is not reliable enough here. Real solves were found by inspecting candidate bundles and using independent/manual audits, while the normal verifier also strongly promoted several invalid families.

## Inputs And Harness

Main 10-row input:

```text
recipes/aceproof-tts/dataset/repeat10-newckpt-nosolutions-20260608.jsonl
```

Reduced rerun input after excluding confirmed `N14` and `N24`:

```text
recipes/aceproof-tts/dataset/repeat10-newckpt-remaining8-after-audit-20260609.jsonl
```

Prepared reduced input after also excluding confirmed `C39`:

```text
recipes/aceproof-tts/dataset/repeat10-newckpt-remaining7-after-c39-20260610.jsonl
```

N39 repair-seed input:

```text
recipes/aceproof-tts/dataset/repeat10-n39-51-repairseed-20260610.jsonl
```

N39 explicit audit-feedback input:

```text
recipes/aceproof-tts/dataset/repeat10-n39-51-auditfeedback-20260610.jsonl
```

Endpoint/model:

```text
base_url: http://aiapps-053026.dyn.nvidia.com:28000/v1
model: ultra
context: endpoint default, expected 256k
```

Important request detail: configs intentionally omit `tokens_to_generate` / `max_completion_tokens`, so the endpoint default budget is used.

The finalized first ablation used 14 arms:

- Temperatures: `temp08` and `temp10`.
- Prompt families: `case`, `claimcert`, `compactaudit`, `formalsanity`, `lemma`, `proofonly`, `routecompare`.
- R1 only: generation plus verification. Refinement settings exist in configs but were not exercised in these R1-only runs.

The reduced remaining-8 rerun used only the two prompt families that looked most productive or least noisy in the first ablation:

- `proofonly`
- `formalsanity`

The N39 follow-up used two repair-style prompt streams:

- `repairseed`: previous candidate proof plus a coarse audit note about likely gaps, with instructions to reprove everything from the original problem.
- `repairfeedback`: a more explicit problem-specific audit-feedback prompt naming the recurring base-case failures and missing subcases.

## Method And Prompt Details

The repeat10 workstream reused the same AceProof R1 candidate-generation and normal-verification logic, but ran locally against the cloud multiplexer instead of submitting Slurm inference jobs. Configs set `cluster: none`, `gateway_address: http://aiapps-053026.dyn.nvidia.com:28000/v1`, and `profiles: ultra`. They intentionally did not set `tokens_to_generate` or `max_completion_tokens`, so the endpoint's 256k default context budget controlled completions.

R1 ablation structure:

```text
10 target problem rows
  x 2 generation temperatures: temp08 and temp10
  x 7 proof-generation prompt families
  x 64 independent candidate attempts per arm
  -> normal verifier prompt on completed candidates
  -> streaming early-stop verifier policy
  -> candidate bundle extraction and out-of-band audit
```

Reduced remaining-8 rerun structure:

```text
8 remaining target problem rows
  x 2 generation temperatures: temp08 and temp10
  x 2 prompt families: proofonly and formalsanity
  x 64 independent candidate attempts per problem/arm
  -> normal verifier prompt on completed candidates
  -> streaming early-stop verifier policy
  -> candidate bundle extraction and out-of-band audit
```

N39 repair-stream structure:

```text
1 N39 row
  x 2 generation temperatures: temp08 and temp10
  x 64 candidate attempts per arm
  x repairseed or repairfeedback prompt
  -> normal verifier prompt on completed candidates
  -> out-of-band audit of high-value candidates
```

Common mux config values were `n_parallel_proof_gen: 64`, `n_verification_per_proof: 16`, solved threshold `0.99999`, `top_p: 0.95`, request timeout `14400`, and `max_rounds: 1`. The streaming verifier used `min_verifications_per_proof: 4`, `early_stop_only_if_score_lt_1: true`, and `cancel_remaining: true`; this allowed obviously non-perfect candidates to stop after enough evidence while still preserving candidate rows for later audit.

For `temp08`, generation/refinement temperature was `0.8` and verifier temperature was `1.0`. For `temp10`, generation/refinement/verifier temperature was `1.0`.

Temperature-effect conclusion:

| Temperature arm | Confirmed distinct solves | Confirmed proof rows | Notes |
| --- | --- | --- | --- |
| `temp08` | `N24`, `N39` | 3 | `N24` came from `temp08/proofonly`; `N39` had valid repairseed/audit-feedback variants. |
| `temp10` | `N14`, `C39`, `N39` | 5 | `N14` came from `temp10/formalsanity`; `C39` came from `temp10/proofonly`; `N39` also had valid repairseed/audit-feedback variants. |

This does not establish that either temperature is globally better. The sample is small, the prompt families differed, and `N39` contributes multiple rows for the same problem. The practical conclusion is to keep `temperature=1.0` as the default because it is the model-recommended setting and produced more of the repeat10 confirmed wins, while keeping `0.8` as a diversity arm for high-value prompt families. Prompt family choice and candidate-level audit were more important than temperature in this workstream.

Prompt-family contracts:

| Family | Prompt file | Prompt contract / structure |
| --- | --- | --- |
| `proofonly` | `proof_generation_proof_only.yaml` | Writes only `## Solution`; removes self-evaluation, score, search diary, and speculative alternatives. If incomplete, it asks for the exact remaining gap. Produced confirmed `N24` and `C39` candidates. |
| `formalsanity` | `proof_generation_formal_sanity.yaml` | Tracks exact quantifiers and original hypotheses, forbids unproved generic-position assumptions, and adds a compact `## Sanity check` plus self-evaluation. Produced confirmed `N14`. |
| `case` | `proof_generation_case_construction.yaml` | Forces exhaustive case or construction coverage, direct property checks for constructions, explicit contradictions for eliminated cases, and self-evaluation of case completeness. |
| `claimcert` | `proof_generation_claim_certificate.yaml` | Uses a short claim/proof dependency chain, requiring each normalization, transformation, extremal choice, induction step, or bound to be proved before use. |
| `compactaudit` | `proof_generation_compact_audit.yaml` | Asks for one direct route plus detailed self-audit of hidden gaps, with explicit instructions to remove speculative alternatives and report unresolved issues faithfully. |
| `lemma` | `proof_generation_lemma_first.yaml` | Starts from small intermediate lemmas, proves them completely, and then assembles the final solution; the self-evaluation audits each lemma and assembly step. |
| `routecompare` | `proof_generation_route_compare.yaml` | Internally compares multiple routes, presents only the selected route, and adds `## Route check` explaining why the critical steps are justified. |
| `repairseed` | `proof_generation_repair_seed.yaml` | Treats a prior candidate and audit note as non-authoritative scratch work. Reuses ideas only after reproving all lemmas/cases, and explicitly checks finite cases, theorem exceptions, divisibility, sign/order choices, and induction hypotheses. Produced confirmed `N39` candidates. |

Normal verification used `proof_verification.yaml`: the same 0/0.5/1 proof-quality rubric as the baseline harness, requiring a detailed evaluation followed by `\boxed{...}`. The repeat10 workstream did not use strict/case-check verifier prompts for automatic promotion; independent/manual audits were used out-of-band to decide which high- or promising-low-verifier candidates were real solves.

## Confirmed Solves

### `N14`

Problem:

```text
Is it possible to find p,q,r in Q such that p+q+r=0 and pqr=1?
```

Artifact:

```text
outputs/repeat10-ultra-mp-r1-ablation/confirmed_solutions_so_far.jsonl
```

Candidate metadata:

- proof id: `N14_62`
- arm: `temp10/formalsanity`
- prompt family: `proof_generation_formal_sanity.yaml`
- temperature family: `temp10`
- normal verifier: `1/1`, `meanscore=1.0`
- audit status: `confirmed_real_solve_out_of_band`

What worked:

The `formalsanity` prompt asked the model to track quantifiers, avoid strengthening hypotheses, and explicitly check sign/divisibility cases. The successful proof reduced the rational system to pairwise-coprime integer factors whose product is a cube up to sign, then reduced to an impossible nonzero integer solution of the exponent-3 Fermat/Euler equation.

Notes:

This was also the only valid pipeline-promoted solve in the finalized 14-arm ablation. The final arm `temp10/formalsanity` promoted `N14` with `meanscore=1.0`.

### `N24`

Problem:

```text
Let (a_n) be a sequence of positive integers such that any prime larger than 1402 divides some term of the sequence and let b_n=a_1...a_n-1. Prove that infinitely many primes divide at least one term of (b_n).
```

Artifact:

```text
outputs/repeat10-ultra-mp-r1-ablation/confirmed_solutions_so_far.jsonl
```

Candidate metadata:

- proof id: `N24_6`
- arm: `temp08/proofonly`
- prompt family: `proof_generation_proof_only.yaml`
- temperature family: `temp08`
- normal verifier rows for this candidate: `0`
- audit status: `confirmed_real_solve_out_of_band`

What worked:

The `proofonly` prompt stripped away self-evaluation and asked for a concise final proof or a named exact gap. The successful proof used a finite-prime contradiction: assume only finitely many primes divide the `b_n`; use CRT, Dirichlet, and quadratic reciprocity to choose a prime `q > 1402`, `q = 3 mod 4`, such that all primes in the finite set are quadratic residues mod `q`. Since `q` divides some `a_m`, we get `b_m == -1 mod q`; but `b_m` factors over the finite set, so it is a quadratic residue mod `q`, contradiction because `-1` is a nonresidue for `q = 3 mod 4`.

Notes:

This is the clearest example that candidate inspection mattered. The normal finalizer did not promote the valid `N24_6`; instead, `temp08/formalsanity` promoted a different `N24_27` row that audit rejected as invalid. Without out-of-band candidate audit, this real solve would have been missed and a false solve could have been recorded.

### `C39`

Problem:

```text
Let p>3 be a prime and color each of 1,...,p-1 red, blue, or green. All three colors are used. Prove there are x,y,z with pairwise distinct colors such that x+y == z mod p.
```

Artifact:

```text
outputs/repeat10-ultra-mp-r1-remaining8-rerun1/confirmed_solutions_so_far.jsonl
```

Candidate metadata:

- proof id: `C39_7`
- arm: `temp10/proofonly`
- prompt family: `proof_generation_proof_only.yaml`
- temperature family: `temp10`
- normal verifier scores on the confirmed row: `[1.0, 0.5, 0.5, 0.5]`
- normal verifier mean on the confirmed row: `0.625`
- later stronger C39 candidates in the completed remaining8 run: top `C39_51`, `11/12` verifier ones, mean `0.9167`; `C39_8`, `9/10`, mean `0.9`
- audit status: `valid`

What worked:

The successful proof came from `proofonly` at temperature `1.0`. It assumes no rainbow solution, derives pairwise sumset containments such as `R+B subset R union B union {0}`, adjoins zero to color classes, and applies a Vosper/equality inverse-Cauchy-Davenport classification to force arithmetic-progression structure. It then uses the resulting interval/one-sided structure and singleton color cases to reach a contradiction.

Notes:

Audit marked the proof valid assuming the stated Vosper equality theorem is allowed. The first confirmed candidate would not have been promoted by the normal `meanscore >= 0.99999` pipeline threshold. It was found by ranking intermediate candidates and auditing a low-mean but structurally promising proof.

### `N39`

Problem:

```text
Let (a_n)_{n>=1} be a sequence of positive integers such that for all m,n>=1, all prime factors of a_m+a_n are among the prime factors of m+n. Prove that a_n=n for all n.
```

Primary artifact:

```text
outputs/repeat10-ultra-mp-n39repairseed/confirmed_solutions_so_far.jsonl
```

Generic repairseed confirmed candidates:

| Proof id | Arm | Verifier scores | Audit status |
| --- | --- | --- | --- |
| `N39_9` | `temp08/repairseed` | `[1.0, 1.0, 0.5, 0.5]`, mean `0.75` | valid |
| `N39_51` | `temp10/repairseed` | `[1.0, 0.5, 1.0, 1.0]`, mean `0.875` | valid |

What worked:

The generic `repairseed` stream used a previous promising `N39_51` candidate plus a coarse audit note, but did not provide the exact explicit modular repairs later used in `repairfeedback`. It instructed the model to treat the seed as non-authoritative scratch work and to reprove all steps from the original problem.

The successful proof family establishes the base values and then runs a Bertrand/Zsigmondy induction. The key repairs compared with earlier invalid candidates were:

- handling `(a_1,a_2)=(8,1)` rigorously;
- splitting `B=1`, `B=2`, and `B>=3` for the `5^B-1` branch;
- using primitive prime divisors of `5^B-1` correctly to rule out extra prime factors beyond `2,3`;
- eliminating the `a_3=24` branch at `n=4`, including the `a_4=1` case;
- checking Zsigmondy exceptions and auxiliary induction indices.

Seed 9 repairs the recurring `(8,1)` gap by eliminating that branch at `n=4` without needing uniqueness for `2^A=5^B+7`. Seed 51 is cleaner: it rules out `(8,1)` using `a_1=8,a_2=1` and `n=4`, then handles the `p^e-1` Zsigmondy exception in the induction with a `p<2n-1` / `p=2n-1` split.

Explicit audit-feedback variants also solved N39:

```text
outputs/repeat10-ultra-mp-n39auditfeedback/confirmed_solutions_so_far.jsonl
```

| Proof id | Arm | Verifier scores | Audit status |
| --- | --- | --- | --- |
| `N39_35` | `temp08/repairfeedback` | `[0.5, 0.5, 0.5, 0.5]`, mean `0.5` | valid |
| `N39_51` | `temp10/repairfeedback` | `[0.5, 1.0, 1.0, 0.5]`, mean `0.75` | valid |
| `N39_57` | `temp10/repairfeedback` | `[1.0, 0.5, 1.0, 0.5]`, mean `0.75` | valid with minor wording cleanup |

These are useful evidence that the route is robust, but the generic repairseed solves are the more important harness result because they did not rely on the explicit problem-specific repair checklist.

## Current Target Status

| Problem | User-provided circumstance | Current repeat10 status | Notes |
| --- | --- | --- | --- |
| `C39` | pipeline-unsolved | Confirmed solved | `temp10/proofonly`, audited valid with Vosper caveat. Later C39 candidates had stronger verifier support. |
| `N1` | pipeline-unsolved | Not solved | `N1_63` was invalid: it proves set equality but then uses the stronger unsupported claim `a_i=i`. Final top still `N1_63`; no new credible candidate. |
| `N14` | pipeline-unsolved | Confirmed solved | `temp10/formalsanity`, audited valid. |
| `N24` | pipeline-unsolved | Confirmed solved | `temp08/proofonly`, valid but not promoted by normal verifier. |
| `N39` | pipeline-unsolved | Confirmed solved | Two valid generic repairseed proofs plus three explicit audit-feedback variants. |
| `proofbench133_028` | pipeline-unsolved | Not solved | Completed remaining8 top candidates are all `0.5`-style verifier support, no ones. |
| `proofbench133_109` | gpt-rejected | Not solved; many false positives | Repeated Gaussian-integer candidates miss unit/sign branches. |
| `C50` | pipeline-unsolved | Not solved | Completed remaining8 top mean `0.375`, no verifier ones. |
| `proofbench133_030` | pipeline-unsolved | Not solved | Game/minimax candidates had unsupported strategy invariants; `030_38` was audited invalid despite `3/4` verifier ones. |
| `G25` | gpt-rejected | Not solved; verifier false-positive hotspot | Many candidates got perfect verifier scores, but they use invalid Pitot/Ptolemy reasoning. |

## Completed Run Totals

Final remaining8 rerun totals:

```text
outputs/repeat10-ultra-mp-r1-remaining8-rerun1
proof rows: 1,864
verified proof groups: 1,575
verifier rows: 6,442
outer problem-arm rows: 32/32 completed
```

Final N39 repairseed totals:

```text
outputs/repeat10-ultra-mp-n39repairseed
proof rows: 128
verifier rows: 488
outer rows: 2/2 completed
confirmed valid proof rows: 2
```

Final N39 auditfeedback totals:

```text
outputs/repeat10-ultra-mp-n39auditfeedback
proof rows: 128
verifier rows: 500
outer rows: 2/2 completed
confirmed valid proof rows: 3
```

Combined confirmed-solution handoff:

```text
outputs/repeat10-confirmed-solutions-combined.jsonl
```

This file currently has 8 confirmed proof rows over 4 distinct problems: `C39`, `N14`, `N24`, and `N39`.

Combined potential-candidate handoff:

```text
outputs/repeat10-potential-candidates-combined-final.jsonl
```

This file has 634 triage rows across remaining8, generic N39 repairseed, and N39 audit-feedback. It intentionally includes known verifier false positives; use the audit notes before treating any row as solved.

Remaining8 final artifacts:

```text
outputs/repeat10-ultra-mp-r1-remaining8-rerun1/all_proofs_with_verification_final.jsonl
outputs/repeat10-ultra-mp-r1-remaining8-rerun1/potential_candidates_final.jsonl
outputs/repeat10-ultra-mp-r1-remaining8-rerun1/per_problem_summary_final.json
```

Line counts:

```text
outputs/repeat10-confirmed-solutions-combined.jsonl                         8
outputs/repeat10-potential-candidates-combined-final.jsonl                634
outputs/repeat10-ultra-mp-r1-remaining8-rerun1/all_proofs_with_verification_final.jsonl  1864
outputs/repeat10-ultra-mp-r1-remaining8-rerun1/potential_candidates_final.jsonl           500
outputs/repeat10-ultra-mp-n39repairseed/confirmed_solutions_so_far.jsonl       2
outputs/repeat10-ultra-mp-n39auditfeedback/confirmed_solutions_so_far.jsonl    3
```

## Comparison To Finished New-Checkpoint Runs

The fully finalized 14-arm multiplexer ablation ended with these finalizer-level promotions:

| Arm | Pipeline-promoted solve ids | Audit result |
| --- | --- | --- |
| `temp10/formalsanity` | `N14` | valid |
| `temp08/formalsanity` | `N24` | invalid promoted row (`N24_27`) |
| all other finalized arms | none | no finalizer solve |

This understated and overstated quality in different ways:

- It understated quality because `N24_6` was a real solve from `temp08/proofonly` but had no verifier rows and was not promoted.
- It overstated quality because `N24_27` was pipeline-promoted but invalid.
- The reduced remaining8 rerun found `C39`, but the first audited valid row had only mean `0.625`.
- The N39 repairseed stream found valid proofs whose verifier scores were mixed rather than threshold-perfect.

Relative to the new checkpoint's original setup, this harness workstream adds confirmed audited solves on `N14`, `N24`, `C39`, and `N39`.

## What Worked Well

Prompt family diversity helped, but only a subset was actually productive in this endpoint run:

- `formalsanity` produced the confirmed `N14` solve.
- `proofonly` produced the confirmed `N24` and `C39` solves.
- `repairseed` produced the confirmed `N39` solves.
- Both `temp08` and `temp10` produced useful candidates. There is no clean evidence from this small sample that `0.8` dominates `1.0` or vice versa; use `1.0` as the default and keep `0.8` as a diversity arm.

Candidate-level auditing was essential:

- `N24_6` would have been missed by finalizer-only analysis.
- `C39_7` had only mean `0.625`, so it would have been discarded by solved-threshold promotion, but audit found it valid.
- `N39_35` was independently valid despite normal verifier scores `[0.5, 0.5, 0.5, 0.5]`.
- Several high-verifier candidates were invalid, so high normal-verifier score alone was not enough.

Using no explicit token cap was fine for these runs:

- Configs omitted `tokens_to_generate` / `max_completion_tokens`.
- Requests used the endpoint's default context budget rather than forcing per-request generation caps.

The reduced rerun strategy was useful:

- After `N14` and `N24` were confirmed, rerunning only the remaining 8 with the two most useful prompt families found `C39`.
- After `C39`, the targeted N39 repair stream found multiple valid N39 variants.

## What Did Not Work / Failure Modes

Endpoint stability affected the first 14-arm run:

- The multiplexer became unhealthy during the first launch. Several arms finalized as zero-proof timeout artifacts.
- Zero-proof arms should not be interpreted as evidence that those prompts are bad.
- Later reduced and N39-specific runs completed successfully after endpoint recovery.

Normal verification produced serious false positives:

- `G25`: many candidates reached perfect or near-perfect verifier scores. The common flaw is treating Pitot side-sum equality as sufficient for a convex quadrilateral to be tangential, or otherwise misusing Ptolemy/Pitot converses. Representative `G25_30` was audited invalid; the same pattern appears broadly.
- `proofbench133_109`: multiple high-score candidates had Gaussian-integer unit/sign branch failures. After absorbing a Gaussian unit, they assume cube-root coordinates can be taken positive and then treat signed factors as positive squares. `proofbench133_109_6` and earlier `109` candidates were audited invalid; `109_2` spot-inspection shows the same unsupported normalization.
- `proofbench133_030`: high-scoring grid-game candidates did not prove the minimax invariant. `proofbench133_030_38` reached `3/4` verifier ones but was audited invalid: a `3 x 2` rectangle with a one-cell tail attached to the middle of a long side breaks the claimed rectangle-plus-tail invariant.
- `N1_63`: invalid induction; it proves only unordered set equality for first prime-index blocks but then uses pointwise equality.
- `C39_26`: invalid Combinatorial Nullstellensatz proof; it used ordinary coefficient vanishing where only the reduced remainder modulo grid polynomials is controlled.
- `N39_60`: invalid induction proof; the `a_n > n` case had a fatal divisibility arithmetic error.

The self-evaluation text inside generations was not enough for promotion:

- Some invalid candidates self-scored or presented as complete.
- The useful signal was mostly in candidate structure plus independent audit, not in the model's own boxed self-score.

## Repair Seeds And Candidate Lessons

### `N39`

Status: solved by repairseed.

The earlier report treated `N39_51` only as a promising repair seed. The completed run changed that conclusion. Generic repairseed produced two valid audited proofs (`N39_9` and `N39_51`), and explicit auditfeedback produced three more valid or valid-with-minor-wording-fix variants.

The route is now one of the strongest positive findings from repeat10. It also illustrates that standard verifier majority is not sufficient: the valid candidates did not reach a near-perfect verifier ratio, while unrelated false-positive families did.

### `proofbench133_109`

Status: repair direction still unclear.

The repeated route through Gaussian integers may still be viable, but every high-scoring candidate so far misses a signed branch. Any future prompt should explicitly force a complete unit/sign case split or an absolute-value formulation that proves all branches.

### `proofbench133_030`

Status: verifier hotspot, not a confirmed solve.

The repeated route through a rectangle/tail game invariant is not currently rigorous. Future prompts should either ask for a formal invariant with exhaustive responses to every Shayan move, or switch to a different game-theoretic proof route. The candidate arithmetic alone is not enough.

### `G25`

Status: verifier hotspot, not a good promotion candidate without a targeted geometry gate.

The current normal verifier repeatedly accepts invalid Pitot-converse arguments. Future verifier sidecars for this problem should explicitly ask whether side-sum equality is being used as a sufficient tangency condition, and should reject proofs that do not prove the actual tangential condition.

## Artifacts

Local working-tree artifacts from the completed repeat10 run:

```text
outputs/repeat10-confirmed-solutions-combined.jsonl
outputs/repeat10-potential-candidates-combined-final.jsonl
outputs/repeat10-ultra-mp-r1-remaining8-rerun1/all_proofs_with_verification_final.jsonl
outputs/repeat10-ultra-mp-r1-remaining8-rerun1/potential_candidates_final.jsonl
outputs/repeat10-ultra-mp-r1-remaining8-rerun1/per_problem_summary_final.json
outputs/repeat10-ultra-mp-n39repairseed/confirmed_solutions_so_far.jsonl
outputs/repeat10-ultra-mp-n39auditfeedback/confirmed_solutions_so_far.jsonl
```

Earlier cluster copy, with files/directories set to `a+rX`:

```text
/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/igitman/aceproof-share/20260610-aceproof-tts/repeat10/
```

That cluster copy was created before the final N39/remaining8 tails completed, so prefer the local working-tree artifacts listed above for final repeat10 conclusions unless the cluster copy is refreshed.

Run notes:

```text
recipes/aceproof-tts/configs/repeat10-mp-ablation/README.md
```

Configs:

```text
recipes/aceproof-tts/configs/repeat10-mp-ablation/
recipes/aceproof-tts/configs/repeat10-mp-repair/
```

Prompts that produced confirmed solves:

```text
recipes/aceproof-tts/prompts/proof_generation_formal_sanity.yaml
recipes/aceproof-tts/prompts/proof_generation_proof_only.yaml
recipes/aceproof-tts/prompts/proof_generation_repair_seed.yaml
```

## Original Broad Open53 Run

The earlier, broader AWS-DFW workstream used a 53-row sanitized open set built from the remaining ProofBench rows plus the GEMS unsolved file, with solution fields removed before use.

Primary input artifacts in the original AceProof repo:

```text
recipes/aceproof-tts/dataset/gems-proofbench-unsolved-20260529-sanitized-nosolutions.jsonl
recipes/aceproof-tts/dataset/gems-proofbench-unsolved-20260529-open53-nosolutions.jsonl
recipes/aceproof-tts/dataset/future-open53-proofbench-gems-after024-20260530-nosolutions.jsonl
```

The broad run launched many R1 prompt variants on AWS-DFW FP4/long-context serving. The main families were `gapresistant`, `dualbounds`, `formalsanity`, `proofonly`, `equivguard`, `obligationselfcritique`, `routecompare`, `counterguard`, `decompose`, `claimcert`, `lemma`, `briefaudit`, `case`, `compactaudit`, `obligation`, and `invariant`.

Because several verifier tails became extremely long, the run used a finish-only policy: active chunks were allowed to write until they flattened or hit walltime; stale non-growing tails were treated as incomplete, partial async verifier rows were preserved, and finalization was repaired with clean dependency chains. The resulting branch-level solved counts should be read as candidate-discovery evidence, not as fully independent proof of correctness.

After finalization, high-scoring candidates were extracted and audited out-of-band:

```text
recipes/aceproof-tts/audits/open53-potential-solutions-20260604.jsonl
recipes/aceproof-tts/audits/open53-potential-solutions-with-subagent-audit-20260604.jsonl
recipes/aceproof-tts/audits/open53-candidate-audit-merged-20260604.jsonl
recipes/aceproof-tts/audits/open53-subagent-real-solved-solutions-20260605.jsonl
```

Audit extraction scope: branch-final candidates with `meanscore >= 0.95` across 16 finalized open53 R1 prompt variants. This produced 263 candidate rows over 30 problem IDs. Five independent high-reasoning audit workers reviewed disjoint problem-ID subsets using only the problem statement and candidate proof.

Problem-level audit result:

- `27` real solves.
- `1` likely solve: `N49`, requiring a tighter final bound check.
- `2` no-solve false positives: `C19` and `G10`.

Audited real-solve IDs from the broad run:

```text
A32, A48, C39, C4, C49, C5, C50, C8, G11, G12, G16, G25, G34,
N1, N10, N13, N14, N22, N24, N39,
proofbench133_026, proofbench133_028, proofbench133_030,
proofbench133_040, proofbench133_075, proofbench133_109,
proofbench133_130
```

This broad run is the source of the “previously successful ideas” that motivated the smaller repeat10 multiplexer rerun. Compared with the broad open53 result, the repeat10 multiplexer/new-checkpoint branch is much narrower: it tries to reproduce or transfer success on only 10 selected rows through the cloud endpoint and now has confirmed audited repeat10 gains on `N14`, `N24`, `C39`, `N39`, and `N1`.

## Fixed-Harness Promotion Update

The continuation work after the initial report tested whether manual candidate selection could be replaced by fixed, reproducible promotion gates.

Recommended high-precision rule:

1. Require a nonempty, complete proof row.
2. Apply `recipes/aceproof-tts/pipeline/run_static_promotion_filters.py`; any `static_promotion_reject=true` is a hard no-promote.
3. Require completed required verifier sidecars and unanimity: any `0`, `0.5`, malformed output, missing required family, or incomplete sidecar means no promote.
4. Use problem-family gates:
   - normal strict/theorem/congruence verifier families for number-theory/divisibility proofs;
   - Pitot/false-converse static gate plus false-converse verifier for G25-style geometry proofs;
   - Gaussian cube sign/sector static gate plus Gaussian sign verifier for `proofbench133_109`;
   - theorem/countermodel verifier unanimity for `proofbench133_030` game-invariant proofs.

Observed fit on current artifacts:

- `N1:89:repairseed/rounds:ebcd6c24a6fc47cc` passes static filtering and has unanimous sidecar support across the located strict/theorem/congruence/Gaussian-style rows. This is the strongest fixed-harness promotion success so far.
- Current G25 candidate families are rejected by the Pitot/opposite-side-sum static gate even when generic verifiers score them highly.
- Top `proofbench133_109` candidates are rejected by the Gaussian cube sign/sector gate; independent audit confirmed the fatal gap is exactly missing unit/sector/sign branches after the Gaussian cube reduction.
- Top older `proofbench133_030` candidates fail adversarial audit because they do not cover Shayan filling the bay after Ali creates an L-shape. The newer repairseed candidates `proofbench133_030:86:repairseed:0b1569a1cdd4ea76` and `proofbench133_030:7:repairseed:b179525779e35400` are also invalid; seed 7 is closer but still only proves bay persistence when Shayan plays `k=1`, not when Shayan fills the initial bay with `k=2`. A narrow static gate now blocks the clearest “Shayan always plays k=1” bay-strategy false pattern, but promotion should still require adversarial verifier unanimity.

Endpoint robustness note: the local multiplexer sidecar runs can die if a single early request receives an nginx/OpenAI-compatible `500`. For the active local run I patched the staged NeMo-Skills `BaseModel.generate_async` path to retry transient OpenAI-compatible 429/500/502/503/504/connection/timeout errors with exponential backoff, and relaunched `repeat10-current3-slow-focus-retry2` at lower concurrency. This is an operational robustness change, not a prompt or selection change.

## Bottom Line

The repeat10/new-checkpoint target set now has confirmed audited gains on `N14`, `N24`, `C39`, `N39`, and `N1`. The strongest new continuation update is `N1`: generic repairseed produced a valid route that a fixed high-precision verifier ensemble can also promote.

The most reliable pattern is prompt-family diversity plus candidate-level audit, with repairseed as the most promising next-stage mechanism when a near-solve has a localized flaw. The biggest caution is verifier reliability: `G25`, `proofbench133_030`, and `proofbench133_109` show that normal verifier ratios can be confidently wrong on recurring proof-pattern failures. Fixed promotion should combine verifier unanimity with narrow static blockers for known false-positive proof families.


## Continuation: Fixed Harness And Remaining Problems, 2026-06-11 00:45 PDT

Current confirmed repeat10/new-checkpoint solves remain `N14`, `N24`, `C39`, `N39`, and `N1`. No new valid solve has been confirmed in this continuation window.

New fixed-harness artifacts added or staged:

```text
recipes/aceproof-tts/pipeline/evaluate_fixed_promotion.py
recipes/aceproof-tts/prompts/proof_generation_self_contained_repair.yaml
recipes/aceproof-tts/dataset/repeat10-028-c50-legacy-attempt-memory-20260611.jsonl
recipes/aceproof-tts/dataset/repeat10-newckpt-current-unsolved2-028-c50-20260611.jsonl
recipes/aceproof-tts/scripts/launch_repeat10_genonly_v5_109_030_verifiers_tmux.sh
recipes/aceproof-tts/scripts/launch_repeat10_legacy_028_c50_verifiers_tmux.sh
recipes/aceproof-tts/scripts/launch_repeat10_current2_028_c50_focus_tmux.sh
recipes/aceproof-tts/scripts/launch_repeat10_028_c50_attempt_memory_repair_tmux.sh
```

`evaluate_fixed_promotion.py` is the executable version of the high-precision promotion rule: static filter must pass, every required verifier family must have enough completed valid rows, and every required verifier score must be exactly `1.0`. Missing sidecar rows, incomplete rows, malformed scores, `0`, or `0.5` all reject. A dry run on the incomplete live v5 sidecars correctly promoted zero candidates.

The deterministic static filter was tightened for `proofbench133_030`: any proof that bases Shayan's strategy on the recurring “Shayan can/should always play `k=1`” family is now no-promote unless a separate adversarial verifier process is explicitly used. This catches all current repairseed `030` rows seen so far and does not reject the previously confirmed non-`030` solutions in the local confirmation files.

Candidate/audit status by remaining problem:

- `proofbench133_109`: current live filtered bundle contains 26 non-static-rejected candidates, mostly repairseed variants. Prior high-value Gaussian-cube candidates were invalid due missing unit/sector/sign branches. A v5 verifier sidecar over 25 earlier `109` survivors plus 4 `030` rows was launched, but as of this note it has zero completed verifier rows because the multiplexer is returning nginx/OpenAI-compatible `500` responses.
  A fresh audit of seven fair-promptwide static survivors also found no valid solution. The recurring failures were broader than the regex static gate: wrong Gaussian unit middle/extreme case reductions, false dismissal of `u=-1`/`u=-i` branches, a false `gcd(x, x^2-3y^2)=1` step that ignores a common divisor `3`, and one false modulo-16 odd-cube exclusion. These should be handled by the Gaussian-sign/adversarial verifier family rather than by trusting normal verifier ratios.
- `proofbench133_030`: live repairseed now has several perimeter/adjacency candidates, all in the same always-`k=1` strategy family. Earlier independent audits found seeds 7 and 86 invalid; the strengthened static gate rejects the current family for promotion.
- `proofbench133_028`: four legacy candidates were independently audited. None is valid as written; three have a plausible induction skeleton but fail to prove the base cases such as `f(1)=1` and `f(2)=2` fully. The staged attempt-memory repair input uses only model-generated candidates plus normal verifier comments, not the independent audit notes.
- `C50`: four legacy candidates were independently audited. None is a standalone proof; the best ones reduce the problem to a black-box additive-combinatorics inequality of the form `|A+A| >= c |A-A|^{3/5}|A|^{2/5}` but do not prove or adequately justify the lemma. The self-contained repair prompt asks the model to prove any invoked black-box lemma or mark the proof incomplete.
- `G25`: current rows remain a false-positive hotspot and are filtered by the Pitot/opposite-side-sum static gate. The one v6 filtered survivor is a decomposition subproblem algebra row, not a final proof candidate.

Active local multiplexer sessions launched in this continuation:

```text
repeat10-genonly-v5-109-030-verifiers
repeat10-legacy-028-c50-verifiers
repeat10-current2-028-c50-focus
```

They were launched at low concurrency with transient-error retries. At the latest health check, the multiplexer reported all worker slots busy, zero pending jobs, and roughly 22.9k jobs in flight; new focused runs were still at zero completed rows and retrying transient `500` errors. The redundant `repeat10-current3-slow-focus-retry2` session was stopped after many synchronized retries and zero rows, because its scope is covered by the smaller targeted sessions above.

Next queued experiment once the endpoint accepts work again: launch `repeat10-028-c50-attempt-memory-repair` from `launch_repeat10_028_c50_attempt_memory_repair_tmux.sh`, using `proof_generation_self_contained_repair.yaml` over `repeat10-028-c50-legacy-attempt-memory-20260611.jsonl`. This is intended to test a fair fixed-harness repair loop: previous model attempts plus model verifier comments, with no independent audit text fed back to the model.

## Continuation: Endpoint Recovery And C50 Verifier False Positive, 2026-06-11 09:35 PDT

The nginx/OpenAI-compatible `500` failures above were observed overnight around `2026-06-11 00:50-00:53 PDT`. A live check at `2026-06-11 09:12 PDT` showed `/v1/models` returning `200` and the health endpoint reporting healthy `ultra` workers. A tiny chat-completions probe no longer returned an immediate nginx `500`; it timed out after 45 seconds with no bytes, consistent with queue pressure rather than the earlier front-door failure mode.

The zero-row streams that failed during the overnight `500` window were relaunched:

```text
repeat10-028-c50-attempt-memory-repair
repeat10-current2-028-c50-focus
repeat10-genonly-v5-109-030-verifiers
repeat10-legacy-028-c50-verifiers
```

An endpoint-failure monitor was added and started in tmux:

```text
recipes/aceproof-tts/scripts/monitor_repeat10_endpoint_failures.sh
tmux session: repeat10-endpoint-failure-monitor
```

It scans recently modified local logs for fresh nginx/LiteLLM `500` signatures and calls `/home/igitman/.claude/bin/notify-slack` if a new signature appears. As of this update, it has not seen a fresh endpoint-failure signature after the relaunch.

The relaunched jobs are progressing through async chunk files rather than final `output.jsonl` files. The most informative current counts are partial:

```text
attempt-memory repair: 75 async proof-gen rows
current2 028/C50 proofonly/obligation/invariantextremal/decompose: 5/8/1/3 async proof-gen rows
v5 109/030 strict/theorem/congruence/Gaussian-sign: 221/227/224/6 async verifier rows
legacy 028/C50 strict/theorem/congruence: 62 async / 64 final / 63 async verifier rows
```

The partial legacy fixed-promotion check without the self-contained gate promoted one `C50` candidate:

```text
C50:noseed:repeat10-ultra-mp-r1-remaining8-rerun1/temp10:f5b929ae10f567ff
```

That candidate had unanimous `1.0` sidecar support from the generic strict, theorem-gate, and congruence families, but manual inspection showed it is not a standalone proof. It relies on a black-box Garaev additive-combinatorics lemma and labels the key proof as a sketch. The candidate's own self-evaluation also says the omitted lemma proof is a material gap.

This exposed a concrete verifier false-positive mode: `proof_verification_theorem_gate.yaml` permits “standard theorem whose hypotheses exactly match the problem,” so it can still accept a research-level or contest-nonstandard named theorem if the model treats it as standard.

To repair this fixed-harness path, a new verifier prompt was added:

```text
recipes/aceproof-tts/prompts/proof_verification_self_contained_theorem_use.yaml
```

This gate requires every nontrivial named theorem, cited result, black-box inequality, classification, or proof-sketch dependency to be fully proved in the candidate. Routine elementary facts remain allowed. A partial run of this gate immediately scored the promoted `C50` candidate as `0.0`, explicitly rejecting the unproved Garaev lemma and crossing-number sketch. Re-running fixed promotion with this self-contained family included reduced the partial legacy `028/C50` promotion count from `1/8` to `0/8`.

## Continuation: Endpoint Working But Loaded, 2026-06-11 10:50 PDT

Live endpoint status improved compared with the overnight nginx `500` failure mode. At `2026-06-11 10:37 PDT`, the `ultra` queue reported `ok=True`, `39/39` workers alive, `8568/9984` slots busy, `11` pending jobs, and `7976` in-flight jobs. Recent local logs had no fresh nginx `500` or LiteLLM endpoint-error signatures. A tiny direct chat-completions probe did not return an immediate `500`; it timed out after 60 seconds with no bytes. The practical interpretation is that the endpoint is working and processing queued work, but remains heavily loaded and not responsive to short synchronous probes.

The relaunched verifier streams are producing rows again. On the attempt-memory `028/C50` snapshot, all returned verifier rows are still hard rejections:

```text
snapshot78 strict/theorem/congruence/self-contained: 30/29/24/39 returned rows, all score 0
legacy 028/C50 self-contained theorem-use: 41 returned rows, all score 0
```

The v5 `109/030` fixed-promotion rerun with the latest strict/theorem/congruence/Gaussian-sign sidecars still promotes `0/29` candidates, even with `min_scores_per_family=1`. Some `proofbench133_109` rows receive Gaussian-sign `1.0` scores, but every such row is blocked by non-1.0 scores from strict, theorem, and/or congruence verifier families.

A new current2 high-self-eval `proofbench133_028` candidate exposed another generic verifier false positive. The candidate specializes the original condition at `k=2` as if the right-hand side were `x^2-y^2` before proving `f(2)=2`; this is circular because the true specialized condition is `f(x)+f(y) | x^{f(2)}-y^{f(2)}`. The generic strict verifier scored this candidate `1.0` and repeated the invalid specialization as correct.

A targeted substitution-integrity verifier prompt was added to catch this family:

```text
recipes/aceproof-tts/prompts/proof_verification_substitution_integrity.yaml
```

This verifier explicitly lists each specialization of the original condition and rejects proofs that replace unknown quantities such as `f(2)`, `f(3)`, or `f(n)` by their intended final values before those values are proved. The first returned substitution-integrity row already rejected a related `028` candidate for invalid divisibility manipulation; the specific strict-verifier false-positive row is still pending in that verifier stream.

## Continuation: Afternoon Fixed-Gate Status, 2026-06-11 14:03 PDT

The multiplexer endpoint is working again, though still loaded. A live health sample at `2026-06-11 13:57 PDT` reported `ok=True`, `42/42` ultra workers alive, `8261/10752` slots busy, `0` pending jobs, `8179` in-flight jobs, and `273750` completed jobs. A scan of locally modified logs from the prior 30 minutes found no fresh nginx `500`, LiteLLM connection, transient OpenAI-compatible endpoint, or `job not found` signatures.

The v5 `proofbench133_109` / `proofbench133_030` sidecars now have enough rows to rerun the fixed gate:

```text
strict: 237 rows
theorem: 235 final rows, including 3 literal null merge-artifact rows
congruence: 237 rows
Gaussian-sign: 225 rows
fixed promotion with min_scores_per_family=1: 0/29 promoted
```

The promotion loader now skips non-object JSONL rows so finalization artifacts such as literal `null` lines do not break monitoring. The best current `proofbench133_109` candidate has 26/32 complete verifier scores equal to `1.0`, but it is still rejected by non-1.0 scores in strict, theorem, congruence, and Gaussian-sign families. No v5 `109` or `030` candidate currently passes a fixed fair gate.

The attempt-memory `028/C50` snapshot also remains non-promoting:

```text
snapshot78 strict/theorem/congruence/self-contained rows: 77/77/76/75
score distribution: strict all 0; theorem 76 zero and 1 one; congruence 75 zero and 1 one; self-contained 74 zero and 1 half
fixed promotion with min_scores_per_family=1: 0/78 promoted
```

The attempt-memory generator has continued past the original snapshot. A delta input was created for 56 fresh candidates beyond the 78-row snapshot:

```text
outputs/repeat10-028-c50-attempt-memory-repair-verification/delta_after_snapshot78/input_candidates_20260611_1408.jsonl
```

A new tmux verifier session was launched over that delta with the same x1 sidecar families:

```text
repeat10-attempt-memory-delta56-verifiers
strict_audit_x1
theorem_gate_x1
congruence_guard_x1
selfcontained_theorem_use_x1
```

For `proofbench133_028`, the targeted false-positive row `proofbench133_028_49` is now fully diagnosed. The generic strict verifier scored it `1.0`, and the self-contained theorem-use verifier also scored it `1.0`; both missed the circular specialization from `x^{f(2)}-y^{f(2)}` to `x^2-y^2`. The substitution-integrity gate scored the exact same candidate `0.0` and identified the invalid premature substitution. This makes substitution-integrity a necessary fixed verifier sidecar for future `028` promotion attempts; self-contained theorem-use alone is not sufficient for this failure mode.

A second current-decomposition fixed-gate batch was also staged from the newer planner and decomposition-assembly rows, because those were not covered by the earlier v5 verifier bundle. Static filtering left 98 candidates: `C50` 26, `G25` 1, `proofbench133_028` 32, `proofbench133_030` 9, and `proofbench133_109` 30. The verifier input is:

```text
outputs/repeat10-genonly-candidate-verification/current_decomp_planner_assembly_verify_static64_x4_20260611.jsonl
```

It was launched in tmux session `repeat10-current-decomp-fixed-verifiers` with x4 strict, theorem, congruence, self-contained theorem-use, Gaussian-sign, and substitution-integrity sidecars.

## Continuation: Modular-Root False Positive, 2026-06-11 14:38 PDT

The current-decomposition verifier batch produced one early fixed-gate promotion under the existing sidecars:

```text
problem: proofbench133_109
candidate_uid: proofbench133_109:21:planner/rounds:4fbde754cbe373e7
source: outputs/repeat10-ultra-mp-current5-decomp-planner/temp10/planner/rounds/R1/proof_gen/output_chunk_0.jsonl-async
existing sidecar support at first check: strict 3/3, theorem 1/1, congruence 4/4, self-contained 3/3, Gaussian-sign 3/3, substitution 2/2 all score 1.0
```

Manual inspection found that this is not a valid solve. The candidate is a decomposition-planner output, and its odd-prime argument contains an invalid modular-root step: it goes from `a^4 == -b^4 (mod p)` to `a^2 == +/- b^2 (mod p)`. That implication is false; `(-b^2)^2 = b^4`, not `-b^4`. This is a concrete false-positive mode for the existing strict/congruence/Gaussian-sign sidecars.

A new general verifier prompt was added:

```text
recipes/aceproof-tts/prompts/proof_verification_modular_root_integrity.yaml
```

This gate audits root extraction, sign changes, cancellation, units, associates, and branch choices in modular algebra. It is general harness logic and does not contain a problem-specific hint. A one-row smoke test on the false-positive candidate and a broader x4 current-decomposition root-integrity sidecar were launched. Early broader root-integrity rows are returning mostly `0.0`, but they have not yet reached the exact false-positive candidate. Re-running fixed promotion with root-integrity required currently gives `0/98` promoted because no candidate has complete support from all required families including root-integrity.

## Continuation: Static Modular-Root Blocker, 2026-06-11 14:48 PDT

The modular-root verifier prompt did not reliably catch the `proofbench133_109:21:planner/rounds:4fbde754cbe373e7` false positive. The one-row root-integrity smoke completed after about 17 minutes and scored the candidate `1.0`; the broader root-integrity sidecar also returned `1.0` rows for the same candidate. The model verifier repeated the same invalid implication instead of rejecting it.

A deterministic static promotion filter was therefore added to `recipes/aceproof-tts/pipeline/run_static_promotion_filters.py`. It rejects `proofbench133_109` candidates that combine a negative fourth-power congruence such as `a^4 == -b^4 (mod p)` with an invalid root extraction of the form `a^2 == +/- b^2`. This is a general algebraic false-pattern blocker, not a prompt hint fed back to generation.

Validation on the false-positive one-row file:

```text
recipes/aceproof-tts/pipeline/run_static_promotion_filters.py
input: outputs/repeat10-genonly-candidate-verification/current_decomp_109_root_falsepositive_x1.jsonl
rows=1 static_rejected=1
```

Re-running the current-decomposition fixed promotion with the updated static gate and existing sidecars gives:

```text
outputs/repeat10-genonly-candidate-verification/current_decomp_fixed_promotion_20260611_1448_staticroot_min1_partial.jsonl
candidates=98 promoted=0
```

Current conclusion: this `proofbench133_109` candidate is invalid, and fixed promotion should require the static modular-root blocker in addition to verifier sidecars. The root-integrity model verifier can remain as a diagnostic sidecar, but it is not reliable enough to be the only defense for this failure mode.

## Continuation: Fixed-Gate Rerun And Independent Audit, 2026-06-11 14:55 PDT

The endpoint remains healthy but saturated. A live health sample at `2026-06-11 14:49 PDT` reported `ok=True`, `56/56` ultra workers alive, `14100/14336` slots busy, `0` pending jobs, `13903` in-flight jobs, and `276764` completed jobs. A fresh scan of local logs found no new nginx `500`, LiteLLM internal server error, transient endpoint error, connection error, or `job not found` signatures.

The latest fixed-promotion reruns still promote no new candidates:

```text
current decomposition planner/assembly, with static modular-root blocker and root-integrity sidecar:
  outputs/repeat10-genonly-candidate-verification/current_decomp_fixed_promotion_20260611_1452_staticroot_min1_withroot_partial.jsonl
  candidates=98 promoted=0

v5 proofbench133_109/proofbench133_030:
  outputs/repeat10-genonly-candidate-verification/current_candidates_v5_109_030_fixed_promotion_20260611_1452_min1_partial.jsonl
  candidates=29 promoted=0

attempt-memory delta after snapshot78:
  outputs/repeat10-028-c50-attempt-memory-repair-verification/delta_after_snapshot78/fixed_promotion_20260611_1452_min1_partial.jsonl
  candidates=56 promoted=0

current2 high-self-eval proofbench133_028 rows:
  outputs/repeat10-genonly-candidate-verification/current2_new_high_selfeval_028_verifiers/fixed_promotion_20260611_1452_min1_partial.jsonl
  candidates=2 promoted=0
```

The current-decomposition near miss remains the same `proofbench133_109` planner row. It has the largest verifier support, but the new static blocker rejects it for deriving `a^2 == +/- b^2` from a negative fourth-power congruence. The next-best current-decomposition row is a `G25` planner candidate, but it is rejected by strict, theorem, and self-contained sidecars. The v5 `proofbench133_109` candidates still have partial support only; the best one has many `1.0` verifier scores but is rejected by strict, theorem, congruence, and Gaussian-sign families.

Independent best-model audits of the highest-priority candidate set also found no valid full solution:

```text
C50:
  4 audited candidates.
  Verdict: promising but incomplete or invalid.
  Common missing piece: the crossing-number/Garaev-type inequality is invoked or sketched but not proved, so the candidates are reductions rather than standalone proofs.

proofbench133_028:
  4 audited candidates.
  Verdict: promising but incomplete or invalid.
  Common missing piece: base-case work for f(1), f(2), and sometimes f(3) is asserted or uses premature substitution before those values are proved.

proofbench133_030 / proofbench133_109:
  7 audited candidates.
  Verdict: no valid full solution.
  proofbench133_030 has a plausible perimeter/minimax direction but leaves the key invariant unproved.
  proofbench133_109 candidates fail on Gaussian-unit case analysis, false coprimality, or incorrect modular/cubic congruence steps.
```

This aligns with the fixed harness: no additional solved problem should be claimed from the current repeat10/new-checkpoint streams. The active value of these runs is diagnostic. They identify which proof families are close enough to deserve targeted harness pressure, and they expose verifier false positives that need deterministic blockers or stronger verifier prompts before promotion.

## Continuation: Model-Memory Repair Stream, 2026-06-11 15:08 PDT

A new fair attempt-memory stream was launched for the five repeat10/new-checkpoint problems still not solved by the fixed gate: `proofbench133_028`, `proofbench133_109`, `C50`, `proofbench133_030`, and `G25`. The setup uses only model-generated candidates and model-generated verifier comments from the current streams; it does not use the independent best-model audit text.

Input construction now preserves existing `candidate_uid` values from curated candidate files so verifier sidecar comments attach correctly, and it skips literal non-object JSONL rows such as `null` merge artifacts. The committed inputs are:

```text
recipes/aceproof-tts/dataset/repeat10-current-modelmemory-input-20260611.jsonl
  full context: 185 deduplicated candidates, up to 10 candidates/problem, up to 4 comments/candidate, about 522 KB

recipes/aceproof-tts/dataset/repeat10-current-modelmemory-compact-input-20260611.jsonl
  compact context: same candidate pool, up to 4 candidates/problem, up to 2 comments/candidate, about 123 KB
```

Both inputs are being run through `recipes/aceproof-tts/prompts/proof_generation_memory_synthesis.yaml` at temperature `1.0`. The full-context tmux session is `repeat10-current-modelmemory-synthesis`; the compact-context tmux session is `repeat10-current-modelmemory-compact-synthesis`. Both were relaunched with the NeMo-Skills venv prepended to `PATH` because the first full-context launch failed locally before any endpoint request with `python: command not found` inside the generated nemo-run script.

As of `2026-06-11 15:08 PDT`, both memory-synthesis runs are alive but have not returned their first rows. Endpoint health remains `ok=True` with no fresh endpoint-failure signatures in local logs.

## Continuation: Memory-Guided Solver Launches, 2026-06-11 15:55 PDT

The compact model-memory synthesis completed for all five remaining problems. Its notes all self-evaluated as `1.0` and are stored at:

```text
outputs/repeat10-current-modelmemory-synthesis/temp10compact/memory/output.jsonl
```

The full-context model-memory synthesis has returned four notes so far: `G25`, `C50`, `proofbench133_109`, and `proofbench133_028`; `proofbench133_030` is still pending in the full-context stream.

Based on those memory notes, the following memory-guided proof-generation streams were launched with `proof_generation_with_memory.yaml`, temperature `1.0`, and no explicit token cap:

```text
repeat10-current-modelmemory-compact-solver-x32
  input: recipes/aceproof-tts/dataset/repeat10-current-modelmemory-compact-solver-input-partial-x32-20260611.jsonl
  coverage: proofbench133_028, proofbench133_109, C50, G25; 32 attempts each

repeat10-current-modelmemory-compact-solver-030-x32
  input: recipes/aceproof-tts/dataset/repeat10-current-modelmemory-compact-solver-input-030-x32-20260611.jsonl
  coverage: proofbench133_030; 32 attempts

repeat10-current-modelmemory-full-solver-c50-109-x32
  input: recipes/aceproof-tts/dataset/repeat10-current-modelmemory-full-solver-input-c50-109-x32-20260611.jsonl
  coverage: C50 and proofbench133_109; 32 attempts each

repeat10-current-modelmemory-full-solver-028-x32
  input: recipes/aceproof-tts/dataset/repeat10-current-modelmemory-full-solver-input-028-x32-20260611.jsonl
  coverage: proofbench133_028; 32 attempts
```

Early compact-solver rows are all `G25`. As of `2026-06-11 15:55 PDT`, 26 `G25` rows have returned and every one is deterministically rejected by the static promotion filter for a Pitot/incircle converse/sufficiency false pattern. No verifier sidecars were launched for those rows because they failed the deterministic gate. The non-`G25` compact rows and all rows from the other solver streams are still pending. Endpoint health remains `ok=True` and no fresh endpoint-failure signatures have appeared in local logs.

## Continuation: Memory-Guided Solver Relaunches, 2026-06-11 16:40 PDT

The full-context model-memory synthesis has now completed for all five remaining problems:

```text
outputs/repeat10-current-modelmemory-synthesis/temp10/memory/output.jsonl
```

All five full-context memory notes self-evaluated as `1.0` and stopped naturally. Because `proofbench133_030` was still pending when the first full-memory solver streams were launched, a matching full-memory 030 solver input was added after the full note arrived:

```text
recipes/aceproof-tts/dataset/repeat10-current-modelmemory-full-solver-input-030-20260611.jsonl
recipes/aceproof-tts/dataset/repeat10-current-modelmemory-full-solver-input-030-x32-20260611.jsonl
```

This uses the same prompt as the other memory-guided solver streams:

```text
recipes/aceproof-tts/prompts/proof_generation_with_memory.yaml
```

and is running as:

```text
repeat10-current-modelmemory-full-solver-030-x32
  input: recipes/aceproof-tts/dataset/repeat10-current-modelmemory-full-solver-input-030-x32-20260611.jsonl
  output: outputs/repeat10-current-modelmemory-solver/temp10full/memorysolver_030_x32
  temperature: 1.0
  max_concurrent_requests: 32
  token cap: none set explicitly; use the endpoint/context limit
```

The four earlier memory-guided solver streams were relaunched at `max_concurrent_requests=32` after a transient endpoint connection-error wave around `16:03 PDT`. The endpoint health page returned to `ok=True`, and the jobs are being monitored rather than cancelled because individual requests may legitimately take one to two hours on this endpoint. As of this update, the compact partial stream has produced 27 rows, all for `G25`; every returned `G25` row is still rejected by deterministic static promotion filters, so no verifier sidecars have been launched from this memory-solver family yet.

## Continuation: proofbench-ultra Endpoint Switch, 2026-06-11 16:40 PDT

The memory-guided solver streams were switched from model name `ultra` to the new `proofbench-ultra` endpoint after the user pointed out that it runs fewer requests per node and may be faster. The replacement launches use the same gateway URL, prompts, inputs, temperature, top-p, concurrency, and no explicit token cap; only the `--model` value and output directories changed. This keeps the speed comparison clean.

The active replacement outputs are:

```text
outputs/repeat10-current-modelmemory-solver-proofbench-ultra/temp10compact/memorysolver_x32
outputs/repeat10-current-modelmemory-solver-proofbench-ultra/temp10compact/memorysolver_030_x32
outputs/repeat10-current-modelmemory-solver-proofbench-ultra/temp10full/memorysolver_c50_109_x32
outputs/repeat10-current-modelmemory-solver-proofbench-ultra/temp10full/memorysolver_028_x32
outputs/repeat10-current-modelmemory-solver-proofbench-ultra/temp10full/memorysolver_030_x32
```

The old `ultra` memory-solver clients were stopped after the proofbench-ultra sessions reached the async request loop. Their partial outputs remain on disk for comparison. Before the 16:03 endpoint connection-error wave and 16:28 relaunch, the original compact partial stream had produced 26 rows between roughly 15:25 and 15:55. After the 16:28 relaunch, it added only one more row before the endpoint switch, while the compact 030, full C50/109, full 028, and full 030 streams were still at zero rows. That means the compact `G25` slice was clearly moving better before 16:28, but there is no evidence that the other slices were faster before 16:28.

## Continuation: Endpoint Monitor Fix And Elastic proofbench-ultra Expansion, 2026-06-11 17:02 PDT

The repeated Slack endpoint alerts after the `16:03-16:04 PDT` connection-error wave were stale-log false positives. The monitor scanned recently modified log files and matched old timestamped error lines inside those logs. It now filters matches to error lines whose embedded timestamp is within the last five minutes before notifying. Live endpoint health at the time of this fix was still `ok=True` for both `ultra` and `proofbench-ultra`; no fresh `proofbench-ultra` connection/500/timeout signatures were present.

The user clarified that the endpoint is elastic and can schedule more nodes as needed, so new proofbench-ultra submissions should not be locally throttled for queue concerns. Based on that, additional memory-guided solver inputs were created for seeds `32..127` and launched against `proofbench-ultra` with `max_concurrent_requests=256` per stream:

```text
recipes/aceproof-tts/dataset/repeat10-current-modelmemory-compact-solver-input-partial-s32-127-20260611.jsonl
recipes/aceproof-tts/dataset/repeat10-current-modelmemory-compact-solver-input-030-s32-127-20260611.jsonl
recipes/aceproof-tts/dataset/repeat10-current-modelmemory-full-solver-input-c50-109-s32-127-20260611.jsonl
recipes/aceproof-tts/dataset/repeat10-current-modelmemory-full-solver-input-028-s32-127-20260611.jsonl
recipes/aceproof-tts/dataset/repeat10-current-modelmemory-full-solver-input-030-s32-127-20260611.jsonl
```

These add 864 proof attempts total. Output directories are under:

```text
outputs/repeat10-current-modelmemory-solver-proofbench-ultra-delta/
```

The original proofbench-ultra `0..31` seed run is already faster than the stalled `ultra` relaunch: by `16:51 PDT`, it had returned rows for compact `G25`, compact `proofbench133_030`, full `proofbench133_109`, and full `proofbench133_030`. Static filtering left three full-memory `proofbench133_109` candidates. Seven x8 verifier sidecars were launched on those candidates using `proofbench-ultra`: strict audit, theorem gate, congruence guard, self-contained theorem-use, Gaussian sign branch, Gaussian unit, and modular-root integrity.

Independent audits rejected all three current `proofbench133_109` candidates. Candidates `proofbench133_109_11` and `proofbench133_109_29` miss Gaussian sign branches. Candidate `proofbench133_109_21` initially received one valid audit, but two independent follow-up audits both found the same fatal algebra error in the negative branch: it uses the false expansion

```text
s^12 + 27s^8u^4 + 243s^4u^8 + 729u^12 = (s^4 + 3u^4)^3
```

The correct cube matching those middle coefficients would not be `(s^4+3u^4)^3`; this makes the branch argument tautological rather than contradictory. A deterministic static blocker was added in `recipes/aceproof-tts/pipeline/run_static_promotion_filters.py` for this exact `proofbench133_109` false-positive pattern. With partial verifier sidecars and that static blocker, the fixed promotion gate promotes zero of the three current candidates.

At `17:05 PDT`, after the user clarified again that the mux endpoint is elastic and should not be locally queued, two more non-overlapping seed bands were launched for the same five memory-solver slices:

```text
seeds 128..255:
  recipes/aceproof-tts/dataset/repeat10-current-modelmemory-compact-solver-input-partial-s128-255-20260611.jsonl
  recipes/aceproof-tts/dataset/repeat10-current-modelmemory-compact-solver-input-030-s128-255-20260611.jsonl
  recipes/aceproof-tts/dataset/repeat10-current-modelmemory-full-solver-input-c50-109-s128-255-20260611.jsonl
  recipes/aceproof-tts/dataset/repeat10-current-modelmemory-full-solver-input-028-s128-255-20260611.jsonl
  recipes/aceproof-tts/dataset/repeat10-current-modelmemory-full-solver-input-030-s128-255-20260611.jsonl

seeds 256..383:
  recipes/aceproof-tts/dataset/repeat10-current-modelmemory-compact-solver-input-partial-s256-383-20260611.jsonl
  recipes/aceproof-tts/dataset/repeat10-current-modelmemory-compact-solver-input-030-s256-383-20260611.jsonl
  recipes/aceproof-tts/dataset/repeat10-current-modelmemory-full-solver-input-c50-109-s256-383-20260611.jsonl
  recipes/aceproof-tts/dataset/repeat10-current-modelmemory-full-solver-input-028-s256-383-20260611.jsonl
  recipes/aceproof-tts/dataset/repeat10-current-modelmemory-full-solver-input-030-s256-383-20260611.jsonl
```

The compact partial and full C50/109 streams use `max_concurrent_requests=512`; the smaller single-problem streams use `256`. All ten added sessions reached the async request loop against `proofbench-ultra` by `17:06 PDT`.

## Continuation: Elastic proofbench-ultra Wide Expansion, 2026-06-11 17:40 PDT

After the user clarified that the `proofbench-ultra` endpoint is elastic and should not be locally queued, the memory-guided solver streams were widened substantially. In addition to the original `0..31` and `32..127` seed ranges, independent seed-band sessions were launched through `2047` for the same five memory-solver slices:

```text
compact partial: proofbench133_028, proofbench133_109, C50, G25
compact 030: proofbench133_030
full C50/109: C50, proofbench133_109
full 028: proofbench133_028
full 030: proofbench133_030
```

Each band keeps the same prompt and endpoint setup:

```text
prompt: recipes/aceproof-tts/prompts/proof_generation_with_memory.yaml
model: proofbench-ultra
gateway: http://aiapps-053026.dyn.nvidia.com:28000/v1
temperature/top_p: 1.0 / 0.95
token cap: none set explicitly; use endpoint/context limit
```

The wider launches reached async loops for the active solver streams, with three early-return slices restarted after they exited before the async loop and produced no output:

```text
compact partial s1920-2047
full 028 s1664-1791
full C50/109 s1536-1663
```

A fresh static-pass collection at `17:31 PDT` found `393` deduplicated static-pass candidates from `1988` returned solver rows:

```text
C50: 107
proofbench133_028: 118
proofbench133_030: 60
proofbench133_109: 108
```

This snapshot is stored at:

```text
outputs/repeat10-genonly-candidate-verification/proofbench_ultra_modelmemory_staticpass_all_20260611_1731_candidates_all.jsonl
```

Verifier sidecars were launched for the `316` candidates that were new relative to the `17:15` snapshot:

```text
outputs/repeat10-genonly-candidate-verification/proofbench_ultra_modelmemory_staticpass_delta_1731/
```

Broad verifier families are normal, strict audit, theorem gate, congruence guard, self-contained theorem-use, and countermodel audit. For the 109 subset, Gaussian sign branch, Gaussian unit, and modular-root integrity sidecars were also launched. Two 109-specific verifier arms (`gaussiansign`, `root`) initially exited before async-loop startup and were relaunched as retry sessions; the retry sessions reached async loops.

Endpoint health stayed `ok=True`, but a real transient retry wave appeared around `17:37 PDT` in the `proofbench-ultra-staticpass-delta-1731` theorem/countermodel verifier logs:

```text
litellm.InternalServerError: InternalServerError: OpenAIException - Connection error
```

These were retryable client-side warnings (`retrying 1/24` and `2/24`), not final failures. A Slack notification was sent per the user's instruction. New launches were paused after this wave while existing solver/verifier requests continue retrying and completing. As of the partial `17:40 PDT` reducer, no candidate is promoted by the fixed verifier gate; most candidates in the newest snapshot still lack completed broad verifier rows.

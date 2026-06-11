# Repeat10 Multiplexer/New-Checkpoint Report

Status as of 2026-06-10 16:20 PDT. This report summarizes the local repeat10 workstream that reruns previously successful AceProof harness variants against the `ultra` cloud multiplexer endpoint/new checkpoint, with no explicit token cap in requests.

## Executive Summary

We targeted 10 rows that the new checkpoint still needed help with under the reference setup:

- `C39`, `N1`, `N14`, `N24`, `N39`, `proofbench133_028`, `proofbench133_109`, `C50`, `proofbench133_030`, `G25`.

Confirmed audited solves from this repeat10 workstream so far:

| Problem | Source | Method / arm | Verifier signal | Final status |
| --- | --- | --- | --- | --- |
| `N14` | `gems-remarkable` | `temp10/formalsanity`, R1 direct generation | `1/1` normal verifier | Confirmed valid by audit |
| `N24` | `gems-remarkable` | `temp08/proofonly`, R1 direct generation | no verifier rows for the valid candidate | Confirmed valid out-of-band by audit |
| `C39` | `gems-remarkable` | `temp10/proofonly`, reduced remaining-8 rerun | `1/4` full verifier votes, mean `0.625` | Confirmed valid by audit, assuming Vosper equality theorem is allowed |

The most important operational finding is that final pipeline promotion alone is not reliable enough here. The workstream found real solves by inspecting candidate bundles and using independent/manual audits, while the normal verifier also strongly promoted several invalid families.

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

The second reduced rerun used only the two prompt families that looked most productive or least noisy in the first ablation:

- `proofonly`
- `formalsanity`

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
- normal verifier scores: `[1.0, 0.5, 0.5, 0.5]`
- normal verifier mean: `0.625`
- audit status: `valid`

What worked:

The successful proof again came from `proofonly`, but at temperature `1.0`. It assumes no rainbow solution, derives pairwise sumset containments such as `R+B subset R union B union {0}`, adjoins zero to color classes, and applies a Vosper/equality inverse-Cauchy-Davenport classification to force arithmetic-progression structure. It then uses the resulting interval/one-sided structure and singleton color cases to reach a contradiction.

Notes:

Audit marked the proof valid assuming the stated Vosper equality theorem is allowed. This candidate would not have been promoted by the normal `meanscore >= 0.99999` pipeline threshold. It was found by ranking intermediate candidates and auditing a low-mean but structurally promising proof.

## Current Target Status

| Problem | User-provided circumstance | Current repeat10 status | Notes |
| --- | --- | --- | --- |
| `C39` | pipeline-unsolved | Confirmed solved | `temp10/proofonly`, audited valid with Vosper caveat. |
| `N1` | pipeline-unsolved | Not solved | `N1_63` was invalid: it proves set equality but then uses the stronger unsupported claim `a_i=i`. |
| `N14` | pipeline-unsolved | Confirmed solved | `temp10/formalsanity`, audited valid. |
| `N24` | pipeline-unsolved | Confirmed solved | `temp08/proofonly`, valid but not promoted by normal verifier. |
| `N39` | pipeline-unsolved | Not solved; repair seed found | `N39_51` has a promising Bertrand/Zsigmondy induction, but base-case exponential checks are not rigorously proved. |
| `proofbench133_028` | pipeline-unsolved | Not solved | No confirmed candidate in the inspected repeat10 artifacts. |
| `proofbench133_109` | gpt-rejected | Not solved; many false positives | Repeated Gaussian-integer candidates miss unit/sign branches. |
| `C50` | pipeline-unsolved | Not solved | No confirmed candidate in the inspected repeat10 artifacts. |
| `proofbench133_030` | pipeline-unsolved | Not solved | Game/minimax candidates had unsupported strategy invariants. |
| `G25` | gpt-rejected | Not solved; verifier false-positive hotspot | Many candidates got near-perfect or perfect verifier scores, but they use invalid Pitot/Ptolemy reasoning. |

## Comparison To Finished New-Checkpoint Runs

The fully finalized 14-arm multiplexer ablation ended with these finalizer-level promotions:

| Arm | Pipeline-promoted solve ids | Audit result |
| --- | --- | --- |
| `temp10/formalsanity` | `N14` | valid |
| `temp08/formalsanity` | `N24` | invalid promoted row (`N24_27`) |
| all other finalized arms | none | no finalizer solve |

This understates and overstates quality in different ways:

- It understates quality because `N24_6` was a real solve from `temp08/proofonly` but had no verifier rows and was not promoted.
- It overstates quality because `N24_27` was pipeline-promoted but invalid.
- The reduced remaining-8 rerun is not finalized yet. Its current audited contribution is `C39_7`, a real solve found from interim proof/verify files, plus `N39_51` as a repair seed.

Relative to the new checkpoint's original setup, this harness workstream adds confirmed audited solves on `N14`, `N24`, and `C39`. The rest of the target set remains unsolved by this repeat10 analysis.

## What Worked Well

Prompt family diversity helped, but only a subset was actually productive in this endpoint run:

- `formalsanity` produced the confirmed `N14` solve.
- `proofonly` produced the confirmed `N24` and `C39` solves.
- Both `temp08` and `temp10` produced useful candidates. There is no clean evidence from this small sample that `0.8` dominates `1.0` or vice versa.

Candidate-level auditing was essential:

- `N24_6` would have been missed by finalizer-only analysis.
- `C39_7` had only mean `0.625`, so it would have been discarded by solved-threshold promotion, but audit found it valid.
- Several high-verifier candidates were invalid, so high normal-verifier score alone was not enough.

Using no explicit token cap was fine for these runs:

- Configs omitted `tokens_to_generate` / `max_completion_tokens`.
- Requests used the endpoint's default context budget rather than forcing per-request generation caps.

The reduced rerun strategy was useful:

- After `N14` and `N24` were confirmed, rerunning only the remaining 8 with the two most useful prompt families found `C39` and a serious `N39` repair seed.

## What Did Not Work / Failure Modes

Endpoint stability dominated the first 14-arm run:

- The multiplexer became unhealthy during the first launch. Several arms finalized as zero-proof timeout artifacts.
- Zero-proof arms should not be interpreted as evidence that those prompts are bad.

Normal verification produced serious false positives:

- `G25`: many candidates reached perfect or near-perfect verifier scores. The common flaw is treating Pitot side-sum equality as sufficient for a convex quadrilateral to be tangential, or otherwise misusing Ptolemy/Pitot converses. These are invalid despite strong verifier support.
- `proofbench133_109`: multiple high-score candidates had Gaussian-integer unit/sign branch failures. After absorbing a Gaussian unit, they assume cube-root coordinates can be taken positive and then treat signed factors as positive squares. `proofbench133_109_60` reached `3/3` verifier ones and was still invalid by audit.
- `proofbench133_030`: high-scoring grid-game candidates did not prove the minimax invariant. They assumed Shayan leaf behavior, ignored possible `k >= 3` or multi-threat positions, or used unsupported pair-sum invariants.
- `C39_26`: invalid Combinatorial Nullstellensatz proof; it used ordinary coefficient vanishing where only the reduced remainder modulo grid polynomials is controlled.
- `N39_60`: invalid induction proof; the `a_n > n` case had a fatal divisibility arithmetic error.
- `N1_63`: invalid induction; it proved only unordered set equality for first prime-index blocks but then used pointwise equality.

The self-evaluation text inside generations was not enough for promotion:

- Some invalid candidates self-scored or presented as complete.
- The useful signal was mostly in candidate structure plus independent audit, not in the model's own boxed self-score.

## Repair Seeds And Next Candidates

### `N39_51`

Artifact:

```text
outputs/repeat10-ultra-mp-r1-remaining8-rerun1/potential_candidates_interim.jsonl
recipes/aceproof-tts/dataset/repeat10-n39-51-repairseed-20260610.jsonl
```

Status: not solved, but promising.

The proof uses a Bertrand-prime induction and Zsigmondy primitive-prime-divisor obstructions. Independent audit found the main induction strategy substantially sound. The first gap is in the base-case exponential checking, especially the `(a_1,a_2)=(8,1)` branch. A generic repair prompt has been staged that includes the original problem, the candidate proof, and the audit note, while instructing the model not to trust the candidate unless it can make every step rigorous.

Prepared repair configs:

```text
recipes/aceproof-tts/configs/repeat10-mp-repair/repeat10-ultra-mp-n39repairseed-temp08.yaml
recipes/aceproof-tts/configs/repeat10-mp-repair/repeat10-ultra-mp-n39repairseed-temp10.yaml
```

These were staged but not launched as part of this report.

### `proofbench133_109`

Status: repair direction unclear.

The repeated route through Gaussian integers may still be viable, but every high-scoring candidate so far misses a signed branch. Any future prompt should explicitly force a complete unit/sign case split or an absolute-value formulation that proves all branches.

### `G25`

Status: verifier hotspot, not a good promotion candidate without a targeted geometry gate.

The current normal verifier repeatedly accepts invalid Pitot-converse arguments. Future verifier sidecars for this problem should explicitly ask whether side-sum equality is being used as a sufficient tangency condition, and should reject proofs that do not prove the actual tangential condition.

## Artifacts

Cluster copy, with files/directories set to `a+rX`:

```text
/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/igitman/aceproof-share/20260610-aceproof-tts/repeat10/
```

This path is under the `llmservice_nemo_reasoning` project tree. It is readable
to users who can traverse that project directory, i.e. `llmservice` project
users on AWS-DFW.

The cluster copy includes the full local interim bundle that was too large to
commit to Git:

```text
/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/igitman/aceproof-share/20260610-aceproof-tts/repeat10/outputs/repeat10-ultra-mp-r1-remaining8-rerun1/potential_candidates_interim.jsonl
```

Confirmed solves:

```text
outputs/repeat10-ultra-mp-r1-ablation/confirmed_solutions_so_far.jsonl
outputs/repeat10-ultra-mp-r1-remaining8-rerun1/confirmed_solutions_so_far.jsonl
```

Candidate bundles:

```text
outputs/repeat10-ultra-mp-r1-ablation/potential_candidates_so_far.jsonl
outputs/repeat10-ultra-mp-r1-remaining8-rerun1/potential_candidates_audited_subset.jsonl

# Full local interim bundle, not committed because it is large:
outputs/repeat10-ultra-mp-r1-remaining8-rerun1/potential_candidates_interim.jsonl
```

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

This broad run is the source of the “previously successful ideas” that motivated the smaller repeat10 multiplexer rerun. Compared with the broad open53 result, the repeat10 multiplexer/new-checkpoint branch is much narrower: it tries to reproduce or transfer success on only 10 selected rows through the cloud endpoint and currently has confirmed audited repeat10 gains on `N14`, `N24`, and `C39`.

## Bottom Line

The most reliable pattern so far is not a single prompt or temperature; it is prompt-family diversity plus candidate-level audit. For this repeat10/new-checkpoint target set, the confirmed gains are `N14`, `N24`, and `C39`. The best next follow-up is a small `N39_51` repair run, because its main proof strategy appears sound and the known gap is narrow. The biggest caution is verifier reliability: `G25` and `proofbench133_109` show that normal verifier ratios can be confidently wrong on recurring proof-pattern failures.

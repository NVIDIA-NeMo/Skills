# Repeat10 Multiplexer Temperature Ablation

Purpose: repeat the prompt families that previously solved the 10 selected problems on the `ultra` multiplexer endpoint, while ablating proof-generation temperature.

Input file:

```text
recipes/aceproof-tts/dataset/repeat10-newckpt-nosolutions-20260608.jsonl
```

Endpoint/model:

```text
http://aiapps-053026.dyn.nvidia.com:28000/v1
model: ultra
```

Arms:

- `temp08`: generation/refinement temperature `0.8`, verifier temperature `1.0`.
- `temp10`: generation/refinement/verifier temperature `1.0`.

For this R1-only run, refinement is not used; the relevant ablation is proof-generation temperature. Verification stays at `1.0` in both arms.

Prompt families:

```text
lemma, claimcert, case, proofonly, routecompare, compactaudit, formalsanity
```

Token budgets are intentionally omitted from all configs so requests do not send `max_completion_tokens`; the endpoint default context budget should apply.

Prepared guarded launcher:

```bash
/home/igitman/workspace/NeMo-Skills-aceproof-mp/recipes/aceproof-tts/scripts/launch_repeat10_mp_ablation.sh
```

It does not launch unless passed `--run`:

```bash
/home/igitman/workspace/NeMo-Skills-aceproof-mp/recipes/aceproof-tts/scripts/launch_repeat10_mp_ablation.sh --run
```

## Running Notes

- 2026-06-08: launched all 14 arms through tmux against `model=ultra` with
  no explicit `tokens_to_generate` / `max_completion_tokens`.
- Installed `openai[aiohttp]` in the shared NeMo-Skills venv so
  `NEMO_SKILLS_OPENAI_AIOHTTP=1` uses the aiohttp client path.
- The endpoint became unhealthy during the run: dashboard reached
  `ultra` workers `0/38` alive with roughly 11k global pending jobs.
  Several arms then hit the 14,400s request timeout. This makes the
  timeout-finalized zero-proof rows a serving/outage artifact, not an
  accuracy result.
- Six arms finalized with 10 outer rows each and `num_proofs=0` for all
  problems: `temp08/{case,claimcert,compactaudit}` and
  `temp10/{case,claimcert,routecompare}`.
- During recovery, `temp08/formalsanity` produced proof `N24_27` with
  13/13 verifier scores at `1.0`, but manual audit found it invalid: it
  maps infinitely many primes to positive integers via the first index
  `m(p)` and incorrectly applies pigeonhole to conclude one index has
  infinitely many preimages. This is a verifier false positive, not a
  solved problem.
- Monitoring policy update: multiplexer requests can legitimately take
  1-2 hours. Do not cancel or resubmit solely because output is flat over
  a short interval. Cancel/resubmit only with stronger evidence, such as
  dead local clients, repeated endpoint-side failures, completed zero-proof
  artifacts from request timeouts, or dashboard/local logs showing the
  request path is no longer making progress.
- 2026-06-09 01:23: `temp10/compactaudit` also finalized as a
  zero-proof timeout-shaped artifact: 10 outer rows, all `meanscore=0`,
  `num_proofs=0`, and no candidate proof.
- 2026-06-09 01:38: `temp08/routecompare` finalized with one non-empty
  `N24` candidate (`meanscore=0.75`, 4 verifier rows). Manual audit found
  it invalid: from `p ∤ b_n = A_n - 1` it concludes `A_n ≡ 1 (mod p)`,
  but the actual implication is only `A_n ≠ 1 (mod p)`. This is another
  verifier false positive, not a solve.
- 2026-06-09 00:53-02:38 monitor window: six arms remained live
  (`temp08/{formalsanity,lemma,proofonly}` and
  `temp10/{formalsanity,lemma,proofonly}`), but proof/verifier row counts stayed
  flat at 52/59. Dashboard showed many alive replicas at times, but queue
  `slots_busy` stayed at 0 and global `jobs_done` moved only from 20591 to
  20593. Decision: do not cancel solely for latency; wait for natural timeout
  or finalization, because immediate resubmission would likely enter the same
  stalled queue.
- 2026-06-09 candidate audit update: extracted
  `outputs/repeat10-ultra-mp-r1-ablation/potential_candidates_so_far.jsonl`
  with 44 non-empty proof candidates and verifier metadata. Independent audit
  confirmed two real solves so far, written to
  `outputs/repeat10-ultra-mp-r1-ablation/confirmed_solutions_so_far.jsonl`:
  `N14_62` from `temp10/formalsanity` (verifier mean 1.0 over one verifier
  row; valid via reduction to FLT for exponent 3/Euler) and `N24_6` from
  `temp08/proofonly` (no verifier rows; valid via CRT + Dirichlet + quadratic
  reciprocity). This means the endpoint run produced useful candidates even
  though the live verifier/finalizer path is heavily degraded by endpoint
  timeouts.
- 2026-06-09 02:39-04:19 slow monitor window: six clients remained live
  but all output counts stayed flat (`proof=52`, `verify=59`, `outer=80`).
  Dashboard `jobs_done` stayed at 20593 and `slots_busy` stayed 0 despite
  fluctuating alive workers. This is consistent with a backend scheduler/queue
  stall. Decision: keep clients alive for now because late requests may still
  return, but do not launch replacements into this stalled queue.
- Prepared reduced future-rerun input excluding confirmed solves `N14` and `N24`:
  `recipes/aceproof-tts/dataset/repeat10-newckpt-remaining8-after-audit-20260609.jsonl`.
  This is not launched yet; it is for future reruns once the multiplexer
  generation queue is healthy.
- 2026-06-09 04:51 passive monitor: async rows moved slightly
  (`outer_async=48 -> 52`) while proof/verifier files stayed flat. The new
  `temp08/formalsanity` async rows include `N14` with `meanscore=0.5` and
  11 proof candidates, and `N24` marked solved from the already-audited invalid
  `N24_27` verifier false positive. This confirms the clients can still return
  late rows, but high verifier scores still need audit.
- 2026-06-09 05:51 passive monitor: four more arms finalized:
  `temp08/formalsanity`, `temp08/lemma`, `temp10/lemma`, and
  `temp10/proofonly`. No new confirmed problem solves. `temp08/formalsanity`
  finalized `N24` as solved using the already-invalid `N24_27` false positive
  and included valid `N14` partials; `temp08/lemma` and `temp10/lemma` were
  effectively empty; `temp10/proofonly` only added another `N14` partial.
  Remaining live arms: `temp08/proofonly` and `temp10/formalsanity`.
- 2026-06-09 05:52-06:52 final-arms monitor: only
  `temp08/proofonly` and `temp10/formalsanity` remained live. No new rows
  appeared (`proof=52`, `verify=59`, `outer=120`, `outer_async=16`). Endpoint
  health fluctuated from mostly alive to `1/49` alive and back to `8/45`, but
  `slots_busy` stayed 0 and `jobs_done` stayed 20593. Decision: keep the last
  two clients alive for natural timeout/finalization; do not submit replacement
  runs into the stalled backend.
- 2026-06-09 06:53-08:23 sparse monitor: remaining live arms stayed
  `temp08/proofonly` and `temp10/formalsanity`; no output counts changed.
  Dashboard degraded to `0/46` or `0/47` alive workers with `slots=0/0`,
  `jobs_done=20593` flat, and pending/in-flight unchanged. Slack update sent
  because this is a persistent endpoint-generation issue. Decision: leave the
  two clients alive, but do not submit anything new until generation throughput
  resumes.
- 2026-06-09 09:23 finalization: all 14 arms finalized (`outer=140`,
  `outer_async=0`). Last two arms were `temp08/proofonly` and
  `temp10/formalsanity`. `temp10/formalsanity` finalized `N14` as solved with
  the confirmed valid `N14_62` proof (`meanscore=1.0`). `temp08/proofonly`
  finalized `N14` at `meanscore=0.5` and `N24` at `meanscore=0.0`; its valid
  `N24_6` candidate remains an out-of-band confirmed solve because verifier
  did not promote it. Final confirmed audited solves from this run remain 2:
  `N14` and `N24`.

- 2026-06-10 09:07: after the multiplexer endpoint was reported healthy,
  launched a reduced rerun over the remaining 8 unsolved selected problems,
  excluding already-confirmed `N14` and `N24`:
  `outputs/repeat10-ultra-mp-r1-remaining8-rerun1`. Arms launched:
  `temp08/proofonly`, `temp10/proofonly`, `temp08/formalsanity`, and
  `temp10/formalsanity`, all using the cloud multiplexer endpoint and no
  explicit max-token cap.
- 2026-06-10 13:29-13:37 endpoint health looked usable, not like the prior
  outage: `13-16` live workers, busy slots, zero or small pending queue, and
  global `jobs_done` advancing (`216511 -> 216749`). Local verifier rows also
  advanced (`1994 -> 2091`). Client logs still print LiteLLM retry boilerplate,
  but because output files continue to grow this is being treated as transient
  retry noise rather than a cancellation signal.
- Interim candidate bundle for the reduced rerun:
  `outputs/repeat10-ultra-mp-r1-remaining8-rerun1/potential_candidates_interim.jsonl`.
  It includes verifier scores, candidate proofs, and manual/independent audit
  status for triaged candidates.
- Audited false positives from the reduced rerun:
  - `proofbench133_030_45` (`temp10/formalsanity`, verifier mean `0.833`) is
    invalid as written. The perimeter arithmetic is fine, but the proof does
    not establish the minimax invariant; it assumes Shayan's leaf behavior and
    does not prove Ali cannot create `k >= 3` or multi-threat positions.
  - `proofbench133_030_27` (`temp10/formalsanity`, verifier mean `0.75`) is
    invalid as written. The claimed pair-sum invariant depends on false or
    unsupported claims such as a unique `k=2` cell and exact contribution `3`
    for each Shayan-Ali pair.
  - `C39_26` (`temp10/formalsanity`, verifier scores `[1.0, 0.0, 1.0, 0.5]`)
    is invalid. It uses a false ordinary-coefficient form of Combinatorial
    Nullstellensatz; vanishing on a finite grid only constrains the reduced
    remainder modulo the grid vanishing polynomials.
  - `N39_60` (`temp08/proofonly`, verifier scores `[0.5, 1.0, 0.5, 0.5]`) is
    invalid. The `a_n < n` induction half is mostly sound, but the `a_n > n`
    case has a fatal arithmetic error: from `p | d` it claims
    `p | 2n + 2d`, while actually `2n + 2d == 2n (mod p)` and `p > n`.

  - `proofbench133_109_17` (`temp08/formalsanity`, verifier scores
    `[1.0, 0.5, 0.5, 1.0]`) is invalid but closer than the earlier 109
    false positives. Independent audit found the Gaussian integer setup mostly
    sound, but the case split is incomplete: in Case A, the subcase `3 | y`
    is not symmetric to the handled `3 | x` case, so the proof leaves one
    branch unproved. This is a plausible repair seed rather than a solve.

  - `proofbench133_109_63` (`temp08/formalsanity`, verifier scores
    `[1.0, 1.0, 0.0, 0.5]`) is invalid but also a repair seed. It fixes some
    earlier sign handling by using absolute values, but the Gaussian Case II
    proof applies modulo `8` to `3x^4-z^4` without proving the sign; if the
    expression is negative, `|3x^4-z^4|` can be `1 mod 8`, so the contradiction
    does not cover all branches.

- Confirmed solve from the reduced rerun:
  - `C39_7` from `temp10/proofonly` (`mean_score=0.625`, verifier scores
    `[1.0, 0.5, 0.5, 0.5]`) was independently audited as valid, assuming the
    stated Vosper equality theorem is allowed. The proof converts the no-rainbow
    assumption into `R'+B'=R'∪B'` after adjoining zero, applies the Vosper
    equality case to force common-difference arithmetic progressions, and then
    derives contradictions from the one-sided interval structure and singleton
    color cases. Extracted to
    `outputs/repeat10-ultra-mp-r1-remaining8-rerun1/confirmed_solutions_so_far.jsonl`.

- Prepared future remaining-7 input excluding newly confirmed `C39`:
  `recipes/aceproof-tts/dataset/repeat10-newckpt-remaining7-after-c39-20260610.jsonl`.
  Remaining IDs: `N1`, `N39`, `proofbench133_028`, `proofbench133_109`,
  `C50`, `proofbench133_030`, `G25`.

  - `proofbench133_109_59` (`temp10/proofonly`, verifier scores
    `[1.0, 0.5, 0.5, 0.5]`) is invalid. It uses a different parity-based
    Gaussian argument, but the mod `16` elimination is incomplete: from
    `v == 3 (mod 4)` it incorrectly concludes `v^2 == 9 (mod 16)`, missing
    branches such as `v == 7 (mod 8)` where `v^2 == 1 (mod 16)`.

  - `proofbench133_109_6` (`temp08/proofonly`, currently 3/3 verifier-1s) is
    invalid. It collapses Gaussian unit/sign cases into one displayed branch,
    omitting the genuine Case B branch, and also falsely treats the `3 | d`
    subcase as symmetric to `3 | c` inside Case A. This is another high-verifier
    false positive for the 109 family.

  - `proofbench133_109_31` (`temp08/proofonly`, 2/2 verifier-1s when audited)
    is invalid. It absorbs the Gaussian unit into a cube root, then treats
    coprime factors like `c` and `c^2-3d^2` as positive squares. The factor
    product is positive, but both factors can be negative; the signed-square
    branches are omitted.


- 2026-06-10 16:04 endpoint check: cloud multiplexer looked healthy:
  `14/14` workers, `3584/3584` slots busy, `pending=0`, and `jobs_done`
  advancing. Local proof/verify rows continued growing, so no endpoint action
  was needed.
- Additional reduced-rerun audits:
  - `N39_51` (`temp08/formalsanity`, verifier scores `[0.5, 0.5, 1.0, 1.0]`)
    is a strong repair seed but not a clean solve as written. Independent audit
    found the main induction/Zsigmondy strategy substantially sound; the first
    gap is that the base-case exponential checks are asserted rather than
    proved, especially the `(a1,a2)=(8,1)` branch.
  - `proofbench133_109_60` (`temp08/proofonly`, currently 3/3 verifier-1s) is
    invalid. Independent audit found the same Gaussian unit/sign failure as
    earlier 109 false positives: after absorbing the unit, the proof
    unjustifiably takes the cube-root coordinates positive and then treats
    signed factors as positive squares.
  - `N1_63` (`temp10/proofonly`, currently verifier scores `[1.0, 0.5]`) is
    invalid by local triage. It proves only set equality
    `{a_1,...,a_q}={1,...,q}` but then uses the stronger unsupported claim
    `a_i=i` for all `i<=q`; the later smooth-interval step is also asserted.


- 2026-06-10 16:24 audit update:
  - `proofbench133_109_39` (`temp10/proofonly`, verifier reached 4/4 ones) is
    invalid. Independent audit found the same Gaussian unit/sign branch failure:
    the proof eliminates all units except `u=1` by assuming signs of `pi^3` are
    already positive, missing the unit-rotated branch where both coordinates can
    become positive. This is another severe verifier false positive for 109.

# Nemotron-IMO-TTS: the IMO 2026 ensemble proof pipeline

This recipe is the inference pipeline described in
[*An Open Recipe for IMO Gold: Training Nemotron for Olympiad Mathematics*](paper.pdf) (Section 4). Three Nemotron 3 Ultra
checkpoints (the general-availability model and the released RL and SFT specialists) search for a proof of each
problem in natural language, and a separate high-compute stage selects the proof to submit. The pipeline uses no
formal prover, no tools, and no internet access; it only needs the checkpoints served behind an OpenAI-compatible
chat-completions endpoint.

| Stage | What happens | Per problem |
|---|---|---|
| Round 1 | 8 generation prompts x 3 checkpoints x 16 samples | 384 proof attempts |
| Verification | RL and SFT each judge every distinct proof 8 times; accepted only if all 16 judgments are 1 | 16 judgments per proof |
| Refinement (rounds 2 to 8) | the 16 best pool proofs, each with up to 8 verifier critiques, refined by every checkpoint 4 times | 192 attempts per round |
| Final selection | GA, RL, and SFT each grade every finalist 16 times with an IMO-style 0 to 7 judge prompt | 48 judgments per finalist |

All counts are derived from the config and checked at launch. The prompts in `prompts/` are byte-identical to
the report's Appendix B.

## Requirements

- A NeMo-Skills checkout (this recipe imports nothing from `nemo_skills`; the repository requirements already
  include `openai`, `httpx`, `pyyaml`, and `transformers`).
- The three checkpoints served behind one or more OpenAI-compatible `/v1` endpoints: a hosted API, a gateway,
  or local [vLLM](https://docs.vllm.ai) servers. Each model entry in the config can point at its own URL. The
  server must accept the configured context window (`max_len`): completions run up to 512k tokens.
- An API key in an environment variable when the endpoint requires one.
- Optionally the served model's tokenizer (a Hugging Face id or a local path) so prompt tokens can be counted
  exactly and completion budgets fitted to the context window. Without it, `max_tokens` is sent as configured.
- An open-file limit above the configured concurrency lanes (`ulimit -n 4096` is enough for the shipped
  config; the launcher checks this).

## Quickstart

1. Prepare the problems as JSONL, one row per problem, with the statement in `problem` (or `question`) and a
   unique `problem_idx` (or `id`):

   ```json
   {"problem_idx": "P1", "problem": "Let $n$ be a positive integer. Prove that ..."}
   ```

   To rebuild the report's 30-problem development set from public datasets, see
   [Rebuilding the development set](#rebuilding-the-development-set).

2. Copy the pipeline config and fill in the `<your-...>` placeholders: the endpoint base URL, the served model
   names, the tokenizer, and the input file.

   ```bash
   cp recipes/nemotron-imo-tts/configs/imo2026-ensemble.yaml my-run.yaml
   ```

   Against a hosted endpoint:

   ```yaml
   endpoint:
     base_url: https://integrate.api.nvidia.com/v1
     api_key_env: NVIDIA_API_KEY
   ```

   Against local vLLM servers that need no key (one URL per checkpoint):

   ```yaml
   endpoint:
     base_url: http://localhost:8000/v1
     api_key_env: null
   checkpoints:
     - id: ga
       model: nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
       base_url: http://ga-host:8000/v1
       ...
   ```

3. Check everything without sending a request:

   ```bash
   export NVIDIA_API_KEY=...
   python recipes/nemotron-imo-tts/run.py --config my-run.yaml --output-dir runs/my-run --dry-run
   ```

   The dry run validates the config, derives the counts, freezes the input, writes the manifests, loads the
   tokenizer, and warns about configured model names the endpoint does not list.

4. Run. The process runs in the foreground until every problem is finished; keep it in `tmux` or similar.

   ```bash
   python recipes/nemotron-imo-tts/run.py --config my-run.yaml --output-dir runs/my-run
   ```

   Start with `configs/smoke.yaml` (the same shape with tiny counts) to check a new endpoint end to end.

5. Read `runs/my-run/submissions.jsonl`: one row per problem with the selected proof. `results.jsonl` has the
   full per-problem record (finalists, scores, counts).

## How a problem is processed

- **Generation.** Each of the eight prompts in `search.generation_prompts` is rendered with the statement and
  sampled `generation_samples_per_prompt` times by every checkpoint (`seed` = sample slot). A response is split
  into the `## Solution` section (the proof) and the `## Self Evaluation` section; truncated responses and
  responses without a solution produce no candidate. Identical proof texts are merged and verified once.
- **Verification.** Every candidate is judged by every verifier `judgments_per_proof` times with the
  verification prompt. A judgment is valid when the response finished and ends in a boxed score (0, 0.5, or 1;
  anything else is invalid). The panel is complete when every verifier returned all of its judgments valid.
  A proof is **accepted** when the panel is complete and every judgment is 1. Complete panels also give the
  proof a mean verifier score (the mean of the per-verifier means) and make it a pool record; an incomplete
  panel is kept only as audit rows.
- **Checkpoint-level stop.** An accepted proof becomes the finalist of the checkpoint that produced it, and that
  checkpoint's remaining generation and verification for the problem are cancelled. Each checkpoint stops
  independently, as soon as one of its own proofs is accepted; a checkpoint that has not produced an accepted
  proof keeps going until the round ends. If any checkpoint has a finalist when the round ends, the search
  stops with one to three finalists.
- **Refinement.** Otherwise the pool is ranked by mean verifier score (then self-evaluation score, later round,
  proof id) and the top `refinement_prompts_per_round` proofs each get one refinement prompt: the standard
  generation prompt as the instruction, the proof, and up to 8 of its verifier critiques (at most 4 per score
  value when the scores are mixed, balanced across verifiers). Every checkpoint samples the prompt
  `refinement_samples_per_prompt` times. Refined proofs are verified like round-1 candidates and join the same
  pool. This repeats up to `max_rounds`.
- **Fallback.** After the last round without acceptance, the top-ranked pool proof is the sole finalist. A
  problem with no complete panel at all ends with `status: no_candidates` and an empty submission.
- **Final selection.** Every finalist is graded by every judge `judgments_per_finalist` times with the
  IMO-style prompt (an integer 0 to 7). A judgment that is truncated or has no parseable score is re-sampled
  with a fresh seed up to 3 more times; a slot still invalid afterwards is excluded. Finalists are ranked by the
  mean over valid judgments, then shorter proof text, then checkpoint order; the first is the submission.
  With `judges: null` the stage is skipped and finalists are ranked by acceptance, mean verifier score, and
  proof length.

## Configuration reference

Everything under `endpoint` except `extra_body`, every per-model `base_url` and `api_key_env` override, and
everything under `concurrency` is **operational**: change it freely between relaunches. All other keys define
the experiment and are pinned by the run manifest (see [Resume](#resume-stop-and-relaunch)).

| Key | Meaning |
|---|---|
| `endpoint.base_url` | Default OpenAI-compatible base URL (`.../v1`). |
| `endpoint.api_key_env` | Environment variable with the API key; `null` for servers that need no key. |
| `endpoint.timeout_s` | Per-request client timeout. |
| `endpoint.retry_window_s`, `backoff_base_s`, `backoff_max_s` | Transient failures (408, 429, 5xx, timeouts, connection errors) are retried with jittered exponential backoff until the window elapses, then the request is parked. |
| `endpoint.extra_body` | Extra fields merged into every request body, for example `chat_template_kwargs`. |
| `checkpoints[]` | Generation and refinement models: `id`, `model` (served name), `max_len` (context window), `max_tokens` (completion budget), `generation_samples_per_prompt`, `refinement_samples_per_prompt`, optional `base_url` and `api_key_env`. |
| `verifiers[]` | Search-time verifiers: `id`, `model`, `max_len`, `max_tokens`, `judgments_per_proof` (equal for all verifiers). |
| `judges[]` or `null` | Final-selection judges: `id`, `model`, `max_len`, `max_tokens`, `judgments_per_finalist`. |
| `search.generation_prompts` | Ordered prompt file stems under `prompts/`. |
| `search.max_rounds` | Round 1 generates; rounds 2 and up refine. |
| `search.refinement_prompts_per_round` | Pool proofs refined per round. |
| `sampling.temperature`, `sampling.top_p` | Sampling for every request. |
| `concurrency.problems` | Problems advanced concurrently. |
| `concurrency.generation`, `verification`, `refinement`, `judging` | In-flight request lanes per role. |
| `context_budget.tokenizer`, `safety_margin_tokens` | Optional prompt-token counting; the completion budget becomes `min(max_tokens, max_len - prompt_tokens - margin)`, and a prompt that leaves no budget is recorded and skipped. |
| `input.path` | The problems file. |

Requests that fail deterministically (HTTP 400, 409, 413, 415, 422, or a context-length error) are parked
without retry. A parked request is contained: one fewer sample, vote, or judgment, recorded in `errors.jsonl`
and re-issued on relaunch. HTTP 401, 403, and 404 abort the run, since a bad key or model name is never a
per-request problem.

## Outputs

```
<output-dir>/
  run_manifest.json          experiment keys, their hash, code commit
  input.jsonl                the frozen problems
  prompt_manifest.json       prompt ids and hashes
  run.log
  problems/<problem_idx>/
    rounds/R1/gen.jsonl      one row per generation attempt
    rounds/R<r>/refine.jsonl one row per refinement attempt (rounds 2 and up)
    rounds/R<r>/verify.jsonl one row per verifier judgment
    rounds/R<r>/errors.jsonl parked and skipped requests
    selection/judge.jsonl    one row per judge attempt (slot and attempt index)
    selection/errors.jsonl
  proof_pool/<source_name>/<problem_idx>.jsonl   every proof with a complete panel, with scores and critiques
  results.jsonl              one row per finished problem: status, rounds, finalists with scores, selection
  submissions.jsonl          one row per finished problem: the selected proof
```

Rows carry their provenance (`generation_model_id`, `generation_prompt_id`, `generation_random_seed`,
`aggregation_trial_idx`, `dep_proof_ids`, `verifier_id`, `verification_seed`, `judge_id`, `slot`, `attempt`),
the response (`generation`, `reasoning_content` when the server returns it, `finish_reason`, `usage`), and the
context budget that was applied. Vote and judge rows reference proofs by `proof_sha256`.

## Resume, STOP, and relaunch

Relaunch the same command to resume. Every request has a deterministic identity, and every response is
persisted as one row; on launch each problem directory is scanned and only missing work is issued. Problems
already in `results.jsonl` are skipped. A relaunch into the same output directory must resolve to the same
experiment keys (models, counts, budgets, prompts, sampling, tokenizer, input hash); otherwise the launcher
lists the drifted keys and refuses. Use a fresh output directory for a different experiment.

To stop cleanly, create a file named `STOP` in the output directory (or press Ctrl-C). In-flight requests are
abandoned, completed responses are kept, and the process exits with code 0. Remove `STOP` before relaunching.

## IMO 2026 submissions

The six proofs the pipeline submitted at IMO 2026 (30 of 42 points), plus the later Problem 6 proof from the
continued run, are in [`imo-proofs/`](imo-proofs/README.md) as JSONL rows with the problem statement, the proof,
and the score.

## Rebuilding the development set

The report's development set has 20 problems from Nemotron-IMO-Bench and 10 from public competitions
(IMO-ProofBench, EGMO via MathNet, IMO 2025 via MathArena). `scripts/build_devset30.py` downloads them from
Hugging Face and writes a pipeline input file in the report's order:

```bash
python recipes/nemotron-imo-tts/scripts/build_devset30.py \
    --nemotron-imo-bench <hugging-face-dataset-id-or-url> --output devset30.jsonl
```

Pass `--public-only` to write just the ten public problems. Each row records its source dataset and license.

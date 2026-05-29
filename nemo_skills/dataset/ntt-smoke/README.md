# NTT-SMOKE

`NTT-SMOKE` is a compact NemotronTranscribe smoke benchmark with English and
multilingual variants:

- `ntt-smoke.en`
- `ntt-smoke.multi`

The benchmark is intentionally a mixed manifest with subtests recorded in
`ntt_subtask` and `subset_for_metrics`. This keeps evaluation and summarization
consolidated while still producing subtask-level metrics.

## Source Data

Prepare the source benchmarks first, then point `ntt-smoke` at the prepared
data root. For the full reproducible suite, provide Preference-ASR directly
rather than the broader ASR parent directory:

```bash
ns prepare_data asr-leaderboard fleurs covost2 contextasr-bench librispeech-pc musan gpqa --data_dir <skills_data>

export NTT_SMOKE_SOURCE_DATA_DIR=<skills_data>
export NTT_SMOKE_PREFERENCE_ASR_DIR=/path/to/preference-asr-bench

ns prepare_data ntt-smoke --data_dir <skills_data>
```

The default full suite uses 200 as the minimum metric-group size:

```bash
--audio-samples 200
--prompt-groups 200
--preference-asr-samples-per-group 200
--preference-asr-prompt-variants original,direct
--long-samples 200
--text-samples 200
```

If Preference-ASR is absent, the audio-instruction subtest falls back to
`librispeech-pc`; such fallback runs are useful for local smoke tests but are
not the intended full NTT-SMOKE configuration.

## Subtests

- clean ASR from read, conversational, and media-style speech, with each clean
  subgroup sampled independently;
- noisy conversational ASR;
- noisy media ASR;
- very short ASR;
- stitched long-form ASR from about 20 minutes to 1 hour;
- non-speech hallucination;
- prompt robustness using paired prompt variants;
- audio-related instruction following through Preference-ASR subgroups for
  normalization, entities, disfluencies, case, and standard prompts, each with
  multiple prompt variants for prompt-adherence checks;
- ContextASR contextless/coarse/fine modes;
- superficial text multiple-choice checks.

Generated noisy and long-form audio is written under `ntt-smoke/data/`.
Original-source samples preserve their original `/data/<source>/...` paths and
therefore expect the source datasets to remain under the same prepared data
root used at evaluation time. Preference-ASR samples preserve absolute paths
under `NTT_SMOKE_PREFERENCE_ASR_DIR`, so that directory must be readable or
mounted at evaluation time. The default long-form setting cycles through
20-minute, 40-minute, and 60-minute stitched samples.

## Metrics

NTT-SMOKE reports WER macro, corpus WER, substitutions, insertions, deletions,
reference words, and correct words for ASR-style groups. Preference-ASR rows
use the Preference-ASR preference-aware normalizer when scoring, so the WER
reflects whether the requested formatting or transcription preference was
followed rather than erased by generic normalization.

See `MEMO.md` for the benchmark rationale, group definitions, and
reproducibility notes.

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
data root:

```bash
ns prepare_data asr-leaderboard fleurs covost2 contextasr-bench librispeech-pc musan gpqa --data_dir <skills_data>
NTT_SMOKE_SOURCE_DATA_DIR=<skills_data> ns prepare_data ntt-smoke --data_dir <skills_data>
```

Optional preference-ASR data can be provided with:

```bash
NTT_SMOKE_PREFERENCE_ASR_DIR=/path/to/preference-asr-bench
```

If preference-ASR data is absent, the audio-instruction subtest falls back to
`librispeech-pc`.

## Subtests

- clean ASR from read, conversational, and media-style speech;
- noisy conversational ASR;
- noisy media ASR;
- very short ASR;
- stitched long-form ASR from about 20 minutes to 1 hour;
- non-speech hallucination;
- prompt robustness using paired prompt variants;
- audio-related instruction following through punctuation/capitalization ASR;
- ContextASR contextless/coarse/fine modes;
- superficial text multiple-choice checks.

Generated noisy and long-form audio is written under `ntt-smoke/data/`.
Original-source samples preserve their original `/data/<source>/...` paths and
therefore expect the source datasets to remain under the same prepared data
root used at evaluation time. The default long-form setting generates one
20-minute, one 40-minute, and one 60-minute stitched sample per variant.

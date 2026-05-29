# NTT-SMOKE Memo

## Purpose

NTT-SMOKE is a compact NemotronTranscribe evaluation suite intended to catch
regressions across the major speech-model behaviors before larger benchmark
runs. It has English and multilingual variants:

- `ntt-smoke.en`
- `ntt-smoke.multi`

The suite is a mixed manifest. Every row records `ntt_subtask`,
`subset_for_metrics`, `origin_dataset`, `origin_manifest`, `origin_id`, and
language metadata so metrics can be aggregated overall or inspected by source
and subtask.

## Group Sizing

The default configuration targets a minimum metric-group size of 200 rows.
Groups that are expanded by prompt variants have at least 200 underlying
prompt groups and at least 400 rows when the default `original,direct`
Preference-ASR prompt variants are used.

Default preparation knobs:

```bash
--audio-samples 200
--prompt-groups 200
--preference-asr-samples-per-group 200
--preference-asr-prompt-variants original,direct
--long-samples 200
--text-samples 200
--multi-multiplier 2
```

## Subtests

- `asr.clean_read`: clean read speech from ASR leaderboard LibriSpeech.
- `asr.clean_conversational`: clean conversational speech from AMI.
- `asr.clean_media`: clean media-style speech from TED-LIUM and GigaSpeech.
- `asr.short`: very short speech, capped by `--short-max-seconds`.
- `asr.noisy_conversational`: AMI mixed with MUSAN noise at fixed SNRs.
- `asr.noisy_media`: media-style speech mixed with MUSAN noise at fixed SNRs.
- `asr.long`: stitched 20, 40, and 60 minute recordings.
- `hallucination.nonspeech`: MUSAN non-speech audio with production and
  explicit-abstention prompts.
- `prompt_robustness`: the same ASR examples with multiple text prompts.
- `audio_instruction_following.preference_asr.normalization`
- `audio_instruction_following.preference_asr.entities`
- `audio_instruction_following.preference_asr.disfluencies`
- `audio_instruction_following.preference_asr.case`
- `audio_instruction_following.preference_asr.standard`
- `context_biasing.contextless`
- `context_biasing.coarse`
- `context_biasing.fine`
- `text.superficial`: small GPQA multiple-choice checks.

The multilingual suite includes the English-style groups plus
`asr.clean_multilingual` from FLEURS/Covost2 and extra multilingual prompt
robustness rows.

## Preference-ASR

Preference-ASR is the intended source for the audio-instruction-following
groups. NTT-SMOKE samples each `preference_type` separately and expands each
selected row into prompt variants. The original Preference-ASR instruction is
preserved, and the direct variant wraps it as an explicit preference command.

Preference-ASR rows are scored with the Preference-ASR preference-aware
normalizer from `NTT_SMOKE_PREFERENCE_ASR_DIR/normalizer`. This matters because
generic ASR normalization would erase exactly the punctuation, casing,
normalization, disfluency, or entity preference being tested.

## Metrics

ASR-style groups report:

- `wer_macro`: unweighted average per-row WER.
- `wer`: corpus WER from summed edit counts.
- `substitutions`, `insertions`, `deletions`: raw corpus edit operations.
- `ref_words`: reference word count.
- `correct_words`: aligned hit count.
- `success_rate`: fraction of rows with WER below the evaluator threshold.
- `*_ci95`: normal-approximation 95% confidence-interval half-widths for
  subset macro WER and rate metrics with at least two observations.

Specialized groups also report hallucination rate, strict non-empty output
rate, context-biasing named-entity WER/FNR, prompt WER delta, prompt text-match
rate, language macro WER, and punctuation/capitalization metrics where
available.

## Reproducibility

Prepare source datasets first, then run:

```bash
export NTT_SMOKE_SOURCE_DATA_DIR=/path/to/skills_data
export NTT_SMOKE_PREFERENCE_ASR_DIR=/path/to/preference-asr-bench

ns prepare_data ntt-smoke --data_dir "$NTT_SMOKE_SOURCE_DATA_DIR"
```

Baseline reports should record the inference path used to produce the numbers.
The initial Qwen ASR and Nemotron Omni baselines are self-hosted cluster runs,
not hosted `inference.nvidia.com` calls: Qwen ASR loads
`Qwen/Qwen3-ASR-1.7B` from a local Hugging Face snapshot/cache, and Nemotron
Omni is served from an audio-capable vLLM/OpenAI-compatible server using
`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`. Hosted
`inference.nvidia.com` runs should be reported under a distinct run label so
API/backend provenance is not mixed with self-hosted results.

The prepare script uses stable SHA-based sampling, so the same source manifests
and options produce the same rows. Generated noisy and long-form audio is
written under `ntt-smoke/data`. Preference-ASR rows preserve absolute paths
under `NTT_SMOKE_PREFERENCE_ASR_DIR`, so that directory must be readable or
mounted during evaluation.

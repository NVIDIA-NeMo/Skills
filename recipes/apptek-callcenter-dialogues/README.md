# AppTek Call-Center Dialogues — Parakeet v3 reproduction

`reproduce_parakeet_v3.py` is a standalone validator that transcribes the AppTek
manifest with `nvidia/parakeet-tdt-0.6b-v3` and scores it with the same
normalization used by the benchmark. It is *not* part of the NeMo Skills
evaluation pipeline; it exists so a contributor can cross-check the AppTek WER
against the dataset paper.

Two modes are supported:

- `--mode direct` (default): transcribe each full-channel WAV directly, with
  the model-card local-attention setting for long audio.
- `--mode silero`: reproduce the AppTek paper's Silero-VAD-segmented condition.

The script depends on `nemo_toolkit[asr]` and (for `silero` mode)
`silero-vad`. Install those into your environment before running.

```bash
python recipes/apptek-callcenter-dialogues/reproduce_parakeet_v3.py \
    --mode direct \
    --work-dir <path-outside-repo>
```

`--manifest` defaults to
`nemo_skills/dataset/apptek-callcenter-dialogues/test.jsonl`, so run
`ns prepare_data apptek-callcenter-dialogues` first.

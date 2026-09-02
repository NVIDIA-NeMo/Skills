# Nemotron-IMO-TTS

Nemotron-IMO-TTS is the natural-language proof pipeline that scored 30 of 42 points at IMO 2026, released with
the report [*An Open Recipe for IMO Gold: Training Nemotron for Olympiad Mathematics*](https://github.com/NVIDIA-NeMo/Skills/blob/main/recipes/nemotron-imo-tts/paper.pdf).
Three Nemotron 3 Ultra
checkpoints (the general-availability model and the released RL and SFT specialists) generate candidate proofs
from eight complementary prompts, a two-checkpoint verifier panel scores every candidate and writes critiques,
and up to seven refinement rounds revise the best candidates. A separate high-compute stage then grades the
finalists with an IMO-style judge prompt and picks the proof to submit.

The recipe talks to any OpenAI-compatible chat-completions endpoint (a hosted API, a gateway, or local vLLM
servers), persists every request as a resumable JSONL row, and ships the report's prompts byte for byte.

For the walkthrough, configuration reference, output layout, and resume semantics see
[recipes/nemotron-imo-tts/README.md](https://github.com/NVIDIA-NeMo/Skills/blob/main/recipes/nemotron-imo-tts/README.md)
in the repository. The proofs submitted at IMO 2026 are in
[recipes/nemotron-imo-tts/imo-proofs](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/nemotron-imo-tts/imo-proofs).

## Run it

```bash
cp recipes/nemotron-imo-tts/configs/imo2026-ensemble.yaml my-run.yaml   # fill in the <your-...> placeholders
export NVIDIA_API_KEY=...
python recipes/nemotron-imo-tts/run.py --config my-run.yaml --output-dir runs/my-run --dry-run
python recipes/nemotron-imo-tts/run.py --config my-run.yaml --output-dir runs/my-run
```

`runs/my-run/submissions.jsonl` holds the selected proof per problem; `results.jsonl` holds the full record.
Relaunching the same command resumes from the persisted rows.

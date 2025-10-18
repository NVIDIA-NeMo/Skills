# ASR and TTS Generation with Riva NIMs

This recipe provides generation modules for Text-to-Speech (TTS) and Automatic Speech Recognition (ASR) using NVIDIA Riva NIM containers. It integrates with the NeMo Skills generation pipeline to provide batch processing, chunking support, and skip-filled functionality.


## Prerequisites

### NGC API Key
The `NVIDIA_API_KEY or NGC_API_KEY` environment variable must be present in your cluster configuration to pull Riva NIM containers from NGC. Add it to your cluster config file (e.g., `cluster_configs/dfw.yaml`):

```yaml
env_vars:
  NGC_API_KEY: "your-ngc-api-key-here"
```

### Dependencies
The Riva client library will be installed automatically via the `installation_command` in the scripts:
```bash
pip install nvidia-riva-client==2.21.1
```

## Usage

### TTS Generation

Run TTS generation on a cluster:
```bash
./scripts/run_tts_nim_cluster.sh
```

### ASR Transcription

Run ASR transcription on a cluster:
```bash
./scripts/run_asr_nim_cluster.sh
```

## SLURM Tests

There are SLURM tests which you can use to verify your setup. Read the following docs for instructions how to run them:

* [ASR](tests/slurm-tests/asr_nim/README.md)
* [TTS](tests/slurm-tests/tts_nim/README.md)


## Input File Formats

### TTS Input

For standard TTS with predefined voices, create a JSONL file:

```jsonl
{"text": "Hello, this is a test of the text to speech system."}
{"text": "How are you doing today?", "voice": "Magpie-Multilingual.EN-US.Sofia"}
{"text": "The quick brown fox jumps over the lazy dog.", "language_code": "en-US"}
{"text": "Custom parameters per sample.", "sample_rate_hz": 22050}
```

### ASR Input

For ASR transcription, provide paths to audio files:

```jsonl
{"audio_path": "/path/to/audio/sample1.wav"}
{"audio_path": "/path/to/audio/sample2.wav"}
```
## Tested containers

TTS: nvcr.io/nvstaging/nim/magpie-tts-multilingual:1.3.0-34013444

ASR: nvcr.io/nim/nvidia/parakeet-tdt-0.6b-v2:1.0.0

### Batch Processing
The recipe supports:
- `--num_chunks`: Split input into chunks for parallel processing
- `--skip_filled`: Skip already processed samples
- Async processing for improved throughput

## Support

For issues or questions, refer to:
- [NVIDIA Riva Documentation](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/index.html)
- [NeMo Skills Documentation](https://github.com/NVIDIA-NeMo/Skills)

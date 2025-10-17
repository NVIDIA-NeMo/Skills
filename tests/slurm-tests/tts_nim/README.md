# TTS NIM Slurm Test

This test validates the TTS (Text-to-Speech) NIM implementation using the Magpie TTS Multilingual container.

## Test Modes

The test can run in three different modes:

### 0. Full command example

```bash
WORKSPACE="/experiments/tts_nim_full_test_$(date +%Y%m%d_%H%M%S)" && \
python tests/slurm-tests/tts_nim/run_test.py \
--workspace "$WORKSPACE" \
--cluster name  \
--expname_prefix tts-full-test-$(date +%H%M%S) \
--mode full \
--config_file recipes/asr_tts/nim_configurations.py  \
--config_key "magpie-tts-multilingual:1.3.0-34013444"

```

### 1. Server Only Mode
Starts only the TTS NIM server. Handy if you want to debug client. Set mode parameter to `--mode server`.


### 2. Generation Only Mode
Runs generation using an existing/running TTS NIM server. You will need to know the node where the server is running.

**Command**

```bash
WORKSPACE="/experiments/tts_nim_full_test_$(date +%Y%m%d_%H%M%S)" && \
python tests/slurm-tests/tts_nim/run_test.py \
--workspace "$WORKSPACE" \
--cluster name \
--expname_prefix tts-final-$(date +%H%M%S) \
--mode generation \
--server_host cw-dfw-h100-004-211-033 \
--server_port 5000 \
--server_node cw-dfw-h100-004-211-033
```

**Validation**:
- Audio files generated for all 6 test samples
- All audio files have non-zero size

### 3. Full Pipeline Mode (Default)
The command from the 0th point. Starts the server and runs generation in a single command (similar to `ns generate` with `--server_gpus` not 0).

**Validation**:
- Audio files generated for all 6 test samples
- All audio files have non-zero size

## Configuration

If you want to experiment with images or different NIM parameters adjust them in this [configuration file](recipes/asr_tts/nim_configurations.py). Below is the default content.

``` python
nim_configurations = {
    "magpie-tts-multilingual:1.3.0-34013444": {
        "image_uri": "nvcr.io/nvstaging/nim/magpie-tts-multilingual:1.3.0-34013444",
        "server_args": "--nim-tags-selector batch_size=32 --nim-disable-model-download false",
        "server_entrypoint": "DISABLE_RIVA_REALTIME_SERVER=True python3 -m nemo_skills.inference.server.serve_riva_nim",
        "installation_command": "pip install nvidia-riva-client==2.21.1",
        "local_images": {} # items should be "cluster_name": "path_to_image.sqsh"
    },
    "parakeet-tdt-0.6b-v2:1.0.0": {
        "image_uri": "nvcr.io/nim/nvidia/parakeet-tdt-0.6b-v2:1.0.0",
        "server_args": "--CONTAINER_ID parakeet-tdt-0.6b-v2 --nim-tags-selector name=parakeet-tdt-0.6b-v2,mode=ofl  --nim-disable-model-download false",
        "server_entrypoint": "python3 -m nemo_skills.inference.server.serve_riva_nim",
        "installation_command": "pip install nvidia-riva-client==2.21.1",
        "local_images": {} # items should be "cluster_name": "path_to_image.sqsh"
    }
}
```

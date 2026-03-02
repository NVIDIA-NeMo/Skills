# Unified TTS Slurm Test

`run_test.py` defaults to:

- `--server_container nvcr.io/nvidia/nemo:25.11`
- `--model nvidia/magpie_tts_multilingual_357m`
- `--codec_model nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps`

## Temporary note about `--code_path`

The current `magpie_tts` unified backend imports `magpietts_inference` from NeMo.
In current test environments, this module is not consistently available from the stock container alone.

For now, pass a NeMo code tree explicitly via `--code_path`.
When a newer NeMo image includes the required Magpie TTS modules, this manual `--code_path` override should no longer be necessary.

## Local image usage (recommended for current cluster runs)

Keep the default in code as NVCR, but override `--server_container` at runtime.

### DFW example

```bash
python tests/slurm-tests/unified_tts/run_test.py \
  --cluster dfw \
  --config_dir "$PWD/cluster_configs" \
  --workspace /lustre/fsw/portfolios/convai/users/<user>/experiments/dialog_scripts2tts/unified_tts_dfw \
  --expname_prefix unified_tts_dfw \
  --server_container /lustre/fsw/portfolios/convai/users/<user>/workspace/images/nemo-25.11.sqsh \
  --code_path /lustre/fsw/portfolios/convai/users/<user>/workspace/code/NeMo_tts
```

### IAD (draco_oci) example

```bash
python tests/slurm-tests/unified_tts/run_test.py \
  --cluster iad \
  --config_dir "$PWD/cluster_configs" \
  --workspace /lustre/fsw/portfolios/llmservice/users/<user>/experiments/dialog_scripts2tts/unified_tts_iad \
  --expname_prefix unified_tts_iad \
  --server_container /lustre/fsw/portfolios/llmservice/users/<user>/workspace/images/nemo-25.11.sqsh \
  --code_path /lustre/fsw/portfolios/llmservice/users/<user>/workspace/code/NeMo_tts
```

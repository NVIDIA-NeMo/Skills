# AceProof TTS (NeMo-Skills)

This recipe implements the DeepSeek-Math-V2 proof generation -> verification -> refinement loop
using NeMo-Skills `generate` and `run_cmd` with SLURM dependencies.

## Quick start

```bash
python /nemo_run/code/recipes/aceproof-tts/pipeline/run_pipeline.py \
  --config /nemo_run/code/recipes/aceproof-tts/configs/aceproof-tts.yaml
```

Edit `configs/aceproof-tts.yaml` to switch `model_profile` between `deepseek_v32` (sglang)
and `nano_v3` (vllm). For stage-specific routing, set `model_profile_by_stage` with
`proof_gen`, `verify`, and `refine`.

Each model profile supports `system_prompt` (inline text) or `system_prompt_path` (file),
which is prepended as a system message to the model input. If you need different system
prompts per stage, define multiple model profiles and point each stage to the desired profile.

## Outputs

All outputs are under:

```
output_dir/rounds/R*/{proof_gen,refine,verify}/output.jsonl
```

Proof pool is stored in `output_dir/proof_pool`.

## Resume and skip behavior

The pipeline now writes `.done` markers for each stage:
- `rounds/R*/proof_gen/input.jsonl.done` (prepare)
- `rounds/R*/proof_gen/output.jsonl.done`
- `rounds/R*/verify/input.jsonl.done` (aggregate)
- `rounds/R*/verify/output.jsonl.done`
- `rounds/R*/refine/input.jsonl.done`
- `rounds/R*/refine/output.jsonl.done`
- `proof_final.done`

On resubmission, stages with existing `.done` files are skipped entirely (no jobs submitted).
To force a re-run, delete the corresponding `.done` file or set `pipeline.rerun_done: true`
(this also disables `skip_filled` for generation so outputs are regenerated).

## Single-experiment scheduling (optional)

By default each stage is submitted as its own nemo-run experiment. For large runs this can
add noticeable delays between stages because dependencies are resolved by querying the previous
experiment status. You can avoid that by submitting all stages into one experiment and using
internal task handles for dependencies.

Enable this in the config:

```yaml
pipeline:
  single_experiment: true
  experiment_name: aceproof_pipeline_imoproofbench   # optional
```

Or via CLI:

```bash
python /nemo_run/code/recipes/aceproof-tts/pipeline/run_pipeline.py \
  --config /nemo_run/code/recipes/aceproof-tts/configs/aceproof-tts.yaml \
  --single_experiment \
  --experiment_name aceproof_pipeline_imoproofbench
```

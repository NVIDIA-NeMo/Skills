# Math (formal language)

We support formal-math evaluation in Lean 4. The task is to generate a Lean 4 proof of a given formal statement.

Under the hood we assemble the final Lean 4 program and evaluate it in a sandboxed Lean 4 environment, capturing a proof status for each item and writing it back to the originating files.

## How we execute and evaluate

Evaluation constructs a complete Lean 4 file from the model output and dataset metadata, then uses a sandboxed Lean 4 checker to validate it.

At a high level, for each JSONL line in your result manifests we do the following:

1. Build the final Lean program to check.
   All the parameters mentioned here are *evaluation* parameters which are controlled with `--extra_eval_args="++eval_config.<parameter_name>=<parameter_value>`
      - Normalize the model output to keep only the intended Lean code. By default we only extract generation after `**FINAL ANSWER**`, but you can change the split phrase with `final_answer_key` parameter. If the `final_answer_key` phrase isn't present, we use the full generation instead.
      - Insert the dataset-provided header (imports and configuration) to ensure a consistent environment (always included from dataset).
      - Use the dataset’s original theorem (formal statement). If the model output includes a theorem declaration, we strip it and replace it with the canonical statement from the dataset to prevent statement tampering (controlled by `restate_formal_statement`; stripping governed by  `strip_theorem_from_proof`).
      - If you're changing default prompt, it's best to ask the model to emit only the proof body; any generated theorem declaration will be replaced as above.

2. Run the assembled program in the Lean sandbox
      - The sandboxed checker returns a status indicating success, error, or timeout. If the proof typechecks but uses `sorry`, this is recorded explicitly.

3. Persist results
      - We write back the assembled proof and its `proof_status` into the same JSONL files, replacing them atomically.

## Key configuration and evaluation considerations

Important configuration options and considerations for reliable Lean 4 evaluation:

- Producing only the proof body
  - Encourage the model to output only the proof body. If a theorem header is generated, it will be stripped and the dataset statement will be used.
- Providing a preassembled proof
  - If the line already includes a complete proof artifact, it will be used directly; otherwise the proof is assembled from the model’s generated text and dataset metadata.
- `restate_formal_statement` (default: True)
  - Controls whether the dataset’s `formal_statement` is inserted before the proof. Keeping this enabled enforces the canonical theorem; disabling it relies on the model’s emitted statement and is generally not recommended for benchmarking.
- `timeout` (default: 30.0 seconds)
  - Per-item execution timeout. A timeout returns `proof_status='timeout'`.
- Stateless execution for Lean 4
  - Evaluation is stateless; prior context is not persisted across items.


## Supported benchmarks

### minif2f

- Benchmark is defined in [`nemo_skills/dataset/minif2f/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/minif2f/__init__.py)
- Original benchmark source is [here](https://github.com/openai/miniF2F).

### putnam-bench

- Benchmark is defined in [`nemo_skills/dataset/putnam-bench/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/putnam-bench/__init__.py)
- Original benchmark source is [here](https://github.com/trishullab/PutnamBench).

### proofnet

- Benchmark is defined in [`nemo_skills/dataset/proofnet/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/proofnet/__init__.py)
- Original benchmark source is [here](https://github.com/zhangir-azerbayev/ProofNet).

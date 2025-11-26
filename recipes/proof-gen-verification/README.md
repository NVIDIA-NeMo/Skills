# Scaling Generative Verifiers For Natural Language Mathematical Proof Verification And Selection


## Prerequisites

To get started with NeMo-Skills, follow the instructions [here](https://nvidia-nemo.github.io/Skills/basics). Once you have set up your cluster configs, you can proceed with the steps below.

## Dataset Preparation

First, generate the dataset files. Run the following command for each dataset you need. For instance, for `aime25`:

```bash
python3 nemo_skills/dataset/aime25/prepare.py
```

**Available datasets:**
- `proof-bench-judge`
- `proof-arena-judge`
- `open-proof-corpus-judge`
- `aime25`
- `challenge19`

## Proof Verification

Run the `run_evals` stage to evaluate proof verification baselines. For example, to run proof verification on `proof-arena-judge`:

```bash
python3 recipes/proof-gen-verification/pipeline/eval_judge.py --stages run_evals ++eval_name=proof-arena-judge
```

> **Note:** By default, this runs proof verification for all prompts and models across 32 seeds. To customize the configuration, modify the model configs or number of seeds in `configs/judge-eval.yaml`.

## Proof Selection

Run the `generic_bon_eval` stage to evaluate proof selection baselines. For example, to run proof selection on `proof-bench-judge`:

```bash
python3 recipes/proof-gen-verification/pipeline/eval_judge.py --stages generic_bon_eval ++eval_name=proof-bench-judge
```

## Test-Time Compute Methods

The `run_end_to_end_eval` stage provides end-to-end proof generation and selection. For example, to generate and select proofs for `challenge19`:

```bash
python3 recipes/proof-gen-verification/pipeline/eval_judge.py --stages run_end_to_end_eval ++end_to_end_eval=challenge19
```

This runs the proposed hybrid genselect and LLM-as-a-judge methods across multiple settings. To run a specific setting, modify the `run_end_to_end_eval.runs` config in the config file.

> **Note:** The evaluation script launched at the end only works for integer-final-answer problems. For proof-based tasks without ground-truth answers, this evaluation step will fail.

## Other Experiments

Additional experiments are available, including:
- Step agent
- Judgement genselect
- Balanced final-answer proof generation

You can run these experiments using their respective stages in a similar manner to the examples above.

## Notes

- **Hardware requirements:** All scripts assume each node has 8 GPUs with at least 80GB memory per GPU. You can configure the number of GPUs and nodes in `configs/judge-eval.yaml` and `pipeline/eval_judge.py`.

- **Parallelization:** The config script supports dividing datasets into chunks for maximum parallelization, which is useful when running across multiple GPU nodes.

- **Model configurations:** We provide configurations (number of GPUs, nodes, temperature, etc.) for supported models in `pipeline/eval_judge.py`. To run other models, define their configurations in `MODEL_CONFIGS`.

- **Custom datasets:** You can evaluate custom datasets as long as they follow the standard NeMo-Skills format and include the required fields for each script.

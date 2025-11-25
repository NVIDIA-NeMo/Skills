To get started with Nemo-Skills please follow the instructions [here](https://nvidia-nemo.github.io/Skills/basics). Once you setup your cluster configs, you can proceed with the following.

## Dataset preparation

First, you need to generate the dataset data files. For each dataset, you need the following command to populate the data file:

```bash
python3 nemo_skills/dataset/{dataset}/prepare.py
```
where `{dataset}` is one of `proof-bench-judge`, `proof-arena-judge`, `open-proof-corpus-judge`, `aime25`, and `challenge19`.

## Proof Verification

You can run the `run_evals` to run proof verification baselines. For instance, the following runs proof verification for `proof-arena-judge`
```bash
python3 recipes/proof-gen-verification/pipeline/eval_judge.py --stages run_evals ++eval_name=proof-arena-judge
```
Note that the above code runs proof verification for all possible prompts and models for 32 seeds. If you need a particularl config, you can change the model configs or number of seeds in `configs/judge-eval.yaml`.

## Proof Selection

You can run the `generic_bon_eval` pipeline to run proof selection baselines. For instance, the following runs proof selection for the `proof-bench-judge` dataset.
```bash
python3 recipes/proof-gen-verification/pipeline/eval_judge.py --stages generic_bon_eval ++eval_name=proof-bench-judge
```

## Test-Time Compute methods

The `run_end_to_end_eval` stage provides the end-to-end proof generation and selection method. For instance, to generate proofs, and select them for `challenge19` you can run:
```bash
python3 recipes/proof-gen-verification/pipeline/eval_judge.py --stages generic_bon_eval ++end_to_end_eval=challenge19
```
This will run the proposed hyrbric genselect and llm-as-a-judge methods for many settings. If you need only one particular setting, you need to change the `run_end_to_end_eval.runs` config in the config file. Once the runs are finished, the stage launches an evaluation script as well, but it only works for integer-final-answer problems, and for proofs this will fail as there is no ground-truth to compare with.


## Other experiments

We have provided code and prompts to other experiments such as step agent, judgement genselect, and balanced final-answer proof generation scripts as well, which you can run their stages in a similar way as above.

## Notes

* For all the scripts, we assume each node has 8 gpus with at least 80GB memory per GPU. You can configure the number of gpus and nodes in `configs/judge-eval.yaml` and `pipeline/eval_judge.py`.
* The config script is configurable to divide the datasets into many chunks for maximum parallelization. This is particularly useful to run over several GPU nodes.
* We provide configurations (such as number of gpus, nodes, temperature) for the models we use in `pipeline/eval_judge.py`. If you need to run other models, you need to define their configurations in `MODEL_CONFIGS`.
* You can evaluate custom datasets as long as they are prepared as standard nemo-skills format, and have the required fields by each script.

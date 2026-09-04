# GenCorrect

GenCorrect runs iterative solution improvement with the `nemotron-ccc-ultra-nvfp4` model. Each round generates 200 candidates per problem, forms a score-blind `similarity10` shortlist, treats all ten shortlisted candidates as submissions, and carries the highest-scoring submission forward. Three `gap_targeted` references are shown in the next round.

Prepare a mounted data directory containing:

```text
<data-dir>/ccc/gencorrect.jsonl
<data-dir>/ccc/gencorrect_metadata.json
```

The JSONL must contain one row per problem with `id`, `problem_id`, `subtask`, and `problem`. The metadata file uses the [CCC evaluator](../../nemo_skills/evaluation/evaluator/ccc.py) format.

Run five rounds:

```bash
python recipes/gencorrect/run.py \
  --cluster <cluster-config> \
  --config-dir <cluster-config-directory> \
  --data-dir <mounted-data-directory> \
  --output-dir <mounted-output-directory> \
  --server-container <vllm-container>
```

The cluster config must mount the model, data, and output paths. Pass `--model` when the mounted model path is not `nemotron-ccc-ultra-nvfp4`. Re-running the command safely fills unfinished generations.

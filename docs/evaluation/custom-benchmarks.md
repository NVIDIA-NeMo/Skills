# Custom benchmarks

NeMo-Skills supports defining benchmarks in external repositories. This lets you
keep proprietary data private, iterate on benchmarks independently of NeMo-Skills
releases, and share team-owned benchmarks without modifying the main repository.

An external benchmark can customize every part of the evaluation pipeline:
dataset preparation, prompt template, generation logic, evaluator, and metrics.

## Quick start

1. **Create a repo** with `benchmark_map.json`, a dataset `__init__.py`, and a `prepare.py`.
2. **Set the env var** `NEMO_SKILLS_EXTRA_BENCHMARK_MAP` to point at your `benchmark_map.json` (`name -> path` structure).
3. **Install** the repo (`pip install -e .`) so that Python can import your modules.
4. **Run** `ns prepare_data <name>` and `ns eval --benchmarks=<name> ...` as usual.

The rest of this page walks through a complete example.

## Walkthrough: a "word_count" benchmark

We will build a small benchmark that asks a model to count the words in a sentence.
This is deliberately simple so the focus stays on the plugin wiring rather than
the task itself.

### Step 1 - Repository layout

```
my-benchmark-repo/
├── pyproject.toml
├── benchmark_map.json
└── my_benchmarks/
    ├── dataset/word_count/
    │   ├── __init__.py
    │   └── prepare.py
    ├── inference/word_count.py
    ├── evaluation/word_count.py
    ├── metrics/word_count.py
    └── prompt/eval/word_count/
        └── default.yaml
```

**`pyproject.toml`** - makes the repo installable so that `my_benchmarks.*` is
importable:

```toml title="pyproject.toml"
[project]
name = "my-benchmarks"
version = "0.1.0"
```

**`benchmark_map.json`** - maps short names to dataset directories (paths are
relative to this file):

```json title="benchmark_map.json"
{
    "word_count": "./my_benchmarks/dataset/word_count"
}
```

### Step 2 - Dataset `__init__.py`

This file tells the eval pipeline which prompt, evaluator, metrics, and generation
module to use by default. All of these can still be overridden from the command line.

```python title="my_benchmarks/dataset/word_count/__init__.py"
from nemo_skills.pipeline.utils.packager import (
    register_external_repo,
    RepoMetadata,
)
from pathlib import Path

# Register repo so it gets packaged inside containers.
# ignore_if_registered avoids errors when the module is imported more than once.
register_external_repo(
    RepoMetadata(name="my_benchmarks", path=Path(__file__).parents[2]),
    ignore_if_registered=True,
)

# Metrics class - use module::Class format for custom metrics
METRICS_TYPE = "my_benchmarks.metrics.word_count::WordCountMetrics"

# Default generation arguments
# prompt_config ending in .yaml is resolved relative to repo root
GENERATION_ARGS = (
    "++prompt_config=my_benchmarks/prompt/eval/word_count/default.yaml "
    "++eval_type=my_benchmarks.evaluation.word_count::WordCountEvaluator"
)

# Custom generation module (optional - remove this line to use the default)
GENERATION_MODULE = "my_benchmarks.inference.word_count"
```


### Step 3 - `prepare.py`

This script creates the test data. It is called by `ns prepare_data word_count`.

```python title="my_benchmarks/dataset/word_count/prepare.py"
import json
from pathlib import Path

SAMPLES = [
    {"sentence": "The quick brown fox", "expected_answer": 4},
    {"sentence": "Hello world", "expected_answer": 2},
    {"sentence": "NeMo Skills is great for evaluation", "expected_answer": 6},
    {"sentence": "One", "expected_answer": 1},
    {"sentence": "A B C D E F G", "expected_answer": 7},
]

if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    output_file = data_dir / "test.jsonl"
    with open(output_file, "wt", encoding="utf-8") as fout:
        for sample in SAMPLES:
            fout.write(json.dumps(sample) + "\n")
```

### Step 4 - Prompt template

Prompt configs live in your external repo and are referenced as a relative path
ending in `.yaml`. You can also reference it as an absolute path, which in container
would need to start with `/nemo_run/code/`, the directory where all packaged code lives.

```yaml title="my_benchmarks/prompt/eval/word_count/default.yaml"
user: |-
  Count the number of words in the quoted sentence below.
  Put your final answer (just the number) inside \boxed{{}}.

  {sentence}
```

In `GENERATION_ARGS` this is referenced as:

```
++prompt_config=my_benchmarks/prompt/eval/word_count/default.yaml
```

### Step 5 - Custom generation module (optional)

A custom generation module lets you change how the model is called - for example
to implement multi-step generation.

This example adds an optional **verify** step where the model is asked to double-check
its own answer.

```python title="my_benchmarks/inference/word_count.py"
import logging

import hydra

from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig
from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class WordCountGenerationConfig(GenerationTaskConfig):
    # Add a custom flag that controls whether to do a verification step
    verify: bool = False


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_generation_config", node=WordCountGenerationConfig)


class WordCountGenerationTask(GenerationTask):
    """Generation task with an optional verification step."""

    async def process_single_datapoint(self, data_point, all_data, prompt_format=None):
        # Step 1: normal generation
        result = await super().process_single_datapoint(data_point, all_data)

        if not self.cfg.verify:
            return result

        # Step 2: ask the model to verify its own answer
        verify_prompt = (
            f"You previously answered the following question:\n\n"
            f"{data_point['problem']}\n\n"
            f"Your answer was:\n{result['generation']}\n\n"
            f"Please verify this is correct. "
            f"If it is, repeat the same answer inside \\boxed{{}}. "
            f"If not, provide the corrected answer inside \\boxed{{}}."
        )
        new_data_point = [{"role": "user", "content": verify_prompt}]
        # We use prompt_format=openai as we already prepared the full message
        verify_result = await super().process_single_datapoint(
            new_data_point,
            all_data,
            prompt_format="openai",
        )
        # Replace generation with the verified answer
        result["generation"] = verify_result["generation"]
        return result


GENERATION_TASK_CLASS = WordCountGenerationTask


@hydra.main(version_base=None, config_name="base_generation_config")
def generate(cfg: WordCountGenerationConfig):
    cfg = WordCountGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    task = WordCountGenerationTask(cfg)
    task.generate()


if __name__ == "__main__":
    generate()
```

If you don't need custom generation logic, simply remove the `GENERATION_MODULE`
line from `__init__.py` and the default
[`nemo_skills.inference.generate`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/inference/generate.py)
module will be used.

### Step 6 - Custom evaluator

Here is an example of a basic evaluator class. The extra "predicted_answer" and "is_correct" fields will be added
to the output jsonl produced by the generation step.

```python title="my_benchmarks/evaluation/word_count.py"
import re

from nemo_skills.evaluation.evaluator.base import BaseEvaluator


class WordCountEvaluator(BaseEvaluator):
    async def eval_single(self, data_point):
        """Extract predicted answer and compare to expected."""
        match = re.search(r"\\boxed\{(\d+)\}", data_point["generation"])
        predicted = int(match.group(1)) if match else None

        return {
            "predicted_answer": predicted,
            "is_correct": predicted == data_point["expected_answer"],
        }
```

This is referenced in `GENERATION_ARGS` using the `module::Class` format:

```
++eval_type=my_benchmarks.evaluation.word_count::WordCountEvaluator
```

### Step 7 - Custom metrics

The metrics class reads the evaluated JSONL and computes summary statistics.

```python title="my_benchmarks/metrics/word_count.py"
from nemo_skills.evaluation.metrics.base import BaseMetrics


class WordCountMetrics(BaseMetrics):
    def _get_score_dict(self, prediction):
        return {"is_correct": prediction.get("is_correct", False)}

    def get_incorrect_sample(self, prediction):
        # used for automatic filtering data based on length
        # (we mark too long examples as incorrect using this method)
        prediction = prediction.copy()
        prediction["is_correct"] = False
        prediction["predicted_answer"] = None
        return prediction

    def update(self, predictions):
        # base class provides convenient helpers for calculating
        # common metrics like majority / pass
        super().update(predictions)
        predicted_answers = [pred["predicted_answer"] for pred in predictions]
        self._compute_pass_at_k(
            predictions=predictions,
            predicted_answers=predicted_answers,
        )
        self._compute_majority_at_k(
            predictions=predictions,
            predicted_answers=predicted_answers,
        )
```

Referenced in `__init__.py` as:

```
METRICS_TYPE = "my_benchmarks.metrics.word_count::WordCountMetrics"
```

### Step 8 - Running the benchmark

Install your repo and set the env var:

```bash
cd my-benchmark-repo
pip install -e .
export NEMO_SKILLS_EXTRA_BENCHMARK_MAP=$(pwd)/benchmark_map.json
```

Prepare the data:

```bash
ns prepare_data word_count
```

Run evaluation (using an API model as an example):

```bash
ns eval \
    --cluster=local \
    --server_type=openai \
    --model=nvidia/nemotron-3-nano-30b-a3b \
    --server_address=https://integrate.api.nvidia.com/v1 \
    --benchmarks=word_count \
    --output_dir=/workspace/test-eval
```

View results:

```bash
ns summarize_results --cluster=local /workspace/test-eval
```

!!! note

    You can also skip creating a benchmark map json, but then you'd need to reference
    benchmark as an absolute path to the folder with __init__.py and prepare.py


## Minimal example

If your benchmark can reuse built-in evaluation and metrics (e.g. the standard math
evaluator), you only need two files:

```python title="my_benchmarks/dataset/my_simple_bench/__init__.py"
METRICS_TYPE = "math"
GENERATION_ARGS = "++prompt_config=generic/math ++eval_type=math"
```

```python title="my_benchmarks/dataset/my_simple_bench/prepare.py"
import json
from pathlib import Path

if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    with open(data_dir / "test.jsonl", "wt", encoding="utf-8") as fout:
        fout.write(json.dumps({
            "problem": "What is 2 + 2?",
            "expected_answer": 4,
        }) + "\n")
```

And a `benchmark_map.json`:

```json
{
    "my_simple_bench": "./my_benchmarks/dataset/my_simple_bench"
}
```

No custom generation module, evaluator, or metrics needed.

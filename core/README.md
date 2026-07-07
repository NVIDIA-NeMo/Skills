# Core / Pipeline Dependency Boundary

NeMo Skills is split into **Core** (agent runtime) and **Pipeline** (orchestration). The rule is simple:

```
Pipeline can import from Core.
Core CANNOT import from Pipeline.
```

Core modules are everything under `nemo_skills/` **except** `nemo_skills/pipeline/`. They must never have top-level imports from `nemo_skills.pipeline` or `nemo_run`. This boundary is enforced by `tests/test_dependency_isolation.py` which verifies that core modules import successfully when `nemo_run` is blocked.

## Dependency placement

When adding a new dependency, put it in the right requirements file:

| If the dependency is needed for... | Add it to |
|---|---|
| Inference, evaluation, tool calling, any benchmark evaluator | `core/requirements.txt` |
| CLI commands (`ns`), cluster orchestration, experiment tracking | `requirements/pipeline.txt` |

There is no separate `main.txt` — `pyproject.toml` composes the default install from `core/requirements.txt` + `requirements/pipeline.txt`. Each dependency lives in exactly one file.

**Boundary definition:**

- **Core** = everything needed to run inference + evaluation locally (including all benchmark evaluator deps)
- **Pipeline** = orchestration-only deps (`nemo_run`, `typer`, `click`, `nemo-evaluator-launcher`)

All benchmark-specific dependencies (e.g., `faiss-cpu`, `sacrebleu`, `datasets`, `func-timeout`) go in `core/requirements.txt`. Eventually these should migrate to JIT (just-in-time) install so that benchmark deps are installed on demand at runtime, but until that is implemented, they must be in core so evaluators do not crash at runtime.

## Examples of correct placement

- `httpx` -> `core/requirements.txt` (used by model inference clients)
- `sympy` -> `core/requirements.txt` (used by math graders)
- `sacrebleu` -> `core/requirements.txt` (used by translation benchmark evaluator)
- `faiss-cpu` -> `core/requirements.txt` (used by BFCL benchmark evaluator)
- `nemo_run` -> `requirements/pipeline.txt` (cluster job orchestration)
- `wandb` -> `core/requirements.txt` (used by summarize-results)

## Examples of mistakes to avoid

- Adding `nemo_run` to `core/requirements.txt` -- it is a pipeline/orchestration dependency, core must not depend on it.
- Adding `typer` to `core/requirements.txt` -- it is the CLI framework, only used by the pipeline layer.

## Writing new core code

- If you need something from `nemo_skills.pipeline`, your code probably belongs in pipeline, not core. Move it.
- If you have a function that works locally but *also* needs a cluster variant, keep both paths in the same function but use a **lazy import** for the pipeline code inside the branch that needs it (see `dataset/utils.py:get_dataset_module` for the pattern). Never add a top-level import.
- The pipeline layer (`nemo_skills/pipeline/`) can provide thin wrappers or re-exports for convenience (see `pipeline/dataset.py`), but all local logic should live in core.

## Dataset loading example

The boundary shows up concretely in dataset loading:

```python
# Core: local-only dataset loading (no cluster deps)
from nemo_skills.dataset.utils import get_dataset_module
module, data_path = get_dataset_module("gsm8k")

# Pipeline: cluster-aware wrapper (SSH downloads, mount resolution)
from nemo_skills.pipeline.dataset import get_dataset_module
module, data_path = get_dataset_module("gsm8k", cluster_config=cfg)
```

The core version has zero pipeline imports. The pipeline wrapper delegates to core for local resolution and only adds cluster-specific logic (mount-path unmounting, SSH file downloads) when needed.

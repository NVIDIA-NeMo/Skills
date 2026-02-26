# Installation & Dependency Groups

NeMo Skills provides two installable packages:

- **`nemo-skills`** (root) -- full install with CLI, cluster orchestration, all benchmarks
- **`nemo-skills-core`** (`core/` subdirectory) -- lightweight runtime only

## Default installation

`pip install nemo-skills` gives you **everything** (inference, evaluation, CLI,
cluster orchestration, benchmarks):

```bash
pip install git+https://github.com/NVIDIA-NeMo/Skills.git
# or, from a local clone:
pip install -e .
```

## Lightweight installation

If you only need inference, evaluation, and tool calling (no cluster orchestration):

```bash
pip install "nemo-skills-core @ git+https://github.com/NVIDIA-NeMo/Skills.git#subdirectory=core"
# or, from a local clone:
pip install -e core/
```

## Extras (dependency groups)

| Extra | Requirements file | What it provides |
|-------|-------------------|------------------|
| `core` | `core/requirements.txt` | Agent runtime: inference, evaluation, tool calling (MCP), prompt formatting, math/code grading. No cluster orchestration. |
| `pipeline` | `requirements/pipeline.txt` | CLI (`ns` command), cluster management, experiment tracking (`nemo_run`, `typer`, `wandb`). |
| `dev` | `requirements/common-tests.txt`, `requirements/common-dev.txt` | Development and testing tools (`pytest`, `ruff`, `pre-commit`). |

### Examples

```bash
# Full install (default)
pip install -e .

# Core only -- lightweight runtime for downstream integrations
pip install -e core/

# Development (everything + dev tools)
pip install -e ".[dev]"
```

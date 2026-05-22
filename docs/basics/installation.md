# Installation & Dependency Groups

NeMo Skills provides three installable packages:

- **`nemo-skills`** (root) -- full install with CLI, cluster orchestration, all benchmarks
- **`nemo-skills-tools`** (`tools/` subdirectory) -- tool runtime only (`ToolManager`, built-in tools such as `DirectPythonTool`), without model-client dependencies such as LiteLLM/OpenAI
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

If you only need the tool runtime (`ToolManager` and built-in tools such as `DirectPythonTool`):

```bash
pip install "nemo-skills-tools @ git+https://github.com/NVIDIA-NeMo/Skills.git#subdirectory=tools"
# or, from a local clone:
pip install -e tools/
```

The current `tools` package is a Phase 1 split: it reuses the existing MCP/runtime layout as-is, so it may still install a few transitive runtime dependencies beyond the absolute minimum. It intentionally excludes model-client dependencies such as `litellm` and `openai`.

## Extras (dependency groups)

| Extra | Requirements file | What it provides |
|-------|-------------------|------------------|
| `tools` | `tools/requirements.txt` | Tool runtime: `ToolManager`, built-in MCP/direct tools, and sandbox-backed `DirectPythonTool`. No model-client dependencies such as LiteLLM/OpenAI. |
| `core` | `core/requirements.txt` | Agent runtime: inference, evaluation, tool calling (MCP), prompt formatting, math/code grading. No cluster orchestration. |
| `pipeline` | `requirements/pipeline.txt` | CLI (`ns` command), cluster management, experiment tracking (`nemo_run`, `typer`, `wandb`). |
| `dev` | `requirements/common-tests.txt`, `requirements/common-dev.txt` | Development and testing tools (`pytest`, `ruff`, `pre-commit`). |

### Examples

```bash
# Full install (default)
pip install -e .

# Core only -- lightweight runtime for downstream integrations
pip install -e core/

# Tools only -- tool runtime for downstream integrations
pip install -e tools/

# Development (everything + dev tools)
pip install -e ".[dev]"
```

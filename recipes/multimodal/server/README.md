# Unified Server and Backends

This directory contains a backend-agnostic inference server with an OpenAI Chat Completions compatible API, plus pluggable backends for different model families.

## Purpose

- Expose one HTTP surface (`/v1/chat/completions`) for multiple model types.
- Keep server concerns (request parsing, batching, response shaping) separate from model-specific logic.
- Let each backend define only model load/inference behavior behind a shared interface.

## Main Components

- `unified_server.py`: FastAPI server, OpenAI-compatible response format, batch scheduling.
- `backends/base.py`: shared data models and abstract interface:
  - `BackendConfig`
  - `GenerationRequest`
  - `GenerationResult`
  - `InferenceBackend`
  - `Modality`
- `backends/__init__.py`: backend registry and lazy loading (`BACKEND_REGISTRY`, `get_backend()`).
- `nemo_skills/inference/server/serve_unified.py`: CLI entrypoint used by local runs and cluster jobs.

## Built-in Backends

- `nemo_asr` -> `backends/nemo_asr_backend.py`
- `magpie_tts` -> `backends/magpie_tts_backend.py`

## How To Add a New Backend

1. Create a config dataclass inheriting `BackendConfig`.
2. Create a backend class inheriting `InferenceBackend`.
3. Register it in `backends/__init__.py` (`BACKEND_REGISTRY`).
4. Start server with `--backend <your_backend_name>` and pass backend-specific args.
5. Add tests (unit + slurm where applicable).

### Required Interface

Your backend class must implement:

- `@classmethod get_config_class(cls) -> type`
- `name` property
- `supported_modalities` property
- `load_model(self) -> None`
- `generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]`

You should also override as needed:

- `validate_request(self, request) -> Optional[str]` for strict input validation
- `health_check(self) -> Dict[str, Any]` for backend-specific health metadata
- `@classmethod get_extra_routes(cls, backend_instance)` if custom endpoints are needed

### Minimal Skeleton

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from recipes.multimodal.server.backends.base import (
    BackendConfig,
    GenerationRequest,
    GenerationResult,
    InferenceBackend,
    Modality,
)


@dataclass
class MyBackendConfig(BackendConfig):
    my_flag: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MyBackendConfig":
        known = {"model_path", "device", "dtype", "my_flag"}
        return cls(
            **{k: v for k, v in d.items() if k in known},
            extra_config={k: v for k, v in d.items() if k not in known},
        )


class MyBackend(InferenceBackend):
    @classmethod
    def get_config_class(cls) -> type:
        return MyBackendConfig

    @property
    def name(self) -> str:
        return "my_backend"

    @property
    def supported_modalities(self) -> Set[Modality]:
        return {Modality.TEXT}

    def load_model(self) -> None:
        self._model = ...
        self._is_loaded = True

    def generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        results = []
        for req in requests:
            try:
                text_out = f"echo: {req.text or ''}"
                results.append(GenerationResult(text=text_out, request_id=req.request_id))
            except Exception as e:
                results.append(GenerationResult(error=str(e), request_id=req.request_id))
        return results
```

Then register in `recipes/multimodal/server/backends/__init__.py`:

```python
BACKEND_REGISTRY = {
    # ...
    "my_backend": ("my_backend", "MyBackend"),
}
```

## Running Slurm Tests (Unified ASR and Unified TTS)

Activate env first:

```bash
source .venv/bin/activate
source ~/.env
```

Run unified ASR backend slurm test:

```bash
python tests/slurm-tests/unified_asr/run_test.py \
  --workspace /lustre/fsw/portfolios/convai/users/$USER/experiments/slurm-tests/unified_asr \
  --cluster dfw \
  --expname_prefix unified_asr_test \
  --server_container /lustre/fsw/portfolios/convai/users/$USER/workspace/images/nemo-25.11.sqsh
```

Run unified TTS backend slurm test:

```bash
python tests/slurm-tests/unified_tts/run_test.py \
  --workspace /lustre/fsw/portfolios/convai/users/$USER/experiments/slurm-tests/unified_tts \
  --cluster dfw \
  --expname_prefix unified_tts_test \
  --server_container /lustre/fsw/portfolios/convai/users/$USER/workspace/images/nemo-25.11.sqsh \
  --code_path /lustre/fsw/portfolios/convai/users/$USER/workspace/code/NeMo_tts
```

Optional flags for both:

- `--skip_check` to skip checker job
- `--config_dir` to select a different cluster config directory

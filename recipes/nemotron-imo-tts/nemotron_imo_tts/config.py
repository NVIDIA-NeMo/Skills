# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run configuration: YAML -> validated frozen dataclasses, derived counts, experiment keys."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

KEYLESS_API_KEY = "EMPTY"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, kw_only=True)
class EndpointDefaults:
    base_url: str
    api_key_env: str | None
    timeout_s: float = 14400.0
    retry_window_s: float = 3600.0
    backoff_base_s: float = 5.0
    backoff_max_s: float = 120.0
    extra_body: dict = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ModelSpec:
    id: str
    model: str
    max_len: int
    max_tokens: int
    base_url: str
    api_key_env: str | None


@dataclass(frozen=True, kw_only=True)
class Checkpoint(ModelSpec):
    generation_samples_per_prompt: int
    refinement_samples_per_prompt: int


@dataclass(frozen=True, kw_only=True)
class Verifier(ModelSpec):
    judgments_per_proof: int


@dataclass(frozen=True, kw_only=True)
class Judge(ModelSpec):
    judgments_per_finalist: int


@dataclass(frozen=True, kw_only=True)
class SearchConfig:
    generation_prompts: tuple
    max_rounds: int
    refinement_prompts_per_round: int


@dataclass(frozen=True, kw_only=True)
class Sampling:
    temperature: float
    top_p: float


@dataclass(frozen=True, kw_only=True)
class Concurrency:
    problems: int
    generation: int
    verification: int
    refinement: int
    judging: int


@dataclass(frozen=True, kw_only=True)
class ContextBudget:
    tokenizer: str
    safety_margin_tokens: int


@dataclass(frozen=True, kw_only=True)
class InputConfig:
    path: str


@dataclass(frozen=True, kw_only=True)
class Config:
    endpoint: EndpointDefaults
    checkpoints: tuple
    verifiers: tuple
    judges: tuple | None
    search: SearchConfig
    sampling: Sampling
    concurrency: Concurrency
    context_budget: ContextBudget | None
    input: InputConfig

    @property
    def checkpoint_order(self):
        return [c.id for c in self.checkpoints]

    @property
    def verifier_order(self):
        return [v.id for v in self.verifiers]

    def checkpoint(self, model_id):
        return next(c for c in self.checkpoints if c.id == model_id)


def _require(mapping, key, location, expected_type):
    if not isinstance(mapping, dict) or key not in mapping:
        raise ConfigError(f"{location}: missing required key '{key}'")
    return _typed(mapping[key], f"{location}.{key}", expected_type)


def _typed(value, location, expected_type):
    if expected_type is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if isinstance(value, bool) and expected_type in (int, float):
        raise ConfigError(f"{location}: expected {expected_type.__name__}, got a boolean")
    if not isinstance(value, expected_type):
        raise ConfigError(f"{location}: expected {expected_type.__name__}, got {type(value).__name__}")
    return value


def _positive(value, location):
    if value <= 0:
        raise ConfigError(f"{location}: must be positive, got {value!r}")
    return value


def _optional(mapping, key, location, expected_type, default=None):
    if not isinstance(mapping, dict) or key not in mapping or mapping[key] is None:
        return default
    return _typed(mapping[key], f"{location}.{key}", expected_type)


def _placeholder(value):
    return isinstance(value, str) and value.startswith("<") and value.endswith(">")


def _reject_unknown(mapping, allowed, location):
    if not isinstance(mapping, dict):
        raise ConfigError(f"{location}: expected a mapping")
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ConfigError(f"{location}: unknown key(s) {unknown}")


MODEL_KEYS = ("id", "model", "max_len", "max_tokens", "base_url", "api_key_env")


def _model_common(raw, location, defaults):
    model_id = _require(raw, "id", location, str).strip()
    if not model_id:
        raise ConfigError(f"{location}.id: must be a non-empty string")
    model = _require(raw, "model", location, str)
    if not model.strip() or _placeholder(model):
        raise ConfigError(f"{location}.model: fill in the served model name (got {model!r})")
    max_len = _positive(_require(raw, "max_len", location, int), f"{location}.max_len")
    max_tokens = _positive(_require(raw, "max_tokens", location, int), f"{location}.max_tokens")
    if max_tokens >= max_len:
        raise ConfigError(f"{location}: max_tokens ({max_tokens}) must be smaller than max_len ({max_len})")
    base_url = _optional(raw, "base_url", location, str, defaults.base_url)
    if _placeholder(base_url):
        raise ConfigError(f"{location}.base_url: fill in the endpoint base URL (got {base_url!r})")
    # A present null means the server needs no key; an absent key inherits the endpoint default.
    api_key_env = _optional(raw, "api_key_env", location, str) if "api_key_env" in raw else defaults.api_key_env
    return dict(
        id=model_id, model=model, max_len=max_len, max_tokens=max_tokens, base_url=base_url, api_key_env=api_key_env
    )


def _model_list(raw, location, defaults, builder, count_fields, required=True):
    if raw is None:
        if required:
            raise ConfigError(f"{location}: at least one entry is required")
        return None
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{location}: expected a non-empty list")
    specs = []
    ids = set()
    for idx, entry in enumerate(raw):
        loc = f"{location}[{idx}]"
        _reject_unknown(entry, MODEL_KEYS + tuple(count_fields), loc)
        common = _model_common(entry, loc, defaults)
        if common["id"] in ids:
            raise ConfigError(f"{loc}.id: duplicate id {common['id']!r}")
        ids.add(common["id"])
        counts = {name: _positive(_require(entry, name, loc, int), f"{loc}.{name}") for name in count_fields}
        specs.append(builder(**common, **counts))
    return tuple(specs)


RECIPE_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_config(path, prompts_dir=RECIPE_PROMPTS_DIR):
    """Load and validate a run config. Prompt ids are checked against the recipe's prompts directory."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: the config must be a mapping")
    return parse_config(raw, prompts_dir=prompts_dir)


def parse_config(raw, prompts_dir=None):
    _reject_unknown(
        raw,
        (
            "endpoint",
            "checkpoints",
            "verifiers",
            "judges",
            "search",
            "sampling",
            "concurrency",
            "context_budget",
            "input",
        ),
        "config",
    )
    ep = raw.get("endpoint")
    if not isinstance(ep, dict):
        raise ConfigError("endpoint: missing section")
    _reject_unknown(
        ep,
        ("base_url", "api_key_env", "timeout_s", "retry_window_s", "backoff_base_s", "backoff_max_s", "extra_body"),
        "endpoint",
    )
    base_url = _require(ep, "base_url", "endpoint", str)
    if _placeholder(base_url):
        raise ConfigError(f"endpoint.base_url: fill in the endpoint base URL (got {base_url!r})")
    extra_body = _optional(ep, "extra_body", "endpoint", dict, {})
    defaults = EndpointDefaults(
        base_url=base_url,
        api_key_env=_optional(ep, "api_key_env", "endpoint", str),
        timeout_s=_positive(_optional(ep, "timeout_s", "endpoint", float, 14400.0), "endpoint.timeout_s"),
        retry_window_s=_optional(ep, "retry_window_s", "endpoint", float, 3600.0),
        backoff_base_s=_optional(ep, "backoff_base_s", "endpoint", float, 5.0),
        backoff_max_s=_optional(ep, "backoff_max_s", "endpoint", float, 120.0),
        extra_body=dict(extra_body),
    )
    for name in ("retry_window_s", "backoff_base_s", "backoff_max_s"):
        if getattr(defaults, name) < 0:
            raise ConfigError(f"endpoint.{name}: must not be negative")

    checkpoints = _model_list(
        raw.get("checkpoints"),
        "checkpoints",
        defaults,
        Checkpoint,
        ("generation_samples_per_prompt", "refinement_samples_per_prompt"),
    )
    verifiers = _model_list(raw.get("verifiers"), "verifiers", defaults, Verifier, ("judgments_per_proof",))
    if len({v.judgments_per_proof for v in verifiers}) != 1:
        raise ConfigError("verifiers: every verifier must have the same judgments_per_proof (equally weighted panel)")
    judges = _model_list(raw.get("judges"), "judges", defaults, Judge, ("judgments_per_finalist",), required=False)

    search_raw = raw.get("search")
    if not isinstance(search_raw, dict):
        raise ConfigError("search: missing section")
    _reject_unknown(search_raw, ("generation_prompts", "max_rounds", "refinement_prompts_per_round"), "search")
    prompts = _require(search_raw, "generation_prompts", "search", list)
    if not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
        raise ConfigError("search.generation_prompts: expected a non-empty list of prompt ids")
    if len(set(prompts)) != len(prompts):
        raise ConfigError("search.generation_prompts: duplicate prompt id")
    if prompts_dir is not None:
        for prompt_id in prompts:
            if not (Path(prompts_dir) / f"{prompt_id}.yaml").is_file():
                raise ConfigError(f"search.generation_prompts: prompt file not found: {prompts_dir}/{prompt_id}.yaml")
    search = SearchConfig(
        generation_prompts=tuple(prompts),
        max_rounds=_positive(_require(search_raw, "max_rounds", "search", int), "search.max_rounds"),
        refinement_prompts_per_round=_positive(
            _require(search_raw, "refinement_prompts_per_round", "search", int), "search.refinement_prompts_per_round"
        ),
    )

    sampling_raw = raw.get("sampling") or {}
    _reject_unknown(sampling_raw, ("temperature", "top_p"), "sampling")
    sampling = Sampling(
        temperature=_optional(sampling_raw, "temperature", "sampling", float, 1.0),
        top_p=_optional(sampling_raw, "top_p", "sampling", float, 0.95),
    )

    conc_raw = raw.get("concurrency") or {}
    _reject_unknown(conc_raw, ("problems", "generation", "verification", "refinement", "judging"), "concurrency")
    concurrency = Concurrency(
        **{
            name: _positive(_optional(conc_raw, name, "concurrency", int, default), f"concurrency.{name}")
            for name, default in (
                ("problems", 30),
                ("generation", 320),
                ("verification", 1280),
                ("refinement", 320),
                ("judging", 256),
            )
        }
    )

    cb_raw = raw.get("context_budget")
    context_budget = None
    if cb_raw is not None:
        _reject_unknown(cb_raw, ("tokenizer", "safety_margin_tokens"), "context_budget")
        tokenizer = _require(cb_raw, "tokenizer", "context_budget", str)
        if _placeholder(tokenizer):
            raise ConfigError(f"context_budget.tokenizer: fill in a tokenizer id or path (got {tokenizer!r})")
        context_budget = ContextBudget(
            tokenizer=tokenizer,
            safety_margin_tokens=_optional(cb_raw, "safety_margin_tokens", "context_budget", int, 4096),
        )

    input_raw = raw.get("input")
    if not isinstance(input_raw, dict):
        raise ConfigError("input: missing section")
    _reject_unknown(input_raw, ("path",), "input")
    input_path = _require(input_raw, "path", "input", str)
    if _placeholder(input_path):
        raise ConfigError(f"input.path: fill in the problems file (got {input_path!r})")
    input_cfg = InputConfig(path=input_path)

    return Config(
        endpoint=defaults,
        checkpoints=checkpoints,
        verifiers=verifiers,
        judges=judges,
        search=search,
        sampling=sampling,
        concurrency=concurrency,
        context_budget=context_budget,
        input=input_cfg,
    )


def derived_counts(cfg):
    return {
        "generations_per_problem": len(cfg.search.generation_prompts)
        * sum(c.generation_samples_per_prompt for c in cfg.checkpoints),
        "judgments_per_proof": sum(v.judgments_per_proof for v in cfg.verifiers),
        "refinements_per_round": cfg.search.refinement_prompts_per_round
        * sum(c.refinement_samples_per_prompt for c in cfg.checkpoints),
        "judgments_per_finalist": sum(j.judgments_per_finalist for j in cfg.judges) if cfg.judges else 0,
    }


def _spec_keys(spec, extra):
    return {"id": spec.id, "model": spec.model, "max_len": spec.max_len, "max_tokens": spec.max_tokens, **extra}


def experiment_keys(cfg):
    """Everything that changes what is computed. Base URLs, keys, timeouts, retries, and lanes are excluded."""
    return {
        "checkpoints": [
            _spec_keys(
                c,
                {
                    "generation_samples_per_prompt": c.generation_samples_per_prompt,
                    "refinement_samples_per_prompt": c.refinement_samples_per_prompt,
                },
            )
            for c in cfg.checkpoints
        ],
        "verifiers": [_spec_keys(v, {"judgments_per_proof": v.judgments_per_proof}) for v in cfg.verifiers],
        "judges": None
        if cfg.judges is None
        else [_spec_keys(j, {"judgments_per_finalist": j.judgments_per_finalist}) for j in cfg.judges],
        "search": {
            "generation_prompts": list(cfg.search.generation_prompts),
            "max_rounds": cfg.search.max_rounds,
            "refinement_prompts_per_round": cfg.search.refinement_prompts_per_round,
        },
        "sampling": {"temperature": cfg.sampling.temperature, "top_p": cfg.sampling.top_p},
        "extra_body": cfg.endpoint.extra_body,
        "context_budget": None
        if cfg.context_budget is None
        else {
            "tokenizer": cfg.context_budget.tokenizer,
            "safety_margin_tokens": cfg.context_budget.safety_margin_tokens,
        },
    }


def resolve_api_key(spec):
    """The API key for one model entry: from its environment variable, or the keyless placeholder."""
    if spec.api_key_env is None:
        return KEYLESS_API_KEY
    value = os.environ.get(spec.api_key_env)
    if not value:
        raise ConfigError(
            f"Environment variable {spec.api_key_env} (API key for model entry {spec.id!r}) is not set. "
            "Export it, or set api_key_env: null for a server that needs no key."
        )
    return value


def all_model_specs(cfg):
    specs = list(cfg.checkpoints) + list(cfg.verifiers)
    if cfg.judges:
        specs.extend(cfg.judges)
    return specs

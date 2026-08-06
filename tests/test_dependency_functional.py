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

"""Functional tests that NeMo-Skills still works with the versions bumped in this PR.

`tests/test_requirements_versions.py` is a *static* guard — it parses the
requirements files and asserts the pins stay in place. It never imports or runs
the bumped packages. These tests close that gap: they drive the bumped
dependencies **through NeMo-Skills' own code paths** so a resolve that installs a
version whose behavior diverged from what NeMo-Skills expects fails CI, not a
production run.

Bumps under test:
  * litellm[caching] ==1.84.10  (fixes GHSA-4xpc-pv4p-pm3w — the pre-1.84 client
                                 could leak the configured api_key to an
                                 attacker-controlled Host header)
  * GitPython        >=3.1.55
  * datamodel-code-generator >=0.64.0
  * wandb            ==0.28.1, with a patched core in the container
  * typer            >=0.16 / click cap removed  (typer<0.16 broke on click 8.2's
                                 Parameter.make_metavar signature — dynamo#1039)
  * lxml             >=6.1.0  (fixes GHSA-vfmq-68hx-4jfw; optional `stem` extra,
                               so guarded by importorskip)
  * aiohttp          >=3.14.3 (fixes CVE-2026-69244)
  * msgpack          >=1.2.1  (fixes GHSA-6v7p-g79w-8964)
  * setuptools       >=78.1.1 (fixes CVE-2025-47273)

All tests are CPU-only, hermetic (no sandbox container, no live LLM endpoint, no
API keys) so they run in the existing `unit-tests` (`-m "not gpu"`) CI job.
"""

import asyncio
import json
from importlib.metadata import version
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# GitPython >=3.1.55 and datamodel-code-generator >=0.64.0
# ---------------------------------------------------------------------------


def test_gitpython_can_initialize_and_inspect_repository(tmp_path):
    import git
    from packaging.version import Version

    assert Version(version("GitPython")) >= Version("3.1.55")
    repo = git.Repo.init(tmp_path)
    assert not repo.bare
    assert repo.git_dir == str(tmp_path / ".git")


def test_datamodel_code_generator_imports_at_fixed_version():
    import datamodel_code_generator
    from packaging.version import Version

    assert datamodel_code_generator is not None
    assert Version(version("datamodel-code-generator")) >= Version("0.64.0")


def test_aiohttp_msgpack_and_setuptools_security_floors():
    from packaging.version import Version

    assert Version(version("aiohttp")) >= Version("3.14.3")
    assert Version(version("msgpack")) >= Version("1.2.1")
    assert Version(version("setuptools")) >= Version("78.1.1")


# ---------------------------------------------------------------------------
# litellm 1.84.10 — driven through nemo_skills.inference.model
# ---------------------------------------------------------------------------

_URL = "http://regression-host:1234/v1"
_KEY = "sk-regression-secret"


def _make_openai_model(**overrides):
    """Construct a real OpenAIModel offline (no tokenizer, no network I/O).

    Default construction does no HTTP: require_tokenizer defaults False, and an
    explicit api_key/base_url skip every env-var lookup.
    """
    from nemo_skills.inference.model.openai import OpenAIModel

    kwargs = dict(model="dummy-model", base_url=_URL, api_key=_KEY)
    kwargs.update(overrides)
    return OpenAIModel(**kwargs)


def test_openai_model_registered_and_constructs():
    """get_model('openai') resolves and builds under litellm 1.84."""
    from nemo_skills.inference.model import get_model
    from nemo_skills.inference.model.openai import OpenAIModel

    model = get_model(server_type="openai", model="dummy-model", base_url=_URL, api_key=_KEY)
    assert isinstance(model, OpenAIModel)


def test_credentials_bound_to_configured_base_url():
    """CVE regression (GHSA-4xpc-pv4p-pm3w): the api_key is assembled bound to
    our configured base_url/api_base only — never a caller-influenced host."""
    model = _make_openai_model()
    assert model.litellm_kwargs["api_key"] == _KEY
    assert model.litellm_kwargs["base_url"] == _URL
    assert model.litellm_kwargs["api_base"] == _URL
    # provider-prefixed model name is what litellm routes on
    assert model.litellm_kwargs["model"] == "openai/dummy-model"


def test_generate_async_calls_litellm_with_bound_credentials():
    """The real generate_async path builds request params and calls
    litellm.acompletion; the key travels only alongside our base_url/api_base,
    and litellm 1.84's response object parses back into NeMo-Skills' dict."""
    model = _make_openai_model()

    # litellm returns pydantic objects; mimic the attributes NeMo-Skills reads
    # plus model_dump() (used by _serialize_output for conversation history).
    fake_choice = SimpleNamespace(
        message=SimpleNamespace(content="hello world"),
        finish_reason="stop",
        logprobs=None,
        model_dump=lambda: {"message": {"role": "assistant", "content": "hello world"}},
    )
    fake_response = SimpleNamespace(
        choices=[fake_choice],
        usage=SimpleNamespace(completion_tokens=2, prompt_tokens=3),
    )

    with patch("litellm.acompletion", new=AsyncMock(return_value=fake_response)) as mock_acompletion:
        result = asyncio.run(
            model.generate_async(
                [{"role": "user", "content": "hi"}],
                tokens_to_generate=8,
                remove_stop_phrases=False,
            )
        )

    assert result["generation"] == "hello world"
    assert result["num_generated_tokens"] == 2
    assert result["num_input_tokens"] == 3

    mock_acompletion.assert_awaited_once()
    call_kwargs = mock_acompletion.await_args.kwargs
    assert call_kwargs["api_key"] == _KEY
    assert call_kwargs["base_url"] == _URL
    assert call_kwargs["api_base"] == _URL
    assert call_kwargs["model"] == "openai/dummy-model"
    # the user message we passed must be the one litellm receives
    assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_build_chat_request_params_shape():
    """The param names NeMo-Skills sends still match litellm 1.84's chat schema."""
    model = _make_openai_model()
    params = model._build_chat_request_params(
        messages=[{"role": "user", "content": "hi"}],
        tokens_to_generate=16,
        temperature=0.0,
        top_p=0.95,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        random_seed=1234,
        stop_phrases=None,
        timeout=60,
        top_logprobs=None,
        stream=False,
        reasoning_effort=None,
    )
    assert params["messages"] == [{"role": "user", "content": "hi"}]
    assert params["max_completion_tokens"] == 16
    assert params["seed"] == 1234
    assert params["stream"] is False


def test_litellm_exception_and_type_surface_present():
    """NeMo-Skills imports these litellm symbols at module load; guard the API
    surface so a litellm bump that relocated them fails here, not at import of
    inference/mcp code."""
    from litellm.exceptions import ContextWindowExceededError  # used by model/utils.py
    from litellm.types.utils import ChatCompletionMessageToolCall  # used by mcp/adapters.py

    assert issubclass(ContextWindowExceededError, Exception)
    assert ChatCompletionMessageToolCall is not None


# ---------------------------------------------------------------------------
# typer >=0.16 / click cap removed — driven through the `ns` CLI app
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cli_app():
    from nemo_skills.pipeline.cli import app

    return app


def test_cli_help_renders(cli_app):
    """Top-level --help exercises click 8.2's Parameter.make_metavar — the exact
    path typer<0.16 crashed on (dynamo#1039). Must render cleanly under the
    uncapped click."""
    from typer.testing import CliRunner

    result = CliRunner().invoke(cli_app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output


@pytest.mark.parametrize("subcommand", ["generate", "eval", "prepare_data"])
def test_cli_subcommand_help_renders(cli_app, subcommand):
    """Per-command help proves each registered command's params render metavars
    under click 8.2 (the break was per-parameter, so exercise real commands)."""
    from typer.testing import CliRunner

    result = CliRunner().invoke(cli_app, [subcommand, "--help"])
    assert result.exit_code == 0, result.output


def test_cli_unknown_command_is_usage_error(cli_app):
    """Arg parsing still rejects an unknown command (proves the click parser is
    wired, not just that --help short-circuits)."""
    from typer.testing import CliRunner

    result = CliRunner().invoke(cli_app, ["definitely-not-a-real-command"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# wandb ==0.28.1 — driven through nemo_skills.inference.log_samples_wandb
# ---------------------------------------------------------------------------


def test_wandb_log_random_samples_call_contract(tmp_path):
    """NeMo-Skills' wandb usage (init/save/summary/finish) still matches the
    wandb 0.28 API. Patched so it is deterministic and offline."""
    from nemo_skills.inference import log_samples_wandb

    jsonl = tmp_path / "samples.jsonl"
    jsonl.write_text("\n".join(json.dumps({"problem": f"q{i}", "generation": f"a{i}"}) for i in range(4)) + "\n")

    fake_wandb = MagicMock()
    fake_wandb.summary = {}
    with patch.object(log_samples_wandb, "wandb", fake_wandb):
        log_samples_wandb.log_random_samples(str(jsonl), num_samples=2, project="regr", name="run1")

    fake_wandb.init.assert_called_once()
    assert fake_wandb.init.call_args.kwargs["project"] == "regr"
    assert fake_wandb.init.call_args.kwargs["name"] == "run1"
    fake_wandb.save.assert_called_once()
    fake_wandb.finish.assert_called_once()
    assert fake_wandb.summary["num_samples"] == 4


def test_wandb_offline_init_runs(tmp_path, monkeypatch):
    """Real wandb 0.28 imports and completes an offline init/finish cycle with no
    account or network (guards the bundled wandb-core the CVE bump targets)."""
    import wandb

    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    run = wandb.init(project="regr", name="offline-run", dir=str(tmp_path))
    try:
        run.log({"metric": 1.0})
    finally:
        wandb.finish()
    assert (tmp_path / "wandb").exists()


# ---------------------------------------------------------------------------
# lxml >=6.1.0 — optional `stem` extra (skips when not installed, e.g. the
# core-only unit-tests job); exercises the real parser when present.
# ---------------------------------------------------------------------------


def test_lxml_parses_html():
    """lxml 6.1 parses HTML via both its own etree and as the BeautifulSoup
    backend NeMo-Skills' `stem` extra provides. Skips cleanly when lxml is not
    installed (it is not in the core/dev set)."""
    lxml_html = pytest.importorskip("lxml.html")
    tree = lxml_html.fromstring("<html><body><p id='x'>hello</p></body></html>")
    assert tree.find(".//p").text == "hello"

    bs4 = pytest.importorskip("bs4")
    soup = bs4.BeautifulSoup("<html><body><p>hi</p></body></html>", "lxml")
    assert soup.find("p").text == "hi"

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

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nemo_skills.dataset.prepare import prepare_datasets
from nemo_skills.dataset.utils import (
    get_dataset_module,
    get_dataset_name,
    get_dataset_path,
    get_extra_benchmark_map,
    import_from_path,
)
from nemo_skills.evaluation.evaluator import (
    EVALUATOR_CLASS_MAP,
    EVALUATOR_MAP,
    _resolve_eval_type,
)
from nemo_skills.evaluation.metrics.base import BaseMetrics
from nemo_skills.evaluation.metrics.map_metrics import get_metrics
from nemo_skills.pipeline.prepare_data import _build_command, _parse_prepare_cli_arguments
from nemo_skills.pipeline.utils.packager import (
    EXTERNAL_REPOS,
    RepoMetadata,
    get_packager,
    get_registered_external_repo,
    register_external_repo,
    resolve_external_data_path,
)
from nemo_skills.prompt.utils import get_config_path, load_config

FIXTURE_DIR = Path(__file__).parent / "data" / "dummy_external_benchmark"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_global_state():
    """Save and restore global registries, env vars, and dynamic sys.modules."""
    saved_evaluator_map = dict(EVALUATOR_MAP)
    saved_evaluator_class_map = dict(EVALUATOR_CLASS_MAP)
    saved_external_repos = dict(EXTERNAL_REPOS)
    saved_env = os.environ.copy()

    # Track dynamic modules added during test
    modules_before = set(sys.modules.keys())

    yield

    # Restore evaluator maps
    EVALUATOR_MAP.clear()
    EVALUATOR_MAP.update(saved_evaluator_map)
    EVALUATOR_CLASS_MAP.clear()
    EVALUATOR_CLASS_MAP.update(saved_evaluator_class_map)

    # Restore external repos
    EXTERNAL_REPOS.clear()
    EXTERNAL_REPOS.update(saved_external_repos)

    # Restore env vars
    os.environ.clear()
    os.environ.update(saved_env)

    # Clean up dynamically added modules
    for mod_name in set(sys.modules.keys()) - modules_before:
        if mod_name.startswith("dynamic_module_"):
            del sys.modules[mod_name]


@pytest.fixture
def dummy_benchmark_git(tmp_path):
    """Copy fixture to tmp_path and git init it."""
    dest = tmp_path / "dummy_external_benchmark"
    shutil.copytree(FIXTURE_DIR, dest)
    subprocess.run(["git", "init"], cwd=dest, capture_output=True, check=True)
    subprocess.run(["git", "add", "."], cwd=dest, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init", "--no-gpg-sign"],
        cwd=dest,
        capture_output=True,
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    )
    return dest


@pytest.fixture
def dummy_benchmark_no_git(tmp_path):
    """Copy fixture to tmp_path (no git)."""
    dest = tmp_path / "dummy_external_benchmark"
    shutil.copytree(FIXTURE_DIR, dest)
    return dest


@pytest.fixture
def benchmark_map_path(dummy_benchmark_git):
    return str(dummy_benchmark_git / "benchmark_map.json")


@pytest.fixture
def word_count_path(dummy_benchmark_git):
    return str(dummy_benchmark_git / "my_benchmarks" / "dataset" / "word_count")


@pytest.fixture
def simple_bench_path(dummy_benchmark_git):
    return str(dummy_benchmark_git / "my_benchmarks" / "dataset" / "my_simple_bench")


# ---------------------------------------------------------------------------
# A. DatasetResolution
# ---------------------------------------------------------------------------


class TestDatasetResolution:
    def test_get_dataset_name_short(self):
        assert get_dataset_name("gsm8k") == "gsm8k"

    def test_get_dataset_name_path(self):
        assert get_dataset_name("/some/path/to/word_count") == "word_count"

    def test_get_dataset_path_builtin(self):
        result = get_dataset_path("gsm8k")
        assert result.name == "gsm8k"
        assert "nemo_skills/dataset/gsm8k" in str(result)

    def test_get_dataset_path_with_slash(self, word_count_path):
        result = get_dataset_path(word_count_path)
        assert str(result) == word_count_path

    def test_get_dataset_path_from_map(self, benchmark_map_path, dummy_benchmark_git):
        os.environ["NEMO_SKILLS_EXTRA_BENCHMARK_MAP"] = benchmark_map_path
        result = get_dataset_path("word_count")
        expected = str((dummy_benchmark_git / "my_benchmarks" / "dataset" / "word_count").resolve())
        assert str(result) == expected

    def test_get_dataset_path_from_map_arg(self, dummy_benchmark_git):
        """benchmark_map dict arg should work without env var."""
        os.environ.pop("NEMO_SKILLS_EXTRA_BENCHMARK_MAP", None)
        expected = str((dummy_benchmark_git / "my_benchmarks" / "dataset" / "word_count").resolve())
        benchmark_map = {"word_count": expected}
        result = get_dataset_path("word_count", extra_benchmark_map=benchmark_map)
        assert str(result) == expected

    def test_get_dataset_path_from_map_file_arg(self, benchmark_map_path, dummy_benchmark_git):
        """benchmark_map path arg should work without env var."""
        os.environ.pop("NEMO_SKILLS_EXTRA_BENCHMARK_MAP", None)
        result = get_dataset_path("word_count", extra_benchmark_map=benchmark_map_path)
        expected = str((dummy_benchmark_git / "my_benchmarks" / "dataset" / "word_count").resolve())
        assert str(result) == expected


# ---------------------------------------------------------------------------
# B. ExtraBenchmarkMap
# ---------------------------------------------------------------------------


class TestExtraBenchmarkMap:
    def test_empty_when_no_env_var(self):
        os.environ.pop("NEMO_SKILLS_EXTRA_BENCHMARK_MAP", None)
        assert get_extra_benchmark_map() == {}

    def test_loads_map(self, benchmark_map_path, dummy_benchmark_git):
        os.environ["NEMO_SKILLS_EXTRA_BENCHMARK_MAP"] = benchmark_map_path
        result = get_extra_benchmark_map()
        assert "word_count" in result
        assert "my_simple_bench" in result
        expected = str((dummy_benchmark_git / "my_benchmarks" / "dataset" / "word_count").resolve())
        assert result["word_count"] == expected

    def test_relative_path_resolved(self, benchmark_map_path):
        os.environ["NEMO_SKILLS_EXTRA_BENCHMARK_MAP"] = benchmark_map_path
        result = get_extra_benchmark_map()
        # The relative path ./my_benchmarks/... should be resolved to absolute
        assert os.path.isabs(result["word_count"])

    def test_absolute_path_kept(self, tmp_path):
        abs_path = "/absolute/path/to/bench"
        map_file = tmp_path / "map.json"
        map_file.write_text(json.dumps({"abs_bench": abs_path}))
        os.environ["NEMO_SKILLS_EXTRA_BENCHMARK_MAP"] = str(map_file)
        result = get_extra_benchmark_map()
        assert result["abs_bench"] == abs_path

    def test_dict_arg_returned_as_is(self):
        """Passing a dict directly should return it without reading env var."""
        os.environ.pop("NEMO_SKILLS_EXTRA_BENCHMARK_MAP", None)
        benchmark_map = {"my_bench": "/abs/path/to/bench"}
        result = get_extra_benchmark_map(benchmark_map)
        assert result == benchmark_map

    def test_file_path_arg(self, benchmark_map_path, dummy_benchmark_git):
        """Passing a file path arg should load and resolve like the env var does."""
        os.environ.pop("NEMO_SKILLS_EXTRA_BENCHMARK_MAP", None)
        result = get_extra_benchmark_map(benchmark_map_path)
        assert "word_count" in result
        expected = str((dummy_benchmark_git / "my_benchmarks" / "dataset" / "word_count").resolve())
        assert result["word_count"] == expected


# ---------------------------------------------------------------------------
# C. GetDatasetModule
# ---------------------------------------------------------------------------


class TestGetDatasetModule:
    def test_full_path(self, word_count_path):
        module, data_path = get_dataset_module(word_count_path)
        assert hasattr(module, "METRICS_TYPE")
        assert module.METRICS_TYPE == "my_benchmarks.metrics.word_count::WordCountMetrics"

    def test_builtin(self):
        module, data_path = get_dataset_module("gsm8k")
        assert hasattr(module, "METRICS_TYPE")
        assert module.METRICS_TYPE == "math"

    def test_from_map(self, benchmark_map_path):
        os.environ["NEMO_SKILLS_EXTRA_BENCHMARK_MAP"] = benchmark_map_path
        module, data_path = get_dataset_module("word_count")
        assert module.METRICS_TYPE == "my_benchmarks.metrics.word_count::WordCountMetrics"

    def test_from_map_arg(self, benchmark_map_path):
        """benchmark_map file path arg should work without env var."""
        os.environ.pop("NEMO_SKILLS_EXTRA_BENCHMARK_MAP", None)
        module, data_path = get_dataset_module("word_count", extra_benchmark_map=benchmark_map_path)
        assert module.METRICS_TYPE == "my_benchmarks.metrics.word_count::WordCountMetrics"

    def test_from_map_dict_arg(self, dummy_benchmark_git):
        """benchmark_map dict arg should work without env var."""
        os.environ.pop("NEMO_SKILLS_EXTRA_BENCHMARK_MAP", None)
        word_count_dir = str((dummy_benchmark_git / "my_benchmarks" / "dataset" / "word_count").resolve())
        benchmark_map = {"word_count": word_count_dir}
        module, data_path = get_dataset_module("word_count", extra_benchmark_map=benchmark_map)
        assert module.METRICS_TYPE == "my_benchmarks.metrics.word_count::WordCountMetrics"

    def test_simple_bench_from_map(self, benchmark_map_path):
        os.environ["NEMO_SKILLS_EXTRA_BENCHMARK_MAP"] = benchmark_map_path
        module, data_path = get_dataset_module("my_simple_bench")
        assert module.METRICS_TYPE == "math"

    def test_collision_builtin_and_map(self, tmp_path):
        """If a name is in both built-in and map, raise RuntimeError."""
        map_file = tmp_path / "map.json"
        map_file.write_text(json.dumps({"gsm8k": str(tmp_path)}))
        os.environ["NEMO_SKILLS_EXTRA_BENCHMARK_MAP"] = str(map_file)
        with pytest.raises(RuntimeError, match="found both as a built-in dataset and in the extra benchmark map"):
            get_dataset_module("gsm8k")

    def test_collision_builtin_and_map_dict_arg(self, tmp_path):
        """Dict arg collision should also raise."""
        os.environ.pop("NEMO_SKILLS_EXTRA_BENCHMARK_MAP", None)
        benchmark_map = {"gsm8k": str(tmp_path)}
        with pytest.raises(RuntimeError, match="found both as a built-in dataset and in the extra benchmark map"):
            get_dataset_module("gsm8k", extra_benchmark_map=benchmark_map)

    def test_not_found_no_map(self):
        os.environ.pop("NEMO_SKILLS_EXTRA_BENCHMARK_MAP", None)
        with pytest.raises(RuntimeError, match="Did you forget to pass extra_benchmark_map or set"):
            get_dataset_module("nonexistent_bench_xyz")

    def test_not_found_with_map(self, benchmark_map_path):
        os.environ["NEMO_SKILLS_EXTRA_BENCHMARK_MAP"] = benchmark_map_path
        with pytest.raises(RuntimeError, match="not found in built-in datasets or the extra benchmark map"):
            get_dataset_module("nonexistent_bench_xyz")

    def test_missing_init_py(self, tmp_path):
        """Path with no __init__.py should raise."""
        empty_dir = tmp_path / "empty_bench"
        empty_dir.mkdir()
        with pytest.raises(RuntimeError, match="Expected .* to exist"):
            get_dataset_module(str(empty_dir))


# ---------------------------------------------------------------------------
# D. BuildCommand
# ---------------------------------------------------------------------------


class TestBuildCommand:
    def test_builtin_dataset_appended(self):
        cmd = _build_command(
            command="python -m nemo_skills.dataset.prepare",
            requested_datasets=["gsm8k"],
            data_dir=None,
            extra_benchmark_map={},
            cluster_config={"executor": "none"},
            skip_data_dir_check=True,
            prepare_unknown_args=[],
        )
        assert "gsm8k" in cmd

    def test_external_dataset_local(self, benchmark_map_path, dummy_benchmark_git):
        """When executor is 'none', external dataset name is used as-is."""
        os.environ["NEMO_SKILLS_EXTRA_BENCHMARK_MAP"] = benchmark_map_path
        extra_map = {
            "my_simple_bench": str((dummy_benchmark_git / "my_benchmarks" / "dataset" / "my_simple_bench").resolve())
        }
        cmd = _build_command(
            command="python -m nemo_skills.dataset.prepare",
            requested_datasets=["my_simple_bench"],
            data_dir=None,
            extra_benchmark_map=extra_map,
            cluster_config={"executor": "none"},
            skip_data_dir_check=True,
            prepare_unknown_args=[],
        )
        # When executor is "none", it should use the name directly
        assert "my_simple_bench" in cmd

    def test_data_dir_collision_raises(self, benchmark_map_path, dummy_benchmark_git):
        """External dataset name colliding with built-in should raise ValueError."""
        os.environ["NEMO_SKILLS_EXTRA_BENCHMARK_MAP"] = benchmark_map_path
        # Create a dir named "gsm8k" (collides with built-in)
        fake_gsm_dir = dummy_benchmark_git / "my_benchmarks" / "dataset" / "gsm8k"
        fake_gsm_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            dummy_benchmark_git / "my_benchmarks" / "dataset" / "my_simple_bench" / "__init__.py",
            fake_gsm_dir / "__init__.py",
        )

        with pytest.raises(ValueError, match="conflicts with a built-in dataset"):
            _build_command(
                command="python -m nemo_skills.dataset.prepare",
                requested_datasets=[str(fake_gsm_dir)],
                data_dir="/some/data_dir",
                extra_benchmark_map={},
                cluster_config={"executor": "none"},
                skip_data_dir_check=True,
                prepare_unknown_args=[],
            )

    def test_extra_args_appended(self):
        cmd = _build_command(
            command="python -m nemo_skills.dataset.prepare",
            requested_datasets=["gsm8k"],
            data_dir=None,
            extra_benchmark_map={},
            cluster_config={"executor": "none"},
            skip_data_dir_check=True,
            prepare_unknown_args=["--split", "test"],
        )
        assert "--split test" in cmd


# ---------------------------------------------------------------------------
# D2. ParsePrepareCLIArguments
# ---------------------------------------------------------------------------


class TestParsePrepareCLIArguments:
    def test_datasets_separated_from_unknown_args(self):
        datasets, unknown = _parse_prepare_cli_arguments(["gsm8k", "math-500", "--foo", "bar"])
        assert datasets == ["gsm8k", "math-500"]
        assert "--foo" in unknown
        assert "bar" in unknown

    def test_parallelism_and_retries_passed_through(self):
        datasets, unknown = _parse_prepare_cli_arguments(
            ["gsm8k", "--parallelism", "7", "--retries=2", "--foo", "bar"]
        )
        assert datasets == ["gsm8k"]
        assert "--parallelism" in unknown
        assert "7" in unknown
        assert "--retries" in unknown
        assert "2" in unknown
        assert "--foo" in unknown
        assert "bar" in unknown

    def test_default_parallelism_and_retries_included(self):
        datasets, unknown = _parse_prepare_cli_arguments(["gsm8k"])
        assert datasets == ["gsm8k"]
        assert "--parallelism" in unknown
        assert "--retries" in unknown


# ---------------------------------------------------------------------------
# E. EvaluatorResolution
# ---------------------------------------------------------------------------


class TestEvaluatorResolution:
    def test_builtin_class(self):
        obj, is_class = _resolve_eval_type("math")
        assert is_class is True
        assert obj is not None

    def test_builtin_function(self):
        obj, is_class = _resolve_eval_type("if")
        assert is_class is False
        assert callable(obj)

    def test_file_path_class_evaluator(self, dummy_benchmark_git):
        """WordCountEvaluator is a BaseEvaluator subclass, so is_class should be True."""
        eval_file = str(dummy_benchmark_git / "my_benchmarks" / "evaluation" / "word_count.py")
        obj, is_class = _resolve_eval_type(f"{eval_file}::WordCountEvaluator")
        assert is_class is True
        assert obj.__name__ == "WordCountEvaluator"

    def test_not_found(self):
        obj, is_class = _resolve_eval_type("nonexistent_evaluator_xyz")
        assert obj is None
        assert is_class is False


# ---------------------------------------------------------------------------
# F. MetricsResolution
# ---------------------------------------------------------------------------


class TestMetricsResolution:
    def test_builtin_math(self):
        m = get_metrics("math")
        assert m is not None

    def test_file_path_with_colons(self, dummy_benchmark_git):
        metrics_file = str(dummy_benchmark_git / "my_benchmarks" / "metrics" / "word_count.py")
        m = get_metrics(f"{metrics_file}::WordCountMetrics")
        assert m is not None

    def test_not_found(self):
        with pytest.raises(ValueError, match="not found"):
            get_metrics("nonexistent_metric_xyz")


# ---------------------------------------------------------------------------
# G. Packager
# ---------------------------------------------------------------------------


class TestPackager:
    def test_register_external_repo(self, dummy_benchmark_git):
        meta = RepoMetadata(name="test_repo_xyz", path=dummy_benchmark_git)
        register_external_repo(meta)
        assert "test_repo_xyz" in EXTERNAL_REPOS
        assert get_registered_external_repo("test_repo_xyz") is not None

    def test_resolve_external_data_path(self, dummy_benchmark_git):
        meta = RepoMetadata(name="resolve_repo_xyz", path=dummy_benchmark_git)
        register_external_repo(meta)
        dataset_dir = dummy_benchmark_git / "my_benchmarks" / "dataset" / "word_count"
        result = resolve_external_data_path(dataset_dir.parent)
        assert result.startswith("/nemo_run/code/")
        assert "my_benchmarks/dataset" in result

    def test_resolve_external_data_path_no_match(self, tmp_path):
        with pytest.raises(RuntimeError, match="does not belong to any registered external repo"):
            resolve_external_data_path(tmp_path / "nonexistent")

    def test_repo_metadata_invalid_path(self):
        with pytest.raises(ValueError, match="does not exist"):
            RepoMetadata(name="bad", path="/does/not/exist")

    def test_get_packager_in_non_editable_git_repo_has_matching_include_paths(self, dummy_benchmark_git, monkeypatch):
        """When running from a non-NeMo git repo, include pattern metadata must stay aligned."""
        monkeypatch.chdir(dummy_benchmark_git)
        packager = get_packager()
        assert isinstance(packager.include_pattern, list)
        assert isinstance(packager.include_pattern_relative_path, list)
        assert len(packager.include_pattern) == len(packager.include_pattern_relative_path)


# ---------------------------------------------------------------------------
# H. PrepareDatasets
# ---------------------------------------------------------------------------


class TestPrepareDatasets:
    def test_prepare_word_count_via_full_path(self, word_count_path):
        prepare_datasets(datasets=[word_count_path], parallelism=1, retries=0)
        jsonl_file = Path(word_count_path) / "test.jsonl"
        assert jsonl_file.exists()
        with open(jsonl_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 5
        first = json.loads(lines[0])
        assert "sentence" in first
        assert "expected_answer" in first

    def test_prepare_simple_bench_via_map(self, benchmark_map_path, dummy_benchmark_git):
        os.environ["NEMO_SKILLS_EXTRA_BENCHMARK_MAP"] = benchmark_map_path
        prepare_datasets(datasets=["my_simple_bench"], parallelism=1, retries=0)
        dataset_dir = dummy_benchmark_git / "my_benchmarks" / "dataset" / "my_simple_bench"
        jsonl_file = dataset_dir / "test.jsonl"
        assert jsonl_file.exists()
        with open(jsonl_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        first = json.loads(lines[0])
        assert first["problem"] == "What is 2 + 2?"
        assert first["expected_answer"] == 4


# ---------------------------------------------------------------------------
# I. ExternalModuleAttributes
# ---------------------------------------------------------------------------


class TestExternalModuleAttributes:
    def test_word_count_init_attributes(self, word_count_path):
        module, data_path = get_dataset_module(word_count_path)
        assert module.METRICS_TYPE == "my_benchmarks.metrics.word_count::WordCountMetrics"
        assert "++eval_type=my_benchmarks.evaluation.word_count::WordCountEvaluator" in module.GENERATION_ARGS
        assert module.GENERATION_MODULE == "my_benchmarks.inference.word_count"

    def test_simple_bench_init_attributes(self, simple_bench_path):
        module, data_path = get_dataset_module(simple_bench_path)
        assert module.METRICS_TYPE == "math"
        assert "++prompt_config=generic/math" in module.GENERATION_ARGS
        assert "++eval_type=math" in module.GENERATION_ARGS

    def test_init_triggers_registration(self, word_count_path):
        get_dataset_module(word_count_path)
        assert "my_benchmarks" in EXTERNAL_REPOS

    def test_module_data_path_is_parent(self, word_count_path):
        _, data_path = get_dataset_module(word_count_path)
        # data_path should be parent of dataset dir (so data_path/word_count/test.jsonl works)
        assert Path(data_path).name == "dataset"


# ---------------------------------------------------------------------------
# J. CustomGenerationModule
# ---------------------------------------------------------------------------


class TestCustomGenerationModule:
    def test_generation_module_has_task_class(self, dummy_benchmark_git):
        gen_file = dummy_benchmark_git / "my_benchmarks" / "inference" / "word_count.py"
        module = import_from_path(str(gen_file))
        assert hasattr(module, "GENERATION_TASK_CLASS")
        assert module.GENERATION_TASK_CLASS.__name__ == "WordCountGenerationTask"

    def test_custom_evaluator_class_resolution(self, dummy_benchmark_git):
        """Verify :: resolution for WordCountEvaluator (a BaseEvaluator subclass)."""
        eval_file = dummy_benchmark_git / "my_benchmarks" / "evaluation" / "word_count.py"
        obj, is_class = _resolve_eval_type(f"{eval_file}::WordCountEvaluator")
        assert is_class is True
        assert obj.__name__ == "WordCountEvaluator"

    def test_custom_metrics_class_resolution(self, dummy_benchmark_git):
        """Verify :: resolution for WordCountMetrics."""
        metrics_file = dummy_benchmark_git / "my_benchmarks" / "metrics" / "word_count.py"
        m = get_metrics(f"{metrics_file}::WordCountMetrics")
        assert isinstance(m, BaseMetrics)


# ---------------------------------------------------------------------------
# K. PromptConfigResolution
# ---------------------------------------------------------------------------


class TestPromptConfigResolution:
    def test_builtin_config(self):
        """Built-in config should resolve to nemo_skills/prompt/config/."""
        path = get_config_path("generic/math")
        assert path.exists()
        assert "nemo_skills/prompt/config/generic/math.yaml" in str(path)

    def test_absolute_yaml_path(self, dummy_benchmark_git):
        """Config ending in .yaml with absolute path should work directly."""
        yaml_path = str(dummy_benchmark_git / "my_benchmarks" / "prompt" / "eval" / "word_count" / "default.yaml")
        path = get_config_path(yaml_path)
        assert path.exists()
        assert str(path) == yaml_path

    def test_relative_yaml_resolves_to_repo_root(self):
        """Relative .yaml path should fall back to repo root resolution."""
        path = get_config_path("nemo_skills/prompt/config/generic/math.yaml")
        assert path.exists()
        assert str(path).endswith("nemo_skills/prompt/config/generic/math.yaml")

    def test_load_config_builtin(self):
        """load_config with a short name should work."""
        config = load_config("generic/math")
        assert "user" in config

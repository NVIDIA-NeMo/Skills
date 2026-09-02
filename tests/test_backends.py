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

import shlex
import subprocess

import pytest

from nemo_skills.pipeline.utils.backends import get_backend_name, get_execution_backend, is_ray_backend_name


def test_kubernetes_ray_selector_with_image_label_translation():
    cluster_config = {
        "executor": "slurm",
        "backend": {
            "name": "ray",
            "control_plane": "kubernetes",
            "precreated_cluster": True,
            "endpoint": "ray://ray-head.ray.svc.cluster.local:10001",
            "kubernetes": {
                "mode": "offline",
                "entrypoint_label_selector": {"type": "worker"},
                "image_label_key": "nemo/image",
            },
        },
    }

    backend = get_execution_backend(cluster_config)
    metadata = backend.stage_metadata(container_image="nvcr.io/nvidia/pytorch:25.02-py3")

    assert metadata["execution_backend"] == "kubernetes-ray"
    assert metadata["ray_control_plane"] == "kubernetes"
    assert metadata["ray_cluster_mode"] == "precreated"
    assert metadata["kubernetes_mode"] == "offline"
    assert metadata["entrypoint_label_selector"] == {
        "type": "worker",
        "nemo/image": "nvcr.io-nvidia-pytorch-25.02-py3",
    }
    assert metadata["ray_entrypoint_label_selector"] == metadata["entrypoint_label_selector"]


def test_kubernetes_ray_alias_uses_same_selector_logic():
    cluster_config = {
        "executor": "slurm",
        "backend": {
            "name": "kubernetes-ray",
            "endpoint": "ray://ray-head.ray.svc.cluster.local:10001",
            "entrypoint_label_selector": {"type": "worker"},
        },
    }

    backend = get_execution_backend(cluster_config)
    metadata = backend.stage_metadata(container_image="nvcr.io/nvidia/pytorch:25.02-py3")

    assert metadata["execution_backend"] == "kubernetes-ray"
    assert metadata["entrypoint_label_selector"] == {"type": "worker"}


def test_image_translation_does_not_override_explicit_selector_value():
    cluster_config = {
        "executor": "slurm",
        "backend": {
            "name": "ray",
            "control_plane": "kubernetes",
            "endpoint": "ray://ray-head.ray.svc.cluster.local:10001",
            "entrypoint_label_selector": {
                "type": "worker",
                "nemo/image": "preset-image-label",
            },
            "image_label_key": "nemo/image",
        },
    }

    backend = get_execution_backend(cluster_config)
    metadata = backend.stage_metadata(container_image="nvcr.io/nvidia/pytorch:25.02-py3")

    assert metadata["entrypoint_label_selector"]["nemo/image"] == "preset-image-label"
    assert metadata["entrypoint_label_selector"]["type"] == "worker"


def test_image_label_selectors_add_explicit_key_value_pairs():
    cluster_config = {
        "executor": "slurm",
        "containers": {
            "nemo-skills": "/containers/nemo-skills.sqsh",
            "nemo-rl": "/containers/nemo-rl.sqsh",
        },
        "backend": {
            "name": "ray",
            "control_plane": "kubernetes",
            "endpoint": "ray://ray-head.ray.svc.cluster.local:10001",
            "kubernetes": {
                "mode": "offline",
                "entrypoint_label_selector": {"type": "worker"},
                "image_label_selectors": {
                    "nemo-skills": {
                        "key": "nemo/workload",
                        "value": "skills",
                    },
                    "nemo-rl": {
                        "nemo/workload": "rl",
                    },
                },
            },
        },
    }

    backend = get_execution_backend(cluster_config)
    metadata = backend.stage_metadata(container_image="/containers/nemo-skills.sqsh")

    assert metadata["entrypoint_label_selector"]["type"] == "worker"
    assert metadata["entrypoint_label_selector"]["nemo/workload"] == "skills"


def test_image_label_selectors_prefer_static_selector_keys():
    cluster_config = {
        "executor": "slurm",
        "containers": {
            "nemo-skills": "/containers/nemo-skills.sqsh",
        },
        "backend": {
            "name": "ray",
            "control_plane": "kubernetes",
            "endpoint": "ray://ray-head.ray.svc.cluster.local:10001",
            "kubernetes": {
                "mode": "offline",
                "entrypoint_label_selector": {"nemo/workload": "static"},
                "image_label_selectors": {
                    "nemo-skills": {
                        "nemo/workload": "dynamic",
                    }
                },
            },
        },
    }

    backend = get_execution_backend(cluster_config)
    metadata = backend.stage_metadata(container_image="/containers/nemo-skills.sqsh")

    assert metadata["entrypoint_label_selector"]["nemo/workload"] == "static"


def test_ray_backend_forwards_required_env_vars_to_runtime_env():
    cluster_config = {
        "executor": "slurm",
        "required_env_vars": ["MY_JUDGE_KEY=secret-123"],
        "backend": {"name": "ray", "dashboard_url": "http://ray-head:8265"},
    }

    backend = get_execution_backend(cluster_config)
    runtime_env = backend._build_runtime_env()

    assert runtime_env is not None
    assert runtime_env["env_vars"]["MY_JUDGE_KEY"] == "secret-123"


def test_dashboard_only_ray_config_is_precreated_not_embedded():
    # backend.name: ray + dashboard_url (no endpoint) targets a pre-provisioned
    # cluster via the Jobs API, so it must resolve as precreated and must not
    # request an embedded Ray cluster.
    cluster_config = {
        "executor": "slurm",
        "backend": {"name": "ray", "dashboard_url": "http://ray-head:8265"},
    }

    backend = get_execution_backend(cluster_config)
    assert backend.precreated_cluster is True

    metadata = backend.stage_metadata(container_image="nvcr.io/nvidia/pytorch:25.02-py3")
    assert "use_with_ray_cluster" not in (metadata or {})


def test_dashboard_only_precreated_preflight_does_not_require_endpoint():
    # A dashboard-only (Jobs API) precreated cluster has no Ray Client endpoint,
    # so preflight must not require one: require_ray_endpoint now defaults to
    # bool(endpoint)=False, so the checks short-circuit without touching ray.init.
    from nemo_skills.pipeline.utils.backends import BackendRunOptions
    from nemo_skills.pipeline.utils.ray_backend import RayBackend

    backend = RayBackend(dashboard_url="http://ray-head:8265", precreated_cluster=True)
    assert backend.endpoint is None

    # No preflight section -> defaults; dry_run=False to exercise the require_reachable default.
    backend._run_preflight({}, BackendRunOptions(dry_run=False))


def test_dashboard_only_explicit_require_endpoint_still_raises():
    # An explicit preflight.require_ray_endpoint=true is still honored: a connectivity
    # check needs a Ray Client endpoint, which a dashboard-only cluster lacks.
    from nemo_skills.pipeline.utils.backends import BackendRunOptions
    from nemo_skills.pipeline.utils.ray_backend import RayBackend

    backend = RayBackend(dashboard_url="http://ray-head:8265", precreated_cluster=True)
    assert backend.endpoint is None

    cluster_config = {"preflight": {"require_ray_endpoint": True}}
    with pytest.raises(RuntimeError, match="endpoint"):
        backend._run_preflight(cluster_config, BackendRunOptions(dry_run=False))


def test_non_precreated_ray_honors_use_with_ray_cluster_flag():
    # Without a precreated cluster (no endpoint/dashboard_url), stage_metadata must
    # honor the caller's use_with_ray_cluster flag rather than force it on.
    cluster_config = {"executor": "slurm", "backend": {"name": "ray"}}
    backend = get_execution_backend(cluster_config)
    assert backend.precreated_cluster is False

    assert backend.stage_metadata(use_with_ray_cluster=True).get("use_with_ray_cluster") is True
    assert "use_with_ray_cluster" not in (backend.stage_metadata(use_with_ray_cluster=False) or {})


def test_ray_executor_none_without_dashboard_raises():
    # backend.name: ray + executor: none targets a pre-provisioned Ray Jobs cluster,
    # which needs a dashboard URL. Without one (and no precreated flag), fail fast with
    # an actionable message instead of the opaque nemo-run SlurmExecutor error.
    cluster_config = {"executor": "none", "backend": {"name": "ray"}}
    with pytest.raises(ValueError, match="dashboard_url"):
        get_execution_backend(cluster_config)


def test_ray_backend_runtime_env_normalizes_and_filters_values():
    from nemo_skills.pipeline.utils.ray_backend import RayBackend

    backend = RayBackend(dashboard_url="http://ray-head:8265", env_vars={"A": "1", "B": None, "C": 2})

    # RAY_OVERRIDE_JOB_RUNTIME_ENV is always injected so a job driver that re-inits Ray
    # (e.g. NeMo-RL GRPO/rollout) can merge its runtime_env instead of erroring on a shared key.
    assert backend._build_runtime_env() == {"env_vars": {"A": "1", "C": "2", "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1"}}


def test_ray_backend_runtime_env_sets_override_even_with_no_env():
    from nemo_skills.pipeline.utils.ray_backend import RayBackend

    backend = RayBackend(dashboard_url="http://ray-head:8265", env_vars={})

    # Even with no forwarded env vars the runtime_env still carries the override flag (never None).
    runtime_env = backend._build_runtime_env()
    assert runtime_env == {"env_vars": {"RAY_OVERRIDE_JOB_RUNTIME_ENV": "1"}}
    assert {"working_dir", "py_modules", "pip", "conda"}.isdisjoint(runtime_env)


def test_ray_backend_working_dir_enables_install_free_code_delivery():
    """The opt-in working dir is the only runtime-env code-delivery field.

    Ray packages this already-present directory/archive and changes the job cwd;
    no dependency installer configuration is synthesized.
    """
    from nemo_skills.pipeline.utils.ray_backend import RayBackend

    backend = RayBackend(
        dashboard_url="http://ray-head:8265",
        env_vars={"AIRGAP": "1"},
        working_dir="  /opt/nvflow  ",
    )

    assert backend._build_runtime_env() == {
        "env_vars": {"AIRGAP": "1", "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1"},
        "working_dir": "/opt/nvflow",
    }


def test_ray_job_submission_receives_configured_working_dir():
    from types import SimpleNamespace

    from nemo_skills.pipeline.utils.ray_backend import RayBackend

    submitted = []

    class RecordingClient:
        def submit_job(self, **kwargs):
            submitted.append(kwargs)
            return "job-1"

        def get_job_status(self, job_id):
            assert job_id == "job-1"
            return "SUCCEEDED"

        def get_job_logs(self, job_id):
            assert job_id == "job-1"
            return ""

    backend = RayBackend(
        dashboard_url="http://ray-head:8265",
        working_dir="/opt/nvflow",
    )
    exp = SimpleNamespace(_title="airgap-code-delivery")

    backend._submit_jobs_concurrently(
        RecordingClient(),
        exp,
        [{"task_name": "prepare-data", "command": "python -m nvflow.prepare_data"}],
    )

    assert len(submitted) == 1
    assert submitted[0]["runtime_env"] == {
        "env_vars": {"RAY_OVERRIDE_JOB_RUNTIME_ENV": "1"},
        "working_dir": "/opt/nvflow",
    }
    assert "pip" not in submitted[0]["runtime_env"]
    assert "conda" not in submitted[0]["runtime_env"]
    prepared_command = shlex.split(submitted[0]["entrypoint"])[2]
    assert prepared_command.startswith('export NEMO_RUN_CODE_DIR="$PWD" && export RAY_ADDRESS=auto && ')


def test_ray_backend_without_working_dir_preserves_entrypoint_command():
    from nemo_skills.pipeline.utils.ray_backend import RayBackend

    backend = RayBackend(dashboard_url="http://ray-head:8265")

    assert backend._prepare_job_entrypoint_command("echo ok") == "export RAY_ADDRESS=auto && echo ok"


def test_ray_working_dir_paths_expand_after_cd_in_all_quote_contexts(tmp_path):
    from nemo_skills.pipeline.utils.backends import rewrite_ray_job_code_paths
    from nemo_skills.pipeline.utils.ray_backend import RayBackend

    code_dir = tmp_path / "uploaded code"
    skills_dir = code_dir / "nemo_skills"
    gym_dir = tmp_path / "Gym"
    code_dir.mkdir()
    skills_dir.mkdir()
    (skills_dir / "__init__.py").write_text("# packaged test module\n")
    gym_dir.mkdir()

    command = (
        f"cd {shlex.quote(str(gym_dir))} && printf '%s\\n' "
        "/nemo_run/code/nvflow/unquoted "
        '"/nemo_run/code/nvflow/double quoted" '
        "'/nemo_run/code/nvflow/single quoted' "
        "'/nemo_run/code/nemo_skills/dataset/test.jsonl'"
    )
    rewritten = rewrite_ray_job_code_paths(command)
    prepared = RayBackend(
        dashboard_url="http://ray-head:8265",
        working_dir="/opt/nvflow-ray-code.zip",
    )._prepare_job_entrypoint_command(rewritten)

    result = subprocess.run(
        ["/bin/bash", "-c", prepared],
        cwd=code_dir,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        str(code_dir / "nvflow/unquoted"),
        str(code_dir / "nvflow/double quoted"),
        str(code_dir / "nvflow/single quoted"),
        str(skills_dir / "dataset/test.jsonl"),
    ]
    assert prepared.index('export NEMO_RUN_CODE_DIR="$PWD"') < prepared.index(f"cd {shlex.quote(str(gym_dir))}")


def test_ray_backend_rejects_non_string_working_dir():
    from nemo_skills.pipeline.utils.ray_backend import RayBackend

    with pytest.raises(ValueError, match="working_dir must be a string"):
        RayBackend(dashboard_url="http://ray-head:8265", working_dir=["/opt/nvflow"])


@pytest.mark.parametrize("backend_name", ["ray", "kubernetes-ray"])
def test_ray_backend_resolves_working_dir_from_cluster_config(backend_name):
    cluster_config = {
        "executor": "none",
        "backend": {
            "name": backend_name,
            "dashboard_url": "http://ray-head:8265",
            "working_dir": "/opt/nvflow-source.zip",
        },
    }

    backend = get_execution_backend(cluster_config)

    assert backend.working_dir == "/opt/nvflow-source.zip"
    assert backend._build_runtime_env()["working_dir"] == "/opt/nvflow-source.zip"


def test_default_slurm_backend_ignores_ray_working_dir_and_stays_lazy():
    """Ray-only code delivery must not alter the default Slurm backend."""
    backend = get_execution_backend(
        {
            "executor": "slurm",
            "backend": {"name": "default", "working_dir": "/opt/nvflow"},
        }
    )

    assert backend.name == "default"
    assert backend.stage_metadata() is None
    assert not hasattr(backend, "working_dir")


# ---------------------------------------------------------------------------
# Dependency resolver (in-batch vs cross-experiment deps)
# ---------------------------------------------------------------------------


def test_compute_in_batch_dep_names_includes_handles_and_task_names():
    from nemo_skills.pipeline.utils.ray_backend import RayBackend

    jobs = [
        {"task_name": "train", "task_handle": "train-handle"},
        {"task_name": "judge", "task_handle": "judge-handle"},
        {"command": "echo no-names"},  # missing both keys -> contributes nothing
    ]

    names = RayBackend._compute_in_batch_dep_names(jobs)

    assert names == {"train", "judge", "train-handle", "judge-handle"}


def test_deps_satisfied_false_for_in_batch_dep_not_yet_completed():
    from nemo_skills.pipeline.utils.ray_backend import RayBackend

    # Premature-start guard: dep is in flight in this batch and not SUCCEEDED.
    in_batch = {"train", "train-handle"}
    assert RayBackend._deps_satisfied(["train"], {}, in_batch) is False
    # Recorded under a non-success terminal state still blocks.
    assert RayBackend._deps_satisfied(["train"], {"train": "FAILED"}, in_batch) is False


def test_deps_satisfied_true_when_in_batch_dep_succeeded_by_task_name_or_handle():
    from nemo_skills.pipeline.utils.ray_backend import RayBackend

    in_batch = {"train", "train-handle"}
    # Recorded SUCCEEDED under task_name.
    assert RayBackend._deps_satisfied(["train"], {"train": "SUCCEEDED"}, in_batch) is True
    # Recorded SUCCEEDED under the nemo-run handle.
    assert RayBackend._deps_satisfied(["train-handle"], {"train-handle": "SUCCEEDED"}, in_batch) is True


def test_deps_satisfied_true_for_cross_experiment_dep_gated_upstream():
    from nemo_skills.pipeline.utils.ray_backend import RayBackend

    # Dep matches no job in this batch -> gated upstream -> treated satisfied.
    in_batch = {"judge", "judge-handle"}
    assert RayBackend._deps_satisfied(["prior-experiment-handle"], {}, in_batch) is True


def test_deps_satisfied_mixed_in_batch_pending_and_cross_experiment():
    from nemo_skills.pipeline.utils.ray_backend import RayBackend

    in_batch = {"train", "train-handle"}
    deps = ["train", "prior-experiment-handle"]
    # Blocked while the in-batch dep is still pending, even though the other is gated.
    assert RayBackend._deps_satisfied(deps, {}, in_batch) is False
    # Unblocks once the in-batch dep succeeds.
    assert RayBackend._deps_satisfied(deps, {"train": "SUCCEEDED"}, in_batch) is True


def test_deps_satisfied_true_for_empty_or_none_deps():
    from nemo_skills.pipeline.utils.ray_backend import RayBackend

    assert RayBackend._deps_satisfied([], {}, set()) is True
    assert RayBackend._deps_satisfied(None, {}, set()) is True


# ---------------------------------------------------------------------------
# Poll-failure cleanup
# ---------------------------------------------------------------------------


def test_handle_poll_failure_stops_jobs_once_and_chains_exception():
    from nemo_skills.pipeline.utils.ray_backend import RayBackend

    backend = RayBackend(dashboard_url="http://ray-head:8265")

    calls = []

    def _record(client, job_ids, *, reason):
        calls.append((client, list(job_ids), reason))

    backend._stop_jobs_best_effort = _record  # type: ignore[method-assign]

    client = object()
    job_ids = ["job-1", "job-2"]
    original = ValueError("poll boom")

    with pytest.raises(RuntimeError) as excinfo:
        backend._handle_poll_failure(client, job_ids, original)

    # Cleanup attempted exactly once for the given job ids with the poll-failure reason.
    assert calls == [(client, ["job-1", "job-2"], "poll-failure")]
    # Wrapped error is chained from the original via ``from exc``.
    assert excinfo.value.__cause__ is original


def test_stop_jobs_best_effort_continues_when_one_stop_raises():
    from nemo_skills.pipeline.utils.ray_backend import RayBackend

    backend = RayBackend(dashboard_url="http://ray-head:8265")

    stopped = []

    class FlakyClient:
        def stop_job(self, job_id):
            stopped.append(job_id)
            if job_id == "job-1":
                raise RuntimeError("stop failed")

        def get_job_status(self, job_id):
            # Report terminal immediately so verification does not block.
            return "STOPPED"

    # Should not propagate the per-job stop error and should attempt every job.
    backend._stop_jobs_best_effort(FlakyClient(), ["job-1", "job-2"], reason="poll-failure")

    assert sorted(stopped) == ["job-1", "job-2"]


# ---------------------------------------------------------------------------
# Backend-name parsing helpers (string-form safe)
# ---------------------------------------------------------------------------


def test_get_backend_name_handles_string_and_dict_and_alias_forms():
    # Bare-string form must not crash and must normalize like the dict form.
    assert get_backend_name({"backend": "ray"}) == "ray"
    assert get_backend_name({"backend": {"name": "Ray"}}) == "ray"
    assert get_backend_name({"backend": "  Kubernetes-Ray  "}) == "kubernetes-ray"
    assert get_backend_name({"execution_backend": "ray"}) == "ray"
    assert get_backend_name({}) == ""
    assert get_backend_name({"backend": {}}) == ""


def test_is_ray_backend_name_recognizes_aliases_and_rejects_default():
    assert is_ray_backend_name({"backend": "ray"}) is True
    assert is_ray_backend_name({"backend": {"name": "kubernetes-ray"}}) is True
    assert is_ray_backend_name({"backend": "ray_kubernetes"}) is True
    assert is_ray_backend_name({"backend": "ray-kubernetes"}) is True
    assert is_ray_backend_name({"backend": "default"}) is False
    assert is_ray_backend_name({"backend": "none"}) is False
    assert is_ray_backend_name({}) is False


def test_default_backend_path_does_not_import_ray_backend():
    """The default / Slurm resolution path must not import the heavy ray_backend module
    (the decoupling nit). Verified in a fresh interpreter so it is order-independent, while
    the historical ``from ...backends import RayBackend`` re-export still works (lazily).
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import nemo_skills.pipeline.utils.backends as b\n"
        "assert 'nemo_skills.pipeline.utils.ray_backend' not in sys.modules, 'imported at module load'\n"
        "b.get_execution_backend({'executor': 'slurm'})\n"
        "b.get_execution_backend({'executor': 'slurm', 'backend': {'name': 'default'}})\n"
        "embedded = b.get_execution_backend({'executor': 'slurm'}, with_ray=True)\n"
        "assert embedded.name == 'default'\n"
        "assert embedded.stage_metadata(use_with_ray_cluster=True) == "
        "{'use_with_ray_cluster': True}\n"
        "assert 'nemo_skills.pipeline.utils.ray_backend' not in sys.modules, 'imported on default path'\n"
        "from nemo_skills.pipeline.utils.backends import RayBackend\n"
        "assert RayBackend.__name__ == 'RayBackend'\n"
        "assert 'nemo_skills.pipeline.utils.ray_backend' in sys.modules, 're-export should import it'\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# run_exp must honor --dry_run before any live remote I/O (non-Ray regression guard)
# ---------------------------------------------------------------------------


def test_run_exp_dry_run_skips_remote_mount_check(monkeypatch):
    """A --dry_run must not perform live remote I/O on the default (non-Ray) path: the mount
    check opens an SSH tunnel and stats each source on the cluster. It must be skipped on a
    dry run so a dry run stays offline-safe (guards the regression where the early dry-run
    short-circuit was removed)."""
    from unittest.mock import MagicMock

    from nemo_skills.pipeline.utils import exp as exp_mod

    calls = []
    monkeypatch.setattr(exp_mod, "get_mounts_from_config", lambda cc: cc["mounts"])
    monkeypatch.setattr(exp_mod, "check_remote_mount_directories", lambda *a, **k: calls.append(("check", a)))

    cluster_config = {"executor": "slurm", "mounts": ["/src:/dst"]}
    # Default backend + slurm + dry_run: must return without touching the cluster.
    exp_mod.run_exp(MagicMock(), cluster_config, dry_run=True)
    assert calls == []

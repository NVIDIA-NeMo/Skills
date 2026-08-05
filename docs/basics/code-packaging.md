# Code packaging

We use [NeMo-Run](https://github.com/NVIDIA-NeMo/Run) for managing our experiments with local and slurm-based
execution supported (please open an issue if you need to run our code on other kinds of clusters).
This means that even if you need to submit jobs on slurm, you can do it from your local machine by defining an
appropriate cluster config and nemo-run will package and upload your code, data and manage
all complexities of slurm scheduling. Check their documentation to learn how to fetch logs, check status,
cancel jobs, etc.

To decide which code to package we use the following logic:

1. If you run commands from inside a cloned Nemo-Skills repository, we will package that repository.
2. If you run commands from inside a git repository which is not Nemo-Skills (doesn't have `nemo_skills` top-level folder),
   we will package your current repository and also include `nemo_skills` subfolder from its installed location.
3. If you run commands from outside of any git repository, we will only package `nemo_skills` subfolder from its installed
   location.

Put simply, we will always include `nemo_skills` and will additionally include your personal git repository if you're
running commands from it.

!!! note

    When packaging a git repository, NeMo-Run will only package the code tracked by git
    (as well as all jsonl files from `nemo_skills/dataset`).
    Any non-tracked files will not be automatically available inside the container or uploaded to slurm.

    When packaging `nemo_skills` from its installed location (which might not be a git repository), we will
    upload **all** the files inside `nemo_skills` subfolder. Make sure you do not store any large files there
    to avoid uploading them on the cluster with each experiment!

!!! note

    When you run commands from a git repo with uncommitted changes, NeMo-Run throws the following error
    ```
    RuntimeError: Your repo has uncommitted changes. Please commit your changes or set check_uncommitted_changes to False to proceed with packaging.
    ```
    This error can be avoided by either taking care of the uncommitted changes (via commit/revert), or setting the environment variable:
    ```bash
    export NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1
    ```
    In all cases, uncommitted code will not be used.

!!! note

    You can override the default packaging behavior with the following environment variables:

    - `NEMO_SKILLS_FORCE_PATTERN_PACKAGER=1` — Skip git-based packaging entirely and always use the installed
      `nemo_skills` package tree (PatternPackager). Useful when you have an editable install and don't want
      packaging tied to the git state of your current directory.
    - `NEMO_SKILLS_FORCE_INSTALLED_PACKAGE=1` — When running from a git repo, use the installed `nemo_skills`
      package instead of the repo's `nemo_skills/` directory. The git repo is still packaged, but `nemo_skills`
      is picked up from the installed location. Useful when your repo checkout has extra files you don't want
      uploaded.

    Note that `NEMO_SKILLS_FORCE_INSTALLED_PACKAGE` has no effect when `NEMO_SKILLS_FORCE_PATTERN_PACKAGER`
    is also set, since the latter bypasses the git repo branch entirely.


Finally, it's important to keep in mind that whenever you submit a new experiment, NeMo-Run will create a copy of your
code package both locally (inside `~/.nemo_run`) and on cluster (inside `ssh_tunnel/job_dir` path in your cluster config).
If you submit multiple experiments from the same Python script, they will all share code, so only one copy will be
created per run of that script. Even so, at some point, the code copies will be accumulated and you will run out of
space both locally and on cluster. There is currently no automatic cleaning, so you have to monitor for that and
periodically remove local and cluster nemo-run folders to free up space. There is no side effect of doing that (they will
be automatically recreated) as long as you don't have any running jobs when you remove the folders.
If you want to have more fine-grained control over code reuse, you can directly specify `--reuse_code_exp` argument when submitting jobs

While our job submission is somewhat complicated and goes through NeMo-Run, at the end, we simply execute a particular sbatch file
that is uploaded to the cluster. It is helpful sometimes to see what's in it and modify directly. You can find sbatch file(s)
for each job inside `ssh_tunnel.job_dir` cluster folder that is defined in your cluster config.

## Ray Jobs code delivery

The Ray Jobs backend can optionally deliver source code with Ray's native
`runtime_env.working_dir` packaging:

```yaml
executor: none
backend:
  name: ray
  dashboard_url: http://<ray-head>:8265
  working_dir: /opt/my-project  # local directory or local .zip on the submitter
```

When `working_dir` is set, Ray uploads that directory or archive and makes it
the submitted job's current directory. It does not run `pip`, `conda`, or `uv`,
so every dependency must already be present in the Ray worker image. For a
strict-airgap launch, point it at an immutable source directory or archive baked
into the launcher image. If the option is absent, Ray Jobs retain their existing
behavior and no working directory is delivered.

Before starting the command, the backend captures Ray's initial uploaded-code
directory in `NEMO_RUN_CODE_DIR`. Legacy `/nemo_run/code/...` command paths are
rewritten to that absolute root, so they continue to work if a workload later
changes directory (for example, `cd /opt/Gym`). Legacy
`/nemo_run/code/nemo_skills/...` paths resolve separately from the NeMo-Skills
package baked into the worker image; the working-directory archive does not need
to duplicate that package. These rewrites apply only to Ray Jobs with an explicit
`working_dir`; local `executor: none`, embedded Ray-on-Slurm, and ordinary Slurm
execution keep their existing path and packaging behavior.

The backend rewrites paths in generated entrypoint command strings. It does not
edit YAML, JSON, or other files inside the uploaded archive. Configuration files
that must refer to the delivered source after changing directory should resolve
`NEMO_RUN_CODE_DIR` themselves (and may retain `/nemo_run/code` as their non-Ray
default).

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

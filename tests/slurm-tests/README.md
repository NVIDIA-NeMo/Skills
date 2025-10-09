# Slurm tests

To add a new Slurm test follow this process:
1. Create a new folder with the descriptive name for the test.
2. Add a run_test.py that will launch the main test jobs. Can run arbitrary pipelines here.
3. Add a check_results.py that will operate on the output of run_test.py and do quality checks. E.g. can check benchmark accuracy range or the presence of certain files, etc.
4. Update run_test.py to schedule check_results.py as the final job.
5. Add your new test into [./run_all.sh](./run_all.sh)

You can always run tests manually from any branch by running

```bash
./run_all.sh <cluster name>
```

You can change CURRENT_DATE to any value there to ensure you don't
accidentally override results of existing pipeline.

See [./clone_and_run.sh](./clone_and_run.sh) for how to register tests to run on schedule with cron.

**Note on SWE-bench tests ([qwen3coder_30b_swebench](qwen3coder_30b_swebench)):** by default, run_all.sh assumes your cluster config has a mount called `/swe-bench-images` where the SWE-bench Verified images are downloaded. See the [SWE-bench docs](https://nvidia.github.io/NeMo-Skills/evaluation/code/#data-preparation) for more details. To use a different path, you can modify the test's --container_formatter parameter, or you can remove it entirely to pull the images from Dockerhub every time the test is run (not recommended).

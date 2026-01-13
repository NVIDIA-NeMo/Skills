# Code

More details are coming soon!

## Supported benchmarks

### swe-bench

!!! note
    While swe-bench evaluation will work out-of-the-box without extra setup, it won't be efficient as we will be re-downloading docker containers
    each time it's launched. Please read [below](#data-preparation) for the details of how to prepare the containers beforehand to avoid this.
    The downloaded containers will take around 650Gb of space, but will make evaluations considerably faster.

- Benchmark is defined in [`nemo_skills/dataset/swe-bench/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/swe-bench/__init__.py)
- Original benchmark source is [here](https://github.com/SWE-bench/SWE-bench).

Nemo-Skills can run inference (rollout) on SWE-bench-style datasets using 2 agentic frameworks: [SWE-agent](https://swe-agent.com/latest/) and [OpenHands](https://www.all-hands.dev/). It can then evaluate the generated patches on SWE-bench Verified/Lite/Full using the [official SWE-bench harness](https://www.swebench.com/SWE-bench/guides/evaluation/).

#### Data preparation

Before running `ns eval`, you will need to prepare the data with this command:

```
ns prepare_data swe-bench
```

This command downloads the SWE-bench Verified dataset. If you want to use a different dataset, you can use the **--dataset_name** and **--split** options to set the HuggingFace path and split respectively.

By default the dataset is downloaded to `nemo_skills/dataset/swe-bench/default.jsonl`. To download to a different file, use the **--setup** option, e.g. `--setup custom` will download to `nemo_skills/dataset/swe-bench/custom.jsonl`. You can then evaluate on this dataset with the `--split` option of `ns eval`, e.g. `ns eval --split custom`.

SWE-bench inference and evaluation runs inside of prebuilt container images from the SWE-bench team. By default, this command will configure them to be downloaded from Dockerhub every time you run `ns eval`. To avoid this we recommend to download the images beforehand in .sif format and include that path in the data file, so it
can be used in the evaluation job.
Note that you can follow the steps below irrespective of whether you're running locally or on Slurm, assuming you have enough disk space (~650Gb) to store all containers.

Here's how you can use it to download all images for SWE-bench Verified:

1. Start by preparing the data with the default command: `ns prepare_data swe-bench`
2. Determine the folder you want to download the images into. Make sure it is accessible from inside the Nemo-Skills container, e.g. mounted in your cluster config.
3. Run the download script on the cluster:
   ```
   ns run_cmd \
     --cluster=<CLUSTER_NAME> \
     --command="python nemo_skills/dataset/swe-bench/dump_images.py \
                nemo_skills/dataset/swe-bench/default.jsonl \
                <MOUNTED_PATH_TO_IMAGES_FOLDER>"
   ```
   If any images fail to download, you can rerun the exact same command and it will automatically re-attempt to download the missing images, skipping the ones that were already downloaded.

4. Rerun `ns prepare_data`, using the `--container_formatter` option to specify the path to the newly downloaded images, as shown below.

   ```
   ns prepare_data swe-bench \
       --container_formatter "<MOUNTED_PATH_TO_IMAGES_FOLDER>/swebench_sweb.eval.x86_64.{instance_id}.sif"
   ```

You can use any existing mounted path in your cluster config or define a new one, e.g.

```
mounts:
  - <CLUSTER_PATH_TO_FOLDER_WITH_IMAGES>:/swe-bench-images
```

When this path is accessed during evaluation, `{instance_id}` will be replaced by the value of the instance_id column in the dataset, replacing `__` with `_1776_`. For example, `astropy__astropy-12907` becomes `astropy_1776_astropy-12907`.

#### SWE-bench-specific parameters

There are a few parameters specific to SWE-bench. They have to be specified with the `++` prefix. All of them are optional, except for ++agent_framework.

- **++agent_framework:** which agentic framework to use. Must be either `swe_agent` or `openhands`. No default, must be specified explicitly.

- **++agent_framework_repo:** URL of the repository to use for SWE-agent/OpenHands. Allows you to pass in a custom fork of these repositories. If you do this, you may find it helpful to check [nemo_skills/inference/eval/swebench.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/inference/eval/swebench.py) to understand how the frameworks are used internally. This is passed directly as an argument to `git clone`. Defaults to the official repositories: [`https://github.com/SWE-agent/SWE-agent.git`](https://github.com/SWE-agent/SWE-agent) for SWE-agent, [`https://github.com/All-Hands-AI/OpenHands.git`](https://github.com/All-Hands-AI/OpenHands) for OpenHands.

- **++agent_framework_commit:** The commit hash, branch or tag to checkout after cloning agent_framework_repo. Allows you to pin SWE-agent/OpenHands to a specific version. Defaults to `HEAD`, i.e. the latest commit.

- **++agent_config:** The config file to use for SWE-agent/OpenHands.
    - For SWE-agent, this is a YAML file. See the [SWE-agent docs](https://swe-agent.com/latest/config/config/).
    - For OpenHands, this is a TOML file. Nemo-Skills runs OpenHands via their SWE-bench evaluation script, so the only settings you can set are the LLM settings under the `[llm.model]` section. For more details, see the [OpenHands evaluation README](https://github.com/All-Hands-AI/OpenHands/blob/main/evaluation/README.md). Note that Nemo-Skills always uses the `[llm.model]` config section and therefore does not support multiple LLM configurations in one TOML file.
    - Nemo-Skills overrides certain parameters, even if they are specified in the config file. These parameters are listed in a comment in the default config files below.
    - Defaults to [eval/swe-bench/swe-agent/default](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config/eval/swe-bench/swe-agent/default.yaml) for SWE-agent, [eval/swe-bench/openhands/default](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config/eval/swe-bench/openhands/default.toml) for OpenHands. Note that if you store your configs in your local Nemo-Skills repo, then the path can be relative to the `nemo_skills/prompt` folder and the file extension is added automatically (same as how it works with regular [prompt configs](../basics/prompt-format.md)).

- **++agent_max_turns:** The maximum number of turns the agent is allowed to take before the trajectory is forcibly terminated. Defaults to 100 for both SWE-agent and OpenHands.

- **++eval_harness_repo:** URL of the repository to use for the evaluation harness. This is passed directly as an argument to `git clone`. Defaults to [`https://github.com/Kipok/SWE-bench.git`](https://github.com/Kipok/SWE-bench), our fork of SWE-bench that supports local evaluation.

- **++eval_harness_commit:** The commit hash, branch or tag to checkout after cloning eval_harness_repo. Defaults to `HEAD`, i.e. the latest commit.

- **++setup_timeout:** The timeout for downloading & installing the agent framework and the evaluation harness, in seconds. Defaults to 1200, i.e. 20 minutes.

- **++swebench_tests_timeout:** The timeout for tests after applying the generated patch during evaluation, in seconds. Defaults to 1800, i.e. 30 minutes.

- **++max_retries:** How many times to try running setup, inference and evaluation until a valid output file is produced. Defaults to 3.

- **++min_retry_interval, ++max_retry_interval:** The interval between retries, in seconds. Selected randomly between min and max on each retry. Defaults to 60 and 180 respectively.

#### Inference parameters

For this benchmark, inference parameters work a bit differently. This is because it does not use the Nemo-Skills LLM client, instead the interaction with the LLM server is handled by SWE-agent/OpenHands. Most inference parameters are not passed to the LLM by default if you don't explicitly specify them, and some parameters may be unsupported, e.g. when using OpenHands.

In order for a parameter to work, it needs to be supported in 2 places: by the agentic framework and by the LLM server itself. For framework support, see the following table:

| Nemo-Skills inference parameter | Behavior when using SWE-agent          | Behavior when using OpenHands          |
| :------------------------------ | :------------------------------------- | :------------------------------------- |
| temperature                     | ✅ Always passed to LLM. Default: 0     | ✅ Always passed to LLM. Default: 0     |
| top_p                           | ✅ Always passed to LLM. Default: 0.95  | ✅ Always passed to LLM. Default: 0.95  |
| top_k                           | 🟡 Only passed to LLM if set explicitly | 🟡 Only passed to LLM if set explicitly |
| tokens_to_generate              | 🟡 Only passed to LLM if set explicitly | 🟡 Only passed to LLM if set explicitly |
| random_seed                     | 🟡 Only passed to LLM if set explicitly | 🟡 Only passed to LLM if set explicitly |
| min_p                           | 🟡 Only passed to LLM if set explicitly | ⛔ Not supported, will fail if set      |
| repetition_penalty              | 🟡 Only passed to LLM if set explicitly | ⛔ Not supported, will fail if set      |
| top_logprobs                    | 🟡 Only passed to LLM if set explicitly | ⛔ Not supported, will fail if set      |

In addition, keep in mind certain parameters may not be supported by your LLM server, because not all of them are part of the official [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create). However, VLLM and SGLang do support all of these parameters.

It's worth noting that when using VLLM with a HuggingFace model, any parameters that are not passed to the server will be taken from the model's config on HuggingFace by default. This may or may not be what you want. To disable this, you can add `--generation-config vllm` to the `--server_args` parameter. See [VLLM docs](https://docs.vllm.ai/en/latest/configuration/engine_args.html#-generation-config).

#### Tool calling

SWE-bench requires models to call custom tools. By default SWE-agent & OpenHands expect that the LLM server supports *native tool calling*, which means the server can parse the model's tool calls and return them in a structured format separately from the rest of the model's output. This is convenient because the agentic framework doesn't have to know what the model's preferred tool call format is. In order to set this up, you need to add these arguments to `--server_args`:

- for VLLM: `--enable-auto-tool-choice --tool-call-parser <PARSER_NAME>`
- for SGLang: `--tool-call-parser <PARSER_NAME>`

For more details and the list of supported parsers, see the docs: [VLLM](https://docs.vllm.ai/en/stable/features/tool_calling.html#automatic-function-calling), [SGLang](https://docs.sglang.ai/advanced_features/function_calling.html).

In addition, both SWE-agent and OpenHands can run without native tool calling. This means the tool calls will be parsed by the agentic framework itself. To try this out, you can use the following configs with the `++agent_config` parameter:

- for SWE-agent: [eval/swe-bench/swe-agent/swe-agent-lm-32b](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config/eval/swe-bench/swe-agent/swe-agent-lm-32b.yaml). This was the config used for [SWE-agent-LM-32B](https://huggingface.co/SWE-bench/SWE-agent-LM-32B). Note that there are significant differences with the default config.
- for OpenHands: [eval/swe-bench/openhands/no-native-tool-calling](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config/eval/swe-bench/openhands/no-native-tool-calling.toml). This simply sets `native_tool_calling` to `false`.

Keep in mind that by default the tool call format expected by these frameworks will likely be different from the one that the model was trained on.

#### Sample run

Here's how to run a sample evaluation of [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) with OpenHands on a Slurm cluster.

1. Prepare the data following instructions [above](#data-preparation).
2. Run
```
ns eval \
    --cluster=<CLUSTER_NAME> \
    --model=Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --server_type=vllm \
    --server_args="--enable-auto-tool-choice --tool-call-parser qwen3_coder" \
    --server_nodes=1 \
    --server_gpus=8 \
    --benchmarks=swe-bench \
    --output_dir=<OUTPUT_DIR> \
    --num_chunks=10 \
    ++agent_framework=openhands \
    ++inference.temperature=0.7 \
    ++inference.top_p=0.8 \
    ++inference.top_k=20
```
replacing <...> with your desired parameters.

After all jobs are complete, you can check the results in `<OUTPUT_DIR>/eval-results/swe-bench/metrics.json`. They should look something like this:
```
{
  "swe-bench": {
    "pass@1": {
      "num_entries": 500,
      "gen_seconds": 7172,
      "issues_resolved": 45.0,
      "no_patch": 0.4,
      "patch_cant_apply": 0.8
    }
  }
}
```
Keep in mind there is some variance between runs, so we recommend running evaluation multiple times and averaging out the resolve rate. To do that automatically, you can set `--benchmarks=swe-bench:N`, where N is your desired number of repeats.

To evaluate the same model with SWE-agent,
all you need to do is replace `openhands` with `swe_agent` in the command above.

!!! note
    There are some instances where the gold (ground truth) patches do not pass the evaluation tests. Therefore, it's likely that on those instances even patches that resolve the issue will be incorrectly evaluated as "unresolved". We have observed 11 such instances in SWE-bench Verified: `astropy__astropy-7606`, `astropy__astropy-8707`, `astropy__astropy-8872`, `django__django-10097`, `psf__requests-1724`, `psf__requests-1766`, `psf__requests-1921`, `psf__requests-2317`, `pylint-dev__pylint-6528`, `pylint-dev__pylint-7080`, `pylint-dev__pylint-7277`. Depending on your setup, this set of instances may be different.

!!! note
    For evaluation, we use a [custom fork](https://github.com/Kipok/SWE-bench) of the SWE-bench repository that supports running evaluation inside of an existing container. It may not always have the latest updates from the upstream repo.

### compute-eval

- Benchmark is defined in [`nemo_skills/dataset/compute-eval/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/compute-eval/__init__.py)
- Original benchmark source is [here](https://github.com/NVIDIA/compute-eval).

ComputeEval is a benchmark for evaluating Large Language Models on CUDA code generation tasks. It features handcrafted CUDA programming challenges that test an LLM's capability at writing reliable CUDA code. The benchmark includes functional correctness evaluation through compilation and execution against held-out test suites.

**Prerequisites:** NVIDIA GPU with CUDA Toolkit 12 or greater must be installed, and `nvcc` must be available in your PATH.

#### Data Preparation

First, prepare the dataset by running the `ns prepare_data` command. You can optionally specify a release version:

```bash
ns prepare_data compute-eval --release 2025-1
```

If no release is specified, the default release will be downloaded. This will generate an `eval.jsonl` file in the `nemo_skills/dataset/compute-eval/` directory.

**Note:** You need to set the `HF_TOKEN` environment variable because the dataset requires authentication.

#### Running the Evaluation

Once the data is prepared, you can run the evaluation. Replace `<...>` placeholders with your cluster and directory paths.

This command runs an evaluation of [OpenReasoning-Nemotron-32B](https://huggingface.co/nvidia/OpenReasoning-Nemotron-32B) on a Slurm cluster:

```bash
ns eval \
    --cluster=<CLUSTER_NAME> \
    --model=nvidia/OpenReasoning-Nemotron-32B \
    --server_type=vllm \
    --server_args="--async-scheduling" \
    --server_nodes=1 \
    --server_gpus=8 \
    --benchmarks=compute-eval \
    --data_dir=<DATA_DIR> \
    --output_dir=<OUTPUT_DIR> \
    ++inference.temperature=0.6 \
    ++inference.top_p=0.95 \
    ++inference.tokens_to_generate=16384
```

**Security Note:** ComputeEval executes machine-generated CUDA code. While the benchmark is designed for evaluation purposes, we strongly recommend running evaluations in a sandboxed environment (e.g., a Docker container or virtual machine) to minimize security risks.

#### Verifying Results

After all jobs are complete, you can check the results in `<OUTPUT_DIR>/eval-results/compute-eval/metrics.json`. You can also review `<OUTPUT_DIR>/eval-results/compute-eval/summarized-results/main_*`. They should look something like this:

```
---------------------------- compute-eval -----------------------------
evaluation_mode | num_entries | avg_tokens | gen_seconds | accuracy
pass@1          | 50          | 8432       | 1245        | 64.00%
```

The benchmark reports:
- **accuracy**: Percentage of problems where generated code compiled and passed all tests
- **pass@1**: Same as accuracy for single-solution generation
- **pass@k**: Success rate when generating k solutions per problem (if configured)

### swe-bench-multilingual

- Benchmark is defined in [`nemo_skills/dataset/swe-bench-multilingual/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/swe-bench-multilingual/__init__.py)
- Original benchmark source is [here](https://www.swebench.com/multilingual.html).

SWE-bench Multilingual uses mostly the same logic as regular SWE-bench, so most of the [SWE-bench docs](#swe-bench) apply to it as well. The differences are as follows:

1. For both OpenHands and SWE-agent, instead of using the official repos, we default to using our forks with multilingual-specific fixes and enhancements: https://github.com/ludwig-n/OpenHands and https://github.com/ludwig-n/SWE-agent. In both forks we use the `ns-swe-bench-multilingual` branch by default.
2. For OpenHands, we use the [Multi-SWE-bench entrypoint script](https://github.com/ludwig-n/OpenHands/blob/ns-swe-bench-multilingual/evaluation/benchmarks/multi_swe_bench/scripts/run_infer.sh) instead of the standard SWE-bench one.
3. For SWE-agent, we default to a [different config file](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config/eval/swe-bench/swe-agent/multilingual.yaml) with language-specific prompting.

#### Sample run

Here's how to run a sample evaluation of [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) with OpenHands on a Slurm cluster.

1. Prepare the data following the same [instructions](#data-preparation) as for SWE-bench, replacing `ns prepare_data swe-bench` with `ns prepare_data swe-bench-multilingual`. This will download [SWE-bench Multilingual](https://huggingface.co/datasets/SWE-bench/SWE-bench_Multilingual) by default instead of Verified. The container names have the same format. For downloading images, you can use the same `dump_images.py` script as for SWE-bench.
2. Run
```
ns eval \
    --cluster=<CLUSTER_NAME> \
    --model=Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --server_type=vllm \
    --server_args="--enable-auto-tool-choice --tool-call-parser qwen3_coder" \
    --server_nodes=1 \
    --server_gpus=8 \
    --benchmarks=swe-bench-multilingual \
    --output_dir=<OUTPUT_DIR> \
    --num_chunks=6 \
    ++agent_framework=openhands \
    ++inference.temperature=0.7 \
    ++inference.top_p=0.8 \
    ++inference.top_k=20
```
replacing <...> with your desired parameters.

After all jobs are complete, you can check the results in `<OUTPUT_DIR>/eval-results/swe-bench-multilingual/metrics.json`. They should look something like this:
```
{
  "swe-bench-multilingual": {
    "pass@1": {
      "num_entries": 300,
      "gen_seconds": 83685,
      "issues_resolved": 33.33333333333336,
      "no_patch": 0.6666666666666665,
      "patch_cant_apply": 1.0
    }
  }
}
```
Keep in mind there is some variance between runs, so we recommend running evaluation multiple times and averaging out the resolve rate. To do that automatically, you can set `--benchmarks=swe-bench-multilingual:N`, where N is your desired number of repeats.

To evaluate the same model with SWE-agent,
all you need to do is replace `openhands` with `swe_agent` in the command above.

!!! note
    There are some instances where the gold (ground truth) patches do not pass the evaluation tests. Therefore, it's likely that on those instances even patches that resolve the issue will be incorrectly evaluated as "unresolved". We have observed 2 such instances in SWE-bench Multilingual: `jqlang__jq-2681` and `tokio-rs__tokio-4384`. In addition, 5 instances behave inconsistently (gold patches sometimes pass and sometimes fail): `axios__axios-4731`, `axios__axios-4738`, `axios__axios-5892`, `caddyserver__caddy-5995`, `valkey-io__valkey-928`. Depending on your setup, this set of instances may be different.

!!! note
    For evaluation, we use a [custom fork](https://github.com/Kipok/SWE-bench) of the SWE-bench repository that supports running evaluation inside of an existing container. It may not always have the latest updates from the upstream repo.


### IOI

We currently support IOI24 and are working to support IOI25 for evaluation. The original data for IOI24 can be seen [here](https://huggingface.co/datasets/open-r1/ioi).

#### Data Preparation

First, prepare the dataset by running the `ns prepare_data` command. The arguments below will generate `ioi24.jsonl` and `ioi24_metadata.json`.

```
ns prepare_data ioi
```

#### Running the Evaluation

Once the data is prepared, you can run the evaluation. Replace `<...>` placeholders with your cluster and directory paths.
Note you have to provide the path to the metadata test file generated from preparing the data. To follow IOI submission rules, we generate 50 solutions per sub-task.

This command runs an evaluation of [OpenReasoning-Nemotron-32B](https://huggingface.co/nvidia/OpenReasoning-Nemotron-32B) on a Slurm cluster.


```
ns eval \
    --cluster=<CLUSTER_NAME> \
    --model=nvidia/OpenReasoning-Nemotron-32B \
    --server_type=vllm \
    --server_args="--async-scheduling" \
    --server_nodes=1 \
    --server_gpus=8 \
    --benchmarks=ioi24:50 \
    --with_sandbox \
    --split=ioi24 \
    --data_dir=<DATA_DIR> \
    --output_dir=<OUTPUT_DIR> \
    --eval_subfolder=eval-results/ioi24/ \ # set the folder if you want to differentiate subsets.
    --extra_eval_args="++eval_config.test_file=<PATH_TO_METADATA_TEST_DIR>/ioi24_metadata.json" \
    ++inference.temperature=0.6 \
    ++inference.top_p=0.95 \
    ++inference.tokens_to_generate=65536
```

##### Verifying Results

After all jobs are complete, you can check the results in `<OUTPUT_DIR>/eval-results/ioi24/ioi/metrics.json`. You can also take a look at `<OUTPUT_DIR>/eval-results/ioi24/ioi/summarized-results/main_*`. They should look something like this:

```
------------------------------------ ioi24 -------------------------------------
evaluation_mode | num_entries | avg_tokens | gen_seconds | correct | total_score
pass@50          | 39          | 52225      | 99630       | 23.08%  | 500
```

### livecodebench

- Benchmark is defined in [`nemo_skills/dataset/livecodebench/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/livecodebench/__init__.py)
- Original benchmark source is [here](https://github.com/LiveCodeBench/LiveCodeBench).

#### Data Preparation

First, prepare the dataset by running the `ns prepare_data` command. The arguments below will generate `test_v6_2408_2505.jsonl`.

```
ns prepare_data livecodebench --release_version v6 --start_date 2024-08 --end_date 2025-05
```

##### For Pypy3 Evaluation:
If you plan to evaluate using the Pypy3 interpreter, you must add the `--keep_all_columns` flag during data preparation. This will download a larger dataset (~1.9GB) containing the necessary test cases. So, we recommend downloading the dataset into a Slurm cluster location.

```
ns prepare_data livecodebench --release_version v6 --start_date 2024-08 --end_date 2025-05 --keep_all_columns --cluster=<CLUSTER_NAME> --data_dir=<DATA_DIR>
```

#### Running the Evaluation

Once the data is prepared, you can run the evaluation. Replace `<...>` placeholders with your cluster and directory paths.

##### Standard Python Evaluation

This command runs an evaluation of [OpenReasoning-Nemotron-32B](https://huggingface.co/nvidia/OpenReasoning-Nemotron-32B) on a Slurm cluster.

```
ns eval \
    --cluster=<CLUSTER_NAME> \
    --model=nvidia/OpenReasoning-Nemotron-32B \
    --server_type=vllm \
    --server_args="--async-scheduling" \
    --server_nodes=1 \
    --server_gpus=8 \
    --benchmarks=livecodebench \
    --split=test_v6_2408_2505 \
    --data_dir=<DATA_DIR> \
    --output_dir=<OUTPUT_DIR> \
    ++parse_reasoning=True \
    ++eval_config.interpreter=python \
    ++inference.temperature=0.6 \
    ++inference.top_p=0.95 \
    ++inference.tokens_to_generate=65536
```

##### Pypy3 Evaluation

To run with the Pypy3 interpreter, we need to use sandbox. Therefore, pass these flags `--with_sandbox --keep_mounts_for_sandbox` and also add the following arguments
```
++eval_config.interpreter=pypy3 ++eval_config.test_file=<DATA_DIR>/livecodebench/test_v6_2408_2505.jsonl
```

##### Verifying Results

After all jobs are complete, you can check the results in `<OUTPUT_DIR>/eval-results/livecodebench/metrics.json`. You can also take a look at `<OUTPUT_DIR>/eval-results/livecodebench/summarized-results/main_*` They should look something like this:

```
-------------------------- livecodebench --------------------------
evaluation_mode | num_entries | avg_tokens | gen_seconds | accuracy
pass@1          | 454         | 15995      | 2188        | 71.15%


------------------------ livecodebench-easy -----------------------
evaluation_mode | num_entries | avg_tokens | gen_seconds | accuracy
pass@1          | 110         | 5338       | 1806        | 99.09%


------------------------ livecodebench-hard -----------------------
evaluation_mode | num_entries | avg_tokens | gen_seconds | accuracy
pass@1          | 203         | 23031      | 2188        | 46.31%


----------------------- livecodebench-medium ----------------------
evaluation_mode | num_entries | avg_tokens | gen_seconds | accuracy
pass@1          | 141         | 14178      | 1889        | 85.11%
```

##### Advanced: Averaging Multiple Runs

Due to variance between runs, you can automatically repeat the evaluation and average the results. To run the evaluation 3 times, for example, set the `--benchmarks` flag as follows:

```
--benchmarks=livecodebench:3
```

### BIRD

The [BIRD benchmark](https://bird-bench.github.io/) is currently the only text-to-SQL benchmark that is supported. Evaluation is based on the SQL evaluation accuracy calculated in [this file](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/evaluation.py) provided in the BIRD GitHub repository.

#### Data Preparation


First, the data must be downloaded and prepared, which you can do by running:
```bash
ns prepare_data birdbench --cluster=<CLUSTER_NAME> --data_dir=<DATA_DIR>
```

This will download and unpack a file into `<DATA_DIR>/birdbench/dev_20240627`, which contains the BIRD dev manifest, table information, and database schemas.
The script will also process the original manifest into `<DATA_DIR>/birdbench/dev.jsonl`, which will be the input for evaluation.
`<DATA_DIR>` should be a path to the mount point where you want this data to be stored.

See [the "Using data on cluster" documentation](./index.md#Using-data-on-cluster) for more information.

#### Running the Evaluation

The following command runs an evaluation of [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) on a Slurm cluster.

```bash
ns eval \
     --cluster=<CLUSTER_NAME> \
     --server_type='sglang' \
     --server_gpus=8 \
     --model=Qwen/Qwen3-8B \
     --benchmarks=birdbench \
     --data_dir=<DATA_DIR> \
     --output_dir=<OUTPUT_DIR> \
     ++inference.tokens_to_generate=10000 \
     ++inference.temperature=0.6 \
     ++inference.top_p=0.95 \
     ++inference.top_k=20 \
     ++max_concurrent_requests=1024 \
```
You should specify: `<CLUSTER_NAME>`, which should match your cluster config name; `<DATA_DIR>`, which should be the location where your dataset is mounted from the cluster; and `<OUTPUT_DIR>`.
The former two arguments should match what you used in `prepare_data`.

### livecodebench-cpp

- Benchmark is defined in [`nemo_skills/dataset/livecodebench-cpp/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/livecodebench-cpp/__init__.py)
- Original benchmark source is [here](https://huggingface.co/datasets/nvidia/LiveCodeBench-CPP).
- Data preparation and evaluation: you can prepare the dataset by running `ns prepare_data livecodebench-cpp`. The command will generate two dataset splits: `v5_2408_2501.jsonl` and `v6_2408_2505.jsonl`. When evaluating, make sure to target the C++ benchmark entrypoint (`--benchmarks=livecodebench-cpp`) and set `--split` to either `v5_2408_2501` or `v6_2408_2505`. The remaining flags mirror the livecodebench instructions above.


### livecodebench-pro

- Benchmark is defined in [`nemo_skills/dataset/livecodebench-pro/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/livecodebench-pro/__init__.py)
- Original benchmark source is [here](https://github.com/GavinZhengOI/LiveCodeBench-Pro).

#### Data Preparation

First, prepare the dataset by running the `ns prepare_data` command. The arguments below will generate `test_24q4.jsonl`, `test_25q1.jsonl`, `test_25q2.jsonl`, and `test_25q3.jsonl` files.

```
ns prepare_data livecodebench-pro --cluster=local --data_dir=/workspace/ns-data
```

Note that, this will also download testcases and keep it at `/workspace/ns-data/livecodebench-pro/testcases`. We recommend using a cluster data location since the testcases directory would be of size 15GB.

#### Running the Evaluation

```
ns eval \
    --cluster=<CLUSTER_NAME> \
    --model=nvidia/OpenReasoning-Nemotron-32B \
    --server_type=vllm \
    --server_args="--async-scheduling" \
    --server_nodes=1 \
    --server_gpus=8 \
    --benchmarks=livecodebench-pro \
    --split=test_25q2 \
    --data_dir=/workspace/ns-data/livecodebench-pro \
    --output_dir=<OUTPUT_DIR> \
    ++parse_reasoning=True \
    ++eval_config.test_file=/workspace/ns-data/livecodebench-pro/test_25q2.jsonl \
    ++eval_config.test_dir=/workspace/ns-data/livecodebench-pro/testcases \
    ++inference.temperature=0.6 \
    ++inference.top_p=0.95 \
    ++inference.tokens_to_generate=65536
```

### human-eval

- Benchmark is defined in [`nemo_skills/dataset/human-eval/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/human-eval/__init__.py)
- Original benchmark source is [here](https://github.com/openai/human-eval).

### mbpp

- Benchmark is defined in [`nemo_skills/dataset/mbpp/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/mbpp/__init__.py)
- Original benchmark source is [here](https://github.com/google-research/google-research/tree/master/mbpp).

### bigcodebench

- Benchmark is defined in [`nemo_skills/dataset/bigcodebench/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/bigcodebench/__init__.py)
- Original benchmark source is [here](https://github.com/bigcode-project/bigcodebench).

### livebench-coding

- Benchmark is defined in [`nemo_skills/dataset/livebench-coding/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/livebench-coding/__init__.py)
- Original benchmark source is [here](https://huggingface.co/datasets/livebench/coding).

### human-eval-infilling

- Benchmark is defined in [`nemo_skills/dataset/human-eval-infilling/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/human-eval-infilling/__init__.py)
- Original benchmark source is [here](https://github.com/openai/human-eval-infilling).

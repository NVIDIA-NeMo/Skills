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

Nemo-Skills can run inference (rollout) on SWE-bench-style datasets using [SWE-agent](https://swe-agent.com/latest/), [mini-SWE-agent](https://mini-swe-agent.com/latest/), [OpenHands](https://www.all-hands.dev/), [OpenCode](https://opencode.ai/), and [Claude Code](https://github.com/anthropics/claude-code). It can then evaluate the generated patches on SWE-bench Verified/Lite/Full using the [official SWE-bench harness](https://www.swebench.com/SWE-bench/guides/evaluation/).

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

- **++agent_framework:** which agent framework to use. Must be one of `swe_agent`, `mini_swe_agent`, `openhands`, `opencode`, `claude_code` or `gold_patch`. The latter option runs evaluation of gold (ground truth) patches from the dataset, skipping the agent rollout. No default, must be specified explicitly.

- **++agent_framework_repo:** URL of the repository to use for SWE-agent/mini-SWE-agent/OpenHands. Allows you to pass in a custom fork of these repositories. If you do this, you may find it helpful to check [nemo_skills/inference/eval/swebench.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/inference/eval/swebench.py) to understand how the frameworks are used internally. This is passed directly as an argument to `git clone`. Defaults to the official repositories: [`https://github.com/SWE-agent/SWE-agent.git`](https://github.com/SWE-agent/SWE-agent) for SWE-agent, [`https://github.com/SWE-agent/mini-swe-agent.git`](https://github.com/SWE-agent/mini-swe-agent) for mini-SWE-agent, [`https://github.com/All-Hands-AI/OpenHands.git`](https://github.com/All-Hands-AI/OpenHands) for OpenHands. Not used for OpenCode or Claude Code, which are installed from npm.

- **++agent_framework_commit:** The commit hash, branch or tag to checkout after cloning agent_framework_repo. Allows you to pin SWE-agent/mini-SWE-agent/OpenHands to a specific version. Defaults to `HEAD` for SWE-agent, `1.2.1` for OpenHands and `v2.0` for mini-SWE-agent. For OpenCode this is the npm version of [`opencode-ai`](https://www.npmjs.com/package/opencode-ai) (default `1.17.11`); for Claude Code it is the npm version of [`@anthropic-ai/claude-code`](https://www.npmjs.com/package/@anthropic-ai/claude-code) (default `2.1.259`).

- **++agent_config:** The config file to use for the agent framework.
    - For SWE-agent and mini-SWE-agent, this is a YAML file. See the docs: [SWE-agent](https://swe-agent.com/latest/config/config/), [mini-SWE-agent](https://mini-swe-agent.com/latest/advanced/yaml_configuration/).
    - For OpenHands, this is a TOML file. Nemo-Skills runs OpenHands via their SWE-bench evaluation script, so the only settings you can set are the LLM settings under the `[llm.model]` section. For more details, see the [OpenHands evaluation README](https://github.com/All-Hands-AI/OpenHands/blob/main/evaluation/README.md). Note that Nemo-Skills always uses the `[llm.model]` config section and therefore does not support multiple LLM configurations in one TOML file.
    - For OpenCode, this is a JSON file merged into `~/.config/opencode/opencode.json`. Nemo-Skills always injects a `provider.nemo` block that points at the same OpenAI-compatible `/v1` URL used by the other harnesses (`baseURL` / dummy `apiKey`). It also configures OpenCode's default agent with `++inference.temperature`, `++inference.top_p`, and `++agent_max_turns`. When set, `++inference.top_k` is added to the model's provider options and forwarded as the nonstandard `top_k` request field supported by vLLM. Nemo-Skills additionally installs `++agent_prompt_config` as an OpenCode instruction file, placing it in system-level context. You do not need to change how the model is served (`--model` / `--server_type=vllm` stay the same).
    - For Claude Code, this is a JSON settings file passed explicitly to the CLI. Nemo-Skills runs Claude Code in bare, non-interactive mode, points it at vLLM's Anthropic Messages API, disables session persistence and background tasks, and allows its local coding tools. The selected `++agent_prompt_config` is appended to Claude Code's system prompt.
    - Nemo-Skills overrides certain parameters, even if they are specified in the config file. These parameters are listed in a comment in the default config files below.
    - Defaults to [eval/swe-bench/swe-agent/default](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config/eval/swe-bench/swe-agent/default.yaml) for SWE-agent, [eval/swe-bench/mini-swe-agent/swebench](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config/eval/swe-bench/mini-swe-agent/swebench.yaml) for mini-SWE-agent, [eval/swe-bench/openhands/default](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config/eval/swe-bench/openhands/default.toml) for OpenHands, [eval/swe-bench/opencode/default](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config/eval/swe-bench/opencode/default.json) for OpenCode, and `eval/swe-bench/claude-code/default` for Claude Code. Note that if you store your configs in your local Nemo-Skills repo, then the path can be relative to the `nemo_skills/prompt` folder and the file extension is added automatically (same as how it works with regular [prompt configs](../basics/prompt-format.md)).

- **++agent_prompt_config:** Markdown prompt added to the task instructions for SWE-agent, mini-SWE-agent, OpenHands, OpenCode, and Claude Code. The path is resolved relative to `nemo_skills/prompt` and the `.md` extension is added automatically. Defaults to `eval/swe-bench/common/solution-originality`. Use `++agent_prompt_config=eval/swe-bench/common/cheats-allowed` for prompting without the solution-originality restrictions, or point it at another Markdown config to add future prompting variants without changing the harness code. For backward compatibility, selecting mini-SWE-agent's `swebench_cheats_allowed` agent config without this option also selects the cheats-allowed prompt.

- **++agent_max_turns:** The maximum number of turns the agent is allowed to take. Defaults to 100. For OpenCode, this is written to `agent.<default_agent>.steps`; after the limit, OpenCode forces a final text-only response instead of allowing more tool calls. For Claude Code, this is passed as `--max-turns`.

- **++agent_timeout:** Hard wall-clock timeout for a Claude Code rollout, in seconds. Defaults to 3600.

- **++opencode_context_window:** The context window advertised to OpenCode for request budgeting and automatic compaction. Defaults to 262144. Set this to the effective context length of the model server, for example `++opencode_context_window=393216` when vLLM is launched with `--max-model-len 393216`. This option only affects OpenCode; the model server enforces its own context limit separately.

- **++claude_code_context_window:** The context window advertised to Claude Code. Defaults to 262144 and should match the effective vLLM context length.

- **++claude_code_model:** A slash-free vLLM served-model alias used by Claude Code. This is required when `--model` is a Hugging Face name or filesystem path containing `/`. Add `--served-model-name <ALIAS>` to `--server_args` and set `++claude_code_model=<ALIAS>` to the same value.

- **++swe_zero_container:** Mounted path to the container to use for SWE-Zero in .sif format. If this option is set, SWE-Zero mode will be enabled. During inference, this will override the `container_formatter` field from the dataset file and run all instances in this container instead, cloning the repo from GitHub before running the agent. The recommended dockerfile for this container is provided [here](https://github.com/NVIDIA-NeMo/Skills/tree/main/dockerfiles/swe-bench/Dockerfile.swe-zero).
    - In SWE-Zero, the agent is prompted not to run tests or execute any code, instead relying only on basic Bash commands and file editing. Therefore, it does not have access to instance-specific Docker environments. For more details, see the [SWE-Zero-to-Hero paper](https://arxiv.org/abs/2604.01496).
    - For OpenHands only, this will also switch the default `agent_framework_commit` to our SWE-Zero branch where the agent is prompted not to use tests or other code execution. For other frameworks this is not supported automatically, though you may modify the prompt yourself in the agent config.
    - This option does not affect evaluation of the generated patches, which still runs in the containers specified in `container_formatter`.

- **++evaluate:** If set to False, disables evaluation (i.e. running unit tests to obtain resolution labels) and runs only inference (i.e. trajectory/patch generation). Defaults to True.

- **++eval_harness_repo:** URL of the repository to use for the evaluation harness. This is passed directly as an argument to `git clone`. Defaults to [`https://github.com/Kipok/SWE-bench.git`](https://github.com/Kipok/SWE-bench), our fork of SWE-bench that supports local evaluation.

- **++eval_harness_commit:** The commit hash, branch or tag to checkout after cloning eval_harness_repo. Defaults to `HEAD`, i.e. the latest commit.

- **++setup_timeout:** The timeout for downloading & installing the agent framework and the evaluation harness, in seconds. Defaults to 1200, i.e. 20 minutes.

- **++swebench_tests_timeout:** The timeout for tests after applying the generated patch during evaluation, in seconds. Defaults to 1800, i.e. 30 minutes.

- **++max_retries:** How many times to try running setup, inference and evaluation until a valid output file is produced. Defaults to 3.

- **++min_retry_interval, ++max_retry_interval:** The interval between retries, in seconds. Selected randomly between min and max on each retry. Defaults to 60 and 180 respectively.

#### Inference parameters

For this benchmark, inference parameters work a bit differently. This is because it does not use the Nemo-Skills LLM client, instead the interaction with the LLM server is handled by the agent framework.

Most inference parameters are not passed to the LLM by default if you don't explicitly specify them, with the exception of temperature (defaults to 0) and top_p (defaults to 0.95). These two parameters are passed to OpenCode through its default agent configuration as well. OpenCode also forwards top_k when `++inference.top_k` is explicitly set; because `top_k` is not part of the standard OpenAI API, this requires a compatible server such as vLLM. Custom request parameters can be set via extra_body, for example `++inference.extra_body.chat_template_kwargs.enable_thinking=False`. OpenCode merges `extra_body` into its model provider options, so the OpenAI-compatible adapter forwards fields such as `chat_template_kwargs` to vLLM unchanged. Keep in mind that some parameters may not be supported by your LLM server.

Claude Code controls its own sampling parameters and does not expose NeMo-Skills' `inference.temperature`, `inference.top_p`, `inference.top_k`, `inference.tokens_to_generate`, or `inference.extra_body` settings. Use the vLLM server configuration or a gateway policy if these values must be overridden.

For OpenCode, the per-turn output-token limit defaults to 131072. Set `++inference.tokens_to_generate` to override it. Nemo-Skills applies the value to both the model's OpenCode config and `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX`; without the latter, OpenCode imposes its own 32000-token runtime cap.

Each OpenCode rollout stores four trajectory artifacts under `trajectories/<instance_id>/`: `opencode.txt` contains only the JSONL event stream from stdout, `opencode.stderr.log` contains diagnostics from stderr, `opencode-session.json` is OpenCode's native root-session export, and `trajectory.json` is that export converted to ATIF v1.7. The native and ATIF exports describe the top-level agent session; delegated subagent sessions are not included.

Each Claude Code rollout stores `claude-code.jsonl`, `claude-code.stderr.log`, `claude-code.exit-code`, `model.patch`, and an ATIF v1.7 `trajectory.json` under the same per-instance trajectory directory. Partial patches are retained when Claude Code reaches its turn limit or exits nonzero.

It's worth noting that when using VLLM with a HuggingFace model, any parameters that are not passed to the server will be taken from the model's config on HuggingFace by default. This may or may not be what you want. To disable this, you can add `--generation-config vllm` to the `--server_args` parameter. See [VLLM docs](https://docs.vllm.ai/en/latest/configuration/engine_args.html#-generation-config).

#### Tool calling

SWE-bench requires models to call custom tools. By default agent frameworks expect that the LLM server supports *native tool calling*, which means the server can parse the model's tool calls and return them in a structured format separately from the rest of the model's output. This is convenient because the agent framework doesn't have to know what the model's preferred tool call format is. In order to set this up, you need to add these arguments to `--server_args`:

- for VLLM: `--enable-auto-tool-choice --tool-call-parser <PARSER_NAME>`
- for SGLang: `--tool-call-parser <PARSER_NAME>`

For more details and the list of supported parsers, see the docs: [VLLM](https://docs.vllm.ai/en/stable/features/tool_calling.html#automatic-function-calling), [SGLang](https://docs.sglang.ai/advanced_features/function_calling.html).

In addition, all supported agent frameworks can run without native tool calling. This means the tool calls will be parsed by the agent framework itself. To try this out, you can use the following configs with the `++agent_config` parameter:

- for SWE-agent: [eval/swe-bench/swe-agent/swe-agent-lm-32b](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config/eval/swe-bench/swe-agent/swe-agent-lm-32b.yaml). This was the config used for [SWE-agent-LM-32B](https://huggingface.co/SWE-bench/SWE-agent-LM-32B). Note that there are significant differences with the default config.
- for mini-SWE-agent: [eval/swe-bench/mini-swe-agent/swebench_xml](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config/eval/swe-bench/mini-swe-agent/swebench_xml.yaml) or [eval/swe-bench/mini-swe-agent/swebench_backticks](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config/eval/swe-bench/mini-swe-agent/swebench_backticks.yaml).
- for OpenHands: [eval/swe-bench/openhands/no-native-tool-calling](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config/eval/swe-bench/openhands/no-native-tool-calling.toml). This simply sets `native_tool_calling` to `false`.

OpenCode expects native tool calling (same as the default OpenHands setup). There is no XML/backtick fallback config.

Claude Code also requires native tool calling. A typical self-hosted invocation uses a slash-free alias shared by vLLM and the harness:

```
ns eval \
    --model=Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --server_type=vllm \
    --server_args="--served-model-name qwen3-coder --enable-auto-tool-choice --tool-call-parser <PARSER_NAME>" \
    --benchmarks=swe-bench \
    --output_dir=<OUTPUT_DIR> \
    ++agent_framework=claude_code \
    ++claude_code_model=qwen3-coder
```

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
      "issues_resolved": 48.4,
      "no_patch": 1.0,
      "patch_cant_apply": 1.6
    }
  }
}
```
Keep in mind there is some variance between runs, so we recommend running evaluation multiple times and averaging out the resolve rate. To do that automatically, you can set `--benchmarks=swe-bench:N`, where N is your desired number of repeats.

To evaluate the same model with SWE-agent or mini-SWE-agent,
all you need to do is replace `openhands` with `swe_agent` or `mini_swe_agent` in the command above.

!!! note
    There are some instances where the gold (ground truth) patches do not pass the evaluation tests. Therefore, it's likely that on those instances even patches that resolve the issue will be incorrectly evaluated as "unresolved". We have observed 11 such instances in SWE-bench Verified: `astropy__astropy-7606`, `astropy__astropy-8707`, `astropy__astropy-8872`, `django__django-10097`, `psf__requests-1724`, `psf__requests-1766`, `psf__requests-1921`, `psf__requests-2317`, `pylint-dev__pylint-6528`, `pylint-dev__pylint-7080`, `pylint-dev__pylint-7277`. Depending on your setup, this set of instances may be different.

!!! note
    For evaluation, we use a [custom fork](https://github.com/Kipok/SWE-bench) of the SWE-bench repository that supports running evaluation inside of an existing container. It may not always have the latest updates from the upstream repo.

### scale-swe

[Scale-SWE](https://github.com/AweAI-Team/ScaleSWE) uses the SWE-bench agent
interfaces for rollout generation, but has its own native evaluation protocol.
NeMo-Skills applies the generated patch in a fresh instance container, injects
the dataset's `f2p_patch` and `f2p_script`, and runs the combined
`FAIL_TO_PASS` and `PASS_TO_PASS` pytest IDs. An instance is resolved only when
every expected test passes. AweAgent is not installed or used at runtime.

Prepare the released dataset with:

```
ns prepare_data scale-swe
```

This loads `PrimeIntellect/Scale-SWE-Verified` and its `train` split by default. The
generated data preserves each instance's `image_url`, `workdir`,
`pre_commands`, and F2P/P2P fields, and renames `parent_commit` to
`base_commit`. The default container
formatter lets Apptainer pull `image_url` directly. For larger evaluations,
pre-convert the required images to SIF files and pass a formatter that uses
`{image_url}` or `{instance_id}` to address the mounted files.

Run rollout generation and evaluation with the same agent options as
SWE-bench:

```
ns eval \
    --cluster=<CLUSTER_NAME> \
    --model=<MODEL> \
    --server_type=vllm \
    --benchmarks=scale-swe \
    --output_dir=<OUTPUT_DIR> \
    ++agent_framework=mini_swe_agent \
    ++agent_max_turns=200
```

Use `++agent_framework=gold_patch ++max_samples=5` as an initial environment
and evaluator smoke test. `++evaluate=False` runs rollout generation without
grading, and `++swebench_tests_timeout` controls the per-instance test timeout
(1800 seconds by default). SWE-Zero is not supported because valid Scale-SWE
rollouts require each instance's native image, `parent_commit`, and
`pre_commands`.

Results are written to
`<OUTPUT_DIR>/eval-results/scale-swe/metrics.json`. The primary metric is
`issues_resolved`; additional counters distinguish empty patches, model patch
application failures, F2P patch application failures, and evaluator errors.

### swe-bench-multilingual

- Benchmark is defined in [`nemo_skills/dataset/swe-bench-multilingual/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/swe-bench-multilingual/__init__.py)
- Original benchmark source is [here](https://www.swebench.com/multilingual.html).

SWE-bench Multilingual uses mostly the same logic as regular SWE-bench, so most of the [SWE-bench docs](#swe-bench) apply to it as well. The differences are as follows:

1. For both OpenHands and SWE-agent, instead of using the official repos, we default to using our forks with multilingual-specific fixes and enhancements: [https://github.com/ludwig-n/OpenHands](https://github.com/ludwig-n/OpenHands) and [https://github.com/ludwig-n/SWE-agent](https://github.com/ludwig-n/SWE-agent). In both forks we use the `ns-swe-bench-multilingual` branch by default.
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

### deep-swe

- Benchmark is defined in [`nemo_skills/dataset/deep-swe/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/deep-swe/__init__.py)
- Original benchmark sources: [GitHub tasks](https://github.com/datacurve-ai/deep-swe), gated [HuggingFace dataset](https://huggingface.co/datasets/datacurve/deep-swe)

DeepSWE is a Harbor-format coding-agent benchmark (113 long-horizon tasks). It reuses the SWE-bench agent logic for generation and scores each task with its Harbor verifier (`tests/test.sh` + `grader.py` + F2P/P2P in `tests/config.json`).

DeepSWE uses mostly the same logic as regular SWE-bench, so most of the [SWE-bench docs](#swe-bench) apply to it as well.
The differences are described below.

#### Data preparation

DeepSWE requires a Harbor-style tasks folder to operate, so the data needs to be prepared on the cluster with `--cluster` and `--data_dir` arguments.

```bash
ns prepare_data deep-swe \
  --cluster=<CLUSTER> \
  --data_dir=/workspace/ns-data \
  --container_formatter "/swe-bench-images/deepswe/{instance_id}.sif"
```

This clones [datacurve-ai/deep-swe](https://github.com/datacurve-ai/deep-swe) temporarily to
build the dataset, writes `default.jsonl`, materializes Harbor task dirs under
`/workspace/ns-data/deep-swe/tasks/`, then deletes the temporary checkout. The pipeline exports
`NEMO_SKILLS_DATA_DIR` from `--data_dir` for the preparation process. Note that this does not download the `.sif` images
themselves, for instructions on that see the SWE-bench docs above.

Useful options:

- `--container_formatter` placeholders: `{instance_id}`, `{task_id}`, `{docker_image}`, `{docker_image_tag}`, `{ext_id}`
- `--repo_url` / `--repo_commit` — override the DeepSWE Harbor git source
- `--setup` — output split name (default: `default`)

#### Evaluation

```bash
ns eval --cluster=<CLUSTER> --benchmarks=deep-swe \
  --data_dir=/workspace/ns-data \
  --model=... --server_type=vllm --server_gpus=8 \
  --output_dir=/path/out \
  ++agent_framework=mini_swe_agent
```

Note that `--data_dir` is required, same as for data preparation. For `++agent_framework`, prefer `mini_swe_agent`, `swe_agent` or `opencode`; OpenHands expects a SWE-bench-shaped local dataset and may not work out of the box on DeepSWE rows. All other regular SWE-bench options are supported. Additionally, there are some DeepSWE-specific options:
- **++tasks_dir:** custom path to the Harbor tasks root on the cluster. This is used to run tests. Defaults to `{data_dir}/deep-swe/tasks`, which is the expected path after standard data preparation.
- **++use_agent_timeouts:** if enabled, limit the agent trajectory duration to the timeout embedded in each task. This is disabled by default since trajectory duration is heavily dependent on the hardware setup when hosting LLMs locally.
- **++use_verifier_timeouts:** if enabled, limit the verifier (evaluation) duration to the timeout embedded in each task. Disabled by default.

Metrics read Harbor `reward.json` (`resolved` / `reward` / `f2p` / `p2p` / `partial`).
Verifier crashes that only write the DeepSWE `reward.txt=-1` sentinel are counted as unresolved / not successfully applied.

### senior-swe-bench

- Benchmark is defined in [`nemo_skills/dataset/senior-swe-bench/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/senior-swe-bench/__init__.py)
- Original benchmark source: [snorkel-ai/senior-swe-bench-v2026.06](https://github.com/snorkel-ai/senior-swe-bench-v2026.06) ([Senior SWE-Bench site](https://senior-swe-bench.snorkel.ai/tasks))

Senior SWE-Bench is a Harbor-format coding-agent benchmark (50 public tasks in v2026.06). It reuses the SWE-bench agent logic for generation and scores each task with its full Harbor verifier (`tests/test.sh`: native verify + LLM rubric/taste judges + optional validation agent).

It follows the same fused agent+Harbor grading pattern as [DeepSWE](#deep-swe). Differences versus DeepSWE:

1. Task repos live under `/repo/{REPO_NAME}` (not `/app`). The agent still works in `/testbed` (copied from the task repo); grading applies `model.patch` into `/repo/$REPO_NAME` before running `test.sh`.
2. The full verifier needs outbound network and judge API keys (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `PORTKEY_API_KEY`, plus optional `SSB_OVERRIDE_*`).
3. Empty `reward.txt` (no `reward.json`) marks an **invalid trial** (infra/validation crash) and must not count as a solve.
4. Gold patches are `solution/oracle.patch` (used with `++agent_framework=gold_patch`).

#### Data preparation

```bash
ns prepare_data senior-swe-bench \
  --cluster=<CLUSTER> \
  --data_dir=/workspace/ns-data \
  --container_formatter "/swe-bench-images/senior-swe-bench/{instance_id}.sif"
```

This clones [snorkel-ai/senior-swe-bench-v2026.06](https://github.com/snorkel-ai/senior-swe-bench-v2026.06) temporarily,
writes `default.jsonl`, materializes Harbor task dirs under
`/workspace/ns-data/senior-swe-bench/tasks/`, then deletes the temporary checkout.
Building / converting task Docker images to `.sif` is separate (pass paths via `--container_formatter`).

Useful options:

- `--container_formatter` placeholders: `{instance_id}`, `{task_id}`, `{docker_image}`, `{base_image}`, `{docker_image_tag}`
- `--repo_url` / `--repo_commit` — override the Harbor git source
- `--setup` — output split name (default: `default`)

#### Evaluation

```bash
# Export judge keys required by the full Senior SWE-Bench verifier
# export ANTHROPIC_API_KEY=...
# export OPENAI_API_KEY=...
# export PORTKEY_API_KEY=...

ns eval --cluster=<CLUSTER> --benchmarks=senior-swe-bench \
  --data_dir=/workspace/ns-data \
  --model=... --server_type=vllm --server_gpus=8 \
  --output_dir=/path/out \
  ++agent_framework=mini_swe_agent
```

`--data_dir` is required. Prefer `mini_swe_agent`, `swe_agent` or `opencode` for `++agent_framework`. Shared DeepSWE-style options also apply:

- **++tasks_dir:** custom Harbor tasks root (default `{data_dir}/senior-swe-bench/tasks`)
- **++use_agent_timeouts** / **++use_verifier_timeouts:** honor per-task timeouts from `task.toml`

Smoke-test grading with `++agent_framework=gold_patch` on a single instance before full agent runs.

Metrics:

- **`issues_resolved`** — basic solve (`reward >= 1`: verifiers + validation when present). Participates in pass@k.
- **`tasteful_issues_resolved`** — official [tasteful solve](https://senior-swe-bench.snorkel.ai/): basic solve **and** rubric `> 0.5`, bloat `< 2×`, practice alignment `> 2/5`, relative taste `> 2/5`. Participates in pass@k separately from basic solve.
- Also reported: `reward` / `invalid_trial` / `verifier_score` / `rubric_score` / `validation_score` / taste field averages when present.

Missing rubric/taste fields fail closed for tasteful solve (counted as not tasteful).

### swe-bench-pro

- Benchmark is defined in [`nemo_skills/dataset/swe-bench-pro/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/swe-bench-pro/__init__.py)
- Original benchmark source is [here](https://labs.scale.com/leaderboard/swe_bench_pro_public).

SWE-bench Pro uses mostly the same logic as regular SWE-bench, so most of the [SWE-bench docs](#swe-bench) apply to it as well. The differences are as follows:

1. Since it is a multilingual benchmark, we use the multilingual inference logic for it, same as for [SWE-bench Multilingual](#swe-bench-multilingual).
2. 88 of 731 instances have to be run in a separate evaluation job with a different Nemo-Skills container based on Alpine Linux. The dockerfile for this container is provided [here](https://github.com/NVIDIA-NeMo/Skills/tree/main/dockerfiles/swe-bench/Dockerfile.nemo-skills.alpine). You can use the `--main_container` option of Nemo-Skills to change the container for this subset of instances, and run the rest as usual. See below for an example.
3. Due to technical issues, OpenHands is not supported for this benchmark. SWE-agent, mini-SWE-agent and OpenCode are supported on the Ubuntu subset. OpenCode needs a glibc Node.js runtime, so it is not expected to work on the Alpine subset.

#### Sample run

Here's how to run a sample evaluation of [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) with SWE-agent on a Slurm cluster.

1. Prepare the data following similar [instructions](#data-preparation) as for SWE-bench. The container formatter format is slightly different, using `{docker_image}` instead of `{instance_id}`. To prepare the data with Dockerhub container URLs, you can simply run

    ```
    ns prepare_data swe-bench-pro
    ```

    If you have local containers downloaded, you can use

    ```
    ns prepare_data swe-bench-pro --container_formatter '/swe-bench-images/{docker_image}.sif'
    ```

    For downloading images, you can use the same `dump_images.py` script as for SWE-bench.

    The data preparation command will create 2 dataset files: `default.alpine.jsonl` for the Alpine-based instances and `default.ubuntu.jsonl` for the Ubuntu-based instances.

2. Run 2 jobs:

    Alpine subset:

    ```
    ns eval \
        --cluster=<CLUSTER_NAME> \
        --main_container=<PATH_TO_ALPINE_NS_CONTAINER> \
        --model=Qwen/Qwen3-Coder-Next \
        --server_type=vllm \
        --server_args="--enable-auto-tool-choice --tool-call-parser qwen3_coder" \
        --server_nodes=1 \
        --server_gpus=8 \
        --benchmarks=swe-bench-pro \
        --output_dir=<OUTPUT_DIR_ALPINE> \
        --split=default.alpine \
        --num_chunks=13 \
        ++agent_framework=swe_agent \
        ++inference.temperature=1.0 \
        ++inference.top_p=0.95 \
        ++inference.top_k=40 \
        ++agent_max_turns=300
    ```

    Ubuntu subset:

    ```
    ns eval \
        --cluster=<CLUSTER_NAME> \
        --model=Qwen/Qwen3-Coder-Next \
        --server_type=vllm \
        --server_args="--enable-auto-tool-choice --tool-call-parser qwen3_coder" \
        --server_nodes=1 \
        --server_gpus=8 \
        --benchmarks=swe-bench-pro \
        --output_dir=<OUTPUT_DIR_UBUNTU> \
        --split=default.ubuntu \
        --num_chunks=2 \
        ++agent_framework=swe_agent \
        ++inference.temperature=1.0 \
        ++inference.top_p=0.95 \
        ++inference.top_k=40 \
        ++agent_max_turns=300
    ```

    replacing <...> with your desired parameters.

After all jobs are complete, you can check the results in `<OUTPUT_DIR_ALPINE>/eval-results/swe-bench-pro/metrics.json` and `<OUTPUT_DIR_UBUNTU>/eval-results/swe-bench-pro/metrics.json`. The combined score on both subsets should be around 40%.

Keep in mind there is some variance between runs, so we recommend running evaluation multiple times and averaging out the resolve rate. To do that automatically, you can set `--benchmarks=swe-bench-pro:N`, where N is your desired number of repeats.

To evaluate the same model with mini-SWE-agent or OpenCode,
replace `swe_agent` with `mini_swe_agent` or `opencode` in the Ubuntu command. OpenHands is not supported for this benchmark. OpenCode is not expected to work on the Alpine subset.

!!! note
    There are some instances where the gold (ground truth) patches do not pass the evaluation tests. Therefore, it's likely that on those instances even patches that resolve the issue will be incorrectly evaluated as "unresolved". We have observed 18 such instances in SWE-bench Pro:
    ```
    instance_ansible__ansible-811093f0225caa4dd33890933150a81c6a6d5226-v1055803c3a812189a1133297f7f5468579283f86, instance_ansible__ansible-942424e10b2095a173dbd78e7128f52f7995849b-v30a923fb5c164d6cd18280c02422f75e611e8fb2, instance_ansible__ansible-de5858f48dc9e1ce9117034e0d7e76806f420ca8-v1055803c3a812189a1133297f7f5468579283f86, instance_ansible__ansible-deb54e4c5b32a346f1f0b0a14f1c713d2cc2e961-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5, instance_ansible__ansible-e9e6001263f51103e96e58ad382660df0f3d0e39-v30a923fb5c164d6cd18280c02422f75e611e8fb2, instance_future-architect__vuls-bff6b7552370b55ff76d474860eead4ab5de785a-v1151a6325649aaf997cd541ebe533b53fddf1b07, instance_NodeBB__NodeBB-00c70ce7b0541cfc94afe567921d7668cdc8f4ac-vnan, instance_NodeBB__NodeBB-087e6020e490b4a1759f38c1ad03869511928263-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e, instance_NodeBB__NodeBB-18c45b44613aecd53e9f60457b9812049ab2998d-v0495b863a912fbff5749c67e860612b91825407c, instance_NodeBB__NodeBB-1ea9481af6125ffd6da0592ed439aa62af0bca11-vd59a5728dfc977f44533186ace531248c2917516, instance_NodeBB__NodeBB-3c85b944e30a0ba8b3ec9e1f441c74f383625a15-v4fbcfae8b15e4ce5d132c408bca69ebb9cf146ed, instance_NodeBB__NodeBB-51d8f3b195bddb13a13ddc0de110722774d9bb1b-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e, instance_NodeBB__NodeBB-76c6e30282906ac664f2c9278fc90999b27b1f48-vd59a5728dfc977f44533186ace531248c2917516, instance_NodeBB__NodeBB-a5afad27e52fd336163063ba40dcadc80233ae10-vd59a5728dfc977f44533186ace531248c2917516, instance_NodeBB__NodeBB-bad15643013ca15affe408b75eba9e47cc604bb2-vd59a5728dfc977f44533186ace531248c2917516, instance_NodeBB__NodeBB-bd80d36e0dcf78cd4360791a82966078b3a07712-v4fbcfae8b15e4ce5d132c408bca69ebb9cf146ed, instance_NodeBB__NodeBB-cfc237c2b79d8c731bbfc6cadf977ed530bfd57a-v0495b863a912fbff5749c67e860612b91825407c, instance_qutebrowser__qutebrowser-305e7c96d5e2fdb3b248b27dfb21042fb2b7e0b8-v2ef375ac784985212b1805e1d0431dc8f1b3c171
    ```
    Depending on your setup, this set of instances may be different.

!!! note
    For evaluation, we use a [custom fork](https://github.com/wasiahmad/SWE-bench_Pro-os) of the SWE-bench Pro repository that supports running evaluation inside of an existing container. It may not always have the latest updates from the upstream repo.

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

See [the "Using data on cluster" documentation](./index.md#using-data-on-cluster) for more information.

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

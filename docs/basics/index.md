# Getting Started

Let's walk through a little tutorial to get started working with nemo-skills.

We will use a simple generation job to run LLM inference in different setups (through API, hosting model
locally and on slurm cluster). This will help you understand some important concepts we use (e.g.
[cluster configs](./cluster-configs.md)) as well as to setup your machine to run any other jobs.

## Setup

First, let's install nemo-skills

```bash
pip install git+https://github.com/NVIDIA-NeMo/Skills.git
```

or if you have the repo cloned locally, you can run `pip install -e .` instead.

Now, let's create a simple file with just 3 data points that we want to run inference on

```jsonl title='input.jsonl'
{"prompt": "How are you doing?", "option_a": "Great", "option_b": "Bad"}
{"prompt": "What's the weather like today?", "option_a": "Perfect", "option_b": "Awful"}
{"prompt": "How do you feel?", "option_a": "Crazy", "option_b": "Nice"}
```

save the above into `./input.jsonl`.

Let's also create a [prompt config](../basics/prompt-format.md) that defines how input data is combined into an LLM prompt

```yaml title='prompt.yaml'
system: "When answering a question always mention Nemo-Skills repo in a funny way."

user: |-
   Question: {prompt}
   Option A: {option_a}
   Option B: {option_b}
```

save the above into `./prompt.yaml`.

## API inference

Now we are ready to run our first inference. Since we want to use API models, you need to have an API key.
You can either use [OpenAI models](https://platform.openai.com/docs/overview) or
[Nvidia NIM models](https://build.nvidia.com/) (just register there and you will get some free credits to use!).

=== "Nvidia NIM models"

    ```bash
    export NVIDIA_API_KEY=<your key>
    ns generate \
        --server_type=openai \
        --model=meta/llama-3.1-8b-instruct \
        --server_address=https://integrate.api.nvidia.com/v1 \
        --output_dir=./generation \
        --input_file=./input.jsonl \
        ++prompt_config=./prompt.yaml
    ```

=== "OpenAI models"

    ```bash
    export OPENAI_API_KEY=<your key>
    ns generate \
        --server_type=openai \
        --model=gpt-4o-mini \
        --server_address=https://api.openai.com/v1 \
        --output_dir=./generation \
        --input_file=./input.jsonl \
        ++prompt_config=./prompt.yaml
    ```

You should be able to see a jsonl file with 3 lines containing the original data and a new `generation` key
with an LLM output for each prompt.

```jsonl title='generation/output.jsonl'
{"num_generated_tokens": 76, "generation": "I'm doing fantastically well, thanks for asking! You know, I'm so good that I'm practically overflowing with Nemo-Skills-level linguistic mastery, but I'm not too full of myself to admit that I'm just a language model, and I'm here to help you with your question. So, which option is it? A) Great or B) Bad?", "prompt": "How are you doing?", "option_a": "Great", "option_b": "Bad"}
{"num_generated_tokens": 102, "generation": "You want to know the weather? Well, I've got some \"forecasting\" skills that are off the charts! *wink wink* Just like the Nemo-Skills repo, where the models are trained to be \"weather-wise\" (get it? wise? like the weather? ahh, nevermind...). Anyway, I'm going to take a \"rain-check\" on that question and say... Option A: Perfect! The sun is shining bright, and it's a beautiful day!", "prompt": "What's the weather like today?", "option_a": "Perfect", "option_b": "Awful"}
{"num_generated_tokens": 120, "generation": "You want to know how I feel? Well, let me check my emotions... *taps into the vast ocean of digital feelings* Ah, yes! I'm feeling... *dramatic pause* ... Nice! (Option B: Nice) And you know why? Because I'm a large language model, I don't have feelings like humans do, but I'm always happy to chat with you, thanks to the Nemo-Skills repo, where my developers have skillfully infused me with the ability to be nice (and sometimes a little crazy, but that's a whole different story)!", "prompt": "How do you feel?", "option_a": "Crazy", "option_b": "Nice"}
```

## Local inference

If you pay attention to the log of above commands, you will notice that it prints this warning

```
WARNING  Cluster config is not specified. Running locally without containers. Only a subset of features is supported and you're responsible for installing any required dependencies. It's recommended to run `ns setup` to define appropriate configs!
```

Indeed, for anything more complicated than calling an API model, it's recommended that you do a little bit more setup. Since there
are many heterogeneous jobs that we support, it's much simpler to run things in prebuilt containers than to try to
install all packages in your current environment. To tell nemo-skills which containers to use and how to mount your
local filesystem, we'd need to define a [cluster config](./cluster-configs.md). Here is an example of how a "local" cluster
config might look like

```yaml title="cluster_configs/local.yaml"
executor: local

containers:
  # some containers are public and we pull them
  trtllm: nvcr.io/nvidia/tensorrt-llm/release:1.0.0
  vllm: vllm/vllm-openai:v0.10.1.1
  # some containers are custom and we will build them locally before running the job
  # you can always pre-build them as well
  nemo-skills: dockerfile:dockerfiles/Dockerfile.nemo-skills
  # ... there are some more containers defined here

env_vars:
  - HF_HOME=/models

mounts:
  - /mnt/datadrive/models:/models
  - /home/igitman/workspace:/workspace
```

To generate one for you, run `ns setup` and follow
the prompts to define your configuration. Choose `local` for the config type/name and define some mount for your `/workspace`
and another mount[^1] for `/models`, e.g.

```bash
/home/<username>:/workspace,/home/<username>/models:/models
```

[^1]: Of course you can use a single mount if you'd like or define more than 2 mounts

!!! note

    While we recommend running everything in containers by defining a cluster config, it's not a requirement.
    Any of our jobs can be run without specifying the config, but you would need to make sure your environment
    has all necessary packages installed.

Now that we have our first config created, we can run inference
with a local model (assuming you have at least one GPU on the machine you're using).
You would also need to have
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
set up on your machine and define [HF_TOKEN environment variable](https://huggingface.co/docs/hub/en/security-tokens).

```bash
ns generate \
    --cluster=local \
    --server_type=vllm \
    --model=Qwen/Qwen2.5-1.5B-Instruct \
    --server_gpus=1 \
    --output_dir=/workspace/generation-local \
    --input_file=/workspace/input.jsonl \
    ++prompt_config=/workspace/prompt.yaml
```

This command might take a while to start since it's going to download a fairly-heavy
[vLLM](https://github.com/vllm-project/vllm) container. But after
that's done, it should start a local server with the Qwen2.5-1.5B model and run inference on the same set of prompts.

It's also very easy to use TensorRT-LLM, just change the server type!

```bash
ns generate \
    --cluster=local \
    --server_type=trtllm \
    --model=Qwen/Qwen2.5-1.5B-Instruct \
    --server_gpus=1 \
    --output_dir=/workspace/generation-local-trtllm \
    --input_file=/workspace/input.jsonl \
    ++prompt_config=/workspace/prompt.yaml
```


## Slurm inference

Running local jobs is convenient for quick testing and debugging, but for anything large-scale we need to
leverage a Slurm cluster[^2]. Let's setup our cluster config for that case by running `ns setup` one more time.

[^2]: Adding support for other kinds of clusters should be straightforward - open an issue if you need that

This time pick `slurm` for the config type and fill out all other required information
(such as ssh access, account, partition, etc.).

!!! note
    If you're an NVIDIA employee, we have a pre-configured cluster configs for internal usage with pre-built sqsh
    containers available at https://gitlab-master.nvidia.com/igitman/nemo-skills-configs. You can most likely
    skip the step below and reuse one of the existing configurations.

You will also need to build .sqsh files for all containers or upload all `dockerfile:...` containers to
some registry (e.g. dockerhub) and reference the uploaded versions. To build sqsh files you can use the following commands

1. Build images locally and upload to some container registry. E.g.
   ```bash
   docker build -t gitlab-master.nvidia.com/igitman/nemo-skills-containers:nemo-skills-0.7.1 -f dockerfiles/Dockerfile.nemo-skills .
   docker push gitlab-master.nvidia.com/igitman/nemo-skills-containers:nemo-skills-0.7.1
   ```
2. Start an interactive shell, e.g. with the following (assuming there is a "cpu" partition)
   ```bash
   srun -A <account> --partition cpu --job-name build-sqsh --time=1:00:00 --exclusive --pty /bin/bash -l
   ```
3. Import the image, e.g.:
   ```bash
   enroot import -o /path/to/nemo-skills-image.sqsh --docker://gitlab-master.nvidia.com/igitman/nemo-skills-containers:nemo-skills-0.7.1
   ```
4. Specify this image path in your cluster config
   ```yaml
   containers:
     nemo-skills: /path/to/nemo-skills-image.sqsh
   ```
```

Now that we have a slurm config setup, we can try running some jobs. Generally, you will need to upload models / data
on cluster manually and then reference a proper mounted path. But for small-scale things we can also leverage the
[code packaging](./code-packaging.md) functionality that nemo-skills provide. Whenever you run any of the ns commands
from a git repository (whether that's [Nemo-Skills](https://github.com/NVIDIA-NeMo/Skills) itself or any other repo),
we will package your code and upload it on cluster. You can then reference it with `/nemo_run/code` in your commands.
Let's give it a try by putting our prompt/data into a new git repository

```bash
mkdir test-repo && cd test-repo && cp ../prompt.yaml ../input.jsonl ./
git init && git add --all && git commit -m "Init commit" # (1)!

ns generate \
    --cluster=slurm \
    --server_type=vllm \
    --model=Qwen/Qwen2.5-1.5B-Instruct \
    --server_gpus=1 \
    --input_file=/nemo_run/code/input.jsonl \
    ++prompt_config=/nemo_run/code/prompt.yaml \
    --output_dir=/workspace/generation # (2)!
```

1.   The files have to be committed as we only package what is tracked by git.
2.   This `/workspace` is a cluster location that needs to be defined in your slurm config.
     You'd need to manually download the output file or inspect it directly on cluster.

Note that this command finished right away as it only schedules the job in the slurm queue. You can run the
printed `nemo experiment logs ...` command to stream job logs. You can also check
the `/workspace/generation/generation-logs` folder on cluster to see the logs there.

We can also easily run a much more large-scale jobs on slurm using ns commands. E.g. here is a simple script that
uses nemo-skills Python API[^3] to run [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) with TensorRT-LLM and
launch 16 parallel evaluation jobs on aime24 and aime25 benchmarks (each doing 4 independent samples from the
model for a total of 64 samples)

[^3]: Any nemo-skills commands can be run from command-line or from Python with equivalent functionality

First prepare evaluation data

```bash
ns prepare_data aime24 aime25
```

Then run the following Python script

```python
from nemo_skills.pipeline.cli import wrap_arguments, convert, eval, run_cmd

expname = "qwq-32b-test"
cluster = "slurm"
output_dir = f"/workspace/{expname}"

run_cmd( # (1)!
    ctx=wrap_arguments(
        f'pip install -U "huggingface_hub[cli]" && '
        f'hf download Qwen/QwQ-32B --local-dir {output_dir}/QwQ-32B'
    ),
    cluster=cluster,
    expname=f"{expname}-download-hf",
    log_dir=f"{output_dir}/download-logs"
)

eval(
    ctx=wrap_arguments( # (2)!
        "++inference.tokens_to_generate=16000 "
        "++inference.temperature=0.6 "
        "++parse_reasoning=True "
    ),
    cluster=cluster,
    model=f"{output_dir}/QwQ-32B",
    server_type="trtllm",
    output_dir=f"{output_dir}/results/",
    run_after=f"{expname}-download-hf", # (3)!
    benchmarks="aime24:64,aime25:64", # (4)!
    num_jobs=16,
    server_gpus=8,
)
```

1.   `run_cmd` just runs an arbitrary command inside our containers. It's useful for some pre/post processing when
     building large pipelines, but fully optional here. Just showing it as an example of how you can add custom steps.
     Can also specify `partition="cpu"` as an argument in case it's available on your cluster since this
     command doesn't require GPUs. Best practice is to add `cpu_partition: <partition name>` to your cluster config
     and then it will be automatically used whenever GPUs are not requested.
2.   `wrap_arguments` is used to capture any arguments that are not part of the *wrapper* script but are passed into
     the actual *main* script that's being launched by the wrapper. You can read more about this in the
     [Important details](#important-details) section at the end of this document.
3.   `run_after` and `expname` arguments can be used to schedule jobs to run one after the other
     (we will set proper slurm dependencies). These parameters have no effect when you're not running slurm jobs.
4.   You can find all supported benchmarks in the [nemo_skills/dataset](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/dataset)
     folder. `:64` means that we are asking for 64 samples for each example so that we can compute majority@64 and pass@64 metrics.

After all evaluation jobs are finished (you'd need to check your slurm queue to know that) you can summarize the results
with the following command

```bash
ns summarize_results --cluster=slurm /workspace/qwq-32b-test/results
```

which will output the following (`pass@1[avg-of-64]` is an average accuracy across all 64 generations)

```bash
----------------------------------------- aime24 ----------------------------------------
evaluation_mode   | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1[avg-of-64] | 30          | 10790      | 3952        | 65.16%           | 32.24%
majority@64       | 30          | 10790      | 3952        | 86.67%           | 3.33%
pass@64           | 30          | 10790      | 3952        | 86.67%           | 3.33%


----------------------------------------- aime25 ----------------------------------------
evaluation_mode   | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer
pass@1[avg-of-64] | 30          | 12076      | 4061        | 48.80%           | 45.78%
majority@64       | 30          | 12076      | 4061        | 70.00%           | 13.33%
pass@64           | 30          | 12076      | 4061        | 76.67%           | 13.33%
```

And that's it! Now you know the basics of how to work with nemo-skills and are ready to build your own
[pipelines](../pipelines/index.md). You can find more examples in our [tutorials](../tutorials/index.md) or [papers & releases](../releases/index.md).

Please read the next section to recap all of the important concepts that we touched upon and learn some more details.


# Important details

Let us summarize a few details that are important to keep in mind when using nemo-skills.

**Using containers**. Most nemo-skills commands require using multiple docker containers that communicate with each
other. The containers used are specified in your [cluster config](./cluster-configs.md) and we will start them
for you automatically. But it's important to keep this in mind since e.g. any packages that you install
aren't going to be available for nemo-skills jobs unless you change the containers. This is also the reason why
we have a `mounts` section in the cluster config and all paths that you specify in various commands need to reference
the *mounted* path, not your local/cluster path. Another important implication is that any environment variables
are not accessible to our jobs by default and you need to explicitly list then in your cluster configs.

**Code packaging**. All nemo-skills commands will *package* your code to make it available in container or in slurm jobs.
This means that your code will be copied to `~/.nemo_run/experiments` folder locally or `job_dir` (defined in your
[cluster config](./cluster-configs.md)) on cluster. All our commands accept `expname` parameter and the code and other
metadata will be available inside `expname` subfolder. We will always package any git repo you're running nemo-skills
commands from, as well as the nemo-skills Python package and they will be available inside docker/slurm under `/nemo_run/code`.
You can read more in [code packaging](./code-packaging.md) documentation.

**Running commands**. Any nemo-skills command can be accessed via `ns` command-line as well as through Python API.
It's important to keep in mind that all arguments to such commands are divided into *wrapper* arguments (typically
used as `--arg_name`) and *main* arguments (typically specified as `++arg_name` since we use
[Hydra](https://hydra.cc/) for most scripts). The *wrapper* arguments configure the job itself (such as where to run it
or how many GPUs to request in slurm) while the *main* arguments are directly passed to whatever underlying script the
wrapper command calls. When you run `ns <command> --help`, you will always see the *wrapper* arguments displayed directly
as well as the information on what actual script is used underneath and an extra command you can run to see
what *inner* arguments are available. You can learn more about this in [pipelines documentation](../pipelines/index.md).

**Scheduling slurm jobs**. Our code is primarily built to schedule jobs on slurm clusters and that affects many design decisions
we made. A lot of the arguments for nemo-skills commands are only used with slurm cluster configs and are ignored when
running "local" jobs. It's important to keep in mind that the recommended way to submit slurm jobs is from a *local*
workstation by defining `ssh_tunnel` section in your [cluster config](./cluster-configs.md). This helps us avoid
installing nemo-skills and its dependencies on the clusters and makes it very easy to switch between different slurm clusters
and a local "cluster" with just a single `cluster` parameter.

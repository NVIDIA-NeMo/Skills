# Inference

Here are the instructions on how to run inference with our repo.

## Download/convert the model

Get the model you want to use. You can use any model that's supported by vLLM, sglang, TensorRT-LLM or Megatron.
You can also use [Nvidia NIM API](https://www.nvidia.com/en-us/ai/) for models that are hosted there.

## Start the server

Start the server hosting your model. Skip this step if you want to use cloud models through an API.

```bash
ns start_server \
    --cluster local \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --server_type vllm \
    --server_gpus 1 \
    --server_nodes 1
```

If the model needs to execute code, add `--with_sandbox`

You could also launch an interactive web chat application by adding `--launch_chat_interface`, for more details see the [Chat Interface documentation](chat_interface.md).

## Send inference requests

Click on :material-plus-circle: symbols in the snippet below to learn more details.


=== "Self-hosted models"

    ```python
    from nemo_skills.inference.model import get_model
    from nemo_skills.prompt.utils import get_prompt
    import asyncio

    llm = get_model(model="meta-llama/Llama-3.1-8B-Instruct", server_type="vllm")  # localhost by default
    prompt_obj = get_prompt('generic/default') # (1)!
    prompt = prompt_obj.fill({'question': "What's 2 + 2?"})
    print(prompt) # (2)!
    output = asyncio.run(llm.generate_async(prompt=prompt))
    print(output["generation"]) # (3)!
    ```

    1.   Here we use [generic/default](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt/config/generic/default.yaml) config.

         See [nemo_skills/prompt/config](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt/config) for more config options
         or [create your own prompts](prompt-format.md)


    2.   This should print

         ```python-console
         >>> print(prompt)
         [{'role': 'user', 'content': "What's 2 + 2?"}]
         ```

         If you don't want to use our prompt class, just create this list yourself

    3.   This should print
         ```python-console
         >>> print(output["generation"])
         2 + 2 = 4.
         ```

=== "API models"

    ```python
    from nemo_skills.inference.model import get_model
    from nemo_skills.prompt.utils import get_prompt
    import asyncio

    llm = get_model( # (1)!
        server_type="openai",  # NIM models are using OpenAI API
        base_url="https://integrate.api.nvidia.com/v1",
        model="meta/llama-3.1-8b-instruct",
    )
    prompt_obj = get_prompt('generic/default') # (2)!

    prompt = prompt_obj.fill({'question': "What's 2 + 2?"})

    print(prompt) # (3)!
    output = asyncio.run(llm.generate_async(prompt=prompt))
    print(output["generation"]) # (4)!
    ```

    1.   Don't forget to define `NVIDIA_API_KEY`.

         To use OpenAI models, use `OPENAI_API_KEY` and set `base_url=https://api.openai.com/v1`.

    2.   Here we use [generic/default](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt/config/generic/default.yaml) config.

         See [nemo_skills/prompt/config](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt/config) for more config options
         or [create your own prompts](prompt-format.md)


    3.   This should print

         ```python-console
         >>> print(prompt)
         [{'role': 'user', 'content': "What's 2 + 2?"}]
         ```

         If you don't want to use our prompt class, just create this list yourself

    4.   This should print
         ```python-console
         >>> print(output["generation"])
         2 + 2 = 4.
         ```

=== "With code execution"

    ``` python
    from nemo_skills.code_execution.sandbox import get_sandbox
    from nemo_skills.inference.model import get_code_execution_model
    from nemo_skills.prompt.utils import get_prompt

    sandbox = get_sandbox()  # localhost by default
    llm = get_code_execution_model(
        model="meta-llama/Llama-3.1-8B-Instruct",
        server_type="vllm",
        sandbox=sandbox,
    )
    system_message = ( # (1)!
        "Environment: ipython\n\n"
        "Use Python to solve this math problem."
    )
    prompt_obj = get_prompt( # (2)!
        'generic/default',
        code_tags='llama3',
        system_message=system_message
     )
    prompt = prompt_obj.fill({'question': "What's 2 + 2?"})
    print(prompt) # (3)!
    output = await llm.generate_async(
        prompt=prompt,
        **prompt.get_code_execution_args() # (4)!
     )
    print(output["generation"]) # (5)!
    ```

    1.   8B model doesn't always follow these instructions, so using 70B or 405B for code execution is recommended.

    2.   Here we use [generic/default](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt/config/generic/default.yaml) config.

         Note how we are updating system message on the previous line (you can also include it in the config directly).

         See [nemo_skills/prompt/config](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt/config) for more config options
         or [create your own prompts](prompt-format.md)

    3.   This should print

         ```python-console
         >>> print(prompt)
         [
            {'role': 'system', 'content': 'Environment: ipython\n\nUse Python to solve this math problem.'},
            {'role': 'user', 'content': "What's 2 + 2?"}
         ]
         ```

         If you don't want to use our prompt class, just create this object yourself

    4.   `prompt.get_code_execution_args()` simply returns a dictionary with start/stop tokens,
         so that we know when to stop LLM generation and how to format the output.

         If you don't want to use our prompt class, just define those parameters directly.

    5.   This should print
         ```python-console
         >>> print(output["generation"])
         <|python_tag|>print(2 + 2)<|eom_id|><|start_header_id|>ipython<|end_header_id|>

         completed
         [stdout]
         4
         [/stdout]<|eot_id|><|start_header_id|>assistant<|end_header_id|>

         The answer is 4.
         ```

         The "4" in the stdout is coming directly from Python interpreter running in the sandbox.

If you want to use completions api, you can also provide `tokenizer` parameter to `get_prompt` and it will use
tokenizer's chat template to format messages and return you a string.

You can learn more about how our prompt formatting works in the [prompt format docs](../basics/prompt-format.md).

!!! note

    You can also use slurm config when launching a server. If you do that, add `host=<slurm node hostname>`
    to the `get_model/sandbox` calls and define `NEMO_SKILLS_SSH_KEY_PATH` and `NEMO_SKILLS_SSH_SERVER` env vars
    to set the connection through ssh.
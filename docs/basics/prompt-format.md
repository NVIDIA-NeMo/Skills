# Prompt utilities

Our prompts are configured via two yaml files:

1. **Prompt config** - contains the actual prompt text with placeholders
2. **Code tags** - specifies code formatting tokens, required for code execution


## Prompt config

The prompt config contains user and system messages with placeholders for keys from a data file.
The configs are model independent (any model can be used with any config).
All of the configs that we support by default are available in
[nemo_skills/prompt/config](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/prompt/config)
folder. Here is an example prompt for
[math evaluations](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/prompt/config/generic/math.yaml):

```yaml
# default prompt for all math benchmarks (e.g. gsm8k, math)

few_shot_examples:
  prefix: "Here are some examples of problems and solutions you can refer to.\n\n"
  template: "Problem:\n{problem}\n\nSolution:\n{solution}\n\n\n\n\n\n"
  suffix: "Here is the problem you need to solve:\n"
  # this is built as <prefix>{template.format(example1)}{template.format(example2)}...{template.format(exampleN)}<suffix>
  # and available as {examples} key in the final prompt
  # if examples_type is not specified, then {examples} will be empty
  # by default there are no examples, but can be changed from code/cmd

# optional system message
# system: ""

user: |-
  Solve the following math problem. Make sure to put the answer (and only answer) inside \boxed{{}}.

  {examples}{problem}
```

Note that we use `{problem}`, `{solution}` and `{examples}` format strings here. The `{examples}` is a special
key that will be used to include few shot examples you specify above (it's empty unless you add `++examples_type` or
specify it in the config).
All other keys will need to be specified when you call `prompt.fill`
(more on that in the [prompt-api section](#prompt-api)) so that we can replace placeholders with actual input.

The input for few shot examples always comes from one of the available example types in
[here](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/prompt/few_shot_examples/__init__.py).


## Code tags

Code tags define the special tokens that models use to mark executable code blocks and their output. Code tags are required when using code execution.
All code tags that we support by default are available in
[nemo_skills/prompt/code_tags](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/prompt/code_tags).

Here is an example code tags file for the [llama3](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/prompt/code_tags/llama3.yaml) family:

```yaml
# Code tags for llama3 family models

# used to execute code within these tags
code_begin: "<|python_tag|>"
code_end: "<|eom_id|>"

# used to extract the code output
code_output_begin: "<|start_header_id|>ipython<|end_header_id|>"
code_output_end: "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

# how to post-process the captured output (choices: llama, qwen)
code_output_format: "llama"
```

## Prompt API

If you're running one of the pipeline scripts, you can control the prompt by using:

```bash
++prompt_config=...
++code_tags=...
++examples_type=...
```

#### Example 1 - Prompt formatted as a message list

If you're implementing a new script, you can use the following code to create a prompt and then use it:

```python
from nemo_skills.prompt.utils import get_prompt

prompt = get_prompt('generic/math')
print(prompt.fill({'problem': "What's 2 + 2?"}))
```

which outputs

```python-console
[
  {
    'role': 'user',
    'content': "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.\n\nWhat's 2 + 2?"
  }
]
```

!!!note
    If your data is already formatted as a list of openai messages, you can directly use it as an input to the pipeline scripts
    if you set `++prompt_format=openai`.


#### Example 2 - Prompt formatted as a string

If you want to use completions API, you can set `++inference.endpoint_type=text`. This will use model's tokenizer to format
messages as a string (you can specify a custom tokenizer with `++tokenizer=...` argument).

Here is an example of the input to completions api

```python
from nemo_skills.prompt.utils import get_prompt

# code_tags parameter is optional and only needed for code execution
prompt = get_prompt('generic/math', tokenizer='Qwen/Qwen2.5-32B-Instruct')
# By default, the prompt.fill method returns a list of messages.
# Here we override that by passing "format_as_string=True"
print(prompt.fill({'problem': "What's 2 + 2?"}, format_as_string=True))
```

which outputs

```python-console
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Solve the following math problem. Make sure to put the answer (and only answer) inside \boxed{}.

What's 2 + 2?<|im_end|>
<|im_start|>assistant
```



#### Example 3 - Overriding the System Message

In the above example, the system message comes from the tokenizer. We can override this system message by setting `++system_message=...`. Here is an example of overriding the system message while creating a prompt

```python
from nemo_skills.prompt.utils import get_prompt

prompt = get_prompt(
  "generic/math",
  tokenizer="Qwen/Qwen2.5-32B-Instruct",
  system_message="You are a helpful chatbot"
)
print(prompt.fill({'problem': "What's 2 + 2?"}, format_as_string=True))
```

which outputs

```python-console
<|im_start|>system
You are a helpful chatbot<|im_end|>
<|im_start|>user
Solve the following math problem. Make sure to put the answer (and only answer) inside \boxed{}.

What's 2 + 2?<|im_end|>
<|im_start|>assistant
```

#### Example 4 - Formatting the Assistant Response (Non-Reasoning)

For SFT, we need to format the output as well. Here we show an example of how the prompt API formats the response for a non-reasoning model.

```python
from nemo_skills.prompt.utils import get_prompt

prompt = get_prompt(
  "generic/math",
  tokenizer="Qwen/Qwen2.5-32B-Instruct",
)

print(prompt.format_assistant_response(content="The answer is 4"))
```

which outputs

```python-console
The answer is 4<|im_end|>
```

This example illustrates how the prompt API is used under the hood in [the SFT data preparation script](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/training/prepare_data.py).


#### Example 5 - Formatting the Assistant Response (Reasoning)

For a reasoning model, the assistant response has the additional `thinking` field.
Below we illustrate how to pass in this extra field

```python
from nemo_skills.prompt.utils import get_prompt

prompt = get_prompt(
  "generic/math",
  tokenizer="openai/gpt-oss-120b",
  system_message="You are a helpful chatbot"
)

print(
  prompt.format_assistant_response(
    content="The answer is 4",
    thinking="Let me think step by step.... The answer is 4"
  )
)
```

which outputs

```python-console
<|channel|>analysis<|message|>Let me think step by step.... The answer is 4<|end|><|start|>assistant<|channel|>final<|message|>The answer is 4<|return|>
```

!!!note
    The `thinking` part is <span style="color: red;">**NOT**</span> automatically added to the assistant response by certain tokenizers like `Qwen/Qwen3-4B`.
    We suggest adding the thinking part to the `content` field as part of preprocessing in such cases.


You can also have a look at the [tests](https://github.com/NVIDIA-NeMo/Skills/tree/main/tests/test_prompts.py) to see more examples of using our prompt API.

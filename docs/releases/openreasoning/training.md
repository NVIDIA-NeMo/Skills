# Model training

## Download data and convert to SFT format

OpenReasoning dataset consists of 5 independent parts:

* Math CoT data
* Math TIR data
* Math GenSelect data
* Code CoT data
* Science CoT data

All datasets except GenSelect are now released. You can use code snippets below to download them and prepare for SFT.
For final training dataset, you should concatenate all of the data together.

### Math CoT data

Math CoT data is released as part of the [nvidia/Nemotron-Post-Training-Dataset-v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) dataset.

```python
from functools import partial
from datasets import load_dataset
from nemo_skills.prompt.utils import get_prompt

def apply_format(elem, prompt):
    assert len(elem['messages']) == 2
    elem['input'] = prompt.fill({'problem': elem['messages'][0]['content']}, format_as_string=True)
    elem['output'] = prompt.format_assistant_response(elem['messages'][1]['content'])
    return elem

dataset = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v1", split="math")

prompt = get_prompt('generic/math', tokenizer='Qwen/Qwen2.5-32B-Instruct', system_message="")
func = partial(apply_format, prompt=prompt)
dataset = dataset.map(func, num_proc=20)
dataset = dataset.remove_columns(['messages'])

dataset.to_json("open-reasoning-math-cot.jsonl")
```

### Math TIR data

We re-use math TIR data from [nvidia/OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) dataset.
While we included this data in training and our released models are capable of TIR inference, we found that results are
generally worse than using CoT. To fix this, TIR data would need to be re-generated using newer models, but this is not
done in our current release.

To get this data, follow instructions for the **second-round** SFT data in [OpenMathReasoning documentation](../openmathreasoning/training.md#second-round-sft).

### Math GenSelect data

We have not released this data yes. Please open an issue if you need it.

### Code CoT data

Code CoT data is released as part of the [nvidia/Nemotron-Post-Training-Dataset-v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) dataset.

```python
import json
from functools import partial
from datasets import load_dataset
from nemo_skills.prompt.utils import get_prompt

question_datasets = {
    "taco": load_dataset("BAAI/TACO"),
    "apps": load_dataset("codeparrot/apps"),
    "code_contests": load_dataset("deepmind/code_contests"),
    "open-r1/codeforces": load_dataset("open-r1/codeforces")
}


def get_question(ds_name, split, index):
    benchmark = question_datasets[ds_name][split][int(index)]
    if ds_name == "code_contests":
        return benchmark["description"]
    elif ds_name in ["taco", "apps"]:
        return benchmark["question"]
    elif ds_name == "open-r1/codeforces":
        question = benchmark["description"]
        if benchmark["input_format"]:
            question += "\n\nInput\n\n" + benchmark["input_format"]
        if benchmark["output_format"]:
            question += "\n\nOutput\n\n" + benchmark["output_format"]
        if benchmark["examples"]:
            question += "\n\nExamples"
            for example in benchmark["examples"]:
                if "input" in example:
                    question += "\n\nInput\n\n" + example["input"]
                if "output" in example:
                    question += "\n\nOutput\n\n" + example["output"]
        if benchmark["note"]:
            question += "\n\nNote\n\n" + benchmark["note"]
        return question
    else:
        raise RuntimeError("Something wrong with the data!")


def apply_format(elem, prompt):
    metadata = json.loads(elem['metadata'])
    question = get_question(metadata['dataset'], metadata['split'], int(metadata['index']))

    elem['input'] = prompt.fill({'question': question}, format_as_string=True)
    elem['output'] = prompt.format_assistant_response(elem['messages'][1]['content'])
    return elem

dataset = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v1", split="code")

prompt = get_prompt('eval/livecodebench/default_reasoning', tokenizer='Qwen/Qwen2.5-32B-Instruct', system_message="")
func = partial(apply_format, prompt=prompt)
dataset = dataset.map(func, num_proc=20)
dataset = dataset.remove_columns(['messages'])

dataset.to_json("open-reasoning-code-cot.jsonl")
```

### Science CoT data

Science CoT data is released as [nvidia/OpenScienceReasoning-2](https://huggingface.co/datasets/nvidia/OpenScienceReasoning-2) dataset.

```python
from functools import partial
from datasets import load_dataset
from nemo_skills.prompt.utils import get_prompt

def apply_format(elem, prompt):
    elem['input'] = prompt.fill({'question': elem['input']}, format_as_string=True)
    elem['output'] = prompt.format_assistant_response(elem['output'])
    return elem

dataset = load_dataset("nvidia/OpenScienceReasoning-2", split="train")

prompt = get_prompt('generic/default', tokenizer='Qwen/Qwen2.5-32B-Instruct', system_message="")  # data already includes instruction
func = partial(apply_format, prompt=prompt)
dataset = dataset.map(func, num_proc=20)

dataset.to_json("open-reasoning-science-cot.jsonl")
```


## Train the models

We mostly use the same training commands as for [OpenMathReasoning models](../openmathreasoning/training.md#run-training). The only difference
is that we pack sequences to 49152 length and use a little different hyperparameters detailed in the following table.
Note that unlike OpenMathReasoning, we are not starting from *Math* models, but are using standard base models for all model sizes.

|                  | **lr** | **min_lr** | **TP** | **PP** | **CP** |
| ---------------- | ------ | ---------- | ------ | ------ | ------ |
| **Qwen2.5-1.5B** | 1e-4   | 1e-7       | 1      | 1      | 4      |
| **Qwen2.5-7B**   | 1e-4   | 1e-7       | 4      | 1      | 4      |
| **Qwen2.5-14B**  | 1e-4   | 1e-7       | 8      | 1      | 4      |
| **Qwen2.5-32B**  | 1e-4   | 1e-7       | 8      | 2      | 4      |

All models are trained for 30000 steps with a single round of SFT and we take the last checkpoint as the final model.
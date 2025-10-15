---
date: 2025-08-28
readtime: 20
---

# Building an Efficient Inference Engine for Solving Math Problems

This tutorial guides you through creating a high-performance inference engine using [Nemo-Skills](https://nvidia-nemo.github.io/Skills/) to tackle complex math problems. It demonstrates the inference pipeline used to win the [AIMO-2 competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/nemoskills-1st-place-solution-nemoskills). With FP8 quantization and ReDrafter speculative decoding, we demonstrate up to 4× faster batched inference compared to BF16 on two H100 GPUs.

We will leverage [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) for optimized model serving, including an advanced technique called ReDrafter for speculative decoding.

By the end of this tutorial and companion notebook, you will have a local setup capable of running efficient inference with a large language model (LLM) integrated with a code execution sandbox.

<!-- more -->

## What We'll Cover

1.  **Setting up Your Environment**: Get your system ready by installing necessary libraries within a suitable container.
2.  **Preparing Model Weights**: Download a pre-trained OpenMath model and convert it into an optimized TensorRT-LLM engine using FP8 quantization.
3.  **Accelerating Inference with ReDrafter**: Discover ReDrafter, a speculative decoding technique, train a draft model, and integrate it into our TensorRT-LLM engine for faster generation.
4.  **Launching the Inference Server**: Set up the LLM server and a parallel code execution sandbox to handle the tool-use capabilities of our model.
5.  **Running Inference**: Finally, we'll send math problems to our custom inference engine and observe its problem-solving abilities.

See the [companion notebook](https://github.com/NVIDIA-NeMo/Skills/tree/main/docs/tutorials/notebooks/demo_aimo_inference.ipynb) for launching the inference server and benchmarking.

-----

## 1\. Setting Up Your Environment

Our first step is to establish a consistent and isolated environment. We will use a NVIDIA PyTorch NGC container and install the essential libraries: TensorRT-LLM for model optimization and Nemo-Skills for the overall pipeline management.
FP8 inference requires a GPU that supports FP8 inference such as Ada Lovelace or Hopper architecture or later. For this example we assume two GPUs are available.

### Container Setup and Library Installation

Once inside the `nvcr.io/nvidia/pytorch:25.05-py3` container, run the following commands to install TensorRT-LLM and Nemo-Skills:

```bash
# Ensure no conflicting TensorRT installations and install TensorRT-LLM
[ -f /etc/pip/constraint.txt ] && : > /etc/pip/constraint.txt
pip uninstall -y tensorrt
pip3 install tensorrt_llm==1.1.0rc0

# Install Nemo-Skills
pip install git+https://github.com/NVIDIA-NeMo/Skills.git
```

-----

## 2\. Preparing Model Weights

Now that our environment is ready, the next step is to prepare our Large Language Model (LLM). We'll download the `nvidia/OpenMath-Nemotron-14B-Kaggle` model and transform it into an optimized TensorRT-LLM engine using FP8 quantization.

**Note on FP8 Quantization:** FP8 (8-bit floating point) quantization is highly efficient but requires GPUs that support `E4M3 FP8` (like NVIDIA Hopper GPUs). For other GPUs, `int8_wo` (8-bit integer with weight-only quantization) is recommended and does not require calibration.

### Downloading Model Weights and Dataset

Generate a Hugging Face token and export it as an environment variable, then use the Hugging Face CLI to download the necessary models and datasets.

```bash
# Export your Hugging Face token
export HF_TOKEN=hf_YOUR_HUGGING_FACE_TOKEN # Replace with your actual token

# Install Hugging Face CLI
pip install -U "huggingface_hub[cli]"

# Download the 14B parameter main model
hf download nvidia/OpenMath-Nemotron-14B-kaggle --local-dir OpenMath-Nemotron-14B-kaggle

# Download the OpenMathReasoning dataset for calibration
hf download nvidia/OpenMathReasoning --repo-type dataset --local-dir OpenMathReasoning
```

### Preparing the Calibration Dataset for FP8 Quantization

For FP8 quantization, a small calibration dataset representative of inference data is essential. We'll use a subset of the `OpenMathReasoning` dataset to create it. Save the following as `prepare_calibration_data.py`:

```python title="prepare_calibration_data.py"
import os
from itertools import islice

from datasets import Dataset, load_dataset

from nemo_skills.prompt.utils import get_prompt

# Define paths and parameters
LOCAL_DATASET_PATH = './calibration_dataset'
CALIB_DATASET_NAME = "nvidia/OpenMathReasoning"
CALIB_SPLIT = 'tir'
CALIB_SIZE = 4096

# Load samples, format them, and save as a Parquet file
print(f"Loading and formatting {CALIB_SIZE} samples for calibration...")
ds_samples = load_dataset(CALIB_DATASET_NAME, split=CALIB_SPLIT, streaming=True)
ds_samples = list(islice(ds_samples, CALIB_SIZE))

prompt_template = get_prompt('generic/math', tokenizer='nvidia/OpenMath-Nemotron-14B-kaggle')
calibration_dataset = Dataset.from_dict(
    {
        "text": [
            prompt_template.format_assistant_response(
                prompt_template.fill(
                    {k: v for k, v in sample.items() if k in ['problem', 'generated_solution']},
                    start_assistant_response_key='generated_solution',
                )
            )
            for sample in ds_samples
        ]
    }
)

os.makedirs(LOCAL_DATASET_PATH, exist_ok=True)
calibration_dataset.to_parquet(f"{LOCAL_DATASET_PATH}/data.parquet")
print(f"Calibration dataset saved to {LOCAL_DATASET_PATH}/data.parquet")
```

Run the script:

```bash
python prepare_calibration_data.py
```

### Converting and Quantizing to TensorRT-LLM Engine

Now, convert the Hugging Face model to a TensorRT-LLM engine, applying FP8 quantization and using the prepared calibration dataset. This step generates the FP8 quantized LLM inference engine.

```bash
ns convert \
    --input_model OpenMath-Nemotron-14B-kaggle \
    --output_model OpenMath-Nemotron-14B-kaggle-fp8-trtllm \
    --convert_from hf \
    --convert_to trtllm \
    --num_gpus 2 \
    --dtype fp8 \
    --hf_model_name nvidia/OpenMath-Nemotron-14B-kaggle \
    --model_type qwen \
    --max_input_len 30000 \
    --max_seq_len 32000 \
    --no-trt_reuse_tmp_engine \
    --calib_dataset ./calibration_dataset
```

After this command, your FP8 LLM engine is ready for deployment.

-----

## 3\. Accelerating Inference with ReDrafter

To push our inference efficiency further, we will integrate [ReDrafter](https://machinelearning.apple.com/research/redrafter-nvidia-tensorrt-llm). This speculative decoding technique uses a smaller "draft" model to predict tokens, allowing the main LLM to generate responses much faster. ReDrafter is an RNN-based inference method developed by Apple. The ReDrafter [implementation](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/redrafter) is compatible with most models supported within the TensorRT-LLM library.

### Installing and Training ReDrafter

First, install the ReDrafter library. The tokenizer and training data for the draft model should be the same as those used for the base model. If the original training data is not available, base model generations can also be used for training the draft model.

```bash
# Install the ReDrafter library
pip install --no-binary=protobuf --ignore-requires-python \
        "git+https://github.com/apple/ml-recurrent-drafter.git#egg=recurrent-drafting[dev,train]"

# Train the ReDrafter model
ns run_cmd --log_dir ./logs/ \
torchrun --nproc_per_node=2 -m nemo_skills.training.train_redrafter \
    --llm_name_or_path 'OpenMath-Nemotron-14B-kaggle' \
    --dataset "OpenMathReasoning" \
    --dataset_split "tir" \
    --bf16 True \
    --output_dir "redrafter_output" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --save_strategy "no" \
    --learning_rate 0.001 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 20 \
    --tf32 True \
    --model_max_length 2048 \
    --dataset_nrows 50000 \
    --drafter_predict_n_tokens 3 \
    --drafter_num_layers 2 \
    --rnn True \
    --phase train \
    --report_to wandb # Remove if not using wandb
```

During training, observe the `redrafter2_top1` score. Aiming for above `0.6` indicates good performance (60% of steps accept the next three drafted tokens).

### Building the TensorRT-LLM Engine for the Draft Model

Now, we'll convert our trained ReDrafter model into a TensorRT-LLM checkpoint and then combine it with our main LLM to create the final, accelerated TensorRT-LLM engine.

First, clone the TensorRT-LLM repository to access its conversion scripts:

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM/
```

Next, convert the trained ReDrafter PyTorch checkpoint to a TensorRT-LLM checkpoint.

```bash
# Base model intermediate checkpoint from FP8 quantisation step
export BASE_TRTLLM_CKPT=$(pwd)/OpenMath-Nemotron-14B-kaggle-fp8-trtllm-tmp-ckpt
# Trained draft checkpoint
export REDRAFTER_PYTORCH_CKPT=$(pwd)/redrafter_output/redrafter__redrafter_OpenMath-Nemotron-14B-kaggle_n_3_lr_0.001_layers_2
export REDRAFTER_TRTLLM_CKPT=$(pwd)/OpenMath-Nemotron-14B-kaggle-fp8-draft-ckpt

cd ./TensorRT-LLM/examples/redrafter
python convert_checkpoint.py \
    --base_model_checkpoint_dir $BASE_TRTLLM_CKPT \
    --drafter_model_dir $REDRAFTER_PYTORCH_CKPT \
    --output_dir $REDRAFTER_TRTLLM_CKPT \
    --dtype bfloat16 \
    --tp_size 2 \
    --redrafter_num_beams 1 \
    --redrafter_draft_len_per_beam 3
cd ../../../
```

Finally, build the combined TensorRT-LLM engine - base model with a draft head for speculative decoding.
```bash
trtllm-build \
    --checkpoint_dir $REDRAFTER_TRTLLM_CKPT \
    --output_dir OpenMath-Nemotron-14B-kaggle-fp8-redrafter-trtllm \
    --gemm_plugin fp8 \
    --use_paged_context_fmha=enable \
    --max_batch_size 32 \
    --max_seq_len 32000 \
    --max_input_len 32000 \
    --max_num_tokens 32000 \
    --speculative_decoding_mode explicit_draft_tokens \
    --max_beam_width 1 \
    --kv_cache_type paged
```

Your TensorRT-LLM engine, now supercharged with ReDrafter, is ready to be served!

-----

## 4\. Benchmarking and results

We’ve prepared a [companion notebook](https://github.com/NVIDIA-NeMo/Skills/tree/main/docs/tutorials/notebooks/demo_aimo_inference.ipynb) where you can try out the full pipeline yourself. The notebook was run with the same container setup and installations as section 1 above, along with 2 H100 GPUs for inference.
In the notebook, you can:

- Run inference on different TensorRT-LLM engines (BF16, FP8, FP8+ReDrafter).
- Compare performance benchmarks.
- Explore advanced controls like **terminating after the first N generations complete**.
- Run inference with tool-calling.

Here’s a sample of the kind of benchmark results you’ll see:

| Metric                            | BF16  | FP8  | FP8+ReDrafter |
| --------------------------------- | ----- | ---- | ------------- |
| Total Generation Time (s)         | 144.2 | 64.7 | 30.5          |
| Average Sample Throughput (Tok/s) | 34.6  | 75.2 | 138.5         |

*(full benchmarks and code available in the notebook)*

-----

## 5\. Optional: Enabling tool-calling and the code execution sandbox

The OpenMath LLM is a powerful model that can use Python as a tool integrated in its reasoning. This means it doesn't just generate text; it can also write and execute Python code in a sandbox to solve problems. In the companion notebook, we provide an example how to launch both the LLM server and its accompanying code execution sandbox.

The interaction works like this:

1.  The LLM generates Python code wrapped in `<tool_call>` and `</tool_call>` tokens.
2.  The inference engine extracts and sends this code to the sandbox.
3.  The sandbox executes the code and returns the results.
4.  The output is fed back to the LLM for continued generation or to finalize its answer.

Here's an example of such an interaction:
<details markdown><summary><b>Show Example Output</b></summary>

Therefore, b = k - 7 = 21 or 49. So, same result. Therefore, sum is 70.

Alternatively, maybe I can write a small program to check for all bases b > 9, compute 9b + 7 and b + 7, check if the latter divides the former, and collect all such bases. Then sum them. Let's do that to verify.

Here's a Python code to perform the check:

```
    <tool_call>
    # Initialize a list to store valid bases
    valid_bases = []

    # Check bases from 10 upwards
    for b in range(10, 10000):  # Arbitrary large upper limit
        num1 = 9 * b + 7
        num2 = b + 7
        if num1 % num2 == 0:
            valid_bases.append(b)
            print(f"Found base: {b}")

    # Sum the valid bases
    sum_bases = sum(valid_bases)
    print(f"Sum: {sum_bases}")

    # If sum is over 1000, take modulo 1000
    if sum_bases > 1000:
        result = sum_bases % 1000
    else:
        result = sum_bases

    print(f"Final Result: {result}")
    </tool_call>
    ```output
    Found base: 21
    Found base: 49
    Sum: 70
    Final Result: 70
    ```
```
The code confirms that the valid bases are 21 and 49, summing to 70.

</details>

To turn off tool-calling in the [companion notebook](https://github.com/NVIDIA-NeMo/Skills/tree/main/docs/tutorials/notebooks/demo_aimo_inference.ipynb) use `get_model` instead of `get_code_execution_model` as shown in the Nemo-Skills [docs](https://nvidia-nemo.github.io/Skills/).

# Tool-calling

## Supported benchmarks

## bfcl_v3

BFCL v3 consists of seventeen distinct evaluation subsets that comprehensively test various aspects of function calling capabilities, from simple function calls to complex multi-turn interactions.

- Benchmark is defined in [`nemo_skills/dataset/bfcl_v3/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/bfcl_v3/__init__.py)
- Original benchmark source is [here](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard).

### Data Preparation

To prepare BFCL v3 data for evaluation:

```bash
ns prepare_data bfcl_v3
```

This command performs the following operations:

- Downloads the complete set of BFCL v3 evaluation files
- Processes and organizes data into seventeen separate subset folders
- Creates standardized test files in JSONL format

**Example output structure**:
```
nemo_skills/dataset/bfcl_v3/
├── simple/test.jsonl
├── parallel/test.jsonl
├── multiple/test.jsonl
└── ... (other subsets)
```

### Challenges of tool-calling tasks

There are three key steps in tool-calling which differentiate it from typical text-only tasks:

1. **Tool Presentation**: Presenting the available tools to the LLM
2. **Response Parsing**: Extracting and validating tool calls from model-generated text
3. **Tool Execution**: Executing the tool calls and communicating the results back to the model

For 1 and 3, we borrow the implementation choices from the [the BFCL repo](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard). For tool call parsing, we support both client side and server side implementations.


#### Client-Side Parsing (Default)

**When to use**: Standard models supported by the BFCL repository

**How it works**: Utilizes the parsing logic from [BFCL's local inference handlers](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/bfcl_eval/model_handler/local_inference)

**Configuration Requirements**:
- Model name specification via `++model_name=<model_id>`

**Sample Command**:

```bash hl_lines="9"
ns eval \
  --benchmarks bfcl_v3 \
  --cluster dfw \
  --model /hf_models/Qwen3-4B \
  --server_gpus 2 \
  --server_type vllm \
  --output_dir /workspace/qwen3-4b-client-parsing/ \
  ++parse_reasoning=True \
  ++inference.tokens_to_generate=8192 \
  ++model_name=Qwen/Qwen3-4B-FC \
```

#### Server-Side Parsing

**When to use**:
- Models not supported by BFCL client-side parsing
- Custom tool-calling formats

**Configuration Requirements**:
- Set `++use_client_parsing=False` and
- Specify appropriate server arguments. For example, evaluating Qwen models with vllm server would require setting the server_args as follows:
```bash
--server_args="--enable-auto-tool-choice --tool-call-parser hermes"
```

**Sample Command**:

The following command evaluates the `Qwen3-4B` model which uses a standard tool-calling format supported by vllm

```bash hl_lines="9-10"
ns eval \
  --benchmarks bfcl_v3 \
  --cluster dfw \
  --model Qwen/Qwen3-4B \
  --server_gpus 2 \
  --server_type vllm \
  --output_dir /workspace/qwen3-4b-server-parsing/ \
  ++parse_reasoning=True \
  ++inference.tokens_to_generate=8192 \
  ++use_client_parsing=False \
  --server_args="--enable-auto-tool-choice --tool-call-parser hermes"
```


**Custom parsing example (NVIDIA Llama-3.3-Nemotron-Super-49B-v1.5)**

Some models implement bespoke tool-calling formats that require specialized parsing logic. For example, the [Llama-3.3-Nemotron-Super-49B-v1.5](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5) model implements its tool calling logic which requires passing the model-specific parsing script, [llama_nemotron_toolcall_parser_no_streaming.py](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5/blob/main/llama_nemotron_toolcall_parser_no_streaming.py), to the server.


```bash hl_lines="12-16"
ns eval \
    --cluster=local \
    --benchmarks=bfcl_v3 \
    --model=/workspace/Llama-3_3-Nemotron-Super-49B-v1_5/ \
    --server_gpus=2 \
    --server_type=vllm \
    --output_dir=/workspace/llama_nemotron_49b_1_5_tool_calling/ \
    ++parse_reasoning=True \
    ++inference.tokens_to_generate=65536 \
    ++inference.temperature=0.6 \
    ++inference.top_p=0.95 \
    ++system_message='' \
    ++use_client_parsing=False \
    --server_args="--tool-parser-plugin \"/workspace/Llama-3_3-Nemotron-Super-49B-v1_5/llama_nemotron_toolcall_parser_no_streaming.py\" \
                    --tool-call-parser \"llama_nemotron_json\" \
                    --enable-auto-tool-choice"
```

### Configuration Parameters

| Configuration          | True                        | False                            |
| ---------------------- | --------------------------- | -------------------------------- |
| `++use_client_parsing` | Default                     | -                                |
| `++model_name`         | Required for client parsing | -                                |
| `--server_args`        | -                           | Required for server-side parsing |



!!!note
    To evaluate individual splits of `bfcl_v3`, such as `simple`, use `benchmarks=bfcl_v3.simple`.

!!!note
    Currently, ns summarize_results does not support benchmarks with custom aggregation requirements like BFCL v3. To handle this, the evaluation pipeline automatically launches a dependent job that processes the individual subset scores using [our scoring script](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/bfcl_v3/bfcl_score.py).
# Checkpoint conversion

!!! info

    This pipeline starting script is [nemo_skills/pipeline/convert.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/pipeline/convert.py)

    All extra parameters are passed to one of the following scripts

    * For conversion to NeMo:
        - If `--model_type=llama`: [nemo_skills/conversion/hf_to_nemo_llama.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/conversion/hf_to_nemo_llama.py)
        - If `--model_type=qwen`: [nemo_skills/conversion/hf_to_nemo_qwen.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/conversion/hf_to_nemo_qwen.py)

    * For conversion to HuggingFace:
        - If `--model_type=llama`: [nemo_skills/conversion/nemo_to_hf_llama.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/conversion/nemo_to_hf_llama.py)
        - If `--model_type=qwen`: [nemo_skills/conversion/nemo_to_hf_qwen.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/conversion/nemo_to_hf_qwen.py)


You only need to convert models if you want to use NeMo-Aligner for training. All other server and training backends
should work with HuggingFace format.

To convert the checkpoint from one format to another use a command like this

```bash
ns convert \
    --cluster=slurm \
    --input_model=/hf_models/Meta-Llama-3.1-70B-Instruct \
    --output_model=/nemo_models/llama3.1-70b-instruct \
    --convert_from=hf \
    --convert_to=nemo \
    --model_type=llama \
    --num_gpus=8 \
    --hf_model_name=meta-llama/Meta-Llama-3.1-70B-Instruct
```

You can provide any extra arguments that will be passed directly to the underlying conversion scripts.


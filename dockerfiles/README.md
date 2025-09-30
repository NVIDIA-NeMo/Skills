# How to build all necessary dockerfiles

Some dockerfiles are directly included in this folder and for some others the instructions to build them are below.
To build one of the existing dockerfiles use a command like this

```
docker build -t igitman/nemo-skills:0.7.1 -f dockerfiles/Dockerfile.nemo-skills .
```
It might take a long time for some of the images.

## Building for arm64/aarch64

To build for arm64 architecture (e.g. to use with GB200 machines) first follow the installation process at
https://docs.docker.com/build/building/multi-platform/#install-qemu-manually

Then run the same docker command but adding `--platform linux/arm64`.

## Building trtllm image

We directly use official `nvcr.io/nvidia/tensorrt-llm/release:1.0.0` image for both amd64 and arm64.

## Building sglang image

We directly use official `lmsysorg/sglang:v0.5.3rc1-cu126` image.

For arm64 we instead use `lmsysorg/sglang:blackwell-cu129-arm64` image.

## Building vllm image

We directly use official `vllm/vllm-openai:v0.10.1.1` image.

For arm64 we instead use `vllm/vllm-openai:v0.10.2` image.

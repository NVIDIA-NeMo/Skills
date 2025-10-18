# Building Docker Images

Some dockerfiles are directly included in this folder and for some others the instructions to build them are below.

The dockerfiles can be built using the standard docker build command. e.g.,
```shell
docker build -t nemo-skills-image:0.7.1 -f dockerfiles/Dockerfile.nemo-skills .
```

In addition, we provide a utility script which provides sane build defaults
```shell
./build.sh Dockerfile.nemo-skills
```

Key configuration environment variables for `build.sh`:
- `DOCKER_NAME`: A fully qualified name of the docker image. The default is inferred from the git repository attributes.
- `DOCKER_TAG`: Docker tag to use. Defaults to `yyyy.mm.dd-<commit_hash>`
- `DOCKER_PUSH`: When set, pushes image after building.
- `DOCKER_PLATFORM`: Directly passed to `--platform` for [multi-platform builds](https://docs.docker.com/build/building/multi-platform/).

## Building for arm64/aarch64

To build for arm64 architecture (e.g. to use with GB200 machines) first follow the installation process at
https://docs.docker.com/build/building/multi-platform/#install-qemu-manually

Then run the same docker command but adding `--platform linux/arm64` or
set `DOCKER_PLATFORM=linux/arm64` for the build script described above.

## Building trtllm image

We directly use official `nvcr.io/nvidia/tensorrt-llm/release:1.0.0` image for both amd64 and arm64.

## Building sglang image

We directly use official `lmsysorg/sglang:v0.5.3rc1-cu126` image.

For arm64 we instead use `lmsysorg/sglang:blackwell-cu129-arm64` image.

## Building vllm image

We directly use official `vllm/vllm-openai:v0.10.1.1` image.

For arm64 we instead use `vllm/vllm-openai:v0.10.2` image.

# Starting a Model Server

!!! info

    This pipeline starting script is [nemo_skills/pipeline/start_server.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/pipeline/start_server.py)

This pipeline provides a convenient way to run model server as a separate process that can then be utilized via its server address.

To run a vLLM model available at the path `/workspace/models/Qwen3-4B-Thinking-2507` (as mounted in the cluster configuration),
```bash
ns start_server \
    --cluster=local \
    --server_type=vllm \
    --model=/workspace/models/Qwen3-4B-Thinking-2507 \
    --server_gpus=1 \
    --log_dir=/workspace/logs/start_server \
```

Now, the model server is available at `localhost:5000` by default. Note that you can also use `Qwen/Qwen3-4B-Thinking-2507` directly as the model name if you don't have it pre-downloaded.

To launch a sandbox server in parallel, provide the `--with_sandbox` argument, which will be
available at `localhost:6000` by default.

## Remote Server

It is also possible to remotely start the server on a slurm cluster and access it locally.
This approach is especially useful for quick debugging or when your local workstation does not have all the required compute resources available.

Consider a vLLM model at path `/workspace/models/Qwen3-235B-A22B-Thinking-2507` that requires 16 GPUs.
We start the server with `--create_tunnel` as,
```bash
ns start_server \
    --cluster=<cluster_name> \
    --server_type=vllm \
    --model=/workspace/models/Qwen3-235B-A22B-Thinking-2507 \
    --server_gpus=8 \
    --server_nodes=2 \
    --log_dir=/workspace/logs/start_server \
    --create_tunnel
```

Once the server is launched, it is available at `localhost:5000` by default.
Similarly, in case `--with_sandbox` is set, it is available at `localhost:6000` by default.

!!! tip

    Pressing `ctrl + c` twice will terminate all tunnels and shutdown the launched slurm job as well.

### Remote and Tunnel Ports

The local port for the model server can be changed using the `--server_tunnel_port` argument. For instance,
setting,
```bash
ns start_server ... --server_tunnel_port=9999
```
will make the model server available at `localhost:9999`.

Similarly, the local port for the sandbox server can be changed using `--sandbox_tunnel_port` argument.

!!! tip

    To avoid port conflicts on the remote hosts in case of partial node jobs (i.e. requesting fewer GPUs than total available on the node) where other jobs may be running, use `--get_random_port` to randomly assign ports to launched server. Using this argument does not change the default tunnel ports exposed locally, and the server will still be
    accessible at `localhost:5000` by default (and sandbox at `localhost:6000` by default).

## Using the Server

To use this started server in [Evaluation](evaluation.md) or [Generation](generation.md),
all the model-related arguments can now be replaced with `--server_type=openai` and `server_address` arguments.

For instance, for the vLLM model server above, the `eval` pipeline arguments can be modified as,
```bash
ns eval ... \
    --model=/workspace/models/Qwen3-4B-Thinking-2507 \
    --server_type=openai \
    --server_address=http://localhost:5000/v1
```

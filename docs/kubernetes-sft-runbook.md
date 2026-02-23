# Kubernetes SFT Runbook

## Session Summary

This document captures the current state of multi-node multi-GPU SFT support on Kubernetes and provides commands to manually run validation tests.

### What Was Built

| Feature | Files | Status |
|---------|-------|--------|
| Multi-node K8s backend | `backends/kubernetes.py`, `backends/base.py` | Indexed Job + Headless Service |
| SFT pipeline K8s routing | `pipeline/nemo_rl/sft.py` | `_run_sft_kubernetes()` |
| Distributed env vars | `backends/kubernetes.py` | MASTER_ADDR, MASTER_PORT, WORLD_SIZE, NODE_RANK |
| RDMA/IB resources | `backends/kubernetes.py` | Opt-in via `rdma` config |
| DNS init container | `backends/kubernetes.py` | Default enabled for multi-node |
| Pod anti-affinity | `backends/kubernetes.py` | Spreads pods across nodes |
| Kaniko build tooling | `scripts/k8s-tests/image-build/` | Build + cache + distribute |
| E2E validation | `scripts/k8s-tests/validate_sft_e2e.sh` | 5 checkpoints |
| NCCL log checker | `scripts/k8s-tests/check_nccl_logs.py` | Parses NCCL_DEBUG=INFO |
| Backend-aware timeout | `pipeline/utils/cluster.py` | `6h`/`30m` K8s-style supported |
| Unit tests | `tests/test_backends.py` | 133 passed, 1 skipped |

### Container Image

- **Image**: `docker.io/nemo-skills/nemo-rl:latest`
- **Base**: `nvcr.io/nvidia/pytorch:25.03-py3` (configurable via `BASE_IMAGE` env var)
- **Contents**: nemo_skills 0.7.0 + KubernetesBackend + all pipeline backends
- **Cached tarball**: `~/nemo-skills/nemo-rl.tar` (default `build-and-load.sh` cache path)

### Cluster Notes

> **Important**: Node names, GPU counts, and driver versions are cluster-specific.
> The scripts and YAML manifests do NOT hardcode node names. Adjust the examples
> below for your cluster. For multi-node tests, ensure all target nodes have the
> **same number of GPUs** to avoid `torchrun` world-size mismatches.

**Example cluster**:
- **gpu-node-1**: 8 GPUs — verified for single-node and multi-node
- **gpu-node-2**: 8 GPUs — verified for single-node and multi-node
- **gpu-node-3**: 7 GPUs — **exclude from multi-node tests** (heterogeneous GPU count)

Check your CUDA driver version to pick a compatible PyTorch container:
```bash
# Run on a GPU node to check driver version
kubectl run driver-check --image=nvcr.io/nvidia/pytorch:25.03-py3 --restart=Never \
    --overrides='{"spec":{"tolerations":[{"key":"nvidia.com/gpu","operator":"Exists","effect":"NoSchedule"}]}}' \
    -- nvidia-smi --query-gpu=driver_version --format=csv,noheader
sleep 15 && kubectl logs driver-check && kubectl delete pod driver-check --ignore-not-found
```
Then check compatibility at: https://docs.nvidia.com/deeplearning/frameworks/support-matrix/

**For multi-node on heterogeneous clusters**: add a `nodeAffinity` to your Job spec
to restrict scheduling to nodes with matching GPU counts:
```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: kubernetes.io/hostname
          operator: In
          values: ["your-node-1", "your-node-2"]
```

---

## Manual Run Commands (Start-to-Finish Order)

Follow these steps in order. All commands assume:
```bash
cd ~/nemo-skills/Skills
```

### 1. Build/Rebuild Container Image (do this first)

Build from local code and distribute to GPU nodes. The script:
- Copies local repo to a build node via `kubectl cp`
- Runs Kaniko to build the image as a tarball
- Caches the tarball at `~/nemo-skills/nemo-rl.tar`
- Copies the tarball to each target node and imports into containerd

```bash
# List your GPU nodes first
kubectl get nodes -l nvidia.com/gpu.present=true -o custom-columns=NAME:.metadata.name,GPUS:.status.capacity.nvidia\\.com/gpu

# Build and load onto specific nodes (replace with your node names)
./scripts/k8s-tests/image-build/build-and-load.sh --nodes "node-1 node-2"

# Or load onto ALL GPU nodes (default)
./scripts/k8s-tests/image-build/build-and-load.sh
```

To use a different base image:
```bash
BASE_IMAGE=nvcr.io/nvidia/pytorch:25.03-py3 ./scripts/k8s-tests/image-build/build-and-load.sh --nodes "node-1 node-2"
```

> **Notes**:
> - `--nodes` requires a **quoted** string (e.g., `"node-1 node-2"`)
> - Build takes ~5 minutes, tarball copy + import takes ~15-20 minutes per node
> - If the script has already run, it will skip the build and use the cached tarball
> - To force a rebuild, delete the cache: `rm ~/nemo-skills/nemo-rl.tar`
> - For multi-node tests, target nodes with the **same GPU count**

### 2. Unit Tests

```bash
.venv/bin/python -m pytest tests/test_backends.py -s -q --tb=short
# Expected: 133 passed, 1 skipped
```

### 3. Single-Node Multi-GPU E2E Validation

```bash
./scripts/k8s-tests/validate_sft_e2e.sh --gpus 2
```

With more GPUs or a custom image:
```bash
./scripts/k8s-tests/validate_sft_e2e.sh --gpus 4
./scripts/k8s-tests/validate_sft_e2e.sh --gpus 2 --image docker.io/nemo-skills/nemo-rl:latest
```

Or apply the YAML manifest directly:
```bash
kubectl apply -f scripts/k8s-tests/manifests/single-node-sft-test.yaml
kubectl wait --for=condition=complete --timeout=600s job/sft-single-node -n default
kubectl logs -l job-name=sft-single-node -n default
kubectl delete -f scripts/k8s-tests/manifests/single-node-sft-test.yaml
```

**NeMo-Skills code paths validated (5 checkpoints):**

| Checkpoint | NeMo-Skills Code Path | What It Proves |
|------------|----------------------|----------------|
| 1. Imports | `nemo_skills.pipeline.utils.declarative.Pipeline` | Declarative pipeline API loadable |
| 1. Imports | `nemo_skills.pipeline.backends.kubernetes.KubernetesBackend` | K8s backend code present |
| 1. Imports | `nemo_skills.pipeline.backends.base.JobSpec` | Multi-node JobSpec with `num_nodes` field |
| 1. Imports | `nemo_skills.inference.generate.InferenceConfig` | Inference pipeline loadable |
| 2. CLI routing | `sft_nemo_rl(executor="kubernetes", dry_run=True)` | CLI detects K8s executor and routes to `_run_sft_kubernetes()` |
| 2. CLI routing | `Pipeline` constructed with 2 jobs | Training job (GPU) + conversion job (CPU-only) with dependency |
| 3. NCCL | `dist.init_process_group(backend="nccl")` | GPU communication initialized |
| 4. DDP | `DistributedDataParallel(model)` | Gradient sync across GPUs via NCCL |
| 5. All-reduce | `dist.all_reduce(tensor)` | Collective communication verified |

**Pass signal**: `E2E SFT VALIDATION: PASSED`

### 4. Multi-Node Multi-GPU Validation

```bash
./scripts/k8s-tests/validate_sft_e2e_multinode.sh
```

With custom options:
```bash
./scripts/k8s-tests/validate_sft_e2e_multinode.sh --nodes 2 --gpus 4
./scripts/k8s-tests/validate_sft_e2e_multinode.sh --image custom:tag
```

Or apply the YAML manifest directly:
```bash
kubectl apply -f scripts/k8s-tests/manifests/multi-node-sft-test.yaml
kubectl wait --for=condition=complete --timeout=900s job/sft-multi-node -n default
kubectl get pods -l job-name=sft-multi-node -o custom-columns=POD:.metadata.name,NODE:.spec.nodeName
kubectl logs -l job-name=sft-multi-node -n default
kubectl delete -f scripts/k8s-tests/manifests/multi-node-sft-test.yaml
```

**What the script does:**

1. Creates a **Headless Service** for DNS-based pod discovery (same pattern as `KubernetesBackend._build_headless_service()` in `nemo_skills/pipeline/backends/kubernetes.py`)
2. Creates an **Indexed Job** with `completionMode: Indexed` (same pattern as `KubernetesBackend._build_job_manifest()` when `num_nodes > 1`)
3. Sets distributed env vars (`MASTER_ADDR`, `MASTER_PORT`, `NODE_RANK` from `JOB_COMPLETION_INDEX`) — mirrors `KubernetesBackend._inject_distributed_env_vars()`
4. Applies **podAntiAffinity** to force pods onto separate nodes — mirrors `KubernetesBackend._build_pod_anti_affinity()`
5. Uses **podAntiAffinity** to force one pod per node (add **nodeAffinity** if your cluster has heterogeneous GPU counts)

**NeMo-Skills code paths validated inside each container:**

| Import / Call | NeMo-Skills Module | What It Proves |
|---------------|-------------------|----------------|
| `import nemo_skills` | `nemo_skills/__init__.py` | Package installed and loadable |
| `from nemo_skills.pipeline.backends.kubernetes import KubernetesBackend` | `backends/kubernetes.py` | Multi-node backend code is present |
| `dist.init_process_group(backend="nccl")` | PyTorch + NCCL | GPU communication works across nodes |
| `DDP(model, device_ids=[local_rank])` | PyTorch DDP | Distributed training works |
| `dist.all_reduce(tensor)` | NCCL | Cross-node collective verified (sum of ranks) |

**Pass criteria** (all checked automatically):
- `MULTI-NODE PASSED` in rank-0 logs
- `nemo_skills` import confirmed
- `KubernetesBackend: OK` confirmed
- Training loss is not NaN
- Pods ran on 2+ distinct nodes (hard-fail otherwise)

**Pass signal**: `MULTI-NODE SFT VALIDATION: PASSED`

### 5. DDP-Only Smoke Tests (no NeMo-Skills dependencies)

For quick NCCL verification without needing the nemo-rl image:

```bash
# Single-node (uses generic PyTorch image)
PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:25.03-py3 ./scripts/k8s-tests/smoke_test_single_node.sh --gpus 2

# Multi-node
PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:25.03-py3 ./scripts/k8s-tests/smoke_test_multi_node.sh --nodes 2 --gpus 2
```

---

## Troubleshooting

### `kubectl cp` produces 0-byte files
`kubectl cp` internally wraps file transfers in a tar stream and relies on the destination container having a compatible `tar` binary to extract it. Alpine's busybox `tar` does not support the tar flags that `kubectl cp` uses, so the extraction silently produces a 0-byte file — even though `kubectl cp` reports exit code 0. The fix is to use `debian:bookworm-slim` for loader pods, which has GNU `tar`. The `build-and-load.sh` script uses debian by default.

Additionally, failed previous attempts can leave stale 0-byte `/raid/nemo-rl.tar` files on nodes. The script now deletes any existing file before copying to prevent `kubectl cp` from failing to overwrite.

### Image not found / `ImagePullBackOff`
The image is loaded locally into containerd, not pulled from a registry. Ensure:
- `imagePullPolicy: Never` is set on the container spec
- The image was imported on the specific node where the pod is scheduled
- Check with: `kubectl describe pod <name>` → Events section

### CUDA driver too old
`pytorch:26.01-py3` requires CUDA 13.1 but DGX nodes have driver 12080 (CUDA 12.8). Use `pytorch:25.03-py3` or older. Check compatibility at: https://docs.nvidia.com/deeplearning/frameworks/support-matrix/

### Stale 0-byte tarballs on nodes
Previous failed attempts can leave empty `/raid/nemo-rl.tar` files. The `build-and-load.sh` script cleans these automatically before copying, but if running manually: `kubectl exec <pod> -- rm -f /raid/nemo-rl.tar`

### Heterogeneous GPU counts across nodes
If nodes have different GPU counts, multi-node `torchrun` jobs will fail with world-size mismatches. Either target only homogeneous nodes (via `--nodes` flag or `nodeAffinity`) or set `--nproc_per_node` to the minimum GPU count across all nodes.

---

## Cluster Details (Example: Pleiades)

> These details are specific to the Pleiades cluster used during development.
> Adjust for your environment.

- **K8s**: v1.32+ with containerd runtime
- **GPU Operator**: NVIDIA device plugin installed (`nvidia.com/gpu` resource available)
- **`ctr` path**: varies by distro (check `find / -name ctr -type f 2>/dev/null`)
- **Local storage**: `build-and-load.sh` uses `/raid` as scratch on each node (configurable)

To find your cluster's details:
```bash
kubectl get nodes -l nvidia.com/gpu.present=true -o custom-columns=NAME:.metadata.name,GPUS:.status.capacity.nvidia\\.com/gpu
```

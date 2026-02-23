# NeMo-RL SFT on Kubernetes — Full Stack Runbook

This runbook covers running **real NeMo-RL SFT training** on Kubernetes using `start_sft.py`, the actual NeMo-RL training entry point with Ray, DTensor/FSDP, and sequence packing.

> **See also**: [kubernetes-sft-runbook.md](./kubernetes-sft-runbook.md) for the K8s backend infrastructure tests (Pipeline, KubernetesBackend, NCCL validation).

---

## What This Proves

Unlike the infrastructure tests that validate imports and DDP, this runbook exercises the **full NeMo-RL training stack**:

| Component | What Runs | NeMo-RL Code Path |
|-----------|----------|-------------------|
| Training entry point | `start_sft.py` | `nemo_skills/training/nemo_rl/start_sft.py` |
| Config system | Hydra + sft.yaml | `nemo_skills/training/nemo_rl/configs/sft.yaml` |
| Data loading | PromptResponseDataset | `start_sft.py:PromptResponseDataset` |
| Tokenization | Auto-detected chat template | `start_sft.py:sft_preprocessor()` |
| Sequence packing | modified_first_fit_decreasing | `start_sft.py:setup_data()` |
| Distributed backend | Ray + DTensor FSDP | NeMo-RL `DTensorPolicyWorkerV2` |
| Model | Qwen/Qwen2.5-0.5B (500M) | Downloaded from HuggingFace |
| Multi-node | Ray head + worker nodes | Ray cluster via Headless Service DNS |
| Tensor Parallelism | TP=2 within each node | `++policy.tensor_model_parallel_size=2` |

### Validated Results

| Test | Status | Evidence |
|------|--------|---------|
| Single-node (2 GPU) | **PASSED** | `start_sft.py` completed, training steps executed |
| Multi-node (2 nodes x 2 GPU) | **PASSED** | Ray 2-node cluster, NCCL cross-node, training completed |

---

## Prerequisites

### 1. Container Image
Build the full NeMo-RL image (~25GB, includes Ray, vLLM, TransformerEngine, Megatron-Core):

```bash
# The Kaniko-compatible Dockerfile is at:
# scripts/k8s-tests/image-build/Dockerfile.nemo-rl-k8s
#
# Build and distribute using the standard flow:
# (see kubernetes-sft-runbook.md for the full Kaniko build process)
#
# Cached tarball location:
ls ~/nemo-skills/images/nemo-rl-full.tar
```

### 2. Training Data
```bash
cd ~/nemo-skills/Skills
python scripts/k8s-tests/generate_sft_data.py 50 > /tmp/train.jsonl
kubectl delete configmap sft-train-data -n default --ignore-not-found
kubectl create configmap sft-train-data -n default --from-file=train.jsonl=/tmp/train.jsonl
```

### 3. Image on Nodes
Ensure `nemo-skills/nemo-rl-full:latest` is loaded in containerd on all target GPU nodes. Use the `build-and-load.sh` flow or manual `kubectl cp` + `ctr import`.

---

## Single-Node Real SFT (SNMG)

Runs `start_sft.py` on a single node with 2 GPUs using DTensor/FSDP.

```bash
kubectl apply -f scripts/k8s-tests/manifests/real-nemo-rl-sft-test.yaml
kubectl wait --for=condition=complete --timeout=1800s job/sft-nemo-rl-real -n default
kubectl logs job/sft-nemo-rl-real -n default | grep -E "▶|step|loss|COMPLETED|Qwen"
kubectl delete -f scripts/k8s-tests/manifests/real-nemo-rl-sft-test.yaml
```

**What to look for:**
```
▶ Setting up data...
▶ Setting up compute cluster...     ← Ray initializes
▶ Setting up model...               ← Qwen2.5-0.5B loads via DTensorPolicyWorkerV2
▶ Preparing batch...                ← Sequence packing (modified_first_fit_decreasing)
▶ Taking a training step...         ← FSDP training with NCCL P2P/CUMEM
  • Total step time: X.XXs
Max number of steps has been reached, stopping training early
=== REAL NEMO-RL SFT COMPLETED ===
```

**Pass signal**: `REAL NEMO-RL SFT COMPLETED`

**NeMo-RL code exercised**:
- `start_sft.py` → `main()` → `setup_data()` → `setup()` → `sft_train()`
- `PromptResponseDataset` loads and tokenizes synthetic math Q&A data
- `sft_preprocessor()` handles sequence packing with `modified_first_fit_decreasing`
- Ray launches `DTensorPolicyWorkerV2` actors for each GPU
- FSDP distributes model parameters across GPUs
- Training loop runs with gradient sync via NCCL

---

## Multi-Node Real SFT (MNMG)

Runs `start_sft.py` across 2 nodes with tensor parallelism (TP=2) within each node.

```bash
kubectl apply -f scripts/k8s-tests/manifests/real-nemo-rl-sft-multinode-test.yaml
kubectl wait --for=condition=complete --timeout=1800s job/sft-nemo-rl-mn -n default

# Verify pods ran on different nodes
kubectl get pods -l job-name=sft-nemo-rl-mn -o custom-columns=POD:.metadata.name,STATUS:.status.phase,NODE:.spec.nodeName

# Check rank-0 logs
kubectl logs $(kubectl get pods -l job-name=sft-nemo-rl-mn -o jsonpath='{.items[0].metadata.name}') -n default | grep -E "▶|step|COMPLETED|Ray cluster|nodes"

kubectl delete -f scripts/k8s-tests/manifests/real-nemo-rl-sft-multinode-test.yaml
```

**Architecture:**
```
Node 0 (Ray head, dgx-30):          Node 1 (Ray worker, dgx-29):
┌──────────────────────────┐         ┌──────────────────────────┐
│  Ray Head + start_sft.py │         │  Ray Worker (--block)    │
│  ┌────────┬────────┐     │  NCCL   │  ┌────────┬────────┐    │
│  │ GPU 0  │ GPU 1  │     │◄─Socket─►│  │ GPU 0  │ GPU 1  │    │
│  │  TP=2 (tensor   │     │ inter-  │  │  TP=2 (tensor   │    │
│  │   parallel)     │     │ node    │  │   parallel)     │    │
│  └────────┴────────┘     │         │  └────────┴────────┘    │
│    P2P/CUMEM (NVLink)    │         │    P2P/CUMEM (NVLink)   │
└──────────────────────────┘         └──────────────────────────┘
```

**Key K8s patterns used (same as KubernetesBackend)**:
- **Headless Service**: DNS-based pod discovery for Ray cluster formation
- **Indexed Job**: `JOB_COMPLETION_INDEX` determines head (0) vs worker (1+)
- **hostNetwork**: Required for Ray multi-node (arbitrary port communication)
- **podAntiAffinity**: Forces one pod per physical node
- **emptyDir Memory**: 16Gi `/dev/shm` for NCCL shared memory segments

**What to look for:**
```
✓ Ray cluster initialized with 2 nodes       ← Both nodes in Ray cluster
Initializing lm_policy workers: 4 workers     ← Workers on both nodes
▶ Taking a training step...
  • Total step time: X.XXs                    ← Training running
  • policy_training: X.XXs (99.9%)
Max number of steps has been reached
=== REAL NEMO-RL MNMG SFT COMPLETED ===
```

**Pass signal**: `REAL NEMO-RL MNMG SFT COMPLETED`

---

## Troubleshooting

### `/dev/shm` too small (NCCL error)
NCCL requires shared memory for inter-process communication. K8s defaults to 64MB which is insufficient for multi-GPU training. Both YAML manifests include a 16Gi `emptyDir` with `medium: Memory` mounted at `/dev/shm`.

Error: `Error while creating shared memory segment /dev/shm/nccl-... No space left on device`

### GPT-2 doesn't work with DTensor
GPT-2 uses `Conv1D` layers which DTensor/FSDP can't shard. Error: `Missing key in checkpoint state_dict: transformer.h.0.attn.c_attn.bias`. Use a modern architecture like `Qwen/Qwen2.5-0.5B` instead.

### Ray worker exits immediately
Use `ray start --block` (not just `ray start`) to keep the worker process alive. Without `--block`, `ray start` launches the daemon in background and the script exits, killing the pod.

### Ray worker exits with non-zero on shutdown
When the head pod finishes and Ray shuts down, the worker's `ray start --block` exits with a non-zero code. The manifest uses `|| true` to suppress this so the job shows Complete 2/2.

### Read-only filesystem for data cache
`start_sft.py` creates a cache directory next to the training data file. ConfigMap mounts are read-only. Fix: copy data to `/tmp/sft-data/` (writable) before training.

### `accelerate` missing
If using HuggingFace Trainer (not NeMo-RL), the container may need `pip install accelerate`. The NeMo-RL path doesn't need this — it uses its own DTensor backend.

### CUDA Forward Compatibility
The `cuda-dl-base:25.05` image uses CUDA 12.9 but runs on driver 570 (CUDA 12.8) via forward compatibility mode. This is expected and safe.

### Ray can't connect between pods
Multi-node Ray requires unrestricted network between pods. If using K8s pod networking (not hostNetwork), Ray may fail to communicate on its randomly-assigned ports. Fix: use `hostNetwork: true` with `dnsPolicy: ClusterFirstWithHostNet`.

---

## Files Reference

| File | Purpose |
|------|---------|
| `scripts/k8s-tests/image-build/Dockerfile.nemo-rl-k8s` | Kaniko-compatible Dockerfile for full NeMo-RL image |
| `scripts/k8s-tests/manifests/real-nemo-rl-sft-test.yaml` | Single-node real SFT (ConfigMap + Job) |
| `scripts/k8s-tests/manifests/real-nemo-rl-sft-multinode-test.yaml` | Multi-node real SFT (ConfigMap + Service + Indexed Job) |
| `scripts/k8s-tests/generate_sft_data.py` | Synthetic training data generator |
| `nemo_skills/training/nemo_rl/start_sft.py` | NeMo-RL SFT entry point |
| `nemo_skills/training/nemo_rl/configs/sft.yaml` | Default SFT Hydra config |

---

## Differences from Infrastructure Tests

| Aspect | Infrastructure Tests (`kubernetes-sft-runbook.md`) | Real NeMo-RL SFT (this doc) |
|--------|---------------------------------------------------|----------------------------|
| Training | Raw PyTorch DDP | NeMo-RL `start_sft.py` with Ray |
| Model | Random Linear / GPT-2 via HF Trainer | Qwen2.5-0.5B via DTensor FSDP |
| Data | Synthetic inline | `PromptResponseDataset` with sequence packing |
| Orchestration | Pipeline + KubernetesBackend | K8s Job directly (proves container works) |
| Multi-node | torchrun | Ray cluster (head + worker) |
| Image | `nemo-skills/nemo-rl:latest` (12GB) | `nemo-skills/nemo-rl-full:latest` (25GB) |
| TP/PP | N/A | TP=2 supported |

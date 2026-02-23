# NeMo-Skills on Kubernetes: Complete Guide

This guide walks through running NeMo-Skills on Kubernetes from scratch. By the end, you'll have a working setup for running LLM inference, evaluation, and training jobs on K8s.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Cluster Setup](#cluster-setup)
4. [Configuration](#configuration)
5. [Running Jobs](#running-jobs)
6. [Monitoring & Debugging](#monitoring--debugging)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Kubernetes cluster (1.24+) with GPU nodes
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/) installed
- `kubectl` configured to access your cluster
- NeMo-Skills installed: `pip install nemo-skills`

Verify GPU operator is working:

```bash
# Check GPU nodes are labeled
kubectl get nodes -l nvidia.com/gpu.present=true

# Check GPU resources are available
kubectl describe node <gpu-node> | grep nvidia.com/gpu
```

---

## Quick Start

If you're impatient, here's the minimum to get running:

```bash
# 1. Apply RBAC and storage manifests
kubectl apply -f cluster_configs/kubernetes/rbac.yaml
kubectl apply -f cluster_configs/kubernetes/storage.yaml

# 2. Create your cluster config
cat > cluster_configs/my-k8s.yaml << 'EOF'
executor: kubernetes
namespace: nemo-skills
containers:
  vllm: vllm/vllm-openai:latest
  nemo-skills: nvcr.io/nvidia/nemo:24.07
service_account: nemo-skills-sa
storage:
  models:
    pvc_name: nemo-models-pvc
    mount_path: /models
  results:
    pvc_name: nemo-results-pvc
    mount_path: /results
env_vars:
  - HF_HOME=/models/hf-cache
skip_hf_home_check: true
EOF

# 3. Run a job
ns generate \
  --cluster my-k8s \
  --model Qwen/Qwen2.5-Math-7B \
  --server-type vllm \
  --server-gpus 1 \
  --benchmarks gsm8k \
  --output-dir /results/quick-test
```

---

## Cluster Setup

### Step 1: Create Namespace

```bash
kubectl create namespace nemo-skills
```

### Step 2: Set Up RBAC

NeMo-Skills needs permissions to create and manage Jobs. Apply the RBAC configuration:

```bash
kubectl apply -f cluster_configs/kubernetes/rbac.yaml
```

Or create manually:

```yaml
# rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: nemo-skills-sa
  namespace: nemo-skills
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: nemo-skills-job-manager
  namespace: nemo-skills
rules:
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["create", "delete", "get", "list", "watch", "patch"]
  - apiGroups: [""]
    resources: ["pods", "pods/log"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["services"]
    verbs: ["create", "delete", "get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: nemo-skills-binding
  namespace: nemo-skills
subjects:
  - kind: ServiceAccount
    name: nemo-skills-sa
roleRef:
  kind: Role
  name: nemo-skills-job-manager
  apiGroup: rbac.authorization.k8s.io
```

`services` permissions are required for multi-node jobs because NeMo-Skills
creates/deletes a Headless Service per distributed job.

### Step 3: Create Storage (PVCs)

NeMo-Skills needs persistent storage for models, data, and results:

```bash
kubectl apply -f cluster_configs/kubernetes/storage.yaml
```

Or create manually (adjust `storageClassName` for your cluster):

```yaml
# storage.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nemo-models-pvc
  namespace: nemo-skills
spec:
  accessModes:
    - ReadWriteMany  # Required for multi-pod access
  resources:
    requests:
      storage: 500Gi
  storageClassName: your-storage-class  # e.g., nfs, efs, azurefile
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nemo-results-pvc
  namespace: nemo-skills
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: your-storage-class
```

### Step 4: Create Image Pull Secret (for NGC)

If using NVIDIA NGC images:

```bash
kubectl create secret docker-registry nvcr-secret \
  --namespace=nemo-skills \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=YOUR_NGC_API_KEY
```

### Step 5: Upload Models to Storage

Option A: Use a temporary pod to copy models:

```bash
# Start a pod with the PVC mounted
kubectl run model-loader -n nemo-skills \
  --image=ubuntu:latest \
  --restart=Never \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "loader",
        "image": "ubuntu:latest",
        "command": ["sleep", "infinity"],
        "volumeMounts": [{"name": "models", "mountPath": "/models"}]
      }],
      "volumes": [{
        "name": "models",
        "persistentVolumeClaim": {"claimName": "nemo-models-pvc"}
      }]
    }
  }'

# Copy models (from your local machine or another source)
kubectl cp /path/to/models nemo-skills/model-loader:/models/

# Clean up
kubectl delete pod model-loader -n nemo-skills
```

Option B: Use HuggingFace Hub (models download automatically):

```bash
# Models are downloaded to HF_HOME on first use
# Just ensure HF_HOME is on a mounted PVC
```

---

## Configuration

### Cluster Configuration File

Create `cluster_configs/my-kubernetes.yaml`:

```yaml
# Executor type - tells NeMo-Skills to use Kubernetes
executor: kubernetes

# Kubernetes namespace for all jobs
namespace: nemo-skills

# Optional: path to kubeconfig (uses default if omitted)
# kubeconfig: ~/.kube/config

# Container images - map logical names to actual images
containers:
  vllm: vllm/vllm-openai:v0.6.0
  nemo-skills: nvcr.io/nvidia/nemo:24.07
  sglang: lmsysorg/sglang:latest
  trtllm: nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3
  sandbox: python:3.11-slim

# Image pull secrets for private registries
image_pull_secrets:
  - nvcr-secret

# Service account with RBAC permissions
service_account: nemo-skills-sa

# Resource pools - map partition names to node selectors
# Use the same partition names as your Slurm config for easy migration
resource_pools:
  gpu-a100:
    node_selector:
      nvidia.com/gpu.product: NVIDIA-A100-SXM4-80GB
    tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule

  gpu-h100:
    node_selector:
      nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
    tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule

  cpu:
    node_selector:
      node-type: cpu-worker

# Storage - PVCs mounted to all containers
storage:
  models:
    pvc_name: nemo-models-pvc
    mount_path: /models
  data:
    pvc_name: nemo-data-pvc
    mount_path: /data
  results:
    pvc_name: nemo-results-pvc
    mount_path: /results

# Job timeout (default for all jobs)
default_timeout: "6h"

# Environment variables for all containers
env_vars:
  - HF_HOME=/models/hf-cache
  - TOKENIZERS_PARALLELISM=false

# Skip HF_HOME validation (useful when setting up)
skip_hf_home_check: false
```

### Environment Variables

Important environment variables:

| Variable | Purpose | Example |
|----------|---------|---------|
| `HF_HOME` | HuggingFace cache directory | `/models/hf-cache` |
| `HF_TOKEN` | HuggingFace API token (for gated models) | `hf_...` |
| `WANDB_API_KEY` | Weights & Biases logging | `...` |
| `OPENAI_API_KEY` | For API-based models | `sk-...` |

---

## Running Jobs

### Basic Generation

```bash
# Generate on a benchmark
ns generate \
  --cluster my-kubernetes \
  --model Qwen/Qwen2.5-Math-7B \
  --server-type vllm \
  --server-gpus 1 \
  --benchmarks gsm8k \
  --output-dir /results/qwen-gsm8k
```

### Evaluation with Multiple Models

```bash
# Compare two models on multiple benchmarks
ns eval \
  --cluster my-kubernetes \
  --model Qwen/Qwen2.5-Math-7B Qwen/Qwen2.5-Math-72B \
  --server-type vllm vllm \
  --server-gpus 1 4 \
  --benchmarks gsm8k,math500 \
  --output-dir /results/model-comparison
```

### Custom Data

```bash
# Run on your own data
ns generate \
  --cluster my-kubernetes \
  --model /models/my-fine-tuned-llama \
  --server-type vllm \
  --server-gpus 2 \
  --input-file /data/my-problems.jsonl \
  --output-dir /results/custom-eval
```

### With Code Execution (Sandbox)

```bash
# Enable sandbox for code generation benchmarks
ns eval \
  --cluster my-kubernetes \
  --model Qwen/Qwen2.5-Coder-7B \
  --server-type vllm \
  --server-gpus 1 \
  --benchmarks human-eval \
  --with-sandbox \
  --output-dir /results/code-eval
```

### Specifying Resources

```bash
# Large model - memory is unlimited by default (uses all available on node)
ns generate \
  --cluster my-kubernetes \
  --model /models/llama-70b \
  --server-type vllm \
  --server-gpus 8 \
  --partition gpu-a100 \
  --output-dir /results/llama-70b
```

**Memory Management**: By default, NeMo-Skills sets a memory **request** (for scheduling) but no **limit** (pods can burst to use all available node memory). This is optimal for LLM inference. Auto-calculated request: `16GB + 32GB × num_gpus`.

- To override the request: `HardwareConfig(memory_request_gb=128.0)` (changes what K8s reserves for scheduling)
- To set a hard limit: `HardwareConfig(memory_limit_gb=256.0)` (caps maximum usage, use on multi-tenant clusters)

**Job Naming**: NeMo-Skills automatically handles job naming for both Slurm and Kubernetes:
1. **Sanitization** (K8s only): Names are converted to K8s-compliant format (lowercase, alphanumeric + hyphens, max 63 chars)
2. **Uniqueness**: A timestamp suffix is added (e.g., `-12345`) to prevent collisions on re-runs

Examples:
- `Qwen2.5_Math_7B` → `qwen2-5-math-7b-12345`
- `my/experiment/test` → `my-experiment-test-12345`

Warnings are logged when names are sanitized. The mapping is shown in logs so you can find your jobs.

**Job Dependencies**: Kubernetes doesn't have native job dependencies like Slurm's `afterok`. When pipelines have dependencies, NeMo-Skills automatically runs jobs **sequentially** (waiting for each to complete before starting the next). A warning is logged when this happens.

### Dry Run (Validate Without Running)

```bash
# See what would be submitted without actually running
ns generate \
  --cluster my-kubernetes \
  --model Qwen/Qwen2.5-Math-7B \
  --server-type vllm \
  --server-gpus 1 \
  --benchmarks gsm8k \
  --output-dir /results/test \
  --dry-run
```

---

## Monitoring & Debugging

### Check Job Status

```bash
# List all jobs
kubectl get jobs -n nemo-skills

# Watch job status
kubectl get jobs -n nemo-skills -w

# Get job details
kubectl describe job <job-name> -n nemo-skills
```

### View Logs

```bash
# Get pod name for a job
kubectl get pods -n nemo-skills -l job-name=<job-name>

# View logs from server container
kubectl logs -n nemo-skills <pod-name> -c server

# View logs from client container
kubectl logs -n nemo-skills <pod-name> -c client

# Follow logs in real-time
kubectl logs -n nemo-skills <pod-name> -c server -f
```

### Debug Failed Jobs

```bash
# Check pod events
kubectl describe pod <pod-name> -n nemo-skills

# Common issues:
# - ImagePullBackOff: Check image name and pull secrets
# - Pending: Check resource requests vs available resources
# - OOMKilled: Increase memory in HardwareConfig
# - CrashLoopBackOff: Check container logs for errors
```

### Cancel a Job

```bash
kubectl delete job <job-name> -n nemo-skills
```

---

## Examples

### Example 1: Quick Math Evaluation

Evaluate Qwen on GSM8K:

```bash
ns eval \
  --cluster my-kubernetes \
  --model Qwen/Qwen2.5-Math-7B \
  --server-type vllm \
  --server-gpus 1 \
  --benchmarks gsm8k \
  --output-dir /results/qwen-math
```

After completion, results are in `/results/qwen-math/eval-results/gsm8k/metrics.json`.

### Example 2: Multi-Model Comparison

Compare base vs instruct models:

```bash
ns eval \
  --cluster my-kubernetes \
  --model meta-llama/Llama-3-8B meta-llama/Llama-3-8B-Instruct \
  --server-type vllm vllm \
  --server-gpus 1 1 \
  --benchmarks gsm8k,math500 \
  --output-dir /results/llama-comparison
```

### Example 3: Large Model (70B+)

Run a 70B model with tensor parallelism:

```bash
ns generate \
  --cluster my-kubernetes \
  --model meta-llama/Llama-3-70B-Instruct \
  --server-type vllm \
  --server-gpus 4 \
  --server-args "--tensor-parallel-size 4" \
  --partition gpu-a100 \
  --benchmarks gsm8k \
  --output-dir /results/llama-70b
```

### Example 4: Code Generation with Sandbox

```bash
ns eval \
  --cluster my-kubernetes \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --server-type vllm \
  --server-gpus 2 \
  --benchmarks human-eval,mbpp \
  --with-sandbox \
  --output-dir /results/code-eval
```

### Example 5: Python SDK Usage

```python
#!/usr/bin/env python3
"""Example: Running NeMo-Skills on Kubernetes via Python SDK."""

from nemo_skills.pipeline.utils.declarative import (
    Command,
    CommandGroup,
    HardwareConfig,
    Pipeline,
)
from nemo_skills.pipeline.utils.scripts import (
    ServerScript,
    GenerationClientScript,
)
from nemo_skills.pipeline.utils.cluster import load_cluster_config

# Load cluster configuration
cluster_config = load_cluster_config("my-kubernetes")

# Create server script (vLLM inference server)
server = ServerScript(
    server_type="vllm",
    model_path="Qwen/Qwen2.5-Math-7B",
    num_gpus=1,
    cluster_config=cluster_config,
)

# Create client script (generation client)
client = GenerationClientScript(
    output_dir="/results/python-example",
    servers=[server],
    model_names=["Qwen/Qwen2.5-Math-7B"],
    server_types=["vllm"],
    extra_arguments="++split=test ++prompt_config=generic/math",
)

# Wrap in Commands with container assignments
server_cmd = Command(script=server, container="vllm", name="server")
client_cmd = Command(script=client, container="nemo-skills", name="client")

# Create CommandGroup (runs in single multi-container Pod)
group = CommandGroup(
    commands=[server_cmd, client_cmd],
    hardware=HardwareConfig(num_gpus=1, partition="gpu-a100"),
    name="inference",
    log_dir="/results/logs",
)

# Create and run pipeline
pipeline = Pipeline(
    name="python-k8s-example",
    cluster_config=cluster_config,
    jobs=[{"name": "generate", "group": group}],
)

# Run (set dry_run=True to validate without executing)
pipeline.run(dry_run=False)
```

---

## Troubleshooting

### Job Stuck in Pending

```bash
# Check why pod isn't scheduled
kubectl describe pod -l job-name=<job-name> -n nemo-skills

# Common causes:
# 1. Insufficient GPU resources
kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'

# 2. Node selector doesn't match
kubectl get nodes --show-labels | grep nvidia

# 3. PVC not bound
kubectl get pvc -n nemo-skills
```

### Image Pull Errors

```bash
# Check secret exists
kubectl get secret nvcr-secret -n nemo-skills

# Verify secret is correct
kubectl get secret nvcr-secret -n nemo-skills -o yaml

# Test image pull manually
kubectl run test --image=nvcr.io/nvidia/nemo:24.07 \
  --overrides='{"spec":{"imagePullSecrets":[{"name":"nvcr-secret"}]}}' \
  -n nemo-skills --rm -it -- echo "OK"
```

### Out of Memory (OOMKilled)

```bash
# Check pod status
kubectl describe pod <pod-name> -n nemo-skills | grep -A5 "Last State"

# Memory management:
# - memory_request_gb: What K8s reserves for scheduling (auto: 16GB + 32GB×GPUs)
# - memory_limit_gb: Hard cap on usage (None = unlimited, can use all available)

# Override request (affects scheduling):
hardware=HardwareConfig(num_gpus=8, memory_request_gb=128.0)

# Set hard limit (for multi-tenant clusters):
hardware=HardwareConfig(num_gpus=8, memory_limit_gb=512.0)

# If you're hitting OOM despite no limits, check node memory:
kubectl describe node <node-name> | grep -A5 "Allocated resources"
```

### Storage Permission Issues

```bash
# Check PVC is mounted correctly
kubectl exec -it <pod-name> -n nemo-skills -c client -- ls -la /models

# Check PVC access mode (should be ReadWriteMany for shared access)
kubectl get pvc -n nemo-skills -o yaml | grep accessModes
```

### Server Not Starting

```bash
# Check server container logs
kubectl logs <pod-name> -n nemo-skills -c server

# Common issues:
# - Model not found: Check HF_HOME and model path
# - CUDA errors: Check GPU allocation and driver compatibility
# - Port conflicts: Usually resolved by Kubernetes
```

### Job Name Issues

NeMo-Skills automatically handles job naming:
1. Sanitizes names for K8s compliance (lowercase, valid chars)
2. Adds a unique timestamp suffix to prevent collisions

If you see logs like:

```text
Job name 'Qwen2.5_Math_7B' is not Kubernetes-compliant. Sanitized to 'qwen2-5-math-7b'.
Job 'Qwen2.5_Math_7B' will be submitted as 'qwen2-5-math-7b-12345'
```

This is normal. To find your job:

```bash
# Use the logged name (with suffix) or search by prefix
kubectl get jobs -n nemo-skills | grep qwen2-5-math

# Or list all NeMo-Skills jobs
kubectl get jobs -n nemo-skills -l app=nemo-skills
```

The unique suffix ensures you can re-run experiments without "job already exists" errors.

### Jobs Running Out of Order (Dependencies)

If you have pipelines with job dependencies and see:

```text
Pipeline has job dependencies but sequential=False.
Kubernetes does not support native job dependencies (like Slurm's afterok).
Auto-enabling sequential mode to ensure correct execution order.
```

This means NeMo-Skills detected dependencies and is running jobs sequentially to ensure correctness. Jobs will wait for their dependencies to complete before starting.

---

## Architecture Overview

When you run `ns generate --cluster my-kubernetes`, here's what happens:

```text
┌─────────────────────────────────────────────────────────────────┐
│                         NeMo-Skills CLI                          │
│                      (ns generate/eval/train)                    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Orchestration                        │
│                                                                  │
│  1. Creates CommandGroup with server + client Commands          │
│  2. Detects executor=kubernetes in cluster config               │
│  3. Routes to _run_kubernetes() method                          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes Backend                            │
│                                                                  │
│  1. Converts CommandGroup → JobSpec with multiple containers    │
│  2. Adds PVC mounts, env vars, resource requests                │
│  3. Submits K8s Job via API                                     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                            │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Multi-Container Pod                   │   │
│  │                                                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │    Server    │  │    Client    │  │   Sandbox    │  │   │
│  │  │   (vLLM)     │  │ (NeMo-Skills)│  │  (Optional)  │  │   │
│  │  │              │  │              │  │              │  │   │
│  │  │  Port 8000   │◄─│ localhost:   │──│  Port 6000   │  │   │
│  │  │              │  │    8000      │  │              │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  │                                                          │   │
│  │  Shared: /models (PVC), /results (PVC), /data (PVC)     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

Key points:
- **Multi-container Pod**: Server, client, and sandbox run in the same Pod
- **localhost networking**: Containers communicate via localhost (no Service needed)
- **Shared storage**: All containers mount the same PVCs
- **GPU scheduling**: Only server container requests GPUs

---

## Comparison: Slurm vs Kubernetes

| Aspect | Slurm | Kubernetes |
|--------|-------|------------|
| Job submission | `sbatch` via NeMo-Run | K8s Jobs API |
| Multi-container | Heterogeneous jobs | Multi-container Pods |
| GPU scheduling | Partitions | Node selectors + device plugin |
| Inter-container comms | `$SLURM_MASTER_NODE` | `localhost` (same Pod) |
| Storage | Shared filesystem | PersistentVolumeClaims |
| Logs | File-based | `kubectl logs` |
| Job dependencies | `--dependency=afterok` | Sequential execution (auto-enabled) |
| Job naming | Name + unique suffix | Name + sanitize + unique suffix |

**Same commands work on both:**

```bash
# Slurm
ns generate --cluster my-slurm --model llama-8b --server-gpus 1 ...

# Kubernetes (identical!)
ns generate --cluster my-kubernetes --model llama-8b --server-gpus 1 ...
```

The only difference is the cluster config file.

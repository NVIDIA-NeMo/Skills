# Migrating NeMo-Skills from Slurm to Kubernetes

This guide walks through migrating NeMo-Skills workloads from Slurm to Kubernetes.

## Overview

NeMo-Skills supports both Slurm and Kubernetes as compute backends. The same job logic works on both - only the cluster configuration changes.

**Key differences:**

| Aspect | Slurm | Kubernetes |
|--------|-------|------------|
| Job submission | `sbatch` + NeMo-Run | K8s Jobs API |
| Multi-container | Heterogeneous jobs | Multi-container Pods |
| GPU scheduling | Partitions | Node selectors + device plugin |
| Inter-container comms | `$SLURM_MASTER_NODE` | `localhost` (same Pod) |
| Storage | Shared filesystem | PersistentVolumeClaims |

## Pre-Migration Checklist

### Infrastructure Requirements

- [ ] Kubernetes cluster 1.24+ deployed
- [ ] [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/) installed
- [ ] Storage class with ReadWriteMany (RWX) support (NFS, EFS, Azure Files, etc.)
- [ ] Network access to container registry (NGC, Docker Hub, etc.)

### Verify GPU Operator

```bash
# Check GPU operator pods
kubectl get pods -n gpu-operator

# Check nodes have GPU resources
kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'

# Expected: "8" or similar for GPU nodes
```

## Step 1: Apply Kubernetes Manifests

```bash
cd cluster_configs/kubernetes/

# Create namespace and RBAC
kubectl apply -f rbac.yaml

# Create image pull secret for NGC
kubectl create secret docker-registry nvcr-secret \
  --namespace=nemo-skills \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=YOUR_NGC_API_KEY

# Edit storage.yaml - replace REPLACE_WITH_YOUR_STORAGE_CLASS
# Then apply
kubectl apply -f storage.yaml

# Verify RBAC (required for multi-node headless services)
kubectl auth can-i create jobs --as=system:serviceaccount:nemo-skills:nemo-skills-sa -n nemo-skills
kubectl auth can-i create services --as=system:serviceaccount:nemo-skills:nemo-skills-sa -n nemo-skills
kubectl auth can-i delete services --as=system:serviceaccount:nemo-skills:nemo-skills-sa -n nemo-skills
```

## Step 2: Convert Cluster Configuration

### Slurm Config → Kubernetes Config

**Before (Slurm):**
```yaml
# cluster_configs/my-slurm.yaml
executor: slurm
account: research-team
partition: gpu-a100
cpu_partition: cpu

containers:
  nemo-skills: nvcr.io/nvidia/nemo-skills:latest
  vllm: nvcr.io/nvidia/vllm:latest

mounts:
  - /shared/models:/models
  - /shared/data:/data

default_timeout: "06:00:00"

env_vars:
  - HF_HOME=/models/hf-cache
```

**After (Kubernetes):**
```yaml
# cluster_configs/my-kubernetes.yaml
executor: kubernetes
namespace: nemo-skills
# kubeconfig: ~/.kube/config  # Optional, uses default

containers:
  nemo-skills: nvcr.io/nvidia/nemo-skills:latest
  vllm: nvcr.io/nvidia/vllm:latest

resource_pools:
  gpu-a100:
    node_selector:
      nvidia.com/gpu.product: NVIDIA-A100-SXM4-80GB
    tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
  cpu:
    node_selector:
      node-type: cpu

storage:
  models:
    pvc_name: nemo-models-pvc
    mount_path: /models
  data:
    pvc_name: nemo-data-pvc
    mount_path: /data

default_timeout: "6h"
service_account: nemo-skills-sa
image_pull_secrets:
  - nvcr-secret

env_vars:
  - HF_HOME=/models/hf-cache
```

### Key Mappings

| Slurm Config | Kubernetes Config |
|--------------|-------------------|
| `account` | `namespace` |
| `partition` | `resource_pools.*.node_selector` |
| `mounts` | `storage.*.pvc_name` + `mount_path` |
| `default_timeout: "06:00:00"` | `default_timeout: "6h"` |

**Migration Notes:**

- **Job naming**: Both Slurm and K8s now get unique timestamp suffixes (e.g., `my-job-12345`) to prevent collisions on re-runs. For K8s, names are also sanitized to be K8s-compliant (lowercase, alphanumeric + hyphens, max 63 chars). Example: `Qwen2.5_Math_7B` → `qwen2-5-math-7b-12345`.

- **Job dependencies**: Slurm's `--dependency=afterok` has no direct K8s equivalent. NeMo-Skills handles this by **auto-enabling sequential mode** when dependencies exist, ensuring jobs run in the correct order.

## Step 3: Populate Storage

Copy your models and data to the PVCs:

```bash
# Option 1: Use a temporary pod to copy data
kubectl run data-loader --image=ubuntu:latest -n nemo-skills \
  --overrides='
{
  "spec": {
    "containers": [{
      "name": "loader",
      "image": "ubuntu:latest",
      "command": ["sleep", "infinity"],
      "volumeMounts": [
        {"name": "models", "mountPath": "/models"},
        {"name": "data", "mountPath": "/data"}
      ]
    }],
    "volumes": [
      {"name": "models", "persistentVolumeClaim": {"claimName": "nemo-models-pvc"}},
      {"name": "data", "persistentVolumeClaim": {"claimName": "nemo-data-pvc"}}
    ]
  }
}'

# Copy data into the pod
kubectl cp /path/to/models nemo-skills/data-loader:/models/
kubectl cp /path/to/data nemo-skills/data-loader:/data/

# Clean up
kubectl delete pod data-loader -n nemo-skills
```

```bash
# Option 2: Direct NFS mount (if using NFS storage)
mount -t nfs <nfs-server>:/exports/models /mnt/models
cp -r /shared/models/* /mnt/models/
```

## Step 4: Test Basic Job

Run a simple test job:

```bash
# Test with local echo command (no GPU)
ns run-cmd \
  --cluster my-kubernetes \
  --cmd "echo 'Hello from Kubernetes!'"
```

## Step 5: Run Inference Pipeline

The same commands work on both Slurm and Kubernetes:

```bash
# Slurm
ns generate \
  --cluster my-slurm \
  --model Qwen/Qwen2.5-Math-7B \
  --server-gpus 8 \
  --output-dir /data/results/slurm-test

# Kubernetes (identical, just different cluster config)
ns generate \
  --cluster my-kubernetes \
  --model Qwen/Qwen2.5-Math-7B \
  --server-gpus 8 \
  --output-dir /data/results/k8s-test
```

## Step 6: Verify Multi-Container Jobs

Test the server+client pattern (multi-container Pod):

```bash
# This creates a Pod with vLLM server + generation client
ns generate \
  --cluster my-kubernetes \
  --model /models/llama-8b \
  --server-type vllm \
  --server-gpus 8 \
  --output-dir /data/results/inference
```

Monitor the job:

```bash
# Watch job status
kubectl get jobs -n nemo-skills -w

# Check pod status
kubectl get pods -n nemo-skills -l app=nemo-skills

# View logs from server container
kubectl logs -n nemo-skills -l job-name=<job-name> -c server

# View logs from client container
kubectl logs -n nemo-skills -l job-name=<job-name> -c client
```

## Troubleshooting

### Job stuck in Pending

```bash
# Check events
kubectl describe job <job-name> -n nemo-skills
kubectl describe pod -l job-name=<job-name> -n nemo-skills

# Common causes:
# - Insufficient GPU resources
# - Node selector doesn't match any nodes
# - PVC not bound
```

### GPU not allocated

```bash
# Verify GPU requests in pod spec
kubectl get pod -l job-name=<job-name> -n nemo-skills -o yaml | grep -A5 resources

# Check node GPU capacity
kubectl describe nodes | grep -A5 "Allocated resources"
```

### Image pull errors

```bash
# Check secret exists
kubectl get secret nvcr-secret -n nemo-skills

# Test pull manually
kubectl run test --image=nvcr.io/nvidia/pytorch:24.01-py3 \
  --overrides='{"spec":{"imagePullSecrets":[{"name":"nvcr-secret"}]}}' \
  -n nemo-skills --rm -it -- echo "Pull OK"
```

### Storage issues

```bash
# Check PVC status
kubectl get pvc -n nemo-skills

# Check if PV is bound
kubectl describe pvc nemo-models-pvc -n nemo-skills
```

## Parallel Operation (Recommended)

During migration, run both clusters in parallel:

```yaml
# Keep both configs
cluster_configs/
├── production-slurm.yaml    # Existing Slurm config
├── production-k8s.yaml      # New Kubernetes config
```

1. Run new/experimental workloads on Kubernetes
2. Keep production workloads on Slurm
3. Gradually migrate as confidence grows
4. Deprecate Slurm config when fully migrated

## Rollback

If Kubernetes has issues, simply switch back to Slurm:

```bash
# Just use the Slurm cluster config
ns generate --cluster production-slurm ...
```

No code changes needed - the backend is determined entirely by the cluster config.

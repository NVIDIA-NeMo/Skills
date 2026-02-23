# NeMo-Skills Kubernetes Setup

This directory contains Kubernetes manifests for setting up NeMo-Skills on a Kubernetes cluster.

## Prerequisites

- Kubernetes cluster 1.24+ (for native job dependencies)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/getting-started.html) installed
- Storage class that supports ReadWriteMany (RWX) access mode
- `kubectl` configured to access your cluster

## Quick Start

```bash
# 1. Create namespace and RBAC
kubectl apply -f rbac.yaml

# 2. Create image pull secret for NGC (if using NVIDIA containers)
kubectl create secret docker-registry nvcr-secret \
  --namespace=nemo-skills \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=YOUR_NGC_API_KEY

# 3. Update storage.yaml with your storage class, then apply
# Edit storage.yaml first: replace REPLACE_WITH_YOUR_STORAGE_CLASS
kubectl apply -f storage.yaml

# 4. Copy and customize the cluster config
cp ../example-kubernetes.yaml ../my-cluster.yaml
# Edit my-cluster.yaml with your settings
```

## Files

| File | Description |
|------|-------------|
| `rbac.yaml` | ServiceAccount, Role, and RoleBinding for jobs/pods/services management |
| `storage.yaml` | PVC templates for models, data, and results |
| `image-pull-secret.yaml` | Instructions for creating image pull secrets |

## Verification

```bash
# Check namespace and service account
kubectl get namespace nemo-skills
kubectl get serviceaccount -n nemo-skills

# Check RBAC
kubectl auth can-i create jobs --as=system:serviceaccount:nemo-skills:nemo-skills-sa -n nemo-skills
kubectl auth can-i create services --as=system:serviceaccount:nemo-skills:nemo-skills-sa -n nemo-skills
kubectl auth can-i delete services --as=system:serviceaccount:nemo-skills:nemo-skills-sa -n nemo-skills

# Check PVCs
kubectl get pvc -n nemo-skills

# Check GPU nodes
kubectl get nodes -l nvidia.com/gpu.present=true
```

## Troubleshooting

### Jobs stuck in Pending
```bash
kubectl describe job <job-name> -n nemo-skills
kubectl describe pod -l job-name=<job-name> -n nemo-skills
```

### Image pull errors
```bash
kubectl get events -n nemo-skills --field-selector reason=Failed
```

### GPU not available
```bash
# Check GPU operator
kubectl get pods -n gpu-operator

# Check node GPU resources
kubectl describe node <node-name> | grep nvidia.com/gpu
```

## Multi-Tenant Setup

For multiple teams, create separate namespaces with resource quotas:

```bash
# Create team namespace
kubectl create namespace team-alpha

# Apply RBAC (update namespace in rbac.yaml)
sed 's/nemo-skills/team-alpha/g' rbac.yaml | kubectl apply -f -

# Add resource quota
kubectl apply -f - <<EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: team-alpha-quota
  namespace: team-alpha
spec:
  hard:
    requests.nvidia.com/gpu: "16"
    requests.memory: "128Gi"
    pods: "20"
EOF
```

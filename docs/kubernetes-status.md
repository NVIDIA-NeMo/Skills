# Kubernetes Backend: Status & Known Issues

This document tracks the implementation status, known limitations, and open issues for the Kubernetes backend port of NeMo-Skills.

**Last Updated**: 2026-02-18
**Status**: Beta - Single-node and multi-node multi-GPU SFT validated on H100 cluster

---

## Implementation Status Summary

### Fully Implemented

| Feature | Status | Notes |
|---------|--------|-------|
| Job submission | ✅ Complete | Creates K8s Jobs via API |
| Multi-container pods | ✅ Complete | Server + client + sandbox in same pod |
| GPU scheduling | ✅ Complete | `nvidia.com/gpu` resource requests |
| PVC storage mounts | ✅ Complete | Configurable in cluster config |
| Node selectors & tolerations | ✅ Complete | Via `resource_pools` config |
| Job status tracking | ✅ Complete | Watch API integration |
| Log streaming | ✅ Complete | `kubectl logs` equivalent |
| Job cancellation | ✅ Complete | Deletes K8s Job |
| Health check | ✅ Complete | K8s API connectivity |
| Timeout handling | ✅ Complete | `activeDeadlineSeconds` |
| Memory auto-calculation | ✅ Complete | 16GB + 32GB per GPU |
| Job name sanitization | ✅ Complete | K8s-compliant names |
| RDMA/IB resource requests | ✅ Complete | Opt-in via `rdma` config for multi-node jobs |
| Multi-node RBAC preflight | ✅ Complete | Fast failure if `services` RBAC is missing |

### Degraded Functionality

| Feature | Slurm Behavior | K8s Behavior | Impact |
|---------|----------------|--------------|--------|
| Job dependencies | Native `--dependency=afterok` | Sequential fallback only | Medium - No parallel DAG execution |
| Heterogeneous jobs | Can span different partitions/nodes | Same node only (multi-container pod) | Medium - Large model + small client must fit on one node |
| External dependencies | Resolves other experiment handles | Skipped with warning | Low - Cross-pipeline deps don't work |

### Not Implemented

| Feature | Slurm Support | K8s Status | Severity | Workaround |
|---------|---------------|------------|----------|------------|
| Multi-node training | `--num_nodes=N` | ✅ Backend + cluster validated (Indexed Job + Headless Service) | N/A | Pipeline integration (`ns nemo_rl sft`) pending |
| Code packaging | NeMo-Run auto-packages | ❌ Not implemented | **Medium** | Bake code into images or mount via PVC |
| SSH tunneling | `--create_tunnel` | ❌ Not implemented | **Medium** | Manual `kubectl port-forward` |
| Log file paths | Writes to `{log_dir}/*.log` | ❌ Different | **Low** | Use `kubectl logs` instead |

---

## Documentation Gaps

These limitations are not documented elsewhere and were discovered during code review:

| Gap | Where It Manifests | Notes |
|-----|-------------------|-------|
| Multi-node training needs MPI Operator | `ns nemo_rl sft --num_nodes=8` | K8s Jobs are single-pod; distributed training needs Kubeflow/MPI Operator |
| Job dependencies force sequential | `declarative.py:641-647` | Logged as warning at runtime only |
| SSH tunneling not available | `ns start_server --create_tunnel` | No equivalent K8s implementation |
| External dependencies skipped | `declarative.py:692-694` | Logged as warning at runtime only |
| Heterogeneous jobs are same-node | Multi-container pod design | Slurm het jobs can span nodes; K8s cannot |
| Code upload patterns don't apply | `--reuse_code` flags | NeMo-Run packager is Slurm-specific |

---

## Known Issues & TODOs

### From Source Code

These TODOs are documented in the codebase:

**`integration.py:57-92`** - NeMo-Run Integration
- [ ] No KubernetesExecutor in nemo-run (options: contribute upstream, adapter class, or wrapper)
- [ ] Code packaging for K8s (options: image builds, PVC mounting, init containers)
- [ ] SlurmBackend is thin wrapper (needs full lifecycle management)
- [ ] Unified CLI entry point for K8s

**`slurm.py:28-46`** - Slurm Backend Gaps (if using new backend interface)
- [ ] `get_status()` - implement via sacct
- [ ] `wait_for_completion()` - implement polling
- [ ] `cancel_job()` - implement via scancel
- [ ] `get_logs()` - read from Slurm output files

### Discovered During Review

- [ ] Multi-node distributed training - need to integrate with MPI Operator or Kubeflow
- [ ] Port-forwarding automation for `--create_tunnel` equivalent
- [ ] Init container for code fetching (git clone or S3 download)
- [ ] Documentation for CI/CD image build workflow

---

## Testing Checklist

Use this checklist when testing on a K8s cluster:

### Infrastructure Setup
- [ ] K8s cluster 1.24+ accessible via kubectl
- [ ] NVIDIA GPU Operator installed and working
- [ ] PVCs created for models/data/results
- [ ] RBAC configured (namespace, service account, roles)
- [ ] Image pull secrets for NGC (if using NVIDIA images)

### Basic Functionality
- [ ] `ns generate` with API model (no GPU needed)
- [ ] `ns generate` with vLLM server (single GPU)
- [ ] `ns eval` on a benchmark (e.g., gsm8k)
- [ ] Multi-container job (server + client + sandbox)
- [ ] Job cancellation via CLI or kubectl
- [ ] Log streaming works

### Edge Cases
- [ ] Job name with special characters (sanitization)
- [ ] Job timeout triggers correctly
- [ ] Failed job reports correct status
- [ ] Large model requiring 8 GPUs on single node
- [ ] Sequential job chain with dependencies

### Multi-Node Training (NEW)
- [x] Single-node multi-GPU SFT via Pipeline+KubernetesBackend (2x H100, GPT-2 124M, NCCL P2P/CUMEM)
- [x] Multi-node multi-GPU SFT via Pipeline+KubernetesBackend (2 nodes x 2 GPUs, Indexed Job + Headless Service)
- [x] NCCL P2P/CUMEM intra-node transport (NVLink-backed on H100)
- [x] NCCL NET/Socket inter-node transport (IB not exposed in container — TCP fallback works)
- [x] Multi-node ring+tree topology verified (nRanks=4, nNodes=2)
- [ ] Pipeline integration (`ns nemo_rl sft --num_nodes=2` on K8s)

### Known to Fail
- [ ] `--create_tunnel` for remote server access
- [ ] External experiment dependencies (cross-pipeline)
- [ ] Code packaging (`--reuse_code` patterns)

---

## Issue Tracking

As you encounter issues during testing, add them here:

### Issue Template

```markdown
### [ISSUE-XXX] Brief Description

**Severity**: High / Medium / Low
**Category**: Missing Feature / Bug / Documentation
**Discovered**: YYYY-MM-DD
**Status**: Open / In Progress / Resolved

**Description**:
What happened or what's missing.

**Steps to Reproduce**:
1. ...
2. ...

**Expected Behavior**:
What should happen.

**Actual Behavior**:
What actually happens.

**Workaround**:
If any.

**Resolution**:
(Fill in when resolved)
```

---

### [ISSUE-001] Placeholder for First Issue

**Severity**: TBD
**Category**: TBD
**Discovered**: TBD
**Status**: Open

**Description**:
(Add first discovered issue here)

---

## References

- [Kubernetes Guide](./kubernetes-guide.md) - Full setup and usage documentation
- [Kubernetes Migration](./kubernetes-migration.md) - Slurm to K8s migration guide
- [Cluster Configs](./basics/cluster-configs.md) - Configuration reference
- [Example K8s Config](../cluster_configs/example-kubernetes.yaml) - Sample configuration

### External Resources
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)
- [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/training/)
- [MPI Operator](https://github.com/kubeflow/mpi-operator)

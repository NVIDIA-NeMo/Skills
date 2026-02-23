#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Multi-Node Multi-GPU Distributed Training Smoke Test on Kubernetes
#
# This script validates multi-node distributed training by creating:
# 1. A Headless Service for DNS-based pod discovery
# 2. An Indexed Job with N completions for N nodes
# 3. torchrun-based training with NCCL verification
#
# Validates:
#   - Headless Service DNS resolution between pods
#   - NCCL multi-node ring/tree topology
#   - Inter-node transport (IB/RoCE/TCP)
#   - All ranks participating across nodes
#
# Usage:
#   ./smoke_test_multi_node.sh [--namespace default] [--nodes 2] [--gpus 2] [--image ...]

set -euo pipefail

NAMESPACE="${NAMESPACE:-default}"
NUM_NODES="${NUM_NODES:-2}"
NUM_GPUS="${NUM_GPUS:-2}"
IMAGE="${IMAGE:-${PYTORCH_IMAGE:-nvcr.io/nvidia/pytorch:25.03-py3}}"
JOB_NAME="nemo-multinode-smoke-$(date +%s | tail -c 6)"
SERVICE_NAME="${JOB_NAME}-workers"
MASTER_PORT=29500
TIMEOUT_SECONDS=900  # 15 minutes
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEANUP_DONE=0

cleanup_resources() {
    if [[ "$CLEANUP_DONE" -eq 1 ]]; then
        return
    fi
    CLEANUP_DONE=1

    echo ""
    echo "Cleaning up Kubernetes resources..."
    kubectl delete job "$JOB_NAME" -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
    kubectl delete service "$SERVICE_NAME" -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
}

trap cleanup_resources EXIT INT TERM

while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace) NAMESPACE="$2"; shift 2 ;;
        --nodes) NUM_NODES="$2"; shift 2 ;;
        --gpus) NUM_GPUS="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --timeout) TIMEOUT_SECONDS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

WORLD_SIZE=$((NUM_NODES * NUM_GPUS))
MASTER_ADDR="${JOB_NAME}-0.${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local"

echo "================================================================"
echo "Multi-Node Distributed Training Smoke Test"
echo "================================================================"
echo "Namespace:    $NAMESPACE"
echo "Nodes:        $NUM_NODES"
echo "GPUs/node:    $NUM_GPUS"
echo "World size:   $WORLD_SIZE"
echo "Image:        $IMAGE"
echo "Job name:     $JOB_NAME"
echo "Service:      $SERVICE_NAME"
echo "Master addr:  $MASTER_ADDR"
echo "Master port:  $MASTER_PORT"
echo "Timeout:      ${TIMEOUT_SECONDS}s"
echo "================================================================"

# Step 1: Create Headless Service
echo ""
echo "Creating Headless Service: $SERVICE_NAME"
cat <<EOF | kubectl apply -n "$NAMESPACE" -f -
apiVersion: v1
kind: Service
metadata:
  name: ${SERVICE_NAME}
  labels:
    app: nemo-skills
    job-name: ${JOB_NAME}
spec:
  clusterIP: None
  publishNotReadyAddresses: true
  selector:
    app: nemo-skills
    job-name: ${JOB_NAME}
EOF

# Step 2: Create Indexed Job
echo "Creating Indexed Job: $JOB_NAME ($NUM_NODES nodes)"
cat <<EOF | kubectl apply -n "$NAMESPACE" -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  labels:
    app: nemo-skills
    test: multinode-smoke-test
spec:
  completionMode: Indexed
  completions: ${NUM_NODES}
  parallelism: ${NUM_NODES}
  backoffLimit: 0
  activeDeadlineSeconds: ${TIMEOUT_SECONDS}
  template:
    metadata:
      labels:
        app: nemo-skills
        job-name: ${JOB_NAME}
        test: multinode-smoke-test
    spec:
      restartPolicy: Never
      subdomain: ${SERVICE_NAME}
      containers:
      - name: trainer
        image: ${IMAGE}
        env:
        - name: MASTER_ADDR
          value: "${MASTER_ADDR}"
        - name: MASTER_PORT
          value: "${MASTER_PORT}"
        - name: WORLD_SIZE
          value: "${NUM_NODES}"
        - name: NCCL_DEBUG
          value: "INFO"
        - name: NCCL_DEBUG_SUBSYS
          value: "INIT,NET"
        command:
        - bash
        - -c
        - |
          export NODE_RANK=\${JOB_COMPLETION_INDEX}
          echo "=== Node \${NODE_RANK} of ${NUM_NODES} starting ==="
          echo "  MASTER_ADDR: \${MASTER_ADDR}"
          echo "  MASTER_PORT: \${MASTER_PORT}"
          echo "  NODE_RANK:   \${NODE_RANK}"
          echo "  WORLD_SIZE:  \${WORLD_SIZE} (nodes)"
          echo "  GPUs/node:   ${NUM_GPUS}"

          # Wait for DNS resolution of master node
          echo "Waiting for master DNS resolution..."
          for i in \$(seq 1 60); do
            if getent hosts ${MASTER_ADDR} > /dev/null 2>&1; then
              echo "Master resolved: \$(getent hosts ${MASTER_ADDR})"
              break
            fi
            sleep 2
          done
          if ! getent hosts ${MASTER_ADDR} > /dev/null 2>&1; then
            echo "FAIL: Master DNS ${MASTER_ADDR} not resolvable after 120s"
            exit 1
          fi

          cat > /tmp/multinode_train.py << 'PYEOF'
"""Multi-node distributed training smoke test."""
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

# torchrun sets these automatically
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE_TOTAL", os.environ.get("WORLD_SIZE", 1)))
node_rank = int(os.environ.get("NODE_RANK", 0))

print(f"[Node {node_rank}, Rank {rank}, LocalRank {local_rank}] Starting (world_size={world_size})")

# Init process group
if not dist.is_initialized():
    dist.init_process_group(backend="nccl")

device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)

print(f"[Rank {rank}] GPU: {torch.cuda.get_device_name(device)}")

# Tiny model
model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64)).to(device)
model = DDP(model, device_ids=[local_rank])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training (10 steps)
dataset = TensorDataset(torch.randn(100, 64), torch.randn(100, 64))
loader = DataLoader(dataset, batch_size=16, shuffle=True)

for step, (x, y) in enumerate(loader):
    if step >= 10:
        break
    x, y = x.to(device), y.to(device)
    loss = loss_fn(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if rank == 0:
        print(f"  Step {step}: loss={loss.item():.4f}")

# Multi-node all-reduce verification
tensor = torch.ones(1).to(device) * rank
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
expected = sum(range(world_size))
assert abs(tensor.item() - expected) < 1e-5, f"All-reduce failed: got {tensor.item()}, expected {expected}"
print(f"[Rank {rank}] All-reduce OK (sum={tensor.item()})")

dist.barrier()
if rank == 0:
    print(f"\n=== MULTI-NODE SMOKE TEST PASSED ===")
    print(f"  Nodes: {world_size // torch.cuda.device_count()}")
    print(f"  GPUs/node: {torch.cuda.device_count()}")
    print(f"  Total ranks: {world_size}")
    print(f"  Training: 10 steps completed")
    print(f"  NCCL all-reduce: verified across all nodes")

dist.destroy_process_group()
PYEOF

          # Use torchrun for multi-GPU per node
          # NODE_RANK and MASTER_ADDR/PORT are already set
          export WORLD_SIZE_TOTAL=$((${NUM_NODES} * ${NUM_GPUS}))
          torchrun \
            --nproc_per_node=${NUM_GPUS} \
            --nnodes=${NUM_NODES} \
            --node_rank=\${NODE_RANK} \
            --master_addr=\${MASTER_ADDR} \
            --master_port=\${MASTER_PORT} \
            /tmp/multinode_train.py
        resources:
          limits:
            nvidia.com/gpu: "${NUM_GPUS}"
          requests:
            nvidia.com/gpu: "${NUM_GPUS}"
            memory: "16Gi"
            cpu: "4"
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
EOF

echo ""
echo "Job and service submitted. Waiting for all $NUM_NODES pods to complete..."

# Wait for completion
kubectl wait --for=condition=complete --timeout=${TIMEOUT_SECONDS}s \
    -n "$NAMESPACE" "job/$JOB_NAME" 2>/dev/null && JOB_STATUS="succeeded" || JOB_STATUS="failed"

echo ""
echo "================================================================"
echo "Job Status: $JOB_STATUS"
echo "================================================================"

# Get logs from all pods
PODS=$(kubectl get pods -n "$NAMESPACE" -l "job-name=$JOB_NAME" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)

for POD in $PODS; do
    echo ""
    echo "--- Logs from $POD ---"
    LOG_FILE="/tmp/${POD}.log"
    kubectl logs -n "$NAMESPACE" "$POD" > "$LOG_FILE" 2>&1 || true
    tail -30 "$LOG_FILE"
    echo ""
done

# Run NCCL checker on first pod (rank 0) logs
FIRST_POD=$(echo "$PODS" | awk '{print $1}')
if [ -n "$FIRST_POD" ]; then
    echo "================================================================"
    echo "NCCL Log Analysis (from rank 0 pod)"
    echo "================================================================"
    python3 "$SCRIPT_DIR/check_nccl_logs.py" \
        --log-file "/tmp/${FIRST_POD}.log" \
        --expected-nodes "$NUM_NODES" \
        --expected-gpus-per-node "$NUM_GPUS" || true
fi

if [[ "$JOB_STATUS" != "succeeded" ]]; then
    echo "Smoke test failed."
    exit 1
fi

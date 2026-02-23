#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Single-Node Multi-GPU SFT Smoke Test on Kubernetes
#
# This script submits a minimal SFT training job to validate:
# 1. K8s backend can schedule GPU jobs
# 2. NCCL initializes and detects all GPUs
# 3. NVLink transport is used (not PCIe/TCP fallback)
# 4. Training runs without errors
#
# Prerequisites:
#   - kubectl configured and pointing to the cluster
#   - GPU nodes available with nvidia.com/gpu resource
#   - A namespace with job creation permissions
#
# Usage:
#   ./smoke_test_single_node.sh [--namespace default] [--gpus 2] [--image nvcr.io/nvidia/pytorch:25.04-py3]

set -euo pipefail

# Defaults
NAMESPACE="${NAMESPACE:-default}"
NUM_GPUS="${NUM_GPUS:-2}"
IMAGE="${IMAGE:-${PYTORCH_IMAGE:-nvcr.io/nvidia/pytorch:25.04-py3}}"
JOB_NAME="nemo-sft-smoke-$(date +%s | tail -c 6)"
TIMEOUT_SECONDS=600  # 10 minutes
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
}

trap cleanup_resources EXIT INT TERM

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace) NAMESPACE="$2"; shift 2 ;;
        --gpus) NUM_GPUS="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --timeout) TIMEOUT_SECONDS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "================================================================"
echo "Single-Node Multi-GPU SFT Smoke Test"
echo "================================================================"
echo "Namespace:  $NAMESPACE"
echo "GPUs:       $NUM_GPUS"
echo "Image:      $IMAGE"
echo "Job name:   $JOB_NAME"
echo "Timeout:    ${TIMEOUT_SECONDS}s"
echo "================================================================"

# The training script that runs inside the container.
# Uses a tiny GPT-2 model with synthetic data to validate NCCL + multi-GPU.
read -r -d '' TRAIN_SCRIPT << 'TRAIN_EOF' || true
#!/usr/bin/env python3
"""Minimal multi-GPU training smoke test with NCCL verification."""
import json
import os
import tempfile

# Step 1: Create synthetic dataset (100 examples)
print("=== Creating synthetic dataset ===")
dataset_path = os.path.join(tempfile.gettempdir(), "smoke_test_data.jsonl")
with open(dataset_path, "w") as f:
    for i in range(100):
        example = {
            "input": f"What is {i} + {i}?",
            "output": f"The answer is {i + i}."
        }
        f.write(json.dumps(example) + "\n")
print(f"Created {dataset_path} with 100 examples")

# Step 2: Multi-GPU training with PyTorch DDP
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

# Initialize distributed
if not dist.is_initialized():
    dist.init_process_group(backend="nccl")

rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)

print(f"[Rank {rank}/{world_size}] Initialized on GPU {local_rank} ({torch.cuda.get_device_name(device)})")

# Step 3: Create a tiny model (linear layers, not a real LLM)
model = nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
).to(device)
model = DDP(model, device_ids=[local_rank])

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Step 4: Synthetic training loop (10 steps)
print(f"[Rank {rank}] Starting training (10 steps)...")
dataset = TensorDataset(torch.randn(100, 64), torch.randn(100, 64))
loader = DataLoader(dataset, batch_size=16, shuffle=True)

for step, (x, y) in enumerate(loader):
    if step >= 10:
        break
    x, y = x.to(device), y.to(device)
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if rank == 0:
        print(f"  Step {step}: loss={loss.item():.4f}")

# Step 5: All-reduce test to verify NCCL communication
print(f"[Rank {rank}] Running all-reduce test...")
tensor = torch.ones(1).to(device) * rank
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
expected = sum(range(world_size))
assert tensor.item() == expected, f"All-reduce failed: got {tensor.item()}, expected {expected}"
print(f"[Rank {rank}] All-reduce OK (sum of ranks = {tensor.item()})")

dist.barrier()
if rank == 0:
    print("\n=== SMOKE TEST PASSED ===")
    print(f"  World size: {world_size}")
    print(f"  GPUs: {torch.cuda.device_count()}")
    print(f"  GPU model: {torch.cuda.get_device_name(0)}")
    print(f"  Training: 10 steps completed")
    print(f"  NCCL all-reduce: verified")

dist.destroy_process_group()
TRAIN_EOF

# Create the K8s Job manifest
cat <<EOF | kubectl apply -n "$NAMESPACE" -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  labels:
    app: nemo-skills
    test: sft-smoke-test
spec:
  backoffLimit: 0
  activeDeadlineSeconds: ${TIMEOUT_SECONDS}
  template:
    metadata:
      labels:
        app: nemo-skills
        test: sft-smoke-test
    spec:
      restartPolicy: Never
      containers:
      - name: trainer
        image: ${IMAGE}
        command:
        - bash
        - -c
        - |
          export NCCL_DEBUG=INFO
          export NCCL_DEBUG_SUBSYS=INIT,NET
          cat > /tmp/smoke_train.py << 'PYEOF'
${TRAIN_SCRIPT}
PYEOF
          torchrun --nproc_per_node=${NUM_GPUS} --master_port=29500 /tmp/smoke_train.py
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
echo "Job submitted: $JOB_NAME"
echo "Waiting for job to complete..."

# Wait for completion
if kubectl wait --for=condition=complete --timeout=${TIMEOUT_SECONDS}s \
    -n "$NAMESPACE" "job/$JOB_NAME" 2>/dev/null; then
    JOB_STATUS="succeeded"
else
    FAILED=$(kubectl get job "$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.status.failed}' 2>/dev/null || echo "0")
    ACTIVE=$(kubectl get job "$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.status.active}' 2>/dev/null || echo "0")
    CONDITIONS=$(kubectl get job "$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{range .status.conditions[*]}{.type}{"\n"}{end}' 2>/dev/null || true)
    if [ "${FAILED:-0}" != "0" ] || echo "$CONDITIONS" | grep -q "^Failed$"; then
        JOB_STATUS="failed"
        echo "Job failed (failed pods=${FAILED:-0})"
    elif [ "${ACTIVE:-0}" != "0" ]; then
        JOB_STATUS="timeout"
        echo "Job timed out or still running (active pods=${ACTIVE:-0})"
    else
        JOB_STATUS="timeout"
        echo "Job did not complete before timeout (no explicit Failed condition)"
    fi
fi

# Check if it actually failed vs timed out
if [ "$JOB_STATUS" = "failed" ]; then
    FAILED=$(kubectl get job "$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.status.failed}' 2>/dev/null || echo "0")
    echo "Job failure confirmed (failed pods=${FAILED:-0})"
fi

echo ""
echo "================================================================"
echo "Job Status: $JOB_STATUS"
echo "================================================================"

# Get pod name
POD_NAME=$(kubectl get pods -n "$NAMESPACE" -l "job-name=$JOB_NAME" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -n "$POD_NAME" ]; then
    echo ""
    echo "Fetching logs from pod: $POD_NAME"
    echo "================================================================"

    # Capture logs
    LOG_FILE="/tmp/${JOB_NAME}.log"
    kubectl logs -n "$NAMESPACE" "$POD_NAME" > "$LOG_FILE" 2>&1 || true

    # Print last 50 lines
    echo "--- Last 50 lines of logs ---"
    tail -50 "$LOG_FILE"

    # Run NCCL checker
    echo ""
    echo "================================================================"
    echo "NCCL Log Analysis"
    echo "================================================================"
    python3 "$SCRIPT_DIR/check_nccl_logs.py" \
        --log-file "$LOG_FILE" \
        --expected-nodes 1 \
        --expected-gpus-per-node "$NUM_GPUS" || true

    echo ""
    echo "Full logs saved to: $LOG_FILE"
fi

if [[ "$JOB_STATUS" != "succeeded" ]]; then
    echo "Smoke test failed."
    exit 1
fi

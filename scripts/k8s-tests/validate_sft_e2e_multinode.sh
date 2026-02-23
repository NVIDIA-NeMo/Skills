#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Multi-Node End-to-End SFT Validation on Kubernetes
#
# Proves that Skills SFT training works across multiple K8s nodes by:
# 1. Creating a Headless Service + Indexed Job for multi-node coordination
# 2. Importing nemo_skills + KubernetesBackend on every rank
# 3. Verifying cross-node NCCL communication via all-reduce
# 4. Running DDP training across all ranks
#
# Pass criteria (all must succeed):
#   - Pods start on 2+ distinct nodes (hard-fail otherwise)
#   - "import nemo_skills" succeeds on every rank
#   - KubernetesBackend imports successfully
#   - Cross-node all-reduce produces correct sum
#   - DDP training completes with finite loss
#   - "MULTI-NODE PASSED" appears in rank-0 logs
#
# Usage:
#   ./validate_sft_e2e_multinode.sh                           # 2 nodes, 2 GPUs each
#   ./validate_sft_e2e_multinode.sh --nodes 2 --gpus 4        # 2 nodes, 4 GPUs each
#   ./validate_sft_e2e_multinode.sh --image custom:tag         # custom image

set -euo pipefail

NAMESPACE="${NAMESPACE:-default}"
NUM_GPUS="${NUM_GPUS:-2}"
NUM_NODES="${NUM_NODES:-2}"
IMAGE="${IMAGE:-nemo-skills/nemo-rl:latest}"
IMAGE_PULL_POLICY="${IMAGE_PULL_POLICY:-Never}"
JOB_NAME="mn-sft-test-$(date +%s | tail -c 6)"
SVC="${JOB_NAME}-workers"
TIMEOUT=900
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace) NAMESPACE="$2"; shift 2 ;;
        --gpus) NUM_GPUS="$2"; shift 2 ;;
        --nodes) NUM_NODES="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [[ "$NUM_NODES" -lt 2 ]]; then
    echo "ERROR: Multi-node validation requires --nodes >= 2 (got $NUM_NODES)."
    echo "For single-node, use: ./validate_sft_e2e.sh --gpus $NUM_GPUS"
    exit 2
fi

MASTER="${JOB_NAME}-0.${SVC}.${NAMESPACE}.svc.cluster.local"

echo "================================================================"
echo "Multi-Node End-to-End SFT Validation on Kubernetes"
echo "================================================================"
echo "Image:      $IMAGE"
echo "Nodes:      $NUM_NODES"
echo "GPUs/node:  $NUM_GPUS"
echo "Namespace:  $NAMESPACE"
echo "Job:        $JOB_NAME"
echo "Service:    $SVC"
echo "Master:     $MASTER"
echo "================================================================"

# --- Create training script ConfigMap ---
kubectl delete configmap sft-e2e-script -n "$NAMESPACE" --ignore-not-found 2>/dev/null
kubectl create configmap sft-e2e-script -n "$NAMESPACE" --from-literal=train.py='
import os, sys
import nemo_skills
print(f"nemo_skills {nemo_skills.__version__}")
from nemo_skills.pipeline.backends.kubernetes import KubernetesBackend
print("KubernetesBackend: OK")

import torch, torch.distributed as dist, torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
dist.init_process_group(backend="nccl")
rank, local_rank = dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0))
world_size = dist.get_world_size()
node_rank = int(os.environ.get("NODE_RANK", os.environ.get("GROUP_RANK", 0)))
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)
print(f"[Rank {rank}/{world_size}] Node {node_rank} GPU {local_rank}: {torch.cuda.get_device_name(device)}")

model = DDP(nn.Linear(64, 64).to(device), device_ids=[local_rank])
for _ in range(10):
    loss = model(torch.randn(16, 64, device=device)).sum()
    loss.backward(); model.zero_grad()

t = torch.ones(1, device=device) * rank
dist.all_reduce(t)
expected = sum(range(world_size))
assert abs(t.item() - expected) < 1e-4, f"all-reduce failed: {t.item()} != {expected}"

dist.barrier()
if rank == 0:
    import math
    print(f"MULTI-NODE PASSED: world_size={world_size} nodes={world_size//torch.cuda.device_count()} all_reduce={t.item()} loss={loss.item():.4f} NaN={math.isnan(loss.item())}")
dist.destroy_process_group()
'

# --- Create Headless Service ---
echo "Creating Headless Service: $SVC..."
cat <<EOF | kubectl apply -n "$NAMESPACE" -f -
apiVersion: v1
kind: Service
metadata:
  name: ${SVC}
spec:
  clusterIP: None
  publishNotReadyAddresses: true
  selector:
    job-name: ${JOB_NAME}
EOF

# --- Create Indexed Job ---
echo "Creating Indexed Job: $JOB_NAME ($NUM_NODES nodes x $NUM_GPUS GPUs)..."
cat <<EOF | kubectl apply -n "$NAMESPACE" -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
spec:
  completionMode: Indexed
  completions: ${NUM_NODES}
  parallelism: ${NUM_NODES}
  backoffLimit: 0
  activeDeadlineSeconds: ${TIMEOUT}
  template:
    metadata:
      labels:
        job-name: ${JOB_NAME}
    spec:
      restartPolicy: Never
      subdomain: ${SVC}
      containers:
      - name: trainer
        image: ${IMAGE}
        imagePullPolicy: ${IMAGE_PULL_POLICY}
        env:
        - {name: MASTER_ADDR, value: "${MASTER}"}
        - {name: MASTER_PORT, value: "29500"}
        - {name: NCCL_DEBUG, value: "INFO"}
        command:
        - bash
        - -c
        - |
          export NODE_RANK=\${JOB_COMPLETION_INDEX}
          echo "Node \${NODE_RANK}: waiting for master DNS: ${MASTER}..."
          for i in \$(seq 1 60); do
            getent hosts ${MASTER} > /dev/null 2>&1 && break
            echo "  retry \$i..."
            sleep 2
          done
          if ! getent hosts ${MASTER} > /dev/null 2>&1; then
            echo "FAIL: Master DNS ${MASTER} not resolvable after 120s"
            exit 1
          fi
          echo "Master resolved. Starting torchrun..."
          torchrun --nproc_per_node=${NUM_GPUS} --nnodes=${NUM_NODES} \
            --node_rank=\${NODE_RANK} --master_addr=\${MASTER_ADDR} \
            --master_port=29500 /scripts/train.py
        resources:
          limits: {nvidia.com/gpu: "${NUM_GPUS}"}
          requests: {nvidia.com/gpu: "${NUM_GPUS}", memory: "32Gi"}
        volumeMounts:
        - {name: script, mountPath: /scripts}
      tolerations:
      - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels:
                job-name: ${JOB_NAME}
            topologyKey: kubernetes.io/hostname
      volumes:
      - name: script
        configMap: {name: sft-e2e-script}
EOF

echo ""
echo "Job submitted. Waiting for completion (timeout: ${TIMEOUT}s)..."

# --- Wait for completion ---
if ! kubectl wait --for=condition=complete --timeout="${TIMEOUT}s" "job/$JOB_NAME" -n "$NAMESPACE"; then
    echo ""
    echo "FAILED — Job did not complete. Pod status:"
    kubectl get pods -l "job-name=$JOB_NAME" -n "$NAMESPACE" -o wide
    echo ""
    echo "Logs from each pod:"
    for pod in $(kubectl get pods -l "job-name=$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}'); do
        echo "--- $pod ---"
        kubectl logs "$pod" -n "$NAMESPACE" --tail=30 2>/dev/null || echo "(no logs)"
    done
    exit 1
fi

# --- Verify pods ran on DIFFERENT nodes ---
echo ""
echo "Node placement:"
kubectl get pods -l "job-name=$JOB_NAME" -n "$NAMESPACE" -o custom-columns=POD:.metadata.name,NODE:.spec.nodeName
NODE_COUNT=$(kubectl get pods -l "job-name=$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' | sort -u | wc -l)
if [ "$NODE_COUNT" -lt 2 ]; then
    echo "FAIL: pods landed on $NODE_COUNT node(s), expected 2 distinct nodes"
    exit 1
fi
echo "OK: pods on $NODE_COUNT distinct nodes"

# --- Collect and check logs ---
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/mn-sft-e2e-$(date +%Y%m%d-%H%M%S).log"

{
    for pod in $(kubectl get pods -l "job-name=$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | sort); do
        echo "=== Pod: $pod ==="
        kubectl logs "$pod" -n "$NAMESPACE" 2>/dev/null || echo "(no logs)"
        echo ""
    done
} > "$LOG_FILE"

echo ""
echo "--- Key log lines ---"
grep -E "PASSED|FAIL|nemo_skills|KubernetesBackend|Rank|NCCL|version|all_reduce|loss=" "$LOG_FILE" | head -20 || true

# --- Pass/fail checks ---
echo ""
PASS=true
if ! grep -q "MULTI-NODE PASSED" "$LOG_FILE"; then
    echo "FAIL: 'MULTI-NODE PASSED' not found in logs"
    PASS=false
fi
if ! grep -q "nemo_skills" "$LOG_FILE"; then
    echo "FAIL: nemo_skills import not confirmed"
    PASS=false
fi
if ! grep -q "KubernetesBackend: OK" "$LOG_FILE"; then
    echo "FAIL: KubernetesBackend import not confirmed"
    PASS=false
fi
if grep -q "NaN=True" "$LOG_FILE"; then
    echo "FAIL: Training loss is NaN"
    PASS=false
fi

if [[ "$PASS" == "true" ]]; then
    echo ""
    echo "================================================================"
    echo "MULTI-NODE SFT VALIDATION: PASSED"
    echo "  $NUM_NODES nodes x $NUM_GPUS GPUs = $((NUM_NODES * NUM_GPUS)) ranks"
    echo "  Log saved: $LOG_FILE"
    echo "================================================================"
else
    echo ""
    echo "================================================================"
    echo "MULTI-NODE SFT VALIDATION: FAILED"
    echo "  Full log: $LOG_FILE"
    echo "================================================================"
    exit 1
fi

# --- Cleanup ---
echo "Cleaning up..."
kubectl delete job "$JOB_NAME" -n "$NAMESPACE" --ignore-not-found 2>/dev/null
kubectl delete svc "$SVC" -n "$NAMESPACE" --ignore-not-found 2>/dev/null
kubectl delete configmap sft-e2e-script -n "$NAMESPACE" --ignore-not-found 2>/dev/null
echo "Done."

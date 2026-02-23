#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Build nemo-rl container image via Kaniko, cache on login node, load to GPU nodes.
#
# Flow:
#   1. If cached tarball exists at $CACHE_PATH, skip build
#   2. Otherwise: Kaniko Job builds on one GPU node → kubectl cp back to login node
#   3. For each GPU node: kubectl cp tarball to node → import into containerd
#
# Usage:
#   ./scripts/k8s-tests/image-build/build-and-load.sh
#   ./scripts/k8s-tests/image-build/build-and-load.sh --nodes "node-01"  # single node only
#
# Acceptance Criteria:
#   1. Image imported into containerd k8s.io namespace on target GPU nodes
#   2. Pod with imagePullPolicy: Never can start and import nemo_skills

set -euo pipefail

NAMESPACE="${NAMESPACE:-default}"
CACHE_PATH="${CACHE_PATH:-$HOME/nemo-skills/nemo-rl.tar}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
BASE_IMAGE="${BASE_IMAGE:-${PYTORCH_IMAGE:-nvcr.io/nvidia/pytorch:25.04-py3}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TARGET_NODES=""
LOCAL_CONTEXT=true  # Default: build from local code (reproducible)
GIT_COMMIT=""       # For --remote mode: pin to specific commit (default: HEAD/main)
CLEANUP_PODS=()
CLEANUP_JOBS=()

cleanup_resources() {
    local pod
    local job

    for pod in "${CLEANUP_PODS[@]}"; do
        kubectl delete pod "$pod" -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
    done
    for job in "${CLEANUP_JOBS[@]}"; do
        kubectl delete job "$job" -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
    done
}

trap cleanup_resources EXIT INT TERM

while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace) NAMESPACE="$2"; shift 2 ;;
        --nodes) TARGET_NODES="$2"; shift 2 ;;
        --cache) CACHE_PATH="$2"; shift 2 ;;
        --remote) LOCAL_CONTEXT=false; shift ;;  # Use GitHub ref instead of local code
        --commit) GIT_COMMIT="$2"; shift 2 ;;    # Pin remote build to specific commit/ref
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ -z "$TARGET_NODES" ]; then
    TARGET_NODES=$(kubectl get nodes -l nvidia.com/gpu.present=true -o jsonpath='{.items[*].metadata.name}')
fi

if [ -z "$TARGET_NODES" ]; then
    echo "ERROR: No GPU nodes found with label nvidia.com/gpu.present=true"
    exit 1
fi

echo "================================================================"
echo "NeMo-RL Image Build & Load"
echo "================================================================"
echo "Namespace:    $NAMESPACE"
echo "Cache:        $CACHE_PATH"
echo "Target nodes: $TARGET_NODES"
echo "================================================================"

# ─── Step 1: Build (or use cache) ───────────────────────────────────
if [ -f "$CACHE_PATH" ]; then
    echo ""
    echo "Step 1: CACHED — using existing tarball at $CACHE_PATH"
    ls -lh "$CACHE_PATH"
else
    echo ""
    FIRST_NODE=$(echo "$TARGET_NODES" | awk '{print $1}')

    if [ "$LOCAL_CONTEXT" = true ]; then
        echo "Step 1: Building nemo-rl image from LOCAL code (reproducible)..."
        echo "  Source: $REPO_ROOT"

        # Copy local repo to the build node
        kubectl delete pod build-prep -n "$NAMESPACE" --ignore-not-found 2>/dev/null
        kubectl run build-prep --image=busybox:1.36 --restart=Never \
            --overrides="{\"spec\":{\"nodeName\":\"$FIRST_NODE\",\"containers\":[{\"name\":\"prep\",\"image\":\"busybox:1.36\",\"command\":[\"sleep\",\"3600\"],\"volumeMounts\":[{\"name\":\"raid\",\"mountPath\":\"/raid\"}]}],\"volumes\":[{\"name\":\"raid\",\"hostPath\":{\"path\":\"/raid\"}}]}}" \
            -n "$NAMESPACE" 2>/dev/null
        CLEANUP_PODS+=("build-prep")
        kubectl wait --for=condition=Ready pod/build-prep -n "$NAMESPACE" --timeout=60s
        kubectl exec build-prep -n "$NAMESPACE" -- rm -rf /raid/build-context 2>/dev/null || true
        kubectl cp "$REPO_ROOT" "$NAMESPACE/build-prep:/raid/build-context"

        # Create Dockerfile locally then copy to build node
        TMPDF=$(mktemp)
        cat > "$TMPDF" << DEOF
FROM $BASE_IMAGE

# NVIDIA base images pin package versions via pip constraints.
# Clear all constraint mechanisms so pip can resolve nemo-skills deps freely.
RUN rm -f /etc/pip/constraint*.txt 2>/dev/null; \
    echo '' > /etc/pip/constraint.txt 2>/dev/null || true
ENV PIP_CONSTRAINT=""

# Install NeMo-Skills
COPY . /opt/NeMo-Skills/
WORKDIR /opt/NeMo-Skills
RUN pip install --no-cache-dir -e .

# Install NeMo-RL runtime
RUN pip install --no-cache-dir nemo-run

# Verify
RUN python -c "from nemo_skills.pipeline.backends.kubernetes import KubernetesBackend; print('KubernetesBackend: OK')"
RUN python -c "import nemo_skills; print('nemo_skills OK')"
DEOF
        kubectl cp "$TMPDF" "$NAMESPACE/build-prep:/raid/build-context/Dockerfile"
        rm -f "$TMPDF"
        kubectl delete pod build-prep -n "$NAMESPACE" --ignore-not-found

        # Run Kaniko as init container, keep a lightweight main container alive
        # for kubectl cp. No hostPath for output, no privileged access needed.
        kubectl delete pod build-nemo-rl-local -n "$NAMESPACE" --ignore-not-found 2>/dev/null
        cat <<PODEOF | kubectl apply -n "$NAMESPACE" -f -
apiVersion: v1
kind: Pod
metadata:
  name: build-nemo-rl-local
  labels:
    app: nemo-skills
    purpose: image-build
spec:
  restartPolicy: Never
  nodeName: $FIRST_NODE
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  activeDeadlineSeconds: 7200
  initContainers:
  - name: kaniko
    image: gcr.io/kaniko-project/executor:v1.23.2
    args:
    - --dockerfile=/workspace/Dockerfile
    - --context=/workspace
    - --no-push
    - --tarPath=/output/nemo-rl.tar
    - --cache=false
    - --log-format=text
    - --destination=nemo-skills/nemo-rl:latest
    volumeMounts:
    - name: context
      mountPath: /workspace
    - name: output
      mountPath: /output
  containers:
  - name: output
    image: busybox:1.36
    command: ["sleep", "3600"]
    volumeMounts:
    - name: output
      mountPath: /output
  volumes:
  - name: context
    hostPath:
      path: /raid/build-context
  - name: output
    emptyDir:
      sizeLimit: 75Gi
PODEOF
        BUILD_POD_NAME="build-nemo-rl-local"
        CLEANUP_PODS+=("$BUILD_POD_NAME")
    else
        REMOTE_REF="${GIT_COMMIT:-refs/heads/main}"
        echo "Step 1: Building nemo-rl image from REMOTE (ref: $REMOTE_REF)..."
        kubectl delete job build-nemo-rl -n "$NAMESPACE" --ignore-not-found 2>/dev/null
        # Inject GIT_REF into the manifest so the build uses the pinned commit
        sed "s|value: refs/heads/main|value: $REMOTE_REF|g" "$SCRIPT_DIR/build-nemo-rl.yaml" \
            | kubectl apply -n "$NAMESPACE" -f -
        echo "  Remote build pinned to: $REMOTE_REF"
        JOB_NAME="build-nemo-rl"
        CLEANUP_JOBS+=("$JOB_NAME")
    fi

    echo "Waiting for build (this may take 5-10 minutes)..."
    if [ "$LOCAL_CONTEXT" = true ]; then
        # Pod-based build: wait for init container (kaniko) to finish,
        # then main container (output) becomes Ready.
        if ! kubectl wait --for=condition=Ready "pod/$BUILD_POD_NAME" -n "$NAMESPACE" --timeout=7200s; then
            echo "FAILED — build logs:"
            kubectl logs "$BUILD_POD_NAME" -c kaniko -n "$NAMESPACE" --tail=50
            exit 1
        fi
    else
        # Job-based build (remote): wait for job completion
        if ! kubectl wait --for=condition=complete --timeout=7200s "job/$JOB_NAME" -n "$NAMESPACE"; then
            echo "FAILED — build logs:"
            kubectl logs "job/$JOB_NAME" -n "$NAMESPACE" --tail=50
            exit 1
        fi
    fi
    echo "Build succeeded."

    # Copy tarball back to login node as cache
    echo "Copying tarball to $CACHE_PATH..."
    mkdir -p "$(dirname "$CACHE_PATH")"
    if [ "$LOCAL_CONTEXT" = true ]; then
        # Tarball is in the output emptyDir, accessible via the main container
        kubectl cp "$NAMESPACE/$BUILD_POD_NAME:/output/nemo-rl.tar" "$CACHE_PATH"
        kubectl delete pod "$BUILD_POD_NAME" -n "$NAMESPACE" --ignore-not-found
    else
        # Remote build: tarball is on hostPath /raid, need a helper pod
        BUILD_POD=$(kubectl get pods -l "job-name=$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}')
        BUILD_NODE=$(kubectl get pod "$BUILD_POD" -n "$NAMESPACE" -o jsonpath='{.spec.nodeName}')
        kubectl delete pod cache-copy -n "$NAMESPACE" --ignore-not-found 2>/dev/null
        kubectl run cache-copy --image=busybox:1.36 --restart=Never \
            --overrides="{
                \"spec\": {
                    \"nodeName\": \"$BUILD_NODE\",
                    \"containers\": [{
                        \"name\": \"copy\",
                        \"image\": \"busybox:1.36\",
                        \"command\": [\"sleep\", \"3600\"],
                        \"volumeMounts\": [{\"name\": \"raid\", \"mountPath\": \"/raid\"}]
                    }],
                    \"volumes\": [{\"name\": \"raid\", \"hostPath\": {\"path\": \"/raid\"}}]
                }
            }" -n "$NAMESPACE" 2>/dev/null
        CLEANUP_PODS+=("cache-copy")
        kubectl wait --for=condition=Ready pod/cache-copy -n "$NAMESPACE" --timeout=60s
        kubectl cp "$NAMESPACE/cache-copy:/raid/nemo-rl.tar" "$CACHE_PATH"
        kubectl delete pod cache-copy -n "$NAMESPACE" --ignore-not-found
    fi
    if [ ! -s "$CACHE_PATH" ]; then
        echo "ERROR: Cache tarball is missing or empty: $CACHE_PATH"
        exit 1
    fi
    echo "Cached at $CACHE_PATH ($(du -h "$CACHE_PATH" | cut -f1))"
fi

# ─── Step 2: Load onto each GPU node ────────────────────────────────
echo ""
echo "Step 2: Loading image onto GPU nodes..."
for NODE in $TARGET_NODES; do
    echo ""
    echo "--- Node: $NODE ---"

    # Create a helper pod on this node
    # Security note: loader must run privileged and mount host-root + containerd socket
    # to execute ctr import on the host runtime. This is high risk and intended only
    # for trusted test clusters; namespace scoping/readOnly host-root do not eliminate risk.
    LOAD_POD="load-$NODE"
    kubectl delete pod "$LOAD_POD" -n "$NAMESPACE" --ignore-not-found 2>/dev/null
    kubectl run "$LOAD_POD" --image=debian:bookworm-slim --restart=Never \
        --overrides="{
            \"spec\": {
                \"nodeName\": \"$NODE\",
                \"containers\": [{
                    \"name\": \"loader\",
                    \"image\": \"debian:bookworm-slim\",
                    \"command\": [\"sleep\", \"3600\"],
                    \"securityContext\": {\"privileged\": true},
                    \"volumeMounts\": [
                        {\"name\": \"raid\", \"mountPath\": \"/raid\"},
                        {\"name\": \"containerd-sock\", \"mountPath\": \"/run/containerd\"},
                        {\"name\": \"host-root\", \"mountPath\": \"/host\", \"readOnly\": true}
                    ]
                }],
                \"volumes\": [
                    {\"name\": \"raid\", \"hostPath\": {\"path\": \"/raid\"}},
                    {\"name\": \"containerd-sock\", \"hostPath\": {\"path\": \"/run/containerd\"}},
                    {\"name\": \"host-root\", \"hostPath\": {\"path\": \"/\"}}
                ]
            }
        }" -n "$NAMESPACE" 2>/dev/null
    CLEANUP_PODS+=("$LOAD_POD")
    kubectl wait --for=condition=Ready "pod/$LOAD_POD" -n "$NAMESPACE" --timeout=120s

    # Copy tarball to the node's /raid via kubectl cp
    # Remove any stale/empty tarball first (from previous failed attempts)
    kubectl exec "$LOAD_POD" -n "$NAMESPACE" -- rm -f /raid/nemo-rl.tar 2>/dev/null || true
    echo "  Copying tarball to $NODE ($(du -h "$CACHE_PATH" | cut -f1))..."
    kubectl cp "$CACHE_PATH" "$NAMESPACE/$LOAD_POD:/raid/nemo-rl.tar"

    # Import into containerd — find ctr on the host (various distro/CM paths)
    echo "  Importing into containerd..."
    kubectl exec "$LOAD_POD" -n "$NAMESPACE" -- sh -c '
        CTR=""
        for p in \
            /host/cm/local/apps/containerd/2.1.3/bin/ctr \
            /host/usr/local/bin/ctr \
            /host/usr/bin/ctr \
            $(find /host/cm -name ctr -type f 2>/dev/null | head -1) \
            $(find /host/opt -name ctr -type f 2>/dev/null | head -1); do
            if [ -x "$p" ]; then CTR="$p"; break; fi
        done
        if [ -z "$CTR" ]; then
            echo "ERROR: ctr not found on host filesystem"
            echo "  Searched under /host/{usr,cm,opt}"
            exit 1
        fi
        echo "  Using ctr at: $CTR"
        TARBALL=/raid/nemo-rl.tar
        if [ ! -s "$TARBALL" ]; then
            echo "ERROR: tarball is empty or missing at $TARBALL"
            exit 1
        fi
        echo "  Tarball size: $(du -h "$TARBALL" | cut -f1)"
        "$CTR" --address /run/containerd/containerd.sock -n k8s.io images import "$TARBALL"
        echo "  Import complete on $(hostname)"
    '

    kubectl delete pod "$LOAD_POD" -n "$NAMESPACE" --ignore-not-found
    echo "  Done: $NODE"
done

# ─── Step 3: Validate ───────────────────────────────────────────────
echo ""
echo "Step 3: Validating image is loadable..."
FIRST_NODE=$(echo "$TARGET_NODES" | awk '{print $1}')
kubectl delete pod nemo-rl-validate -n "$NAMESPACE" --ignore-not-found 2>/dev/null
kubectl run nemo-rl-validate --image="nemo-skills/nemo-rl:$IMAGE_TAG" --restart=Never \
    --image-pull-policy=Never \
    --overrides="{\"spec\":{\"nodeName\":\"$FIRST_NODE\"}}" \
    -n "$NAMESPACE" -- python -c "import nemo_skills; print(f'nemo_skills OK: {nemo_skills.__version__}')"
CLEANUP_PODS+=("nemo-rl-validate")
if ! kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/nemo-rl-validate -n "$NAMESPACE" --timeout=180s; then
    echo "Validation FAILED. Pod details:"
    kubectl describe pod nemo-rl-validate -n "$NAMESPACE" || true
    kubectl logs nemo-rl-validate -n "$NAMESPACE" || true
    exit 1
fi
kubectl logs nemo-rl-validate -n "$NAMESPACE"
kubectl delete pod nemo-rl-validate -n "$NAMESPACE" --ignore-not-found

echo ""
echo "================================================================"
echo "Image loaded on: $TARGET_NODES"
echo "Cached at: $CACHE_PATH"
echo "Next: run SFT smoke test with --image nemo-skills/nemo-rl:$IMAGE_TAG"
echo "================================================================"

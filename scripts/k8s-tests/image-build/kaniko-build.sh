#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Generic Kaniko build script for NeMo-Skills container images.
#
# Builds a Dockerfile via Kaniko on a GPU node, saves the tarball to /raid,
# then copies it back to the login node for caching and redistribution.
#
# Usage:
#   # Build nemo-skills image
#   ./kaniko-build.sh nemo-skills
#
#   # Build vllm image
#   ./kaniko-build.sh vllm
#
#   # Build all simple images (excludes nemo-rl which needs special handling)
#   ./kaniko-build.sh --all
#
#   # Build on a specific node
#   ./kaniko-build.sh nemo-skills --node my-gpu-node
#
# Supported images: megatron, nemo-skills, sandbox, vllm, verl
# For nemo-rl: use build-and-load.sh (multi-stage, needs special handling)
#
# Output: $CACHE_DIR/<name>.tar (default: ~/nemo-skills/images/)

set -euo pipefail

NAMESPACE="${NAMESPACE:-default}"
CACHE_DIR="${CACHE_DIR:-$HOME/nemo-skills/images}"
NODE=""
BUILD_ALL=false
TIMEOUT=7200  # 2 hours
GIT_REF=""  # empty = use HEAD (integration/latest mode)

# Simple Dockerfiles that Kaniko can build directly from git context
SIMPLE_IMAGES="megatron nemo-skills sandbox vllm verl"

usage() {
    echo "Usage: $0 <image-name> [--node NODE] [--timeout SECS] [--commit SHA]"
    echo "       $0 --all [--node NODE] [--commit SHA]"
    echo ""
    echo "Images: $SIMPLE_IMAGES"
    echo "For nemo-rl: use build-and-load.sh (multi-stage, needs special handling)"
    echo ""
    echo "Options:"
    echo "  --commit SHA   Pin build to exact git commit (reproducible mode)."
    echo "                 Default: refs/heads/main (integration/latest mode)."
    echo "  --node NODE    Build on a specific K8s node."
    echo "  --timeout SECS Max build time in seconds (default: 7200)."
    exit 1
}

IMAGE_NAME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --all) BUILD_ALL=true; shift ;;
        --node) NODE="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --commit) GIT_REF="$2"; shift 2 ;;
        --help|-h) usage ;;
        -*) echo "Unknown option: $1"; usage ;;
        *) IMAGE_NAME="$1"; shift ;;
    esac
done

# Resolve git ref
if [ -z "$GIT_REF" ]; then
    GIT_REF="refs/heads/main"
    echo "NOTE: Building from refs/heads/main (integration/latest mode)."
    echo "      For reproducible builds, use --commit <SHA>."
fi

if [ "$BUILD_ALL" = false ] && [ -z "$IMAGE_NAME" ]; then
    usage
fi

mkdir -p "$CACHE_DIR"

build_image() {
    local name="$1"
    local dockerfile="Skills/dockerfiles/Dockerfile.${name}"
    local job_name="kaniko-build-${name}"
    local tarball="/raid/nemo-skills-${name}.tar"
    local cache_path="${CACHE_DIR}/${name}.tar"
    local dest_tag="nemo-skills/${name}:latest"

    echo ""
    echo "================================================================"
    echo "Building: $name"
    echo "Dockerfile: $dockerfile"
    echo "Cache: $cache_path"
    echo "================================================================"

    # Skip if cached
    if [ -f "$cache_path" ]; then
        echo "CACHED — $cache_path exists ($(du -h "$cache_path" | cut -f1))"
        echo "Delete it to force rebuild."
        return 0
    fi

    # Node selector
    local node_selector=""
    if [ -n "$NODE" ]; then
        node_selector="\"nodeName\": \"$NODE\","
    fi

    # Clean up previous
    kubectl delete job "$job_name" -n "$NAMESPACE" --ignore-not-found 2>/dev/null

    # Create Kaniko Job
    cat <<EOF | kubectl apply -n "$NAMESPACE" -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${job_name}
  labels:
    app: nemo-skills
    purpose: image-build
    image: ${name}
spec:
  backoffLimit: 0
  activeDeadlineSeconds: ${TIMEOUT}
  template:
    metadata:
      labels:
        app: nemo-skills
        purpose: image-build
        image: ${name}
    spec:
      restartPolicy: Never
      ${node_selector:+nodeName: ${NODE}}
      nodeSelector:
        nvidia.com/gpu.present: "true"
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - name: kaniko
        image: gcr.io/kaniko-project/executor:v1.23.2
        args:
        - --dockerfile=${dockerfile}
        - --context=git://github.com/NVIDIA-NeMo/Skills.git#${GIT_REF}
        - --no-push
        - --tarPath=${tarball}
        - --cache=false
        - --log-format=text
        - --destination=${dest_tag}
        resources:
          requests:
            cpu: "4"
            memory: "16Gi"
          limits:
            cpu: "8"
            memory: "32Gi"
        volumeMounts:
        - name: raid
          mountPath: /raid
      volumes:
      - name: raid
        hostPath:
          path: /raid
          type: DirectoryOrCreate
EOF

    echo "Waiting for build job $job_name..."
    if ! kubectl wait --for=condition=complete --timeout="${TIMEOUT}s" "job/$job_name" -n "$NAMESPACE"; then
        echo "FAILED — logs:"
        kubectl logs "job/$job_name" -n "$NAMESPACE" --tail=30
        return 1
    fi

    echo "Build succeeded. Copying tarball to $cache_path..."

    # Get the node it ran on and copy tarball back
    local build_pod
    build_pod=$(kubectl get pods -l "image=$name,purpose=image-build" -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}')
    local build_node
    build_node=$(kubectl get pod "$build_pod" -n "$NAMESPACE" -o jsonpath='{.spec.nodeName}')

    # Helper pod to access /raid on that node
    local copy_pod="copy-${name}"
    kubectl delete pod "$copy_pod" -n "$NAMESPACE" --ignore-not-found 2>/dev/null
    kubectl run "$copy_pod" --image=busybox:1.36 --restart=Never \
        --overrides="{
            \"spec\": {
                \"nodeName\": \"$build_node\",
                \"containers\": [{
                    \"name\": \"copy\",
                    \"image\": \"busybox:1.36\",
                    \"command\": [\"sleep\", \"600\"],
                    \"volumeMounts\": [{\"name\": \"raid\", \"mountPath\": \"/raid\"}]
                }],
                \"volumes\": [{\"name\": \"raid\", \"hostPath\": {\"path\": \"/raid\"}}]
            }
        }" -n "$NAMESPACE" 2>/dev/null
    cleanup_copy_pod() {
        kubectl delete pod "$copy_pod" -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
    }
    trap cleanup_copy_pod EXIT ERR
    kubectl wait --for=condition=Ready "pod/$copy_pod" -n "$NAMESPACE" --timeout=60s
    kubectl cp "$NAMESPACE/$copy_pod:${tarball}" "$cache_path"
    trap - EXIT ERR
    cleanup_copy_pod

    echo "Cached: $cache_path ($(du -h "$cache_path" | cut -f1))"
    kubectl delete job "$job_name" -n "$NAMESPACE" --ignore-not-found
}

# Build
if [ "$BUILD_ALL" = true ]; then
    echo "Building all images: $SIMPLE_IMAGES"
    for img in $SIMPLE_IMAGES; do
        build_image "$img" || echo "WARNING: $img build failed, continuing..."
    done
    echo ""
    echo "================================================================"
    echo "All builds complete. Cached images:"
    ls -lh "$CACHE_DIR/"*.tar 2>/dev/null || echo "(none)"
    echo ""
    echo "For nemo-rl, run: ./build-and-load.sh"
    echo "================================================================"
else
    build_image "$IMAGE_NAME"
fi

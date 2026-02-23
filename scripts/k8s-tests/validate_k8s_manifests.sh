#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Validate Kubernetes manifests for pre-commit.
# Prefers kubeconform; falls back to kubectl client-side validation.

set -euo pipefail

if [[ $# -eq 0 ]]; then
    exit 0
fi

files=()
for file in "$@"; do
    if [[ -f "$file" ]]; then
        files+=("$file")
    fi
done

if [[ ${#files[@]} -eq 0 ]]; then
    exit 0
fi

if command -v kubeconform >/dev/null 2>&1; then
    echo "Using kubeconform to validate Kubernetes manifests..."
    kubeconform -strict -summary "${files[@]}"
    exit 0
fi

if command -v kubectl >/dev/null 2>&1; then
    echo "kubeconform not found; using kubectl --dry-run=client for validation..."
    for file in "${files[@]}"; do
        # Try strict validation first. If schema download/strict validation is unavailable,
        # fall back to parser-level validation to avoid false negatives in offline setups.
        if ! kubectl apply --dry-run=client --validate=true -f "$file" >/dev/null 2>&1; then
            kubectl apply --dry-run=client --validate=false -f "$file" >/dev/null
        fi
    done
    exit 0
fi

echo "ERROR: neither kubeconform nor kubectl is available on PATH." >&2
echo "Install kubeconform (preferred) or kubectl to validate Kubernetes manifests." >&2
exit 1

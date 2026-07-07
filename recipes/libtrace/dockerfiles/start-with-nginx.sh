#!/bin/bash
# LibTrace sandbox entrypoint
#
# Wraps the standard nemo-skills start-with-nginx.sh with environment
# tweaks needed for the scientific libraries in this image (PySCF,
# RDKit, OpenBabel, etc.).
#
# The standard entrypoint is copied into the image at /start-with-nginx-base.sh
# during the Docker build.  This script sets env vars and then exec's it.

set -e

# ── MPI / InfiniBand ────────────────────────────────────────────────
# Prevent MPI/InfiniBand memory allocation failures when scientific
# libraries (numpy, scipy, PySCF, etc.) try to use MPI simultaneously
# across workers.  Forces MPI to use TCP only, which is sufficient for
# the single-node sandbox.
export OMPI_MCA_btl="^openib"
export OMPI_MCA_btl_openib_allow_ib=0
export UCX_NET_DEVICES=""
export UCX_TLS="tcp,self,sm"
export OMPI_MCA_mca_base_component_show_load_errors=0
export UCX_WARN_UNUSED_ENV_VARS=n

# ── PySCF ───────────────────────────────────────────────────────────
export PYSCF_MAX_MEMORY=4000  # MB per worker

# ── Thread limits ───────────────────────────────────────────────────
# One thread per worker prevents resource contention when user code
# calls numpy/scipy with multi-threaded BLAS backends.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ── Micromamba activation ───────────────────────────────────────────
# Activate the sandbox env so conda activation hooks (e.g., environment
# variables for scientific packages) are applied.
if command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook -s bash)" || true
    micromamba activate sandbox || true
fi

# ── Delegate to base entrypoint ─────────────────────────────────────
exec /start-with-nginx-base.sh "$@"

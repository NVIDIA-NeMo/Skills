#!/usr/bin/env bash
set -euo pipefail

: "${RUNTIME_CACHE_ARCHIVE:?Set the destination RUNTIME_CACHE_ARCHIVE}"
: "${SKILLS_IMAGE:?Set the Pyxis SKILLS_IMAGE}"

export SWE_AGENT_REPO=${SWE_AGENT_REPO:-https://github.com/SWE-agent/SWE-agent.git}
export SWE_AGENT_COMMIT=${SWE_AGENT_COMMIT:-3ea751c087f32b16e039a2233dd6eefecef325d5}
export SWE_BENCH_REPO=${SWE_BENCH_REPO:-https://github.com/Kipok/SWE-bench.git}
export SWE_BENCH_COMMIT=${SWE_BENCH_COMMIT:-c6cd63f0f48956fa3db9ba1e0fadff3a4cd2e2c6}

local_cache="/tmp/${USER:-swe}-swe-runtime-cache-${SLURM_JOB_ID}"
publish_archive="${RUNTIME_CACHE_ARCHIVE}.tmp.${SLURM_JOB_ID}"
mkdir -p "${local_cache}" "$(dirname -- "${RUNTIME_CACHE_ARCHIVE}")"
if [[ -e "${RUNTIME_CACHE_ARCHIVE}" || -e "${publish_archive}" ]]; then
  echo "Refusing to overwrite an existing runtime cache" >&2
  exit 2
fi

srun \
  --container-image="${SKILLS_IMAGE}" \
  --container-mounts="${local_cache}:/root" \
  --no-container-mount-home \
  --container-env=HTTPS_PROXY,HTTP_PROXY,NO_PROXY,SWE_AGENT_REPO,SWE_AGENT_COMMIT,SWE_BENCH_REPO,SWE_BENCH_COMMIT \
  bash -lc '
    set -euo pipefail
    retry() {
      local attempt
      for attempt in $(seq 1 10); do
        if "$@"; then return 0; fi
        sleep $((attempt * 5))
      done
      return 1
    }

    curl -Lf https://astral.sh/uv/install.sh | sh
    source /root/.local/bin/env
    export UV_PYTHON_INSTALL_DIR=/root/uv/python
    export UV_TOOL_DIR=/root/uv/tool
    export UV_TOOL_BIN_DIR=/root/uv/tool-bin

    retry git clone "${SWE_AGENT_REPO}" /root/SWE-agent
    cd /root/SWE-agent
    git checkout "${SWE_AGENT_COMMIT}"
    uv venv --python 3.12 --managed-python venv
    source venv/bin/activate
    uv pip install -e .
    uv pip install rich==14.2.0 orjson
    deactivate

    retry git clone "${SWE_BENCH_REPO}" /root/SWE-bench
    cd /root/SWE-bench
    git checkout "${SWE_BENCH_COMMIT}"
    uv venv --python 3.12 --managed-python venv
    source venv/bin/activate
    uv pip install -e .

    test -x /root/SWE-agent/venv/bin/python
    test -x /root/SWE-bench/venv/bin/python
    touch /root/.swebench_setup_ready
  '

tar -C "${local_cache}" -cf "${publish_archive}" .
archive_list="/tmp/${USER:-swe}-swe-runtime-cache-${SLURM_JOB_ID}.list"
tar -tf "${publish_archive}" >"${archive_list}"
grep -qx './SWE-agent/venv/bin/python' "${archive_list}"
grep -qx './SWE-bench/venv/bin/python' "${archive_list}"
grep -qx './.swebench_setup_ready' "${archive_list}"
mv "${publish_archive}" "${RUNTIME_CACHE_ARCHIVE}"
echo "Published runtime cache: ${RUNTIME_CACHE_ARCHIVE}"

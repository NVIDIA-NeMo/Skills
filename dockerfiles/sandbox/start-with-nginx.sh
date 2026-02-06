#!/bin/bash
# Start nginx load balancer with multiple uwsgi workers
# Uses TCP sockets for workers, supporting both single-node and multi-node deployments.
#
# Multi-node is auto-detected from SLURM environment variables.
# Falls back to single-node (localhost) when SLURM is not available.
#
# =============================================================================
# Environment Variables
# =============================================================================
#
# Required (set by Dockerfile defaults if not provided):
#   NGINX_PORT              Port nginx listens on (default: 6000, set in Dockerfile)
#
# Optional — Worker Configuration:
#   NUM_WORKERS             Number of uWSGI workers per node (default: $(nproc --all))
#   SANDBOX_WORKER_BASE_PORT
#                           Starting TCP port for workers (default: 50001). Workers
#                           bind to sequential ports: base, base+1, ..., base+N-1.
#                           If a port is already in use, the startup algorithm retries
#                           with offset increments.
#   STATEFUL_SANDBOX        Set to 1 (default) for stateful mode: each uWSGI worker
#                           runs a single process to preserve Jupyter kernel sessions
#                           across requests. Set to 0 for stateless mode where
#                           UWSGI_PROCESSES and UWSGI_CHEAPER take effect.
#   UWSGI_PROCESSES         uWSGI processes per worker (default: 1). Only used when
#                           STATEFUL_SANDBOX=0.
#   UWSGI_CHEAPER           uWSGI cheaper mode: minimum number of active processes
#                           (default: 1). Only used when STATEFUL_SANDBOX=0.
#
# Optional — Multi-Node (SLURM):
#   SLURM_JOB_NODELIST      SLURM-provided compressed nodelist (e.g., "node[001-016]").
#                           Presence of this variable triggers multi-node mode.
#                           Automatically set by SLURM — do not set manually.
#   SLURM_JOB_ID            SLURM job ID, used to namespace the port coordination
#                           directory. Automatically set by SLURM.
#   SANDBOX_PORTS_DIR       Explicit path for cross-node port coordination files.
#                           Must be on a shared filesystem (e.g., Lustre). If unset,
#                           defaults to /nemo_run/sandbox_ports_<SLURM_JOB_ID> in
#                           SLURM jobs, or /tmp/sandbox_ports_<PID> for single-node.
#
# Optional — Security:
#   NEMO_SKILLS_SANDBOX_BLOCK_NETWORK
#                           Set to 1 to enable network blocking for sandboxed code.
#                           Uses /etc/ld.so.preload to intercept socket() calls in
#                           all new processes. Applied AFTER nginx/uWSGI start so
#                           the API remains functional. (default: 0)
#
# =============================================================================

set -e

export NUM_WORKERS=${NUM_WORKERS:-$(nproc --all)}

# =============================================================================
# Helper function: Expand SLURM nodelist without scontrol
# =============================================================================
# Parses compressed SLURM nodelist formats like:
#   - "node001" -> "node001"
#   - "node[001-003]" -> "node001 node002 node003"
#   - "node[001,003,005]" -> "node001 node003 node005"
#   - "gpu[01-02],cpu[01-03]" -> "gpu01 gpu02 cpu01 cpu02 cpu03"
expand_nodelist() {
    local nodelist="$1"

    # If empty, return empty
    [ -z "$nodelist" ] && return

    # Use Python for reliable parsing (available in sandbox container)
    python3 -c "
import re
import sys

def expand_nodelist(nodelist):
    '''Expand SLURM nodelist to individual hostnames.'''
    if not nodelist:
        return []

    nodes = []
    # Split by comma, but not commas inside brackets
    # First, handle each bracketed group separately
    remaining = nodelist

    while remaining:
        # Find a complete node specification (prefix + optional bracket range)
        match = re.match(r'([^\[\],]+)(?:\[([^\]]+)\])?(?:,|$)', remaining)
        if not match:
            break

        prefix = match.group(1)
        ranges = match.group(2)
        remaining = remaining[match.end():]

        if ranges is None:
            # Simple hostname without range
            if prefix.strip():
                nodes.append(prefix.strip())
        else:
            # Has range specification like '001-003' or '001,003,005' or '001-003,005'
            for range_part in ranges.split(','):
                range_part = range_part.strip()
                if '-' in range_part:
                    # Range like 001-003
                    parts = range_part.split('-', 1)
                    start_str, end_str = parts[0], parts[1]
                    # Preserve leading zeros
                    width = len(start_str)
                    try:
                        for i in range(int(start_str), int(end_str) + 1):
                            nodes.append(f'{prefix}{i:0{width}d}')
                    except ValueError:
                        # If parsing fails, just add as-is
                        nodes.append(f'{prefix}{range_part}')
                else:
                    # Single number
                    nodes.append(f'{prefix}{range_part}')

    return nodes

nodelist = sys.argv[1]
nodes = expand_nodelist(nodelist)
print(' '.join(nodes))
" "$nodelist" 2>/dev/null
}

# =============================================================================
# Node discovery (auto-detect from SLURM, fallback to localhost)
# =============================================================================
# Debug: Print all relevant environment variables for troubleshooting
_H=$(hostname)
echo "[$_H] === Environment Debug ==="
echo "[$_H] SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-<not set>}"
echo "[$_H] SLURM_NNODES: ${SLURM_NNODES:-<not set>}"
echo "[$_H] SLURM_NODEID: ${SLURM_NODEID:-<not set>}"
echo "[$_H] SLURM_PROCID: ${SLURM_PROCID:-<not set>}"
echo "[$_H] NGINX_PORT: ${NGINX_PORT:-<not set>}"
echo "[$_H] SANDBOX_WORKER_BASE_PORT: ${SANDBOX_WORKER_BASE_PORT:-<not set>}"
echo "[$_H] NUM_WORKERS: ${NUM_WORKERS:-<not set>}"
echo "[$_H] ==========================="

# Parse SLURM_JOB_NODELIST if available, otherwise use localhost
if [ -n "$SLURM_JOB_NODELIST" ]; then
    echo "[$_H] Expanding SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
    ALL_NODES=$(expand_nodelist "$SLURM_JOB_NODELIST")
    if [ -z "$ALL_NODES" ]; then
        echo "[$_H] WARNING: Failed to expand nodelist, falling back to localhost"
        ALL_NODES="127.0.0.1"
    fi
else
    echo "[$_H] No SLURM environment detected, using localhost"
    ALL_NODES="127.0.0.1"
fi

# Determine master (first node) and count
MASTER_NODE=$(echo "$ALL_NODES" | awk '{print $1}')
NODE_COUNT=$(echo "$ALL_NODES" | wc -w)

echo "[$_H] Resolved nodes: $ALL_NODES"
echo "[$_H] Master node: $MASTER_NODE, Total nodes: $NODE_COUNT"

CURRENT_NODE=$(hostname)
# Normalize hostnames (strip domain if present for comparison)
CURRENT_NODE_SHORT="${CURRENT_NODE%%.*}"
MASTER_NODE_SHORT="${MASTER_NODE%%.*}"

# For localhost/127.0.0.1 fallback, we're always the master
echo "[$_H] === Master Detection Debug ==="
echo "[$_H] ALL_NODES: $ALL_NODES"
echo "[$_H] CURRENT_NODE: $CURRENT_NODE"
echo "[$_H] CURRENT_NODE_SHORT: $CURRENT_NODE_SHORT"
echo "[$_H] MASTER_NODE: $MASTER_NODE"
echo "[$_H] MASTER_NODE_SHORT: $MASTER_NODE_SHORT"
echo "[$_H] ================================"

if [ "$ALL_NODES" = "127.0.0.1" ] || [ "$CURRENT_NODE_SHORT" = "$MASTER_NODE_SHORT" ]; then
    IS_MASTER=1
    echo "[$_H] This node is the MASTER node"
    echo "[$_H]   Reason: ALL_NODES='$ALL_NODES' or CURRENT_NODE_SHORT='$CURRENT_NODE_SHORT' matches MASTER_NODE_SHORT='$MASTER_NODE_SHORT'"
else
    IS_MASTER=0
    echo "[$_H] This node is a WORKER node (master: $MASTER_NODE)"
fi

# TCP mode: workers listen on ports
# Use high port range (50000+) to avoid commonly used ports like 6666
SANDBOX_WORKER_BASE_PORT=${SANDBOX_WORKER_BASE_PORT:-50001}

# =============================================================================
# Dynamic port reporting setup (for multi-node deployments)
# =============================================================================
# Each node will find free ports dynamically and report them to shared storage.
# Master will collect all port assignments before generating nginx config.
# IMPORTANT: Must use shared filesystem (Lustre), NOT /tmp (which is node-local)
#
# Directory priority (must be mounted and shared across nodes):
#   1. SANDBOX_PORTS_DIR env var (explicit override)
#   2. /nemo_run/sandbox_ports_* (always mounted in training jobs)
#   3. /workspace/sandbox_ports_* (legacy, may not be mounted)
#   4. /tmp/sandbox_ports_* (local only, for single-node testing)
if [ -n "$SANDBOX_PORTS_DIR" ]; then
    PORTS_REPORT_DIR="$SANDBOX_PORTS_DIR"
elif [ -n "$SLURM_JOB_ID" ]; then
    # In SLURM jobs, use /nemo_run which is reliably mounted to shared storage
    if [ -d "/nemo_run" ]; then
        PORTS_REPORT_DIR="/nemo_run/sandbox_ports_${SLURM_JOB_ID}"
    elif [ -d "/workspace" ]; then
        # Fallback to /workspace if mounted
        PORTS_REPORT_DIR="/workspace/sandbox_ports_${SLURM_JOB_ID}"
    else
        echo "ERROR: Neither /nemo_run nor /workspace are mounted - cannot share ports across nodes"
        echo "Available mounts:"
        mount | grep -E '(nemo_run|workspace|lustre)' || echo "  (none found)"
        exit 1
    fi
else
    PORTS_REPORT_DIR="/tmp/sandbox_ports_$$"
fi
mkdir -p "$PORTS_REPORT_DIR"
# Clean stale port files from previous runs (e.g., if SANDBOX_PORTS_DIR is reused)
rm -f "$PORTS_REPORT_DIR"/*_ports.txt 2>/dev/null || true
echo "[$_H] Port reporting directory: $PORTS_REPORT_DIR"
echo "[$_H] Directory exists: $([ -d "$PORTS_REPORT_DIR" ] && echo 'yes' || echo 'no')"

# Array to track actual ports assigned to workers (may differ from base calculation if ports are busy)
declare -a ACTUAL_WORKER_PORTS

echo "Configuration:"
echo "  Master node: $MASTER_NODE"
echo "  Total nodes: $NODE_COUNT"
echo "  Workers per node: $NUM_WORKERS"
echo "  Base worker port: $SANDBOX_WORKER_BASE_PORT (actual ports may vary if busy)"

echo "Workers per node: $NUM_WORKERS, Nginx port: $NGINX_PORT"

# =============================================================================
# uWSGI configuration
# =============================================================================
# Allow callers to opt-out of single-process state-preserving mode where each worker is given one process
: "${STATEFUL_SANDBOX:=1}"
if [ "$STATEFUL_SANDBOX" -eq 1 ]; then
    UWSGI_PROCESSES=1
    UWSGI_CHEAPER=1
else
    # In stateless mode, honour caller-supplied values
    : "${UWSGI_PROCESSES:=1}"
    : "${UWSGI_CHEAPER:=1}"
fi

export UWSGI_PROCESSES UWSGI_CHEAPER

echo "UWSGI settings: PROCESSES=$UWSGI_PROCESSES, CHEAPER=$UWSGI_CHEAPER"

# Validate and fix uwsgi configuration
if [ -z "$UWSGI_PROCESSES" ]; then
    UWSGI_PROCESSES=2
fi

if [ -z "$UWSGI_CHEAPER" ]; then
    UWSGI_CHEAPER=1
elif [ "$UWSGI_CHEAPER" -le 0 ]; then
    echo "WARNING: UWSGI_CHEAPER ($UWSGI_CHEAPER) must be at least 1"
    UWSGI_CHEAPER=1
    echo "Setting UWSGI_CHEAPER to $UWSGI_CHEAPER"
elif [ "$UWSGI_CHEAPER" -ge "$UWSGI_PROCESSES" ]; then
    echo "WARNING: UWSGI_CHEAPER ($UWSGI_CHEAPER) must be lower than UWSGI_PROCESSES ($UWSGI_PROCESSES)"
    if [ "$UWSGI_PROCESSES" -eq 1 ]; then
        # For single process, disable cheaper mode entirely
        echo "Disabling cheaper mode for single process setup"
        UWSGI_CHEAPER=""
    else
        UWSGI_CHEAPER=$((UWSGI_PROCESSES - 1))
        echo "Setting UWSGI_CHEAPER to $UWSGI_CHEAPER"
    fi
fi

export UWSGI_PROCESSES
if [ -n "$UWSGI_CHEAPER" ]; then
    export UWSGI_CHEAPER
    echo "UWSGI config - Processes: $UWSGI_PROCESSES, Cheaper: $UWSGI_CHEAPER"
else
    echo "UWSGI config - Processes: $UWSGI_PROCESSES, Cheaper: disabled"
fi

# =============================================================================
# Nginx config generation will happen AFTER workers start and report their ports
# =============================================================================
UPSTREAM_FILE="/tmp/upstream_servers.conf"
echo "Nginx upstream config will be generated after all nodes report their ports"

# =============================================================================
# Log setup
# =============================================================================
mkdir -p /var/log/nginx
# Remove symlinks if present and create real log files
rm -f /var/log/nginx/access.log /var/log/nginx/error.log
touch /var/log/nginx/access.log /var/log/nginx/error.log
chmod 644 /var/log/nginx/*.log
# Pre-create per-worker log files so uWSGI writes to regular files
for i in $(seq 1 $NUM_WORKERS); do
    touch /var/log/worker${i}.log
done
chmod 644 /var/log/worker*.log || true

# Mirror logs to stdout/stderr for docker logs
tail -f /var/log/nginx/access.log &> /dev/stdout &
tail -f /var/log/nginx/error.log &> /dev/stderr &
tail -f /var/log/worker*.log &> /dev/stderr &

# =============================================================================
# Worker management
# =============================================================================
echo "Starting $NUM_WORKERS workers in parallel..."
WORKER_PIDS=()

# Function to cleanup on exit
cleanup() {
    echo "Shutting down workers and nginx..."
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    pkill -f nginx || true
    # Clean up temp directories created during startup
    [ -n "$HEALTH_CHECK_DIR" ] && rm -rf "$HEALTH_CHECK_DIR" 2>/dev/null || true
    [ -n "$REMOTE_HEALTH_DIR" ] && rm -rf "$REMOTE_HEALTH_DIR" 2>/dev/null || true
    exit 0
}

trap cleanup SIGTERM SIGINT

# Function to start a single worker (fast, no waiting)
# Spawns uwsgi in background and returns immediately
# Returns "pid:port" - caller must verify worker is healthy
start_worker_fast() {
    local i=$1
    local WORKER_PORT=$2

    # Create a custom uwsgi.ini for this worker
    cat > /tmp/worker${i}_uwsgi.ini << EOF
[uwsgi]
module = main
callable = app
processes = ${UWSGI_PROCESSES}
http-socket = 0.0.0.0:${WORKER_PORT}
vacuum = true
master = true
die-on-term = true
memory-report = true

# Connection and request limits to prevent overload
listen = 100
http-timeout = 300
socket-timeout = 300

# NO auto-restart settings to preserve session persistence
# max-requests and reload-on-rss would kill Jupyter kernels

# Logging for debugging 502 errors
disable-logging = false
log-date = true
log-prefix = [worker${i}]
logto = /var/log/worker${i}.log
EOF

    if [ -n "$UWSGI_CHEAPER" ]; then
        echo "cheaper = ${UWSGI_CHEAPER}" >> /tmp/worker${i}_uwsgi.ini
    fi

    # Clear any old log
    > /var/log/worker${i}.log

    # Start worker with custom config (returns immediately)
    (
        cd /app && env WORKER_NUM=$i uwsgi --ini /tmp/worker${i}_uwsgi.ini
    ) &
    local pid=$!

    echo "$pid:$WORKER_PORT"
}

# Check if a worker failed due to port conflict (check its log file)
worker_had_port_conflict() {
    local i=$1
    grep -q "Address already in use" /var/log/worker${i}.log 2>/dev/null
}

# Check if a worker process is alive
worker_is_alive() {
    local pid=$1
    kill -0 "$pid" 2>/dev/null
}

# Wrapper for monitoring loop restarts - uses existing port assignment
start_worker() {
    local i=$1
    local idx=$((i - 1))
    # Use existing port if available (from initial startup or previous restart)
    local port=${ACTUAL_WORKER_PORTS[$idx]:-$((SANDBOX_WORKER_BASE_PORT + i - 1))}
    start_worker_fast $i $port
}

# Start all workers simultaneously with parallel retry for port conflicts
echo "[$_H] === Starting Workers (Parallel Algorithm) ==="
echo "[$_H] IS_MASTER: $IS_MASTER"
echo "[$_H] SANDBOX_WORKER_BASE_PORT: $SANDBOX_WORKER_BASE_PORT"
echo "[$_H] NUM_WORKERS: $NUM_WORKERS"
echo "[$_H] Port range: $SANDBOX_WORKER_BASE_PORT to $((SANDBOX_WORKER_BASE_PORT + NUM_WORKERS - 1))"
echo "[$_H] =========================="

# =============================================================================
# PARALLEL STARTUP ALGORITHM:
# 1. Spawn ALL workers instantly (no waiting between them)
# 2. Give them a moment to attempt port binding
# 3. Check which workers failed due to port conflicts
# 4. Restart failed workers with new port ranges (parallel)
# 5. Repeat until all workers are running or max retries exceeded
# =============================================================================

MAX_STARTUP_RETRIES=5
PORT_INCREMENT=200  # Jump by this much on each retry round

# Initialize arrays for tracking workers
for i in $(seq 1 $NUM_WORKERS); do
    WORKER_PIDS+=("")
    ACTUAL_WORKER_PORTS+=("")
done

# Phase 1: Start all workers at once (no waiting)
echo "Starting $NUM_WORKERS workers in parallel (no waiting between spawns)..."
START_SPAWN=$(date +%s)

for i in $(seq 1 $NUM_WORKERS); do
    port=$((SANDBOX_WORKER_BASE_PORT + i - 1))
    result=$(start_worker_fast $i $port)
    pid="${result%%:*}"
    idx=$((i - 1))
    WORKER_PIDS[$idx]=$pid
    ACTUAL_WORKER_PORTS[$idx]=$port
done

SPAWN_ELAPSED=$(($(date +%s) - START_SPAWN))
echo "All $NUM_WORKERS workers spawned in ${SPAWN_ELAPSED}s - checking for port conflicts..."

# Phase 2: Parallel retry loop for port conflicts
retry_round=0
while [ $retry_round -lt $MAX_STARTUP_RETRIES ]; do
    # Wait briefly for workers to attempt binding (all in parallel)
    sleep 1

    # Check for port conflicts by scanning logs
    FAILED_WORKERS=()
    for i in $(seq 1 $NUM_WORKERS); do
        idx=$((i - 1))
        pid=${WORKER_PIDS[$idx]}

        # Skip if process is alive (might be okay)
        if worker_is_alive "$pid"; then
            continue
        fi

        # Process died - check if it was a port conflict
        if worker_had_port_conflict $i; then
            FAILED_WORKERS+=($i)
        fi
    done

    if [ ${#FAILED_WORKERS[@]} -eq 0 ]; then
        echo "No port conflicts detected after retry round $retry_round"
        break
    fi

    echo "Retry round $((retry_round + 1)): ${#FAILED_WORKERS[@]} workers had port conflicts, restarting with offset..."

    # Calculate new port offset for this retry round
    PORT_OFFSET=$(( (retry_round + 1) * PORT_INCREMENT ))

    # Restart failed workers with new ports (all at once, parallel)
    for i in "${FAILED_WORKERS[@]}"; do
        idx=$((i - 1))
        old_port=${ACTUAL_WORKER_PORTS[$idx]}
        new_port=$((SANDBOX_WORKER_BASE_PORT + i - 1 + PORT_OFFSET))

        echo "  Worker $i: port $old_port conflict -> trying port $new_port"
        result=$(start_worker_fast $i $new_port)
        pid="${result%%:*}"
        WORKER_PIDS[$idx]=$pid
        ACTUAL_WORKER_PORTS[$idx]=$new_port
    done

    retry_round=$((retry_round + 1))
done

if [ $retry_round -ge $MAX_STARTUP_RETRIES ]; then
    echo "WARNING: Max startup retries reached, some workers may have issues"
fi

echo "All $NUM_WORKERS workers spawned - waiting for readiness..."

# =============================================================================
# Wait for workers to be ready (parallel health checks for faster startup)
# =============================================================================
echo "Waiting for workers to start..."
TIMEOUT=180  # Increased timeout since uwsgi takes time to start
START_TIME=$(date +%s)

# Track which workers are ready to avoid redundant checks
declare -A WORKER_READY

# Directory for health check status files (parallel communication)
HEALTH_CHECK_DIR=$(mktemp -d)

# Function to check a single worker's health (runs in background)
check_worker_health() {
    local worker_num=$1
    local status_file="$HEALTH_CHECK_DIR/worker_${worker_num}"
    local idx=$((worker_num - 1))
    local WORKER_PORT=${ACTUAL_WORKER_PORTS[$idx]}
    local HEALTH_URL="http://127.0.0.1:${WORKER_PORT}/health"

    if curl -s -f --connect-timeout 2 --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; then
        echo "ready" > "$status_file"
    fi
}

# Main readiness loop with parallel health checks
READY_WORKERS=0
LAST_PROGRESS_TIME=0

while [ $READY_WORKERS -lt $NUM_WORKERS ]; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    if [ $ELAPSED -gt $TIMEOUT ]; then
        echo "ERROR: Timeout waiting for workers to start"

        # Show worker status and logs
        echo "Worker status:"
        for i in "${!WORKER_PIDS[@]}"; do
            pid=${WORKER_PIDS[$i]}
            worker_num=$((i+1))
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Worker $worker_num (PID $pid): Process Running"
                echo "    Recent log output:"
                tail -20 /var/log/worker${worker_num}.log 2>/dev/null | sed 's/^/      /' || echo "      No log found"
            else
                echo "  Worker $worker_num (PID $pid): Dead"
                echo "    Log:"
                tail -30 /var/log/worker${worker_num}.log 2>/dev/null | sed 's/^/      /' || echo "      No log found"
            fi
        done

        exit 1
    fi

    # Launch parallel health checks for all unready workers
    check_pids=()
    checking_workers=()

    for i in $(seq 1 $NUM_WORKERS); do
        if [ "${WORKER_READY[$i]}" != "1" ]; then
            check_worker_health $i &
            check_pids+=($!)
            checking_workers+=($i)
        fi
    done

    # Wait for all parallel health checks to complete (with timeout)
    for pid in "${check_pids[@]}"; do
        wait $pid 2>/dev/null || true
    done

    # Collect results from status files
    PREV_READY=$READY_WORKERS
    for i in "${checking_workers[@]}"; do
        if [ -f "$HEALTH_CHECK_DIR/worker_${i}" ]; then
            WORKER_READY[$i]=1
            READY_WORKERS=$((READY_WORKERS + 1))
            rm -f "$HEALTH_CHECK_DIR/worker_${i}"

            idx=$((i - 1))
            WORKER_PORT=${ACTUAL_WORKER_PORTS[$idx]}
            echo "  Worker $i (port $WORKER_PORT): Ready! ($READY_WORKERS/$NUM_WORKERS)"
        fi
    done

    # Show progress every 10 seconds if not all ready
    if [ $READY_WORKERS -lt $NUM_WORKERS ]; then
        if [ $((CURRENT_TIME - LAST_PROGRESS_TIME)) -ge 10 ]; then
            echo "  Progress: $READY_WORKERS/$NUM_WORKERS workers ready (${ELAPSED}s elapsed)"
            LAST_PROGRESS_TIME=$CURRENT_TIME
        fi

        # Only sleep if we didn't make progress (avoid busy-waiting but stay responsive)
        if [ $READY_WORKERS -eq $PREV_READY ]; then
            sleep 1
        fi
    fi
done

echo "[$_H] All local workers are ready!"

# Debug: Show what ports are actually listening
echo "[$_H] === Listening Ports Debug ==="
echo "[$_H] First 3 worker ports (actual assignments):"
for i in 0 1 2; do
    if [ $i -lt ${#ACTUAL_WORKER_PORTS[@]} ]; then
        p=${ACTUAL_WORKER_PORTS[$i]}
        # Use word boundary matching for port number (handles various ss output formats)
        if ss -tlnp 2>/dev/null | grep -qE ":${p}($|[^0-9])" ; then
            echo "[$_H]   Worker $((i+1)) port $p: LISTENING"
        else
            echo "[$_H]   Worker $((i+1)) port $p: NOT LISTENING"
        fi
    fi
done
echo "[$_H] ==============================="

# =============================================================================
# Write port assignments to shared storage AFTER workers are verified ready
# This ensures other nodes only see our ports after workers are actually listening
# =============================================================================
PORTS_FILE="$PORTS_REPORT_DIR/${CURRENT_NODE_SHORT}_ports.txt"
echo "[$_H] Writing port assignments to $PORTS_FILE"
> "$PORTS_FILE"
for i in $(seq 1 $NUM_WORKERS); do
    idx=$((i - 1))
    echo "${i}:${ACTUAL_WORKER_PORTS[$idx]}" >> "$PORTS_FILE"
done
echo "PORT_REPORT_COMPLETE" >> "$PORTS_FILE"
# Force flush to shared filesystem (critical for Lustre visibility across nodes)
sync
echo "[$_H] Port assignments written and synced for $NUM_WORKERS workers"

# =============================================================================
# Start nginx (master node only)
# =============================================================================
if [ "$IS_MASTER" = "1" ]; then
    if [ "$NODE_COUNT" -gt 1 ]; then
        # =============================================================================
        # Collect port assignments from all nodes (dynamic port discovery)
        # =============================================================================
        echo "Waiting for all nodes to report their port assignments..."
        PORT_COLLECT_TIMEOUT=120
        PORT_COLLECT_START=$(date +%s)

        while true; do
            PORT_COLLECT_ELAPSED=$(($(date +%s) - PORT_COLLECT_START))
            if [ $PORT_COLLECT_ELAPSED -gt $PORT_COLLECT_TIMEOUT ]; then
                echo "ERROR: Timeout waiting for all nodes to report ports"
                echo "Expected port files from: $ALL_NODES"
                echo "Found in $PORTS_REPORT_DIR:"
                ls -la "$PORTS_REPORT_DIR" || true
                exit 1
            fi

            # Force Lustre cache invalidation - more aggressive than stat/ls
            # 1. Create and delete a temp file to invalidate directory cache
            _tmp_invalidate="$PORTS_REPORT_DIR/.cache_invalidate_$$_$(date +%s%N)"
            touch "$_tmp_invalidate" 2>/dev/null && rm -f "$_tmp_invalidate" 2>/dev/null || true

            # 2. List directory with -la to force metadata refresh
            ls -la "$PORTS_REPORT_DIR" > /dev/null 2>&1

            # 3. Sync to ensure any pending writes are visible
            sync

            NODES_REPORTED=0
            for node in $ALL_NODES; do
                node_short="${node%%.*}"
                port_file="$PORTS_REPORT_DIR/${node_short}_ports.txt"
                # Use cat to actually READ the file (not just stat) - forces Lustre to fetch latest
                if [ -f "$port_file" ] && cat "$port_file" 2>/dev/null | grep -q "PORT_REPORT_COMPLETE"; then
                    NODES_REPORTED=$((NODES_REPORTED + 1))
                fi
            done

            if [ $NODES_REPORTED -ge $NODE_COUNT ]; then
                echo "All $NODE_COUNT nodes have reported their ports"
                break
            fi

            if [ $((PORT_COLLECT_ELAPSED % 10)) -eq 0 ]; then
                echo "  Waiting for port reports: $NODES_REPORTED/$NODE_COUNT nodes (${PORT_COLLECT_ELAPSED}s elapsed)"
            fi
            sleep 1
        done

        # =============================================================================
        # Generate nginx upstream config from collected port assignments
        # =============================================================================
        echo "Generating nginx upstream config from actual port assignments..."
        > $UPSTREAM_FILE

        for node in $ALL_NODES; do
            node_short="${node%%.*}"
            port_file="$PORTS_REPORT_DIR/${node_short}_ports.txt"
            echo "Reading ports from $port_file for node $node"

            while IFS=: read -r worker_num worker_port; do
                # Skip the PORT_REPORT_COMPLETE marker
                [ "$worker_num" = "PORT_REPORT_COMPLETE" ] && continue
                [ -z "$worker_num" ] && continue

                echo "        server ${node}:${worker_port} max_fails=3 fail_timeout=30s;" >> $UPSTREAM_FILE
            done < "$port_file"
        done

        echo "Generated upstream servers from dynamic port assignments:"
        wc -l < $UPSTREAM_FILE | xargs echo "  Total upstream servers:"
        head -5 $UPSTREAM_FILE
        echo "  ..."

        # =============================================================================
        # Create nginx config with dynamic upstream
        # =============================================================================
        sed "s|\${NGINX_PORT}|${NGINX_PORT}|g" /etc/nginx/nginx.conf.template > /tmp/nginx_temp.conf

        awk -v upstream_file="$UPSTREAM_FILE" '
        /\${UPSTREAM_SERVERS}/ {
            while ((getline line < upstream_file) > 0) {
                print line
            }
            close(upstream_file)
            next
        }
        { print }
        ' /tmp/nginx_temp.conf > /etc/nginx/nginx.conf

        echo "Nginx configuration created with dynamic ports"

        # Test nginx configuration
        echo "Testing nginx configuration..."
        if ! nginx -t; then
            echo "ERROR: nginx configuration test failed"
            echo "Generated nginx.conf:"
            cat /etc/nginx/nginx.conf
            exit 1
        fi

        # =============================================================================
        # Wait for remote workers to be healthy (PARALLEL health checks using xargs)
        # =============================================================================
        echo "Verifying all remote workers are healthy (parallel checks)..."
        REMOTE_TIMEOUT=60
        REMOTE_START=$(date +%s)

        # Directory for parallel health check results
        export REMOTE_HEALTH_DIR=$(mktemp -d)

        # Build list of all workers to check
        ENDPOINTS_FILE=$(mktemp)
        for node in $ALL_NODES; do
            node_short="${node%%.*}"
            port_file="$PORTS_REPORT_DIR/${node_short}_ports.txt"
            while IFS=: read -r worker_num worker_port; do
                [ "$worker_num" = "PORT_REPORT_COMPLETE" ] && continue
                [ -z "$worker_num" ] && continue
                echo "${node}:${worker_port}" >> "$ENDPOINTS_FILE"
            done < "$port_file"
        done
        TOTAL_EXPECTED=$(wc -l < "$ENDPOINTS_FILE")
        echo "  Checking $TOTAL_EXPECTED workers across $NODE_COUNT nodes..."

        while true; do
            REMOTE_ELAPSED=$(($(date +%s) - REMOTE_START))
            if [ $REMOTE_ELAPSED -gt $REMOTE_TIMEOUT ]; then
                echo "WARNING: Timeout waiting for all remote workers, starting nginx anyway"
                break
            fi

            # Run parallel health checks using xargs (64 parallel jobs)
            # Each curl writes a status file if successful
            cat "$ENDPOINTS_FILE" | xargs -P 64 -I {} sh -c '
                endpoint="{}"
                status_file="$REMOTE_HEALTH_DIR/$(echo "$endpoint" | tr ":" "_")"
                [ -f "$status_file" ] && exit 0
                if curl -s -f --connect-timeout 2 --max-time 5 "http://${endpoint}/health" > /dev/null 2>&1; then
                    touch "$status_file"
                fi
            '

            # Count ready workers
            TOTAL_READY=$(find "$REMOTE_HEALTH_DIR" -type f 2>/dev/null | wc -l)

            if [ $TOTAL_READY -ge $TOTAL_EXPECTED ]; then
                echo "All $TOTAL_READY/$TOTAL_EXPECTED remote workers healthy!"
                break
            fi

            echo "  Remote health check: $TOTAL_READY/$TOTAL_EXPECTED workers ready (${REMOTE_ELAPSED}s elapsed)"
            sleep 1
        done

        rm -rf "$REMOTE_HEALTH_DIR" "$ENDPOINTS_FILE"
    else
        # Single-node mode: generate config from local ports only
        echo "Single-node mode: generating nginx config from local ports"
        > $UPSTREAM_FILE
        for i in $(seq 1 $NUM_WORKERS); do
            idx=$((i - 1))
            worker_port=${ACTUAL_WORKER_PORTS[$idx]}
            echo "        server 127.0.0.1:${worker_port} max_fails=3 fail_timeout=30s;" >> $UPSTREAM_FILE
        done

        sed "s|\${NGINX_PORT}|${NGINX_PORT}|g" /etc/nginx/nginx.conf.template > /tmp/nginx_temp.conf
        awk -v upstream_file="$UPSTREAM_FILE" '
        /\${UPSTREAM_SERVERS}/ {
            while ((getline line < upstream_file) > 0) { print line }
            close(upstream_file)
            next
        }
        { print }
        ' /tmp/nginx_temp.conf > /etc/nginx/nginx.conf

        echo "Testing nginx configuration..."
        if ! nginx -t; then
            echo "ERROR: nginx configuration test failed"
            cat /etc/nginx/nginx.conf
            exit 1
        fi
    fi

    echo "Starting nginx on port $NGINX_PORT..."

    # Debug: check if port is already in use before starting nginx
    echo "=== Port $NGINX_PORT Debug ==="
    if ss -tlnp 2>/dev/null | grep -q ":$NGINX_PORT "; then
        echo "WARNING: Port $NGINX_PORT already in use!"
        ss -tlnp 2>/dev/null | grep ":$NGINX_PORT " || true
        netstat -tlnp 2>/dev/null | grep ":$NGINX_PORT " || true
        echo "Processes using port $NGINX_PORT:"
        lsof -i :$NGINX_PORT 2>/dev/null || true
    else
        echo "Port $NGINX_PORT is free"
    fi
    echo "=== End Port Debug ==="

    nginx
else
    # Worker node in multi-node mode: start a local nginx proxy that forwards to master
    # This allows clients to connect to localhost:NGINX_PORT on any node
    echo "Starting local nginx proxy to master node..."

    # Generate a simple proxy config for worker nodes
    cat > /etc/nginx/nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    # Proxy all requests to the master node's nginx LB
    upstream master_lb {
        server ${MASTER_NODE}:${NGINX_PORT};
    }

    server {
        listen ${NGINX_PORT};
        server_name localhost;

        client_max_body_size 10M;
        client_body_buffer_size 128k;

        location / {
            proxy_pass http://master_lb;

            # Forward all headers including X-Session-ID for consistent hashing
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            proxy_set_header X-Session-ID \$http_x_session_id;

            # Match master timeouts
            proxy_connect_timeout 1200s;
            proxy_send_timeout 1200s;
            proxy_read_timeout 1200s;

            proxy_buffering off;
        }

        location /nginx-status {
            stub_status on;
            access_log off;
            allow 127.0.0.1;
            allow ::1;
            deny all;
        }
    }

    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log warn;
}
EOF

    echo "Testing nginx proxy configuration..."
    if ! nginx -t; then
        echo "ERROR: nginx proxy configuration test failed"
        cat /etc/nginx/nginx.conf
        exit 1
    fi

    nginx
    echo "Local nginx proxy started on port $NGINX_PORT -> master $MASTER_NODE:$NGINX_PORT"
fi

# =============================================================================
# Enable network blocking for user code execution if requested
# This MUST happen AFTER nginx/uwsgi start (they need sockets for API)
# Using /etc/ld.so.preload ensures this cannot be bypassed by user code
# Applied on ALL nodes since worker nodes run sandboxed user code too
# =============================================================================
BLOCK_NETWORK_LIB="/usr/lib/libblock_network.so"
if [ "${NEMO_SKILLS_SANDBOX_BLOCK_NETWORK:-0}" = "1" ]; then
    if [ -f "$BLOCK_NETWORK_LIB" ]; then
        echo "$BLOCK_NETWORK_LIB" > /etc/ld.so.preload
        echo "Network blocking ENABLED: All new processes will have network blocked"
        echo "  (API server sockets created before this, so API still works)"
    else
        echo "WARNING: Network blocking requested but $BLOCK_NETWORK_LIB not found"
    fi
fi

# =============================================================================
# Print status summary
# =============================================================================
if [ "$IS_MASTER" = "1" ]; then
    echo "=== Sandbox deployment ready (MASTER) ==="
    echo "Nginx load balancer: http://localhost:$NGINX_PORT"
    echo "Session affinity: enabled (based on X-Session-ID header)"
    echo "Nodes: $NODE_COUNT ($ALL_NODES)"
    echo "Workers per node: $NUM_WORKERS"
    echo "Total workers: $((NODE_COUNT * NUM_WORKERS))"
    echo "Local worker ports: ${ACTUAL_WORKER_PORTS[0]} to ${ACTUAL_WORKER_PORTS[$((NUM_WORKERS-1))]}"
    echo "Nginx status: http://localhost:$NGINX_PORT/nginx-status"
else
    echo "=== Sandbox deployment ready (WORKER NODE) ==="
    echo "Local nginx proxy: http://localhost:$NGINX_PORT -> master $MASTER_NODE:$NGINX_PORT"
    echo "Master node: $MASTER_NODE (nginx LB with consistent hash routing)"
    echo "Local workers: $NUM_WORKERS (ports: ${ACTUAL_WORKER_PORTS[0]} to ${ACTUAL_WORKER_PORTS[$((NUM_WORKERS-1))]})"
fi

echo "UWSGI processes per worker: $UWSGI_PROCESSES"
if [ -n "$UWSGI_CHEAPER" ]; then
    echo "UWSGI cheaper mode: $UWSGI_CHEAPER"
else
    echo "UWSGI cheaper mode: disabled"
fi

# Show process status
echo "Process status:"
for i in "${!WORKER_PIDS[@]}"; do
    pid=${WORKER_PIDS[$i]}
    if kill -0 "$pid" 2>/dev/null; then
        echo "  Worker $((i+1)) (PID $pid): Running"
    else
        echo "  Worker $((i+1)) (PID $pid): Dead"
    fi
done

# =============================================================================
# Monitoring loop
# =============================================================================
echo "Monitoring processes (Ctrl+C to stop)..."

# Only run load monitor on master node
if [ "$IS_MASTER" = "1" ]; then
    monitor_load() {
        echo "Starting worker load monitor (updates every 60s)..."
        while true; do
            sleep 60
            echo "--- Worker Load Stats (Top 10) at $(date) ---"
            grep "upstream:" /var/log/nginx/access.log | awk -F'upstream: ' '{print $2}' | awk -F' session: ' '{print $1}' | sort | uniq -c | sort -nr | head -n 10 || echo "No logs yet"
            echo "--- End Stats ---"
        done
    }
    monitor_load &  # Run in background
fi

while true; do
    # Check if any worker died
    for idx in "${!WORKER_PIDS[@]}"; do
        pid=${WORKER_PIDS[$idx]}
        i=$((idx + 1))
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "WARNING: Worker $i (PID $pid) died - restarting..."
            result=$(start_worker $i)
            new_pid="${result%%:*}"
            new_port="${result##*:}"
            WORKER_PIDS[$idx]=$new_pid
            ACTUAL_WORKER_PORTS[$idx]=$new_port
        fi
    done

    # Check nginx (runs on all nodes)
    if ! pgrep nginx > /dev/null; then
        echo "ERROR: Nginx died unexpectedly"
        cleanup
        exit 1
    fi

    sleep 10
done

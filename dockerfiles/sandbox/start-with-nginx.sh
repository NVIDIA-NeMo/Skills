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
#   SANDBOX_FORCE_SINGLE_NODE
#                           Set to 1 to force single-node mode even when SLURM
#                           variables are present. Useful for debugging or when
#                           multi-node sandbox is not desired.
#
# Optional — Security:
#   NEMO_SKILLS_SANDBOX_BLOCK_NETWORK
#                           Set to 1 to enable network blocking for sandboxed code.
#                           Uses /etc/ld.so.preload to intercept socket() calls in
#                           all new processes. Applied AFTER nginx/uWSGI start so
#                           the API remains functional. Note: in any mode, if a
#                           worker crashes the monitoring loop will attempt to restart
#                           it, but the new process will be unable to bind its socket.
#                           The remaining workers continue serving. (default: 0)
#
# =============================================================================

set -e

export NUM_WORKERS=${NUM_WORKERS:-$(nproc --all)}

# =============================================================================
# Utility functions
# =============================================================================

# Expand SLURM compressed nodelist to space-separated hostnames.
# Parses formats like:
#   - "node001" -> "node001"
#   - "node[001-003]" -> "node001 node002 node003"
#   - "node[001,003,005]" -> "node001 node003 node005"
#   - "gpu[01-02],cpu[01-03]" -> "gpu01 gpu02 cpu01 cpu02 cpu03"
expand_nodelist() {
    local nodelist="$1"
    [ -z "$nodelist" ] && return

    python3 -c "
import re, sys

def expand_nodelist(nodelist):
    if not nodelist:
        return []
    nodes = []
    remaining = nodelist
    while remaining:
        match = re.match(r'([^\[\],]+)(?:\[([^\]]+)\])?(?:,|$)', remaining)
        if not match:
            break
        prefix = match.group(1)
        ranges = match.group(2)
        remaining = remaining[match.end():]
        if ranges is None:
            if prefix.strip():
                nodes.append(prefix.strip())
        else:
            for range_part in ranges.split(','):
                range_part = range_part.strip()
                if '-' in range_part:
                    parts = range_part.split('-', 1)
                    start_str, end_str = parts[0], parts[1]
                    width = len(start_str)
                    try:
                        for i in range(int(start_str), int(end_str) + 1):
                            nodes.append(f'{prefix}{i:0{width}d}')
                    except ValueError:
                        nodes.append(f'{prefix}{range_part}')
                else:
                    nodes.append(f'{prefix}{range_part}')
    return nodes

print(' '.join(expand_nodelist(sys.argv[1])))
" "$nodelist" 2>/dev/null
}

# Start a single uWSGI worker in the background.
# Args: $1=worker_number $2=port
# Prints: "pid:port"
start_worker_fast() {
    local i=$1
    local WORKER_PORT=$2

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
listen = 100
http-timeout = 300
socket-timeout = 300
disable-logging = false
log-date = true
log-prefix = [worker${i}]
logto = /var/log/worker${i}.log
EOF

    if [ -n "$UWSGI_CHEAPER" ]; then
        echo "cheaper = ${UWSGI_CHEAPER}" >> /tmp/worker${i}_uwsgi.ini
    fi

    > /var/log/worker${i}.log
    ( cd /app && env WORKER_NUM=$i uwsgi --ini /tmp/worker${i}_uwsgi.ini ) &
    echo "$!:$WORKER_PORT"
}

# Restart wrapper — reuses the worker's existing port assignment.
start_worker() {
    local i=$1
    local idx=$((i - 1))
    local port=${ACTUAL_WORKER_PORTS[$idx]:-$((SANDBOX_WORKER_BASE_PORT + i - 1))}
    start_worker_fast $i $port
}

worker_had_port_conflict() {
    grep -q "Address already in use" /var/log/worker${1}.log 2>/dev/null
}

worker_is_alive() {
    kill -0 "$1" 2>/dev/null
}

# Generate /etc/nginx/nginx.conf from template + upstream file.
# Uses UPSTREAM_FILE and NGINX_PORT globals.
generate_nginx_config() {
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
}

# Read a node's port file and emit "node:port" lines to stdout.
# Args: $1=node_hostname $2=port_file_path
read_port_file() {
    local node=$1
    local port_file=$2
    while IFS=: read -r worker_num worker_port; do
        [ "$worker_num" = "PORT_REPORT_COMPLETE" ] && continue
        [ -z "$worker_num" ] && continue
        echo "${node}:${worker_port}"
    done < "$port_file"
}

# Wait for all nodes to write their port files to shared storage.
# Uses PORTS_REPORT_DIR, ALL_NODES, NODE_COUNT globals.
wait_for_port_reports() {
    echo "Waiting for all nodes to report their port assignments..."
    local timeout=120
    local start=$(date +%s)

    while true; do
        local elapsed=$(($(date +%s) - start))
        if [ $elapsed -gt $timeout ]; then
            echo "ERROR: Timeout waiting for all nodes to report ports"
            echo "Expected port files from: $ALL_NODES"
            echo "Found in $PORTS_REPORT_DIR:"
            ls -la "$PORTS_REPORT_DIR" || true
            exit 1
        fi

        local reported=0
        for node in $ALL_NODES; do
            local node_short="${node%%.*}"
            local port_file="$PORTS_REPORT_DIR/${node_short}_ports.txt"
            if [ -f "$port_file" ] && grep -q "PORT_REPORT_COMPLETE" "$port_file" 2>/dev/null; then
                reported=$((reported + 1))
            fi
        done

        if [ $reported -ge $NODE_COUNT ]; then
            echo "All $NODE_COUNT nodes have reported their ports"
            return
        fi

        if [ $((elapsed % 10)) -eq 0 ]; then
            echo "  Waiting for port reports: $reported/$NODE_COUNT nodes (${elapsed}s elapsed)"
        fi
        sleep 1
    done
}

# Verify remote workers are reachable (parallel health checks via xargs).
# Args: $1=endpoints_file (one "host:port" per line)
verify_remote_workers() {
    local endpoints_file=$1
    local total_expected=$(wc -l < "$endpoints_file")
    echo "Verifying $total_expected remote workers are healthy (parallel checks)..."

    local timeout=60
    local start=$(date +%s)
    export REMOTE_HEALTH_DIR=$(mktemp -d)

    while true; do
        local elapsed=$(($(date +%s) - start))
        if [ $elapsed -gt $timeout ]; then
            echo "WARNING: Timeout waiting for all remote workers, starting nginx anyway"
            break
        fi

        cat "$endpoints_file" | xargs -P 64 -I {} sh -c '
            endpoint="{}"
            status_file="$REMOTE_HEALTH_DIR/$(echo "$endpoint" | tr ":" "_")"
            [ -f "$status_file" ] && exit 0
            if curl -s -f --connect-timeout 2 --max-time 5 "http://${endpoint}/health" > /dev/null 2>&1; then
                touch "$status_file"
            fi
        '

        local ready=$(find "$REMOTE_HEALTH_DIR" -type f 2>/dev/null | wc -l)
        if [ $ready -ge $total_expected ]; then
            echo "All $ready/$total_expected remote workers healthy!"
            break
        fi

        echo "  Remote health check: $ready/$total_expected workers ready (${elapsed}s elapsed)"
        sleep 1
    done

    rm -rf "$REMOTE_HEALTH_DIR"
}

# =============================================================================
# Node discovery
# =============================================================================
_H=$(hostname)

# Log configured values (only show SLURM vars if they're actually set)
echo "[$_H] NGINX_PORT=$NGINX_PORT NUM_WORKERS=$NUM_WORKERS"
[ -n "$SLURM_JOB_NODELIST" ] && echo "[$_H] SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST SLURM_NNODES=${SLURM_NNODES:-?}"
[ -n "$SANDBOX_FORCE_SINGLE_NODE" ] && echo "[$_H] SANDBOX_FORCE_SINGLE_NODE=$SANDBOX_FORCE_SINGLE_NODE"

if [ "${SANDBOX_FORCE_SINGLE_NODE:-0}" = "1" ]; then
    echo "[$_H] SANDBOX_FORCE_SINGLE_NODE=1, forcing single-node mode"
    ALL_NODES="127.0.0.1"
elif [ -n "$SLURM_JOB_NODELIST" ]; then
    echo "[$_H] Expanding SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
    ALL_NODES=$(expand_nodelist "$SLURM_JOB_NODELIST")
    if [ -z "$ALL_NODES" ]; then
        echo "[$_H] WARNING: Failed to expand SLURM_JOB_NODELIST='$SLURM_JOB_NODELIST'"
        echo "[$_H] Falling back to single-node mode. If multi-node is intended, check that"
        echo "[$_H] SLURM_JOB_NODELIST is correctly set by your SLURM environment."
        ALL_NODES="127.0.0.1"
    fi
else
    echo "[$_H] No SLURM_JOB_NODELIST detected — running in single-node mode"
    ALL_NODES="127.0.0.1"
fi

MASTER_NODE=$(echo "$ALL_NODES" | awk '{print $1}')
NODE_COUNT=$(echo "$ALL_NODES" | wc -w)
CURRENT_NODE_SHORT="${_H%%.*}"
MASTER_NODE_SHORT="${MASTER_NODE%%.*}"

if [ "$ALL_NODES" = "127.0.0.1" ] || [ "$CURRENT_NODE_SHORT" = "$MASTER_NODE_SHORT" ]; then
    IS_MASTER=1
    echo "[$_H] Role: MASTER | Nodes: $NODE_COUNT | Master: $MASTER_NODE"
else
    IS_MASTER=0
    echo "[$_H] Role: WORKER | Master: $MASTER_NODE"
fi

# =============================================================================
# Port coordination setup
# =============================================================================
SANDBOX_WORKER_BASE_PORT=${SANDBOX_WORKER_BASE_PORT:-50001}

if [ -n "$SANDBOX_PORTS_DIR" ]; then
    PORTS_REPORT_DIR="$SANDBOX_PORTS_DIR"
elif [ -n "$SLURM_JOB_ID" ]; then
    if [ -d "/nemo_run" ]; then
        PORTS_REPORT_DIR="/nemo_run/sandbox_ports_${SLURM_JOB_ID}"
    elif [ -d "/workspace" ]; then
        PORTS_REPORT_DIR="/workspace/sandbox_ports_${SLURM_JOB_ID}"
    else
        echo "ERROR: Neither /nemo_run nor /workspace are mounted — cannot share ports across nodes"
        exit 1
    fi
else
    PORTS_REPORT_DIR="/tmp/sandbox_ports_$$"
fi
mkdir -p "$PORTS_REPORT_DIR"
rm -f "$PORTS_REPORT_DIR/${CURRENT_NODE_SHORT}_ports.txt" 2>/dev/null || true
echo "[$_H] Port report dir: $PORTS_REPORT_DIR"

declare -a ACTUAL_WORKER_PORTS
UPSTREAM_FILE="/tmp/upstream_servers.conf"

echo "[$_H] Workers/node: $NUM_WORKERS | Base port: $SANDBOX_WORKER_BASE_PORT | Nginx: $NGINX_PORT"

# =============================================================================
# uWSGI configuration
# =============================================================================
: "${STATEFUL_SANDBOX:=1}"
if [ "$STATEFUL_SANDBOX" -eq 1 ]; then
    UWSGI_PROCESSES=1
    UWSGI_CHEAPER=1
else
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
# Log setup
# =============================================================================
mkdir -p /var/log/nginx
rm -f /var/log/nginx/access.log /var/log/nginx/error.log
touch /var/log/nginx/access.log /var/log/nginx/error.log
chmod 644 /var/log/nginx/*.log
for i in $(seq 1 $NUM_WORKERS); do
    touch /var/log/worker${i}.log
done
chmod 644 /var/log/worker*.log || true

tail -f /var/log/nginx/access.log &> /dev/stdout &
tail -f /var/log/nginx/error.log &> /dev/stderr &
tail -f /var/log/worker*.log &> /dev/stderr &

# =============================================================================
# Worker startup
# =============================================================================
WORKER_PIDS=()

cleanup() {
    echo "Shutting down workers and nginx..."
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    pkill -f nginx || true
    [ -n "$HEALTH_CHECK_DIR" ] && rm -rf "$HEALTH_CHECK_DIR" 2>/dev/null || true
    [ -n "$REMOTE_HEALTH_DIR" ] && rm -rf "$REMOTE_HEALTH_DIR" 2>/dev/null || true
    exit 0
}

trap cleanup SIGTERM SIGINT

MAX_STARTUP_RETRIES=5
PORT_INCREMENT=200

for i in $(seq 1 $NUM_WORKERS); do
    WORKER_PIDS+=("")
    ACTUAL_WORKER_PORTS+=("")
done

# Phase 1: Spawn all workers simultaneously
echo "[$_H] Starting $NUM_WORKERS workers (ports $SANDBOX_WORKER_BASE_PORT-$((SANDBOX_WORKER_BASE_PORT + NUM_WORKERS - 1)))..."
START_SPAWN=$(date +%s)

for i in $(seq 1 $NUM_WORKERS); do
    port=$((SANDBOX_WORKER_BASE_PORT + i - 1))
    result=$(start_worker_fast $i $port)
    WORKER_PIDS[$((i - 1))]="${result%%:*}"
    ACTUAL_WORKER_PORTS[$((i - 1))]=$port
done

echo "[$_H] All $NUM_WORKERS workers spawned in $(($(date +%s) - START_SPAWN))s"

# Phase 2: Retry workers that failed due to port conflicts
retry_round=0
while [ $retry_round -lt $MAX_STARTUP_RETRIES ]; do
    sleep 1

    FAILED_WORKERS=()
    for i in $(seq 1 $NUM_WORKERS); do
        idx=$((i - 1))
        worker_is_alive "${WORKER_PIDS[$idx]}" && continue
        worker_had_port_conflict $i && FAILED_WORKERS+=($i)
    done

    [ ${#FAILED_WORKERS[@]} -eq 0 ] && break

    PORT_OFFSET=$(( (retry_round + 1) * PORT_INCREMENT ))
    echo "[$_H] Retry $((retry_round + 1)): ${#FAILED_WORKERS[@]} port conflicts, offset +$PORT_OFFSET"

    for i in "${FAILED_WORKERS[@]}"; do
        idx=$((i - 1))
        new_port=$((SANDBOX_WORKER_BASE_PORT + i - 1 + PORT_OFFSET))
        result=$(start_worker_fast $i $new_port)
        WORKER_PIDS[$idx]="${result%%:*}"
        ACTUAL_WORKER_PORTS[$idx]=$new_port
    done

    retry_round=$((retry_round + 1))
done

[ $retry_round -ge $MAX_STARTUP_RETRIES ] && echo "WARNING: Max startup retries reached"

# =============================================================================
# Wait for local workers to be ready (parallel health checks)
# =============================================================================
echo "[$_H] Waiting for workers to become ready..."
TIMEOUT=180
START_TIME=$(date +%s)
declare -A WORKER_READY
HEALTH_CHECK_DIR=$(mktemp -d)

check_worker_health() {
    local worker_num=$1
    local idx=$((worker_num - 1))
    local port=${ACTUAL_WORKER_PORTS[$idx]}
    if curl -s -f --connect-timeout 2 --max-time 5 "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
        echo "ready" > "$HEALTH_CHECK_DIR/worker_${worker_num}"
    fi
}

READY_WORKERS=0
LAST_PROGRESS_TIME=0

while [ $READY_WORKERS -lt $NUM_WORKERS ]; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    if [ $ELAPSED -gt $TIMEOUT ]; then
        echo "ERROR: Timeout waiting for workers to start"
        for i in "${!WORKER_PIDS[@]}"; do
            pid=${WORKER_PIDS[$i]}
            w=$((i+1))
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Worker $w (PID $pid): Running"
                tail -20 /var/log/worker${w}.log 2>/dev/null | sed 's/^/    /' || true
            else
                echo "  Worker $w (PID $pid): Dead"
                tail -30 /var/log/worker${w}.log 2>/dev/null | sed 's/^/    /' || true
            fi
        done
        exit 1
    fi

    # Launch parallel health checks for unready workers
    check_pids=()
    checking_workers=()
    for i in $(seq 1 $NUM_WORKERS); do
        if [ "${WORKER_READY[$i]}" != "1" ]; then
            check_worker_health $i &
            check_pids+=($!)
            checking_workers+=($i)
        fi
    done

    for pid in "${check_pids[@]}"; do
        wait $pid 2>/dev/null || true
    done

    PREV_READY=$READY_WORKERS
    for i in "${checking_workers[@]}"; do
        if [ -f "$HEALTH_CHECK_DIR/worker_${i}" ]; then
            WORKER_READY[$i]=1
            READY_WORKERS=$((READY_WORKERS + 1))
            rm -f "$HEALTH_CHECK_DIR/worker_${i}"
            echo "  Worker $i (port ${ACTUAL_WORKER_PORTS[$((i-1))]}): Ready ($READY_WORKERS/$NUM_WORKERS)"
        fi
    done

    if [ $READY_WORKERS -lt $NUM_WORKERS ]; then
        if [ $((CURRENT_TIME - LAST_PROGRESS_TIME)) -ge 10 ]; then
            echo "  Progress: $READY_WORKERS/$NUM_WORKERS workers ready (${ELAPSED}s)"
            LAST_PROGRESS_TIME=$CURRENT_TIME
        fi
        [ $READY_WORKERS -eq $PREV_READY ] && sleep 1
    fi
done

echo "[$_H] All $NUM_WORKERS local workers ready!"

# =============================================================================
# Write port assignments to shared storage (multi-node only)
# =============================================================================
if [ "$NODE_COUNT" -gt 1 ]; then
    PORTS_FILE="$PORTS_REPORT_DIR/${CURRENT_NODE_SHORT}_ports.txt"
    > "$PORTS_FILE"
    for i in $(seq 1 $NUM_WORKERS); do
        echo "${i}:${ACTUAL_WORKER_PORTS[$((i-1))]}" >> "$PORTS_FILE"
    done
    echo "PORT_REPORT_COMPLETE" >> "$PORTS_FILE"
    sync
    echo "[$_H] Port assignments written to $PORTS_FILE"
fi

# =============================================================================
# Nginx setup
# =============================================================================
if [ "$IS_MASTER" = "1" ]; then
    if [ "$NODE_COUNT" -gt 1 ]; then
        # --- Multi-node: collect ports from all nodes, build cross-node upstream ---
        wait_for_port_reports

        > $UPSTREAM_FILE
        ENDPOINTS_FILE=$(mktemp)
        for node in $ALL_NODES; do
            node_short="${node%%.*}"
            port_file="$PORTS_REPORT_DIR/${node_short}_ports.txt"
            for endpoint in $(read_port_file "$node" "$port_file"); do
                echo "        server ${endpoint} max_fails=3 fail_timeout=30s;" >> $UPSTREAM_FILE
                echo "$endpoint" >> "$ENDPOINTS_FILE"
            done
        done
        echo "[$_H] Generated upstream with $(wc -l < $UPSTREAM_FILE) servers across $NODE_COUNT nodes"

        generate_nginx_config
        verify_remote_workers "$ENDPOINTS_FILE"
        rm -f "$ENDPOINTS_FILE"
    else
        # --- Single-node: upstream from local ports only ---
        > $UPSTREAM_FILE
        for i in $(seq 1 $NUM_WORKERS); do
            echo "        server 127.0.0.1:${ACTUAL_WORKER_PORTS[$((i-1))]} max_fails=3 fail_timeout=30s;" >> $UPSTREAM_FILE
        done

        generate_nginx_config
    fi

    echo "[$_H] Starting nginx on port $NGINX_PORT..."
    nginx
else
    # --- Worker node: local nginx proxy forwarding to master ---
    echo "[$_H] Starting nginx proxy to master $MASTER_NODE:$NGINX_PORT..."
    sed -e "s|\${MASTER_NODE}|${MASTER_NODE}|g" \
        -e "s|\${NGINX_PORT}|${NGINX_PORT}|g" \
        /etc/nginx/nginx-worker-proxy.conf.template > /etc/nginx/nginx.conf

    echo "Testing nginx proxy configuration..."
    if ! nginx -t; then
        echo "ERROR: nginx proxy configuration test failed"
        cat /etc/nginx/nginx.conf
        exit 1
    fi

    nginx
    echo "[$_H] Nginx proxy started: localhost:$NGINX_PORT -> $MASTER_NODE:$NGINX_PORT"
fi

# =============================================================================
# Network blocking
# =============================================================================
# ld.so.preload intercepts socket() in all NEW exec'd processes. This is safe
# for nginx/uWSGI that are already running. However, if the monitoring loop
# restarts a crashed worker, the new uWSGI process would be unable to bind its
# listening socket. We set NETWORK_BLOCKING_ACTIVE so the monitoring loop can
# emit a clear diagnostic when this happens.
NETWORK_BLOCKING_ACTIVE=0
BLOCK_NETWORK_LIB="/usr/lib/libblock_network.so"
if [ "${NEMO_SKILLS_SANDBOX_BLOCK_NETWORK:-0}" = "1" ]; then
    if [ -f "$BLOCK_NETWORK_LIB" ]; then
        echo "$BLOCK_NETWORK_LIB" > /etc/ld.so.preload
        NETWORK_BLOCKING_ACTIVE=1
        echo "[$_H] Network blocking ENABLED via ld.so.preload"
        if [ "$NODE_COUNT" -gt 1 ]; then
            echo "[$_H] NOTE: Network blocking is active in multi-node mode. If a worker"
            echo "[$_H]   crashes, the monitoring loop will be unable to restart it because"
            echo "[$_H]   ld.so.preload blocks socket() in new processes. The remaining"
            echo "[$_H]   workers will continue serving requests."
        fi
    else
        echo "[$_H] WARNING: Network blocking requested but $BLOCK_NETWORK_LIB not found"
    fi
fi

# =============================================================================
# Status summary
# =============================================================================
if [ "$IS_MASTER" = "1" ]; then
    echo "=== Sandbox ready (MASTER) ==="
    echo "  Nginx LB: http://localhost:$NGINX_PORT"
    echo "  Nodes: $NODE_COUNT | Workers/node: $NUM_WORKERS | Total: $((NODE_COUNT * NUM_WORKERS))"
    echo "  Local ports: ${ACTUAL_WORKER_PORTS[0]}-${ACTUAL_WORKER_PORTS[$((NUM_WORKERS-1))]}"
else
    echo "=== Sandbox ready (WORKER) ==="
    echo "  Proxy: localhost:$NGINX_PORT -> $MASTER_NODE:$NGINX_PORT"
    echo "  Local workers: $NUM_WORKERS (ports ${ACTUAL_WORKER_PORTS[0]}-${ACTUAL_WORKER_PORTS[$((NUM_WORKERS-1))]})"
fi
echo "  uWSGI: processes=$UWSGI_PROCESSES cheaper=${UWSGI_CHEAPER:-disabled}"

# =============================================================================
# Monitoring loop
# =============================================================================
echo "[$_H] Monitoring processes..."

if [ "$IS_MASTER" = "1" ]; then
    (
        while true; do
            sleep 60
            echo "--- [$_H] Worker Load Stats (Top 10) at $(date) ---"
            grep "upstream:" /var/log/nginx/access.log 2>/dev/null \
                | awk -F'upstream: ' '{print $2}' | awk -F' session: ' '{print $1}' \
                | sort | uniq -c | sort -nr | head -n 10 || echo "No logs yet"
            echo "--- End Stats ---"
        done
    ) &
fi

while true; do
    for idx in "${!WORKER_PIDS[@]}"; do
        pid=${WORKER_PIDS[$idx]}
        i=$((idx + 1))
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[$_H] WARNING: Worker $i (PID $pid) died — restarting..."
            if [ "$NETWORK_BLOCKING_ACTIVE" = "1" ]; then
                echo "[$_H] WARNING: Network blocking (ld.so.preload) is active. The restarted"
                echo "[$_H]   worker may fail to bind its port because socket() is blocked for"
                echo "[$_H]   new processes. Remaining workers continue serving requests."
            fi
            result=$(start_worker $i)
            WORKER_PIDS[$idx]="${result%%:*}"
            ACTUAL_WORKER_PORTS[$idx]="${result##*:}"
        fi
    done

    if ! pgrep nginx > /dev/null; then
        echo "[$_H] ERROR: Nginx died unexpectedly"
        cleanup
        exit 1
    fi

    sleep 10
done

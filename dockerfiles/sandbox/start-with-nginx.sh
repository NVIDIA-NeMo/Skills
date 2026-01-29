#!/bin/bash
# Start nginx load balancer with multiple uwsgi workers
# Simplified approach using background processes

set -e

export NUM_WORKERS=${NUM_WORKERS:-$(nproc --all)}

echo "Starting multi-worker deployment with nginx (unix sockets upstream)..."
echo "Workers: $NUM_WORKERS, Nginx port: $NGINX_PORT"

# Override nginx config for multi-worker mode (single mode uses original config)
echo "Configuring nginx for multi-worker load balancing..."


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

# Generate upstream servers configuration for nginx (using unix sockets)
echo "Generating nginx configuration..."

# Prepare socket directory
SOCKET_DIR="/tmp/uwsgi_sockets"
mkdir -p "$SOCKET_DIR"
chmod 777 "$SOCKET_DIR"

# Write upstream servers to a temp file
UPSTREAM_FILE="/tmp/upstream_servers.conf"
> $UPSTREAM_FILE  # Clear the file
for i in $(seq 1 $NUM_WORKERS); do
    SOCKET_PATH="${SOCKET_DIR}/worker${i}.sock"
    echo "        server unix:${SOCKET_PATH} max_fails=3 fail_timeout=30s;" >> $UPSTREAM_FILE
done

echo "Generated upstream servers for $NUM_WORKERS workers (unix sockets):"
cat $UPSTREAM_FILE

# Create nginx config by replacing placeholders
# First replace the NGINX_PORT, then insert the upstream servers
sed "s|\${NGINX_PORT}|${NGINX_PORT}|g" /etc/nginx/nginx.conf.template > /tmp/nginx_temp.conf

# Replace the upstream servers placeholder with the actual servers
# Use a different approach - split at the placeholder and reassemble
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

echo "Generated nginx config with upstream servers:"
echo "Nginx configuration created successfully"

# Test nginx configuration
echo "Testing nginx configuration..."
if ! nginx -t; then
    echo "ERROR: nginx configuration test failed"
    echo "Generated nginx.conf:"
    cat /etc/nginx/nginx.conf
    exit 1
fi

# Create log directory
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

# Start workers as background processes
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
    # Cleanup leftover unix sockets
    if [ -n "$SOCKET_DIR" ] && [ -d "$SOCKET_DIR" ]; then
        rm -f "$SOCKET_DIR"/worker*.sock 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGTERM SIGINT

# Function to start a single worker and return its PID
start_worker() {
    local i=$1
    SOCKET_PATH="${SOCKET_DIR}/worker${i}.sock"

    echo "Starting worker $i on socket $SOCKET_PATH..." >&2

    # Ensure old socket is removed if present
    if [ -S "$SOCKET_PATH" ]; then
        rm -f "$SOCKET_PATH"
    fi

    # Create a custom uwsgi.ini for this worker that serves HTTP over a unix socket
    cat > /tmp/worker${i}_uwsgi.ini << EOF
[uwsgi]
module = main
callable = app
processes = ${UWSGI_PROCESSES}
http-socket = ${SOCKET_PATH}
chmod-socket = 666
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

    echo "Created custom uwsgi config for worker $i (HTTP unix socket ${SOCKET_PATH})" >&2

    # Start worker with custom config
    (
        # Run uwsgi from /app in a subshell so the current directory of the main script is unaffected
        cd /app && env WORKER_NUM=$i uwsgi --ini /tmp/worker${i}_uwsgi.ini
    ) &

    local pid=$!
    echo "Worker $i started with PID $pid on socket $SOCKET_PATH" >&2
    echo $pid
}

# Start all workers simultaneously
for i in $(seq 1 $NUM_WORKERS); do
    pid=$(start_worker $i)
    WORKER_PIDS+=($pid)
done

echo "All $NUM_WORKERS workers started simultaneously - waiting for readiness..."

# Wait for workers to be ready
echo "Waiting for workers to start..."
READY_WORKERS=0
TIMEOUT=180  # Increased timeout since uwsgi takes time to start
START_TIME=$(date +%s)

# Track which workers are ready to avoid redundant checks
declare -A WORKER_READY

while [ $READY_WORKERS -lt $NUM_WORKERS ]; do
    CURRENT_TIME=$(date +%s)
    if [ $((CURRENT_TIME - START_TIME)) -gt $TIMEOUT ]; then
        echo "ERROR: Timeout waiting for workers to start"

        # Show worker status and logs
        echo "Worker status:"
        for i in "${!WORKER_PIDS[@]}"; do
            pid=${WORKER_PIDS[$i]}
            socket_path="${SOCKET_DIR}/worker$((i+1)).sock"
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Worker $((i+1)) (PID $pid): Process Running"

                # Check if socket exists
                if [ -S "$socket_path" ]; then
                    echo "    Socket $socket_path: Present"
                else
                    echo "    Socket $socket_path: Not present"
                fi

                # Show recent log output
                echo "    Recent log output:"
                tail -20 /var/log/worker$((i+1)).log 2>/dev/null | sed 's/^/      /' || echo "      No log found"
            else
                echo "  Worker $((i+1)) (PID $pid): Dead"
                echo "    Log:"
                tail -30 /var/log/worker$((i+1)).log 2>/dev/null | sed 's/^/      /' || echo "      No log found"
            fi
        done

        exit 1
    fi

    READY_WORKERS=0
    for i in $(seq 1 $NUM_WORKERS); do
        # Skip workers that are already ready
        if [ "${WORKER_READY[$i]}" = "1" ]; then
            READY_WORKERS=$((READY_WORKERS + 1))
            continue
        fi

        SOCKET_PATH="${SOCKET_DIR}/worker${i}.sock"

        # Try the health check via unix socket
        if curl -s -f --connect-timeout 2 --max-time 5 --unix-socket "$SOCKET_PATH" http://localhost/health > /dev/null 2>&1; then
            READY_WORKERS=$((READY_WORKERS + 1))
            WORKER_READY[$i]=1
            echo "  Worker $i (socket $SOCKET_PATH): Ready! ($READY_WORKERS/$NUM_WORKERS)"
        fi
    done

    # Show progress every 10 seconds
    if [ $((CURRENT_TIME % 10)) -eq 0 ] && [ $READY_WORKERS -lt $NUM_WORKERS ]; then
        echo "  Progress: $READY_WORKERS/$NUM_WORKERS workers ready (${CURRENT_TIME}s elapsed)"
    fi

    # Check less frequently to reduce CPU usage and log spam
    sleep 2
done

echo "All workers are ready!"

# Start nginx
echo "Starting nginx on port $NGINX_PORT..."
nginx

# Enable network blocking for user code execution if requested
# This MUST happen AFTER nginx/uwsgi start (they need sockets for API)
# Using /etc/ld.so.preload ensures this cannot be bypassed by user code
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

echo "=== Multi-worker deployment ready ==="
echo "Nginx load balancer: http://localhost:$NGINX_PORT"
echo "Session affinity: enabled (based on JSON session_id)"
echo "Workers: $NUM_WORKERS (unix sockets in ${SOCKET_DIR}/worker{1..$NUM_WORKERS}.sock)"
echo "Nginx status: http://localhost:$NGINX_PORT/nginx-status"
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

# Keep the container running and monitor
echo "Monitoring processes (Ctrl+C to stop)..."
monitor_load() {
    echo "Starting worker load monitor (updates every 60s)..."
    while true; do
        sleep 60
        echo "--- Worker $(hostname) Load Stats (Top 10) at $(date) ---"
        grep "upstream:" /var/log/nginx/access.log | awk -F'upstream: ' '{print $2}' | awk -F' session: ' '{print $1}' | sort | uniq -c | sort -nr | head -n 10 || echo "No logs yet"
        echo "--- End Stats ---"
    done
}
monitor_load &  # Run in background

while true; do
    # Check if any worker died
    for idx in "${!WORKER_PIDS[@]}"; do
        pid=${WORKER_PIDS[$idx]}
        i=$((idx + 1))
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "WARNING: Worker $i (PID $pid) died - restarting..."
            new_pid=$(start_worker $i)
            WORKER_PIDS[$idx]=$new_pid
        fi
    done

    # Check nginx
    if ! pgrep nginx > /dev/null; then
        echo "ERROR: Nginx died unexpectedly"
        cleanup
        exit 1
    fi

    sleep 10
done

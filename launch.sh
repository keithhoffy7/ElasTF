#!/usr/bin/env bash
# launch.sh — Elastic supervisor for ElasTF.
#
# Opens separate macOS Terminal.app windows for the controller and each worker.
# This terminal acts as the supervisor: it monitors worker PIDs and handles
# automatic restart with N-1 workers on failure.
#
# Usage:
#   chmod +x launch.sh
#   ./launch.sh

PROJECT_DIR="/Users/keithhoffmeister/Downloads/ElasTF"
HEARTBEAT_PORT=6000
NUM_WORKERS=3
STARTUP_SLEEP=15
EPOCHS=5
TMPDIR_LAUNCH="/tmp/elastf_launch"

echo "[supervisor] Killing any old ElasTF processes..."
pkill -9 -f "elas_tf.controller" 2>/dev/null
pkill -9 -f "elas_tf.worker" 2>/dev/null
lsof -ti tcp:${HEARTBEAT_PORT} | xargs kill -9 2>/dev/null
sleep 3

echo "[supervisor] Clearing old state..."
rm -rf "$PROJECT_DIR/shared/checkpoints/"* "$PROJECT_DIR/shared/config/"*
mkdir -p "$PROJECT_DIR/shared/checkpoints" "$PROJECT_DIR/shared/config"
rm -rf "$TMPDIR_LAUNCH"
mkdir -p "$TMPDIR_LAUNCH"

# --- Write the controller script ---
cat > "$TMPDIR_LAUNCH/controller.sh" <<EOF
#!/usr/bin/env bash
cd "$PROJECT_DIR"
source .venv/bin/activate
export CONFIG_DIR=shared/config
export CHECKPOINT_DIR=shared/checkpoints
export HEARTBEAT_PORT=$HEARTBEAT_PORT
echo "=== CONTROLLER ==="
echo \$\$ > "$TMPDIR_LAUNCH/controller.pid"
exec python3 -m elas_tf.controller
EOF
chmod +x "$TMPDIR_LAUNCH/controller.sh"

# --- Open controller Terminal ---
echo "[supervisor] Starting controller..."
open -a Terminal "$TMPDIR_LAUNCH/controller.sh"
sleep 3

if [ -f "$TMPDIR_LAUNCH/controller.pid" ]; then
    CTRL_PID=$(cat "$TMPDIR_LAUNCH/controller.pid")
    echo "[supervisor] Controller started (pid=$CTRL_PID)."
else
    echo "[supervisor] Controller started."
fi

# --- Trap Ctrl-C ---
cleanup() {
    echo ""
    echo "[supervisor] Shutting down..."
    pkill -9 -f "elas_tf.controller" 2>/dev/null
    pkill -9 -f "elas_tf.worker" 2>/dev/null
    rm -rf "$TMPDIR_LAUNCH"
    echo "[supervisor] Done."
    exit 0
}
trap cleanup INT TERM

ACTIVE_WORKERS=$NUM_WORKERS
GENERATION=0

while true; do
    GENERATION=$((GENERATION + 1))
    BASE_PORT=$((30000 + RANDOM % 10000))

    if [ $GENERATION -eq 1 ]; then
        SLEEP_TIME=$STARTUP_SLEEP
    else
        SLEEP_TIME=20
    fi

    echo ""
    echo "============================================================"
    echo "[supervisor] Generation $GENERATION: starting $ACTIVE_WORKERS workers"
    echo "[supervisor] TF ports: $BASE_PORT - $((BASE_PORT + ACTIVE_WORKERS - 1))"
    echo "============================================================"

    # Write and launch a worker script for each worker.
    for i in $(seq 0 $((ACTIVE_WORKERS - 1))); do
        TF_PORT=$((BASE_PORT + i))
        PIDFILE="$TMPDIR_LAUNCH/worker_${i}.pid"
        SCRIPT="$TMPDIR_LAUNCH/worker_${i}.sh"
        rm -f "$PIDFILE"

        cat > "$SCRIPT" <<EOF
#!/usr/bin/env bash
cd "$PROJECT_DIR"
source .venv/bin/activate
export WORKER_ID=$i
export CONTROLLER_HOST=localhost
export WORKER_HOST=localhost
export HEARTBEAT_PORT=$HEARTBEAT_PORT
export CONFIG_DIR=shared/config
export CHECKPOINT_DIR=shared/checkpoints
export TF_PORT=$TF_PORT
export STARTUP_SLEEP_SECS=$SLEEP_TIME
export EPOCHS=$EPOCHS
echo "=== WORKER $i (port $TF_PORT) ==="
echo \$\$ > "$PIDFILE"
exec python3 -m elas_tf.worker
EOF
        chmod +x "$SCRIPT"

        open -a Terminal "$SCRIPT"
        echo "[supervisor] Opened Terminal for worker $i (port $TF_PORT)"
    done

    # Wait for PID files.
    echo "[supervisor] Waiting for worker PIDs..."
    sleep 8

    WORKER_PIDS=()
    for i in $(seq 0 $((ACTIVE_WORKERS - 1))); do
        PIDFILE="$TMPDIR_LAUNCH/worker_${i}.pid"
        if [ -f "$PIDFILE" ]; then
            PID=$(cat "$PIDFILE")
            WORKER_PIDS+=($PID)
            echo "[supervisor] Worker $i pid=$PID"
        else
            echo "[supervisor] WARNING: Could not get PID for worker $i"
        fi
    done

    echo ""
    echo "[supervisor] $ACTIVE_WORKERS workers running."
    if [ ${#WORKER_PIDS[@]} -gt 0 ]; then
        echo "[supervisor] To test elastic recovery, open another terminal and run:"
        echo ""
        echo "    kill ${WORKER_PIDS[0]}"
        echo ""
    fi

    # Wait for ANY worker to exit.
    EXITED_PID=""
    while [ -z "$EXITED_PID" ]; do
        for pid in "${WORKER_PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                EXITED_PID=$pid
                break
            fi
        done
        sleep 1
    done

    echo ""
    echo "[supervisor] Worker (pid=$EXITED_PID) has exited."
    echo "[supervisor] Waiting for remaining workers to crash..."
    sleep 8

    # Kill any remaining workers.
    for pid in "${WORKER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    sleep 2

    # If generation > 1 (restarted workers finished), we're done.
    if [ $GENERATION -gt 1 ]; then
        echo ""
        echo "============================================================"
        echo "[supervisor] Training completed after elastic recovery!"
        echo "============================================================"
        break
    fi

    # Generation 1 failure — restart with fewer workers.
    ACTIVE_WORKERS=$((ACTIVE_WORKERS - 1))
    if [ $ACTIVE_WORKERS -lt 1 ]; then
        echo "[supervisor] No workers left. Exiting."
        break
    fi

    echo ""
    echo "============================================================"
    echo "[supervisor] ELASTIC RECOVERY: restarting with $ACTIVE_WORKERS workers"
    echo "[supervisor] Waiting 20s for controller to clear stale heartbeats..."
    echo "============================================================"
    sleep 20
    echo "[supervisor] Restarting from latest checkpoint..."
done

# Clean up.
pkill -9 -f "elas_tf.controller" 2>/dev/null
rm -rf "$TMPDIR_LAUNCH"
echo "[supervisor] Controller stopped. All done."

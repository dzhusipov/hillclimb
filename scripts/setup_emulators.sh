#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="$PROJECT_DIR/docker"
LOG_DIR="$PROJECT_DIR/logs/setup_screenshots"

# Load .env
set -a
source "$DOCKER_DIR/.env"
set +a

ADB_PORT_BASE="${ADB_PORT_BASE:-5555}"
NUM_EMULATORS="${NUM_EMULATORS:-2}"

info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()   { echo -e "\033[1;31m[ERROR]\033[0m $*"; }

# --- Step 1: Kernel module ---
info "Checking binder_linux kernel module..."
if lsmod | grep -q binder_linux; then
    ok "binder_linux already loaded"
else
    info "Loading binder_linux module..."
    sudo modprobe binder_linux devices=binder,hwbinder,vndbinder
    if lsmod | grep -q binder_linux; then
        ok "binder_linux loaded successfully"
    else
        err "Failed to load binder_linux. Check kernel module availability."
        exit 1
    fi
fi

# Persist across reboots
if [ ! -f /etc/modules-load.d/redroid.conf ]; then
    info "Adding binder_linux to auto-load on boot..."
    echo "binder_linux" | sudo tee /etc/modules-load.d/redroid.conf > /dev/null
    ok "Created /etc/modules-load.d/redroid.conf"
fi

if [ ! -f /etc/modprobe.d/redroid.conf ]; then
    info "Setting binder_linux module parameters..."
    echo "options binder_linux devices=binder,hwbinder,vndbinder" | sudo tee /etc/modprobe.d/redroid.conf > /dev/null
    ok "Created /etc/modprobe.d/redroid.conf"
fi

# --- Step 2: Docker Compose ---
info "Starting ReDroid containers ($NUM_EMULATORS emulators)..."

COMPOSE_ARGS=""
if [ "$NUM_EMULATORS" -ge 4 ]; then
    COMPOSE_ARGS="--profile scale-4"
fi
if [ "$NUM_EMULATORS" -ge 5 ]; then
    COMPOSE_ARGS="--profile scale-8"
fi

cd "$DOCKER_DIR"
docker compose $COMPOSE_ARGS up -d
ok "Docker compose started"

# --- Step 3: Wait for emulators to boot ---
info "Waiting for emulators to boot..."

wait_for_boot() {
    local port=$1
    local name=$2
    local timeout=60
    local elapsed=0

    # Wait for ADB connection
    while [ $elapsed -lt $timeout ]; do
        if adb connect "localhost:$port" 2>/dev/null | grep -q "connected"; then
            break
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done

    if [ $elapsed -ge $timeout ]; then
        err "$name: ADB connection timeout after ${timeout}s"
        return 1
    fi

    # Wait for boot_completed
    elapsed=0
    while [ $elapsed -lt $timeout ]; do
        local boot_status
        boot_status=$(adb -s "localhost:$port" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r\n')
        if [ "$boot_status" = "1" ]; then
            ok "$name (port $port): booted"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done

    err "$name: boot timeout after ${timeout}s"
    return 1
}

for i in $(seq 0 $((NUM_EMULATORS - 1))); do
    port=$((ADB_PORT_BASE + i))
    wait_for_boot "$port" "hcr2-$i" &
done
wait
info "All emulators checked"

# --- Step 4: Configure emulators ---
info "Configuring emulators..."

configure_emulator() {
    local port=$1
    local name=$2

    # Disable animations
    adb -s "localhost:$port" shell settings put global window_animation_scale 0 2>/dev/null
    adb -s "localhost:$port" shell settings put global transition_animation_scale 0 2>/dev/null
    adb -s "localhost:$port" shell settings put global animator_duration_scale 0 2>/dev/null

    # Set resolution
    adb -s "localhost:$port" shell wm size "${WIDTH:-480}x${HEIGHT:-800}" 2>/dev/null

    ok "$name: configured (animations off, resolution ${WIDTH:-480}x${HEIGHT:-800})"
}

for i in $(seq 0 $((NUM_EMULATORS - 1))); do
    port=$((ADB_PORT_BASE + i))
    configure_emulator "$port" "hcr2-$i"
done

# --- Step 5: Install APK (if available) ---
APK_DIR="$DOCKER_DIR/apk"
APK_FILES=("$APK_DIR"/*.apk)

if [ -f "${APK_FILES[0]}" ]; then
    info "Found APK files, installing HCR2..."
    for i in $(seq 0 $((NUM_EMULATORS - 1))); do
        port=$((ADB_PORT_BASE + i))
        if adb -s "localhost:$port" install-multiple "$APK_DIR"/*.apk 2>/dev/null; then
            ok "hcr2-$i: APK installed"
            # Launch game
            adb -s "localhost:$port" shell am start -n com.fingersoft.hillclimb/com.fingersoft.hillclimb.MainActivity 2>/dev/null || true
        else
            warn "hcr2-$i: APK install failed (may need ARM translation)"
        fi
    done
else
    warn "No APK files found in $APK_DIR — skipping install"
    warn "Place HCR2 split APKs in docker/apk/ and re-run, or use: ./scripts/manage.sh install-apk"
fi

# --- Step 6: Verification ---
info "Taking verification screenshots..."
mkdir -p "$LOG_DIR"

for i in $(seq 0 $((NUM_EMULATORS - 1))); do
    port=$((ADB_PORT_BASE + i))
    screenshot="$LOG_DIR/hcr2-${i}.png"
    if adb -s "localhost:$port" exec-out screencap -p > "$screenshot" 2>/dev/null; then
        ok "hcr2-$i: screenshot saved to $screenshot"
    else
        warn "hcr2-$i: screenshot failed"
    fi
done

# --- Summary ---
echo ""
info "=== Setup Summary ==="
for i in $(seq 0 $((NUM_EMULATORS - 1))); do
    port=$((ADB_PORT_BASE + i))
    boot_status=$(adb -s "localhost:$port" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r\n')
    if [ "$boot_status" = "1" ]; then
        ok "hcr2-$i (port $port): READY"
    else
        err "hcr2-$i (port $port): NOT READY"
    fi
done

echo ""
info "ws-scrcpy: http://localhost:8000"
info "Screenshots: $LOG_DIR/"

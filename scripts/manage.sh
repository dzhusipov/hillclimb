#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="$PROJECT_DIR/docker"

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

get_port() {
    local id=$1
    echo $((ADB_PORT_BASE + id))
}

compose_profiles() {
    local n=${1:-$NUM_EMULATORS}
    local args=""
    if [ "$n" -ge 4 ]; then
        args="--profile scale-4"
    fi
    if [ "$n" -ge 5 ]; then
        args="--profile scale-8"
    fi
    echo "$args"
}

usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [args]

Commands:
  start [N]          Start N emulators (default: $NUM_EMULATORS from .env)
  stop               Stop all containers
  restart <id>       Restart a specific emulator (0-7)
  status             Show status of all emulators
  install-apk        Install HCR2 APK on all running emulators
  screenshot <id>    Take screenshot of emulator <id>
  shell <id>         Open ADB shell to emulator <id>

Examples:
  $(basename "$0") start          # Start $NUM_EMULATORS emulators
  $(basename "$0") start 4        # Start 4 emulators
  $(basename "$0") status         # Show status table
  $(basename "$0") restart 0      # Restart hcr2-0
  $(basename "$0") screenshot 1   # Screenshot hcr2-1
  $(basename "$0") shell 0        # ADB shell into hcr2-0
EOF
}

cmd_start() {
    local n=${1:-$NUM_EMULATORS}
    info "Starting $n emulators..."

    # Check binder_linux
    if ! lsmod | grep -q binder_linux; then
        warn "binder_linux not loaded. Run setup_emulators.sh first, or:"
        warn "  sudo modprobe binder_linux devices=binder,hwbinder,vndbinder"
        exit 1
    fi

    local profiles
    profiles=$(compose_profiles "$n")

    # Build service list: only start the ones we need
    local services="ws-scrcpy"
    for i in $(seq 0 $((n - 1))); do
        services="$services hcr2-$i"
    done

    cd "$DOCKER_DIR"
    docker compose $profiles up -d $services
    ok "Started $n emulators + ws-scrcpy"

    # Wait for boot
    info "Waiting for boot..."
    for i in $(seq 0 $((n - 1))); do
        local port
        port=$(get_port "$i")
        local timeout=60
        local elapsed=0

        # Connect ADB
        while [ $elapsed -lt $timeout ]; do
            if adb connect "localhost:$port" 2>/dev/null | grep -q "connected"; then
                break
            fi
            sleep 2
            elapsed=$((elapsed + 2))
        done

        # Check boot
        elapsed=0
        while [ $elapsed -lt $timeout ]; do
            local status
            status=$(adb -s "localhost:$port" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r\n')
            if [ "$status" = "1" ]; then
                ok "hcr2-$i (port $port): ready"
                break
            fi
            sleep 2
            elapsed=$((elapsed + 2))
        done

        if [ $elapsed -ge $timeout ]; then
            warn "hcr2-$i: boot timeout"
        fi
    done
}

cmd_stop() {
    info "Stopping all containers..."
    cd "$DOCKER_DIR"
    docker compose --profile scale-4 --profile scale-8 down
    ok "All containers stopped"
}

cmd_restart() {
    local id=${1:?Error: emulator id required (0-7)}
    local port
    port=$(get_port "$id")

    info "Restarting hcr2-$id..."
    cd "$DOCKER_DIR"

    local profiles
    profiles=$(compose_profiles 8)  # include all profiles to find the service
    docker compose $profiles restart "hcr2-$id"

    # Reconnect ADB
    info "Waiting for hcr2-$id to boot..."
    local timeout=60
    local elapsed=0
    sleep 5

    while [ $elapsed -lt $timeout ]; do
        adb connect "localhost:$port" 2>/dev/null || true
        local status
        status=$(adb -s "localhost:$port" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r\n')
        if [ "$status" = "1" ]; then
            ok "hcr2-$id (port $port): ready"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done

    warn "hcr2-$id: boot timeout after ${timeout}s"
}

cmd_status() {
    printf "%-12s %-10s %-12s %-15s\n" "CONTAINER" "PORT" "DOCKER" "BOOT"
    printf "%-12s %-10s %-12s %-15s\n" "---------" "----" "------" "----"

    for i in $(seq 0 7); do
        local port
        port=$(get_port "$i")
        local name="hcr2-$i"

        # Check if container is running
        local docker_status
        docker_status=$(docker inspect --format='{{.State.Status}}' "$name" 2>/dev/null | tr -d '\n' || true)
        [ -z "$docker_status" ] && docker_status="not found"

        local boot_status="-"
        if [ "$docker_status" = "running" ]; then
            adb connect "localhost:$port" > /dev/null 2>&1 || true
            boot_status=$(adb -s "localhost:$port" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r\n')
            if [ "$boot_status" = "1" ]; then
                boot_status="ready"
            else
                boot_status="booting"
            fi
        fi

        printf "%-12s %-10s %-12s %-15s\n" "$name" "$port" "$docker_status" "$boot_status"
    done

    echo ""
    # ws-scrcpy status
    local scrcpy_status
    scrcpy_status=$(docker inspect --format='{{.State.Status}}' ws-scrcpy 2>/dev/null || echo "not found")
    printf "%-12s %-10s %-12s\n" "ws-scrcpy" "8000" "$scrcpy_status"
}

cmd_install_apk() {
    local apk_dir="$DOCKER_DIR/apk"
    local apk_files=("$apk_dir"/*.apk)

    if [ ! -f "${apk_files[0]}" ]; then
        err "No APK files found in $apk_dir"
        err "Place HCR2 split APKs there first. See docker/apk/README.md"
        exit 1
    fi

    info "Installing APK on all running emulators..."
    for i in $(seq 0 7); do
        local port
        port=$(get_port "$i")
        local name="hcr2-$i"

        # Check if container is running
        local docker_status
        docker_status=$(docker inspect --format='{{.State.Status}}' "$name" 2>/dev/null | tr -d '\n' || true)
        [ -z "$docker_status" ] && docker_status="not found"
        if [ "$docker_status" != "running" ]; then
            continue
        fi

        adb connect "localhost:$port" > /dev/null 2>&1 || true
        if adb -s "localhost:$port" install-multiple "$apk_dir"/*.apk 2>/dev/null; then
            ok "$name: APK installed"
        else
            warn "$name: APK install failed (may need ARM translation layer)"
        fi
    done
}

cmd_screenshot() {
    local id=${1:?Error: emulator id required (0-7)}
    local port
    port=$(get_port "$id")
    local out_dir="$PROJECT_DIR/logs"
    mkdir -p "$out_dir"

    local screenshot="$out_dir/screenshot-hcr2-${id}.png"

    adb connect "localhost:$port" > /dev/null 2>&1 || true
    if adb -s "localhost:$port" exec-out screencap -p > "$screenshot" 2>/dev/null; then
        ok "Screenshot saved: $screenshot"
    else
        err "Failed to take screenshot of hcr2-$id"
        exit 1
    fi
}

cmd_shell() {
    local id=${1:?Error: emulator id required (0-7)}
    local port
    port=$(get_port "$id")

    adb connect "localhost:$port" > /dev/null 2>&1 || true
    exec adb -s "localhost:$port" shell
}

# --- Main ---
if [ $# -lt 1 ]; then
    usage
    exit 1
fi

command="$1"
shift

case "$command" in
    start)       cmd_start "$@" ;;
    stop)        cmd_stop ;;
    restart)     cmd_restart "$@" ;;
    status)      cmd_status ;;
    install-apk) cmd_install_apk ;;
    screenshot)  cmd_screenshot "$@" ;;
    shell)       cmd_shell "$@" ;;
    -h|--help|help) usage ;;
    *)
        err "Unknown command: $command"
        usage
        exit 1
        ;;
esac

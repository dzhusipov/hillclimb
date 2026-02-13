#!/usr/bin/env bash
# Последовательный запуск Redroid-эмуляторов с проверкой здоровья.
# Предотвращает одновременную загрузку binder — причину kernel panic.
#
# Использование:
#   ./start-emulators.sh              # запуск по умолчанию (2 эмулятора)
#   ./start-emulators.sh 4            # запуск 4 эмуляторов
#   ./start-emulators.sh 8            # запуск всех 8
#   ./start-emulators.sh 1            # запуск только hcr2-0 (тест)
#   ./start-emulators.sh stop         # остановить все эмуляторы
#   ./start-emulators.sh status       # показать статус

set -euo pipefail

COMPOSE_DIR="$(cd "$(dirname "$0")/../docker" && pwd)"
COMPOSE_FILE="$COMPOSE_DIR/docker-compose.yml"

# Задержка между запуском контейнеров (секунды)
DELAY_BETWEEN=${DELAY_BETWEEN:-15}
# Таймаут ожидания healthy (секунды)
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-120}

ALL_EMUS=(hcr2-0 hcr2-1 hcr2-2 hcr2-3 hcr2-4 hcr2-5 hcr2-6 hcr2-7)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

get_health() {
    docker inspect --format='{{.State.Health.Status}}' "$1" 2>/dev/null || echo "not_found"
}

wait_healthy() {
    local name="$1"
    local elapsed=0
    log "Жду healthy для $name (таймаут ${HEALTH_TIMEOUT}с)..."
    while [ $elapsed -lt "$HEALTH_TIMEOUT" ]; do
        local status
        status=$(get_health "$name")
        case "$status" in
            healthy)
                log "$name: healthy (${elapsed}с)"
                return 0
                ;;
            unhealthy)
                log "$name: unhealthy — возможно проблема, но продолжаю"
                return 1
                ;;
            not_found)
                log "$name: контейнер не найден"
                return 1
                ;;
        esac
        sleep 5
        elapsed=$((elapsed + 5))
    done
    log "$name: таймаут ожидания healthy (${HEALTH_TIMEOUT}с)"
    return 1
}

check_binder_module() {
    if /usr/sbin/lsmod | grep binder_linux > /dev/null 2>&1; then
        log "binder_linux уже загружен"
    else
        log "ОШИБКА: модуль binder_linux не загружен"
        log "Выполни: sudo modprobe binder_linux"
        exit 1
    fi
}

cmd_start() {
    local count=${1:-2}

    if [ "$count" -lt 1 ] || [ "$count" -gt 8 ]; then
        echo "Количество эмуляторов: 1-8"
        exit 1
    fi

    check_binder_module

    # Определяем compose profile
    local profile_args=""
    if [ "$count" -gt 4 ]; then
        profile_args="--profile scale-8"
    elif [ "$count" -gt 2 ]; then
        profile_args="--profile scale-4"
    fi

    # Сначала поднимаем сервисную инфраструктуру (ws-scrcpy, dashboard)
    log "Запускаю ws-scrcpy и dashboard..."
    docker compose -f "$COMPOSE_FILE" $profile_args up -d ws-scrcpy dashboard 2>/dev/null || true

    # Последовательный запуск эмуляторов
    local started=0
    for emu in "${ALL_EMUS[@]}"; do
        if [ $started -ge "$count" ]; then
            break
        fi

        log "--- Запуск $emu ($((started + 1))/$count) ---"
        docker compose -f "$COMPOSE_FILE" $profile_args up -d "$emu"

        # Ждём healthy (кроме последнего — для него не ждём)
        if [ $((started + 1)) -lt "$count" ]; then
            if wait_healthy "$emu"; then
                log "Пауза ${DELAY_BETWEEN}с перед следующим контейнером..."
                sleep "$DELAY_BETWEEN"
            else
                log "ВНИМАНИЕ: $emu не стал healthy, но продолжаю"
                sleep "$DELAY_BETWEEN"
            fi
        else
            # Последний контейнер — тоже ждём, чтобы убедиться
            wait_healthy "$emu" || log "ВНИМАНИЕ: $emu не стал healthy"
        fi

        started=$((started + 1))
    done

    log "=== Запущено эмуляторов: $started ==="
    cmd_status
}

cmd_stop() {
    log "Останавливаю все Redroid контейнеры..."
    for emu in "${ALL_EMUS[@]}"; do
        if docker ps -q --filter "name=$emu" | grep -q .; then
            log "Останавливаю $emu..."
            docker stop -t 30 "$emu" 2>/dev/null || true
        fi
    done
    log "Все эмуляторы остановлены"
}

cmd_status() {
    echo ""
    echo "=== Статус эмуляторов ==="
    for emu in "${ALL_EMUS[@]}"; do
        local running health adb_ok
        running=$(docker inspect --format='{{.State.Running}}' "$emu" 2>/dev/null || echo "false")
        health=$(get_health "$emu")

        if [ "$running" = "true" ]; then
            # Проверяем ADB
            local port=$((5555 + ${emu##hcr2-}))
            if timeout 3 adb connect "localhost:$port" 2>/dev/null | grep -q "connected"; then
                adb_ok="ADB OK"
            else
                adb_ok="ADB ?"
            fi
            printf "  %-8s  running  health=%-10s  %s\n" "$emu" "$health" "$adb_ok"
        else
            printf "  %-8s  stopped\n" "$emu"
        fi
    done
    echo ""
}

# --- Main ---
case "${1:-}" in
    stop)
        cmd_stop
        ;;
    status)
        cmd_status
        ;;
    [1-8])
        cmd_start "$1"
        ;;
    "")
        cmd_start 2
        ;;
    *)
        echo "Использование: $0 [1-8|stop|status]"
        echo ""
        echo "  $0        — запустить 2 эмулятора (по умолчанию)"
        echo "  $0 4      — запустить 4 эмулятора"
        echo "  $0 8      — запустить все 8"
        echo "  $0 stop   — остановить все"
        echo "  $0 status — показать статус"
        exit 1
        ;;
esac

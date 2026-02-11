# Hill Climb Racing 2 AI Agent

## Что это
AI-агент, который играет в Hill Climb Racing 2 на параллельных Android-эмуляторах (ReDroid в Docker). Читает экран через ADB screencap, анализирует состояние через OpenCV, принимает решения (rule-based или PPO RL), и отправляет команды через ADB.

## Платформы
- **Обучение (NAS):** Ubuntu LTS, Ryzen 5 5600, 80GB DDR4, RTX 3090 24GB VRAM
- **Разработка:** macOS, Apple Silicon M2 Max, 32GB
- **Эмуляторы:** ReDroid 14 в Docker, 800x480 landscape, software rendering

## Архитектура
```
┌── Docker ──────────────────────────────────────┐
│  ReDroid #0..#7 (HCR2) ← ADB :5555-:5562      │
│  ws-scrcpy (:8100) — live просмотр             │
│  hcr2-dashboard (:8150) — snapshot polling     │
└────────────────────────────────────────────────┘
         │ screencap (PNG, ~250ms, timeout 10s)
         ▼
┌── Conda: hillclimb ───────────────────────────┐
│  capture.py    — ADB screencap через adbutils  │
│  vision.py     — CV: 9 states, dials, OCR      │
│  controller.py — ADB input: gas/brake/tap      │
│  navigator.py  — state machine + CAPTCHA/OFFLINE│
│  env.py        — Gymnasium HCR2Env + watchdog  │
│  train.py      — PPO SubprocVecEnv (8 envs)   │
└────────────────────────────────────────────────┘
```

## Conda-окружение
```bash
conda activate hillclimb
```
Python 3.11, PyTorch 2.5 (CUDA 12.4), OpenCV, stable-baselines3, adbutils, FastAPI.

## Запуск тестов
```bash
conda activate hillclimb
python -m pytest tests/ -v
```

## Docker-инфраструктура

### Эмуляторы
```bash
cd docker/
docker compose up -d                    # 2 эмулятора (default)
docker compose --profile scale-4 up -d  # 4 эмулятора
docker compose --profile scale-8 up -d  # 8 эмуляторов
```

### Порты
| Сервис | Порт | Описание |
|--------|------|----------|
| hcr2-0..7 | 5555-5562 | ADB |
| ws-scrcpy | 8100 | Браузерный просмотр эмуляторов |
| hcr2-dashboard | 8150 | Веб-дашборд мониторинга |

### Блокировка интернета (обязательно!)
HCR2 использует серверную валидацию. Эмуляторы ДОЛЖНЫ работать офлайн,
иначе сработает CHEAT DETECTED при переносе сейва.
```bash
SUBNET=$(docker network inspect docker_hcr2-net -f '{{range .IPAM.Config}}{{.Subnet}}{{end}}')
sudo iptables -I DOCKER-USER 1 -m conntrack --ctstate ESTABLISHED,RELATED -j RETURN
sudo iptables -I DOCKER-USER 2 -s $SUBNET -d $SUBNET -j RETURN
sudo iptables -I DOCKER-USER 3 -s $SUBNET -j DROP
```
Правила в DOCKER-USER действуют только на подсеть docker_hcr2-net.
Другие контейнеры (Jellyfin, Frigate и т.д.) не затрагиваются.
Правила не переживают ребут — добавить в автозагрузку при необходимости.

### Перенос прогресса HCR2 с физического устройства
```bash
# На Mac (где подключён телефон):
adb backup -f hcr2_save.ab -noapk com.fingersoft.hcr2
scp hcr2_save.ab dasm@NAS-IP:~/develop/hillclimb/

# На NAS — извлечь и запушить на эмуляторы:
python3 -c "
import zlib
with open('hcr2_save.ab','rb') as f:
    [f.readline() for _ in range(4)]
    open('hcr2_save.tar','wb').write(zlib.decompress(f.read()))
"
mkdir -p /tmp/hcr2_restore && tar xf hcr2_save.tar -C /tmp/hcr2_restore

# Для каждого эмулятора (adb root уже включён в ReDroid):
adb -s localhost:5555 root
adb -s localhost:5555 push gamestatus.bin /data/data/com.fingersoft.hcr2/files/
adb -s localhost:5555 push gamestatus.bak /data/data/com.fingersoft.hcr2/files/
# Исправить права:
APP_UID=$(adb -s localhost:5555 shell "stat -c '%u' /data/data/com.fingersoft.hcr2/")
adb -s localhost:5555 shell "chown $APP_UID:$APP_UID /data/data/com.fingersoft.hcr2/files/gamestatus.*"
```

### Запуск HCR2 на эмуляторе
```bash
adb -s localhost:5555 shell am start -n com.fingersoft.hcr2/.AppActivity
adb -s localhost:5555 shell am force-stop com.fingersoft.hcr2
```
**Важно:** Activity класс = `.AppActivity` (НЕ `.game.MainActivity`).

## Основные команды
```bash
# Тест захвата экрана
python -m hillclimb.capture --serial localhost:5555 --benchmark 10

# Тест контроллера
python -m hillclimb.controller --serial localhost:5555 --test

# Rule-based агент
python -m hillclimb.game_loop --agent rules
python -m hillclimb.game_loop --agent rules --episodes 5 --headless

# Обучение RL агента (8 параллельных эмуляторов)
python -m hillclimb.train --timesteps 100000 --num-envs 8

# Обучение на ночь (nohup, лог в файл)
nohup python -u -m hillclimb.train --timesteps 500000 --num-envs 8 > logs/train_run.log 2>&1 &

# Продолжить обучение с чекпоинта
python -m hillclimb.train --timesteps 500000 --num-envs 8 --resume models/ppo_hillclimb

# Мониторинг обучения
tail -f logs/train_run.log
grep "^  EP" logs/train_run.log | tail -20

# Запуск RL агента
python -m hillclimb.game_loop --agent rl
```

## Структура проекта
```
hillclimb/
├── docker/
│   ├── docker-compose.yml    — ReDroid + ws-scrcpy + dashboard
│   ├── Dockerfile.dashboard  — образ для веб-дашборда
│   ├── .env                  — конфигурация эмуляторов
│   └── apk/                  — APK файлы HCR2 (gitignored)
├── web/
│   ├── server.py             — FastAPI дашборд (:8150) + /snapshot endpoint
│   ├── emulator.py           — управление эмуляторами (Docker + ADB)
│   ├── streamer.py           — MJPEG стриминг (legacy, для ws-scrcpy)
│   ├── templates/dashboard.html
│   └── static/{app.js, style.css}  — snapshot polling (800мс)
├── hillclimb/
│   ├── config.py             — координаты кнопок, ROI, пороги (800x480)
│   ├── capture.py            — ADB screencap через adbutils
│   ├── vision.py             — CV: 9 states, dials, template OCR
│   ├── controller.py         — ADB input: gas/brake/tap через adbutils
│   ├── navigator.py          — state machine навигация (9 состояний)
│   ├── agent_rules.py        — rule-based baseline агент
│   ├── agent_rl.py           — RL агент (обёртка над PPO)
│   ├── env.py                — Gymnasium environment
│   ├── game_loop.py          — основной цикл capture→CV→agent→input
│   ├── logger.py             — CSV лог + PNG кадры
│   ├── calibrate.py          — калибровка ROI
│   └── train.py              — обучение PPO
├── scripts/
│   └── manage.sh             — start/stop/restart/install-apk
├── tests/
│   └── test_vision.py        — тесты CV модуля
├── models/                   — RL модели
├── logs/                     — логи
├── templates/                — шаблоны для template matching OCR
├── environment.yml           — conda (PyTorch CUDA 12.4)
├── requirements.txt          — pip dependencies
├── PLAN.md                   — план рефакторинга с прогрессом
└── config.json               — пользовательский конфиг (gitignored)
```

## Game States (9 states)
```
UNKNOWN → MAIN_MENU → VEHICLE_SELECT → RACING →
DRIVER_DOWN → TOUCH_TO_CONTINUE → RESULTS → (retry) → VEHICLE_SELECT
                                           ↗
DOUBLE_COINS_POPUP → (skip) → RACING
CAPTCHA → (handle) → continue
```

## Navigator State Machine
| State | Action | Wait | Expect |
|-------|--------|------|--------|
| MAIN_MENU | tap race_button | 2s | VEHICLE_SELECT |
| VEHICLE_SELECT | tap start_button + dismiss_popups | 3.5s | RACING |
| DOUBLE_COINS_POPUP | tap skip_button | 2s | RACING |
| DRIVER_DOWN | tap center + BACK (skip second chance) | 0.8s | TOUCH_TO_CONTINUE |
| TOUCH_TO_CONTINUE | tap center_screen | 1.5s | RESULTS |
| RESULTS | read OCR → tap retry | 2s | VEHICLE_SELECT |
| CAPTCHA | _solve_captcha (3-step) | varies | any |
| UNKNOWN | BACK + tap center | 1s | retry |

Stuck detection: same state 3 cycles → fallback tap center.
Portrait frame detection: h > w → not in game → relaunch.

### CAPTCHA / OFFLINE обработка
Классификатор: `overall_V < 75` + dark top (>70%) + dark edges (>60%) + НЕТ RPM-циферблата.
Обработчик `_solve_captcha` (3 шага, макс 2 перезапуска):
1. BACK — скипает OFFLINE popup
2. ADVENTURE tap (155, 25) — закрывает OFFLINE popup
3. HOME + `_relaunch_game()` — крайняя мера (настоящая CAPTCHA)

`_relaunch_game()`: force-stop → start → 5s → GOT IT (500,202) → 2× ADVENTURE tap.

## Actions
Discrete(3): `0=nothing, 1=gas, 2=brake`

## Ключевые технические заметки

### ReDroid
- Образ: `redroid/redroid:14.0.0-latest`, 480x800 portrait (landscape в игре = 800x480)
- GPU: software rendering (guest mode), RTX 3090 только для PyTorch
- ADB root по дефолту
- iptables внутри контейнера НЕ работает (нет kernel модуля)
- Airplane mode НЕ блокирует Docker сеть

### HCR2
- Package: `com.fingersoft.hcr2`, Activity: `.AppActivity`
- Сейв: `/data/data/com.fingersoft.hcr2/files/gamestatus.{bin,bak,dat}`
- Формат сейва: зашифрованный бинарный, Cocos2d-x, не документирован
- `allowBackup=true` в манифесте — adb backup работает
- CHEAT DETECTED при переносе сейва + доступ к интернету
- Офлайн режим (интернет заблокирован) — сейв принимается без проблем
- Adventure mode требует ~3500 звёзд — нужен перенос прогресса или грайнд

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

# Исключить эмулятор из обучения (например, для экспериментов)
python -m hillclimb.train --timesteps 500000 --num-envs 8 --skip-envs 0
python -m hillclimb.train --timesteps 500000 --num-envs 8 --skip-envs 0,3

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
| MAIN_MENU | 2× ADVENTURE tap + RACE tap | 3s | VEHICLE_SELECT |
| VEHICLE_SELECT | tap START (730,445); stuck: BACK + swipe right 5× | 4s | RACING |
| DOUBLE_COINS_POPUP | tap skip_button | 2s | RACING |
| DRIVER_DOWN | tap safe area (50,50) — не RESPAWN | 3s | TOUCH_TO_CONTINUE |
| TOUCH_TO_CONTINUE | tap bottom (400,460) — не центр панели | 3s | RESULTS |
| RESULTS | wait 1.5s анимация → двойной тап RETRY (87,448) | 4s | VEHICLE_SELECT |
| CAPTCHA | _solve_captcha (3-step) | varies | any |
| UNKNOWN | wait → BACK → tap center → ADVENTURE → relaunch (escalation) | varies | retry |

Stuck detection: same state 3 cycles → fallback (RESULTS: tap RETRY, остальные: tap center).
Portrait frame detection: h > w → not in game → relaunch.
Mid-race interrupt: CAPTCHA, DOUBLE_COINS, MAIN_MENU, VEHICLE_SELECT → ensure_racing(30s).

### CAPTCHA / OFFLINE обработка
**CAPTCHA** ("ARE YOU A ROBOT?"): `overall_V < 50` + dark top (>70%) + dark edges (>60%) + НЕТ RPM-циферблата + tab_bright < 0.03.
**OFFLINE popup** ("Connection to game server failed!"): early detection в step 0b — `overall_v < 100 + tab_bright > 0.05 + dark_center > 50%` → MAIN_MENU.
Может появиться на ЛЮБОМ табе (ADVENTURE, CUPS и т.д.).

Обработчик `_solve_captcha` (3 шага, макс 2 перезапуска):
1. BACK — скипает OFFLINE popup
2. ADVENTURE tap (155, 25) — закрывает OFFLINE popup
3. HOME + `_relaunch_game()` — крайняя мера (настоящая CAPTCHA)

`_relaunch_game()`: force-stop → start → 4s → GOT IT (315,180) → DON'T ALLOW (400,320) → 2× ADVENTURE tap.

## Vision Classifier (приоритетный порядок)
| Step | Проверка | Результат |
|------|----------|-----------|
| 0 | overall_V<50 + dark top/edges + NO dial + tab_bright<0.03 | CAPTCHA |
| 0b | overall_V<100 + tab_bright>0.05 + dark_center>50% | MAIN_MENU (OFFLINE popup) |
| 1 | orange upper + dark center | DRIVER_DOWN |
| 1b | red upper + green center RESPAWN | DRIVER_DOWN (OUT OF FUEL) |
| 1b | red upper + white bottom | TOUCH_TO_CONTINUE |
| 2 | dark center + white bottom + no orange | TOUCH_TO_CONTINUE (dark overlay) |
| 2b | gray panel + `green_bl_btn<0.02` + `bl_bright<0.10` | TOUCH_TO_CONTINUE (level summary) |
| 3 | RPM dial brightness>30 + red needle>2.5% | RACING |
| 4 | `green_bl_btn>0.06` + `green_br_btn>0.06` (bright buttons) | RESULTS |
| 5 | yellow top + blue skip bottom | DOUBLE_COINS_POPUP |
| 6 | green START right + `green_bl<0.05` + `bottom_white<0.10` | VEHICLE_SELECT |
| 6b | BACK button bright + green upgrades | VEHICLE_SELECT (fallback) |
| 7 | vivid bottom center + no green START | MAIN_MENU |
| 7b | tab bar visible + dark background | MAIN_MENU (fallback) |

Ключевое: `green_bl_btn` (S≥100,V≥130) — яркие UI кнопки, отличает RETRY/NEXT от зелёной травы Countryside.

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

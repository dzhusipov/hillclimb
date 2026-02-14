# Hill Climb Racing 2 AI Agent

AI-агент, который играет в [Hill Climb Racing 2](https://play.google.com/store/apps/details?id=com.fingersoft.hcr2) на параллельных Android-эмуляторах (ReDroid в Docker). Читает экран через ADB screencap, анализирует состояние игры с помощью OpenCV, принимает решения (rule-based или PPO RL), управляет машиной через ADB.

## Как это работает

```
┌── Docker (NAS) ───────────────────────────┐
│  ReDroid #0 (HCR2) ← ADB localhost:5555  │
│  ReDroid #1 (HCR2) ← ADB localhost:5556  │
│  ...до 8 эмуляторов                       │
│  ws-scrcpy (:8100) — live просмотр        │
│  dashboard (:8150) — мониторинг           │
└───────────────────────────────────────────┘
         │ ADB screencap (PNG, ~250ms)
         ▼
┌── Conda: hillclimb ──────────────────────┐
│  capture.py    — ADB screencap (adbutils)│
│  vision.py     — CV: states, dials, OCR  │
│  controller.py — ADB input: gas/brake    │
│  navigator.py  — state machine навигация │
│  env.py        — Gymnasium HCR2Env       │
│  train.py      — PPO (RTX 3090, CUDA)   │
└──────────────────────────────────────────┘
```

Захват через scrcpy H.264 stream (~15 FPS, fallback на ADB screencap). 8 параллельных эмуляторов через SubprocVecEnv.

## Требования

- **Ubuntu LTS** (NAS), или macOS для разработки
- **Python 3.11** (через conda)
- **Docker** с ReDroid (требуются kernel-модули binder_linux и ashmem_linux)
- **NVIDIA GPU** + CUDA 12.4 (для обучения; эмуляторы используют software rendering)

## Установка

### 1. Conda-окружение

```bash
git clone <repo-url> && cd hillclimb
conda env create -f environment.yml
conda activate hillclimb
```

### 2. Docker-инфраструктура

```bash
cd docker/

# 2 эмулятора (default)
docker compose up -d

# 4 эмулятора
docker compose --profile scale-4 up -d

# 8 эмуляторов
docker compose --profile scale-8 up -d
```

### 3. Установить HCR2 на эмуляторы

```bash
# Положить APK в docker/apk/, затем:
./scripts/manage.sh install-apk
```

### 4. Заблокировать интернет для эмуляторов

**Обязательно!** HCR2 использует серверную валидацию — при переносе сейва с физического устройства без блокировки интернета сработает CHEAT DETECTED.

```bash
SUBNET=$(docker network inspect docker_hcr2-net -f '{{range .IPAM.Config}}{{.Subnet}}{{end}}')
sudo iptables -I DOCKER-USER 1 -m conntrack --ctstate ESTABLISHED,RELATED -j RETURN
sudo iptables -I DOCKER-USER 2 -s $SUBNET -d $SUBNET -j RETURN
sudo iptables -I DOCKER-USER 3 -s $SUBNET -j DROP
```

Правила действуют только на подсеть hcr2-net. Другие контейнеры не затрагиваются.
Правила не переживают ребут — добавить в автозагрузку при необходимости.

### 5. Перенести прогресс HCR2 (опционально)

Adventure mode требует ~3500 звёзд. Чтобы не грайндить:

```bash
# На Mac (где подключён телефон с прогрессом):
adb backup -f hcr2_save.ab -noapk com.fingersoft.hcr2
scp hcr2_save.ab dasm@NAS-IP:~/develop/hillclimb/

# На NAS — извлечь и пушнуть:
python3 -c "
import zlib
with open('hcr2_save.ab','rb') as f:
    [f.readline() for _ in range(4)]
    open('hcr2_save.tar','wb').write(zlib.decompress(f.read()))
"
mkdir -p /tmp/hcr2_restore && tar xf hcr2_save.tar -C /tmp/hcr2_restore

# Для каждого эмулятора:
adb -s localhost:5555 root
adb -s localhost:5555 push /tmp/hcr2_restore/apps/com.fingersoft.hcr2/f/gamestatus.bin /data/data/com.fingersoft.hcr2/files/
adb -s localhost:5555 push /tmp/hcr2_restore/apps/com.fingersoft.hcr2/f/gamestatus.bak /data/data/com.fingersoft.hcr2/files/
```

## Быстрый старт

### Запустить HCR2

```bash
adb -s localhost:5555 shell am start -n com.fingersoft.hcr2/.AppActivity
```

### Тест захвата экрана

```bash
python -m hillclimb.capture --serial localhost:5555 --benchmark 10
```

### Тест контроллера

```bash
python -m hillclimb.controller --serial localhost:5555 --test
```

### Калибровка (ROI, кнопки, диалы)

```bash
python -m hillclimb.calibrate
```

### Rule-based агент

```bash
python -m hillclimb.game_loop --agent rules
python -m hillclimb.game_loop --agent rules --episodes 5 --headless
```

### Обучение RL-агента (PPO)

```bash
# Быстрый тест (100k шагов)
python -m hillclimb.train --timesteps 100000 --num-envs 8

# Обучение на ночь (nohup, лог в файл)
nohup python -u -m hillclimb.train --timesteps 500000 --num-envs 8 > logs/train_run.log 2>&1 &

# Продолжить обучение с сохранённой модели
python -m hillclimb.train --timesteps 500000 --num-envs 8 --resume models/ppo_hillclimb

# Исключить эмулятор(ы) из обучения
python -m hillclimb.train --timesteps 500000 --num-envs 8 --skip-envs 0

# Мониторинг обучения
tail -f logs/train_run.log
grep "^  EP" logs/train_run.log | tail -20
```

### Оценка обученной модели

```bash
python -m hillclimb.evaluate --episodes 10
```

## Мониторинг

| URL | Описание |
|-----|----------|
| `http://NAS-IP:8100` | ws-scrcpy — live просмотр эмуляторов |
| `http://NAS-IP:8150` | Веб-дашборд — snapshot polling, статусы, управление |

Дашборд использует snapshot polling (800мс) вместо MJPEG — обходит лимит 6 HTTP-соединений браузера.
Оба сервиса видны в CasaOS.

## Структура проекта

```
hillclimb/
├── docker/
│   ├── docker-compose.yml    — ReDroid + ws-scrcpy + dashboard
│   ├── Dockerfile.dashboard  — образ веб-дашборда
│   ├── .env                  — конфигурация эмуляторов
│   └── apk/                  — APK файлы HCR2 (gitignored)
├── web/
│   ├── server.py             — FastAPI дашборд (:8150)
│   ├── emulator.py           — управление эмуляторами (Docker + ADB)
│   ├── streamer.py           — MJPEG стриминг (legacy)
│   ├── templates/dashboard.html
│   └── static/{app.js, style.css}  — snapshot polling
├── hillclimb/
│   ├── config.py             — координаты кнопок, ROI, пороги (800x480)
│   ├── capture.py            — ADB screencap через adbutils
│   ├── vision.py             — CV: 9 game states, dials, template OCR
│   ├── controller.py         — ADB input: gas/brake/tap через adbutils
│   ├── navigator.py          — state machine навигация (9 состояний)
│   ├── agent_rules.py        — rule-based baseline агент
│   ├── agent_rl.py           — RL агент (PPO)
│   ├── env.py                — Gymnasium HCR2Env
│   ├── memory_reader.py      — чтение позиции из памяти (nodefinder pipe)
│   ├── game_loop.py          — основной цикл: capture → CV → agent → input
│   ├── logger.py             — CSV лог + PNG кадры
│   ├── calibrate.py          — интерактивная калибровка ROI
│   └── train.py              — обучение PPO
├── scripts/
│   ├── manage.sh             — start/stop/restart/install-apk
│   └── nodefinder.c          — C утилита поиска Node в памяти HCR2
├── tests/
│   └── test_vision.py        — тесты CV модуля
├── models/                   — RL модели (gitignored)
├── logs/                     — логи (gitignored)
├── templates/                — шаблоны для template matching OCR
├── environment.yml           — conda (Python 3.11, PyTorch CUDA 12.4)
├── requirements.txt          — pip dependencies
├── PLAN.md                   — план рефакторинга с прогрессом
├── CLAUDE.md                 — инструкции для Claude Code
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

## Actions

Discrete(3): `0=nothing, 1=gas, 2=brake`

В воздухе: gas = наклон назад (нос вверх), brake = наклон вперёд (нос вниз).

## Технологии

- **Эмуляторы:** ReDroid 14 (Docker), 480x800 portrait / 800x480 landscape
- **Computer Vision:** OpenCV (HSV-сегментация, dial gauge reader, template matching OCR)
- **Reinforcement Learning:** Stable-Baselines3 PPO, Gymnasium
- **Screen Capture:** scrcpy H.264 stream через PyAV (~15 FPS) + ADB screencap fallback
- **Input:** ADB shell input swipe через adbutils
- **Hardware Acceleration:** PyTorch CUDA (RTX 3090)
- **Web Dashboard:** FastAPI + Snapshot Polling (800мс)
- **Monitoring:** ws-scrcpy, CasaOS

## Чтение памяти (Memory Reader)

Помимо визуального анализа (OCR дистанции), агент читает позицию машины напрямую из памяти игрового процесса — это точнее и быстрее.

### Как это работает

HCR2 построена на Cocos2d-x / Box2D. Позиция машины хранится в `Node` объекте внутри heap-региона `[anon:scudo:primary]` (~22 MB). Утилита `nodefinder` (C, статическая компиляция) сканирует этот регион и находит нужный Node через структурный паттерн:

1. **Скан** — читаем весь регион (~22 MB) через `process_vm_readv`
2. **Структурный поиск** — ищем Node по 5 признакам: scale [1,1,1], rotation sin²+cos²=1, pos_X copy на +108, car body markers ±0.7071 на +96/+100, дупликат pos_Y
3. **Delta filter** — ждём 2с, перечитываем — кто сдвинулся = живой Node машины
4. **Стриминг** — непрерывно выдаём pos_x / pos_y (float32) по stdout pipe

### Что читаем

| Параметр | Оффсет | Описание |
|----------|--------|----------|
| pos_X | +0 | Мировая X-координата машины (≈ метры) |
| pos_Y | -36 / -32 | Высота машины (дублируется) |
| rotation | -20 / -16 | sin / cos угла поворота |
| tilt | +60 / +64 | cos / sin наклона кузова |
| scale | -12 / -8 / -4 | Масштаб X/Y/Z (всегда 1.0) |

**Дистанция** = `pos_x - initial_x` (начальная позиция фиксируется при старте гонки).

### Интеграция в обучение

`MemoryReader` (`hillclimb/memory_reader.py`) запускается в фоновом потоке при каждом reset(). Через ~8 секунд (когда машина уже едет) nodefinder завершает скан и начинает стриминг. До этого момента используется OCR distance как fallback. После подключения — `state.distance_m` перезаписывается точным значением из памяти.

### Безопасность

- `process_vm_readv` ≤ 70 MB за вызов — безопасно (скан 22 MB ниже лимита)
- 160 MB+ — крашит игру (Android SIGKILL по античиту)
- Точечные чтения (4-8 байт) — без ограничений

### Деплой

```bash
# Компиляция
gcc -O2 -static -o nodefinder scripts/nodefinder.c

# На все эмуляторы
for i in $(seq 0 7); do
  docker cp nodefinder hcr2-$i:/data/local/tmp/
  docker exec hcr2-$i chmod +x /data/local/tmp/nodefinder
done
```

## Тесты

```bash
conda activate hillclimb
python -m pytest tests/ -v
```

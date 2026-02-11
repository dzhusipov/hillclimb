# PLAN: Рефакторинг Hill Climb Racing 2 AI Agent

> **Репозиторий:** https://github.com/dzhusipov/hillclimb
> **Цель:** Полная переработка проекта — переход на параллельные Android-эмуляторы, image-based CNN, веб-мониторинг, и ускорение обучения в 50–100 раз.
> **Целевая платформа обучения:** NAS — Ubuntu LTS, Ryzen 5 5600, 80GB DDR4, RTX 3090 24GB VRAM
> **Платформа разработки:** macOS, Apple Silicon M2 Max, 32GB
> **Среда:** Conda (Python) + Docker (эмуляторы)
> **Игра:** Hill Climb Racing 2, Adventure Mode

---

## Прогресс выполнения

### Шаг 1: environment.yml + requirements.txt — ВЫПОЛНЕН ✅
- [x] `environment.yml` обновлён: PyTorch CUDA 12.4, adbutils, fastapi, uvicorn, pydantic, jinja2
- [x] `requirements.txt` обновлён (pip-зависимости)
- [x] Убраны Mac-специфичные пакеты (mss, pytesseract)
- [x] Проверено: PyTorch 2.5.1, CUDA=True, все пакеты установлены

### Шаг 2: Docker инфраструктура + мониторинг (Фаза 0) — ВЫПОЛНЕН ~90%
- [x] `docker/docker-compose.yml` — 8 ReDroid-контейнеров с profiles (default=2, scale-4, scale-8)
- [x] `docker/.env` — конфигурация (NUM_EMULATORS, разрешение, RAM, CPU)
- [x] `docker/Dockerfile.dashboard` — образ для веб-дашборда (FastAPI + OpenCV + ADB + Docker CLI)
- [x] `docker/apk/` — APK файлы HCR2 v1.70.4 (base + x86_64 + en + mdpi)
- [x] ws-scrcpy контейнер на порту 8100
- [x] Веб-дашборд на порту 8150: MJPEG стримы, статусы, управление контейнерами, start/stop game
- [x] CasaOS labels на dashboard и ws-scrcpy
- [x] `scripts/manage.sh` — start/stop/restart/status/install-apk
- [x] HCR2 установлена и запускается на обоих эмуляторах
- [x] Прогресс из Redmi перенесён на эмуляторы (46338 coins, 110 gems, Adventure)
- [x] Интернет для эмуляторов заблокирован через iptables DOCKER-USER
- [ ] **БАГ:** Touch input через дашборд — координаты не попадают (маппинг norm→screen)

### Шаг 3: Capture + Vision + Controller (Фаза 1) — ВЫПОЛНЕН ✅
- [x] `capture.py` переписан: adbutils, PNG/RAW backends, auto-reconnect, ~2.4 FPS
- [x] `controller.py` переписан: adbutils, параметризованные координаты, без singleton config
- [x] `config.py` обновлён: дефолты для ReDroid 800x480, adb_serial, emulator_serial()
- [x] `vision.py` обновлён: template matching OCR вместо Tesseract (с fallback)
- [x] Smoke-тест: capture (480x800 BGR) + vision (VEHICLE_SELECT detected) + controller OK

### Шаг 4: Калибровка — НЕ НАЧАТ
Нужно определить точные ROI и координаты кнопок для 800x480 ReDroid.
Нужно создать шаблоны цифр для template matching OCR.

### Шаг 5–10: НЕ НАЧАТЫ
Следующий шаг: **Шаг 4 — Калибровка ROI и кнопок**

---

## Контекст проекта

Текущее состояние: рабочий AI-агент для HCR, использующий OpenCV для парсинга экрана и PPO (Stable-Baselines3) для обучения. Работает на одном физическом Android-устройстве через scrcpy + ADB. Скорость ~10 FPS, обучение крайне медленное.

**⚠️ ЧТО УЖЕ РАБОТАЕТ И ДОЛЖНО БЫТЬ СОХРАНЕНО:**
- **Навигация по меню** — в репозитории реализован рабочий переход по менюшкам HCR2 до запуска гонки (navigator.py). Эту логику нужно сохранить и адаптировать, а не переписывать с нуля.
- **Обход капчи** — реализован механизм обхода капчи. Сохранить как есть.
- Перед рефакторингом любого модуля — изучить существующую реализацию в репозитории и перенести рабочую логику.

**Приоритет рефакторинга:**
Основной фокус переделки — **сама гонка и обучение**: захват экрана, CV во время заезда, reward function, Gymnasium env, параллелизация, PPO-тренировка. Всё что касается менюшек, навигации и капчи — максимально сохранять существующий код, адаптируя только интерфейсы под новую архитектуру.

Проблемы текущей версии (касаются именно гонки и обучения):
- Одно устройство — нет параллелизации, невозможно собрать достаточно данных
- Бедный state vector (7 float без истории) — агент не видит контекст
- Нет дистанции в reward — используется шумный optical flow вместо OCR счётчика
- ADB input latency нестабильна — ломает credit assignment

Что уже правильно в текущей версии:
- 3 действия (nothing/gas/brake) — это полный набор управления HCR2 (газ/тормоз также управляют наклоном в воздухе)
- CV pipeline изначально заточен под HCR2 — не требует переделки под другую игру

Целевое состояние: 8 параллельных ReDroid-эмуляторов на NAS, image-based CNN policy с гибридными наблюдениями, OCR дистанции через template matching, веб-дашборд мониторинга, обучение 10M timesteps за 2–5 дней.

---

## Архитектура решения

```
┌─────────────────────────────────────────────────────────────┐
│                    NAS (Ubuntu LTS)                         │
│                                                             │
│  ┌─── Docker ──────────────────────────────────────────┐   │
│  │  ReDroid #0 (HCR2)  ← ADB :5555                    │   │
│  │  ReDroid #1 (HCR2)  ← ADB :5556                    │   │
│  │  ...                                                │   │
│  │  ReDroid #7 (HCR2)  ← ADB :5562                    │   │
│  └─────────────────────────────────────────────────────┘   │
│         │  screencap / scrcpy                               │
│         ▼                                                   │
│  ┌─── Conda env: hillclimb ────────────────────────────┐   │
│  │  SubprocVecEnv (8 workers)                          │   │
│  │  ├── Worker 0: HCR2Env(:5555) → capture → CV → act │   │
│  │  ├── Worker 1: HCR2Env(:5556) → capture → CV → act │   │
│  │  └── ...                                            │   │
│  │         │                                           │   │
│  │         ▼                                           │   │
│  │  PPO + MultiInputPolicy (CnnPolicy)                 │   │
│  │  device=cuda (RTX 3090)                             │   │
│  │         │                                           │   │
│  │         ▼                                           │   │
│  │  Monitor Server (FastAPI + WebSocket)               │   │
│  │  TensorBoard                                        │   │
│  └─────────────────────────────────────────────────────┘   │
│         │  http                                             │
│         ▼                                                   │
│  Веб-дашборд: http://nas-ip:8080                           │
└─────────────────────────────────────────────────────────────┘
         │
    Браузер на Mac (разработка, мониторинг)
```

---

## Фаза 0: Инфраструктура Docker + ReDroid

### 0.1 Docker Compose для эмуляторов

Создать `docker/docker-compose.yml` для запуска N ReDroid-контейнеров. Параметры каждого контейнера:
- Образ: `redroid/redroid:12.0.0_64only-latest`
- Режим: `--privileged`
- RAM: 3GB, CPU: 1.2 ядра
- Разрешение: 480×800, DPI: 160
- GPU mode: `guest` (software rendering — GPU остаётся для PyTorch)
- Порты: ADB на 5555+i для каждого контейнера
- Volume: персистентный `/data` для сохранения состояния игры

Также в compose добавить:
- Сервис `ws-scrcpy` для live-просмотра эмуляторов в браузере (порт 8000)
- Сервис `monitor` для кастомного дашборда (порт 8080)

### 0.2 Мониторинг эмуляторов через CasaOS

**⚠️ Мониторинг поднимается ПЕРВЫМ, до любого обучения. Без визуального контроля — не двигаемся дальше.**

NAS работает под CasaOS — все Docker-сервисы должны быть видны и управляемы через его веб-интерфейс. Задача: после `docker compose up` я открываю CasaOS в браузере и вижу все эмуляторы live.

**ws-scrcpy как основной инструмент визуального мониторинга:**
- Поднимается как Docker-контейнер в том же compose-файле
- Автоматически обнаруживает все ReDroid-инстансы через ADB
- Веб-интерфейс на порте 8000 — показывает все эмуляторы в виде карточек
- Можно кликнуть на любой эмулятор и управлять тачем прямо из браузера
- Это позволяет: проверить что HCR2 запустилась, вручную пройти меню при первичной настройке, наблюдать за агентом в реальном времени

**Интеграция с CasaOS:**
- Создать `docker/casaos/` с описанием приложений для CasaOS App Store (yaml-манифесты)
- Либо: зарегистрировать docker-compose сервисы так, чтобы CasaOS их видел в разделе "Apps" (CasaOS автоматически подхватывает контейнеры запущенные через Docker)
- В CasaOS dashboard должны быть видны:
  - 8 ReDroid-контейнеров (hcr2-0 ... hcr2-7) — статус, ресурсы, логи
  - ws-scrcpy — ссылка на веб-интерфейс :8000
  - Позже: TensorBoard (:6006), кастомный training monitor (:8080)
- Настроить labels/ports в docker-compose чтобы CasaOS корректно отображал имена и ссылки на веб-интерфейсы

**Порядок проверки перед продолжением:**
1. Открыть CasaOS → видны все 8 контейнеров hcr2-0..7, статус Running
2. Открыть ws-scrcpy (http://nas-ip:8000) → видны все 8 эмуляторов
3. Кликнуть на любой эмулятор → видно рабочий стол Android
4. Запустить HCR2 вручную через ws-scrcpy на одном эмуляторе → игра работает
5. Только после этого переходить к Фазе 1

### 0.3 Скрипт инициализации эмуляторов

Создать `scripts/setup_emulators.sh`:
- Проверить наличие kernel-модулей `binder_linux` и `ashmem_linux`, загрузить если отсутствуют
- Запустить `docker compose up -d`
- Дождаться готовности каждого контейнера (polling `adb connect`)
- Установить HCR2 на каждый инстанс через `adb install-multiple` (base.apk + split_config.x86_64.apk)
- Отключить все анимации Android на каждом инстансе (`window_animation_scale`, `transition_animation_scale`, `animator_duration_scale` → 0)
- Установить разрешение `adb shell wm size 480x800`
- Запустить HCR2 на каждом инстансе
- Проверить что игра запустилась (скриншот + базовая валидация)

### 0.4 Скрипт управления

Создать `scripts/manage.sh` с командами:
- `start` — поднять все контейнеры
- `stop` — остановить все контейнеры
- `restart N` — перезапустить конкретный контейнер
- `status` — показать состояние всех контейнеров и ADB-подключений
- `install-apk` — установить/обновить HCR2 на всех инстансах
- `screenshot N` — сделать скриншот конкретного эмулятора

### 0.5 Хранение APK

Создать `docker/apk/` директорию. Добавить её в `.gitignore`. В README описать откуда скачать APK (APKMirror) и какие файлы нужны: `base.apk`, `split_config.x86_64.apk`, `split_config.xxxhdpi.apk`, `split_config.en.apk`.

---

## Фаза 1: Ядро — Capture, CV, Controller

### 1.1 Модуль захвата экрана — `hillclimb/capture.py`

Полностью переписать модуль захвата. Поддержать два backend'а с единым интерфейсом:

**Backend 1: ADB screencap (основной для ReDroid)**
- Использовать `adb -s <serial> exec-out screencap` в RAW формате (без PNG-сжатия) для минимальной латентности
- Парсить RAW output: первые 12 байт — header (width, height, format), остальное — RGBX пиксели
- Конвертировать в numpy array
- Целевая скорость: 10–15 FPS на ReDroid-контейнер

**Backend 2: scrcpy stream (опциональный, для более высокого FPS)**
- Запускать `scrcpy --no-display --video-codec=h264 --max-fps=15 --max-size=480` с выводом в pipe
- Декодировать H264-поток через FFmpeg subprocess или `av` (PyAV)
- Целевая скорость: 15–30 FPS

**Общий интерфейс:**
```
class ScreenCapture:
    def __init__(self, adb_serial: str, backend: str = "screencap")
    def capture(self) -> np.ndarray  # BGR, shape (H, W, 3)
    def close(self)
```

Важно: каждый инстанс ScreenCapture работает с конкретным ADB serial (`localhost:5555`, `localhost:5556`, ...). Соединение должно быть persistent — не переподключаться на каждый кадр.

### 1.2 Модуль компьютерного зрения — `hillclimb/vision.py`

Изучить существующий `vision.py` — в нём уже есть Game State Classifier, Gauge Reader, Vehicle Tilt Detector, Terrain Analyzer, Speed Estimator. Сохранить рабочие компоненты (особенно game state classification — racing/menu/crash/results). Основное изменение: **добавить OCR дистанции через template matching** (сейчас отсутствует) и адаптировать ROI/пороги под HCR2.

Модуль должен из скриншота извлекать:

**1. Дистанцию (distance) — через template matching цифр:**
- Создать шаблоны цифр 0–9 из скриншотов HCR2 (жёлтый шрифт на полупрозрачном фоне)
- ROI дистанции — верхняя часть экрана
- Template matching с порогом confidence ≥ 0.85
- Сортировка найденных цифр по x-позиции → сборка числа
- Возвращать `(value: int, confidence: float, reliable: bool)`
- Sanity check: дистанция не уменьшается, не растёт быстрее 50м/сек

**2. Топливо (fuel) — через цветовой анализ:**
- ROI топливной полоски
- HSV-фильтрация зелёного/жёлтого/красного цвета полоски
- Процент заполненных пикселей → float 0.0–1.0
- Детекция подбора канистры: fuel[t] > fuel[t-1]

**3. Состояние игры (game state) — классификатор:**
- `racing` — идёт заезд
- `crashed` — машина перевернулась (game over)
- `menu` — главное меню / меню выбора
- `results` — экран результатов
- Реализовать через template matching ключевых UI-элементов (кнопка "Retry", "Continue", логотип) или через цветовой анализ характерных областей

**4. Монеты (coins) — опционально:**
- Template matching числового счётчика монет
- Для дополнительного компонента reward

**Общий интерфейс:**
```
class GameVision:
    def __init__(self, config: dict)
    def process(self, frame: np.ndarray) -> GameState
    
@dataclass
class GameState:
    game_phase: str          # "racing" | "crashed" | "menu" | "results"
    distance: int            # метры
    distance_confidence: float
    fuel: float              # 0.0–1.0
    coins: int
    is_reliable: bool        # все OCR-метрики достаточно уверены
```

### 1.3 Калибровка под HCR2 — `hillclimb/calibrate.py`

Переписать утилиту калибровки:
- Подключиться к указанному эмулятору по ADB serial
- Показать скриншот, дать выделить мышкой ROI для: дистанции, топлива, кнопки газа, кнопки тормоза, области игрового поля
- Сохранить все ROI в `config.json`
- Добавить режим "auto-detect" — попробовать найти элементы автоматически через template matching стандартных UI-элементов HCR2
- Добавить команду `--capture-digits` — захватить шаблоны цифр 0–9 из текущего скриншота и сохранить в `templates/`

### 1.4 Контроллер — `hillclimb/controller.py`

Адаптировать под множественные устройства. Набор действий остаётся Discrete(3) — это полное управление HCR2:

**3 дискретных действия:**
| ID | Действие | ADB-команда | На земле | В воздухе |
|----|----------|-------------|----------|-----------|
| 0 | Ничего | — (отпустить всё) | Качение по инерции | Свободный полёт |
| 1 | Газ | tap правая кнопка | Ускорение вперёд | Вращение против часовой стрелки (нос вверх) |
| 2 | Тормоз | tap левая кнопка | Торможение / откат назад | Вращение по часовой стрелке (нос вниз) |

**Требования:**
- Использовать `adb shell input tap` вместо `input swipe` для дискретных нажатий
- Для удержания кнопок — `input swipe x y x y duration_ms`
- Координаты кнопок из `config.json`
- Каждый Controller привязан к конкретному ADB serial
- Реализовать метод `release_all()` — отпустить все кнопки

### 1.5 Навигатор меню — `hillclimb/navigator.py`

**⚠️ СОХРАНИТЬ СУЩЕСТВУЮЩУЮ ЛОГИКУ. В репозитории уже реализованы:**
- Переход по меню до запуска гонки (работает)
- Обход капчи (работает)

**НЕ переписывать с нуля.** Взять существующий `navigator.py`, изучить его полностью и адаптировать минимально:
- Обновить интерфейс, чтобы принимал ADB serial (для работы с конкретным эмулятором вместо единственного устройства)
- Убедиться что обход капчи и навигация по меню работают на ReDroid-эмуляторе (могут быть различия в таймингах — подтюнить при необходимости)
- При необходимости добавить (но не ломать существующее):
  - Обработка всплывающих окон (реклама, предложения покупок) — если не реализовано
  - Детекция зависания — если состояние не менялось N секунд, перезапустить игру через `am force-stop` + `am start`
- Сохранить все координаты, шаблоны и логику обхода капчи

---

## Фаза 2: Gymnasium Environment

### 2.1 Основная среда — `hillclimb/env.py`

Создать полноценную Gymnasium-среду `HCR2Env`:

**Observation space — Dict (гибридный):**
- `"image"`: Box(0, 255, shape=(84, 84, 1), dtype=uint8) — grayscale, resized игровое поле (без HUD)
- `"vector"`: Box с 6 float значениями:
  - fuel (0–1)
  - distance (0–50000)
  - speed — delta distance / dt
  - fuel_delta — изменение топлива за последний шаг
  - distance_delta — приращение дистанции за шаг
  - time_since_last_progress — секунды без прироста дистанции

**Action space:** Discrete(3) — nothing / gas / brake (газ и тормоз также управляют наклоном в воздухе)

**Метод `step(action)`:**
1. Отправить действие через Controller
2. Подождать `action_duration` мс (настраиваемо, дефолт 66мс ≈ 15 FPS)
3. Захватить скриншот через Capture
4. Обработать через Vision → GameState
5. Если OCR ненадёжен — использовать предыдущее значение дистанции
6. Вычислить reward
7. Определить terminated (crash / out of fuel) и truncated (max steps)
8. Вернуть (obs, reward, terminated, truncated, info)

**Метод `reset()`:**
1. Если игра в состоянии `crashed` или `results` — через Navigator нажать retry
2. Если игра в `menu` — через Navigator запустить Adventure Mode
3. Дождаться состояния `racing`
4. Обнулить внутренние счётчики
5. Вернуть начальный observation

**Reward function:**
- `+distance_delta * 1.0` — основной сигнал (приращение дистанции в метрах)
- `+speed_bonus * 0.1` — clip(speed / MAX_SPEED, 0, 1)
- `-5.0` за crash (game over)
- `+0.5` за подбор канистры (fuel увеличился)
- `-0.05 * (0.2 - fuel) / 0.2` — когда fuel < 0.2 (штраф за критически низкое топливо)
- `-0.1` за простой (distance_delta ≈ 0 при состоянии racing)

**Сохранение debug-информации:**
- В `info` dict каждого шага добавлять: raw distance, raw fuel, ocr_confidence, game_phase, action_name
- Опционально: сохранять debug-кадр с overlay в shared dict для мониторинга

### 2.2 Фабрика сред — `hillclimb/env_factory.py`

Функция `make_env(emulator_port, rank, config_path, seed)` возвращающая callable для SubprocVecEnv:
- Каждый вызов создаёт НОВЫЙ экземпляр HCR2Env с собственным ADB-соединением
- Применяет Gymnasium wrappers в правильном порядке:
  1. `TimeLimit` — максимум 3000 шагов на эпизод (~3–5 минут)
  2. `RecordEpisodeStatistics` — трекинг episode reward и length
- Seed для воспроизводимости

### 2.3 Vectorized environment setup — `hillclimb/vec_env.py`

Функция для создания готового vectorized environment:
- Создать SubprocVecEnv из N фабрик `make_env` с `start_method='spawn'`
- Обернуть в `VecFrameStack(n_stack=4)` — стекирование 4 кадров для temporal context
- Обернуть в `VecMonitor` для логирования
- Вернуть готовый env для передачи в PPO

---

## Фаза 3: Обучение

### 3.1 Тренировочный скрипт — `hillclimb/train.py`

Полностью переписать:

**Аргументы CLI:**
- `--num-envs N` — количество параллельных сред (дефолт: 8)
- `--timesteps N` — total timesteps (дефолт: 10_000_000)
- `--resume PATH` — продолжить обучение с чекпоинта
- `--config PATH` — путь к config.json (дефолт: config.json)
- `--device` — cuda/cpu/auto (дефолт: auto)
- `--lr` — начальный learning rate (дефолт: 2.5e-4)
- `--monitor-port` — порт для веб-мониторинга (дефолт: 8080)
- `--save-freq N` — частота сохранения чекпоинтов в шагах (дефолт: 100_000)
- `--eval-freq N` — частота eval (дефолт: 50_000)

**Гиперпараметры PPO (дефолты, тюнить позже):**
- `learning_rate`: linear schedule от 2.5e-4 до 0
- `n_steps`: 128
- `batch_size`: 256
- `n_epochs`: 4
- `gamma`: 0.99
- `gae_lambda`: 0.95
- `clip_range`: linear schedule от 0.1 до 0
- `ent_coef`: 0.01
- `vf_coef`: 0.5
- `max_grad_norm`: 0.5

**Policy:** `MultiInputPolicy` — автоматически использует NatureCNN для image и MLP для vector, конкатенирует features.

**Callbacks:**
- `CheckpointCallback` — сохранять модель каждые `save_freq` шагов в `models/`
- `EvalCallback` — eval на 1 отдельной среде каждые `eval_freq` шагов, сохранять best model
- Кастомный `MonitorCallback` — обновлять shared dict с метриками для веб-дашборда (см. Фазу 5)
- Кастомный `TensorBoardCallback` — логировать кастомные метрики: avg_distance, max_distance, avg_fuel_at_death, ocr_fail_rate

**TensorBoard:** логирование в `logs/tensorboard/`

### 3.2 Оценка модели — `hillclimb/evaluate.py`

Переписать:
- Загрузить обученную модель
- Запустить N эпизодов (дефолт: 20) на одной или нескольких средах
- Собрать статистику: mean/std reward, mean/max distance, survival time
- Опционально: записать видео эпизодов (сохранять кадры → склеить через FFmpeg)
- Вывести результаты в консоль и сохранить в JSON

### 3.3 Rule-based baseline — `hillclimb/agent_rules.py`

Обновить rule-based агента для HCR2:
- Если fuel < 20% → газ (искать канистру)
- Если крутой подъём → газ
- Если в воздухе после прыжка → тормоз (нос вниз для приземления на колёса) или газ (нос вверх при крутом спуске)
- Если скорость ≈ 0 и не crash → тормоз (откат для разгона)
- Иначе → газ

Этот агент нужен для: базовой проверки что pipeline работает, сбора данных для imitation learning, baseline для сравнения с RL.

---

## Фаза 4: Imitation Learning (Behavior Cloning)

### 4.1 Запись демонстраций — `hillclimb/record.py`

Скрипт для записи человеческой игры:
- Подключиться к эмулятору
- Запустить игру
- В цикле: захватывать экран + читать состояние через Vision
- Отслеживать тач-события через `adb shell getevent` → маппить на action ID
- Сохранять пары (observation, action) в файлы: `demos/demo_001.npz`
- Горячая клавиша для старта/стопа записи

### 4.2 Behavior cloning — `hillclimb/pretrain.py`

Предобучение policy на демонстрациях перед RL:
- Загрузить все `.npz` из `demos/`
- Использовать SB3 `PPO.load()` или тренировать отдельный supervised classifier
- Предпочтительный вариант: создать PPO модель, извлечь policy network, обучить через supervised loss (CrossEntropyLoss на action prediction), загрузить веса обратно в PPO
- Альтернатива: использовать библиотеку `imitation` (pip install imitation) для behavior cloning поверх SB3
- Сохранить pretrained model в `models/pretrained.zip`

Цель: 20–30 записанных эпизодов дают агенту базовое понимание "газ = ехать вперёд" до RL-тюнинга.

---

## Фаза 5: Веб-мониторинг

### 5.1 Backend — `hillclimb/monitor/server.py`

FastAPI-сервер с WebSocket:

**Shared state** — multiprocessing-safe dict (через `multiprocessing.Manager` или файловый IPC), обновляемый из training loop:
- Для каждого env: последний кадр (JPEG bytes), distance, fuel, reward, episode_reward, game_phase, ocr_confidence, status
- Глобально: total_steps, episodes_done, avg_reward, best_distance, fps, training_time

**WebSocket endpoint `/ws`:**
- Отправлять JSON с данными всех env + глобальными метриками
- Кадры — base64 JPEG с качеством 40–50% (минимум трафика)
- Частота: 1–2 обновления в секунду
- Включать debug overlay на кадрах (ROI-рамки, распознанные значения, текущее действие)

**REST endpoints:**
- `GET /api/status` — текущее состояние тренировки
- `GET /api/history` — история rewards/distances за последние N эпизодов
- `POST /api/pause` — поставить тренировку на паузу
- `POST /api/resume` — возобновить
- `POST /api/restart-env/{id}` — перезапустить конкретную среду

### 5.2 Frontend — `hillclimb/monitor/static/index.html`

Одностраничное веб-приложение (vanilla JS или minimal React, без сборки):

**Layout дашборда:**

Верхняя панель:
- Статус тренировки (running/paused), total steps, elapsed time, FPS
- Кнопки Pause/Resume

Основная область — сетка N эмуляторов (2×4 для 8 env):
- Каждая ячейка: миниатюра кадра с debug overlay, distance, fuel bar, episode reward, статус (racing/crashed/menu/ocr_fail)
- Клик на ячейку — увеличенный просмотр

Нижняя панель:
- График reward по эпизодам (chart.js или recharts — любая lightweight библиотека)
- График best distance по эпизодам
- Лог последних событий (crashes, records, OCR fails, env restarts)

### 5.3 Интеграция с тренировочным циклом

Мониторинг НЕ должен замедлять тренировку:
- Кадры для дашборда сохраняются в shared memory/dict с debounce (не чаще 2 раз/сек на env)
- FastAPI-сервер работает в отдельном процессе
- Если никто не подключён к WebSocket — данные не сериализуются и не отправляются
- TensorBoard запускается отдельно (`tensorboard --logdir logs/tensorboard`)
- ws-scrcpy запускается через Docker compose (независимо от Python)

---

## Фаза 6: Конфигурация и утилиты

### 6.1 Конфигурация — `hillclimb/config.py` + `config.json`

Единый конфиг-файл с секциями:
- `emulators`: список ADB-серверов с портами
- `capture`: backend (screencap/scrcpy), target_fps
- `vision`: ROI координаты для каждого элемента, пороги confidence
- `controller`: координаты кнопок, action_duration_ms
- `training`: все гиперпараметры PPO, num_envs, save_freq и т.д.
- `monitor`: порт, debug_overlay on/off, frame_quality

Дефолтные значения в `config.py`, пользовательские override через `config.json`.

### 6.2 Логирование — `hillclimb/logger.py`

Переписать:
- CSV-лог каждого эпизода: env_id, episode_num, total_reward, distance, steps, avg_fuel, death_reason, timestamp
- Сохранение проблемных кадров (OCR fails, неожиданные состояния) в `logs/debug_frames/`
- Ротация логов (не больше 1GB)

### 6.3 Тесты — `tests/`

Написать тесты:
- `test_vision.py` — тестирование OCR и game state detection на сохранённых скриншотах из `tests/fixtures/`
- `test_env.py` — тест что env создаётся, step/reset работают (mock capture)
- `test_controller.py` — тест маппинга action → ADB-команды (без реального ADB)
- `test_reward.py` — тест reward function на граничных случаях

---

## Структура проекта (целевая)

```
hillclimb/
├── docker/
│   ├── docker-compose.yml          # ReDroid + ws-scrcpy + monitor
│   ├── apk/                        # .gitignore'd, сюда класть APK
│   └── Dockerfile.monitor          # если monitor в отдельном контейнере
├── scripts/
│   ├── setup_emulators.sh          # Инициализация всех эмуляторов
│   ├── manage.sh                   # start/stop/restart/status
│   └── capture_templates.sh        # Захват шаблонов цифр для OCR
├── hillclimb/
│   ├── __init__.py
│   ├── config.py                   # Дефолтные конфиги + загрузка config.json
│   ├── capture.py                  # Захват экрана (screencap / scrcpy)
│   ├── vision.py                   # CV: OCR дистанции, fuel, game state
│   ├── controller.py               # ADB input: nothing/gas/brake
│   ├── navigator.py                # Навигация по меню HCR2, авто-рестарт
│   ├── calibrate.py                # Интерактивная калибровка ROI
│   ├── env.py                      # Gymnasium HCR2Env
│   ├── env_factory.py              # make_env() + wrappers
│   ├── vec_env.py                  # SubprocVecEnv + VecFrameStack сборка
│   ├── train.py                    # Обучение PPO
│   ├── evaluate.py                 # Оценка модели
│   ├── agent_rules.py              # Rule-based baseline
│   ├── record.py                   # Запись демонстраций
│   ├── pretrain.py                 # Behavior cloning
│   ├── logger.py                   # CSV + debug frame logging
│   └── monitor/
│       ├── server.py               # FastAPI + WebSocket backend
│       └── static/
│           └── index.html          # Веб-дашборд (single page)
├── templates/                      # Шаблоны цифр 0–9 для OCR
├── models/                         # Чекпоинты и обученные модели
├── logs/                           # Логи, TensorBoard, debug frames
├── demos/                          # Записи демонстраций для imitation learning
├── tests/
│   ├── fixtures/                   # Тестовые скриншоты
│   ├── test_vision.py
│   ├── test_env.py
│   ├── test_controller.py
│   └── test_reward.py
├── config.json                     # Пользовательский конфиг
├── environment.yml                 # Conda environment
├── requirements.txt                # pip dependencies
├── CLAUDE.md                       # Инструкции для Claude Code
└── README.md                       # Документация проекта
```

---

## Порядок выполнения

**Строго последовательно. Каждая фаза должна быть рабочей перед переходом к следующей.**

**⚠️ ГЛАВНЫЙ ПРИНЦИП: Перед изменением любого файла — сначала прочитай существующую реализацию в репозитории. В проекте уже работают: навигация по меню HCR2, обход капчи, базовый CV pipeline. Эти модули нужно сохранить и адаптировать, а не выбрасывать. Основной фокус переделки — capture во время гонки, vision во время гонки, reward, env, параллелизация и обучение.**

### Шаг 1: environment.yml + requirements.txt
Обновить зависимости:
- Python 3.11
- pytorch + torchvision (CUDA 12.x)
- stable-baselines3[extra]
- gymnasium
- opencv-python-headless
- numpy
- fastapi + uvicorn + websockets
- adbutils (Python ADB library — удобнее чем subprocess)
- Pillow
- tensorboard

### Шаг 2: Docker инфраструктура + мониторинг (Фаза 0)
Поднять все ReDroid-контейнеры + ws-scrcpy. Убедиться через CasaOS что все контейнеры Running. Открыть ws-scrcpy в браузере — все 8 эмуляторов видны. Установить HCR2, запустить вручную через ws-scrcpy на одном эмуляторе — игра работает. **Не двигаться дальше пока мониторинг не работает.**

### Шаг 3: Capture + Vision + Controller (Фаза 1)
Реализовать, протестировать на одном эмуляторе. Добиться стабильного OCR дистанции с confidence > 90%.

### Шаг 4: Калибровка
Откалибровать ROI на эмуляторе, сохранить config.json.

### Шаг 5: Env + Rule-based агент
Реализовать HCR2Env, запустить с rule-based агентом на одном эмуляторе. Убедиться что цикл step/reset работает стабильно минимум 30 минут без сбоев.

### Шаг 6: Параллелизация
Поднять 8 эмуляторов, запустить SubprocVecEnv с rule-based агентом. Убедиться что все 8 работают параллельно стабильно.

### Шаг 7: Мониторинг (Фаза 5)
Реализовать веб-дашборд, убедиться что видно все 8 эмуляторов с метриками.

### Шаг 8: PPO-обучение (Фаза 3)
Запустить тренировку на 1M шагов как smoke test. Проверить что reward растёт, TensorBoard показывает осмысленные кривые.

### Шаг 9: Imitation Learning (Фаза 4) — опционально
Записать демонстрации, pretrain, запустить fine-tuning PPO.

### Шаг 10: Полное обучение
Запустить 10M+ timesteps. Мониторить через дашборд и TensorBoard.

---

## Важные технические ограничения

1. **ReDroid требует kernel-модули binder и ashmem** — убедиться что они загружены на NAS. Если ядро не поддерживает — использовать AVD в headless режиме как fallback.

2. **HCR2 может требовать Google Play Services** для начальной активации — если так, использовать MicroG или образ ReDroid с GApps (`redroid/redroid:12.0.0_64only-gapps`).

3. **SubprocVecEnv + ADB** — каждый worker-процесс должен создавать своё ADB-соединение ПОСЛЕ fork'а. Нельзя шарить ADB-соединение между процессами.

4. **Memory leak в ADB** — при длительной работе `adb exec-out screencap` может утекать память. Реализовать периодический reconnect (каждые 1000 шагов).

5. **HCR2 Adventure Mode офлайн** — звёзды за рекорды не начисляются без интернета, но геймплей работает полностью. Для RL это не имеет значения.

6. **Эмуляторы не синхронизированы по времени** — каждый env работает со своей скоростью. SubprocVecEnv обрабатывает это корректно (ждёт самый медленный env на каждом step).

7. **GPU разделение** — эмуляторы используют software rendering на CPU, RTX 3090 целиком для PyTorch. Не запускать эмуляторы с GPU passthrough.

8. **OCR fallback** — если template matching не может распознать дистанцию 3 кадра подряд, использовать предыдущее значение и логировать проблемный кадр. Не обнулять дистанцию.

---

## Метрики успеха

- **Pipeline стабильность:** 8 параллельных сред работают без сбоев ≥ 4 часа
- **Throughput:** ≥ 60 steps/sec суммарно по всем средам
- **OCR accuracy:** ≥ 95% reliable reads для дистанции
- **Обучение:** reward монотонно растёт после 500K шагов
- **Результат:** обученный агент проезжает ≥ 1000м в Adventure Mode (Countryside)
- **Baseline:** rule-based агент проезжает ≥ 200м

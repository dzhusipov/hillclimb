# Hill Climb Racing AI Agent

AI-агент, который играет в [Hill Climb Racing](https://play.google.com/store/apps/details?id=com.fingersoft.hillclimb) на Android-устройстве. Читает экран телефона через scrcpy, анализирует состояние игры с помощью OpenCV и управляет машиной через ADB — от простых правил до обучаемого RL-агента (PPO).

## Как это работает

```
Android (Hill Climb Racing)
    │
    ▼  scrcpy (USB, ~35ms)
    │
Screen Capture (mss / ADB fallback)
    │
    ▼
CV Module (OpenCV)
    ├── Game State Classifier   (menu / racing / crash / results)
    ├── Gauge Reader            (fuel, RPM, boost → 0–100%)
    ├── Vehicle Tilt Detector   (угол наклона машины)
    ├── Terrain Analyzer        (наклон поверхности)
    └── Speed Estimator         (optical flow)
    │
    ▼  State Vector [fuel, rpm, boost, tilt, slope, airborne, speed]
    │
Agent (rule-based или PPO)
    │
    ▼  Action: nothing / gas / brake
    │
ADB Controller (input swipe)
    │
    ▼  Touch events → Android
```

Цикл работает на ~10 решений/сек (~100ms на итерацию).

## Требования

- **macOS** (Apple Silicon) или Linux
- **Python 3.11** (через conda)
- **Android-устройство** с Hill Climb Racing, подключённое по USB
- **USB Debugging** включён на устройстве

### Системные зависимости

```bash
brew install scrcpy android-platform-tools
```

## Установка

```bash
# Клонировать репозиторий
git clone <repo-url> && cd hillclimb

# Создать conda-окружение
conda env create -f environment.yml
conda activate hillclimb
```

## Быстрый старт

### 1. Подключить телефон и запустить scrcpy

```bash
# Убедиться что устройство видно
adb devices

# Запустить зеркалирование экрана
scrcpy
```

### 2. Откалибровать области экрана

```bash
python -m hillclimb.calibrate
```

Интерактивная утилита: переключай режимы через `TAB`, выделяй области мышкой, нажми `s` для сохранения.

| Клавиша | Действие |
|---------|----------|
| `TAB` | Переключить режим (fuel ROI, vehicle ROI, кнопки...) |
| `s` | Сохранить конфигурацию в `config.json` |
| `d` | Показать/скрыть debug overlay |
| `q` | Выйти |

### 3. Протестировать управление

```bash
python -m hillclimb.controller --test
```

Машина выполнит: газ 2 сек → тормоз 1 сек → газ 3 сек.

### 4. Запустить rule-based агента

```bash
python -m hillclimb.game_loop --agent rules
```

Агент будет играть бесконечно (Ctrl+C для остановки). Для ограничения по эпизодам:

```bash
python -m hillclimb.game_loop --agent rules --episodes 5
```

### 5. Обучить RL-агента (PPO)

```bash
# Обучение (~1000 эпизодов)
python -m hillclimb.train --episodes 1000

# Или по количеству шагов
python -m hillclimb.train --timesteps 200000

# С визуализацией
python -m hillclimb.train --render

# Продолжить обучение с чекпоинта
python -m hillclimb.train --resume models/ppo_hillclimb.zip
```

Прогресс обучения можно отслеживать через TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
```

### 6. Оценить и запустить обученную модель

```bash
# Оценка (10 эпизодов, статистика)
python -m hillclimb.evaluate --episodes 10

# Запустить RL-агента в игре
python -m hillclimb.game_loop --agent rl
```

## Структура проекта

```
hillclimb/
├── hillclimb/
│   ├── config.py          # Координаты кнопок, ROI, пороги
│   ├── capture.py         # Захват экрана (scrcpy/mss + ADB fallback)
│   ├── vision.py          # CV: state classifier, gauge reader, tilt, terrain
│   ├── controller.py      # ADB input: gas/brake/tap
│   ├── navigator.py       # Навигация по меню, авто-рестарт
│   ├── calibrate.py       # Интерактивная калибровка ROI
│   ├── agent_rules.py     # Rule-based baseline агент
│   ├── agent_rl.py        # RL агент (обёртка над PPO)
│   ├── env.py             # Gymnasium environment
│   ├── game_loop.py       # Основной цикл: capture → CV → agent → input
│   ├── logger.py          # CSV логирование + PNG кадры
│   ├── train.py           # Обучение PPO
│   └── evaluate.py        # Оценка модели
├── tests/
│   └── test_vision.py     # Тесты CV модуля
├── models/                # Сохранённые RL модели
├── logs/                  # Логи обучения и игровых сессий
├── templates/             # Шаблоны для template matching
├── environment.yml        # Conda environment
└── config.json            # Конфигурация (создаётся калибровкой)
```

## Конфигурация

Координаты кнопок и ROI-областей настраиваются тремя способами:

1. **Интерактивно:** `python -m hillclimb.calibrate` — drag-and-drop регионов на экране
2. **Вручную:** редактирование `config.json`
3. **По умолчанию:** значения в `hillclimb/config.py` (Galaxy S-series, landscape)

## RL Agent: детали

**State vector** — 7 значений float32 [0..1]:

| Индекс | Параметр | Описание |
|--------|----------|----------|
| 0 | fuel | Уровень топлива |
| 1 | rpm | Обороты двигателя |
| 2 | boost | Уровень буста |
| 3 | tilt | Наклон машины (нормализованный) |
| 4 | terrain_slope | Наклон поверхности (нормализованный) |
| 5 | airborne | В воздухе (0 или 1) |
| 6 | speed_estimate | Оценка скорости (optical flow) |

**Actions** — Discrete(3): `nothing`, `gas`, `brake`

**Reward:**

| Компонент | Вес | Описание |
|-----------|-----|----------|
| speed_estimate | +1.0 | Ехать вперёд |
| crashed | -10.0 | Штраф за переворот |
| fuel_consumed | -0.1 | Экономия топлива |
| moving with fuel | +0.5 | Бонус за движение |
| extreme tilt (>45 deg) | -0.3 | Штраф за нестабильность |

## Тесты

```bash
conda activate hillclimb
python -m pytest tests/ -v
```

## Технологии

- **Computer Vision:** OpenCV (HSV-сегментация, Canny edges, HoughLines, optical flow)
- **Reinforcement Learning:** Stable-Baselines3 PPO, Gymnasium
- **Screen Capture:** mss (fast) + ADB screencap (fallback)
- **Input:** ADB shell input swipe
- **Hardware Acceleration:** PyTorch MPS (Apple Silicon), CUDA (Nvidia)

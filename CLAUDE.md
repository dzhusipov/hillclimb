# Hill Climb Racing 2 AI Agent

## Что это
AI-агент, который играет в Hill Climb Racing 2 на Android (Redmi Note 8 Pro, 2340x1080). Читает экран телефона через scrcpy, анализирует состояние игры через OpenCV + Tesseract OCR, принимает решения (rule-based или PPO RL), и отправляет команды управления через ADB.

## Архитектура
```
Android (HCR2) → scrcpy (USB) → Mac Screen Capture (mss)
→ CV Module (OpenCV + Tesseract OCR):
  - State classifier (8 game states)
  - Circular dial gauge reader (RPM, Boost via red needle detection)
  - Horizontal bar gauge reader (Fuel)
  - Distance OCR ("103m" text)
  - Tilt, terrain, airborne, speed detection
→ Agent (rules / PPO RL): решение {nothing, gas, brake}
→ ADB Controller → touch events → Android
```

## Conda-окружение
```bash
conda activate hillclimb
```
Создано из `environment.yml`. Python 3.11, PyTorch (MPS на Mac), OpenCV, stable-baselines3, pytesseract.

## Запуск тестов
```bash
conda activate hillclimb
python -m pytest tests/ -v
```

## Основные команды
```bash
# Калибровка CV (интерактивная настройка ROI, кнопок, диалов)
python -m hillclimb.calibrate

# Тест контроллера (gas 2s → brake 1s → gas 3s)
python -m hillclimb.controller --test

# Rule-based агент (основной игровой цикл)
python -m hillclimb.game_loop --agent rules
python -m hillclimb.game_loop --agent rules --episodes 5 --headless

# Обучение RL агента
python -m hillclimb.train --episodes 1000
python -m hillclimb.train --timesteps 200000 --render
python -m hillclimb.train --resume models/ppo_hillclimb.zip

# Оценка обученной модели
python -m hillclimb.evaluate --episodes 10

# Запуск RL агента в игре
python -m hillclimb.game_loop --agent rl
```

## Структура проекта
```
hillclimb/
├── hillclimb/
│   ├── config.py        — координаты кнопок, ROI (Rect+CircleROI), OCR, пороги
│   ├── capture.py       — захват экрана (scrcpy/mss + ADB fallback)
│   ├── vision.py        — CV: 8 states, dial reader, OCR, tilt, terrain
│   ├── controller.py    — ADB input: gas/brake/tap
│   ├── navigator.py     — state machine навигация (8 состояний)
│   ├── agent_rules.py   — rule-based baseline агент
│   ├── agent_rl.py      — RL агент (обёртка над PPO)
│   ├── env.py           — Gymnasium environment для RL (8-dim obs)
│   ├── game_loop.py     — основной цикл capture→CV→agent→input
│   ├── logger.py        — CSV лог (distance_m, coins) + PNG кадры
│   ├── calibrate.py     — калибровка: ROI, кнопки, dial, needle angles
│   └── train.py         — скрипт обучения PPO
├── tests/
│   └── test_vision.py   — 34 теста: dials, OCR, classifier, navigator
├── models/              — сохранённые RL модели
├── logs/                — логи обучения + игровые логи
├── templates/           — шаблоны для template matching
└── environment.yml      — conda environment
```

## Prerequisites
1. scrcpy: `brew install scrcpy`
2. ADB: `brew install android-platform-tools`
3. Tesseract: `brew install tesseract` (at `/opt/homebrew/bin/tesseract`)
4. Android подключён по USB, USB Debugging включён
5. `scrcpy` запущен — экран телефона виден на Mac

## Game States (8 states)
```
UNKNOWN → MAIN_MENU → VEHICLE_SELECT → RACING →
DRIVER_DOWN → TOUCH_TO_CONTINUE → RESULTS → (retry) → VEHICLE_SELECT
                                           ↗
DOUBLE_COINS_POPUP → (skip) → RACING
```

## Navigator State Machine
| State | Action | Wait | Expect |
|-------|--------|------|--------|
| MAIN_MENU | tap race_button | 2s | VEHICLE_SELECT |
| VEHICLE_SELECT | tap start_button | 3s | RACING |
| DOUBLE_COINS_POPUP | tap skip_button | 2s | RACING |
| DRIVER_DOWN | tap center_screen | 1s | TOUCH_TO_CONTINUE |
| TOUCH_TO_CONTINUE | tap center_screen | 1.5s | RESULTS |
| RESULTS | read OCR → tap retry | 2s | VEHICLE_SELECT |
| UNKNOWN | tap center_screen | 1s | retry |

Stuck detection: same state 3 cycles → fallback tap center.

## Конфигурация
Координаты кнопок и ROI регионов настраиваются через:
1. `python -m hillclimb.calibrate` — интерактивно (TAB по режимам)
2. Редактирование `config.json` — вручную
3. Дефолты в `hillclimb/config.py` — для Redmi Note 8 Pro 2340x1080 landscape

### Калибровка диалов
В calibrate.py, режим RPM_DIAL / BOOST_DIAL:
1. Кликнуть центр диала
2. Кликнуть край → задаётся CircleROI
3. Нажать 't' — включить тест overlay (видна маска иглы + угол)
4. Нажать 'n' при игле на 0% → записать min angle
5. Нажать 'm' при игле на 100% → записать max angle
6. Нажать 's' — сохранить

## State Vector (вход RL агента)
8 значений float32 [0..1]:
```
[fuel, rpm, boost, tilt_norm, terrain_slope_norm, airborne, speed_estimate, distance_norm]
```

## Actions
Discrete(3): `0=nothing, 1=gas, 2=brake`

## Reward Function
```
+0.1 * distance_delta_m    — метры вперёд (OCR), fallback speed_estimate
-10.0 * crashed            — штраф за DRIVER_DOWN
-0.1 * fuel_consumed       — экономия топлива
+0.5 * (fuel>0 and moving) — бонус за движение
-0.3 * (|tilt|>45°)        — штраф за экстремальный наклон
```

# Hill Climb Racing AI Agent

## Что это
AI-агент, который играет в Hill Climb Racing на Android. Читает экран телефона через scrcpy, анализирует состояние игры через OpenCV, принимает решения (rule-based или PPO RL), и отправляет команды управления через ADB.

## Архитектура
```
Android (Hill Climb Racing) → scrcpy (USB) → Mac Screen Capture (mss)
→ CV Module (OpenCV): state classifier + gauge reader + tilt detection
→ Agent (rules / PPO RL): решение {nothing, gas, brake}
→ ADB Controller → touch events → Android
```

## Conda-окружение
```bash
conda activate hillclimb
```
Создано из `environment.yml`. Python 3.11, PyTorch (MPS на Mac), OpenCV, stable-baselines3.

## Запуск тестов
```bash
conda activate hillclimb
python -m pytest tests/ -v
```

## Основные команды
```bash
# Phase 1: Калибровка CV (интерактивная настройка ROI)
python -m hillclimb.calibrate

# Phase 2: Тест контроллера (gas 2s → brake 1s → gas 3s)
python -m hillclimb.controller --test

# Phase 3: Rule-based агент (основной игровой цикл)
python -m hillclimb.game_loop --agent rules
python -m hillclimb.game_loop --agent rules --episodes 5 --headless

# Phase 4: Обучение RL агента
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
│   ├── config.py        — координаты кнопок, ROI, пороги (+ config.json)
│   ├── capture.py       — захват экрана (scrcpy/mss + ADB fallback)
│   ├── vision.py        — CV: game state classifier, gauge reader, tilt, terrain
│   ├── controller.py    — ADB input: gas/brake/tap
│   ├── navigator.py     — навигация по меню, авто-рестарт
│   ├── agent_rules.py   — rule-based baseline агент
│   ├── agent_rl.py      — RL агент (обёртка над PPO)
│   ├── env.py           — Gymnasium environment для RL
│   ├── game_loop.py     — основной цикл capture→CV→agent→input
│   ├── logger.py        — CSV лог + PNG кадры
│   ├── calibrate.py     — интерактивная калибровка ROI
│   └── train.py         — скрипт обучения PPO
├── tests/
│   └── test_vision.py   — тесты CV модуля (synthetic frames)
├── models/              — сохранённые RL модели
├── logs/                — логи обучения + игровые логи
├── templates/           — шаблоны для template matching (menu.png, crash.png)
└── environment.yml      — conda environment
```

## Prerequisites
1. scrcpy: `brew install scrcpy`
2. ADB: `brew install android-platform-tools`
3. Android подключён по USB, USB Debugging включён
4. `scrcpy` запущен — экран телефона виден на Mac

## Фазы разработки
- **Phase 1** (Screen Capture + CV): capture.py, vision.py, calibrate.py — читаем экран, определяем состояние
- **Phase 2** (Input Controller): controller.py, navigator.py — управляем игрой через ADB
- **Phase 3** (Rule-Based Agent): agent_rules.py, game_loop.py, logger.py — замыкаем цикл с простыми правилами
- **Phase 4** (RL Agent): env.py, agent_rl.py, train.py — обучаем PPO играть лучше правил

## Конфигурация
Координаты кнопок и ROI регионов настраиваются через:
1. `python -m hillclimb.calibrate` — интерактивно
2. Редактирование `config.json` — вручную
3. Дефолты в `hillclimb/config.py` — для Galaxy S-series landscape

## State Vector (вход RL агента)
7 значений float32 [0..1]: `[fuel, rpm, boost, tilt_norm, terrain_slope_norm, airborne, speed_estimate]`

## Actions
Discrete(3): `0=nothing, 1=gas, 2=brake`

## Reward Function
```
+1.0 * speed_estimate       — ехать вперёд
-10.0 * crashed             — штраф за переворот
-0.1 * fuel_consumed        — экономия топлива
+0.5 * (fuel>0 and moving)  — бонус за движение
-0.3 * (|tilt|>45°)         — штраф за экстремальный наклон
```

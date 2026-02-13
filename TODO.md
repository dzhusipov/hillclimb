# TODO

## [DONE] Ускорение capture: scrcpy видеострим
scrcpy-server v2.4 + PyAV H.264 декодер. `capture()` ≈ 0ms вместо ~400ms screencap.

- [x] `hillclimb/scrcpy_capture.py` — ScrcpyCapture класс (raw H.264 → PyAV decode → frame buffer)
- [x] `hillclimb/capture.py` — фабрика `create_capture()` с fallback scrcpy → raw
- [x] `hillclimb/config.py` — параметры scrcpy (max_fps, max_size, bitrate, server_jar)
- [x] `hillclimb/env.py`, `game_loop.py` — переведены на `create_capture()`
- [x] `scripts/profile_run.py` — `--backend scrcpy` для A/B сравнения
- [x] `vendor/scrcpy-server.jar` — скачан с GitHub releases
- [x] `requirements.txt` — добавлен `av>=12.0`

## [DONE] Navigator: адаптивный polling вместо фиксированных sleep
- [x] `_wait_transition()` — polling каждые 150ms вместо жёстких sleep(2-3s)
- [x] Все переходы (MAIN_MENU, VEHICLE_SELECT, DRIVER_DOWN, RESULTS и др.)
- [x] `_dismiss_popups`, `_relaunch_game`, `_solve_captcha` — уменьшены таймауты

## [DONE] Dashboard: WebSocket streaming 30 FPS
- [x] `web/server.py` — `/ws/stream/{emu_id}` WebSocket endpoint
- [x] `web/static/app.js` — WebSocket клиент, auto-reconnect
- [x] `web/streamer.py` — ScrcpyCapture интеграция, fallback на screencap
- [x] `docker/Dockerfile.dashboard` — av + scrcpy-server.jar
- [x] Пересобран образ, CasaOS контейнер перезапущен

## [DONE] Переключить game_loop / train на scrcpy backend
scid баг пофикшен (0x7FFFFFFF), capture_backend = "scrcpy" в config.py.
Стабильно работает на 8 эмуляторах, ~27 FPS обучения.

- [x] Фикс scid overflow (Java Integer.parseInt signed 32-bit)
- [x] `capture_backend: str = "scrcpy"` в config.py
- [x] Стабильная работа на 8 эмуляторах (подтверждено при обучении)
- [x] No periodic codec reset, AVError catch, 30s stale watchdog

## [PARTIALLY DONE] Dashboard: панель обучения
Базовые метрики реализованы (JSONL логи + API + Chart.js). Остались расширенные метрики.

### [DONE] Базовые метрики
- [x] JSONL логи: `logs/train_episodes.jsonl`, `logs/nav_events.jsonl`, `logs/training_status.json`
- [x] API: `/api/training/{status,episodes,events}` — server.py читает из logs/
- [x] Chart.js графики (distance + reward), stats grid, nav events log, 5s polling
- [x] Docker: `logs/` монтируется как `/app/logs:ro`

### Карточки эмуляторов — расширенные метрики
- [ ] Game state (RACING / DRIVER_DOWN / RESULTS / ...)
- [ ] Текущий reward эпизода
- [ ] Distance (текущая / max за эпизод)
- [ ] Fuel bar
- [ ] Текущее действие агента (GAS / BRAKE / NOTHING)

### Бэкенд — управление обучением
- [ ] `POST /api/training/start` / `stop` / `pause` — управление обучением
- [ ] Shared state через файл или multiprocessing.Manager (train.py → dashboard)

## Ускорение эмуляторов (speedhack)
Ускорить игровое время в 2-3x = пропорционально больше эпизодов.

### GameGuardian (самый простой)
- [ ] Скачать APK, установить на hcr2-0
- [ ] Выбрать HCR2, поставить 2x speed
- [ ] Проверить что vision classifier работает на ускоренной игре
- [ ] Если ОК — установить на все 8

### LD_PRELOAD libgamespeed.so (если GameGuardian не работает)
- [ ] Найти/собрать libgamespeed.so для x86_64 Android 14
- [ ] `LD_PRELOAD=/path/to/lib am start -n com.fingersoft.hcr2/.AppActivity`
- [ ] Подменяет `clock_gettime` — игра думает что время идёт быстрее

## Оптимизация обучения
- [x] Обучение с scrcpy capture — ~27 FPS (8 эмуляторов)
- [x] `--resume` работает, `--skip-envs` для исключения эмуляторов
- [ ] Проверить reward рост в TensorBoard после фиксов навигации
- [ ] Запустить длинное обучение (1M+ timesteps) с новыми фиксами
- [ ] Попробовать 12-16 эмуляторов (Ryzen 5600 может потянуть)

## [DONE] Vision + Navigator фиксы (2026-02-13)
Серия фиксов для надёжной навигации меню между эпизодами.

- [x] **green_bl_btn**: яркий зелёный HSV [35,100,130]-[85,255,255] — отличает UI кнопки от травы (Countryside)
- [x] **OFFLINE step 0b**: раннее определение OFFLINE popup → MAIN_MENU (overall_v<100 + tab_bright>0.05 + dark_center>50%)
- [x] **TOUCH_TO_CONTINUE**: tap (400,460) вместо center (400,240) — center поглощается stats card
- [x] **RESULTS animation wait**: 1.5s перед первым RETRY tap (кнопки не интерактивны во время анимации)
- [x] **RESULTS double-tap**: RETRY + 0.3s + RETRY для надёжности
- [x] **RETRY/NEXT координаты**: Y=448 (было 430, не попадало в кнопку)
- [x] **RESULTS stuck fallback**: tap RETRY вместо бесполезного center
- [x] **Nav frame logging**: JPEG-скриншоты на каждый ensure_racing() → `logs/nav_frames/`
- [x] **Mid-race interrupts**: env.py ловит CAPTCHA, DOUBLE_COINS, MAIN_MENU, VEHICLE_SELECT во время RACING
- [x] **--skip-envs**: исключение эмуляторов из обучения (`--skip-envs 0 3`)

### Нерешённое
- [ ] RESULTS RETRY всё ещё ~48% s1 rate — нужно расследование (возможно анимация длиннее 1.5s)

## Мелкие задачи
- [ ] iptables правила в автозагрузку (пропадают после ребута)
- [ ] Отключить анимации Android на всех эмуляторах:
  ```bash
  for p in 5555 5556 5557 5558 5559 5560 5561 5562; do
    adb -s localhost:$p shell settings put global window_animation_scale 0
    adb -s localhost:$p shell settings put global transition_animation_scale 0
    adb -s localhost:$p shell settings put global animator_duration_scale 0
  done
  ```

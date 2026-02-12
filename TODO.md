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

## [IN PROGRESS] Переключить game_loop / train на scrcpy backend
scid баг пофикшен (0x7FFFFFFF), capture_backend переключен на "scrcpy" в config.py.
Нужно провалидировать на profile_run.

- [x] Фикс scid overflow (Java Integer.parseInt signed 32-bit)
- [x] `capture_backend: str = "scrcpy"` в config.py
- [ ] Прогнать `profile_run.py --backend scrcpy` — убедиться что capture ≈ 0ms
- [ ] Прогнать game_loop на 8 эмуляторах с scrcpy — проверить стабильность
- [ ] Закоммитить финальное переключение

## Dashboard: панель обучения
Сейчас дашборд показывает стримы и Docker-статусы. Нужна информация об обучении.

### Панель метрик (верхняя полоса)
- [ ] Статус обучения: running / paused / stopped
- [ ] Total timesteps / target
- [ ] Elapsed time
- [ ] Steps/sec (текущий throughput)
- [ ] Episodes completed

### Карточки эмуляторов — добавить RL-метрики
- [ ] Game state (RACING / DRIVER_DOWN / RESULTS / ...)
- [ ] Текущий reward эпизода
- [ ] Distance (текущая / max за эпизод)
- [ ] Fuel bar
- [ ] Текущее действие агента (GAS / BRAKE / NOTHING)

### Графики (нижняя панель)
- [ ] Episode reward (скользящее среднее)
- [ ] Best distance по эпизодам
- [ ] Steps per episode
- [ ] Использовать Chart.js (lightweight, без сборки)

### Бэкенд
- [ ] `POST /api/training/start` / `stop` / `pause` — управление обучением
- [ ] `GET /api/training/status` — текущие метрики (JSON)
- [ ] `GET /api/training/history` — история эпизодов (последние N)
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
- [ ] Прогнать обучение с scrcpy capture — ожидаемо 2-3x ускорение
- [ ] Проверить reward рост в TensorBoard
- [ ] Запустить 500k+ с `--resume`
- [ ] Попробовать 12-16 эмуляторов (Ryzen 5600 может потянуть)

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
- [ ] Собрать debug-кадры UNKNOWN стейта для анализа ложных срабатываний

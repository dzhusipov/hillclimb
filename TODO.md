# TODO

## Ускорение capture: scrcpy видеострим
Сейчас screencap = 328ms (97.7% step time). Переход на scrcpy H264 stream даст 30+ FPS.

- [ ] Запустить `scrcpy --no-display --video-codec=h264 --max-fps=15 --max-size=480` с pipe output
- [ ] Декодировать H264 через `av` (PyAV) или FFmpeg subprocess
- [ ] Кольцевой буфер: последний декодированный кадр, `capture()` возвращает мгновенно
- [ ] Фоновый поток на каждый эмулятор: scrcpy → decode → buffer
- [ ] Сравнить latency vs текущий RAW screencap
- [ ] Проверить стабильность на 8 параллельных потоках

Ожидаемый эффект: step 336ms → ~50-100ms (capture ≈ 0ms, ограничение = action_hold 200ms).

## Dashboard: панель обучения
Сейчас дашборд показывает только стримы и Docker-статусы. Нужна информация об обучении.

### Панель метрик (верхняя полоса)
- [ ] Статус обучения: running / paused / stopped
- [ ] Total timesteps / target
- [ ] Elapsed time
- [ ] Steps/sec (текущий throughput)
- [ ] Episodes completed

### Карточки эмуляторов — добавить RL-метрики
- [ ] Game state (RACING / DRIVER_DOWN / RESULTS / ...)
- [ ] Текущий reward эпизода (累積)
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
- [ ] WebSocket для push-обновлений (вместо polling)

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
- [ ] Запустить первое обучение с CNN (100k timesteps, 8 envs)
- [ ] Проверить что reward растёт в TensorBoard
- [ ] Если ОК — запустить 500k+ с `--resume`
- [ ] Попробовать 12-16 эмуляторов (Ryzen 5600 может потянуть)
- [ ] Уменьшить sleep'ы в navigator.py (2.0s → 1.0-1.5s)

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

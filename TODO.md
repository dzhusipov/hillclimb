# TODO — следующая сессия

## Проверить результаты ночного обучения
- [ ] Процесс жив? `ps aux | grep hillclimb.train`
- [ ] Сколько эпизодов? `grep "^  EP" logs/train_run.log | wc -l`
- [ ] Последние эпизоды: `grep "^  EP" logs/train_run.log | tail -20`
- [ ] Best distance: `grep "best=" logs/train_run.log | tail -1`
- [ ] Модель сохранилась? `ls -la models/ppo_hillclimb*`
- [ ] Чекпоинты: `ls -la models/checkpoints/`
- [ ] TensorBoard: `tensorboard --logdir logs/tensorboard`
- [ ] Дашборд: http://NAS-IP:8150 — все 8 эмуляторов живы?

## Исправить проблему UNKNOWN (30% эпизодов)
30% эпизодов завершаются со статусом UNKNOWN — агент не может классифицировать экран.
Это плохо для обучения: episode termination срабатывает без ясной причины.

- [ ] Собрать debug-кадры UNKNOWN: `ls logs/nav_debug/*_unknown.png`
- [ ] Посмотреть что на экране в момент UNKNOWN — какой это реальный стейт?
- [ ] Добавить сохранение debug-кадра в `env.py` при завершении эпизода на UNKNOWN
- [ ] Исправить классификатор `vision.py` для пропущенных стейтов
- [ ] Возможные причины:
  - Переходные кадры между стейтами (анимация загрузки)
  - Новые попапы которые не обрабатываются
  - Тёмные/необычные карты с нестандартными цветами

## Ускорение эмуляторов (главный буст)
Bottleneck — НЕ GPU/CPU, а скорость игры в реальном времени.
Ускорение игры в 2-3x = 2-3x больше эпизодов = 100k за 5-8ч вместо 14ч.

### Вариант 1: Отключить анимации Android (просто, сразу)
Ускоряет переходы по меню на ~1-2с. Применить на все 8 эмуляторов:
```bash
for p in 5555 5556 5557 5558 5559 5560 5561 5562; do
  adb -s localhost:$p shell settings put global window_animation_scale 0
  adb -s localhost:$p shell settings put global transition_animation_scale 0
  adb -s localhost:$p shell settings put global animator_duration_scale 0
done
```

### Вариант 2: GameGuardian (speedhack 2-3x)
- Работает на root-устройствах (ReDroid = root по дефолту)
- Ставим APK, выбираем HCR2, ставим скорость 2x-3x
- Эмуляторы офлайн → античит не сработает
- **Проще всего попробовать**, результат сразу виден
- [ ] Скачать GameGuardian APK
- [ ] Установить на один эмулятор, протестировать
- [ ] Если работает — поставить на все 8

### Вариант 3: Xposed/LSPosed + GameSpeed модуль
- Надёжнее чем GameGuardian (работает на уровне фреймворка)
- Перехватывает `clock_gettime`, `System.nanoTime()` и т.д.
- Сложнее в установке (нужен LSPosed framework на ReDroid)
- [ ] Проверить совместимость LSPosed с ReDroid 14

### Вариант 4: LD_PRELOAD libgamespeed.so
- Самый низкоуровневый — инжект библиотеки через LD_PRELOAD
- Подменяет системные часы для конкретного процесса
- Минимальный overhead, максимальная совместимость
- [ ] Найти/собрать libgamespeed.so для x86_64
- [ ] Протестировать: `LD_PRELOAD=/path/to/lib am start ...`

### Ожидаемый эффект
| Скорость | Эпизод | Timesteps/мин | 100k за |
|----------|--------|--------------|---------|
| 1x (сейчас) | ~90с | ~115 | ~14ч |
| 2x | ~45с | ~200 | ~8ч |
| 3x | ~30с | ~300 | ~5.5ч |

## Оптимизация обучения
- [ ] Увеличить steps на эпизод (сейчас 3-24, нужно 50+)
- [ ] Проанализировать reward function — достаточно ли сигнала?
- [ ] Рассмотреть device='cpu' (PPO с MlpPolicy быстрее на CPU)
- [ ] Уменьшить sleep'ы в navigator.py (много `time.sleep(2.0)`)
- [ ] Попробовать 12-16 эмуляторов на NAS (Ryzen 5600 может потянуть)
- [ ] Если 100k завершилось — запустить 500k с `--resume`

## Windows PC как дополнительный ресурс
PC: i9-11400K, RTX 3080Ti, 80GB RAM, Windows 11.
Вывод: **малополезен** для текущей задачи. Bottleneck = скорость игры, не compute.

| Вариант | Проблема |
|---------|----------|
| ReDroid на Windows | Не работает — нужны Linux kernel modules |
| ReDroid в WSL2 | Модули ядра обычно отсутствуют, нестабильно |
| Android Studio AVD | Тяжёлые эмуляторы, другая настройка, но возможно |
| Сетевой ADB | Latency по сети убьёт screencap |

Если очень хочется:
- [ ] Попробовать WSL2 + ReDroid (неочевидно что заработает)
- [ ] Или Android Studio AVD с headless эмуляторами

## Мелкие задачи
- [ ] iptables для блокировки интернета (если не активны после ребута)
- [ ] Проверить CAPTCHA-логи: `grep "CAPTCHA" logs/train_run.log | wc -l`

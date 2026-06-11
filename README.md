# 🎮 Brawlhalla AI

Интеллектуальный бот для [Brawlhalla](https://brawlhalla.com/) с компьютерным зрением, Kalman-трекингом и reinforcement learning.

## Архитектура

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Screen Cap │────>│ YOLO Vision  │────>│ Game State  │
│  (mss/opencv)│    │ (GPU/CUDA)   │     │ (Kalman F.) │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐            │
│  Input Ctrl │<────│   AI Brain   │<───────────┘
│ (keyboard)  │     │ (RL + Rules) │
└─────────────┘     └──────────────┘
```

## Модули

| Файл | Описание |
|------|----------|
| `main.py` | Точка входа — запуск AI-бота |
| `ai_brain.py` | Боевой ИИ: адаптивное обучение, стратегическое принятие решений, паттерны врагов |
| `game_state.py` | Построение состояния игры: Kalman-фильтрация 2D, blast-zone awareness, предсказание траекторий |
| `vision_client.py` | YOLO-инференс на GPU (RTX): детекция enemy/player/weapon/gadget с временным сглаживанием |
| `ai_runner.py` | Цикл AI: frame budget management, stall recovery, adaptive FPS optimization |
| `input_controller.py` | Управление клавиатурой: нажатия, задержки, комбо-ввод |
| `screen_capture.py` | Захват экрана игры через mss/opencv с автодетекцией окна |
| `movement_ai.py` | AI-движение: уклонение, позиционирование, edge recovery |
| `neural_ai.py` | Нейросетевой модуль для предсказания действий |
| `rl_trainer.py` | Reinforcement Learning тренер (PPO-style) |
| `train_model.py` | Обучение YOLO-модели на датасете Brawlhalla |
| `train_unified.py` | Унифицированный пайплайн обучения: vision + RL вместе |
| `download_model.py` | Скачивание предобученных моделей с Roboflow/Kaggle |
| `replay_system.py` | Система записи и воспроизведения матчей для анализа |
| `gui.py` | PyQt6 интерфейс управления ботом |
| `config.py` | Конфигурация: профили персонажей, matchup-таблицы, параметры обучения |

## Возможности

### Компьютерное зрение
- **YOLOv8** детекция в реальном времени (45 FPS target)
- 4 класса: `enemy`, `player`, `weapon`, `gadget`
- Temporal smoothing — сглаживание позиций между кадрами
- Adaptive confidence threshold

### Трекинг и предсказание
- **Kalman Filter 1D/2D** — фильтрация шума, предсказание траекторий
- Blast-zone awareness — отслеживание зон выброса
- Edge detection + recovery path planning

### ИИ и обучение
- Reinforcement Learning (PPO-style trainer)
- Enemy pattern learning — запоминание паттернов противника
- Strategic action scoring — оценка действий по контексту матча
- Character-specific profiles и matchup таблицы

## Установка

```bash
# Клонировать репозиторий
git clone https://github.com/kamil12431234/brawlhalla-ai.git
cd brawlhalla-ai

# Создать виртуальное окружение
python -m venv .venv && source .venv/bin/activate

# Установить зависимости
pip install -r requirements.txt

# Скачать модель (если нужно)
python download_model.py
```

## Быстрый старт

```bash
# Запустить бота
./run.sh

# Или через Python напрямую
python main.py

# Запустить GUI
python gui.py
```

### Конфигурация

Переменные окружения:
- `BH_CHARACTER` — персонаж (`balanced`, `aggressive`, `defensive`)
- `ROBOFLOW_API_KEY` — ключ для скачивания моделей с Roboflow
- `ROBOFLOW_VERSION` — версия модели (default: 5)

## Обучение модели

```bash
# Датасеты
# Kaggle: https://www.kaggle.com/datasets/patrickgomes/brawlhalla-dataset
# Roboflow: https://universe.roboflow.com/rasheds-workspace/brawlhalla-vision

# Обучить YOLO модель
python train_model.py

# Унифицированное обучение (vision + RL)
python train_unified.py

# RL тренировка отдельно
python rl_trainer.py
```

## Системные требования

- **OS**: Linux (Arch/Ubuntu), Windows с WSL
- **GPU**: NVIDIA (CUDA 12+) — RTX 3060+ рекомендуется
- **RAM**: 8GB минимум
- **Python**: 3.10+

## Датасеты

| Источник | Описание | Ссылка |
|----------|----------|--------|
| Kaggle | Brawlhalla dataset для AI training | [brawlhalla-dataset](https://www.kaggle.com/datasets/patrickgomes/brawlhalla-dataset) |
| Roboflow | Vision detection dataset (annotated) | [brawlhalla-vision](https://universe.roboflow.com/rasheds-workspace/brawlhalla-vision) |

## Лицензия

MIT

# Brawlhalla AI — Vision-Based Game Bot

Autonomous AI player for Brawlhalla using computer vision (YOLOv8), Kalman filtering, and adaptive decision-making.

## Features

- **Real-time object detection** — YOLOv8 ONNX model running on GPU (CUDA)
- **Advanced tracking** — 2D Kalman filters with temporal smoothing and predictive tracking
- **Adaptive confidence threshold** — dynamically adjusts based on inference performance
- **Multiple game modes**:
  - `full` — complete autonomous control
  - `assist` — AI assists while you control movement
  - `combat` — attacks/shields only, manual movement
  - `poke` — light attacks only, in-range engagement
- **Stall detection & recovery** — automatic recovery from stuck states
- **Performance monitoring** — real-time FPS, inference time, and pipeline stats

## Architecture

```
capture_frame -> vision_client.infer() -> build_world_state()
  -> ai.choose_action() -> resolve_action() -> input_controller.apply_keys()
```

### Components

| Module | Purpose |
|--------|---------|
| `vision_client.py` | YOLOv8 ONNX inference with GPU acceleration |
| `game_state.py` | Kalman tracking, world state building, blast zone awareness |
| `ai_brain.py` | Decision engine with character-specific profiles |
| `ai_runner.py` | Main game loop with async pipeline support |
| `input_controller.py` | Keyboard/mouse input simulation via X11 |
| `config.py` | Character profiles, matchup data, training config |

## Requirements

- **Python 3.10+**
- **NVIDIA GPU** (CUDA-enabled, tested with RTX 3070 Ti / 3090)
- **Linux** (X11 window capture via `xdotool`)
- Dependencies: `numpy`, `opencv-python`, `onnxruntime-gpu`

### Install dependencies

```bash
cd brawlhalla-ai
pip install numpy opencv-python onnxruntime-gpu
```

## Usage

### Full autonomous mode

```bash
python main.py
```

### Assist mode (you move, AI attacks)

```bash
python main.py --assist
```

### Combat mode (attacks/shields only)

```bash
python main.py --combat
```

### Dry-run (no input, just logging)

```bash
python main.py --dry-run
```

### Custom character & aggression

```bash
python main.py --character bodvar --aggressive 0.7
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BH_MODEL_PATH` | `./model/brawlhalla_vision.onnx` | Path to ONNX model |
| `BH_DEVICE` | `cuda` | Inference device (`cuda` or `cpu`) |
| `BH_CONF_THRESH` | `0.15` | Confidence threshold |
| `BH_WIDTH` | `1280` | Capture window width |
| `BH_HEIGHT` | `720` | Capture window height |
| `ROBOFLOW_API_KEY` | — | Roboflow API key (for model updates) |

## Model

The project includes a pre-trained YOLOv8 model (`model/brawlhalla_vision.onnx`) detecting:
- `player` — your character
- `enemy` — opponent(s)
- `gadget` / `weapon` — pickups and projectiles

Train or update the model via Roboflow workspace `rasheds-workspace/brawlhalla-vision`.

## Performance

Target: **45 FPS** with ~13ms frame budget. Typical breakdown:
- Capture: <20ms
- Inference: <50ms (GPU)
- Decision: <5ms
- Input: <2ms

## Project Structure

```
brawlhalla-ai/
├── main.py              # Entry point
├── ai_runner.py         # Game loop & async pipeline
├── ai_brain.py          # AI decision engine
├── vision_client.py     # YOLOv8 inference
├── game_state.py        # Tracking & world state
├── input_controller.py  # X11 input simulation
├── config.py            # Config & character profiles
├── replay_system.py     # Replay recording/analysis
└── model/               # ONNX models
```

## License

MIT

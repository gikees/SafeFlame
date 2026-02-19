# SafeFlame

Real-time AI kitchen safety monitor that detects hazards and alerts you before accidents happen.

All inference runs locally — no cloud.

## Overview

SafeFlame watches a kitchen camera feed and combines YOLOv8 object detection with computer-vision heuristics (flame, smoke, boil-over, proximity) to track burner state in real time. A state machine escalates unattended-burner alerts through INFO → WARNING → CRITICAL severity levels. Voice alerts notify you immediately, a local LLM (Ollama llama3.1:8b) provides one-sentence safety advice, and a live web dashboard lets you monitor everything from another device.

## Architecture

```
Camera / Video
      │
      ▼
┌──────────┐    ┌──────────────────┐
│  YOLOv8  │    │  CV Heuristics   │
│ Detector │    │ flame · smoke ·  │
│          │    │ boilover · prox  │
└────┬─────┘    └───────┬──────────┘
     │                  │
     └──────┬───────────┘
            ▼
   ┌─────────────────┐
   │  State Machine   │
   │  (per-burner     │
   │   escalation)    │
   └───────┬─────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐ ┌──────────┐
│  Voice  │ │   LLM    │
│ Alerts  │ │ Advisor  │
│(pyttsx3)│ │ (Ollama) │
└─────────┘ └──────────┘
           │
           ▼
   ┌──────────────┐
   │  Dashboard   │
   │  (FastAPI +  │
   │  WebSocket)  │
   └──────────────┘
```

## Features

- **YOLOv8 object detection** — identifies people, appliances, and flammable objects in the frame
- **Flame detection** — HSV color-space thresholding with morphological filtering
- **Smoke detection** — gray-region HSV detection with temporal persistence to filter out steam
- **Boil-over detection** — motion + edge analysis above burner zones
- **Proximity alerts** — warns when flammable objects (bottles, phones, etc.) are near active burners
- **Unattended burner escalation** — three-tier state machine (INFO at 60s, WARNING at 3min, CRITICAL at 5min)
- **Voice alerts** — pyttsx3 TTS with severity-based volume, macOS/Linux system fallback
- **LLM safety advice** — local Ollama model gives one-sentence actionable tips for WARNING/CRITICAL events
- **Live web dashboard** — real-time video stream, alert feed, and zone management over WebSocket
- **Configurable burner zones** — draw zones in the dashboard UI or POST to `/api/zones`

## Setup

### Prerequisites

- Python 3.10+
- A webcam or video file of a kitchen
- [Ollama](https://ollama.ai) (optional, for LLM safety advice)

### Install

```bash
pip install -r requirements.txt
```

If using LLM advice, pull a model:

```bash
ollama pull llama3.1:8b
```

### Configuration

All thresholds, timers, and settings live in `config.py`. Key settings:

| Setting | Default | Description |
|---|---|---|
| `YOLO_MODEL` | `yolov8n.pt` | YOLO model weight file |
| `YOLO_CONFIDENCE` | `0.4` | Detection confidence threshold |
| `DASHBOARD_PORT` | `8000` | Web dashboard port |
| `UNATTENDED_WARNING_SECONDS` | `180` | Seconds before WARNING escalation |
| `UNATTENDED_CRITICAL_SECONDS` | `300` | Seconds before CRITICAL escalation |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model for safety advice |

## Usage

```bash
python main.py
```

### CLI Flags

| Flag | Description |
|---|---|
| `--video PATH` | Use a video file instead of live camera |
| `--camera INDEX` | Camera device index (default: 0) |
| `--no-display` | Headless mode — no OpenCV window |
| `--no-tts` | Disable voice alerts |
| `--no-llm` | Disable LLM advisor |
| `--port PORT` | Override dashboard port |
| `--demo` | Demo mode — shorter escalation timers (10s/30s/60s) |

### Examples

```bash
# Run on a recorded video, no voice alerts
python main.py --video demo_kitchen.mp4 --no-tts

# Headless mode on camera 1, dashboard on port 9000
python main.py --camera 1 --no-display --port 9000

# Demo mode with fast escalation timers
python main.py --video demo_kitchen.mp4 --demo
```

### Dashboard

Open `http://localhost:8000` in a browser to see the live video feed, alert history, and burner zone status.

## Burner Zones

Burner zones define rectangular regions of interest on the camera frame. SafeFlame uses them to track per-burner state and detect zone-specific hazards (flame overlap, boil-over, proximity).

### Configure via Dashboard

Use the zone management UI in the web dashboard to draw and name burner zones interactively.

### Configure via API

```bash
curl -X POST http://localhost:8000/api/zones \
  -H "Content-Type: application/json" \
  -d '[
    {"name": "Front Left", "x": 100, "y": 200, "w": 200, "h": 200},
    {"name": "Front Right", "x": 400, "y": 200, "w": 200, "h": 200}
  ]'
```

Each zone is a dict with `name`, `x`, `y`, `w`, `h` (pixel coordinates).

## Testing

```bash
python -m pytest tests/ -v   # 82 tests, ~1s
```

## Project Structure

```
SafeFlame/
├── main.py              # Entry point, orchestrates all components
├── config.py            # All thresholds, timers, and settings
├── detector.py          # YOLOv8 inference wrapper
├── heuristics.py        # Rule-based CV: flame, smoke, boil-over, proximity
├── state_machine.py     # Per-burner state tracking and alert escalation
├── alerts.py            # Voice alert manager (TTS)
├── llm_advisor.py       # Ollama LLM integration for safety advice
├── dashboard/
│   ├── server.py        # FastAPI + WebSocket backend
│   └── static/
│       └── index.html   # Dashboard frontend
├── tests/
│   ├── conftest.py      # Shared fixtures and mocks
│   ├── test_main.py
│   ├── test_detector.py
│   ├── test_heuristics.py
│   ├── test_state_machine.py
│   ├── test_alerts.py
│   ├── test_llm_advisor.py
│   ├── test_dashboard.py
│   └── test_integration.py
├── requirements.txt
└── pytest.ini
```

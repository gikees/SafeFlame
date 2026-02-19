# SafeFlame

Real-time AI kitchen safety monitor. Camera feed → YOLOv8 + CV heuristics → state machine → voice alerts + LLM advice + web dashboard.

**Hackathon project** (12-hour NYU hackathon). All inference runs locally on a Dell Pro Max with NVIDIA GB10 — no cloud.

## Workflow

- **Develop locally** → push to GitHub → pull on remote GB10 → run there
- **Remote machine**: `dell@100.89.249.36` (GPU inference, no camera, Ollama with llama3.1:8b)
  - SSH via `sshpass -p <password> ssh dell@100.89.249.36` (password stored locally, not in repo)
  - Working directory on remote: `~/SafeFlame` with venv at `.venv`
- **Dashboard access**: `http://100.89.249.36:8000` from any machine on the network
- For demo without a live camera, use `--video PATH` with a pre-recorded clip

## Quick reference

- **Language**: Python 3.10+
- **Entry point**: `main.py` — orchestrates all components in a capture loop
- **Config**: `config.py` — all thresholds, timers, and settings (no .env files)
- **Tests**: `python -m pytest tests/ -v` (76 tests, ~2s)
- **Run**: `python main.py` (requires camera or `--video PATH`)
- **Dashboard**: FastAPI on port 8000 (`dashboard/server.py` + `dashboard/static/index.html`)

## Architecture

```
main.py (SafeFlame class)
├── detector.py        — YOLOv8 wrapper (ultralytics). detect() → list[dict], draw() → annotated frame
├── heuristics.py      — HeuristicDetector: flame (HSV), smoke (HSV + persistence), boilover (motion/edge), proximity
├── state_machine.py   — KitchenStateMachine: per-zone BurnerState enum, escalation timers, alert dedup via cooldowns
├── alerts.py          — AlertManager: pyttsx3 TTS in background threads, system fallback (say/espeak), log capped at 100
├── llm_advisor.py     — LLMAdvisor: Ollama client (llama3.1:8b), hazard prompts, timeout with FALLBACK_ADVICE, dict cache (max 100)
└── dashboard/
    ├── server.py      — FastAPI app, REST API (/api/status, /api/alerts, /api/zones, /api/config), WebSocket (/ws, /ws/alerts)
    └── static/        — frontend HTML
```

## Key patterns

- **No database** — all state is in-memory (zone configs, alert log, state machine)
- **Config as module globals** — `config.py` values are mutated at runtime (CLI args, dashboard API). Tests reset them via `conftest.py::_reset_config` autouse fixture
- **Imports use bare module names** — `import config`, `from detector import Detector` etc. Project root must be on `sys.path`
- **Dashboard runs in a daemon thread** with its own asyncio event loop. `broadcast_frame_sync` / `broadcast_alert_sync` bridge the main thread to the async loop
- **Alert deduplication** uses `(alert_type, zone)` cooldown keys in `state_machine._alert_cooldowns`

## Testing

```bash
python -m pytest tests/ -v
```

- Tests mock `ultralytics`, `pyttsx3`, `ollama`, and `uvicorn` via `conftest.py` (patched into `sys.modules` before any project import)
- Shared fixtures: `black_frame`, `flame_frame`, `smoke_frame`, `sample_burner_zones`, `person_detection`, `bottle_near_zone`
- Test files mirror source modules: `test_detector.py`, `test_heuristics.py`, `test_state_machine.py`, `test_alerts.py`, `test_llm_advisor.py`, `test_dashboard.py`, `test_integration.py`
- Dashboard tests use `fastapi.testclient.TestClient` — no running server needed
- `pytest.ini` sets `testpaths = tests` and `addopts = -v --tb=short`

## Git conventions

- Commit messages start with a **lowercase** letter
- Do **not** add "Co-Authored-By" lines to commits

## Deployment note

On systems with externally-managed Python (Debian/Ubuntu), use a venv:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## GPU / CUDA note

The Dell GB10 has NVIDIA compute capability 12.1. Current PyTorch CUDA builds (e.g. `torch+cu128`) only officially support up to 12.0. Installing CUDA-enabled PyTorch causes the dashboard video stream to break (frames stop flowing over WebSocket). **Keep PyTorch CPU-only** (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`) until PyTorch adds official support for compute capability 12.1.

"""FastAPI + WebSocket backend for SafeFlame dashboard."""

import asyncio
import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from state_machine import BurnerState

app = FastAPI(title="SafeFlame Dashboard")

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Shared State (set by main.py) ────────────────────────────────────────────
# These are populated by the main loop via set_shared_state()
_shared = {
    "frame": None,           # latest annotated frame (numpy array)
    "status": {},            # kitchen status dict
    "alerts": [],            # alert manager reference
    "state_machine": None,   # KitchenStateMachine reference
    "alert_manager": None,   # AlertManager reference
    "fps": 0.0,
    "inference_ms": 0.0,
}

# Connected WebSocket clients
_frame_clients: list[WebSocket] = []
_alert_clients: list[WebSocket] = []


def set_shared_state(key: str, value):
    """Called by main.py to share data with the dashboard."""
    _shared[key] = value


def get_shared_state(key: str):
    return _shared.get(key)


# ── REST Endpoints ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/api/status")
async def get_status():
    sm = _shared.get("state_machine")
    if sm is None:
        return {"person_present": False, "zones": {}}
    return {
        **sm.get_status(config.BURNER_ZONES),
        "zone_overrides": sm.zone_active_overrides,
        "fps": _shared.get("fps", 0),
        "inference_ms": _shared.get("inference_ms", 0),
    }


@app.get("/api/alerts")
async def get_alerts():
    am = _shared.get("alert_manager")
    if am is None:
        return []
    return am.get_recent_alerts()


@app.get("/api/zones")
async def get_zones():
    return config.BURNER_ZONES


class ZoneConfig(BaseModel):
    name: str
    x: int
    y: int
    w: int
    h: int


@app.post("/api/zones")
async def set_zones(zones: list[ZoneConfig]):
    config.BURNER_ZONES = [z.model_dump() for z in zones]
    return {"status": "ok", "zones": config.BURNER_ZONES}


@app.post("/api/zones/{zone_name}/toggle")
async def toggle_zone(zone_name: str):
    sm = _shared.get("state_machine")
    if sm is None:
        return {"status": "error", "message": "State machine not initialized"}
    current = sm.zone_active_overrides.get(zone_name, False)
    sm.zone_active_overrides[zone_name] = not current
    # Reset escalation timer when toggling off
    if not sm.zone_active_overrides[zone_name]:
        sm.zone_states[zone_name] = BurnerState.OFF
        sm.zone_unattended_since[zone_name] = None
    return {"status": "ok", "zone": zone_name, "active": sm.zone_active_overrides[zone_name]}


class ConfigUpdate(BaseModel):
    key: str
    value: float | int | str


@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    if hasattr(config, update.key):
        setattr(config, update.key, update.value)
        return {"status": "ok", "key": update.key, "value": update.value}
    return {"status": "error", "message": f"Unknown config key: {update.key}"}


# ── WebSocket: Video Frames ─────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_video(websocket: WebSocket):
    await websocket.accept()
    _frame_clients.append(websocket)
    try:
        while True:
            # Keep connection alive, send frames from the broadcast loop
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _frame_clients:
            _frame_clients.remove(websocket)


@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    await websocket.accept()
    _alert_clients.append(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _alert_clients:
            _alert_clients.remove(websocket)


# ── Broadcast Functions (called from background thread) ──────────────────────

async def _broadcast_frame(frame_b64: str):
    """Send a base64 JPEG frame to all connected frame clients."""
    disconnected = []
    for ws in _frame_clients:
        try:
            await ws.send_text(frame_b64)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in _frame_clients:
            _frame_clients.remove(ws)


async def _broadcast_alert(alert_dict: dict):
    """Send an alert JSON to all connected alert clients."""
    msg = json.dumps(alert_dict)
    disconnected = []
    for ws in _alert_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in _alert_clients:
            _alert_clients.remove(ws)


def broadcast_frame_sync(frame: np.ndarray, loop: asyncio.AbstractEventLoop):
    """Encode frame and schedule broadcast on the event loop."""
    _, buffer = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
    )
    frame_b64 = base64.b64encode(buffer).decode("utf-8")
    asyncio.run_coroutine_threadsafe(_broadcast_frame(frame_b64), loop)


def broadcast_alert_sync(alert_dict: dict, loop: asyncio.AbstractEventLoop):
    """Schedule alert broadcast on the event loop."""
    asyncio.run_coroutine_threadsafe(_broadcast_alert(alert_dict), loop)

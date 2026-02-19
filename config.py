"""SafeFlame configuration — all thresholds, timers, zones, and settings."""

# ── Input Source ──────────────────────────────────────────────────────────────
# Set VIDEO_PATH to a file path to use a pre-recorded video; set to None for live webcam.
VIDEO_PATH = None  # e.g., "demo_kitchen.mp4"
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ── YOLO Model ────────────────────────────────────────────────────────────────
# Swap to "yolov8s.pt", "yolov8m.pt", or a custom fine-tuned ".pt" path anytime.
YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.4
YOLO_DEVICE = None  # None = auto-detect (GPU if available), or "cpu", "cuda:0"

# Classes of interest from COCO (by name)
YOLO_CLASSES_OF_INTEREST = [
    "person", "oven", "microwave", "toaster", "knife",
    "cup", "bowl", "bottle", "fork", "spoon",
    "sink", "refrigerator", "cell phone",
    "backpack", "handbag", "tie", "umbrella", "suitcase",
    "wine glass", "scissors", "teddy bear",
    "laptop", "remote", "keyboard", "mouse",
    "potted plant", "vase", "hair drier",
]

# ── Flame Detection (HSV thresholds) ─────────────────────────────────────────
FLAME_H_MIN, FLAME_H_MAX = 0, 40
FLAME_S_MIN, FLAME_S_MAX = 100, 255
FLAME_V_MIN, FLAME_V_MAX = 200, 255
FLAME_MIN_CONTOUR_AREA = 500  # pixels²

# ── Smoke Detection ──────────────────────────────────────────────────────────
SMOKE_H_MIN, SMOKE_H_MAX = 0, 30
SMOKE_S_MIN, SMOKE_S_MAX = 0, 60
SMOKE_V_MIN, SMOKE_V_MAX = 150, 230
SMOKE_MIN_CONTOUR_AREA = 2000
SMOKE_DETECTION_ENABLED = True
SMOKE_PERSISTENCE_SECONDS = 3.0  # must persist this long to count as smoke (not steam)

# ── Boil-Over Detection ──────────────────────────────────────────────────────
BOILOVER_EDGE_THRESHOLD = 80
BOILOVER_MOTION_THRESHOLD = 30
BOILOVER_MIN_AREA = 1000
BOILOVER_DETECTION_ENABLED = True

# ── Proximity Detection ──────────────────────────────────────────────────────
PROXIMITY_DISTANCE_PX = 150  # pixel distance threshold
FLAMMABLE_OBJECTS = [
    "bottle", "cup", "cell phone", "book", "paper",
    "backpack", "handbag", "tie", "umbrella", "suitcase",
    "wine glass", "teddy bear", "laptop", "remote",
    "keyboard", "mouse", "hair drier", "scissors",
    "potted plant", "vase",
]

# ── Burner Zones ──────────────────────────────────────────────────────────────
# List of dicts: {"name": str, "x": int, "y": int, "w": int, "h": int}
# Can be configured at runtime via the dashboard API.
BURNER_ZONES = []
ASSUME_BURNERS_ACTIVE = False  # Skip flame check — treat all zones as active

# ── State Machine Timers (seconds) ───────────────────────────────────────────
UNATTENDED_INFO_SECONDS = 60
UNATTENDED_WARNING_SECONDS = 180
UNATTENDED_CRITICAL_SECONDS = 300

# ── Alert Cooldowns (seconds) ────────────────────────────────────────────────
ALERT_COOLDOWN_SECONDS = 30
CRITICAL_REPEAT_SECONDS = 15

# ── Voice Alert Settings ─────────────────────────────────────────────────────
TTS_RATE = 175  # words per minute
TTS_VOLUME_INFO = 0.7
TTS_VOLUME_WARNING = 0.9
TTS_VOLUME_CRITICAL = 1.0

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8000
JPEG_QUALITY = 70  # 0-100, lower = smaller frames over WebSocket

# ── LLM (Ollama) ─────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
LLM_TIMEOUT_SECONDS = 5.0

# Fallback advice when LLM is unavailable, times out, or --no-llm is set
FALLBACK_ADVICE = {
    "unattended": "Turn off the burner or return to the kitchen immediately.",
    "proximity": "Move flammable objects away from the active burner now.",
    "boilover": "Reduce heat immediately and slide the pot off the burner.",
    "smoke": "Check the source of smoke — turn off heat and ventilate the area.",
    "flame": "If the flame is uncontrolled, turn off heat and cover with a lid.",
    "grease_fire": "Never use water. Turn off heat and smother with a metal lid or baking soda.",
}

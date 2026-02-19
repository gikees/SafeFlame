"""SafeFlame — Real-time AI kitchen safety monitor.

Entry point that orchestrates: Camera → Detector + Heuristics → State Machine → Alerts + LLM → Dashboard
"""

import argparse
import asyncio
import signal
import sys
import threading
import time

import cv2
import numpy as np
import uvicorn

import config
from alerts import AlertManager
from dashboard.server import (
    app,
    broadcast_alert_sync,
    broadcast_frame_sync,
    set_shared_state,
)
from detector import Detector
from heuristics import HeuristicDetector
from llm_advisor import LLMAdvisor
from state_machine import KitchenStateMachine, Severity


def parse_args():
    parser = argparse.ArgumentParser(description="SafeFlame Kitchen Safety Monitor")
    parser.add_argument("--video", type=str, default=None, help="Path to video file (overrides config)")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (overrides config)")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV window (headless mode)")
    parser.add_argument("--no-tts", action="store_true", help="Disable voice alerts")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM advisor")
    parser.add_argument("--port", type=int, default=None, help="Dashboard port (overrides config)")
    parser.add_argument("--demo", action="store_true", help="Demo mode: shorter escalation timers")
    return parser.parse_args()


class SafeFlame:
    """Main application class orchestrating all SafeFlame components."""

    def __init__(self, args):
        self.args = args
        self.running = True

        # Apply CLI overrides
        if args.video:
            config.VIDEO_PATH = args.video
        if args.camera is not None:
            config.CAMERA_INDEX = args.camera
        if args.port:
            config.DASHBOARD_PORT = args.port
        if args.demo:
            config.UNATTENDED_INFO_SECONDS = 10
            config.UNATTENDED_WARNING_SECONDS = 30
            config.UNATTENDED_CRITICAL_SECONDS = 60
            config.ALERT_COOLDOWN_SECONDS = 10
            config.SMOKE_PERSISTENCE_SECONDS = 1.5
            config.ASSUME_BURNERS_ACTIVE = True
            config.SMOKE_DETECTION_ENABLED = False
            config.BOILOVER_DETECTION_ENABLED = False

        # Initialize components
        print("[SafeFlame] Initializing detector...")
        self.detector = Detector()

        print("[SafeFlame] Initializing heuristics engine...")
        self.heuristics = HeuristicDetector()

        print("[SafeFlame] Initializing state machine...")
        self.state_machine = KitchenStateMachine()

        print("[SafeFlame] Initializing alert manager...")
        self.alert_manager = AlertManager() if not args.no_tts else None

        print("[SafeFlame] Initializing LLM advisor...")
        self.llm_advisor = LLMAdvisor() if not args.no_llm else None

        # Share references with dashboard
        set_shared_state("state_machine", self.state_machine)
        set_shared_state("alert_manager", self.alert_manager)

        # Dashboard event loop (set when server starts)
        self.dashboard_loop: asyncio.AbstractEventLoop | None = None

        # Performance tracking
        self.fps = 0.0
        self.inference_ms = 0.0

    def open_capture(self) -> cv2.VideoCapture:
        """Open video capture from configured source."""
        if config.VIDEO_PATH:
            print(f"[SafeFlame] Opening video file: {config.VIDEO_PATH}")
            cap = cv2.VideoCapture(config.VIDEO_PATH)
        else:
            print(f"[SafeFlame] Opening camera index: {config.CAMERA_INDEX}")
            cap = cv2.VideoCapture(config.CAMERA_INDEX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        if not cap.isOpened():
            print("[SafeFlame] ERROR: Could not open video source!")
            sys.exit(1)

        return cap

    def start_dashboard(self):
        """Start the FastAPI dashboard in a background thread."""
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.dashboard_loop = loop

            server_config = uvicorn.Config(
                app,
                host=config.DASHBOARD_HOST,
                port=config.DASHBOARD_PORT,
                log_level="warning",
                loop="asyncio",
            )
            server = uvicorn.Server(server_config)
            loop.run_until_complete(server.serve())

        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        # Give the server a moment to start and set the loop
        time.sleep(1)
        print(f"[SafeFlame] Dashboard running at http://0.0.0.0:{config.DASHBOARD_PORT}")

    def run(self):
        """Main capture + processing loop."""
        # Start dashboard
        self.start_dashboard()

        # Open capture
        cap = self.open_capture()

        print("[SafeFlame] Starting main loop. Press 'q' to quit.")
        print(f"[SafeFlame] Burner zones configured: {len(config.BURNER_ZONES)}")
        if not config.BURNER_ZONES:
            print("[SafeFlame] No burner zones — configure via dashboard at /api/zones")

        frame_count = 0
        fps_start = time.time()

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    if config.VIDEO_PATH:
                        # Loop video
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    print("[SafeFlame] Lost camera feed!")
                    break

                t0 = time.time()

                # ── YOLO Detection ───────────────────────────────────
                detections = self.detector.detect(frame)

                # ── Heuristic Analysis ───────────────────────────────
                heuristic_results = self.heuristics.analyze(
                    frame, detections, config.BURNER_ZONES
                )

                t1 = time.time()
                self.inference_ms = (t1 - t0) * 1000

                # ── State Machine ────────────────────────────────────
                alerts = self.state_machine.update(
                    detections, heuristic_results, config.BURNER_ZONES
                )

                # ── Handle Alerts ────────────────────────────────────
                for alert in alerts:
                    advice = None
                    if (
                        self.llm_advisor
                        and alert.severity in (Severity.WARNING, Severity.CRITICAL)
                    ):
                        context = {"zone": alert.burner_zone}
                        # Add object info for proximity alerts
                        if alert.type == "proximity":
                            for pa in heuristic_results.get("proximity_alerts", []):
                                if pa["zone"] == alert.burner_zone:
                                    context["object"] = pa["object"]
                                    break
                        advice = self.llm_advisor.get_safety_advice(
                            alert.type, context
                        )

                    if self.alert_manager:
                        self.alert_manager.handle_alert(alert, advice)

                    # Broadcast alert to dashboard
                    if self.dashboard_loop:
                        alert_dict = alert.to_dict()
                        alert_dict["advice"] = advice
                        broadcast_alert_sync(alert_dict, self.dashboard_loop)

                # ── Annotate Frame ───────────────────────────────────
                annotated = self.detector.draw(frame, detections)
                annotated = self.heuristics.draw_overlays(
                    annotated, heuristic_results, config.BURNER_ZONES
                )

                # ── FPS Calculation ──────────────────────────────────
                frame_count += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    self.fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = time.time()

                # Update shared state for dashboard
                set_shared_state("fps", self.fps)
                set_shared_state("inference_ms", self.inference_ms)

                # Draw FPS on frame
                cv2.putText(
                    annotated,
                    f"FPS: {self.fps:.1f} | Inference: {self.inference_ms:.0f}ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # ── Broadcast to Dashboard ───────────────────────────
                if self.dashboard_loop:
                    broadcast_frame_sync(annotated, self.dashboard_loop)

                # ── Display (if not headless) ────────────────────────
                if not self.args.no_display:
                    cv2.imshow("SafeFlame", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                else:
                    # Small sleep to avoid busy loop in headless mode
                    time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n[SafeFlame] Interrupted by user.")
        finally:
            self.running = False
            cap.release()
            if not self.args.no_display:
                cv2.destroyAllWindows()
            print("[SafeFlame] Shutdown complete.")


def main():
    args = parse_args()

    # Handle signals
    def signal_handler(sig, frame):
        print("\n[SafeFlame] Shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    app_instance = SafeFlame(args)
    app_instance.run()


if __name__ == "__main__":
    main()

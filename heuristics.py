"""Rule-based computer vision detections for SafeFlame."""

import time
import cv2
import numpy as np

import config


class HeuristicDetector:
    """Performs rule-based CV detections: flame, smoke, boil-over, proximity."""

    def __init__(self):
        self.prev_gray = None
        # Track smoke persistence: zone_key -> first_seen timestamp
        self._smoke_first_seen: dict[str, float] = {}

    def analyze(
        self, frame: np.ndarray, detections: list[dict], burner_zones: list[dict]
    ) -> dict:
        """Run all heuristic analyses on a frame.

        Args:
            frame: BGR image.
            detections: YOLO detections list.
            burner_zones: list of {"name", "x", "y", "w", "h"}.

        Returns dict with keys:
            flames: list of contour bounding rects
            smoke_regions: list of confirmed smoke regions
            boilover_zones: list of zone names with boil-over detected
            proximity_alerts: list of {"object", "zone", "distance"}
            zone_flame_status: {zone_name: bool} — flame present per zone
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flames = self._detect_flames(hsv)
        smoke_regions = self._detect_smoke(hsv, frame) if config.SMOKE_DETECTION_ENABLED else []
        boilover_zones = self._detect_boilover(gray, burner_zones) if config.BOILOVER_DETECTION_ENABLED else []
        zone_flame_status = self._check_zones_for_flame(flames, burner_zones)
        proximity_alerts = self._check_proximity(detections, burner_zones)

        self.prev_gray = gray

        return {
            "flames": flames,
            "smoke_regions": smoke_regions,
            "boilover_zones": boilover_zones,
            "proximity_alerts": proximity_alerts,
            "zone_flame_status": zone_flame_status,
        }

    # ── Flame Detection ──────────────────────────────────────────────────────

    def _detect_flames(self, hsv: np.ndarray) -> list[tuple]:
        """Detect flame-colored regions via HSV thresholding."""
        lower = np.array([config.FLAME_H_MIN, config.FLAME_S_MIN, config.FLAME_V_MIN])
        upper = np.array([config.FLAME_H_MAX, config.FLAME_S_MAX, config.FLAME_V_MAX])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        flames = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= config.FLAME_MIN_CONTOUR_AREA:
                flames.append(cv2.boundingRect(cnt))
        return flames

    # ── Smoke Detection ──────────────────────────────────────────────────────

    def _detect_smoke(self, hsv: np.ndarray, frame: np.ndarray) -> list[tuple]:
        """Detect smoke via gray-ish HSV range + temporal persistence."""
        lower = np.array([config.SMOKE_H_MIN, config.SMOKE_S_MIN, config.SMOKE_V_MIN])
        upper = np.array([config.SMOKE_H_MAX, config.SMOKE_S_MAX, config.SMOKE_V_MAX])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        now = time.time()
        confirmed = []
        active_keys = set()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < config.SMOKE_MIN_CONTOUR_AREA:
                continue
            rect = cv2.boundingRect(cnt)
            key = f"smoke_{rect[0]//50}_{rect[1]//50}"
            active_keys.add(key)
            if key not in self._smoke_first_seen:
                self._smoke_first_seen[key] = now
            elif now - self._smoke_first_seen[key] >= config.SMOKE_PERSISTENCE_SECONDS:
                confirmed.append(rect)

        # Clean up stale entries
        stale = [k for k in self._smoke_first_seen if k not in active_keys]
        for k in stale:
            del self._smoke_first_seen[k]

        return confirmed

    # ── Boil-Over Detection ──────────────────────────────────────────────────

    def _detect_boilover(
        self, gray: np.ndarray, burner_zones: list[dict]
    ) -> list[str]:
        """Detect boil-over by looking for motion/edge spikes above burner zones."""
        if self.prev_gray is None:
            return []

        boilover_zones = []
        for zone in burner_zones:
            x, y, w, h = zone["x"], zone["y"], zone["w"], zone["h"]
            # Check region above the zone (potential boil-over area)
            above_y = max(0, y - h // 2)
            roi_curr = gray[above_y:y, x:x + w]
            roi_prev = self.prev_gray[above_y:y, x:x + w]
            if roi_curr.size == 0 or roi_prev.size == 0:
                continue

            # Motion detection
            diff = cv2.absdiff(roi_curr, roi_prev)
            motion_pixels = np.sum(diff > config.BOILOVER_MOTION_THRESHOLD)

            # Edge intensity
            edges = cv2.Canny(roi_curr, config.BOILOVER_EDGE_THRESHOLD, 200)
            edge_pixels = np.sum(edges > 0)

            if motion_pixels > config.BOILOVER_MIN_AREA or edge_pixels > config.BOILOVER_MIN_AREA:
                boilover_zones.append(zone["name"])

        return boilover_zones

    # ── Zone Flame Check ─────────────────────────────────────────────────────

    def _check_zones_for_flame(
        self, flames: list[tuple], burner_zones: list[dict]
    ) -> dict[str, bool]:
        """Check if any detected flame overlaps with a burner zone."""
        status = {}
        for zone in burner_zones:
            zx, zy, zw, zh = zone["x"], zone["y"], zone["w"], zone["h"]
            has_flame = False
            for fx, fy, fw, fh in flames:
                # Check bounding rect overlap
                if (fx < zx + zw and fx + fw > zx and fy < zy + zh and fy + fh > zy):
                    has_flame = True
                    break
            status[zone["name"]] = has_flame
        return status

    # ── Proximity Detection ──────────────────────────────────────────────────

    def _check_proximity(
        self, detections: list[dict], burner_zones: list[dict]
    ) -> list[dict]:
        """Check if flammable objects are too close to burner zones."""
        alerts = []
        flammable = [d for d in detections if d["class"] in config.FLAMMABLE_OBJECTS]
        for obj in flammable:
            ox1, oy1, ox2, oy2 = obj["bbox"]
            obj_cx, obj_cy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
            for zone in burner_zones:
                zx, zy, zw, zh = zone["x"], zone["y"], zone["w"], zone["h"]
                zone_cx, zone_cy = zx + zw // 2, zy + zh // 2
                dist = np.sqrt((obj_cx - zone_cx) ** 2 + (obj_cy - zone_cy) ** 2)
                if dist < config.PROXIMITY_DISTANCE_PX:
                    alerts.append({
                        "object": obj["class"],
                        "zone": zone["name"],
                        "distance": int(dist),
                    })
        return alerts

    def draw_overlays(
        self, frame: np.ndarray, heuristics: dict, burner_zones: list[dict]
    ) -> np.ndarray:
        """Draw heuristic overlays on a frame."""
        annotated = frame.copy()

        # Draw burner zones
        for zone in burner_zones:
            x, y, w, h = zone["x"], zone["y"], zone["w"], zone["h"]
            flame_on = heuristics["zone_flame_status"].get(zone["name"], False)
            color = (0, 0, 255) if flame_on else (0, 200, 0)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                annotated, zone["name"], (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )

        # Draw flame detections
        for fx, fy, fw, fh in heuristics["flames"]:
            cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)
            cv2.putText(
                annotated, "FLAME", (fx, fy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
            )

        # Draw smoke detections
        for sx, sy, sw, sh in heuristics["smoke_regions"]:
            cv2.rectangle(annotated, (sx, sy), (sx + sw, sy + sh), (128, 128, 128), 2)
            cv2.putText(
                annotated, "SMOKE", (sx, sy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA,
            )

        # Draw proximity warnings
        for pa in heuristics["proximity_alerts"]:
            cv2.putText(
                annotated,
                f"WARNING: {pa['object']} near {pa['zone']}",
                (10, annotated.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA,
            )

        return annotated

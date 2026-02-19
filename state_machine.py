"""Kitchen state tracking and alert decision engine for SafeFlame."""

import time
from dataclasses import dataclass, field
from enum import Enum

import config


class BurnerState(str, Enum):
    OFF = "off"
    ACTIVE_ATTENDED = "active_attended"
    ACTIVE_UNATTENDED = "active_unattended"
    ALERT_INFO = "alert_info"
    ALERT_WARNING = "alert_warning"
    ALERT_CRITICAL = "alert_critical"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    type: str  # e.g., "unattended", "proximity", "smoke", "boilover"
    severity: Severity
    message: str
    timestamp: float = field(default_factory=time.time)
    burner_zone: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "burner_zone": self.burner_zone,
        }


class KitchenStateMachine:
    """Tracks per-burner-zone state and produces alert events."""

    def __init__(self):
        # Per-zone state
        self.zone_states: dict[str, BurnerState] = {}
        self.zone_unattended_since: dict[str, float | None] = {}

        # Person tracking
        self.person_present: bool = False
        self.person_absent_since: float | None = None

        # Alert deduplication: (alert_type, zone) -> last_fired timestamp
        self._alert_cooldowns: dict[tuple[str, str], float] = {}

    def update(
        self,
        detections: list[dict],
        heuristics: dict,
        burner_zones: list[dict],
    ) -> list[Alert]:
        """Process one frame's worth of data and return any new alerts.

        Args:
            detections: YOLO detections.
            heuristics: Output from HeuristicDetector.analyze().
            burner_zones: Current burner zone configs.

        Returns:
            List of Alert objects to be handled.
        """
        now = time.time()
        alerts: list[Alert] = []

        # ── Person presence ──────────────────────────────────────────────
        person_detected = any(d["class"] == "person" for d in detections)
        if person_detected:
            self.person_present = True
            self.person_absent_since = None
        else:
            if self.person_present:
                # Person just left
                self.person_absent_since = now
            self.person_present = False

        # ── Per-zone state updates ───────────────────────────────────────
        zone_flame_status = heuristics.get("zone_flame_status", {})

        for zone in burner_zones:
            name = zone["name"]
            flame_on = zone_flame_status.get(name, False)

            # Initialize zone state if new
            if name not in self.zone_states:
                self.zone_states[name] = BurnerState.OFF
                self.zone_unattended_since[name] = None

            if not flame_on:
                # Burner is off
                self.zone_states[name] = BurnerState.OFF
                self.zone_unattended_since[name] = None
                continue

            # Burner is on
            if self.person_present:
                # Person is present — de-escalate
                self.zone_states[name] = BurnerState.ACTIVE_ATTENDED
                self.zone_unattended_since[name] = None
                continue

            # Burner on, person absent — escalation logic
            if self.zone_unattended_since[name] is None:
                self.zone_unattended_since[name] = self.person_absent_since or now

            absence_duration = now - self.zone_unattended_since[name]

            if absence_duration >= config.UNATTENDED_CRITICAL_SECONDS:
                self.zone_states[name] = BurnerState.ALERT_CRITICAL
                alert = self._maybe_alert(
                    "unattended", name, Severity.CRITICAL,
                    f"CRITICAL: {name} has been unattended for over {int(absence_duration)}s! Turn off the burner immediately.",
                    now, cooldown=config.CRITICAL_REPEAT_SECONDS,
                )
                if alert:
                    alerts.append(alert)

            elif absence_duration >= config.UNATTENDED_WARNING_SECONDS:
                self.zone_states[name] = BurnerState.ALERT_WARNING
                alert = self._maybe_alert(
                    "unattended", name, Severity.WARNING,
                    f"Warning: {name} unattended for {int(absence_duration)}s. Please check your cooking.",
                    now,
                )
                if alert:
                    alerts.append(alert)

            elif absence_duration >= config.UNATTENDED_INFO_SECONDS:
                self.zone_states[name] = BurnerState.ALERT_INFO
                alert = self._maybe_alert(
                    "unattended", name, Severity.INFO,
                    f"Notice: {name} is on and no one is in the kitchen.",
                    now,
                )
                if alert:
                    alerts.append(alert)

            else:
                self.zone_states[name] = BurnerState.ACTIVE_UNATTENDED

        # ── Proximity alerts ─────────────────────────────────────────────
        for pa in heuristics.get("proximity_alerts", []):
            alert = self._maybe_alert(
                "proximity", pa["zone"], Severity.WARNING,
                f"Warning: {pa['object']} detected near {pa['zone']}. Move it away from the heat source.",
                now,
            )
            if alert:
                alerts.append(alert)

        # ── Boil-over alerts ─────────────────────────────────────────────
        for zone_name in heuristics.get("boilover_zones", []):
            alert = self._maybe_alert(
                "boilover", zone_name, Severity.WARNING,
                f"Warning: Potential boil-over detected at {zone_name}! Reduce heat or remove the pot.",
                now,
            )
            if alert:
                alerts.append(alert)

        # ── Smoke alerts ─────────────────────────────────────────────────
        if heuristics.get("smoke_regions"):
            alert = self._maybe_alert(
                "smoke", "kitchen", Severity.CRITICAL,
                "CRITICAL: Smoke detected in the kitchen! Check for fire immediately.",
                now,
            )
            if alert:
                alerts.append(alert)

        # ── General flame with no zone ───────────────────────────────────
        if heuristics.get("flames") and not burner_zones:
            alert = self._maybe_alert(
                "flame", "kitchen", Severity.WARNING,
                "Warning: Open flame detected. Define burner zones for better monitoring.",
                now,
            )
            if alert:
                alerts.append(alert)

        return alerts

    def _maybe_alert(
        self,
        alert_type: str,
        zone: str,
        severity: Severity,
        message: str,
        now: float,
        cooldown: float | None = None,
    ) -> Alert | None:
        """Create an alert if not in cooldown period."""
        key = (alert_type, zone)
        cd = cooldown if cooldown is not None else config.ALERT_COOLDOWN_SECONDS
        last = self._alert_cooldowns.get(key, 0)
        if now - last < cd:
            return None
        self._alert_cooldowns[key] = now
        return Alert(
            type=alert_type,
            severity=severity,
            message=message,
            timestamp=now,
            burner_zone=zone,
        )

    def get_status(self, burner_zones: list[dict]) -> dict:
        """Return current kitchen status for the dashboard."""
        now = time.time()
        absence_time = None
        if self.person_absent_since is not None:
            absence_time = now - self.person_absent_since

        zones_status = {}
        for zone in burner_zones:
            name = zone["name"]
            state = self.zone_states.get(name, BurnerState.OFF)
            zones_status[name] = {
                "state": state.value,
                "unattended_seconds": (
                    now - self.zone_unattended_since[name]
                    if self.zone_unattended_since.get(name) is not None
                    else None
                ),
            }

        return {
            "person_present": self.person_present,
            "person_absent_seconds": absence_time,
            "zones": zones_status,
        }

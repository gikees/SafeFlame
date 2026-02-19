"""YOLOv8 inference wrapper for SafeFlame."""

import cv2
import numpy as np
from ultralytics import YOLO

import config


class Detector:
    """Loads a YOLO model and runs per-frame inference."""

    def __init__(self):
        self.model = YOLO(config.YOLO_MODEL)
        self.device = config.YOLO_DEVICE  # None = auto
        # Build a set of class names we care about for fast lookup
        self.classes_of_interest = set(config.YOLO_CLASSES_OF_INTEREST)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run inference on a frame and return structured detections.

        Returns list of dicts:
            {"class": str, "confidence": float, "bbox": (x1,y1,x2,y2), "label": str}
        """
        results = self.model(
            frame,
            conf=config.YOLO_CONFIDENCE,
            device=self.device,
            verbose=False,
        )
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                if cls_name not in self.classes_of_interest:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "label": f"{cls_name} {conf:.0%}",
                })
        return detections

    def draw(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw bounding boxes and labels on a copy of the frame."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = det["label"]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
            )
        return annotated

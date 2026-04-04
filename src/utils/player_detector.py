from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_name: str  # "player" | "goalkeeper" | "referee" | "ball"


class PlayerDetector(ABC):
    """Detects players (and optionally the ball) in a single frame."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Returns a list of detections found in the frame."""
        ...


class YOLOPlayerDetector(PlayerDetector):
    """Player detector backed by a YOLOv8 model fine-tuned on football data."""

    # Class IDs for a football-fine-tuned model: 0=player, 1=goalkeeper, 2=referee, 3=ball
    _CLASS_NAMES: dict[int, str] = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}

    def __init__(self, model_name: str = "yolov8x.pt", confidence: float = 0.3) -> None:
        from ultralytics import YOLO  # lazy import — model download on first use

        self._model = YOLO(model_name)
        self._confidence = confidence

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model(frame, verbose=False)[0]
        detections: list[Detection] = []
        for box in results.boxes:
            conf = float(box.conf)
            if conf < self._confidence:
                continue
            cls_id = int(box.cls)
            class_name = self._CLASS_NAMES.get(cls_id, "player")
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf, class_name=class_name))
        return detections


class FakePlayerDetector(PlayerDetector):
    """Deterministic detector for tests — cycles through a pre-supplied sequence."""

    def __init__(self, detections_sequence: list[list[Detection]]) -> None:
        self._seq = detections_sequence
        self._idx = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        dets = self._seq[self._idx % len(self._seq)]
        self._idx += 1
        return dets

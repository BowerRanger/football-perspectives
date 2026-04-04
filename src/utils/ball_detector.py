from abc import ABC, abstractmethod
import numpy as np


class BallDetector(ABC):
    """Detects the ball in a single frame and returns its pixel position."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> tuple[float, float] | None:
        """Returns (u, v) pixel coordinates of ball, or None if not found."""
        ...


class YOLOBallDetector(BallDetector):
    """Ball detector using a YOLOv8 model (sports ball class or custom model)."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.3) -> None:
        from ultralytics import YOLO  # lazy import — model download on first use
        self._model = YOLO(model_name)
        self._confidence = confidence
        # COCO class 32 = sports ball
        self._ball_class_id = 32

    def detect(self, frame: np.ndarray) -> tuple[float, float] | None:
        results = self._model(frame, verbose=False)[0]
        best_conf = 0.0
        best_pos = None
        for box in results.boxes:
            if int(box.cls) == self._ball_class_id and float(box.conf) > self._confidence:
                if float(box.conf) > best_conf:
                    best_conf = float(box.conf)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    best_pos = ((x1 + x2) / 2, (y1 + y2) / 2)
        return best_pos


class FakeBallDetector(BallDetector):
    """Deterministic detector for tests — returns pre-supplied positions in sequence."""

    def __init__(self, positions: list[tuple[float, float] | None]) -> None:
        self._positions = positions
        self._idx = 0

    def detect(self, frame: np.ndarray) -> tuple[float, float] | None:
        pos = self._positions[self._idx % len(self._positions)]
        self._idx += 1
        return pos

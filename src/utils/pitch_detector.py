from pathlib import Path
import json
import logging

import numpy as np

from src.stages.calibration import LandmarkDetection, PitchKeypointDetector
from src.utils.pitch import FIFA_LANDMARKS


class ManualJsonPitchDetector(PitchKeypointDetector):
    """Loads per-shot manual pitch landmarks from JSON files."""

    def __init__(self, annotations_dir: Path, min_confidence: float = 0.0) -> None:
        self.annotations_dir = annotations_dir
        self.min_confidence = float(min_confidence)
        self._cache: dict[str, dict[str, dict[str, float]]] = {}

    def _load_shot_annotations(self, shot_id: str) -> dict[str, dict[str, float]]:
        if shot_id in self._cache:
            return self._cache[shot_id]

        annotation_path = self.annotations_dir / f"{shot_id}.json"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing manual landmark file: {annotation_path}")

        data = json.loads(annotation_path.read_text())
        frames = data.get("frames")
        if not isinstance(frames, dict):
            raise ValueError(f"Invalid manual landmark schema in {annotation_path}")

        self._cache[shot_id] = frames
        return frames

    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int | None = None,
        shot_id: str | None = None,
    ) -> dict[str, LandmarkDetection]:
        if frame_idx is None or shot_id is None:
            return {}

        frames = self._load_shot_annotations(shot_id)
        frame_points = frames.get(str(frame_idx), {})
        if not isinstance(frame_points, dict):
            return {}

        detections: dict[str, LandmarkDetection] = {}
        for name, payload in frame_points.items():
            if name not in FIFA_LANDMARKS:
                logging.warning(
                    "Ignoring unknown manual landmark '%s' for shot %s frame %s",
                    name,
                    shot_id,
                    frame_idx,
                )
                continue
            if not isinstance(payload, dict):
                continue
            try:
                u = float(payload["u"])
                v = float(payload["v"])
                confidence = float(payload.get("confidence", 1.0))
            except (KeyError, TypeError, ValueError):
                continue
            if confidence < 0.0 or confidence > 1.0:
                logging.warning(
                    "Ignoring manual landmark '%s' with out-of-range confidence %.3f for shot %s frame %s",
                    name,
                    confidence,
                    shot_id,
                    frame_idx,
                )
                continue
            h, w = frame.shape[:2]
            if u < 0.0 or v < 0.0 or u >= float(w) or v >= float(h):
                logging.warning(
                    "Ignoring manual landmark '%s' with out-of-bounds coords (u=%.1f, v=%.1f) for shot %s frame %s",
                    name,
                    u,
                    v,
                    shot_id,
                    frame_idx,
                )
                continue
            if confidence < self.min_confidence:
                continue
            detections[name] = LandmarkDetection(
                uv=np.array([u, v], dtype=np.float32),
                confidence=confidence,
                source="manual_json",
            )
        return detections

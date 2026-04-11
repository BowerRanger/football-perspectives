from pathlib import Path
import json
import logging

import cv2
import numpy as np

from src.stages.calibration import LandmarkDetection, PitchKeypointDetector
from src.utils.pitch import FIFA_LANDMARKS


class ManualJsonPitchDetector(PitchKeypointDetector):
    """Loads per-shot manual pitch landmarks from JSON files."""

    def __init__(
        self,
        annotations_dir: Path,
        min_confidence: float = 0.0,
        max_cached_shots: int = 64,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence must be in [0.0, 1.0], got {min_confidence}"
            )
        self.annotations_dir = annotations_dir
        self.min_confidence = float(min_confidence)
        self.max_cached_shots = max(1, int(max_cached_shots))
        self._cache: dict[str, dict[str, dict[str, float]]] = {}

    def _load_shot_annotations(self, shot_id: str) -> dict[str, dict[str, float]]:
        if shot_id in self._cache:
            return self._cache[shot_id]

        annotation_path = self.annotations_dir / f"{shot_id}.json"
        if not annotation_path.exists():
            # Cache the miss so we don't re-check the filesystem every frame
            self._cache[shot_id] = {}
            return {}

        data = json.loads(annotation_path.read_text())
        frames = data.get("frames")
        if not isinstance(frames, dict):
            raise ValueError(f"Invalid manual landmark schema in {annotation_path}")

        if len(self._cache) >= self.max_cached_shots:
            # FIFO eviction: Python 3.7+ dict preserves insertion order.
            oldest_shot_id = next(iter(self._cache))
            del self._cache[oldest_shot_id]
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
            except (KeyError, TypeError, ValueError) as exc:
                logging.debug(
                    "Ignoring malformed manual payload for landmark '%s' in shot %s frame %s: %s",
                    name,
                    shot_id,
                    frame_idx,
                    exc,
                )
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


class HeuristicPitchDetector(PitchKeypointDetector):
    """Automatically detects core pitch landmarks from white-line geometry.

    Uses grass colour masking to isolate on-pitch white lines from broadcast
    overlays (scoreboards, logos, ad boards).  Focuses on centre-line landmarks
    and adds corner landmarks when strong vertical boundaries are visible.
    """

    def __init__(self, min_confidence: float = 0.3) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence must be in [0.0, 1.0], got {min_confidence}"
            )
        self.min_confidence = float(min_confidence)

    @staticmethod
    def _grass_mask(hsv: np.ndarray) -> np.ndarray:
        """Return a binary mask of green-grass pixels."""
        return cv2.inRange(hsv, (30, 25, 25), (85, 255, 255))

    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int | None = None,
        shot_id: str | None = None,
    ) -> dict[str, LandmarkDetection]:
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return {}

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- Isolate on-pitch white lines using grass proximity masking ---
        grass = self._grass_mask(hsv)
        grass_ratio = float(np.count_nonzero(grass)) / float(grass.size)
        if grass_ratio < 0.15:
            return {}  # Not enough grass visible — likely a close-up or non-pitch shot

        # Dilate grass to include white lines that sit on grass boundaries
        grass_dilated = cv2.dilate(grass, np.ones((21, 21), np.uint8))

        white_mask = cv2.inRange(hsv, (0, 0, 170), (180, 70, 255))
        # Keep only white pixels that are near grass (removes overlays, ad boards)
        pitch_white = cv2.bitwise_and(white_mask, grass_dilated)

        pitch_white_ratio = float(np.count_nonzero(pitch_white)) / float(pitch_white.size)
        if pitch_white_ratio < 0.002:
            return {}

        edges = cv2.Canny(pitch_white, 40, 120)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=40,
            minLineLength=max(30, int(0.08 * w)),
            maxLineGap=max(8, int(0.02 * w)),
        )
        if lines is None:
            return {}

        # Classify lines by angle (allow up to ~15° from pure H/V for perspective)
        horizontal_lines: list[tuple[float, float, float, float]] = []
        vertical_lines: list[tuple[float, float, float, float]] = []
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [float(v) for v in line]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            length = (dx ** 2 + dy ** 2) ** 0.5
            if length < 1:
                continue
            angle = np.degrees(np.arctan2(dy, dx))
            if angle <= 15.0:
                horizontal_lines.append((x1, y1, x2, y2))
            elif angle >= 75.0:
                vertical_lines.append((x1, y1, x2, y2))

        if len(horizontal_lines) < 2:
            return {}

        # For horizontal lines, use the midpoint y and x-span
        h_mids_y = [float((y1 + y2) / 2.0) for x1, y1, x2, y2 in horizontal_lines]
        h_mids_x = [float((x1 + x2) / 2.0) for x1, y1, x2, y2 in horizontal_lines]

        y_top = float(np.percentile(h_mids_y, 10))
        y_bottom = float(np.percentile(h_mids_y, 90))

        # Vertical lines give sideline x positions
        v_mids_x: list[float] = []
        for x1, y1, x2, y2 in vertical_lines:
            v_mids_x.append(float((x1 + x2) / 2.0))

        # x_center: prefer vertical line median if available, else horizontal line center
        if v_mids_x:
            x_center = float(np.median(v_mids_x))
        else:
            x_center = float(np.median(h_mids_x))

        y_center = float((y_top + y_bottom) / 2.0)

        spread = max(1.0, y_bottom - y_top)
        geometry_score = min(1.0, spread / (0.35 * h))
        base_conf = max(self.min_confidence, min(0.85, 0.45 + 0.4 * geometry_score))

        candidates: dict[str, LandmarkDetection] = {
            "halfway_near": LandmarkDetection(
                uv=np.array([x_center, y_top], dtype=np.float32),
                confidence=base_conf,
                source="heuristic",
            ),
            "halfway_far": LandmarkDetection(
                uv=np.array([x_center, y_bottom], dtype=np.float32),
                confidence=base_conf,
                source="heuristic",
            ),
            "center_spot": LandmarkDetection(
                uv=np.array([x_center, y_center], dtype=np.float32),
                confidence=max(self.min_confidence, min(0.8, base_conf - 0.05)),
                source="heuristic",
            ),
        }

        if len(v_mids_x) >= 2:
            x_left = float(np.percentile(v_mids_x, 10))
            x_right = float(np.percentile(v_mids_x, 90))
            edge_conf = max(self.min_confidence, min(0.75, base_conf - 0.1))
            candidates["corner_near_left"] = LandmarkDetection(
                uv=np.array([x_left, y_top], dtype=np.float32),
                confidence=edge_conf,
                source="heuristic",
            )
            candidates["corner_near_right"] = LandmarkDetection(
                uv=np.array([x_right, y_top], dtype=np.float32),
                confidence=edge_conf,
                source="heuristic",
            )
            candidates["corner_far_left"] = LandmarkDetection(
                uv=np.array([x_left, y_bottom], dtype=np.float32),
                confidence=edge_conf,
                source="heuristic",
            )
            candidates["corner_far_right"] = LandmarkDetection(
                uv=np.array([x_right, y_bottom], dtype=np.float32),
                confidence=edge_conf,
                source="heuristic",
            )

        return {
            name: det
            for name, det in candidates.items()
            if det.confidence >= self.min_confidence and name in FIFA_LANDMARKS
        }


class HybridPitchDetector(PitchKeypointDetector):
    """Combines automatic detector families; keeps the highest-confidence result per landmark."""

    def __init__(self, detectors: list[PitchKeypointDetector]) -> None:
        if not detectors:
            raise ValueError("HybridPitchDetector requires at least one detector")
        self.detectors = detectors

    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int | None = None,
        shot_id: str | None = None,
    ) -> dict[str, LandmarkDetection]:
        merged: dict[str, LandmarkDetection] = {}
        for detector in self.detectors:
            detections = detector.detect(frame, frame_idx=frame_idx, shot_id=shot_id)
            for name, detection in detections.items():
                current = merged.get(name)
                if current is None or detection.confidence > current.confidence:
                    merged[name] = detection
        return merged

import json
from pathlib import Path

import numpy as np
import pytest

from src.stages.calibration import LandmarkDetection
from src.utils.pitch_detector import ManualJsonPitchDetector


def _write_manual_landmarks(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_manual_json_detector_loads_frame_points(tmp_path):
    annotations_dir = tmp_path / "landmarks"
    _write_manual_landmarks(
        annotations_dir / "origi01.json",
        {
            "frames": {
                "0": {
                    "center_spot": {"u": 960.0, "v": 540.0, "confidence": 0.95},
                    "halfway_top": {"u": 960.0, "v": 50.0, "confidence": 0.8},
                }
            }
        },
    )

    detector = ManualJsonPitchDetector(annotations_dir=annotations_dir)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    detections = detector.detect(frame, frame_idx=0, shot_id="origi01")

    assert "center_spot" in detections
    center = detections["center_spot"]
    assert isinstance(center, LandmarkDetection)
    assert np.allclose(center.uv, np.array([960.0, 540.0], dtype=np.float32))
    assert center.confidence == pytest.approx(0.95)


def test_manual_json_detector_filters_invalid_names_and_low_confidence(tmp_path):
    annotations_dir = tmp_path / "landmarks"
    _write_manual_landmarks(
        annotations_dir / "origi01.json",
        {
            "frames": {
                "0": {
                    "center_spot": {"u": 960.0, "v": 540.0, "confidence": 0.2},
                    "not_a_landmark": {"u": 1.0, "v": 2.0, "confidence": 1.0},
                    "halfway_bottom": {"u": 960.0, "v": 1030.0, "confidence": 0.7},
                }
            }
        },
    )

    detector = ManualJsonPitchDetector(annotations_dir=annotations_dir, min_confidence=0.5)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    detections = detector.detect(frame, frame_idx=0, shot_id="origi01")

    assert "center_spot" not in detections
    assert "not_a_landmark" not in detections
    assert "halfway_bottom" in detections


def test_manual_json_detector_missing_shot_file_raises(tmp_path):
    detector = ManualJsonPitchDetector(annotations_dir=tmp_path / "landmarks")
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    with pytest.raises(FileNotFoundError):
        detector.detect(frame, frame_idx=0, shot_id="unknown")


def test_manual_json_detector_returns_empty_when_context_missing(tmp_path):
    detector = ManualJsonPitchDetector(annotations_dir=tmp_path / "landmarks")
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    assert detector.detect(frame, frame_idx=None, shot_id="origi01") == {}
    assert detector.detect(frame, frame_idx=0, shot_id=None) == {}
    assert detector.detect(frame) == {}


def test_manual_json_detector_accepts_confidence_boundaries(tmp_path):
    annotations_dir = tmp_path / "landmarks"
    _write_manual_landmarks(
        annotations_dir / "origi01.json",
        {
            "frames": {
                "0": {
                    "center_spot": {"u": 960.0, "v": 540.0, "confidence": 0.0},
                    "halfway_top": {"u": 960.0, "v": 50.0, "confidence": 1.0},
                }
            }
        },
    )

    detector = ManualJsonPitchDetector(annotations_dir=annotations_dir, min_confidence=0.0)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    detections = detector.detect(frame, frame_idx=0, shot_id="origi01")
    assert detections["center_spot"].confidence == pytest.approx(0.0)
    assert detections["halfway_top"].confidence == pytest.approx(1.0)


def test_manual_json_detector_invalid_payloads_are_ignored(tmp_path):
    annotations_dir = tmp_path / "landmarks"
    _write_manual_landmarks(
        annotations_dir / "origi01.json",
        {
            "frames": {
                "0": {
                    "center_spot": ["bad", "payload"],
                    "halfway_top": {"u": "oops", "v": 20.0, "confidence": 0.9},
                    "halfway_bottom": {"u": 960.0, "v": 1080.0, "confidence": 0.9},
                    "left_penalty_spot": {"u": 300.0, "v": 400.0, "confidence": 0.9},
                }
            }
        },
    )

    detector = ManualJsonPitchDetector(annotations_dir=annotations_dir)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    detections = detector.detect(frame, frame_idx=0, shot_id="origi01")
    assert list(detections.keys()) == ["left_penalty_spot"]
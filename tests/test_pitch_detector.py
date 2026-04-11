import json
from pathlib import Path

import numpy as np
import pytest

from src.stages.calibration import LandmarkDetection
from src.utils.pitch_detector import HeuristicPitchDetector, HybridPitchDetector, ManualJsonPitchDetector


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
                    "halfway_far": {"u": 960.0, "v": 1030.0, "confidence": 0.7},
                }
            }
        },
    )

    detector = ManualJsonPitchDetector(annotations_dir=annotations_dir, min_confidence=0.5)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    detections = detector.detect(frame, frame_idx=0, shot_id="origi01")

    assert "center_spot" not in detections
    assert "not_a_landmark" not in detections
    assert "halfway_far" in detections


def test_manual_json_detector_missing_shot_file_returns_empty(tmp_path):
    detector = ManualJsonPitchDetector(annotations_dir=tmp_path / "landmarks")
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    result = detector.detect(frame, frame_idx=0, shot_id="unknown")
    assert result == {}


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
                    "halfway_near": {"u": 960.0, "v": 50.0, "confidence": 1.0},
                }
            }
        },
    )

    detector = ManualJsonPitchDetector(annotations_dir=annotations_dir, min_confidence=0.0)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    detections = detector.detect(frame, frame_idx=0, shot_id="origi01")
    assert detections["center_spot"].confidence == pytest.approx(0.0)
    assert detections["halfway_near"].confidence == pytest.approx(1.0)


def test_manual_json_detector_invalid_payloads_are_ignored(tmp_path):
    annotations_dir = tmp_path / "landmarks"
    _write_manual_landmarks(
        annotations_dir / "origi01.json",
        {
            "frames": {
                "0": {
                    "center_spot": ["bad", "payload"],
                    "halfway_near": {"u": "oops", "v": 20.0, "confidence": 0.9},
                    "halfway_far": {"u": 960.0, "v": 1080.0, "confidence": 0.9},
                    "left_penalty_spot": {"u": 300.0, "v": 400.0, "confidence": 0.9},
                }
            }
        },
    )

    detector = ManualJsonPitchDetector(annotations_dir=annotations_dir)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    detections = detector.detect(frame, frame_idx=0, shot_id="origi01")
    assert list(detections.keys()) == ["left_penalty_spot"]


def _make_synthetic_pitch_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (40, 120, 40)
    cv2 = pytest.importorskip("cv2")
    cv2.line(frame, (100, 120), (width - 100, 120), (255, 255, 255), 5)
    cv2.line(frame, (100, height - 120), (width - 100, height - 120), (255, 255, 255), 5)
    cv2.line(frame, (width // 2, 90), (width // 2, height - 90), (255, 255, 255), 5)
    return frame


def test_heuristic_detector_detects_core_landmarks():
    detector = HeuristicPitchDetector(min_confidence=0.3)
    frame = _make_synthetic_pitch_frame()

    detections = detector.detect(frame)

    assert "halfway_near" in detections
    assert "halfway_far" in detections
    assert "center_spot" in detections
    assert detections["center_spot"].source == "heuristic"


def test_hybrid_detector_merges_from_automatic_detector():
    frame = _make_synthetic_pitch_frame()
    hybrid = HybridPitchDetector(detectors=[HeuristicPitchDetector(min_confidence=0.3)])

    detections = hybrid.detect(frame)

    assert "center_spot" in detections


def test_hybrid_detector_requires_non_empty_detector_list():
    with pytest.raises(ValueError):
        HybridPitchDetector(detectors=[])


def test_manual_detector_respects_max_cached_shots(tmp_path):
    annotations_dir = tmp_path / "landmarks"
    for shot_id in ["a", "b", "c"]:
        _write_manual_landmarks(
            annotations_dir / f"{shot_id}.json",
            {
                "frames": {
                    "0": {
                        "center_spot": {"u": 10.0, "v": 10.0, "confidence": 0.9}
                    }
                }
            },
        )

    detector = ManualJsonPitchDetector(annotations_dir=annotations_dir, max_cached_shots=2)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detector.detect(frame, frame_idx=0, shot_id="a")
    detector.detect(frame, frame_idx=0, shot_id="b")
    detector.detect(frame, frame_idx=0, shot_id="c")

    assert len(detector._cache) == 2


def test_detector_rejects_invalid_min_confidence(tmp_path):
    with pytest.raises(ValueError):
        ManualJsonPitchDetector(annotations_dir=tmp_path / "landmarks", min_confidence=-0.1)

    with pytest.raises(ValueError):
        HeuristicPitchDetector(min_confidence=1.1)
"""Unit tests for the shared anchor-click scoring helper.

``score_track`` is the single scoring path used by both
``scripts/eval_anchor_clicks.py`` and the camera regression gate
(tests/test_camera_regression.py), so the two can never drift apart.
All fixtures here are synthetic pinhole cameras — no media needed.
"""

import numpy as np
import pytest

from src.utils.anchor_click_eval import score_track
from src.utils.camera_projection import project_world_to_image
from src.utils.virtual_cameras import look_at_view

_K = [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]]
_WORLD_PTS = [
    [0.0, 68.0, 0.0],
    [16.5, 54.16, 0.0],
    [52.5, 34.0, 0.0],
]


def _camera_frame(frame: int) -> dict:
    R, t = look_at_view(np.array([52.5, -20.0, 20.0]),
                        np.array([52.5, 34.0, 0.0]))
    return {"frame": frame, "K": _K,
            "R": [list(r) for r in R], "t": list(t),
            "confidence": 0.8, "is_anchor": True}


def _project(frame_cam: dict, world_xyz: list) -> list:
    p = project_world_to_image(
        np.array(frame_cam["K"]), np.array(frame_cam["R"]),
        np.array(frame_cam["t"]), (0.0, 0.0),
        np.array([world_xyz], dtype=float))[0]
    return [float(p[0]), float(p[1])]


def _anchor(frame: int, cam: dict, offsets=None) -> dict:
    offsets = offsets or [(0.0, 0.0)] * len(_WORLD_PTS)
    landmarks = []
    for w, (dx, dy) in zip(_WORLD_PTS, offsets):
        xy = _project(cam, w)
        landmarks.append({"name": "kp", "world_xyz": list(w),
                          "image_xy": [xy[0] + dx, xy[1] + dy]})
    return {"frame": frame, "landmarks": landmarks}


def _track(frames: list) -> dict:
    return {"clip_id": "synth", "fps": 25.0, "image_size": [1280, 720],
            "distortion": [0.0, 0.0], "frames": frames}


@pytest.mark.unit
def test_exact_clicks_score_zero_residuals():
    cam = _camera_frame(0)
    metrics = score_track({"anchors": [_anchor(0, cam)]}, _track([cam]))
    assert metrics["clicks"] == 3
    assert metrics["med_px"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["p90_px"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["max_px"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_known_pixel_offset_is_measured():
    cam = _camera_frame(0)
    anchors = {"anchors": [_anchor(0, cam, offsets=[(3.0, 4.0)] * 3)]}
    metrics = score_track(anchors, _track([cam]))
    assert metrics["med_px"] == pytest.approx(5.0, abs=1e-6)
    assert metrics["max_px"] == pytest.approx(5.0, abs=1e-6)
    assert metrics["per_anchor"] == [
        {"frame": 0, "clicks": 3,
         "med_px": pytest.approx(5.0, abs=1e-6),
         "max_px": pytest.approx(5.0, abs=1e-6)}]


@pytest.mark.unit
def test_anchor_frame_missing_from_track_counts_as_uncovered():
    cam = _camera_frame(0)
    anchors = {"anchors": [_anchor(0, cam), _anchor(99, cam)]}
    metrics = score_track(anchors, _track([cam]))
    assert metrics["anchor_frames_total"] == 2
    assert metrics["anchor_frames_covered"] == 1
    assert metrics["clicks"] == 3  # uncovered anchor's clicks not scored


@pytest.mark.unit
def test_track_coverage_and_confidence_summarised():
    cam0, cam1 = _camera_frame(0), _camera_frame(1)
    cam1["confidence"] = 0.4
    metrics = score_track({"anchors": [_anchor(0, cam0)]},
                          _track([cam0, cam1]))
    assert metrics["track_frames"] == 2
    assert metrics["mean_confidence"] == pytest.approx(0.6)


@pytest.mark.unit
def test_no_scorable_clicks_yields_none_percentiles():
    cam = _camera_frame(0)
    metrics = score_track({"anchors": [_anchor(99, cam)]}, _track([cam]))
    assert metrics["clicks"] == 0
    assert metrics["med_px"] is None
    assert metrics["p90_px"] is None
    assert metrics["max_px"] is None

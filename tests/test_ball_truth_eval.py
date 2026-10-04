"""Ball truth scorer."""

from __future__ import annotations

import numpy as np

from src.utils import ball_truth_eval as ev
from src.utils import ball_truth_solver as S
from tests.test_ball_truth_solver import CAM_A


def truth_line():
    frames = list(range(0, 11))
    xyz = [[100.0 + i, 34.0 + 0.1 * i, 1.0] for i in range(11)]  # crosses x=105 at i=5
    return {"dense": {"frames": frames, "xyz": xyz}, "outcome": "goal",
            "events": [{"frame": 5, "kind": "touch"}, {"frame": 9, "kind": "bounce"}]}


def test_error_3d_exact_and_offset():
    t = truth_line()
    pipe = ev.pipeline_by_ref(t["dense"]["frames"], t["dense"]["xyz"])
    r = ev.error_3d(t["dense"]["frames"], np.array(t["dense"]["xyz"]), pipe)
    assert r["p50"] == 0 and r["coverage"] == 1 and r["pct_within_0.2m"] == 1
    shifted = {f: p + [0.3, 0, 0] for f, p in pipe.items()}
    r = ev.error_3d(t["dense"]["frames"], np.array(t["dense"]["xyz"]), shifted)
    assert abs(r["p50"] - 0.3) < 1e-9 and r["pct_within_0.2m"] == 0 and r["pct_within_0.5m"] == 1


def test_uncovered_frames_count_as_misses():
    t = truth_line()
    pipe = {f: np.array(p) for f, p in zip(t["dense"]["frames"][:5], t["dense"]["xyz"][:5])}
    r = ev.error_3d(t["dense"]["frames"], np.array(t["dense"]["xyz"]), pipe)
    assert r["n"] == 5 and abs(r["coverage"] - 5 / 11) < 1e-3
    assert abs(r["pct_within_0.2m"] - 5 / 11) < 1e-3
    assert r["pct_within_0.2m_of_covered"] == 1


def test_reprojection_error_px():
    pts = np.array([[40.0, 20.0, 1.0]])
    pipe = {0: np.array([40.5, 20.0, 1.0])}
    r = ev.reprojection_error([0], pts, pipe, {"a": lambda sf: CAM_A}, {"a": 0})
    assert r["a"]["n"] == 1 and r["a"]["p50"] > 1
    assert isinstance(CAM_A, S.Cam)


def test_event_timing():
    r = ev.event_timing(
        [{"frame": 5, "kind": "touch"}, {"frame": 40, "kind": "bounce"}],
        [{"frame": 7, "state": "player_touch"}, {"frame": 6, "state": "bounce"}])
    assert r["n_matched"] == 1 and r["events"][0]["error_frames"] == 2
    assert not r["events"][1]["matched"]
    assert r["mean_abs_error_frames"] == 2


def test_line_cross_truth_vs_pipeline():
    t = truth_line()
    xyz = np.array(t["dense"]["xyz"])
    c = ev.line_cross(t["dense"]["frames"], xyz)
    assert c["line_x"] == 105.0 and abs(c["frame"] - 5.0) < 1e-6
    pipe = ev.pipeline_by_ref(t["dense"]["frames"], (xyz + [0.0, 0.4, 0.3]).tolist())
    res = ev.evaluate(t, pipe)
    assert abs(res["line_cross"]["point_error_m"] - 0.5) < 1e-6
    assert res["line_cross"]["frame_error"] == 0


def test_line_cross_missed_by_pipeline():
    t = truth_line()
    pipe = ev.pipeline_by_ref(t["dense"]["frames"][:4], t["dense"]["xyz"][:4])
    res = ev.evaluate(t, pipe)
    assert res["line_cross"]["matched"] is False

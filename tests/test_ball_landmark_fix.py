"""Landmark-coincidence resolution: point snap, line snap, suggestions."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_landmark_fix import (
    project_onto_segment_2d,
    resolve_landmark_world,
    suggest_pitch_fixes,
)
from src.utils.pitch_landmarks import LANDMARK_CATALOGUE
from src.utils.pitch_lines_catalogue import LINE_CATALOGUE

BALL_R = 0.11


def _camera_pose():
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _project(p, K, R, t):
    cam = R @ np.asarray(p, dtype=float) + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def test_point_landmark_snaps_exactly_ignoring_camera():
    name = "left_goal_left_post_base"
    lm = LANDMARK_CATALOGUE[name]
    world = resolve_landmark_world(
        (999.0, 999.0), name, K=None, R=None, t=None,
        distortion=(0.0, 0.0), ball_radius=BALL_R,
    )
    assert world is not None
    assert world[0] == pytest.approx(lm.world_xyz[0])
    assert world[1] == pytest.approx(lm.world_xyz[1])
    assert world[2] == pytest.approx(BALL_R)


def test_line_landmark_snaps_ground_point_onto_line():
    K, R, t = _camera_pose()
    line_name = sorted(LINE_CATALOGUE)[0]
    (ax, ay, _az), (bx, by, _bz) = LINE_CATALOGUE[line_name]
    # A true point slightly OFF the line at ball height; its click pixel
    # must snap back onto the line.
    mid = np.array([(ax + bx) / 2.0, (ay + by) / 2.0, BALL_R])
    off = mid + np.array([0.3, 0.3, 0.0])
    uv = _project(off, K, R, t)
    world = resolve_landmark_world(
        uv, f"line:{line_name}", K=K, R=R, t=t,
        distortion=(0.0, 0.0), ball_radius=BALL_R,
    )
    assert world is not None
    snapped = project_onto_segment_2d(
        (world[0], world[1]), (ax, ay), (bx, by))
    assert world[0] == pytest.approx(snapped[0], abs=1e-6)
    assert world[1] == pytest.approx(snapped[1], abs=1e-6)
    assert world[2] == pytest.approx(BALL_R)
    # And it landed near the true off-line point (within the 0.3m offset + eps).
    assert np.hypot(world[0] - off[0], world[1] - off[1]) < 0.5


def test_line_landmark_without_camera_returns_none():
    line_name = sorted(LINE_CATALOGUE)[0]
    assert resolve_landmark_world(
        (10.0, 10.0), f"line:{line_name}", K=None, R=None, t=None,
        distortion=(0.0, 0.0), ball_radius=BALL_R,
    ) is None


def test_unknown_name_returns_none():
    assert resolve_landmark_world(
        (10.0, 10.0), "no_such_feature", K=None, R=None, t=None,
        distortion=(0.0, 0.0), ball_radius=BALL_R,
    ) is None


def test_project_onto_segment_clamps_to_endpoints():
    assert project_onto_segment_2d((-5.0, 0.0), (0.0, 0.0), (10.0, 0.0)) == (0.0, 0.0)
    assert project_onto_segment_2d((15.0, 3.0), (0.0, 0.0), (10.0, 0.0)) == (10.0, 0.0)
    assert project_onto_segment_2d((4.0, 3.0), (0.0, 0.0), (10.0, 0.0)) == (4.0, 0.0)


def test_suggest_ranks_nearest_first_and_prefixes_lines():
    name = "left_goal_left_post_base"
    lm = LANDMARK_CATALOGUE[name]
    near = (lm.world_xyz[0] + 0.2, lm.world_xyz[1] + 0.1)
    out = suggest_pitch_fixes(near, max_distance_m=2.0, limit=5)
    assert out, "expected at least the nearby post base"
    assert out[0]["name"] == name
    assert out[0]["kind"] == "landmark"
    assert out[0]["distance_m"] == pytest.approx(np.hypot(0.2, 0.1), abs=1e-6)
    for item in out:
        assert (item["kind"] == "line") == item["name"].startswith("line:")
        assert item["distance_m"] <= 2.0
    assert [i["distance_m"] for i in out] == sorted(i["distance_m"] for i in out)


def test_suggest_ignores_elevated_landmarks():
    # Crossbar endpoints live at z=2.44 — never a grounded-ball fix.
    out = suggest_pitch_fixes((0.0, 30.34), max_distance_m=1.0, limit=10)
    assert all("crossbar" not in i["name"] for i in out if i["kind"] == "landmark")


def test_suggest_empty_when_nothing_in_range():
    assert suggest_pitch_fixes((52.5, 20.0), max_distance_m=0.05, limit=5) == []

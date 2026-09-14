"""Tests for Task A of generic moving-camera support (2026-09-09):
degenerate-solo hardening in ``_solve_one_anchor_full``.

Diagnosis this implements a fix for (gberch-2): anchor frame 162 has one
mislabeled landmark click. Its solo solve is degenerate (fx~50, camera
centre ~250m off the pitch) on every fx-multiplier attempt, but the
historical code returned that degenerate candidate anyway, poisoning
the first joint-pass median seed (see the 1e9-residual relock explosion
in docs/superpowers/specs/2026-09-09-moving-camera-support.md). See that
doc for the full write-up; gate/moving-path/triage tests live in
tests/test_camera_mode_gate.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.schemas.anchor import Anchor, LandmarkObservation
import src.utils.anchor_solver as anchor_solver
from src.utils.anchor_solver import solve_anchors_jointly

IMAGE_SIZE: tuple[int, int] = (1920, 1080)
CX_TRUE = IMAGE_SIZE[0] / 2.0
CY_TRUE = IMAGE_SIZE[1] / 2.0


def _K(fx: float) -> np.ndarray:
    return np.array([[fx, 0.0, CX_TRUE], [0.0, fx, CY_TRUE], [0.0, 0.0, 1.0]])


def _yaw(angle_deg: float) -> np.ndarray:
    look = np.array([0.0, 64.0, -30.0])
    look = look / np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    base = np.array([right, down, look], dtype=float)
    a = np.deg2rad(angle_deg)
    Ry = np.array(
        [[np.cos(a), -np.sin(a), 0.0],
         [np.sin(a), np.cos(a), 0.0],
         [0.0, 0.0, 1.0]],
    )
    return base @ Ry.T


def _project(K: np.ndarray, R: np.ndarray, t: np.ndarray, world: np.ndarray) -> tuple[float, float]:
    cam = R @ world + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def _make_landmark(K, R, t, name: str, world: tuple[float, float, float]) -> LandmarkObservation:
    return LandmarkObservation(
        name=name,
        image_xy=_project(K, R, t, np.asarray(world, dtype=float)),
        world_xyz=world,
    )


_LANDMARK_WORLD: list[tuple[str, tuple[float, float, float]]] = [
    ("near_left_corner", (0.0, 0.0, 0.0)),
    ("near_right_corner", (105.0, 0.0, 0.0)),
    ("far_left_corner", (0.0, 68.0, 0.0)),
    ("far_right_corner", (105.0, 68.0, 0.0)),
    ("halfway_near", (52.5, 0.0, 0.0)),
    ("near_left_corner_flag_top", (0.0, 0.0, 1.5)),
    ("left_goal_crossbar_left", (0.0, 30.34, 2.44)),
    ("left_goal_crossbar_right", (0.0, 37.66, 2.44)),
]


def _rich_anchor(K: np.ndarray, R: np.ndarray, t: np.ndarray, frame: int) -> Anchor:
    return Anchor(
        frame=frame,
        landmarks=tuple(
            _make_landmark(K, R, t, name, xyz) for name, xyz in _LANDMARK_WORLD
        ),
    )


def _camera_centred_anchor(C: np.ndarray, R: np.ndarray, fx: float, frame: int) -> Anchor:
    t = -R @ C
    return _rich_anchor(_K(fx), R, t, frame)


@pytest.mark.unit
def test_solve_one_anchor_full_returns_none_when_every_attempt_degenerate(monkeypatch):
    """Historical bug: _solve_one_anchor_full kept ``candidate_best = primary``
    even when *every* fx-multiplier alternative was ALSO degenerate, so it
    returned a degenerate (K, R, t, fx) tuple instead of signalling failure.
    Forcing every degeneracy check to fail must now yield None."""
    anchor = _rich_anchor(_K(1500.0), _yaw(0.0), np.array([0.0, 0.0, 50.0]), frame=0)
    monkeypatch.setattr(anchor_solver, "_is_degenerate_solo", lambda t, fx: True)
    result = anchor_solver._solve_one_anchor_full(
        anchor, CX_TRUE, CY_TRUE, fx_init=1500.0, K_init=_K(1500.0),
    )
    assert result is None


@pytest.mark.unit
def test_solve_one_anchor_full_still_returns_result_when_non_degenerate():
    """Sanity counterpart: a well-conditioned anchor must still solve
    normally (the hardening must not make every solve return None)."""
    R = _yaw(0.0)
    C = np.array([52.5, -30.0, 30.0])
    anchor = _camera_centred_anchor(C, R, 1500.0, frame=0)
    result = anchor_solver._solve_one_anchor_full(
        anchor, CX_TRUE, CY_TRUE, fx_init=1500.0, K_init=_K(1500.0),
    )
    assert result is not None
    K, R_hat, t_hat, fx = result
    C_hat = -R_hat.T @ t_hat
    assert np.linalg.norm(C_hat - C) < 0.5


@pytest.mark.unit
def test_hybrid_solve_excludes_degenerate_anchor_from_median_seed(caplog):
    """One-level-up integration of the Task A fix: the joint hybrid pass
    (Pass 1's rich-anchor loop) must exclude a degenerate solo solve from
    the t_world median seed rather than let it poison every other anchor
    — this is the exact f162 poisoning pattern from the gberch-2 bug."""
    import logging

    good = [
        _camera_centred_anchor(np.array([52.5, -30.0, 30.0]), _yaw(a), 1500.0, f)
        for a, f in [(-8.0, 0), (0.0, 50), (8.0, 100)]
    ]
    bad_lms = list(
        _camera_centred_anchor(
            np.array([52.5, -30.0, 30.0]), _yaw(4.0), 1500.0, frame=75,
        ).landmarks
    )
    bad_lms[0] = LandmarkObservation(
        name=bad_lms[0].name, image_xy=(-50000.0, 90000.0),
        world_xyz=bad_lms[0].world_xyz,
    )
    poisoned = Anchor(frame=75, landmarks=tuple(bad_lms))

    with caplog.at_level(logging.WARNING):
        sol = solve_anchors_jointly((*good, poisoned), image_size=IMAGE_SIZE)

    # The median seed (t_world) must stay near the 3 good anchors'
    # shared t, not be dragged by the poisoned anchor.
    t_good = -_yaw(0.0) @ np.array([52.5, -30.0, 30.0])
    assert np.linalg.norm(np.asarray(sol.t_world) - t_good) < 5.0

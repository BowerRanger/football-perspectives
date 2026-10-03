"""``mouth`` goal element: pixel ray ∩ goal-line plane inside posts/crossbar."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.goal_geometry import (
    GoalGeometry,
    goal_element_candidates,
    resolve_goal_impact_world,
)

_K = np.array([[1500.0, 0.0, 640.0], [0.0, 1500.0, 360.0], [0.0, 0.0, 1.0]])
# camera at (-10, 34, 1.22) looking down +x at the near goal
_R = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
_C = np.array([-10.0, 34.0, 1.22])
_T = -_R @ _C
_G = GoalGeometry.from_pitch_config({})


def _px(world):
    cam = _R @ np.asarray(world, float) + _T
    return (float(cam[0] * 1500.0 / cam[2] + 640.0), float(cam[1] * 1500.0 / cam[2] + 360.0))


def _resolve(world, element="mouth"):
    return resolve_goal_impact_world(
        _px(world), element, K=_K, R=_R, t=_T, distortion=(0.0, 0.0), geometry=_G)


def test_mouth_resolves_to_goal_line_plane():
    got = _resolve((0.0, 35.5, 1.3))
    assert got[0] == pytest.approx(0.0, abs=1e-6)
    assert got[1] == pytest.approx(35.5, abs=1e-3)
    assert got[2] == pytest.approx(1.3, abs=1e-3)


def test_mouth_rejects_ray_outside_posts():
    with pytest.raises(ValueError):
        _resolve((0.0, 39.0, 1.0))


def test_mouth_rejects_ray_above_crossbar():
    with pytest.raises(ValueError):
        _resolve((0.0, 34.0, 3.0))


def test_mouth_is_not_an_auto_classification_candidate():
    """Auto goal-impact classification must never propose ``mouth`` (any
    shot into the goal would match it)."""
    cands = goal_element_candidates(
        _px((0.0, 35.0, 1.0)), K=_K, R=_R, t=_T, distortion=(0.0, 0.0), geometry=_G)
    assert "mouth" not in {c[0] for c in cands}

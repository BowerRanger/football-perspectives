"""Sanity test: project a known pitch point through the broadcast camera,
then recover it via ``ankle_ray_to_pitch`` at ``plane_z = ball_radius``.

Reuses the foot-anchor helper for the ball; the only difference is the
plane height (ball radius rather than zero)."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.foot_anchor import ankle_ray_to_pitch  # reused for ball


@pytest.mark.unit
def test_ground_projection_returns_ball_radius_z():
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    # Use a physically valid camera (above pitch, looking down).
    # Camera at (52.5, -30, 30) looking at pitch centre (52.5, 34, 0)
    # — same pose used in tests/fixtures/synthetic_clip.py, with
    # exact orthonormal R derived from the look direction.
    look_world = np.array([0.0, 64.0, -30.0])
    look_world = look_world / np.linalg.norm(look_world)
    right_world = np.array([1.0, 0.0, 0.0])
    down_world = np.cross(look_world, right_world)
    R = np.array([right_world, down_world, look_world], dtype=float)
    cam_C = np.array([52.5, -30.0, 30.0])
    t = -R @ cam_C
    pitch_pt = np.array([60.0, 30.0, 0.11])
    cam = R @ pitch_pt + t
    pix = K @ cam
    uv = pix[:2] / pix[2]
    recovered = ankle_ray_to_pitch(uv, K=K, R=R, t=t, plane_z=0.11)
    assert np.allclose(recovered, pitch_pt, atol=1e-3)

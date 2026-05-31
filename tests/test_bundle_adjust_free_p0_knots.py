"""C2 physics: gravity + 2 hard knots determine the arc (depth included)."""
from __future__ import annotations

import numpy as np

from src.utils.bundle_adjust import fit_parabola_to_image_observations


def _cam():
    K = np.array([[1800.0, 0, 960.0], [0, 1800.0, 540.0], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.0])
    return K, R, t


def _project(p, K, R, t):
    cam = R @ p + t
    return (float((K @ cam)[0] / cam[2]), float((K @ cam)[1] / cam[2]))


def test_free_p0_two_knots_recovers_depth():
    K, R, t = _cam()
    # Camera 40 m back so the arc is in front of the camera.
    t = np.array([0.0, 0.0, 40.0])
    p0 = np.array([-8.0, 2.0, 0.11])
    v0 = np.array([10.0, 1.0, 9.0])
    fps = 30.0
    g = np.array([0.0, 0.0, -9.81])
    n = 16
    obs, Ks, Rs, ts = [], [], [], []
    for i in range(n):
        dt = i / fps
        p = p0 + v0 * dt + 0.5 * g * dt ** 2
        obs.append((i, _project(p, K, R, t)))
        Ks.append(K)
        Rs.append(R)
        ts.append(t)
    p_end = p0 + v0 * ((n - 1) / fps) + 0.5 * g * ((n - 1) / fps) ** 2
    # Free p0 (p0_fixed=None) + start & end as hard knots.
    knots = {0: p0, n - 1: p_end}
    rp0, rv0, resid = fit_parabola_to_image_observations(
        obs, Ks=Ks, Rs=Rs, t_world=ts, fps=fps,
        p0_fixed=None, knot_frames=knots,
    )
    assert np.linalg.norm(rp0 - p0) < 0.10        # depth recovered
    assert np.linalg.norm(rv0 - v0) < 0.5
    assert resid < 2.0

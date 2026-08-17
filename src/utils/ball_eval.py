"""Metric primitives for grading the ball track against ground truth.

Part of the sub-20cm accuracy campaign (see
docs/superpowers/specs/2026-08-17-ball-sub20cm-accuracy-design.md §4).
Pure numpy — no torch, no video — so everything here runs in the light venv
and inside unit tests with synthetic cameras.

Ground-truth model:
- A clicked/known pixel defines a camera ray the true ball centre must lie on
  (lateral error is measurable for every anchored or detected frame).
- Ground-level states pin full 3-D via ray ∩ the z = ball_radius plane.
- ``player_touch`` pins depth via the contacting joint projected onto the
  clicked ray (lateral from the click, depth from the body — the same
  identifiability the resolver itself relies on).
- Airborne states are ray-only (their depth is what the physics must supply).
"""

from __future__ import annotations

import numpy as np

from src.schemas.ball_anchor import BallAnchor
from src.utils.ball_anchor_heights import GROUND_LEVEL_STATES
from src.utils.camera_projection import undistort_pixel


def pixel_ray(uv, K, R, t, distortion=(0.0, 0.0)):
    """Camera centre and unit world-space ray direction through pixel ``uv``."""
    uv = np.asarray(uv, dtype=float)
    if tuple(distortion) != (0.0, 0.0):
        uv = undistort_pixel(uv, K, distortion)
    R = np.asarray(R, dtype=float)
    C = -R.T @ np.asarray(t, dtype=float)
    d = R.T @ (np.linalg.inv(np.asarray(K, dtype=float))
               @ np.array([uv[0], uv[1], 1.0]))
    return C, d / np.linalg.norm(d)


def point_ray_distance(P, C, d_hat):
    """(perpendicular distance, along-ray depth) of point ``P`` from a ray."""
    v = np.asarray(P, dtype=float) - C
    along = float(np.dot(v, d_hat))
    return float(np.linalg.norm(v - along * d_hat)), along


def ray_plane_z(C, d_hat, z):
    """Intersect the ray with the horizontal plane ``Z=z`` (forward only)."""
    dz = float(d_hat[2])
    if abs(dz) < 1e-9:
        return None
    s = (float(z) - float(C[2])) / dz
    if s <= 0:
        return None
    return np.asarray(C, dtype=float) + s * np.asarray(d_hat, dtype=float)


# States whose clicked pixel pins full 3-D via the ground plane. ``bounce``
# is at ground level at the bounce instant even though it brackets flight.
_GROUND_EXACT_STATES = frozenset(GROUND_LEVEL_STATES) | {"bounce"}


def anchor_gt_world(anchor: BallAnchor, K, R, t, distortion, *,
                    ball_radius: float, joint_world=None):
    """Best-available ground-truth world position for an anchor.

    Returns ``(xyz | None, kind)`` with kind one of ``"ground_exact"``,
    ``"joint_depth"``, ``"ray_only"``, ``"none"``.
    """
    if anchor.image_xy is None:
        return None, "none"
    C, d = pixel_ray(anchor.image_xy, K, R, t, distortion)
    if anchor.state in _GROUND_EXACT_STATES:
        X = ray_plane_z(C, d, ball_radius)
        return (X, "ground_exact") if X is not None else (None, "ray_only")
    if anchor.state == "player_touch" and joint_world is not None:
        _, along = point_ray_distance(np.asarray(joint_world, float), C, d)
        if along > 0:
            return C + along * d, "joint_depth"
    return None, "ray_only"


__all__ = [
    "pixel_ray",
    "point_ray_distance",
    "ray_plane_z",
    "anchor_gt_world",
]

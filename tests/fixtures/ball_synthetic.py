"""Shared synthetic-scene helpers for ball auto-event / solver tests.

Provides a static broadcast-style camera plus utilities to turn analytic
world trajectories into per-frame pixel observations and TrackerStep
sequences — the inputs the ball stage's event detection and piecewise
solver consume.
"""
from __future__ import annotations

import numpy as np

from src.utils.ball_tracker import TrackerStep
from src.utils.camera_projection import project_world_to_image

FPS = 25.0


def broadcast_camera(
    cam_centre: tuple[float, float, float] = (52.5, -20.0, 15.0),
    target: tuple[float, float, float] = (52.5, 34.0, 0.0),
    focal: float = 1200.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OpenCV-convention camera (z into scene, y down) looking at target."""
    K = np.array([[focal, 0.0, 960.0], [0.0, focal, 540.0], [0.0, 0.0, 1.0]])
    C = np.asarray(cam_centre, dtype=float)
    fwd = np.asarray(target, dtype=float) - C
    fwd = fwd / np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    t = -R @ C
    return K, R, t


def per_frame_cams(n: int, **kwargs):
    K, R, t = broadcast_camera(**kwargs)
    return (
        {i: K for i in range(n)},
        {i: R for i in range(n)},
        {i: t for i in range(n)},
    )


def project_track(
    worlds: dict[int, np.ndarray],
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> dict[int, tuple[float, float]]:
    """Project per-frame world positions to pixels."""
    out: dict[int, tuple[float, float]] = {}
    for fi, w in worlds.items():
        uv = project_world_to_image(
            K, R, t, (0.0, 0.0), np.asarray(w, dtype=float).reshape(1, 3)
        )[0]
        out[fi] = (float(uv[0]), float(uv[1]))
    return out


def steps_from_pixels(
    pixels: dict[int, tuple[float, float] | None],
    n_frames: int,
    p_flight: dict[int, float] | float = 0.0,
) -> list[TrackerStep]:
    """Build TrackerStep list from per-frame pixels (None = missing)."""
    steps = []
    for fi in range(n_frames):
        pf = p_flight.get(fi, 0.0) if isinstance(p_flight, dict) else p_flight
        steps.append(TrackerStep(
            frame=fi, uv=pixels.get(fi), p_flight=float(pf),
            is_outlier=False, is_gap_fill=False,
        ))
    return steps


def rolling_worlds(
    start_xy: tuple[float, float],
    vel_xy_per_frame: tuple[float, float],
    n: int,
    start_frame: int = 0,
    z: float = 0.11,
) -> dict[int, np.ndarray]:
    """Constant-velocity ground roll."""
    return {
        start_frame + i: np.array([
            start_xy[0] + vel_xy_per_frame[0] * i,
            start_xy[1] + vel_xy_per_frame[1] * i,
            z,
        ])
        for i in range(n)
    }


def ballistic_worlds(
    p0: tuple[float, float, float],
    v0: tuple[float, float, float],
    n: int,
    start_frame: int = 0,
    fps: float = FPS,
    restitution: float | None = None,
    floor_z: float = 0.11,
) -> dict[int, np.ndarray]:
    """Simulated gravity flight; optional ground bounce with restitution."""
    g = -9.81
    dt = 1.0 / fps
    p = np.asarray(p0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()
    out = {start_frame: p.copy()}
    for i in range(1, n):
        v[2] += g * dt
        p = p + v * dt
        if p[2] < floor_z and restitution is not None and v[2] < 0:
            p[2] = floor_z + (floor_z - p[2]) * restitution
            v[2] = -v[2] * restitution
        out[start_frame + i] = p.copy()
    return out


class FakePlayerContext:
    """Duck-typed PlayerContext substitute for event-detection tests."""

    def __init__(self, samples_by_frame: dict[int, list] | None = None) -> None:
        self._by_frame = samples_by_frame or {}

    def joints_at(self, frame: int):
        return tuple(self._by_frame.get(int(frame), ()))

    def joints_near_pixel(self, frame, uv, radius_px):
        u, v = float(uv[0]), float(uv[1])
        hits = []
        for s in self.joints_at(frame):
            if s.uv is None:
                continue
            d = ((s.uv[0] - u) ** 2 + (s.uv[1] - v) ** 2) ** 0.5
            if d <= radius_px:
                hits.append((d, s))
        hits.sort(key=lambda pair: pair[0])
        return [s for _, s in hits]

    def joint_world(self, frame, player_id, bone):
        for s in self.joints_at(frame):
            if s.player_id == player_id and s.bone == bone:
                return np.asarray(s.world_xyz, dtype=float)
        return None

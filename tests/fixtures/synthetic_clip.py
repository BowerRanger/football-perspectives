"""Synthetic broadcast clip generator for tests.

Renders a sparse set of pitch landmarks + stadium hoardings into a
sequence of frames using a known camera trajectory (yaw pan + slow
zoom). Returns the trajectory alongside the frames so tests can assert
recovery accuracy.

Camera pose is the broadcast pose validated in decisions log D6: camera
at world ``(52.5, -30, 30)`` looking towards pitch centre ``(52.5, 34, 0)``.
Yaw is applied as a world-z rotation post-multiplied into the base
world->camera matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SyntheticClip:
    frames: list[np.ndarray]      # BGR uint8
    Ks: list[np.ndarray]          # per-frame 3x3
    Rs: list[np.ndarray]          # per-frame 3x3 world->camera
    t_world: np.ndarray           # 3,
    image_size: tuple[int, int]   # (w, h)
    fps: float


def _build_pitch_world_points() -> np.ndarray:
    """Sparse 3D points across the pitch and on stadium structure.

    Many points cluster on pitch lines (high contrast) so that a feature
    detector can lock onto them after rendering. A few non-coplanar
    points (corner flags, crossbars) make K identifiable.
    """
    pts = []
    # Pitch corners + halfway + box corners
    for x in (0.0, 16.5, 52.5, 88.5, 105.0):
        for y in (0.0, 13.84, 24.84, 34.0, 43.16, 54.16, 68.0):
            pts.append([x, y, 0.0])
    # Crossbar endpoints
    pts += [
        [0.0, 30.34, 2.44], [0.0, 37.66, 2.44],
        [105.0, 30.34, 2.44], [105.0, 37.66, 2.44],
    ]
    # Corner flag tops
    pts += [
        [0.0, 0.0, 1.5], [105.0, 0.0, 1.5],
        [0.0, 68.0, 1.5], [105.0, 68.0, 1.5],
    ]
    # Stadium "advertising hoardings" — points behind the touchline at small z
    for x in np.linspace(0, 105, 12):
        pts.append([x, -2.0, 1.0])
        pts.append([x, 70.0, 1.0])
    return np.array(pts, dtype=float)


def render_synthetic_clip(
    n_frames: int = 60,
    pan_total_deg: float = 15.0,
    zoom_factor: float = 1.10,
    fps: float = 30.0,
    image_size: tuple[int, int] = (1280, 720),
) -> SyntheticClip:
    """Render a synthetic broadcast clip with a known camera trajectory.

    Camera at world ``(52.5, -30, 30)`` looking at pitch centre
    ``(52.5, 34, 0)``. Per-frame yaw is applied as a world-z rotation
    post-multiplied into ``R_base`` so the optical axis sweeps across
    the pitch.
    """
    w, h = image_size
    fx0 = 700.0  # wide enough that pitch is in frame at the broadcast pose
    cx = w / 2
    cy = h / 2
    # Camera at world C=(52.5, -30, 30) looking at pitch centre (52.5, 34, 0).
    # Build R_base from exact normalised look-direction so it is orthonormal
    # to floating-point precision (rounded values like 0.424/0.905 produce
    # det(R_base)≈0.998 which then leaks ~2° error through the RQ decomposition
    # in solve_first_anchor — see decisions log D7).
    look_world = np.array([0.0, 64.0, -30.0])
    look_world = look_world / np.linalg.norm(look_world)  # camera +z in world
    right_world = np.array([1.0, 0.0, 0.0])               # camera +x in world
    down_world = np.cross(look_world, right_world)        # camera +y in world
    R_base = np.array([right_world, down_world, look_world], dtype=float)
    cam_C = np.array([52.5, -30.0, 30.0])
    t_world = -R_base @ cam_C                             # = (-52.5, ~14.43, ~39.87)
    pts_world = _build_pitch_world_points()

    Ks: list[np.ndarray] = []
    Rs: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    visible_counts: list[int] = []

    for i in range(n_frames):
        s = i / (n_frames - 1) if n_frames > 1 else 0.0
        yaw = np.deg2rad(pan_total_deg * s)
        zoom = 1.0 + (zoom_factor - 1.0) * s
        K = np.array([[fx0 * zoom, 0, cx], [0, fx0 * zoom, cy], [0, 0, 1.0]])
        R_yaw_world = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0],
             [np.sin(yaw),  np.cos(yaw), 0],
             [0,            0,           1]],
            dtype=float,
        )
        R = R_base @ R_yaw_world.T  # apply yaw in world frame (post-multiply transpose)

        # Project all points to image
        cam = pts_world @ R.T + t_world
        in_front = cam[:, 2] > 0.1
        pix = cam[in_front] @ K.T
        uv = pix[:, :2] / pix[:, 2:3]

        img = np.full((h, w, 3), (60, 110, 60), dtype=np.uint8)  # green pitch tone
        n_visible = 0
        for u, v in uv:
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(img, (int(u), int(v)), 3, (255, 255, 255), -1)
                n_visible += 1

        Ks.append(K)
        Rs.append(R)
        frames.append(img)
        visible_counts.append(n_visible)

    # Sanity check: the fixture is internally consistent only if every
    # rendered frame contains enough visible landmarks for downstream
    # feature matching to have a chance.
    assert visible_counts[0] >= 6, (
        f"first frame must have >=6 visible landmarks, got {visible_counts[0]}"
    )
    assert min(visible_counts) >= 6, (
        f"every frame must have >=6 visible landmarks, "
        f"got min={min(visible_counts)}"
    )

    return SyntheticClip(
        frames=frames,
        Ks=Ks,
        Rs=Rs,
        t_world=t_world,
        image_size=image_size,
        fps=fps,
    )

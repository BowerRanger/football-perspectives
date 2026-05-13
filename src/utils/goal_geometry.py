"""Goal-frame geometry and pixel-ray resolution for ``goal_impact`` ball
anchors.

When the user marks a frame where the ball struck part of the goal, the
ball stage cannot use the generic ``ankle_ray_to_pitch`` ray-to-ground
projection — the contact point is at the goal frame, not on the pitch
plane. This module pins the trajectory to the known 3D location of the
struck element by intersecting the camera pixel ray with the
element-specific geometric primitive (a vertical line for posts, a
horizontal line for the crossbar, a vertical plane for the nets).

The pitch coordinate system follows CLAUDE.md: x along the nearside
touchline (0..length_m), y across the pitch toward the far touchline
(0..width_m), z up. Goals sit at the two end-lines (x=0 and x=length_m)
centred on y = width_m / 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.utils.camera_projection import undistort_pixel


# Numerical tolerance for rejecting near-parallel ray/surface configs and
# for accepting positive-direction ray parameters. Anything smaller than
# this gets treated as "no intersection" so we fall back rather than
# returning a wildly extrapolated point.
_RAY_T_EPS = 1e-6
_PARALLEL_EPS = 1e-9


@dataclass(frozen=True)
class GoalGeometry:
    """Pitch-frame goal geometry.

    Both goals share the same posts (``post_y_left``, ``post_y_right``)
    and crossbar height; only the goal-line x changes between the near
    goal (x=0) and the far goal (x=length_m). ``net_depth`` is the
    distance the back / side nets extend beyond the goal line.
    """
    goal_line_x_near: float
    goal_line_x_far: float
    post_y_left: float
    post_y_right: float
    crossbar_z: float
    net_depth: float

    @classmethod
    def from_pitch_config(cls, pitch_cfg: Mapping[str, float]) -> "GoalGeometry":
        length = float(pitch_cfg.get("length_m", 105.0))
        width = float(pitch_cfg.get("width_m", 68.0))
        goal_height = float(pitch_cfg.get("goal_height_m", 2.44))
        goal_width = float(pitch_cfg.get("goal_width_m", 7.32))
        net_depth = float(pitch_cfg.get("goal_depth_m", 1.5))
        center_y = width / 2.0
        half = goal_width / 2.0
        return cls(
            goal_line_x_near=0.0,
            goal_line_x_far=length,
            post_y_left=center_y - half,
            post_y_right=center_y + half,
            crossbar_z=goal_height,
            net_depth=net_depth,
        )


def resolve_goal_impact_world(
    image_xy: tuple[float, float],
    element: str,
    *,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float] = (0.0, 0.0),
    geometry: GoalGeometry,
) -> np.ndarray:
    """Resolve a goal-impact anchor to a 3D world position.

    Builds a camera ray from the clicked pixel and intersects it with
    the geometric primitive for ``element`` at each of the two goals.
    The intersection with the smallest positive ray parameter (i.e.
    closest to the camera along the ray) is returned. The two goals'
    candidates compete on equal footing — whichever the ray hits first
    wins.

    Raises:
        ValueError: if no valid intersection exists. Callers fall back
            to a generic ray-to-plane projection at ``crossbar_z``.
    """
    C, d = _camera_ray(image_xy, K=K, R=R, t=t, distortion=distortion)
    candidates: list[tuple[float, float, np.ndarray]] = []
    for goal_x in (geometry.goal_line_x_near, geometry.goal_line_x_far):
        sign = -1.0 if goal_x == geometry.goal_line_x_near else 1.0
        candidates.extend(_candidates_for_element(C, d, element, goal_x, sign, geometry))
    if not candidates:
        raise ValueError(
            f"goal_impact ray did not hit any '{element}' element"
        )
    # Each candidate is (residual_distance_m, s, world). Posts and the
    # crossbar use a closest-approach projection — the ray rarely passes
    # exactly through the 1-D line, so we keep a small residual and
    # prefer the line the ray comes nearest. Nets are 2-D planes with
    # zero residual. Ties on residual break by smallest forward ray-t
    # (closer to camera wins).
    candidates.sort(key=lambda h: (h[0], h[1]))
    return candidates[0][2]


def _camera_ray(
    image_xy: tuple[float, float],
    *,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build (camera_centre, ray_direction_world) for a pixel.

    Mirrors ``foot_anchor.ankle_ray_to_pitch`` up to (but not including)
    the ground-plane intersection so the same un-distort + back-project
    semantics apply.
    """
    uv = np.asarray(image_xy, dtype=float)
    if distortion != (0.0, 0.0):
        uv = undistort_pixel(uv, K, distortion)
    C = -R.T @ t
    pixel_h = np.array([uv[0], uv[1], 1.0])
    d_cam = np.linalg.inv(K) @ pixel_h
    d_world = R.T @ d_cam
    return np.asarray(C, dtype=float), np.asarray(d_world, dtype=float)


def _candidates_for_element(
    C: np.ndarray,
    d: np.ndarray,
    element: str,
    goal_x: float,
    sign: float,
    g: GoalGeometry,
) -> list[tuple[float, float, np.ndarray]]:
    """Return candidate intersections as ``(residual_m, s, world)``.

    ``residual_m`` is the ray-to-primitive distance (0 for planes; for
    lines it's the closest-approach distance, used to disambiguate
    between multiple post / crossbar candidates that compete for the
    same ray).
    """
    if element == "post":
        out = []
        for post_y in (g.post_y_left, g.post_y_right):
            hit = _project_ray_onto_vertical_line(
                C, d, x0=goal_x, y0=post_y, z_range=(0.0, g.crossbar_z),
            )
            if hit is not None:
                out.append(hit)
        return out
    if element == "crossbar":
        hit = _project_ray_onto_horizontal_line(
            C, d, x0=goal_x, z0=g.crossbar_z,
            y_range=(g.post_y_left, g.post_y_right),
        )
        return [hit] if hit is not None else []
    if element == "back_net":
        plane_x = goal_x + sign * g.net_depth
        hit = _ray_plane_x(
            C, d, plane_x,
            y_range=(g.post_y_left, g.post_y_right),
            z_range=(0.0, g.crossbar_z),
        )
        return [hit] if hit is not None else []
    if element == "side_net":
        x_min, x_max = sorted((goal_x, goal_x + sign * g.net_depth))
        out = []
        for post_y in (g.post_y_left, g.post_y_right):
            hit = _ray_plane_y(
                C, d, post_y,
                x_range=(x_min, x_max),
                z_range=(0.0, g.crossbar_z),
            )
            if hit is not None:
                out.append(hit)
        return out
    raise ValueError(f"unknown goal_element: {element!r}")


def _project_ray_onto_vertical_line(
    C: np.ndarray,
    d: np.ndarray,
    *,
    x0: float,
    y0: float,
    z_range: tuple[float, float],
) -> tuple[float, float, np.ndarray] | None:
    """Project the ray onto a vertical line at (x0, y0).

    The line is parametrised by z; the ray's closest approach is found
    in the xy-plane by least-squares — that fixes the ray parameter s,
    and z follows from the ray equation. ``residual_m`` is the xy
    distance from the ray (at s) to the line, used to disambiguate
    when two posts compete for the same pixel. We clamp z to
    ``z_range`` and reject hits that fall outside the segment.
    """
    d_xy = np.array([d[0], d[1]])
    denom = float(d_xy @ d_xy)
    if denom < _PARALLEL_EPS:
        return None
    a = np.array([x0 - C[0], y0 - C[1]])
    s = float(a @ d_xy) / denom
    if s < _RAY_T_EPS:
        return None
    z = float(C[2] + s * d[2])
    if not (z_range[0] <= z <= z_range[1]):
        return None
    closest_xy = np.array([C[0] + s * d[0], C[1] + s * d[1]])
    residual = float(np.linalg.norm(closest_xy - np.array([x0, y0])))
    return residual, s, np.array([x0, y0, z], dtype=float)


def _project_ray_onto_horizontal_line(
    C: np.ndarray,
    d: np.ndarray,
    *,
    x0: float,
    z0: float,
    y_range: tuple[float, float],
) -> tuple[float, float, np.ndarray] | None:
    """Project the ray onto a horizontal line at (x0, *, z0).

    The line is parametrised by y; the ray's closest approach is found
    in the xz-plane by least-squares, then y follows from s. Returns
    the closest-approach residual so callers can prefer the goal whose
    crossbar the ray comes nearest to.
    """
    d_xz = np.array([d[0], d[2]])
    denom = float(d_xz @ d_xz)
    if denom < _PARALLEL_EPS:
        return None
    a = np.array([x0 - C[0], z0 - C[2]])
    s = float(a @ d_xz) / denom
    if s < _RAY_T_EPS:
        return None
    y = float(C[1] + s * d[1])
    if not (y_range[0] <= y <= y_range[1]):
        return None
    closest_xz = np.array([C[0] + s * d[0], C[2] + s * d[2]])
    residual = float(np.linalg.norm(closest_xz - np.array([x0, z0])))
    return residual, s, np.array([x0, y, z0], dtype=float)


def _ray_plane_x(
    C: np.ndarray,
    d: np.ndarray,
    x_plane: float,
    *,
    y_range: tuple[float, float],
    z_range: tuple[float, float],
) -> tuple[float, float, np.ndarray] | None:
    if abs(d[0]) < _PARALLEL_EPS:
        return None
    s = (x_plane - C[0]) / d[0]
    if s < _RAY_T_EPS:
        return None
    y = float(C[1] + s * d[1])
    z = float(C[2] + s * d[2])
    if not (y_range[0] <= y <= y_range[1]):
        return None
    if not (z_range[0] <= z <= z_range[1]):
        return None
    return 0.0, float(s), np.array([x_plane, y, z], dtype=float)


def _ray_plane_y(
    C: np.ndarray,
    d: np.ndarray,
    y_plane: float,
    *,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
) -> tuple[float, float, np.ndarray] | None:
    if abs(d[1]) < _PARALLEL_EPS:
        return None
    s = (y_plane - C[1]) / d[1]
    if s < _RAY_T_EPS:
        return None
    x = float(C[0] + s * d[0])
    z = float(C[2] + s * d[2])
    if not (x_range[0] <= x <= x_range[1]):
        return None
    if not (z_range[0] <= z <= z_range[1]):
        return None
    return 0.0, float(s), np.array([x, y_plane, z], dtype=float)

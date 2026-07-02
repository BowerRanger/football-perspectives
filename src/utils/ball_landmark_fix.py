"""Landmark-coincidence ball fixes (spec §5.3).

A grounded ball anchor may name a pitch feature it visibly coincides with
(penalty spot, line crossing, corner arc). The feature's exact FIFA world
coordinates make the anchor an exact hard knot: for a point landmark the
world x,y IS the landmark; for a line the clicked-pixel ground ray is
snapped onto the line (1-D). Pure and torch-free — consumed by both anchor
resolution paths and the web suggest endpoint.
"""

from __future__ import annotations

import numpy as np

from src.utils.foot_anchor import ankle_ray_to_pitch
from src.utils.pitch_landmarks import LANDMARK_CATALOGUE
from src.utils.pitch_lines_catalogue import LINE_CATALOGUE

LINE_PREFIX = "line:"
# Point landmarks above this height (crossbar ends, flag tops) are never
# grounded-ball fixes.
_MAX_GROUND_LANDMARK_Z = 0.2


def project_onto_segment_2d(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float]:
    """Closest point to ``p`` on segment ``a``-``b`` (2-D, clamped)."""
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    px, py = float(p[0]), float(p[1])
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 0.0:
        return (ax, ay)
    s = ((px - ax) * dx + (py - ay) * dy) / denom
    s = min(1.0, max(0.0, s))
    return (ax + s * dx, ay + s * dy)


def resolve_landmark_world(
    image_xy: tuple[float, float] | None,
    landmark: str,
    *,
    K: np.ndarray | None,
    R: np.ndarray | None,
    t: np.ndarray | None,
    distortion: tuple[float, float],
    ball_radius: float,
) -> np.ndarray | None:
    """World position of a landmark-coincident grounded ball anchor.

    Point landmark: the landmark's x,y at ball height (camera-independent).
    ``line:<name>``: clicked-pixel ground ray at z=ball_radius, projected
    onto the line segment. None when unresolvable.
    """
    if landmark.startswith(LINE_PREFIX):
        seg = LINE_CATALOGUE.get(landmark[len(LINE_PREFIX):])
        if seg is None or image_xy is None \
                or K is None or R is None or t is None:
            return None
        try:
            ground = ankle_ray_to_pitch(
                (float(image_xy[0]), float(image_xy[1])),
                K=K, R=R, t=t, plane_z=ball_radius, distortion=distortion,
            )
        except ValueError:
            return None
        (ax, ay, _), (bx, by, _) = seg
        sx, sy = project_onto_segment_2d(
            (float(ground[0]), float(ground[1])), (ax, ay), (bx, by))
        return np.array([sx, sy, ball_radius], dtype=float)
    lm = LANDMARK_CATALOGUE.get(landmark)
    if lm is None:
        return None
    return np.array(
        [lm.world_xyz[0], lm.world_xyz[1], ball_radius], dtype=float)


def suggest_pitch_fixes(
    ground_xy: tuple[float, float],
    *,
    max_distance_m: float = 2.0,
    limit: int = 5,
) -> list[dict]:
    """Pitch features near a ground point, nearest first.

    Lines are named with the ``line:`` prefix so a suggestion's ``name``
    can be stored directly in ``BallAnchor.landmark``.
    """
    gx, gy = float(ground_xy[0]), float(ground_xy[1])
    items: list[dict] = []
    for lm in LANDMARK_CATALOGUE.values():
        if lm.world_xyz[2] > _MAX_GROUND_LANDMARK_Z:
            continue
        d = float(np.hypot(lm.world_xyz[0] - gx, lm.world_xyz[1] - gy))
        if 0.0 < d <= max_distance_m:
            items.append({
                "name": lm.name, "kind": "landmark", "distance_m": d,
                "world_xy": [lm.world_xyz[0], lm.world_xyz[1]],
            })
    for name, ((ax, ay, az), (bx, by, bz)) in LINE_CATALOGUE.items():
        # Only consider ground-level lines (both endpoints at z <= 0.2).
        # Exclude pitch-boundary lines (goal lines, touchlines).
        if az > _MAX_GROUND_LANDMARK_Z or bz > _MAX_GROUND_LANDMARK_Z:
            continue
        if "goal_line" in name or "touchline" in name:
            continue
        sx, sy = project_onto_segment_2d((gx, gy), (ax, ay), (bx, by))
        d = float(np.hypot(sx - gx, sy - gy))
        if 0.0 < d <= max_distance_m:
            items.append({
                "name": f"{LINE_PREFIX}{name}", "kind": "line",
                "distance_m": d, "world_xy": [sx, sy],
            })
    items.sort(key=lambda i: (i["distance_m"], i["name"]))
    return items[:limit]

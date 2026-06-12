"""Centre-circle (and arc) detection from a bootstrap camera.

The straight-line detector (``line_detector.py``) cannot see the centre
circle — it fits straight segments. But the centre circle is exactly the
feature that defines a midfield frame (where the penalty box, and hence most
straight lines, is out of view). Without it those frames have too few
independent straight lines and get dropped by the static-line solve.

This detector samples points around the projected world circle, strip-searches
the painted-line ridge perpendicular to the circle's tangent at each sample
(reusing the line detector's sub-pixel cross-section sampler), and returns the
sub-pixel ``(world_xyz, image_xy)`` correspondences. The static-camera solve
consumes them as weighted point constraints — a strong, well-distributed
constraint that pins rotation + focal on midfield frames.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from src.utils.camera_projection import project_world_to_image
from src.utils.line_detector import (
    DetectorConfig,
    _prepare_frame,
    _sample_centreline_offset,
)

# FIFA centre circle: radius 9.15 m about the centre spot (pitch centre).
CENTRE_CIRCLE_CENTRE = (52.5, 34.0, 0.0)
CENTRE_CIRCLE_RADIUS = 9.15


class CircleDetection(NamedTuple):
    name: str
    world_points: tuple[tuple[float, float, float], ...]
    image_points: tuple[tuple[float, float], ...]
    confidence: float
    """Fraction of in-view samples that yielded a usable ridge detection."""


def detect_circle(
    frame_bgr: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float],
    cfg: DetectorConfig,
    *,
    centre: tuple[float, float, float] = CENTRE_CIRCLE_CENTRE,
    radius: float = CENTRE_CIRCLE_RADIUS,
    name: str = "centre_circle",
    n_samples: int = 72,
    min_points: int = 12,
) -> CircleDetection | None:
    """Detect a painted ground circle near where it projects under (K,R,t).

    Returns ``None`` if too few samples are in front of the camera / in frame
    / yield a ridge (``< min_points``), or confidence < ``cfg.min_confidence``.
    """
    gray, green = _prepare_frame(frame_bgr, cfg)
    h, w = gray.shape

    cx, cy, cz = centre
    thetas = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    world = np.stack(
        [cx + radius * np.cos(thetas), cy + radius * np.sin(thetas),
         np.full(n_samples, cz)], axis=1
    )

    # In-front mask: cam_z = world @ R[2] + t[2] must be positive.
    cam_z = world @ np.asarray(R)[2] + np.asarray(t)[2]
    in_front = cam_z > 0.1
    if in_front.sum() < min_points:
        return None

    proj = np.full((n_samples, 2), np.nan)
    proj[in_front] = project_world_to_image(
        K, R, t, distortion, world[in_front]
    )

    det_world: list[tuple[float, float, float]] = []
    det_img: list[tuple[float, float]] = []
    n_in_view = 0
    for i in range(n_samples):
        p = proj[i]
        if not np.all(np.isfinite(p)):
            continue
        if not (0 <= p[0] < w and 0 <= p[1] < h):
            continue
        n_in_view += 1
        # tangent from finite-difference of neighbours that are in front
        p_prev = proj[(i - 1) % n_samples]
        p_next = proj[(i + 1) % n_samples]
        ref_a = p_prev if np.all(np.isfinite(p_prev)) else p
        ref_b = p_next if np.all(np.isfinite(p_next)) else p
        tangent = ref_b - ref_a
        if np.linalg.norm(tangent) < 1e-6:
            continue
        tangent = tangent / np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        out = _sample_centreline_offset(gray, green, p, normal, cfg)
        if out is None:
            continue
        pt, _strength = out
        det_world.append(tuple(float(x) for x in world[i]))
        det_img.append((float(pt[0]), float(pt[1])))

    if len(det_img) < min_points:
        return None
    confidence = len(det_img) / max(1, n_in_view)
    if confidence < cfg.min_confidence:
        return None
    return CircleDetection(
        name=name,
        world_points=tuple(det_world),
        image_points=tuple(det_img),
        confidence=float(confidence),
    )

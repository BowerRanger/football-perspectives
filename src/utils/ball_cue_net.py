"""Goal-net motion-energy event cue: when a goal is in view, watch for a
camera-compensated frame-difference spike inside the net's projected
region -- a candidate ``goal_impact`` -- and back-project the energy
centroid onto the net's known 3D plane for a location.

Reuses ``ball_motion_flow.frame_homography``/``warp_to_reference`` (the
pipeline's existing camera-compensated background-alignment machinery)
and ``goal_geometry`` (the existing goal-element pixel-ray intersection)
rather than re-deriving either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import cv2
import numpy as np

from src.utils.ball_hybrid_types import CueEvidence
from src.utils.ball_motion_flow import frame_homography, warp_to_reference
from src.utils.camera_projection import project_world_to_image
from src.utils.goal_geometry import GoalGeometry, goal_element_candidates

Camera = tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]  # K, R, t, distortion
Box = tuple[float, float, float, float]

# Minimum polygon area (px^2) for a projected net region to be considered
# "in view" rather than a degenerate sliver at the image edge.
_MIN_POLYGON_AREA_PX = 200.0
# How far outside the image bounds a projected corner may fall and the
# goal still count as (partially) in view -- projected corners near an
# off-screen post/net edge are common and still useful.
_BOUNDS_MARGIN_FRAC = 0.25


def net_region_polygon(
    K: np.ndarray, R: np.ndarray, t: np.ndarray,
    distortion: tuple[float, float],
    geometry: GoalGeometry,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, str] | None:
    """Project one goal's net volume (bounded by the posts/crossbar,
    extruded ``net_depth`` behind the goal line) into pixel space for
    this frame's camera.

    Tries both goals and returns the convex hull (image-clipped
    ``int32`` ``(N,2)`` points) for whichever projects with the larger
    in-image area, or ``None`` if neither goal is meaningfully in view
    (any corner behind the camera, or too little of the hull inside the
    image bounds).
    """
    w, h = image_size
    best: tuple[np.ndarray, str, float] | None = None
    for side, goal_x in (("near", geometry.goal_line_x_near),
                         ("far", geometry.goal_line_x_far)):
        sign = -1.0 if side == "near" else 1.0
        back_x = goal_x + sign * geometry.net_depth
        xs = sorted((goal_x, back_x))
        corners = np.array([
            [x, y, z]
            for x in xs
            for y in (geometry.post_y_left, geometry.post_y_right)
            for z in (0.0, geometry.crossbar_z)
        ], dtype=float)

        cam_pts = (np.asarray(R, float) @ corners.T).T + np.asarray(t, float)
        if np.any(cam_pts[:, 2] <= 1e-3):
            continue  # some corner behind the camera -- skip this goal

        pix = project_world_to_image(
            np.asarray(K, float), np.asarray(R, float), np.asarray(t, float),
            distortion, corners)
        if not np.all(np.isfinite(pix)):
            continue
        margin_w, margin_h = w * _BOUNDS_MARGIN_FRAC, h * _BOUNDS_MARGIN_FRAC
        in_bounds_frac = float(np.mean(
            (pix[:, 0] >= -margin_w) & (pix[:, 0] <= w + margin_w) &
            (pix[:, 1] >= -margin_h) & (pix[:, 1] <= h + margin_h)))
        if in_bounds_frac < 0.5:
            continue
        hull = cv2.convexHull(pix.astype(np.float32)).reshape(-1, 2)
        clipped = np.clip(hull, [0, 0], [w - 1, h - 1]).astype(np.int32)
        area = float(cv2.contourArea(clipped))
        if area < _MIN_POLYGON_AREA_PX:
            continue
        if best is None or area > best[2]:
            best = (clipped, side, area)
    if best is None:
        return None
    return best[0], best[1]


@dataclass(frozen=True)
class NetFrameEnergy:
    frame: int
    energy: float  # mean abs-diff per in-polygon pixel, players excluded
    centroid_uv: tuple[float, float] | None
    goal_side: str


def net_energy_series(
    frame_pairs: Iterable[tuple[int, np.ndarray, np.ndarray]],
    cameras: dict[int, Camera],
    geometry: GoalGeometry,
    image_size: tuple[int, int],
    *,
    player_boxes: dict[int, list[Box]] | None = None,
    diff_thresh: int = 18,
) -> list[NetFrameEnergy]:
    """Per-frame net-region motion energy.

    ``frame_pairs`` yields ``(frame_idx, prev_bgr, cur_bgr)`` for
    consecutive decoded frames (``frame_idx`` indexes ``cur_bgr``).
    ``cameras`` maps ``frame_idx -> (K, R, t, distortion)``. A frame is
    skipped (absent from the output, not zero-filled) when its camera is
    missing or neither goal is in view for it. Player boxes for
    ``frame_idx`` are zeroed out of both the energy sum and the
    diff-mask used for the impact centroid.
    """
    player_boxes = player_boxes or {}
    out: list[NetFrameEnergy] = []
    for frame_idx, prev_bgr, cur_bgr in frame_pairs:
        cam_cur = cameras.get(frame_idx)
        if cam_cur is None:
            continue
        K1, R1, t1, dist = cam_cur
        poly = net_region_polygon(K1, R1, t1, dist, geometry, image_size)
        if poly is None:
            continue
        hull, side = poly
        h, w = cur_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        for (x0, y0, x1, y1) in player_boxes.get(frame_idx, []):
            cv2.rectangle(mask, (int(x0), int(y0)), (int(x1), int(y1)), 0, -1)

        gp = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY) if prev_bgr.ndim == 3 else prev_bgr
        gc = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY) if cur_bgr.ndim == 3 else cur_bgr
        cam_prev = cameras.get(frame_idx - 1)
        if cam_prev is not None:
            K0, R0, _t0, _d0 = cam_prev
            H = frame_homography(K1, R1, K0, R0)
            gp = warp_to_reference(gp, H, (w, h))

        diff = cv2.absdiff(gc, gp).astype(np.float64)
        diff[mask == 0] = 0.0
        area = float(np.count_nonzero(mask))
        energy = float(diff.sum() / area) if area > 0 else 0.0
        centroid = None
        thresh_mask = diff > diff_thresh
        if thresh_mask.any():
            ys, xs = np.nonzero(thresh_mask)
            weights = diff[ys, xs]
            centroid = (float(np.average(xs, weights=weights)),
                        float(np.average(ys, weights=weights)))
        out.append(NetFrameEnergy(frame=frame_idx, energy=energy,
                                   centroid_uv=centroid, goal_side=side))
    return out


def net_energy_onsets(
    series: list[NetFrameEnergy],
    cameras: dict[int, Camera],
    geometry: GoalGeometry,
    *,
    k_mad: float = 4.0,
    min_gap_frames: int = 5,
) -> list[CueEvidence]:
    """Adaptive-threshold peak picking over ``net_energy_series``'s
    energy, with each accepted peak's location back-projected onto the
    net-element the ray through its diff centroid actually hits
    (``goal_geometry.goal_element_candidates``, nearest-hit wins)."""
    if len(series) < 5:
        return []
    ordered = sorted(series, key=lambda s: s.frame)
    energies = np.array([s.energy for s in ordered])
    med = float(np.median(energies))
    mad = float(np.median(np.abs(energies - med))) + 1e-9
    thresh = med + k_mad * mad

    events: list[CueEvidence] = []
    last_frame = -min_gap_frames - 1
    for s in ordered:
        if s.energy <= thresh or s.centroid_uv is None:
            continue
        if s.frame - last_frame < min_gap_frames:
            continue
        cam = cameras.get(s.frame)
        if cam is None:
            continue
        K, R, t, dist = cam
        world = None
        hits = goal_element_candidates(
            s.centroid_uv, K=K, R=R, t=t, distortion=dist, geometry=geometry)
        if hits:
            _el, _res, _s, world = hits[0]
        conf = float(np.clip((s.energy - thresh) / thresh, 0.0, 1.0))
        events.append(CueEvidence(
            frame=s.frame, kind="goal_impact", cue="net_energy", conf=conf,
            xyz=(tuple(float(v) for v in world) if world is not None else None),
            uv=s.centroid_uv,
        ))
        last_frame = s.frame
    return events


__all__ = [
    "net_region_polygon", "NetFrameEnergy", "net_energy_series",
    "net_energy_onsets", "Camera", "Box",
]

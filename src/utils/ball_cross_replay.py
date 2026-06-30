"""Cross-replay triangulation (ball v2 Phase 1.5).

Shots in one sync group film the same real moment from different
cameras. Pairing their per-frame ball detections through the group's
sync offset turns them into an ad-hoc stereo rig: the midpoint of the
two rays' common perpendicular is a 3D fix, and the perpendicular's
length (ray miss) is a built-in consistency gate.

The saved sync offset is refined LOCALLY by minimising median ray miss
over a sub-frame grid — sync_map.json is never written (operator
offsets win); the caller surfaces disagreements as a review cue.

Pure module: no file/video access; the stage owns I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.camera_projection import pixel_ray

# observation maps: frame -> ((u, v), confidence)
Obs = dict[int, tuple[tuple[float, float], float]]
# camera maps: frame -> (K, R, t)
Cams = dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class CrossReplayCfg:
    enabled: bool = True
    min_conf: float = 0.3
    max_ray_miss_m: float = 1.0
    min_parallax_deg: float = 8.0
    offset_search_radius_frames: float = 4.0
    offset_search_step: float = 0.25
    min_pairs_for_refine: int = 8
    fix_weight_px_per_m: float = 30.0


@dataclass(frozen=True)
class PairFix:
    """One triangulated point, keyed by both shots' frames."""

    frame_a: int
    frame_b: int
    xyz: tuple[float, float, float]
    ray_miss_m: float
    parallax_deg: float


def triangulate_rays(
    uv_a: tuple[float, float],
    K_a: np.ndarray,
    R_a: np.ndarray,
    t_a: np.ndarray,
    uv_b: tuple[float, float],
    K_b: np.ndarray,
    R_b: np.ndarray,
    t_b: np.ndarray,
    distortion_a: tuple[float, float] = (0.0, 0.0),
    distortion_b: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, float, float]:
    """Midpoint of the common perpendicular between two pixel rays.

    Returns ``(point, miss_m, parallax_deg)``; ``point`` is NaN and the
    gates are unsatisfiable when either ray depth is non-positive.
    """
    c1, d1 = pixel_ray(uv_a, K_a, R_a, t_a, distortion_a)
    c2, d2 = pixel_ray(uv_b, K_b, R_b, t_b, distortion_b)
    cos = float(np.clip(abs(np.dot(d1, d2)), 0.0, 1.0))
    parallax = float(np.degrees(np.arccos(cos)))
    A = np.stack([d1, -d2], axis=1)
    b = c2 - c1
    (s, u), *_ = np.linalg.lstsq(A, b, rcond=None)
    if s <= 0 or u <= 0:  # intersection behind a camera: invalid pair
        return np.full(3, np.nan), float("inf"), 0.0
    p1 = c1 + s * d1
    p2 = c2 + u * d2
    return (p1 + p2) / 2.0, float(np.linalg.norm(p1 - p2)), parallax


def interp_uv(
    obs: Obs,
    f: float,
    min_conf: float,
    max_span: int = 3,
) -> tuple[float, float] | None:
    """Observation at a (possibly fractional) frame, linearly
    interpolated between the nearest confident detections no more than
    ``max_span`` frames apart."""
    lo, hi = int(np.floor(f)), int(np.ceil(f))
    if lo == hi:
        rec = obs.get(lo)
        if rec is not None and rec[1] >= min_conf:
            return rec[0]
        # Exact frame absent — fall through to neighbour search so that
        # integer-valued queries still interpolate between neighbours.
        hi = lo + 1
    a = next(
        (
            (g, obs[g])
            for g in range(lo, lo - max_span, -1)
            if g in obs and obs[g][1] >= min_conf
        ),
        None,
    )
    b = next(
        (
            (g, obs[g])
            for g in range(hi, hi + max_span)
            if g in obs and obs[g][1] >= min_conf
        ),
        None,
    )
    if a is None or b is None or b[0] - a[0] > max_span:
        return None
    w = (f - a[0]) / (b[0] - a[0])
    ua, va = a[1][0]
    ub, vb = b[1][0]
    return ((1 - w) * ua + w * ub, (1 - w) * va + w * vb)


def _pairs_at_offset(
    obs_a: Obs,
    cams_a: Cams,
    obs_b: Obs,
    cams_b: Cams,
    offset_b_minus_a: float,
    cfg: CrossReplayCfg,
):
    """Yield (frame_a, frame_b, uv_a, uv_b) for every B detection whose
    synced A-frame has (interpolated) evidence.

    Convention (sync_map): B frame f_b shows the instant A saw at
    ``f_a = f_b - offset_b_minus_a``.
    """
    for f_b, (uv_b, conf_b) in sorted(obs_b.items()):
        if conf_b < cfg.min_conf or f_b not in cams_b:
            continue
        f_a = f_b - offset_b_minus_a
        f_a_int = int(round(f_a))
        if f_a_int not in cams_a:
            continue
        uv_a = interp_uv(obs_a, f_a, cfg.min_conf)
        if uv_a is None:
            continue
        yield f_a_int, f_b, uv_a, uv_b


def refine_pair_offset(
    *,
    obs_a: Obs,
    cams_a: Cams,
    obs_b: Obs,
    cams_b: Cams,
    saved_offset: float,
    cfg: CrossReplayCfg,
    distortion_a: tuple[float, float] = (0.0, 0.0),
    distortion_b: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float, int]:
    """Scan offsets around the saved value, minimising median ray miss.

    Returns ``(refined_offset, median_miss_at_refined, n_pairs)``. The
    saved offset is returned unchanged when fewer than
    ``min_pairs_for_refine`` pairs exist at it.
    """

    def _median_miss(offset: float) -> tuple[float, int]:
        misses = []
        for f_a, f_b, uv_a, uv_b in _pairs_at_offset(
            obs_a, cams_a, obs_b, cams_b, offset, cfg,
        ):
            K_a, R_a, t_a = cams_a[f_a]
            K_b, R_b, t_b = cams_b[f_b]
            _, miss, _ = triangulate_rays(
                uv_a, K_a, R_a, t_a, uv_b, K_b, R_b, t_b,
                distortion_a, distortion_b,
            )
            if np.isfinite(miss):
                misses.append(miss)
        if not misses:
            return float("inf"), 0
        return float(np.median(misses)), len(misses)

    base_miss, base_pairs = _median_miss(saved_offset)
    if base_pairs < cfg.min_pairs_for_refine:
        return float(saved_offset), base_miss, base_pairs

    best = (float(saved_offset), base_miss, base_pairs)
    r = cfg.offset_search_radius_frames
    step = cfg.offset_search_step
    for off in np.arange(saved_offset - r, saved_offset + r + step / 2, step):
        miss, n = _median_miss(float(off))
        if n >= cfg.min_pairs_for_refine and miss < best[1]:
            best = (float(off), miss, n)
    return best


def triangulate_pair(
    *,
    obs_a: Obs,
    cams_a: Cams,
    obs_b: Obs,
    cams_b: Cams,
    offset_b_minus_a: float,
    cfg: CrossReplayCfg,
    distortion_a: tuple[float, float] = (0.0, 0.0),
    distortion_b: tuple[float, float] = (0.0, 0.0),
) -> list[PairFix]:
    """Gated triangulation of every synced detection pair."""
    fixes: list[PairFix] = []
    for f_a, f_b, uv_a, uv_b in _pairs_at_offset(
        obs_a, cams_a, obs_b, cams_b, offset_b_minus_a, cfg,
    ):
        K_a, R_a, t_a = cams_a[f_a]
        K_b, R_b, t_b = cams_b[f_b]
        point, miss, parallax = triangulate_rays(
            uv_a, K_a, R_a, t_a, uv_b, K_b, R_b, t_b,
            distortion_a, distortion_b,
        )
        if not np.all(np.isfinite(point)):
            continue
        if miss > cfg.max_ray_miss_m or parallax < cfg.min_parallax_deg:
            continue
        fixes.append(
            PairFix(
                frame_a=f_a,
                frame_b=f_b,
                xyz=(float(point[0]), float(point[1]), float(point[2])),
                ray_miss_m=miss,
                parallax_deg=parallax,
            )
        )
    return fixes

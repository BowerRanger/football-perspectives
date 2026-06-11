"""Advertising-hoarding (LED board) base-edge detection + per-clip geometry.

The board base is a STATIC SCENE LINE parallel to the far touchline:
world model ``y = 68 + d, z = h`` with a single ``(d, h)`` per clip — the
camera rig and the boards don't move. The LED band is the highest-contrast
feature in exactly the far field where pitch lines are starved (the far
touchline hides in its shadow), so once ``(d, h)`` is calibrated the edge
constrains tilt and lens on every frame it is visible — including spans
where no pitch feature is detectable.

Detection targets the band's BOTTOM edge (bright board above, grass below):
a signed step, not a painted ridge, so this module has its own perpendicular
sampler; the line fit and observation format are shared with the painted-line
detector.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from src.utils.camera_projection import project_world_to_image
from src.utils.line_detector import (
    DetectorConfig,
    _parabolic_subpixel,
    _prepare_frame,
)

# Sample the central span only: the boards curve away / change setback near
# the corners in many stadiums, and the model is a straight line.
BOARD_X0 = 10.0
BOARD_X1 = 95.0


class BoardLineModel(NamedTuple):
    d: float          # lateral offset beyond the far touchline (m)
    h: float          # height of the detected edge above ground (m)
    frames: int       # calibration frames with a usable detection
    contrast: float   # median step contrast (grey levels)
    residual: float   # median |point-to-model-line| over calibration (px)


class BoardDetection(NamedTuple):
    name: str
    image_points: tuple[tuple[float, float], ...]
    image_segment: tuple[tuple[float, float], tuple[float, float]]
    world_segment: tuple[tuple[float, float, float], tuple[float, float, float]]
    contrast: float


def board_world_segment(
    d: float, h: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return ((BOARD_X0, 68.0 + d, h), (BOARD_X1, 68.0 + d, h))


def _step_edge_candidates(
    g: np.ndarray, min_step: float, half_w: int = 6, top_k: int = 3,
) -> list[tuple[float, float]]:
    """All prominent bright→dark steps along profile ``g`` (sampled walking
    TOWARD the grass side), strongest first, as (sub-pixel index, contrast).
    Antisymmetric box kernel = mean(above) - mean(below). The caller's
    geometric consensus picks the candidate consistent with a straight
    line — photometric gates alone are too brittle at far field."""
    L = len(g)
    if L < 2 * half_w + 3:
        return []
    kernel = np.concatenate([
        np.full(half_w, 1.0 / half_w), [0.0], np.full(half_w, -1.0 / half_w),
    ]).astype(np.float32)
    resp = np.convolve(g, kernel[::-1], mode="same")
    resp[: half_w + 1] = -np.inf
    resp[-(half_w + 1):] = -np.inf
    out: list[tuple[float, float]] = []
    r = resp.copy()
    for _ in range(top_k):
        peak = int(np.argmax(r))
        if not np.isfinite(r[peak]) or r[peak] < min_step:
            break
        if 0 < peak < L - 1:
            sub = _parabolic_subpixel(resp[peak - 1], resp[peak],
                                      resp[peak + 1])
            out.append((peak + sub, float(resp[peak])))
        # suppress the peak's neighbourhood before the next pick
        lo = max(0, peak - half_w)
        hi = min(L, peak + half_w + 1)
        r[lo:hi] = -np.inf
    return out


def detect_board_line(
    frame_bgr: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float],
    d: float,
    h: float,
    cfg: DetectorConfig | None = None,
    *,
    n_samples: int = 64,
    strip_px: int = 30,
    min_step: float = 25.0,
    min_points: int = 6,
) -> BoardDetection | None:
    """Detect the board base edge around the projected ``(d, h)`` model line.

    Walks perpendicular profiles at ``n_samples`` points along the projected
    line, locates the bright→grass step at each, requires grass (green mask)
    on the below side, and needs ``min_points`` agreeing detections.
    """
    cfg = cfg or DetectorConfig()
    gray, _green = _prepare_frame(frame_bgr, cfg)
    h_img, w_img = gray.shape
    # Far-field grass under floodlights is washed out (saturation below the
    # painted-line detector's S>=40 gate), so the polarity check uses a LOOSE
    # grassiness mask and a RELATIVE comparison: the below-edge patch must be
    # distinctly greener than the above-edge (board/crowd) patch.
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    grassy = ((hsv[..., 0] >= cfg.green_h_low)
              & (hsv[..., 0] <= cfg.green_h_high)
              & (hsv[..., 1] >= 15)).astype(np.float32)

    def _grassiness(u: float, v: float) -> float:
        x, y = int(round(u)), int(round(v))
        if not (3 <= x < w_img - 3 and 3 <= y < h_img - 3):
            return -1.0
        return float(grassy[y - 3:y + 4, x - 3:x + 4].mean())

    xs = np.linspace(BOARD_X0, BOARD_X1, n_samples)
    world = np.stack([xs, np.full_like(xs, 68.0 + d), np.full_like(xs, h)],
                     axis=1)
    cam = world @ np.asarray(R).T + np.asarray(t)
    proj = project_world_to_image(K, R, t, distortion, world)
    # Grass-side reference: a world point 1 m toward the pitch tells us which
    # image direction is "down the step".
    grass = world.copy()
    grass[:, 1] -= 1.0
    grass[:, 2] = 0.0
    proj_g = project_world_to_image(K, R, t, distortion, grass)

    in_view = ((cam[:, 2] > 1.0) & np.isfinite(proj).all(axis=1)
               & (proj[:, 0] >= 0) & (proj[:, 0] < w_img)
               & (proj[:, 1] >= -strip_px) & (proj[:, 1] < h_img))
    if in_view.sum() < min_points:
        return None

    # Collect step candidates per perpendicular (sample index, u, v,
    # contrast, grassiness-below-minus-above as a soft score component).
    cands: list[tuple[int, float, float, float, float]] = []
    offsets = np.arange(-strip_px, strip_px + 1)
    for idx in np.where(in_view)[0]:
        centre = proj[idx]
        normal = proj_g[idx] - centre
        nn = np.linalg.norm(normal)
        if not np.isfinite(nn) or nn < 1e-6:
            continue
        normal = normal / nn
        sx = centre[0] + offsets * normal[0]
        sy = centre[1] + offsets * normal[1]
        ok = (sx >= 0) & (sx < w_img - 1) & (sy >= 0) & (sy < h_img - 1)
        if ok.sum() < 15:
            continue
        x0 = np.floor(sx[ok]).astype(np.int32)
        y0 = np.floor(sy[ok]).astype(np.int32)
        fx = sx[ok] - x0
        fy = sy[ok] - y0
        g = (gray[y0, x0] * (1 - fx) * (1 - fy)
             + gray[y0, x0 + 1] * fx * (1 - fy)
             + gray[y0 + 1, x0] * (1 - fx) * fy
             + gray[y0 + 1, x0 + 1] * fx * fy)
        valid_offsets = offsets[ok].astype(np.float64)
        for sub_idx, contrast in _step_edge_candidates(
                g.astype(np.float32), min_step):
            base = int(np.floor(sub_idx))
            if base < 0 or base >= len(valid_offsets) - 1:
                continue
            frac = sub_idx - base
            off = ((1 - frac) * valid_offsets[base]
                   + frac * valid_offsets[base + 1])
            u = float(centre[0] + off * normal[0])
            v = float(centre[1] + off * normal[1])
            g_rel = (_grassiness(u + 6 * normal[0], v + 6 * normal[1])
                     - _grassiness(u - 6 * normal[0], v - 6 * normal[1]))
            cands.append((int(idx), u, v, contrast, g_rel))

    if len(cands) < min_points:
        return None

    # Geometric consensus: the board base is the straight line supported by
    # the most perpendiculars (one inlier per sample). Crowd/board-graphic
    # steps are uncorrelated with any straight line; ties break toward the
    # grassier-below, higher-contrast candidate set.
    arr = np.array([(c[1], c[2]) for c in cands])
    sample_ids = np.array([c[0] for c in cands])
    rng = np.random.default_rng(0)
    best_inl: np.ndarray | None = None
    best_score = -1.0
    n = len(cands)
    for _ in range(min(200, n * (n - 1) // 2 + 1)):
        i, j = rng.choice(n, size=2, replace=False)
        if sample_ids[i] == sample_ids[j]:
            continue
        p0, p1 = arr[i], arr[j]
        dvec = p1 - p0
        nl = np.linalg.norm(dvec)
        if nl < 20.0:
            continue
        nrm = np.array([-dvec[1], dvec[0]]) / nl
        res = np.abs((arr - p0) @ nrm)
        inl = res < 2.5
        # one inlier per perpendicular: prefer the closest
        seen: dict[int, int] = {}
        for k in np.where(inl)[0]:
            s = int(sample_ids[k])
            if s not in seen or res[k] < res[seen[s]]:
                seen[s] = int(k)
        ks = np.array(sorted(seen.values()))
        if len(ks) < min_points:
            continue
        score = (len(ks)
                 + 0.2 * float(np.mean([cands[k][4] for k in ks]))
                 + 0.001 * float(np.median([cands[k][3] for k in ks])))
        if score > best_score:
            best_score = score
            best_inl = ks
    if best_inl is None:
        return None

    arr = arr[best_inl]
    contrasts = np.array([cands[k][3] for k in best_inl])
    # Refit by PCA on the consensus set.
    mean = arr.mean(axis=0)
    _, _, vt = np.linalg.svd(arr - mean)
    direction = vt[0]
    proj_1d = (arr - mean) @ direction
    a = mean + proj_1d.min() * direction
    b = mean + proj_1d.max() * direction
    return BoardDetection(
        name="board_line",
        image_points=tuple((float(p[0]), float(p[1])) for p in arr),
        image_segment=((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))),
        world_segment=board_world_segment(d, h),
        contrast=float(np.median(contrasts)),
    )


def calibrate_board_line(
    frames_bgr: dict[int, np.ndarray],
    cams: dict[int, dict],
    distortion: tuple[float, float],
    cfg: DetectorConfig | None = None,
    *,
    d_grid: np.ndarray | None = None,
    h_grid: np.ndarray | None = None,
    min_frames: int = 3,
) -> BoardLineModel | None:
    """Solve the per-clip board geometry ``(d, h)``.

    Coarse grid: score each (d, h) by the number of frames with a usable
    detection and the summed point support. Refine: least-squares (d, h)
    minimising the reprojection distance of all detected edge points to the
    projected model line across the calibration frames.
    """
    cfg = cfg or DetectorConfig()
    # With a STATIC camera centre, (d, h) is a one-parameter family: every
    # line on the plane through C and the image edge projects identically in
    # every frame, so the family member is arbitrary — fix h = 0 and solve d
    # alone. d >= 1 keeps the model from binding to the far touchline itself
    # (already a catalogue constraint).
    if d_grid is None:
        d_grid = np.arange(0.5, 10.1, 1.0)
    if h_grid is None:
        h_grid = np.array([0.0])

    best = None  # (score, d, h)
    for d in d_grid:
        for h in h_grid:
            n_pts = 0
            n_frames = 0
            for fid, img in frames_bgr.items():
                c = cams[fid]
                det = detect_board_line(
                    img, np.asarray(c["K"]), np.asarray(c["R"]),
                    np.asarray(c["t"]), distortion, float(d), float(h), cfg)
                if det is None:
                    continue
                n_frames += 1
                n_pts += len(det.image_points)
            if n_frames >= min_frames:
                score = n_pts
                if best is None or score > best[0]:
                    best = (score, float(d), float(h))
    if best is None:
        return None
    _, d0, h0 = best

    # Refine d: solve INDEPENDENTLY per frame (the detected edge is sub-px
    # precise within a frame; the inconsistency lives between frames, whose
    # cameras carry exactly the far-field error this constraint will fix),
    # then take the median across frames. Iterate so detections can follow.
    def _frame_residuals(dd: float, c: dict, pts: np.ndarray) -> np.ndarray:
        xs = np.linspace(BOARD_X0, BOARD_X1, 24)
        world = np.stack([
            xs, np.full_like(xs, 68.0 + dd), np.zeros_like(xs)], axis=1)
        proj = project_world_to_image(
            np.asarray(c["K"]), np.asarray(c["R"]), np.asarray(c["t"]),
            distortion, world)
        fin = np.isfinite(proj).all(axis=1)
        if fin.sum() < 2:
            return np.array([1e3])
        pp = proj[fin]
        dvec = pp[-1] - pp[0]
        nn = np.array([-dvec[1], dvec[0]])
        nl = np.linalg.norm(nn)
        if nl < 1e-6:
            return np.array([1e3])
        nn = nn / nl
        return (pts - pp[0]) @ nn

    d_cur = d0
    last = None
    for _ in range(2):
        d_is: list[float] = []
        contrasts = []
        pxres: list[float] = []
        for fid, img in frames_bgr.items():
            c = cams[fid]
            det = detect_board_line(
                img, np.asarray(c["K"]), np.asarray(c["R"]),
                np.asarray(c["t"]), distortion, d_cur, 0.0, cfg)
            if det is None:
                continue
            pts = np.asarray(det.image_points, dtype=float)
            sol = least_squares(
                lambda p, c=c, pts=pts: _frame_residuals(
                    float(np.clip(p[0], 0.3, 14.0)), c, pts),
                np.array([d_cur]), method="lm", max_nfev=40)
            d_i = float(np.clip(sol.x[0], 0.3, 14.0))
            d_is.append(d_i)
            contrasts.append(det.contrast)
            pxres.append(float(np.median(np.abs(
                _frame_residuals(d_i, c, pts)))))
        if len(d_is) < min_frames:
            return last
        d_cur = float(np.median(d_is))
        # residual = cross-frame consistency: median |d_i - median| expressed
        # in pixels via each frame's own scale is overkill — report the
        # median per-frame fit residual plus the d spread in metres.
        last = BoardLineModel(
            d=d_cur, h=0.0, frames=len(d_is),
            contrast=float(np.median(contrasts)) if contrasts else 0.0,
            residual=float(np.median(np.abs(np.asarray(d_is) - d_cur))))
    return last

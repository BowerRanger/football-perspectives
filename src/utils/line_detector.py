"""Per-frame painted-line detection from a bootstrap camera.

Given a frame and an initial camera estimate (K, R, t, distortion), project
each world line from the line catalogue, search a strip around the
projected line for white painted pixels on green grass, and fit a sub-
pixel-accurate line back from the detected pixels.

Sub-pixel localisation: a painted pitch line has two image-side edges
(grass→paint and paint→grass). Along each perpendicular cross-section
we extract the intensity profile, compute its 1-D gradient, and locate
the positive and negative gradient peaks via quadratic interpolation.
The CENTRELINE of the painted feature lies exactly between those two
peaks — accurate to ~0.1 px under good contrast, regardless of how wide
the painted line is in pixels.

The output is a list of ``DetectedLine`` per frame: refined image
endpoints + a confidence score. The camera solver consumes these as
``LineObservation`` constraints (sub-pixel residuals) instead of point-
landmark clicks (which carry 1–3 px placement noise).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np

from src.utils.camera_projection import project_world_to_image


class DetectedLine(NamedTuple):
    name: str
    image_segment: tuple[tuple[float, float], tuple[float, float]]
    world_segment: tuple[tuple[float, float, float], tuple[float, float, float]]
    confidence: float
    """Fraction of cross-sections that yielded a usable detection (0..1)."""
    n_samples: int
    """Number of cross-sections actually sampled along the projected line."""


@dataclass(frozen=True)
class DetectorConfig:
    search_strip_px: int = 30
    """Perpendicular half-width of the search strip around the projected line."""

    sample_step_px: float = 3.0
    """Spacing along the projected line at which to take cross-sections."""

    min_gradient: float = 10.0
    """Minimum ridge-response magnitude (≈ paint-vs-flank contrast in
    grey levels). Broadcast white paint over green grass shows up at
    contrast 30–80 under typical lighting; 10 is the safe floor that
    captures shadow / far-side lines while still rejecting grass
    texture (which has contrast ≤ 4)."""

    min_paint_width_px: float = 1.0
    max_paint_width_px: float = 20.0
    """Bounds on the gap between the two edges per cross-section. A
    painted football line is 12 cm wide; in the image that projects to
    2–6 px on the near side and 1–2 px on the far side. Cross-sections
    whose edge pair is outside this range are rejected as noise."""

    green_h_low: int = 30
    """HSV-H lower bound for "is on grass" — used to suppress non-pitch
    pixels (crowd, ads). Football grass hue is ≈ 30–90."""

    green_h_high: int = 90

    grass_check_radius_px: int = 10
    """How far around a candidate centreline pixel to look for grass.
    A painted line on grass has green pixels within this distance."""

    grass_check_min_count: int = 5
    """Minimum green-pixel count within the radius for a centreline
    pixel to count as "painted on grass" (vs e.g. ad-board text)."""

    min_confidence: float = 0.4
    """Reject the line if fewer than this fraction of cross-sections
    yielded a usable detection."""


def _prepare_frame(frame_bgr: np.ndarray, cfg: DetectorConfig) -> tuple[np.ndarray, np.ndarray]:
    """Returns (gray_float32, green_mask_uint8).
    Gray is used for sub-pixel edge gradients; green mask is used to
    reject paint detected on non-grass backgrounds (ad boards, crowd)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    H, S = hsv[..., 0], hsv[..., 1]
    green = ((H >= cfg.green_h_low) & (H <= cfg.green_h_high) & (S >= 40)).astype(np.uint8) * 255
    return gray, green


def _parabolic_subpixel(y_prev: float, y_at: float, y_next: float) -> float:
    """Sub-pixel offset of the local extremum near samples (y_prev, y_at, y_next).
    Returns an offset in [-0.5, +0.5] from the centre sample. Used to
    locate gradient peaks at sub-pixel resolution."""
    denom = (y_prev - 2.0 * y_at + y_next)
    if abs(denom) < 1e-9:
        return 0.0
    val = 0.5 * (y_prev - y_next) / denom
    if not np.isfinite(val):
        return 0.0
    return float(np.clip(val, -0.5, 0.5))


def _sample_centreline_offset(
    gray: np.ndarray,
    green_mask: np.ndarray,
    centre: np.ndarray,
    normal: np.ndarray,
    cfg: DetectorConfig,
) -> tuple[np.ndarray, float] | None:
    """Walk a perpendicular cross-section at ``centre`` along ``normal``.
    Returns the sub-pixel image point of the painted-line centreline if
    a valid bright ridge is found inside the search strip; otherwise
    ``None``.

    Algorithm: convolve the intensity profile with a "bright ridge"
    template (positive lobe of width ~paint width, flanked by negative
    lobes of grass background). The peak response identifies the
    centreline; sub-pixel position comes from parabolic interpolation
    around the peak. This is robust to wider light regions in the strip
    (e.g. mowing stripes adjacent to the painted line) because the
    template requires both the central bright lobe AND darker flanks.
    """
    h, w = gray.shape
    n = cfg.search_strip_px
    offsets = np.arange(-n, n + 1)
    xs = centre[0] + offsets * normal[0]
    ys = centre[1] + offsets * normal[1]
    valid = (xs >= 0) & (xs < w - 1) & (ys >= 0) & (ys < h - 1)
    if valid.sum() < 11:
        return None
    xi = xs[valid]; yi = ys[valid]
    x0 = np.floor(xi).astype(np.int32); y0 = np.floor(yi).astype(np.int32)
    fx = xi - x0; fy = yi - y0
    g = (gray[y0, x0] * (1 - fx) * (1 - fy)
         + gray[y0, x0 + 1] * fx * (1 - fy)
         + gray[y0 + 1, x0] * (1 - fx) * fy
         + gray[y0 + 1, x0 + 1] * fx * fy)
    L = len(g)
    if L < 11:
        return None

    # Bright-ridge template: +1 in the centre over [-paint_w/2, +paint_w/2],
    # -1 in flanks 2*paint_w wide, zero-mean so it's response = bright
    # central strip vs darker flanks. Match-filtered correlation gives
    # peak at line centre, with magnitude proportional to (paint-grass)
    # contrast and to paint width. Sub-pixel via parabola on peaks.
    paint_w = 3
    flank_w = 8
    # Build kernel of total length 2*flank_w + paint_w
    klen = paint_w + 2 * flank_w
    half = klen // 2
    kernel = np.zeros(klen, dtype=np.float32)
    kernel[flank_w:flank_w + paint_w] = +1.0 / paint_w
    kernel[:flank_w] = -0.5 / flank_w
    kernel[-flank_w:] = -0.5 / flank_w
    # Convolve (valid mode would shorten; use same with edge handling)
    # Use np.convolve in 'same' mode for simplicity; edges are ignored later.
    resp = np.convolve(g, kernel[::-1], mode='same')
    # Suppress edges
    edge_zone = flank_w + 1
    resp[:edge_zone] = -np.inf
    resp[-edge_zone:] = -np.inf
    peak_idx = int(np.argmax(resp))
    if not np.isfinite(resp[peak_idx]) or resp[peak_idx] < cfg.min_gradient:
        return None
    # Sub-pixel parabola
    if peak_idx <= 0 or peak_idx >= L - 1:
        return None
    sub_off = _parabolic_subpixel(resp[peak_idx - 1], resp[peak_idx], resp[peak_idx + 1])
    mid_sub = peak_idx + sub_off
    if not np.isfinite(mid_sub):
        return None

    # Verify there's a brighter-than-flanks structure at this peak
    centre_int = float(g[peak_idx])
    flank_left = float(g[max(0, peak_idx - flank_w):peak_idx - 1].mean()) if peak_idx > 1 else centre_int
    flank_right = float(g[peak_idx + 2:min(L, peak_idx + flank_w + 1)].mean()) if peak_idx < L - 2 else centre_int
    contrast = centre_int - 0.5 * (flank_left + flank_right)
    if contrast < cfg.min_gradient:
        return None

    valid_offsets = offsets[valid].astype(np.float64)
    if mid_sub < 0 or mid_sub > len(valid_offsets) - 1:
        return None
    base = int(np.floor(mid_sub)); frac = mid_sub - base
    nb = min(base + 1, len(valid_offsets) - 1)
    real_offset = (1.0 - frac) * valid_offsets[base] + frac * valid_offsets[nb]
    cx_sub = float(centre[0] + real_offset * normal[0])
    cy_sub = float(centre[1] + real_offset * normal[1])
    xi_c = int(round(cx_sub)); yi_c = int(round(cy_sub))
    if 0 <= xi_c < w and 0 <= yi_c < h:
        r = cfg.grass_check_radius_px
        y0c = max(0, yi_c - r); y1c = min(h, yi_c + r + 1)
        x0c = max(0, xi_c - r); x1c = min(w, xi_c + r + 1)
        green_count = int(green_mask[y0c:y1c, x0c:x1c].sum() // 255)
        if green_count < cfg.grass_check_min_count:
            return None
    return np.array([cx_sub, cy_sub]), float(contrast)


# A world point just in front of the camera (cam_z above the behind-camera
# threshold) but far off-axis still perspective-divides to a coordinate
# millions of pixels out — a "grazing-incidence endpoint at the horizon".
# No real broadcast frame is anywhere near this large, so a projection past
# this bound means the segment can't be meaningfully strip-searched.
_MAX_PROJ_COORD_PX = 1e5


def _project_endpoints(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float],
    world_a: tuple[float, float, float],
    world_b: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Project the world endpoints under the given camera + distortion.

    Returns ``None`` if either endpoint lies behind the camera *or* at
    grazing incidence to it. The behind-camera check (``cam_z`` near zero)
    alone is not enough: a point with ``cam_z`` comfortably positive but
    far off-axis — a world line crossing the camera's horizon, e.g. the
    far end of a full-length touchline in a tight penalty-box shot — still
    projects to millions of pixels. That degenerate "line" then gets
    strip-searched and the detector locks onto whatever spurious feature
    happens to fall in the meaningless strip, so we reject it here.
    """
    pts = np.array([world_a, world_b], dtype=np.float64)
    cam_a = R @ pts[0] + t
    cam_b = R @ pts[1] + t
    if cam_a[2] <= 0.1 or cam_b[2] <= 0.1:
        return None
    proj = project_world_to_image(K, R, t, distortion, pts)
    if not np.all(np.isfinite(proj)) or np.abs(proj).max() > _MAX_PROJ_COORD_PX:
        return None
    return proj[0], proj[1]


def _clip_segment_to_image(
    pa: np.ndarray, pb: np.ndarray, w: int, h: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip a 2D segment to the image rectangle [0,w) x [0,h).
    Returns the clipped endpoints, or ``None`` if the segment lies
    entirely outside.
    """
    # Liang–Barsky clipping.
    p1 = pa.astype(np.float64)
    p2 = pb.astype(np.float64)
    dx, dy = p2 - p1
    u1, u2 = 0.0, 1.0
    for edge, p, q in (
        ("left",   -dx,    p1[0] - 0.0),
        ("right",   dx,  (w - 1.0) - p1[0]),
        ("bottom", -dy,    p1[1] - 0.0),
        ("top",     dy,  (h - 1.0) - p1[1]),
    ):
        if abs(p) < 1e-9:
            if q < 0:
                return None
        else:
            r = q / p
            if p < 0:
                if r > u2:
                    return None
                if r > u1:
                    u1 = r
            else:
                if r < u1:
                    return None
                if r < u2:
                    u2 = r
    clipped_a = p1 + u1 * np.array([dx, dy])
    clipped_b = p1 + u2 * np.array([dx, dy])
    return clipped_a, clipped_b


def _project_visible_segment(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float],
    world_a: tuple[float, float, float],
    world_b: tuple[float, float, float],
    image_w: int,
    image_h: int,
    *,
    sample_step_m: float = 1.0,
    margin_px: float = 5.0,
    min_run_samples: int = 4,
    min_segment_px: float = 20.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the endpoints of the longest in-frame sub-segment of a
    world line.

    Replaces the ``_project_endpoints`` + ``_clip_segment_to_image`` pair
    for the detection pipeline. A long catalogue line (e.g. the full
    105 m ``near_touchline``) often has BOTH catalogue endpoints outside
    the camera's view — one off-screen, one past the camera horizon —
    yet a middle sub-segment is plainly in frame. The 2-endpoint
    projection misses that middle entirely; this densely samples the
    world segment, projects each sample with distortion, and returns
    the endpoints of the longest contiguous run of samples that fall
    in front of the camera, project to a finite sane pixel coord, and
    sit inside the image rect (with a small ``margin_px``).

    Returns ``None`` when no in-frame run is long enough to strip-search.
    """
    A = np.asarray(world_a, dtype=np.float64)
    B = np.asarray(world_b, dtype=np.float64)
    seg_len_m = float(np.linalg.norm(B - A))
    n = max(2, int(round(seg_len_m / sample_step_m)) + 1)
    ts = np.linspace(0.0, 1.0, n)
    pts = A + ts[:, None] * (B - A)
    # In-front mask: cam_z = (R @ p)[2] + t[2] = pts @ R[2] + t[2]
    cam_z = pts @ R[2] + t[2]
    in_front = cam_z > 0.1
    if not in_front.any():
        return None
    proj = project_world_to_image(K, R, t, distortion, pts[in_front])
    # Reject post-distortion grazing-degenerate projections (same guard
    # _project_endpoints uses on the 2-endpoint path).
    sane = np.isfinite(proj).all(axis=1) & (
        np.abs(proj).max(axis=1) <= _MAX_PROJ_COORD_PX
    )
    full_uv = np.full((n, 2), np.nan, dtype=np.float64)
    full_uv[in_front] = proj
    valid = np.zeros(n, dtype=bool)
    valid[in_front] = sane
    u, v = full_uv[:, 0], full_uv[:, 1]
    in_frame = (
        valid
        & (u >= -margin_px) & (u <= image_w + margin_px)
        & (v >= -margin_px) & (v <= image_h + margin_px)
    )
    if not in_frame.any():
        return None
    # Find the longest contiguous in-frame run.
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, f in enumerate(in_frame):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, n - 1))
    s, e = max(runs, key=lambda r: r[1] - r[0])
    if e - s + 1 < min_run_samples:
        return None
    pa, pb = full_uv[s], full_uv[e]
    if np.linalg.norm(pb - pa) < min_segment_px:
        return None
    return pa, pb


def _fit_line_pca(
    points: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a 2D line via PCA. Returns (centroid, unit_direction)."""
    if weights is None:
        weights = np.ones(len(points))
    w = weights / weights.sum()
    centroid = (points * w[:, None]).sum(axis=0)
    centred = points - centroid
    cov = (centred * w[:, None]).T @ centred
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Principal axis = eigenvector with largest eigenvalue
    direction = eigvecs[:, -1]
    direction = direction / np.linalg.norm(direction)
    return centroid, direction


def detect_painted_line(
    frame_bgr: np.ndarray,
    gray: np.ndarray,
    green_mask: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float],
    name: str,
    world_segment: tuple[tuple[float, float, float], tuple[float, float, float]],
    cfg: DetectorConfig,
) -> DetectedLine | None:
    """Find the actual painted line in the frame near where ``world_segment``
    is projected to under the given camera. Returns ``None`` if the line
    cannot be reliably detected (e.g. outside frame, behind camera,
    occluded, or paint not found).

    Algorithm:
      1. Project the world endpoints to image space (with distortion).
      2. Clip the projected segment to the image rectangle.
      3. Walk along the segment in ``sample_step_px`` steps, taking
         perpendicular intensity cross-sections.
      4. In each cross-section: locate the positive (grass→paint) and
         negative (paint→grass) gradient peaks via quadratic interpolation.
         The painted-line centreline is at the sub-pixel midpoint.
      5. Reject the sample if the edges aren't 1–6 px apart (typical
         painted-line width in image), if gradients are below threshold,
         or if the midpoint lands off the grass.
      6. RANSAC-fit a line through the surviving centreline samples to
         reject any outliers (e.g. a player crossing the line).
      7. Project the clipped predicted endpoints onto the fitted line.
    """
    h, w = gray.shape[:2]
    # Use the visible-sub-segment helper rather than projecting only the
    # two catalogue endpoints + Liang–Barsky clipping. Long lines like
    # the full-length near_touchline have both catalogue endpoints
    # outside the view — the 2-endpoint approach has nothing valid to
    # clip even when the line's middle is plainly in frame.
    seg = _project_visible_segment(
        K, R, t, distortion, world_segment[0], world_segment[1], w, h,
    )
    if seg is None:
        return None
    ca, cb = seg
    seg_len = float(np.linalg.norm(cb - ca))
    direction = (cb - ca) / seg_len
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)

    n_steps = max(3, int(seg_len / cfg.sample_step_px))
    ts = np.linspace(0.0, 1.0, n_steps)
    centroids: list[np.ndarray] = []
    edge_strengths: list[float] = []
    for alpha in ts:
        centre = ca + alpha * (cb - ca)
        out = _sample_centreline_offset(gray, green_mask, centre, normal, cfg)
        if out is None:
            continue
        pt, strength = out
        centroids.append(pt)
        edge_strengths.append(strength)

    if len(centroids) < 4:
        return None
    confidence = len(centroids) / float(n_steps)
    if confidence < cfg.min_confidence:
        return None

    centroids_arr = np.stack(centroids)
    weights_arr = np.array(edge_strengths)

    # RANSAC-style outlier rejection: try a few line fits on random pairs,
    # keep inliers (perpendicular dist < 1 px from fit), pick model with
    # most inliers. Final fit is least-squares on the inlier set.
    best_inliers: np.ndarray | None = None
    rng = np.random.default_rng(seed=0xC0DE)
    n = len(centroids_arr)
    for _trial in range(min(50, n * 2)):
        i, j = rng.choice(n, size=2, replace=False)
        if i == j: continue
        p1 = centroids_arr[i]; p2 = centroids_arr[j]
        d = p2 - p1; dn = float(np.linalg.norm(d))
        if dn < 1e-6: continue
        nx, ny = -d[1] / dn, d[0] / dn
        c = -(nx * p1[0] + ny * p1[1])
        distances = np.abs(centroids_arr @ np.array([nx, ny]) + c)
        inliers = distances < 1.0
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers is None or best_inliers.sum() < 3:
        return None
    inlier_pts = centroids_arr[best_inliers]
    inlier_w = weights_arr[best_inliers]
    cent, axis = _fit_line_pca(inlier_pts, inlier_w)

    def _project_onto(p: np.ndarray) -> np.ndarray:
        return cent + np.dot(p - cent, axis) * axis

    refined_a = _project_onto(ca)
    refined_b = _project_onto(cb)

    final_conf = float(best_inliers.sum() / n_steps)
    return DetectedLine(
        name=name,
        image_segment=(
            (float(refined_a[0]), float(refined_a[1])),
            (float(refined_b[0]), float(refined_b[1])),
        ),
        world_segment=world_segment,
        confidence=final_conf,
        n_samples=int(best_inliers.sum()),
    )


def detect_painted_lines_in_frame(
    frame_bgr: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    distortion: tuple[float, float],
    world_lines: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]],
    cfg: DetectorConfig | None = None,
) -> list[DetectedLine]:
    """Detect every world line in ``world_lines`` that's visible in the frame."""
    if cfg is None:
        cfg = DetectorConfig()
    gray, green_mask = _prepare_frame(frame_bgr, cfg)
    out: list[DetectedLine] = []
    for name, world_segment in world_lines.items():
        # Skip non-ground lines for v1 (goal posts, crossbars, ad boards
        # need different detection — different background, different
        # colour). Focus on z=0 painted pitch lines.
        if world_segment[0][2] != 0.0 or world_segment[1][2] != 0.0:
            continue
        det = detect_painted_line(
            frame_bgr, gray, green_mask,
            K, R, t, distortion,
            name, world_segment, cfg,
        )
        if det is not None:
            out.append(det)
    return out

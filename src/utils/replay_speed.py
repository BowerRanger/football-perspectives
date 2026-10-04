"""Replay playback speed + sync offset from players' pitch positions.

A replay of a live moment shows the same players at the same pitch positions,
only at a different playback speed (slow motion) and starting at a different
instant. Once both shots have tracks and a solved camera, every player's feet
can be projected onto the pitch, and the replay is found by searching the
time map

    live_frame = offset + rate * replay_frame

that makes the replay's players land on live players at the matching instant.
``rate`` < 1 is slow motion (0.34 = one live frame every ~3 replay frames).
No player identities are used, so the estimate does not depend on the sync it
is trying to find.

Cost = mean over replay frames of the mean distance from each replay player to
the nearest live player (truncated at ``trunc_m``). It is computed through a
precomputed (replay frame x live frame) distance table, so the exhaustive
(rate, offset) search is array indexing. Pixel-only cues (player tempo, stride
cadence) were evaluated on hand-labelled reels and rejected as primary signal
(docs/superpowers/specs/2026-10-04-replay-speed.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from scipy.spatial import cKDTree

from src.utils.camera_projection import pixel_ray

TRUNC_M = 3.0
# Pitch bounds (with margin) for accepting a projected foot point.
_PITCH_X = (-10.0, 115.0)
_PITCH_Y = (-10.0, 78.0)
# A speed ramp is reported when a continuous two-rate time map has rates
# differing by more than this factor AND lowers the cost by this fraction.
_RAMP_FACTOR = 1.2
_RAMP_MIN_GAIN = 0.04

CameraFn = Callable[[int], "tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]] | None"]


@dataclass(frozen=True)
class SpeedEstimate:
    rate: float              # live frames per replay frame (1.0 = real time)
    offset: float            # live frame shown at replay frame 0
    cost_m: float            # mean truncated player-to-player distance at the optimum
    coverage: float          # fraction of sampled replay frames that landed on live frames
    margin: float            # (profile median cost - best) / profile median
    confidence: float        # 0..1 from cost and margin
    rate_first: float        # rate before the ramp breakpoint (== rate when no ramp)
    rate_second: float       # rate after it
    ramp: bool               # a two-rate time map fits clearly better
    n_replay_frames: int

    def frame_offset_at(self, replay_frame: float) -> float:
        """Sync-map style offset (replay frame - live frame) at a replay frame."""
        return float(replay_frame - (self.offset + self.rate * replay_frame))


def feet_on_pitch(tracks_doc: dict, camera: CameraFn,
                  classes: tuple[str, ...] = ("player", "goalkeeper", "referee")) -> dict[int, np.ndarray]:
    """Per frame, the pitch-plane (x, y) of every tracked person's box bottom.

    ``camera(frame)`` returns ``(K, R, t, distortion)`` or None. Interpolated
    track frames and points that land off the pitch are skipped.
    """
    pts: dict[int, list[np.ndarray]] = {}
    for tr in tracks_doc.get("tracks", []):
        if tr.get("class_name") not in classes:
            continue
        for fr in tr.get("frames", []):
            if fr.get("interpolated"):
                continue
            cam = camera(int(fr["frame"]))
            if cam is None:
                continue
            K, R, t, dist = cam
            x0, y0, x1, y1 = fr["bbox"]
            C, d = pixel_ray(((x0 + x1) / 2.0, y1), np.asarray(K, float), np.asarray(R, float),
                             np.asarray(t, float), tuple(dist))
            if abs(d[2]) < 1e-9:
                continue
            s = -C[2] / d[2]
            if s <= 0:
                continue
            p = C + s * d
            if _PITCH_X[0] < p[0] < _PITCH_X[1] and _PITCH_Y[0] < p[1] < _PITCH_Y[1]:
                pts.setdefault(int(fr["frame"]), []).append(p[:2])
    return {f: np.asarray(v, float) for f, v in pts.items()}


def _distance_table(live: Mapping[int, np.ndarray], replay: Mapping[int, np.ndarray],
                    rep_frames: np.ndarray, trunc_m: float) -> tuple[np.ndarray, int]:
    """D[j, f - f0]: mean truncated NN distance of replay frame j's players to
    live frame f's players (NaN where live has no points)."""
    f0, f1 = min(live), max(live)
    trees = {f: cKDTree(p) for f, p in live.items() if len(p)}
    D = np.full((len(rep_frames), f1 - f0 + 1), np.nan)
    for i, j in enumerate(rep_frames):
        q = replay[int(j)]
        for f, tree in trees.items():
            d, _ = tree.query(q, k=1)
            D[i, f - f0] = float(np.mean(np.minimum(d, trunc_m)))
    return D, f0


def _costs(D: np.ndarray, f0: int, rep_frames: np.ndarray, rates: np.ndarray,
           offsets: np.ndarray, min_cover: float) -> np.ndarray:
    """Cost for every (rate, offset); inf where coverage is too low."""
    n_live = D.shape[1]
    t = offsets[None, :, None] + rates[:, None, None] * rep_frames[None, None, :] - f0
    lo = np.floor(t).astype(int)
    a = t - lo
    valid = (lo >= 0) & (lo + 1 < n_live)
    lo_c = np.clip(lo, 0, n_live - 2)
    rows = np.arange(len(rep_frames))[None, None, :]
    d0 = D[rows, lo_c]
    d1 = D[rows, lo_c + 1]
    val = (1 - a) * d0 + a * d1
    val = np.where(valid, val, np.nan)
    ok = ~np.isnan(val)
    cover = ok.mean(axis=2)
    with np.errstate(invalid="ignore"):
        mean = np.nansum(val, axis=2) / np.maximum(ok.sum(axis=2), 1)
    return np.where(cover >= min_cover, mean, np.inf)


def _search(D, f0, rep_frames, rates, n_live_frames, min_cover, offset_step=1.0):
    span_max = rates.max() * (rep_frames[-1] - rep_frames[0])
    offsets = np.arange(f0 - span_max, f0 + n_live_frames, offset_step)
    C = _costs(D, f0, rep_frames, rates, offsets, min_cover)
    best = np.unravel_index(np.argmin(C), C.shape)
    return C, offsets, best


def _refine(D, f0, rep_frames, r0, o0, n_live_frames, min_cover):
    rates = np.linspace(r0 / 1.06, r0 * 1.06, 49)
    offsets = np.arange(o0 - 4.0, o0 + 4.001, 0.25)
    C = _costs(D, f0, rep_frames, rates, offsets, min_cover)
    i, k = np.unravel_index(np.argmin(C), C.shape)
    return float(rates[i]), float(offsets[k]), float(C[i, k])


def _costs_for_times(D: np.ndarray, f0: int, T: np.ndarray, min_cover: float) -> np.ndarray:
    """Cost for each row of live times ``T`` (M, n_samples)."""
    n_live = D.shape[1]
    t = T - f0
    lo = np.floor(t).astype(int)
    a = t - lo
    valid = (lo >= 0) & (lo + 1 < n_live)
    lo_c = np.clip(lo, 0, n_live - 2)
    rows = np.arange(T.shape[1])[None, :]
    val = (1 - a) * D[rows, lo_c] + a * D[rows, lo_c + 1]
    val = np.where(valid, val, np.nan)
    ok = ~np.isnan(val)
    cover = ok.mean(axis=1)
    with np.errstate(invalid="ignore"):
        mean = np.nansum(val, axis=1) / np.maximum(ok.sum(axis=1), 1)
    return np.where(cover >= min_cover, mean, np.inf)


def _ramp_fit(D, f0, rep_frames, r0, o0, min_cover):
    """Best continuous two-rate time map (breakpoint at 30-70 % of the replay).

    Returns ``(r1, r2, cost)``; ``r1``/``r2`` are the rates before/after the
    breakpoint and the map is continuous there.
    """
    span = rep_frames[-1] - rep_frames[0]
    r_grid = np.geomspace(r0 / 2.0, r0 * 2.0, 33)
    o_grid = np.arange(o0 - 12.0, o0 + 12.001, 1.0)
    best = (np.inf, r0, r0)
    j = rep_frames - rep_frames[0]
    for frac in (0.3, 0.4, 0.5, 0.6, 0.7):
        b = frac * span
        r1, r2, o = np.meshgrid(r_grid, r_grid, o_grid, indexing="ij")
        r1, r2, o = r1.ravel(), r2.ravel(), o.ravel()
        T = np.where(j[None, :] < b,
                     o[:, None] + r1[:, None] * (j[None, :] + rep_frames[0]),
                     o[:, None] + r1[:, None] * (b + rep_frames[0]) + r2[:, None] * (j[None, :] - b))
        C = _costs_for_times(D, f0, T, min_cover)
        i = int(np.argmin(C))
        if C[i] < best[0]:
            best = (float(C[i]), float(r1[i]), float(r2[i]))
    return best[1], best[2], best[0]


def estimate_speed(
    live: Mapping[int, np.ndarray],
    replay: Mapping[int, np.ndarray],
    *,
    rate_range: tuple[float, float] = (0.08, 1.5),
    n_rates: int = 90,
    max_samples: int = 80,
    trunc_m: float = TRUNC_M,
    min_cover: float = 0.6,
    min_replay_frames: int = 12,
) -> SpeedEstimate | None:
    """Best (rate, offset) mapping the replay onto the live shot, or None when
    there is too little data to say anything."""
    live = {int(f): np.asarray(p, float) for f, p in live.items() if len(p)}
    replay = {int(f): np.asarray(p, float) for f, p in replay.items() if len(p)}
    if len(live) < min_replay_frames or len(replay) < min_replay_frames:
        return None
    frames = np.array(sorted(replay))
    step = max(1, int(np.ceil(len(frames) / max_samples)))
    rep_frames = frames[::step].astype(float)
    D, f0 = _distance_table(live, replay, rep_frames.astype(int), trunc_m)
    n_live_frames = D.shape[1]
    rates = np.geomspace(rate_range[0], rate_range[1], n_rates)
    C, offsets, (i, k) = _search(D, f0, rep_frames, rates, n_live_frames, min_cover)
    if not np.isfinite(C[i, k]):
        return None
    r, o, cost = _refine(D, f0, rep_frames, float(rates[i]), float(offsets[k]),
                         n_live_frames, min_cover)
    # Contrast of the optimum against the rate profile's typical level: a
    # true match is one deep basin; an unrelated replay gives a flat profile.
    per_rate = C.min(axis=1)
    finite = per_rate[np.isfinite(per_rate)]
    background = min(float(np.median(finite)) if len(finite) else trunc_m, trunc_m)
    margin = max(0.0, (background - cost) / background) if background > 0 else 0.0
    t = o + r * rep_frames - f0
    coverage = float(np.mean((t >= 0) & (t < n_live_frames - 1)))
    r1, r2, ramp_cost = _ramp_fit(D, f0, rep_frames, r, o, min_cover)
    ramp = bool(abs(np.log(r1 / r2)) > np.log(_RAMP_FACTOR)
                and ramp_cost < cost * (1.0 - _RAMP_MIN_GAIN))
    if not ramp:
        r1 = r2 = r
    quality = float(np.clip((trunc_m - cost) / (trunc_m - 0.8), 0.0, 1.0))
    confidence = float(np.clip(margin / 0.25, 0.0, 1.0)) * quality
    return SpeedEstimate(
        rate=r, offset=o, cost_m=cost, coverage=coverage, margin=margin,
        confidence=confidence, rate_first=r1, rate_second=r2, ramp=ramp,
        n_replay_frames=len(frames),
    )


# ---------------------------------------------------------------------------
# operator-marked moments: the camera-free path
# ---------------------------------------------------------------------------

# A ramp from marked moments needs interval rates differing by this factor
# AND a straight-line fit missing some moment by more than this many live
# frames (click error alone is about +-1 frame).
_MOMENT_RAMP_MIN_RESIDUAL = 1.5


@dataclass(frozen=True)
class MomentFit:
    rate: float                   # live frames per replay frame
    offset: float                 # live frame at replay frame 0
    residual_frames: float        # worst |live - fit| over the moments
    interval_rates: list[float]   # rate between consecutive moments
    ramp: bool
    n_moments: int


def rate_from_moments(pairs: list[tuple[float, float]]) -> MomentFit:
    """Rate + offset from operator-marked matching moments.

    ``pairs`` are ``(live_frame, replay_frame)``: the same instant marked in
    both clips (a ball contact, a net impact). Two pairs fix the time map
    exactly; more are fitted by least squares, and their interval rates
    reveal a speed ramp. Raises ``ValueError`` for fewer than two moments,
    a repeated replay frame, or moments that run backwards.
    """
    pts = sorted((float(r), float(lv)) for lv, r in pairs)
    if len(pts) < 2:
        raise ValueError("mark at least two matching moments")
    rep = np.array([p[0] for p in pts])
    live = np.array([p[1] for p in pts])
    if np.any(np.diff(rep) <= 0):
        raise ValueError("each moment needs a different replay frame")
    if np.any(np.diff(live) <= 0):
        raise ValueError("moments must run forwards in both clips")
    rate, offset = np.polyfit(rep, live, 1)
    resid = float(np.max(np.abs(live - (offset + rate * rep))))
    intervals = [float(x) for x in np.diff(live) / np.diff(rep)]
    ramp = bool(len(intervals) >= 2
                and max(intervals) / min(intervals) > _RAMP_FACTOR
                and resid > _MOMENT_RAMP_MIN_RESIDUAL)
    return MomentFit(rate=float(rate), offset=float(offset), residual_frames=resid,
                     interval_rates=intervals, ramp=ramp, n_moments=len(pts))

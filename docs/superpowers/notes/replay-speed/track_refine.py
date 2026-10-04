"""Prototype: identity-consistent (track-to-track) refinement of a coarse (rate, offset).

cost(r, o) = weighted mean over replay tracks i of min over live tracks k of
             mean_j min(|p_i(j) - q_k(o + r j)|, trunc)   (live interpolated, needs coverage)
"""
import numpy as np


def track_arrays(tracks: list[dict[int, np.ndarray]], frame_lo: int, frame_hi: int) -> np.ndarray:
    """(n_tracks, n_frames, 2) with NaN where a track has no point."""
    n = frame_hi - frame_lo + 1
    A = np.full((len(tracks), n, 2), np.nan)
    for i, rows in enumerate(tracks):
        for f, p in rows.items():
            if frame_lo <= f <= frame_hi:
                A[i, f - frame_lo] = p
    return A


def track_cost(rep_tracks, live_A, live_lo, r, o, trunc=3.0, min_pts=6):
    """rep_tracks: list of (frames array, (n,2) positions)."""
    n_live = live_A.shape[1]
    total, weight = 0.0, 0
    for fr, P in rep_tracks:
        t = o + r * fr - live_lo
        lo = np.floor(t).astype(int)
        ok = (lo >= 0) & (lo + 1 < n_live)
        if ok.sum() < min_pts:
            continue
        a = (t - lo)[ok]
        q = (1 - a)[None, :, None] * live_A[:, lo[ok]] + a[None, :, None] * live_A[:, lo[ok] + 1]
        d = np.linalg.norm(q - P[ok][None], axis=2)          # (n_live_tracks, n_pts)
        valid = ~np.isnan(d)
        cnt = valid.sum(axis=1)
        dd = np.where(valid, np.minimum(d, trunc), 0.0).sum(axis=1)
        mean = np.where(cnt >= min_pts, dd / np.maximum(cnt, 1), np.inf)
        best = mean.min()
        if not np.isfinite(best):
            best = trunc
        total += best * ok.sum()
        weight += ok.sum()
    return total / weight if weight else np.inf


def refine(rep_tracks, live_tracks, r0, o0, rate_span=1.35, off_span=30.0, n_r=55, off_step=1.0):
    los = [min(t) for t in live_tracks]
    his = [max(t) for t in live_tracks]
    lo, hi = min(los), max(his)
    A = track_arrays(live_tracks, lo, hi)
    best = (np.inf, r0, o0)
    for r in np.geomspace(r0 / rate_span, r0 * rate_span, n_r):
        for o in np.arange(o0 - off_span, o0 + off_span + 0.01, off_step):
            c = track_cost(rep_tracks, A, lo, r, o)
            if c < best[0]:
                best = (c, r, o)
    c, r, o = best
    for rr in np.linspace(r / 1.03, r * 1.03, 31):
        for oo in np.arange(o - 2, o + 2.01, 0.25):
            cc = track_cost(rep_tracks, A, lo, rr, oo)
            if cc < best[0]:
                best = (cc, rr, oo)
    return best

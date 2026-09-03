"""Strike-window high-resolution redetection.

A fast strike (shot/clearance) accelerates the ball past the standard-res
WASB detector's trained scale regime within a handful of frames: it goes
small, motion-blurred, and travels tens of px/frame. The corridor-gated
``ball_second_pass`` module fixes evidence GAPS but re-decodes at the same
scale, so it cannot recover a genuinely too-small/blurred ball — and worse,
the detector often locks onto a confident WRONG static blob near the
strike point rather than reporting nothing (see gberch f344-349: every
frame carries a `source`, but several repeat the same pixel to the fourth
decimal — a static-lock false positive, not real evidence). So the trigger
below is a plain velocity break, NOT gated on evidence being absent: a
strong break always earns a look, whether the track is silent or
confidently wrong there.

This module selects and sizes the redetection windows and does the
crop/upscale coordinate bookkeeping. Pure logic: no video access, no
torch (the stage owns I/O and the actual detector calls), matching the
``ball_second_pass`` split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Sources that mean "a real detector pass looked at this frame" — mirrors
# ball_auto_anchor.AutoAnchorCfg.event_evidence_sources and
# ball_event_resolver._HARD_EVIDENCE_SOURCES. Used only to pick the
# flanking knots for corridor prediction (see predict_corridor_centers):
# frames INSIDE a strike window are deliberately never trusted, evidenced
# or not — that is exactly what a fast strike can poison.
REAL_EVIDENCE_SOURCES: tuple[str, ...] = ("detector", "second_pass", "foot_guided")


@dataclass(frozen=True)
class StrikeWindowCfg:
    enabled: bool = True
    # Trigger: a pixel-speed jump (px/frame) at least this large.
    min_dspeed_px: float = 12.0
    # +/- frames used to estimate speed on either side of a candidate break.
    break_velocity_frames: int = 3
    # Nominal window half-width in frames around the trigger.
    window_radius_frames: int = 8
    # Hard caps bounding total redetection cost per shot.
    max_windows_per_shot: int = 3
    max_crops_per_window: int = 17
    # Corridor crop (square, ORIGINAL pixels) around the predicted centre,
    # then bicubically upscaled by upscale_factor before detection.
    # 160 * 2.0 = 320, matching ball_second_pass's proven zoom_crop_px
    # sizing (letterbox behaviour already validated at that output size) —
    # the difference is this crop covers a much smaller real-world region,
    # so the ball is proportionally larger within the detector's fixed
    # input than the plain (unscaled) second-pass zoom achieves.
    crop_px: int = 160
    upscale_factor: float = 2.0
    # Fixed-radius gate (px) around the corridor prediction — analogous to
    # foot_guided's ball_near_foot_px; strike windows have no per-frame IMM
    # covariance of their own (the corridor is a straight-line prediction,
    # not a filtered posterior).
    corridor_radius_px: float = 60.0
    # Gate fields named to match ball_second_pass.best_gated_candidate's
    # duck-typed cfg contract (corridor_sigma, accept_min) so this cfg can
    # be passed there directly.
    corridor_sigma: float = 3.0
    accept_min: float = 0.2
    candidate_min_score: float = 0.05
    top_k: int = 5
    # +/- frames used to estimate the extrapolation velocity when only one
    # flanking real-evidence knot exists.
    velocity_lookback_frames: int = 5


@dataclass(frozen=True)
class StrikeWindow:
    trigger_frame: int
    start: int
    end: int
    dspeed_px: float


@dataclass(frozen=True)
class StrikeWindowDetection:
    frame: int
    uv: tuple[float, float]
    combined_score: float


def _velocity(
    uvs: dict[int, tuple[float, float] | None],
    f: int,
    w: int,
    sign: int,
) -> np.ndarray | None:
    """Mean pixel velocity (px/frame, forward-in-time) over up to ``w``
    frames before (sign=-1) or after (sign=+1) frame ``f``. Prefers the
    longest available baseline; needs >= 2 frames of separation. Same
    idea as ``ball_auto_events._window_velocity``, reimplemented here to
    keep this module independent (it is deliberately a much simpler local
    metric — no direction-change/segmentation machinery)."""
    base = uvs.get(f)
    if base is None:
        return None
    base_arr = np.asarray(base, dtype=float)
    for off in range(w, 1, -1):
        other = uvs.get(f + sign * off)
        if other is not None:
            return (np.asarray(other, dtype=float) - base_arr) * (sign / off)
    return None


def find_strike_triggers(
    uvs: dict[int, tuple[float, float] | None],
    n_frames: int,
    cfg: StrikeWindowCfg,
) -> list[tuple[int, float]]:
    """Frames where pixel speed jumps by >= ``cfg.min_dspeed_px``,
    paired with the jump magnitude. NOT gated on nearby evidence being
    absent (see module docstring)."""
    w = cfg.break_velocity_frames
    out: list[tuple[int, float]] = []
    for f in range(n_frames):
        v_b = _velocity(uvs, f, w, -1)
        v_a = _velocity(uvs, f, w, +1)
        if v_b is None or v_a is None:
            continue
        dspeed = abs(float(np.linalg.norm(v_a)) - float(np.linalg.norm(v_b)))
        if dspeed >= cfg.min_dspeed_px:
            out.append((f, dspeed))
    return out


def select_strike_windows(
    uvs: dict[int, tuple[float, float] | None],
    n_frames: int,
    cfg: StrikeWindowCfg,
) -> list[StrikeWindow]:
    """Strongest-|dv|-first window selection, budget-capped and
    non-overlapping.

    1. Rank trigger candidates by jump magnitude, strongest first.
    2. Take up to ``cfg.max_windows_per_shot``, skipping any trigger whose
       nominal window already overlaps an accepted one — this also
       subsumes NMS for near-duplicate breaks around the same strike.
    3. Each window is ``[frame - radius, frame + radius]`` clipped to
       ``[0, n_frames - 1]``, then trimmed (centred on the trigger, still
       clipped to bounds) to ``cfg.max_crops_per_window`` frames if still
       oversized.
    """
    triggers = sorted(
        find_strike_triggers(uvs, n_frames, cfg), key=lambda t: -t[1],
    )
    windows: list[StrikeWindow] = []
    for f, dspeed in triggers:
        if len(windows) >= cfg.max_windows_per_shot:
            break
        start = max(0, f - cfg.window_radius_frames)
        end = min(n_frames - 1, f + cfg.window_radius_frames)
        if any(not (end < w.start or start > w.end) for w in windows):
            continue
        span = end - start + 1
        if span > cfg.max_crops_per_window:
            half = cfg.max_crops_per_window // 2
            start = max(0, f - half)
            end = min(n_frames - 1, start + cfg.max_crops_per_window - 1)
            start = max(0, end - cfg.max_crops_per_window + 1)
        windows.append(StrikeWindow(
            trigger_frame=f, start=start, end=end, dspeed_px=dspeed,
        ))
    windows.sort(key=lambda w: w.start)
    return windows


def _last_real(
    uvs: dict[int, tuple[float, float] | None],
    sources: dict[int, str],
    start_f: int,
    step: int,
    bound: int,
    real_sources: tuple[str, ...],
) -> tuple[int, np.ndarray] | None:
    f = start_f
    while (step > 0 and f <= bound) or (step < 0 and f >= bound):
        if sources.get(f) in real_sources and uvs.get(f) is not None:
            return f, np.asarray(uvs[f], dtype=float)
        f += step
    return None


def predict_corridor_centers(
    uvs: dict[int, tuple[float, float] | None],
    sources: dict[int, str],
    window: StrikeWindow,
    n_frames: int,
    real_sources: tuple[str, ...] = REAL_EVIDENCE_SOURCES,
    lookback: int = 5,
) -> dict[int, tuple[float, float]]:
    """Per-frame corridor centres across ``[window.start, window.end]``,
    predicted from the trajectory FLANKING the window — never from
    anything inside it, evidenced or not (that is exactly what a fast
    strike can poison; see module docstring).

    Straight-line interpolation (a 2-D pixel-space analogue of the
    solver's own two-knot primitive) when both a pre- and post-window
    real observation exist; single-sided linear extrapolation — using the
    velocity estimated over ``lookback`` frames on that side — when only
    one flanking knot exists; empty when neither does (nothing to anchor
    the crop on).
    """
    pre = _last_real(uvs, sources, window.start - 1, -1, 0, real_sources)
    post = _last_real(uvs, sources, window.end + 1, +1, n_frames - 1, real_sources)
    frames = range(window.start, window.end + 1)
    out: dict[int, tuple[float, float]] = {}
    if pre is not None and post is not None:
        pf, puv = pre
        qf, quv = post
        span = qf - pf
        for f in frames:
            t = (f - pf) / span if span else 0.0
            xy = puv + t * (quv - puv)
            out[f] = (float(xy[0]), float(xy[1]))
    elif pre is not None:
        pf, puv = pre
        v = _velocity(uvs, pf, lookback, -1)
        if v is None:
            v = np.zeros(2)
        for f in frames:
            xy = puv + v * (f - pf)
            out[f] = (float(xy[0]), float(xy[1]))
    elif post is not None:
        qf, quv = post
        v = _velocity(uvs, qf, lookback, +1)
        if v is None:
            v = np.zeros(2)
        for f in frames:
            xy = quv + v * (f - qf)
            out[f] = (float(xy[0]), float(xy[1]))
    return out


def map_upscaled_crop_candidates(
    candidates: list[tuple[float, float, float]],
    x0: int,
    y0: int,
    scale: float,
) -> list[tuple[float, float, float]]:
    """Map upscaled-crop-space candidates back to full-frame pixels:
    descale by ``scale`` then translate by the crop origin."""
    return [(x0 + u / scale, y0 + v / scale, s) for u, v, s in candidates]

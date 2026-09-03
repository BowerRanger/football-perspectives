"""Strike-window redetection: bridging fast, motion-blurred launches.

A fast strike (shot/clearance) accelerates the ball past the standard-res
WASB detector's trained scale regime within a handful of frames: it goes
small, motion-blurred, and travels tens of px/frame. The corridor-gated
``ball_second_pass`` module fixes evidence GAPS but re-decodes at the same
scale, so it cannot recover a genuinely too-small/blurred ball — and worse,
the detector often locks onto a confident WRONG static blob near the
strike point rather than reporting nothing (see gberch f344-349: every
frame carries a `source`, but several repeat the same pixel to the
thirteenth decimal — a static-lock false positive, not real evidence). So
the trigger below is a plain velocity break, NOT gated on evidence being
absent: a strong break always earns a look, whether the track is silent or
confidently wrong there.

W6 go/no-go smoke test (2026-09-03, gberch f343, real fine-tuned-v1 WASB
detector on MPS): the true ball IS visible at STANDARD resolution — a
plain full-frame low-threshold pass found a smooth, rising-confidence
(0.43->0.92) candidate track across f335-351, matching the known-good
production values at f350/351. Upscaling was never the bottleneck. What
failed was the ACCEPTANCE path: (a) the straight-line corridor's flanking
knot search can be poisoned by a static-lock false positive (see
``find_static_lock_frames``), and (b) even with correct knots, a straight
line 18 frames apart badly undershoots a real strike's curve (drifts up to
~290px), so a single-frame corridor gate either misses the true ball or
prefers a nearer decoy over it.

W6 continuation fixes both: ``find_static_lock_frames`` demotes the
static-lock signature from "real evidence" everywhere it is consulted
(corridor/chain knot selection AND, via the stage's source relabelling,
the solver/event evidence sets), and ``select_kinematic_chain`` replaces
the straight-line corridor's per-frame spatial gate with a temporal one:
low-threshold candidates are gathered FULL-FRAME (no crop — see the smoke
test above) across the window, and a chain is accepted only if it bridges
the break for enough frames at a physically plausible, non-collapsing-
confidence pace. ``predict_corridor_centers``/``map_upscaled_crop_candidates``
remain as general-purpose, independently-tested utilities (a corridor
crop is still a reasonable strategy for a distant/tiny ball where
full-frame detection would be too coarse) but are no longer the
mechanism the ball stage wires strike-window acceptance through.

Pure logic: no video access, no torch (the stage owns I/O and the actual
detector calls), matching the ``ball_second_pass`` split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Sources that mean "a real detector pass looked at this frame" — mirrors
# ball_auto_anchor.AutoAnchorCfg.event_evidence_sources and
# ball_event_resolver._HARD_EVIDENCE_SOURCES. Used to pick the flanking
# knots for corridor prediction and the kinematic chain search (see
# predict_corridor_centers / flanking_knots) and as the population
# find_static_lock_frames scans for the frozen-run signature: frames
# INSIDE a strike window are deliberately never trusted, evidenced or
# not — that is exactly what a fast strike can poison.
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

    # --- Static-lock detection (W6 continuation) -------------------------
    # A run of >= static_lock_min_run_frames consecutive real-evidence
    # frames whose positions repeat to within static_lock_eps_px is a
    # detector artifact (e.g. WASB's temporal ring buffer echoing a stale
    # heatmap peak — see gberch f344-349, bit-identical to the 13th decimal
    # while confidence climbs 0.428->0.916), not a genuinely still ball —
    # UNLESS flagged only when flanked by real motion (see
    # find_static_lock_frames): a legitimately resting ball (free-kick
    # setup, paused replay) has no fast motion on either side and is left
    # alone.
    static_lock_eps_px: float = 0.05
    static_lock_min_run_frames: int = 2
    static_lock_min_context_speed_px: float = 12.0
    static_lock_lookback_frames: int = 5

    # --- Kinematic chain search (W6 continuation) -------------------------
    # Replaces the straight-line corridor as the ACCEPTANCE gate for
    # strike-window redetection (see select_kinematic_chain): candidates
    # are gathered full-frame at a low threshold (reusing
    # candidate_min_score/top_k above — corridor drift is what defeated
    # the old crop+gate approach, not detector sensitivity), then a chain
    # bridging the break is accepted only if it clears all of:
    # chain_min_frames length, chain_min_avg_score, a physically bounded
    # per-step speed (chain_speed_slack over the max of the window's own
    # trigger jump and the flanking knots' approach/departure speed), and
    # a non-collapsing confidence trend.
    chain_min_frames: int = 5
    chain_max_gap_frames: int = 2
    # 3.0, not a tighter value: the gberch f343 smoke test found the
    # window's own (multi-frame-smoothed) dspeed_px trigger estimate
    # under-measures the TRUE single-frame peak speed right at a strike's
    # sharpest instant (a 346px/frame real step at f334->f335 vs a 136.8
    # smoothed trigger reading) -- 2.5x left a hairline-infeasible first
    # edge (342.28 needed vs 342.0 allowed) that silently steered the DP
    # onto a worse first candidate. Harmless there (an existing anchor's
    # confidence protected the frame anyway) but worth the margin.
    chain_speed_slack: float = 3.0
    chain_min_avg_score: float = 0.15
    chain_trend_tolerance: float = 0.25


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


# ---------------------------------------------------------------------------
# Static-lock detection: a run of consecutive real-evidence frames whose
# positions repeat to sub-pixel precision is a detector artifact (a stuck
# temporal buffer / stale heatmap peak), not a real observation — provided
# it is flanked by motion fast enough that a genuinely static ball couldn't
# have produced it. See gberch f344-349 in the module docstring.
# ---------------------------------------------------------------------------

def find_static_lock_frames(
    uvs: dict[int, tuple[float, float] | None],
    sources: dict[int, str],
    n_frames: int,
    cfg: StrikeWindowCfg = StrikeWindowCfg(),
    real_sources: tuple[str, ...] = REAL_EVIDENCE_SOURCES,
) -> frozenset[int]:
    """Frames belonging to a static-lock run: >= ``cfg.static_lock_min_run_frames``
    contiguous real-evidence frames whose positions all fall within
    ``cfg.static_lock_eps_px`` of the run's first position, flanked
    immediately before or after by real motion of at least
    ``cfg.static_lock_min_context_speed_px`` px/frame (estimated over
    ``cfg.static_lock_lookback_frames``) — i.e. physically implausible for
    a ball that is actually moving through this span. A genuinely resting
    ball (no fast motion on either side) is never flagged: nothing implies
    it should have moved.

    Pure geometric/velocity logic — reuses ``_velocity`` so the context
    check is consistent with ``find_strike_triggers``'s own notion of
    local speed.
    """
    eps = cfg.static_lock_eps_px
    min_run = cfg.static_lock_min_run_frames
    min_speed = cfg.static_lock_min_context_speed_px
    lookback = cfg.static_lock_lookback_frames

    runs: list[tuple[int, int]] = []
    f = 0
    while f < n_frames:
        if sources.get(f) not in real_sources or uvs.get(f) is None:
            f += 1
            continue
        base = np.asarray(uvs[f], dtype=float)
        g = f + 1
        while (
            g < n_frames
            and sources.get(g) in real_sources
            and uvs.get(g) is not None
            and float(np.linalg.norm(np.asarray(uvs[g], dtype=float) - base)) <= eps
        ):
            g += 1
        if g - f >= min_run:
            runs.append((f, g - 1))
        f = g if g > f else f + 1

    flagged: set[int] = set()
    for start, end in runs:
        v_before = _velocity(uvs, start, lookback, -1)
        v_after = _velocity(uvs, end, lookback, +1)
        fast = (
            (v_before is not None and float(np.linalg.norm(v_before)) >= min_speed)
            or (v_after is not None and float(np.linalg.norm(v_after)) >= min_speed)
        )
        if fast:
            flagged.update(range(start, end + 1))
    return frozenset(flagged)


# ---------------------------------------------------------------------------
# Flanking knots: shared building block for both the straight-line corridor
# (predict_corridor_centers, above) and the kinematic chain search, below.
# ---------------------------------------------------------------------------

Knot = tuple[int, np.ndarray, np.ndarray]  # (frame, position, local velocity)


def flanking_knots(
    uvs: dict[int, tuple[float, float] | None],
    sources: dict[int, str],
    window: StrikeWindow,
    n_frames: int,
    real_sources: tuple[str, ...] = REAL_EVIDENCE_SOURCES,
    lookback: int = 5,
) -> tuple[Knot | None, Knot | None]:
    """Last real-evidence knot before the window and first after, each as
    ``(frame, position, velocity)`` — velocity is the local approach (pre)
    or departure (post) rate estimated over ``lookback`` frames on that
    side. ``None``/``None`` when neither exists — nothing to anchor a
    prediction on."""
    pre = _last_real(uvs, sources, window.start - 1, -1, 0, real_sources)
    post = _last_real(uvs, sources, window.end + 1, +1, n_frames - 1, real_sources)
    pre_knot: Knot | None = None
    if pre is not None:
        pf, puv = pre
        v = _velocity(uvs, pf, lookback, -1)
        pre_knot = (pf, puv, v if v is not None else np.zeros(2))
    post_knot: Knot | None = None
    if post is not None:
        qf, quv = post
        v = _velocity(uvs, qf, lookback, +1)
        post_knot = (qf, quv, v if v is not None else np.zeros(2))
    return pre_knot, post_knot


# ---------------------------------------------------------------------------
# Kinematic chain search: the ACCEPTANCE gate for strike-window
# redetection. Replaces a straight-line-corridor spatial gate (which
# undershoots a real strike's curve — see module docstring) with a
# temporal/kinematic one: a low-threshold candidate is accepted only as
# part of a chain that (a) bridges the break for at least
# cfg.chain_min_frames frames, (b) never implies a physically-impossible
# per-step speed, and (c) does not have a collapsing confidence trend.
# Candidates are gathered PER FRAME (any source — full-frame, low
# threshold — the stage owns that I/O); this function is pure selection
# logic over the resulting graph.
# ---------------------------------------------------------------------------

def select_kinematic_chain(
    candidates_by_frame: dict[int, list[tuple[float, float, float]]],
    pre: Knot | None,
    post: Knot | None,
    window: StrikeWindow,
    cfg: StrikeWindowCfg = StrikeWindowCfg(),
) -> list[StrikeWindowDetection]:
    """Best kinematically-consistent chain of candidates bridging
    ``window``, or ``[]`` if none clears the acceptance bar.

    A max-total-score DP over per-frame candidates (their number is small
    — bounded by ``top_k`` per frame — so an O(n^2) DP over the whole
    window is cheap). Edges are only feasible when the implied speed is
    within ``cfg.chain_speed_slack`` of the max of: the window's own
    trigger jump, and the flanking knots' approach/departure speed (a
    strike only gets faster through the break, so this is a generous
    upper bound, not a tight one) — this is what lets a smooth,
    accelerating true-ball path beat an isolated high-score decoy that
    cannot connect to the flanking trajectory at a plausible speed.
    ``pre``/``post`` missing on one side relaxes only that side's
    boundary constraint; missing on both sides means there is nothing to
    corroborate against, so no chain is ever accepted.
    """
    if pre is None and post is None:
        return []

    base_speed = window.dspeed_px
    if pre is not None:
        base_speed = max(base_speed, float(np.linalg.norm(pre[2])))
    if post is not None:
        base_speed = max(base_speed, float(np.linalg.norm(post[2])))
    max_speed = max(base_speed, 1.0) * cfg.chain_speed_slack
    max_dt = cfg.chain_max_gap_frames + 1

    frames = sorted(
        f for f in range(window.start, window.end + 1)
        if candidates_by_frame.get(f)
    )
    if not frames:
        return []

    nodes: list[tuple[int, np.ndarray, float]] = []
    for f in frames:
        for (u, v, s) in candidates_by_frame[f]:
            nodes.append((f, np.array([u, v], dtype=float), float(s)))
    n = len(nodes)

    def _speed_ok(f1: int, p1: np.ndarray, f2: int, p2: np.ndarray) -> bool:
        dt = f2 - f1
        if dt <= 0 or dt > max_dt:
            return False
        return float(np.linalg.norm(p2 - p1)) / dt <= max_speed

    best_score = [-np.inf] * n
    best_prev = [-1] * n
    for i, (f, pos, score) in enumerate(nodes):
        if pre is None or _speed_ok(pre[0], pre[1], f, pos):
            best_score[i] = score
        for j in range(i):
            fj, posj, _sj = nodes[j]
            if fj >= f or best_score[j] == -np.inf:
                continue
            if not _speed_ok(fj, posj, f, pos):
                continue
            cand = best_score[j] + score
            if cand > best_score[i]:
                best_score[i] = cand
                best_prev[i] = j

    best_i = -1
    best_total = -np.inf
    for i, (f, pos, _s) in enumerate(nodes):
        if best_score[i] == -np.inf:
            continue
        if post is not None and not _speed_ok(f, pos, post[0], post[1]):
            continue
        if best_score[i] > best_total:
            best_total = best_score[i]
            best_i = i
    if best_i == -1:
        return []

    path_idx: list[int] = []
    i = best_i
    while i != -1:
        path_idx.append(i)
        i = best_prev[i]
    path_idx.reverse()

    if len(path_idx) < cfg.chain_min_frames:
        return []
    scores = [nodes[i][2] for i in path_idx]
    if float(np.mean(scores)) < cfg.chain_min_avg_score:
        return []
    half = len(scores) // 2
    if half > 0:
        first_half = float(np.mean(scores[:half]))
        second_half = float(np.mean(scores[half:]))
        if first_half > 0 and second_half < first_half * (1.0 - cfg.chain_trend_tolerance):
            return []

    return [
        StrikeWindowDetection(
            frame=nodes[i][0],
            uv=(float(nodes[i][1][0]), float(nodes[i][1][1])),
            combined_score=nodes[i][2],
        )
        for i in path_idx
    ]


def apply_chain_detections(
    chain: list[StrikeWindowDetection],
    static_lock: frozenset[int],
    cur_uv: dict[int, tuple[float, float]],
    raw_confidences: dict[int, float],
    sources: dict[int, str],
) -> int:
    """Merge accepted chain detections into the observation stream
    IN PLACE, honouring the "never downgrade a stronger existing
    detection" rule that second_pass/foot_guided also apply — EXCEPT for
    frames the static-lock filter has already demoted.

    A demoted frame's existing confidence is untrustworthy by
    construction: the gberch smoke test found static-lock rows (e.g.
    f346-348) whose stale confidence traces back to the SAME underlying
    detector computation as the chain's correct replacement (only the
    position differs — see the module docstring for why), so a strict
    ``<=`` comparison against that confidence would protect the
    known-wrong frozen value with an exact tie forever. Returns the
    number of frames actually replaced.
    """
    accepted = 0
    for d in chain:
        prev = raw_confidences.get(d.frame)
        if (
            d.frame not in static_lock
            and prev is not None
            and d.combined_score <= prev
        ):
            continue
        cur_uv[d.frame] = d.uv
        raw_confidences[d.frame] = d.combined_score
        sources[d.frame] = "strike_window"
        accepted += 1
    return accepted

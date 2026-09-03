"""Strike-window high-res redetection: trigger selection, corridor
prediction (flanking-trajectory, in-window-evidence-blind), coordinate
round-trip, and budget caps. Pure logic — no video, no torch."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_strike_window import (
    REAL_EVIDENCE_SOURCES,
    StrikeWindow,
    StrikeWindowCfg,
    StrikeWindowDetection,
    apply_chain_detections,
    find_static_lock_frames,
    find_strike_triggers,
    flanking_knots,
    map_upscaled_crop_candidates,
    predict_corridor_centers,
    select_kinematic_chain,
    select_strike_windows,
)


def _cfg(**kw) -> StrikeWindowCfg:
    return StrikeWindowCfg(**kw)


def _piecewise_track(
    n: int, segments: list[tuple[int, float]], y: float = 400.0,
) -> dict[int, tuple[float, float]]:
    """Contiguous 1-D track over frames [0, n): ``segments`` is a list of
    ``(start_frame, px_per_frame)`` — the slope in force from each
    start_frame (inclusive) until the next one. Position is continuous
    across slope changes (no positional teleport, a pure velocity break)."""
    uvs: dict[int, tuple[float, float]] = {}
    x = 0.0
    slope = 0.0
    seg_idx = 0
    for f in range(n):
        if seg_idx < len(segments) and f == segments[seg_idx][0]:
            slope = segments[seg_idx][1]
            seg_idx += 1
        uvs[f] = (x, y)
        x += slope
    return uvs


# ---------------------------------------------------------------------------
# Trigger detection.
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_find_strike_triggers_fires_on_hard_velocity_break():
    # Constant 2 px/frame roll up to f20, then a 40 px/frame strike.
    uvs = {f: (100.0 + 2.0 * f, 400.0) for f in range(20)}
    uvs.update({f: (140.0 + 40.0 * (f - 20), 400.0) for f in range(20, 40)})
    triggers = find_strike_triggers(uvs, n_frames=40, cfg=_cfg(min_dspeed_px=12.0))
    frames = [f for f, _ in triggers]
    assert 20 in frames


@pytest.mark.unit
def test_find_strike_triggers_silent_on_smooth_roll():
    uvs = {f: (100.0 + 5.0 * f, 400.0) for f in range(40)}
    triggers = find_strike_triggers(uvs, n_frames=40, cfg=_cfg(min_dspeed_px=12.0))
    assert triggers == []


@pytest.mark.unit
def test_find_strike_triggers_fires_even_when_evidence_present():
    """The defining fast-strike failure is a confident WRONG detection,
    not always an empty gap — the trigger must not require missing
    evidence (see gberch f343-349: every frame has a `source`, several
    are a static-lock false positive)."""
    uvs = {f: (100.0 + 2.0 * f, 400.0) for f in range(20)}
    uvs.update({f: (140.0 + 40.0 * (f - 20), 400.0) for f in range(20, 40)})
    triggers = find_strike_triggers(uvs, n_frames=40, cfg=_cfg(min_dspeed_px=12.0))
    assert any(f == 20 for f, _ in triggers)


# ---------------------------------------------------------------------------
# Window selection: strongest-first, budget-capped, non-overlapping.
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_select_strike_windows_centers_on_trigger_and_clips_to_bounds():
    uvs = {f: (100.0 + 2.0 * f, 400.0) for f in range(5)}
    uvs.update({f: (108.0 + 40.0 * (f - 5), 400.0) for f in range(5, 20)})
    cfg = _cfg(min_dspeed_px=12.0, window_radius_frames=8, max_windows_per_shot=3)
    windows = select_strike_windows(uvs, n_frames=20, cfg=cfg)
    assert len(windows) == 1
    w = windows[0]
    assert w.trigger_frame == 5
    # Clipped: 5-8 = -3 -> 0.
    assert w.start == 0
    assert w.end == min(19, 5 + 8)


@pytest.mark.unit
def test_select_strike_windows_caps_at_max_windows_per_shot():
    # Four well-separated strikes: slow roll (2 px/frame) punctuated by
    # four fast bursts (40 px/frame), each pair of transitions >> 2*radius
    # apart so their windows can never merge into one.
    n = 220
    segments = [
        (0, 2.0), (20, 40.0), (30, 2.0),
        (70, 2.0), (90, 40.0), (100, 2.0),
        (140, 2.0), (160, 40.0), (170, 2.0),
        (190, 2.0), (210, 40.0),
    ]
    uvs = _piecewise_track(n, segments)
    cfg = _cfg(min_dspeed_px=12.0, window_radius_frames=5, max_windows_per_shot=2)
    windows = select_strike_windows(uvs, n_frames=n, cfg=cfg)
    assert len(windows) == 2  # budget caps 4 real strikes down to 2


@pytest.mark.unit
def test_select_strike_windows_skips_overlapping_weaker_trigger():
    # Two break candidates 3 frames apart both clear the gate; a radius
    # wide enough to merge their windows must collapse them into one.
    n = 40
    segments = [(0, 2.0), (10, 45.0), (13, 5.0)]
    uvs = _piecewise_track(n, segments)
    cfg = _cfg(min_dspeed_px=12.0, window_radius_frames=8, max_windows_per_shot=5)
    windows = select_strike_windows(uvs, n_frames=n, cfg=cfg)
    assert len(windows) == 1
    # Windows must never overlap (general invariant, checked regardless).
    for a, b in zip(windows, windows[1:]):
        assert a.end < b.start


@pytest.mark.unit
def test_select_strike_windows_respects_max_crops_per_window():
    uvs = {f: (100.0 + 2.0 * f, 400.0) for f in range(20)}
    uvs.update({f: (140.0 + 40.0 * (f - 20), 400.0) for f in range(20, 60)})
    cfg = _cfg(
        min_dspeed_px=12.0, window_radius_frames=8, max_crops_per_window=5,
    )
    windows = select_strike_windows(uvs, n_frames=60, cfg=cfg)
    assert windows
    for w in windows:
        assert w.end - w.start + 1 <= 5


@pytest.mark.unit
def test_select_strike_windows_empty_when_no_triggers():
    uvs = {f: (100.0 + 2.0 * f, 400.0) for f in range(20)}
    assert select_strike_windows(uvs, n_frames=20, cfg=_cfg()) == []


# ---------------------------------------------------------------------------
# Corridor prediction: flanking trajectory, blind to in-window evidence.
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_corridor_interpolates_between_flanking_real_evidence():
    """Straight-line interpolation between the last real observation
    before the window and the first real observation after it — even
    when in-window values (e.g. a static-lock false positive) disagree."""
    uvs = {
        9: (100.0, 400.0),    # pre-window real evidence
        # 10..14 inside the window: a static-lock decoy that must be
        # ignored by the corridor prediction.
        10: (900.0, 900.0), 11: (900.0, 900.0), 12: (900.0, 900.0),
        13: (900.0, 900.0), 14: (900.0, 900.0),
        15: (200.0, 400.0),   # post-window real evidence
    }
    sources = {9: "detector", 10: "detector", 11: "detector", 12: "detector",
               13: "detector", 14: "detector", 15: "detector"}
    window = StrikeWindow(trigger_frame=12, start=10, end=14, dspeed_px=50.0)
    centers = predict_corridor_centers(uvs, sources, window, n_frames=16)
    assert set(centers) == {10, 11, 12, 13, 14}
    # Linear from (9, 100) to (15, 200): frame 12 is the midpoint.
    assert centers[12][0] == pytest.approx(150.0, abs=1e-6)
    assert centers[10][0] == pytest.approx(116.667, abs=1e-2)
    assert centers[14][0] == pytest.approx(183.333, abs=1e-2)
    for f in centers:
        assert centers[f][1] == pytest.approx(400.0, abs=1e-6)


@pytest.mark.unit
def test_corridor_extrapolates_forward_when_only_pre_evidence_exists():
    uvs = {f: (100.0 + 5.0 * f, 400.0) for f in range(10)}
    sources = {f: "detector" for f in range(10)}
    window = StrikeWindow(trigger_frame=10, start=10, end=14, dspeed_px=50.0)
    centers = predict_corridor_centers(uvs, sources, window, n_frames=15)
    assert set(centers) == {10, 11, 12, 13, 14}
    # Velocity ~5 px/frame extrapolated from the last real point (f=9, 145).
    assert centers[10][0] == pytest.approx(150.0, abs=1e-6)
    assert centers[14][0] == pytest.approx(170.0, abs=1e-6)


@pytest.mark.unit
def test_corridor_extrapolates_backward_when_only_post_evidence_exists():
    uvs = {f: (500.0 + 5.0 * (f - 20), 400.0) for f in range(20, 30)}
    sources = {f: "detector" for f in range(20, 30)}
    window = StrikeWindow(trigger_frame=17, start=15, end=19, dspeed_px=50.0)
    centers = predict_corridor_centers(uvs, sources, window, n_frames=30)
    assert set(centers) == {15, 16, 17, 18, 19}
    assert centers[19][0] == pytest.approx(495.0, abs=1e-6)


@pytest.mark.unit
def test_corridor_empty_when_no_flanking_evidence():
    uvs = {5: (100.0, 400.0)}
    sources = {5: "bridge"}  # not in REAL_EVIDENCE_SOURCES
    window = StrikeWindow(trigger_frame=5, start=3, end=7, dspeed_px=50.0)
    centers = predict_corridor_centers(uvs, sources, window, n_frames=10)
    assert centers == {}


@pytest.mark.unit
def test_corridor_ignores_non_real_sources_when_searching_flanks():
    uvs = {8: (900.0, 900.0), 9: (100.0, 400.0), 15: (200.0, 400.0)}
    sources = {8: "bridge", 9: "detector", 15: "second_pass"}
    window = StrikeWindow(trigger_frame=12, start=10, end=14, dspeed_px=50.0)
    centers = predict_corridor_centers(uvs, sources, window, n_frames=16)
    assert centers[12][0] == pytest.approx(150.0, abs=1e-6)


def test_real_evidence_sources_matches_documented_set():
    assert set(REAL_EVIDENCE_SOURCES) == {"detector", "second_pass", "foot_guided"}


# ---------------------------------------------------------------------------
# Coordinate round-trip: upscaled-crop candidates map back exactly.
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_map_upscaled_crop_candidates_round_trips_exactly():
    # A ball at full-frame (350.0, 220.0), crop origin (300, 180),
    # upscale 2x -> crop-local (50, 40) -> upscaled (100, 80).
    mapped = map_upscaled_crop_candidates(
        [(100.0, 80.0, 0.7)], x0=300, y0=180, scale=2.0,
    )
    assert mapped == [(350.0, 220.0, 0.7)]


@pytest.mark.unit
def test_map_upscaled_crop_candidates_non_integer_scale():
    mapped = map_upscaled_crop_candidates(
        [(60.0, 30.0, 0.5)], x0=100, y0=50, scale=1.5,
    )
    (u, v, s) = mapped[0]
    assert u == pytest.approx(140.0)
    assert v == pytest.approx(70.0)
    assert s == 0.5


@pytest.mark.unit
def test_map_upscaled_crop_candidates_empty_list():
    assert map_upscaled_crop_candidates([], x0=0, y0=0, scale=2.0) == []


# ---------------------------------------------------------------------------
# Gating reuse: StrikeWindowCfg must be duck-type compatible with
# ball_second_pass.best_gated_candidate (corridor_sigma / accept_min).
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cfg_is_gate_compatible_with_second_pass_gate():
    from src.utils.ball_second_pass import best_gated_candidate

    cfg = _cfg()
    mean, cov = np.array([500.0, 300.0]), np.eye(2) * 25.0
    decoy = (900.0, 300.0, 0.95)
    true_cand = (505.0, 302.0, 0.6)
    best = best_gated_candidate([decoy, true_cand], mean, cov, cfg)
    assert best is not None
    (u, v), combined = best
    assert (u, v) == (505.0, 302.0)


# ---------------------------------------------------------------------------
# Static-lock detection: a frozen, sub-pixel-repeated run of real-evidence
# frames flanked by fast motion is a detector artifact, not evidence.
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_find_static_lock_frames_flags_frozen_run_flanked_by_fast_motion():
    # Fast roll (30 px/frame) up to f19, frozen at f20-25 at a DIFFERENT
    # point (identical to the 1e-13 level, matching gberch's float-noise
    # signature) than the roll would have reached, then a fast resumption
    # far from the frozen point.
    uvs = {f: (100.0 + 30.0 * f, 400.0) for f in range(20)}
    frozen = (100.0 + 30.0 * 20, 400.0)
    for f in range(20, 26):
        uvs[f] = (frozen[0] + f * 1e-13, frozen[1])
    for f in range(26, 40):
        uvs[f] = (frozen[0] + 900.0 + 30.0 * (f - 26), 400.0)
    sources = {f: "detector" for f in range(40)}
    flagged = find_static_lock_frames(uvs, sources, n_frames=40, cfg=_cfg())
    assert flagged == frozenset(range(20, 26))


@pytest.mark.unit
def test_find_static_lock_frames_reproduces_gberch_f347_signature():
    """Literal reproduction of the observed gberch f344-349 static-lock
    (source="detector", uv identical to ~1e-13, confidence rising
    0.428->0.916) flanked by real fast motion at f343 and f350 — the exact
    bug behind the corridor predictor's failure (see module docstring)."""
    uvs = {
        341: (1327.82, 578.97), 342: (1349.61, 575.70),
        343: (1678.8491973876953, 761.5925636291504),
        344: (1678.8491973876953, 761.5925636291503),
        345: (1678.8491973876953, 761.5925636291502),
        346: (1678.8491973876953, 761.5925636291500),
        347: (1678.8491973876953, 761.5925636291499),
        348: (1678.8491973876953, 761.5925636291498),
        349: (1678.8491973876953, 761.5925636291497),
        350: (1338.13, 556.57), 351: (1339.61, 554.55),
    }
    sources = {f: "detector" for f in uvs}
    sources[343] = "foot_guided"
    flagged = find_static_lock_frames(uvs, sources, n_frames=352, cfg=_cfg())
    # 344-349 are the bug; 343 (the genuine foot_guided touch that seeded
    # the frozen value) may or may not be swept in depending on run
    # detection, but 347 specifically -- the frame that poisoned the
    # narrowed-window knot search -- must always be excluded.
    assert 347 in flagged
    assert {344, 345, 346, 348, 349} <= flagged


@pytest.mark.unit
def test_find_static_lock_frames_ignores_genuinely_still_ball():
    # A ball at rest (free-kick setup): identical position, but NO fast
    # motion flanking it -- must not be flagged.
    uvs = {f: (500.0, 400.0) for f in range(30)}
    sources = {f: "detector" for f in range(30)}
    flagged = find_static_lock_frames(uvs, sources, n_frames=30, cfg=_cfg())
    assert flagged == frozenset()


@pytest.mark.unit
def test_find_static_lock_frames_requires_min_run_length():
    uvs = {f: (100.0 + 30.0 * f, 400.0) for f in range(10)}
    uvs[10] = uvs[9]  # a two-frame repeat, shorter than the required run
    uvs.update({f: (uvs[9][0] + 900.0 + 30.0 * (f - 11), 400.0) for f in range(11, 20)})
    sources = {f: "detector" for f in range(20)}
    flagged = find_static_lock_frames(
        uvs, sources, n_frames=20, cfg=_cfg(static_lock_min_run_frames=3),
    )
    assert flagged == frozenset()


@pytest.mark.unit
def test_find_static_lock_frames_ignores_non_real_evidence_sources():
    uvs = {f: (100.0 + 30.0 * f, 400.0) for f in range(20)}
    frozen = uvs[19]
    for f in range(20, 26):
        uvs[f] = (frozen[0] + f * 1e-13, frozen[1])
    for f in range(26, 40):
        uvs[f] = (frozen[0] + 900.0 + 30.0 * (f - 26), 400.0)
    sources = {f: "detector" for f in range(40)}
    for f in range(20, 26):
        sources[f] = "bridge"  # synthetic gap-fill, not real evidence
    flagged = find_static_lock_frames(uvs, sources, n_frames=40, cfg=_cfg())
    assert flagged == frozenset()


@pytest.mark.unit
def test_find_static_lock_frames_empty_on_smooth_track():
    uvs = {f: (100.0 + 5.0 * f, 400.0) for f in range(40)}
    sources = {f: "detector" for f in range(40)}
    assert find_static_lock_frames(uvs, sources, n_frames=40, cfg=_cfg()) == frozenset()


@pytest.mark.unit
def test_static_lock_relabelling_lets_corridor_reach_past_it():
    """Downstream integration check: once a static-lock frame's source is
    relabelled away from a REAL_EVIDENCE_SOURCES member (the ball.py
    wiring's actual mechanism), predict_corridor_centers's flanking search
    skips it and reaches the next genuine knot -- reproducing the fix for
    the reported >800px corridor failure when a narrow window landed
    exactly on the static-lock frame."""
    uvs = {
        330: (800.0, 400.0),
        347: (1678.85, 761.59),  # the static-lock decoy
        360: (900.0, 400.0),
    }
    sources = {330: "detector", 347: "detector", 360: "detector"}
    # A narrowed window whose post-boundary search lands exactly on the
    # static-lock frame 347 (reproducing the reported failure mode when
    # window_radius_frames was narrowed).
    window = StrikeWindow(trigger_frame=345, start=340, end=346, dspeed_px=50.0)
    centers_before = predict_corridor_centers(uvs, sources, window, n_frames=361)
    # The straight line is dragged sharply toward the wrong decoy near the
    # window's end (far from the true ~800-900 trajectory band).
    assert centers_before[346][0] > 1500.0

    demoted_sources = dict(sources)
    demoted_sources[347] = "static_lock"
    centers_after = predict_corridor_centers(uvs, demoted_sources, window, n_frames=361)
    # With 347 demoted, the post-window search skips it and reaches 360
    # instead -- the corridor stays on the true ~800-900 trajectory band.
    assert 700.0 < centers_after[346][0] < 1000.0


# ---------------------------------------------------------------------------
# flanking_knots: shared pre/post real-evidence lookup with local velocity.
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_flanking_knots_returns_both_sides_with_velocity():
    uvs = {f: (100.0 + 5.0 * f, 400.0) for f in range(10)}
    uvs.update({f: (100.0 + 5.0 * f + 200.0, 400.0) for f in range(15, 25)})
    sources = {f: "detector" for f in list(range(10)) + list(range(15, 25))}
    window = StrikeWindow(trigger_frame=12, start=10, end=14, dspeed_px=50.0)
    pre, post = flanking_knots(uvs, sources, window, n_frames=25, lookback=5)
    assert pre is not None and post is not None
    assert pre[0] == 9
    assert post[0] == 15
    assert pre[2][0] == pytest.approx(5.0, abs=1e-6)
    assert post[2][0] == pytest.approx(5.0, abs=1e-6)


@pytest.mark.unit
def test_flanking_knots_none_when_no_real_evidence():
    uvs = {5: (100.0, 400.0)}
    sources = {5: "bridge"}
    window = StrikeWindow(trigger_frame=5, start=3, end=7, dspeed_px=50.0)
    pre, post = flanking_knots(uvs, sources, window, n_frames=10)
    assert pre is None and post is None


# ---------------------------------------------------------------------------
# Kinematic chain search: the ACCEPTANCE gate replacing the straight-line
# corridor. Candidates are pre-computed per frame (the stage's job); this
# tests pure selection logic.
# ---------------------------------------------------------------------------

def _knot(frame: int, uv: tuple[float, float], v: tuple[float, float]) -> tuple:
    return (frame, np.asarray(uv, dtype=float), np.asarray(v, dtype=float))


@pytest.mark.unit
def test_select_kinematic_chain_recovers_true_ball_over_decoy():
    # True ball accelerates smoothly from pre to post through the window;
    # a stationary decoy with higher per-frame score sits far away and
    # cannot reconnect to either flank at a plausible speed.
    window = StrikeWindow(trigger_frame=15, start=10, end=19, dspeed_px=40.0)
    pre = _knot(9, (100.0, 400.0), (10.0, 0.0))
    post = _knot(20, (600.0, 400.0), (60.0, 0.0))
    candidates: dict[int, list[tuple[float, float, float]]] = {}
    x = 100.0
    for i, f in enumerate(range(10, 20)):
        x += 50.0  # smooth ~50px/frame progression toward post
        candidates[f] = [
            (x, 400.0, 0.5 + 0.04 * i),          # true ball: rising confidence
            (1200.0, 900.0, 0.9),                 # decoy: high score, unreachable
        ]
    pre_knot = pre
    post_knot = post
    chain = select_kinematic_chain(
        candidates, pre_knot, post_knot, window,
        cfg=_cfg(chain_min_frames=5, chain_speed_slack=3.0, chain_min_avg_score=0.1),
    )
    assert len(chain) == 10
    frames = sorted(d.frame for d in chain)
    assert frames == list(range(10, 20))
    for d in chain:
        assert d.uv[1] == pytest.approx(400.0, abs=1e-6)
        assert d.uv[0] < 1000.0  # never picks the decoy


@pytest.mark.unit
def test_select_kinematic_chain_rejects_too_short_chain():
    window = StrikeWindow(trigger_frame=15, start=10, end=19, dspeed_px=40.0)
    pre = _knot(9, (100.0, 400.0), (10.0, 0.0))
    post = _knot(20, (600.0, 400.0), (60.0, 0.0))
    # Only two plausible, connectable frames -- below chain_min_frames.
    candidates = {
        10: [(150.0, 400.0, 0.6)],
        11: [(200.0, 400.0, 0.6)],
    }
    chain = select_kinematic_chain(
        candidates, pre, post, window,
        cfg=_cfg(chain_min_frames=5, chain_speed_slack=3.0),
    )
    assert chain == []


@pytest.mark.unit
def test_select_kinematic_chain_rejects_implausible_speed_jumps():
    window = StrikeWindow(trigger_frame=15, start=10, end=19, dspeed_px=10.0)
    pre = _knot(9, (100.0, 400.0), (5.0, 0.0))
    post = _knot(20, (200.0, 400.0), (5.0, 0.0))
    # A single candidate per frame, but it teleports impossibly far --
    # must never chain into an accepted result.
    candidates = {
        f: [(100.0 + (5000.0 if f % 2 else -5000.0), 400.0, 0.9)]
        for f in range(10, 20)
    }
    chain = select_kinematic_chain(
        candidates, pre, post, window,
        cfg=_cfg(chain_min_frames=3, chain_speed_slack=3.0, chain_min_avg_score=0.1),
    )
    assert chain == []


@pytest.mark.unit
def test_select_kinematic_chain_bridges_missing_frames():
    window = StrikeWindow(trigger_frame=15, start=10, end=19, dspeed_px=40.0)
    pre = _knot(9, (100.0, 400.0), (50.0, 0.0))
    post = _knot(20, (600.0, 400.0), (50.0, 0.0))
    # Candidates only on every other frame -- must bridge via
    # chain_max_gap_frames.
    x = 100.0
    candidates = {}
    for f in range(10, 20):
        x += 50.0
        if f % 2 == 0:
            candidates[f] = [(x, 400.0, 0.6)]
    chain = select_kinematic_chain(
        candidates, pre, post, window,
        cfg=_cfg(
            chain_min_frames=3, chain_max_gap_frames=2,
            chain_speed_slack=3.0, chain_min_avg_score=0.1,
        ),
    )
    assert len(chain) >= 3
    assert all(f % 2 == 0 for f in (d.frame for d in chain))


@pytest.mark.unit
def test_select_kinematic_chain_empty_without_any_flanking_evidence():
    window = StrikeWindow(trigger_frame=15, start=10, end=19, dspeed_px=40.0)
    candidates = {f: [(100.0 + 10.0 * f, 400.0, 0.9)] for f in range(10, 20)}
    chain = select_kinematic_chain(candidates, None, None, window, cfg=_cfg())
    assert chain == []


@pytest.mark.unit
def test_select_kinematic_chain_single_sided_pre_only():
    window = StrikeWindow(trigger_frame=15, start=10, end=19, dspeed_px=40.0)
    pre = _knot(9, (100.0, 400.0), (50.0, 0.0))
    x = 100.0
    candidates = {}
    for f in range(10, 20):
        x += 50.0
        candidates[f] = [(x, 400.0, 0.5)]
    chain = select_kinematic_chain(
        candidates, pre, None, window,
        cfg=_cfg(chain_min_frames=5, chain_speed_slack=3.0, chain_min_avg_score=0.1),
    )
    assert len(chain) == 10


@pytest.mark.unit
def test_apply_chain_detections_replaces_static_lock_frame_despite_tie():
    """Regression guard for a real bug found during the gberch f343 smoke
    test: three static-lock frames (f346-348) carried a confidence that
    EXACTLY tied the chain's replacement candidate — the frozen row's
    confidence traces back to the same underlying detector/tracker
    computation as the corrected one (only the position differs; see the
    module docstring). A strict `<=` "never downgrade" comparison would
    protect the known-wrong frozen value forever. A frame the
    static-lock filter has already demoted must not be shielded by its
    own (untrustworthy) confidence."""
    static_lock = frozenset({20})
    cur_uv = {20: (900.0, 650.0)}  # the frozen decoy value
    raw_confidences = {20: 0.811}  # the frozen row's own confidence
    sources = {20: "static_lock"}
    chain = [StrikeWindowDetection(frame=20, uv=(700.0, 500.0), combined_score=0.811)]

    accepted = apply_chain_detections(chain, static_lock, cur_uv, raw_confidences, sources)

    assert accepted == 1
    assert cur_uv[20] == (700.0, 500.0)
    assert raw_confidences[20] == 0.811
    assert sources[20] == "strike_window"


@pytest.mark.unit
def test_apply_chain_detections_still_protects_non_static_lock_tie():
    """The existing "never downgrade" rule (shared with second_pass /
    foot_guided) must be preserved for frames NOT flagged as
    static-lock: a tie or a weaker chain candidate must not overwrite
    perfectly good existing evidence."""
    static_lock: frozenset[int] = frozenset()
    cur_uv = {20: (700.0, 500.0)}
    raw_confidences = {20: 0.9}
    sources = {20: "detector"}
    chain = [StrikeWindowDetection(frame=20, uv=(701.0, 501.0), combined_score=0.9)]

    accepted = apply_chain_detections(chain, static_lock, cur_uv, raw_confidences, sources)

    assert accepted == 0
    assert cur_uv[20] == (700.0, 500.0)
    assert sources[20] == "detector"


@pytest.mark.unit
def test_apply_chain_detections_always_replaces_strictly_higher_score():
    static_lock: frozenset[int] = frozenset()
    cur_uv = {20: (700.0, 500.0)}
    raw_confidences = {20: 0.5}
    sources = {20: "detector"}
    chain = [StrikeWindowDetection(frame=20, uv=(750.0, 520.0), combined_score=0.8)]

    accepted = apply_chain_detections(chain, static_lock, cur_uv, raw_confidences, sources)

    assert accepted == 1
    assert cur_uv[20] == (750.0, 520.0)
    assert sources[20] == "strike_window"


@pytest.mark.unit
def test_apply_chain_detections_new_frame_has_no_prior_to_compare():
    static_lock: frozenset[int] = frozenset()
    cur_uv: dict[int, tuple[float, float]] = {}
    raw_confidences: dict[int, float] = {}
    sources: dict[int, str] = {}
    chain = [StrikeWindowDetection(frame=20, uv=(750.0, 520.0), combined_score=0.2)]

    accepted = apply_chain_detections(chain, static_lock, cur_uv, raw_confidences, sources)

    assert accepted == 1
    assert sources[20] == "strike_window"


@pytest.mark.unit
def test_select_kinematic_chain_rejects_collapsing_confidence_trend():
    window = StrikeWindow(trigger_frame=15, start=10, end=19, dspeed_px=40.0)
    pre = _knot(9, (100.0, 400.0), (50.0, 0.0))
    post = _knot(20, (600.0, 400.0), (50.0, 0.0))
    x = 100.0
    candidates = {}
    for i, f in enumerate(range(10, 20)):
        x += 50.0
        # Confidence collapses from 0.9 to ~0.1 across the chain.
        score = 0.9 - 0.08 * i
        candidates[f] = [(x, 400.0, score)]
    chain = select_kinematic_chain(
        candidates, pre, post, window,
        cfg=_cfg(
            chain_min_frames=5, chain_speed_slack=3.0,
            chain_min_avg_score=0.05, chain_trend_tolerance=0.25,
        ),
    )
    assert chain == []

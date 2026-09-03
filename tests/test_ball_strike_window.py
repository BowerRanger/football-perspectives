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
    find_strike_triggers,
    map_upscaled_crop_candidates,
    predict_corridor_centers,
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

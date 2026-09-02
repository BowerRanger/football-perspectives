"""Tests for per-player translation cleanup in refined_poses.

These cover the new robust per-player pass that runs before the
cross-player consensus and final smoothing:

  - ``_velocity_limit_xy``   : physical speed cap (forward/backward).
  - ``_hampel_outlier_mask`` : local-median position-outlier detection (metres).
  - ``_biphasic_pop_mask``   : local-median velocity-spike detection with a
    reversal test, catching an isolated out-and-back pop too small to
    trip the Hampel position floor.
  - ``_clean_player_translation``: densify short gaps, reject position
    outliers, reject biphasic velocity pops, clamp velocity — carrying
    root_R / thetas / confidence along the same grid.

The artifacts these target (from real hmr_world data): per-player
teleports (single-frame 10-80 m excursions), high-frequency depth
wobble (apparent 15-30 m/s), missing-frame gaps, and isolated
biphasic root-translation pops (out-and-back within 1-3 frames) too
small to trip the Hampel/velocity/acceleration gates individually.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.schemas.refined_pose import RefinedPose
from src.schemas.smpl_world import SmplWorldTrack
from src.schemas.sync_map import Alignment, GroupSync, SyncMap
from src.stages.refined_poses import (
    RefinedPosesStage,
    _accel_limit_xy,
    _biphasic_pop_mask,
    _clean_player_translation,
    _hampel_outlier_mask,
    _velocity_limit_xy,
)


def _make_track(
    *,
    player_id: str = "P001",
    shot_id: str = "play",
    frames: np.ndarray,
    root_t: np.ndarray,
    confidence: np.ndarray | None = None,
    root_R: np.ndarray | None = None,
    thetas: np.ndarray | None = None,
) -> SmplWorldTrack:
    n = root_t.shape[0]
    return SmplWorldTrack(
        player_id=player_id,
        frames=np.asarray(frames, dtype=np.int64),
        betas=np.zeros(10, dtype=np.float32),
        thetas=(thetas if thetas is not None
                else np.zeros((n, 24, 3), dtype=np.float32)),
        root_R=(root_R if root_R is not None
                else np.tile(np.eye(3), (n, 1, 1)).astype(np.float32)),
        root_t=root_t.astype(np.float32),
        confidence=(confidence.astype(np.float32) if confidence is not None
                    else np.full(n, 0.9, dtype=np.float32)),
        shot_id=shot_id,
    )


# ── _velocity_limit_xy ──────────────────────────────────────────────


@pytest.mark.unit
def test_velocity_limit_caps_single_frame_teleport() -> None:
    """A static track with one frame jumping 5 m and back: after the
    limit, no consecutive step exceeds the cap."""
    xy = np.zeros((11, 2))
    xy[5, 0] = 5.0  # teleport out and back
    out = _velocity_limit_xy(xy, max_step=0.4)
    steps = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert float(steps.max()) <= 0.4 + 1e-6


@pytest.mark.unit
def test_velocity_limit_preserves_genuine_sprint() -> None:
    """A constant 0.33 m/frame ramp (≈10 m/s at 30 fps) is below a
    0.4 m/frame cap and must pass through essentially unchanged."""
    xy = np.column_stack([np.arange(40) * 0.33, np.zeros(40)])
    out = _velocity_limit_xy(xy, max_step=0.4)
    np.testing.assert_allclose(out, xy, atol=1e-6)


# ── _accel_limit_xy ─────────────────────────────────────────────────


@pytest.mark.unit
def test_accel_limit_caps_isolated_pop() -> None:
    """A static track with one frame popping 0.4 m out and back: after
    the limit, no consecutive velocity CHANGE exceeds the cap (a pop
    this small never trips the velocity limiter's own displacement cap,
    since the per-frame step is still under it -- this is exactly the
    signature the acceleration limiter exists to catch)."""
    xy = np.zeros((21, 2))
    xy[10, 0] = 0.4  # pop out and back within a single frame
    max_dv = 40.0 / 25.0**2  # a_max_m_s2=40 at 25 fps
    out = _accel_limit_xy(xy, max_dv)
    v = np.diff(out, axis=0)
    dv = np.diff(v, axis=0)
    dv_mag = np.linalg.norm(dv, axis=1)
    assert float(dv_mag.max()) <= max_dv + 1e-9
    # the pop itself must be materially reduced, not just re-centred
    assert abs(float(out[10, 0])) < 0.15


@pytest.mark.unit
def test_accel_limit_preserves_genuine_curved_sprint() -> None:
    """A circular-arc sprint at 8 m/s with a 4.27 m turn radius has
    centripetal acceleration ~15 m/s^2 -- the spec's ceiling for a
    "genuine sprint direction change" -- well under the default 40 m/s^2
    cap, and must pass through the limiter untouched."""
    fps = 25.0
    v_speed = 8.0
    accel = 15.0
    radius = v_speed**2 / accel
    w = v_speed / radius
    n = 60
    t = np.arange(n) / fps
    xy = np.column_stack([
        radius * np.sin(w * t), radius * (1.0 - np.cos(w * t)),
    ])
    max_dv = 40.0 / fps**2
    out = _accel_limit_xy(xy, max_dv)
    # interior points only -- the fwd/bwd limiter anchors its first two
    # taps unconditionally on each pass, so edge frames trivially match
    # regardless of whether the limiter is exercised; interior frames are
    # the ones that prove nothing was clamped.
    np.testing.assert_allclose(out[3:-3], xy[3:-3], atol=1e-6)


@pytest.mark.unit
def test_accel_limit_disabled_when_max_dv_nonpositive() -> None:
    xy = np.zeros((10, 2))
    xy[5, 0] = 3.0
    out = _accel_limit_xy(xy, max_dv=0.0)
    np.testing.assert_array_equal(out, xy)


# ── _hampel_outlier_mask ────────────────────────────────────────────


@pytest.mark.unit
def test_hampel_flags_isolated_spike() -> None:
    """A lone large excursion is flagged; the surrounding clean points
    are not."""
    xy = np.zeros((21, 2))
    xy[10] = [8.0, 0.0]
    mask = _hampel_outlier_mask(xy, window=13, k=3.0, abs_floor=0.6)
    assert mask[10]
    assert not mask[np.arange(21) != 10].any()


@pytest.mark.unit
def test_hampel_preserves_genuine_drift() -> None:
    """A smooth slow drift contains no outliers — nothing flagged."""
    xy = np.column_stack([np.linspace(0, 3, 60), np.linspace(0, 1, 60)])
    mask = _hampel_outlier_mask(xy, window=13, k=3.0, abs_floor=0.6)
    assert not mask.any()


# ── _biphasic_pop_mask ──────────────────────────────────────────────
#
# These fixtures are the discriminator the design is built around: a
# pop must be flagged, but a footstrike-style single-directional
# deceleration or a curved-sprint direction change must never be —
# the reversal test (on raw velocity direction, not on deviation from
# the local median) is what tells them apart. ``window=11`` matches
# the "±5 frames" window in the design (``half = window // 2 == 5``).


@pytest.mark.unit
def test_biphasic_pop_flags_single_frame_out_and_back() -> None:
    """A static path with one frame popping 0.35 m out and immediately
    back: the displaced frame is flagged, its neighbours are not."""
    xy = np.zeros((41, 2))
    xy[20, 0] = 0.35
    mask = _biphasic_pop_mask(
        xy, window=11, k=3.0, floor_m_per_frame=0.12, max_reversal_frames=2,
    )
    assert mask[20]
    assert not mask[np.arange(41) != 20].any()


@pytest.mark.unit
def test_biphasic_pop_flags_multi_frame_pop_within_reversal_window() -> None:
    """A pop that stays displaced for 2 frames before reversing (still
    within the default ``max_reversal_frames=2``) flags both displaced
    frames, not the trusted frames on either side."""
    xy = np.zeros((41, 2))
    xy[20, 0] = 0.30
    xy[21, 0] = 0.32
    mask = _biphasic_pop_mask(
        xy, window=11, k=3.0, floor_m_per_frame=0.12, max_reversal_frames=2,
    )
    assert mask[20] and mask[21]
    assert not mask[19]
    assert not mask[22]


@pytest.mark.unit
def test_biphasic_pop_respects_max_reversal_frames() -> None:
    """A pop that stays displaced for 3 frames before reversing is NOT
    flagged when ``max_reversal_frames=2`` (the reversal falls outside
    the lookahead window) but IS flagged once the window is widened to
    3 — proving the parameter, not a hidden fixed constant, controls
    the lookahead."""
    xy = np.zeros((41, 2))
    xy[20, 0] = 0.30
    xy[21, 0] = 0.31
    xy[22, 0] = 0.33
    narrow = _biphasic_pop_mask(
        xy, window=11, k=3.0, floor_m_per_frame=0.12, max_reversal_frames=2,
    )
    wide = _biphasic_pop_mask(
        xy, window=11, k=3.0, floor_m_per_frame=0.12, max_reversal_frames=3,
    )
    assert not narrow.any()
    assert wide[20] and wide[21] and wide[22]


@pytest.mark.unit
def test_biphasic_pop_flags_asymmetric_real_pop_regression() -> None:
    """Regression fixture captured from real gberch P001 hmr_world
    output (frames 341-352, the "342-343" cluster the design doc
    reports as a surviving pop). The entering leg (341->342) is an
    unambiguous local anomaly; the exiting/reversing leg (342->343) is
    a genuine, large, opposite-direction return -- but its OWN local
    window happens to include other mildly-elevated (non-anomalous)
    real motion a few frames later, which inflates its independent
    threshold enough that requiring BOTH legs to independently clear
    the magnitude gate misses it (verified during development: the
    prior symmetric-partner construction left this specific real pop
    uncorrected). The asymmetric-partner construction (only the
    entering leg must independently clear the gate; the exiting leg
    only needs to clear the absolute floor) catches it."""
    xy = np.array([
        (11.5441, 33.1204),   # frame 341
        (10.9664, 32.3797),   # frame 342 -- pop
        (11.1288, 32.4501),   # frame 343 -- reversal back toward trend
        (11.1013, 32.5937),   # frame 344
        (10.9410, 33.0842),   # frame 345
        (10.9074, 33.2522),   # frame 346
        (10.9786, 33.2327),   # frame 347
        (11.1954, 32.8629),   # frame 348
        (11.1968, 32.9091),   # frame 349
        (11.1329, 33.0340),   # frame 350
        (11.1223, 33.0673),   # frame 351
        (11.2650, 32.8536),   # frame 352
    ])
    mask = _biphasic_pop_mask(
        xy, window=11, k=3.0, floor_m_per_frame=0.12, max_reversal_frames=2,
    )
    # position index 1 == frame 342, the pop this fixture targets.
    assert mask[1]


@pytest.mark.unit
def test_biphasic_pop_never_flags_footstrike_decel() -> None:
    """A player running east at 3.5 m/s decelerates to 0.3 m/s at frame
    20 (footstrike) and stays slow — a large, sustained, ONE-DIRECTIONAL
    change. However abruptly it happens (tested here as an instant
    single-frame step, the sharpest/most adversarial case — a real
    footstrike decelerates more gradually than this), it must never be
    flagged: velocity never reverses direction, only shrinks."""
    fps = 25.0
    n = 41
    fast = 3.5 / fps
    slow = 0.3 / fps
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] + (fast if i <= 20 else slow)
    xy = np.column_stack([x, np.full(n, 5.0)])
    mask = _biphasic_pop_mask(
        xy, window=11, k=3.0, floor_m_per_frame=0.12, max_reversal_frames=2,
    )
    assert not mask.any()


@pytest.mark.unit
def test_biphasic_pop_never_flags_curved_sprint() -> None:
    """A circular-arc sprint at 8 m/s with ~15 m/s^2 centripetal
    acceleration (the spec's "genuine direction change" ceiling) never
    reverses velocity direction frame-to-frame — never flagged."""
    fps = 25.0
    v_speed = 8.0
    accel = 15.0
    radius = v_speed**2 / accel
    w = v_speed / radius
    n = 60
    t = np.arange(n) / fps
    xy = np.column_stack([
        radius * np.sin(w * t), radius * (1.0 - np.cos(w * t)),
    ])
    mask = _biphasic_pop_mask(
        xy, window=11, k=3.0, floor_m_per_frame=0.12, max_reversal_frames=2,
    )
    assert not mask.any()


# ── _clean_player_translation ───────────────────────────────────────


@pytest.mark.unit
def test_clean_translation_disabled_is_passthrough() -> None:
    n = 20
    frames = np.arange(n)
    rt = np.column_stack([np.arange(n) * 0.1, np.zeros(n), np.full(n, 0.9)])
    tr = _make_track(frames=frames, root_t=rt)
    out, stats = _clean_player_translation(tr, fps=30.0, enabled=False)
    np.testing.assert_array_equal(out.root_t, tr.root_t)
    np.testing.assert_array_equal(out.frames, tr.frames)


@pytest.mark.unit
def test_clean_translation_fills_short_gap() -> None:
    """Frames [0,1,2,5,6] (a 2-frame gap at 3,4) become a contiguous
    grid with the gap linearly interpolated."""
    frames = np.array([0, 1, 2, 5, 6])
    # physical motion (0.1 m/frame ≈ 3 m/s at 30 fps) so the velocity
    # limit never engages and the fill stays a pure interpolation.
    rt = np.column_stack([
        frames.astype(float) * 0.1,
        np.zeros(5),
        np.full(5, 0.9),
    ])
    tr = _make_track(frames=frames, root_t=rt)
    out, stats = _clean_player_translation(
        tr, fps=30.0, enabled=True, max_gap_fill_frames=5,
    )
    np.testing.assert_array_equal(out.frames, np.arange(0, 7))
    # x == 0.1 * frame, so the filled frames 3,4 must interpolate to 0.3,0.4
    np.testing.assert_allclose(out.root_t[:, 0], np.arange(0, 7) * 0.1, atol=1e-4)
    assert stats["filled_frames"] == 2


@pytest.mark.unit
def test_clean_translation_does_not_fill_long_gap() -> None:
    """A gap longer than max_gap_fill_frames is left as a hole (the
    track is split into runs); the long gap's frames are not fabricated."""
    frames = np.array([0, 1, 2, 50, 51, 52])
    rt = np.column_stack([
        np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0]),
        np.zeros(6),
        np.full(6, 0.9),
    ])
    tr = _make_track(frames=frames, root_t=rt)
    out, stats = _clean_player_translation(
        tr, fps=30.0, enabled=True, max_gap_fill_frames=10,
    )
    # the 47-frame gap between 2 and 50 must NOT be filled
    assert 25 not in set(out.frames.tolist())
    assert out.frames.max() == 52
    assert stats["filled_frames"] == 0


@pytest.mark.unit
def test_clean_translation_removes_teleport_keeps_position() -> None:
    """A static player with one catastrophic 20 m single-frame
    excursion: after cleanup the excursion is gone and the player
    stays at the true position, with bounded per-frame speed."""
    n = 31
    frames = np.arange(n)
    rt = np.column_stack([
        np.full(n, 30.0), np.full(n, 20.0), np.full(n, 0.9),
    ])
    rt[15, 0] += 20.0  # teleport
    tr = _make_track(frames=frames, root_t=rt)
    out, stats = _clean_player_translation(
        tr, fps=30.0, enabled=True,
        hampel_window_s=0.4, hampel_k=3.0, hampel_floor_m=0.6,
        v_max_m_s=12.0,
    )
    # frame 15 back near the true (30, 20)
    np.testing.assert_allclose(out.root_t[15, :2], [30.0, 20.0], atol=0.5)
    speeds = np.linalg.norm(np.diff(out.root_t[:, :2], axis=0), axis=1) * 30.0
    assert float(speeds.max()) <= 12.0 + 1e-3
    assert stats["rejected_frames"] >= 1


@pytest.mark.unit
def test_clean_translation_accel_limit_catches_pop_below_velocity_and_hampel_gates() -> None:
    """A single-frame 0.4 m out-and-back pop is too small to trip the
    Hampel floor (0.6 m) and too brief to exceed the velocity cap
    (12 m/s / 25 fps = 0.48 m/frame per the problem statement) -- exactly
    the class of super-physical pop the acceleration limiter targets that
    the two existing passes let through untouched. The biphasic pop
    pass is disabled here (pop_max_reversal_frames=0) so this test
    isolates the ACCEL limiter's own catch, independent of the newer
    pass that -- with its defaults enabled -- would repair this exact
    pop earlier in the pipeline and leave nothing for the accel limiter
    to clamp (see test_clean_translation_pop_rejection_catches_pop_below_hampel_and_accel_gates
    for that combined-defaults behaviour)."""
    n = 41
    frames = np.arange(n)
    rt = np.column_stack([np.full(n, 10.0), np.full(n, 5.0), np.full(n, 0.9)])
    rt[20, 0] += 0.4  # pop out and back within one frame
    tr = _make_track(frames=frames, root_t=rt)
    out, stats = _clean_player_translation(
        tr, fps=25.0, enabled=True,
        hampel_window_s=0.4, hampel_k=3.0, hampel_floor_m=0.6,
        pop_max_reversal_frames=0,
        v_max_m_s=12.0, a_max_m_s2=40.0,
    )
    assert stats["rejected_frames"] == 0  # Hampel never sees it
    assert stats["pop_rejected_frames"] == 0  # pop pass disabled for this test
    assert stats["accel_clamped_frames"] >= 1
    np.testing.assert_allclose(out.root_t[20, :2], [10.0, 5.0], atol=0.15)


@pytest.mark.unit
def test_clean_translation_accel_limit_preserves_curved_sprint() -> None:
    """A curved sprint (~15 m/s^2 centripetal acceleration) must survive
    the full cleanup pipeline (Hampel + velocity + acceleration limit)
    essentially unchanged -- the acceptance criterion from the design:
    a genuine direction change is not mistaken for a pop."""
    fps = 25.0
    v_speed = 8.0
    accel = 15.0
    radius = v_speed**2 / accel
    w = v_speed / radius
    n = 60
    frames = np.arange(n)
    t = frames / fps
    xy = np.column_stack([
        radius * np.sin(w * t), radius * (1.0 - np.cos(w * t)),
    ])
    rt = np.column_stack([xy, np.full(n, 0.9)])
    tr = _make_track(frames=frames, root_t=rt)
    out, stats = _clean_player_translation(
        tr, fps=fps, enabled=True,
        hampel_window_s=0.4, hampel_k=3.0, hampel_floor_m=0.6,
        v_max_m_s=12.0, a_max_m_s2=40.0,
    )
    assert stats["accel_clamped_frames"] == 0
    np.testing.assert_allclose(out.root_t[:, :2], xy, atol=1e-6)


@pytest.mark.unit
def test_clean_translation_pop_rejection_catches_pop_below_hampel_and_accel_gates() -> None:
    """A single-frame 0.35 m out-and-back pop is too small to trip the
    Hampel floor (0.6 m) AND too small to trip the default acceleration
    cap (a_max_m_s2=75 at 25 fps) -- exactly the residual class of pop
    the biphasic pass exists to catch after the other two gates let it
    through untouched (see the design note on 2026-09-02 isolated
    biphasic root-translation pops surviving the Wave-4b limiter)."""
    n = 41
    frames = np.arange(n)
    rt = np.column_stack([np.full(n, 10.0), np.full(n, 5.0), np.full(n, 0.9)])
    rt[20, 0] += 0.35  # pop out and back within one frame
    tr = _make_track(frames=frames, root_t=rt)
    out, stats = _clean_player_translation(
        tr, fps=25.0, enabled=True,
        hampel_window_s=0.4, hampel_k=3.0, hampel_floor_m=0.6,
        pop_k=3.0, pop_floor_m_per_frame=0.12, pop_max_reversal_frames=2,
        v_max_m_s=12.0, a_max_m_s2=75.0,
    )
    assert stats["rejected_frames"] == 0       # Hampel never sees it
    assert stats["accel_clamped_frames"] == 0  # nor the accel limiter
    assert stats["pop_rejected_frames"] >= 1
    np.testing.assert_allclose(out.root_t[20, :2], [10.0, 5.0], atol=1e-6)


@pytest.mark.unit
def test_clean_translation_pop_rejection_preserves_footstrike_decel() -> None:
    """A sustained one-directional deceleration (running at 3.5 m/s,
    footstrike at frame 20 drops it to 0.3 m/s, held) must never be
    treated as a pop, however abruptly the deceleration happens --
    tested here as the sharpest case, an instant single-frame step.
    The acceleration limiter is disabled (a_max_m_s2 very large) so
    this test isolates the pop pass specifically: the default-sized
    accel cap (75 m/s^2, see config/default.yaml) legitimately nibbles
    at a step this sharp too (that is the OTHER, pre-existing gate's
    documented, deliberate trade-off -- not this one's)."""
    fps = 25.0
    n = 41
    frames = np.arange(n)
    fast, slow = 3.5 / fps, 0.3 / fps
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] + (fast if i <= 20 else slow)
    rt = np.column_stack([x, np.full(n, 5.0), np.full(n, 0.9)])
    tr = _make_track(frames=frames, root_t=rt)
    out, stats = _clean_player_translation(
        tr, fps=fps, enabled=True,
        hampel_window_s=0.4, hampel_k=3.0, hampel_floor_m=0.6,
        pop_k=3.0, pop_floor_m_per_frame=0.12, pop_max_reversal_frames=2,
        v_max_m_s=12.0, a_max_m_s2=1e6,
    )
    assert stats["pop_rejected_frames"] == 0
    np.testing.assert_allclose(out.root_t[:, 0], x, atol=1e-6)


@pytest.mark.unit
def test_clean_translation_pop_rejection_preserves_curved_sprint() -> None:
    """A curved sprint (~15 m/s^2 centripetal acceleration) must never
    be flagged by the pop pass -- the acceptance criterion from the
    design: a genuine direction change is not mistaken for a pop."""
    fps = 25.0
    v_speed = 8.0
    accel = 15.0
    radius = v_speed**2 / accel
    w = v_speed / radius
    n = 60
    frames = np.arange(n)
    t = frames / fps
    xy = np.column_stack([
        radius * np.sin(w * t), radius * (1.0 - np.cos(w * t)),
    ])
    rt = np.column_stack([xy, np.full(n, 0.9)])
    tr = _make_track(frames=frames, root_t=rt)
    out, stats = _clean_player_translation(
        tr, fps=fps, enabled=True,
        hampel_window_s=0.4, hampel_k=3.0, hampel_floor_m=0.6,
        pop_k=3.0, pop_floor_m_per_frame=0.12, pop_max_reversal_frames=2,
        v_max_m_s=12.0, a_max_m_s2=75.0,
    )
    assert stats["pop_rejected_frames"] == 0
    np.testing.assert_allclose(out.root_t[:, :2], xy, atol=1e-6)


@pytest.mark.unit
def test_clean_translation_carries_rotation_and_thetas() -> None:
    """Filled frames must produce valid rotation matrices and the
    thetas array must stay shape-consistent with the dense frames."""
    frames = np.array([0, 1, 2, 5, 6])
    n = 5
    rt = np.column_stack([np.arange(n, dtype=float), np.zeros(n), np.full(n, 0.9)])
    # distinct rotations so SLERP fill is exercised
    angles = np.linspace(0.0, 1.0, n)
    root_R = np.stack([
        np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        for a in angles
    ]).astype(np.float32)
    thetas = np.tile(np.arange(n, dtype=np.float32).reshape(n, 1, 1), (1, 24, 3))
    tr = _make_track(frames=frames, root_t=rt, root_R=root_R, thetas=thetas)
    out, _ = _clean_player_translation(
        tr, fps=30.0, enabled=True, max_gap_fill_frames=5,
    )
    assert out.root_R.shape == (7, 3, 3)
    assert out.thetas.shape == (7, 24, 3)
    # every rotation stays orthonormal (valid SO(3))
    for R in out.root_R:
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-4)


# ── Stage integration ───────────────────────────────────────────────


def _write_sync_map(output_dir: Path, *, ref: str, offsets: dict[str, int]) -> None:
    sm = SyncMap(groups=[GroupSync(
        group_id="",
        reference_shot=ref,
        alignments=[
            Alignment(shot_id=s, frame_offset=o, method="manual", confidence=1.0)
            for s, o in offsets.items()
        ],
    )])
    (output_dir / "shots").mkdir(parents=True, exist_ok=True)
    sm.save(output_dir / "shots" / "sync_map.json")


@pytest.mark.integration
def test_stage_cleanup_removes_teleport_end_to_end(tmp_path: Path) -> None:
    """A per-player teleport injected in hmr_world arrives bounded in
    the saved RefinedPose, and the summary records cleanup work."""
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="play", offsets={"play": 0})

    n = 60
    teleport_frame = 30
    for pid, base in zip(("P001", "P002", "P003", "P004"),
                         [(10.0, 20.0), (30.0, 40.0), (50.0, 5.0), (70.0, 60.0)]):
        rt = np.tile([base[0], base[1], 0.95], (n, 1))
        rt[teleport_frame, 0] += 25.0  # single-frame teleport
        SmplWorldTrack(
            player_id=pid,
            frames=np.arange(n, dtype=np.int64),
            betas=np.zeros(10, dtype=np.float32),
            thetas=np.zeros((n, 24, 3), dtype=np.float32),
            root_R=np.tile(np.eye(3), (n, 1, 1)).astype(np.float32),
            root_t=rt.astype(np.float32),
            confidence=np.full(n, 0.9, dtype=np.float32),
            shot_id="play",
        ).save(output_dir / "hmr_world" / f"play__{pid}_smpl_world.npz")

    cfg = {
        "refined_poses": {
            "ground_snap_max_distance": 0.0,
            "cleanup": {"enabled": True, "v_max_m_s": 12.0},
        }
    }
    RefinedPosesStage(config=cfg, output_dir=output_dir).run()

    for pid in ("P001", "P002", "P003", "P004"):
        rp = RefinedPose.load(output_dir / "refined_poses" / f"{pid}_refined.npz")
        speeds = np.linalg.norm(np.diff(rp.root_t[:, :2], axis=0), axis=1) * 30.0
        assert float(speeds.max()) <= 12.0 + 0.5, (
            f"{pid} still has a >12 m/s step after cleanup"
        )

    summary = json.loads(
        (output_dir / "refined_poses" / "refined_poses_summary.json").read_text()
    )
    assert "cleanup" in summary
    assert summary["cleanup"]["rejected_frames"] >= 1


@pytest.mark.integration
def test_stage_cleanup_bounds_multishot_switch_teleports(tmp_path: Path) -> None:
    """A player seen in two synced shots whose camera solves disagree
    by 3 m: the highest-confidence pick alternates between the two
    positions frame-to-frame, which would teleport the player. The
    post-assembly cleanup must bound the assembled track's per-frame
    speed to the physical cap."""
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="a", offsets={"a": 0, "b": 0})

    n = 40
    frames = np.arange(n, dtype=np.int64)
    # shot a places P001 at x=10; shot b at x=13 (a 3 m disagreement).
    # Confidence alternates so the per-frame highest-conf pick flips
    # between the two shots every frame.
    conf_a = np.where(frames % 2 == 0, 0.9, 0.4).astype(np.float32)
    conf_b = np.where(frames % 2 == 0, 0.4, 0.9).astype(np.float32)
    for shot, x, conf in (("a", 10.0, conf_a), ("b", 13.0, conf_b)):
        SmplWorldTrack(
            player_id="P001",
            frames=frames,
            betas=np.zeros(10, dtype=np.float32),
            thetas=np.zeros((n, 24, 3), dtype=np.float32),
            root_R=np.tile(np.eye(3), (n, 1, 1)).astype(np.float32),
            root_t=np.tile([x, 10.0, 0.95], (n, 1)).astype(np.float32),
            confidence=conf,
            shot_id=shot,
        ).save(output_dir / "hmr_world" / f"{shot}__P001_smpl_world.npz")

    cfg = {
        "refined_poses": {
            "ground_snap_max_distance": 0.0,
            "cleanup": {"enabled": True, "v_max_m_s": 12.0},
        }
    }
    RefinedPosesStage(config=cfg, output_dir=output_dir).run()

    rp = RefinedPose.load(output_dir / "refined_poses" / "P001_refined.npz")
    assert set(rp.contributing_shots) == {"a", "b"}
    speeds = np.linalg.norm(np.diff(rp.root_t[:, :2], axis=0), axis=1) * 30.0
    assert float(speeds.max()) <= 12.0 + 0.5, (
        "multi-shot switch teleports were not bounded"
    )


@pytest.mark.integration
def test_stage_cleanup_pop_rejection_recorded_in_summary(tmp_path: Path) -> None:
    """A single-frame biphasic pop too small to trip the Hampel/accel
    gates (default a_max_m_s2=75) still arrives corrected in the saved
    RefinedPose, and the stage summary records ``pop_rejected_frames``
    end-to-end (config -> cleanup_kwargs -> cleanup_summary -> JSON)."""
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="play", offsets={"play": 0})

    n = 60
    pop_frame = 30
    rt = np.tile([10.0, 20.0, 0.95], (n, 1))
    rt[pop_frame, 0] += 0.35  # sub-Hampel-floor, sub-accel-cap pop
    SmplWorldTrack(
        player_id="P001",
        frames=np.arange(n, dtype=np.int64),
        betas=np.zeros(10, dtype=np.float32),
        thetas=np.zeros((n, 24, 3), dtype=np.float32),
        root_R=np.tile(np.eye(3), (n, 1, 1)).astype(np.float32),
        root_t=rt.astype(np.float32),
        confidence=np.full(n, 0.9, dtype=np.float32),
        shot_id="play",
    ).save(output_dir / "hmr_world" / "play__P001_smpl_world.npz")

    cfg = {
        "refined_poses": {
            "ground_snap_max_distance": 0.0,
            "cleanup": {"enabled": True},
        }
    }
    RefinedPosesStage(config=cfg, output_dir=output_dir).run()

    rp = RefinedPose.load(output_dir / "refined_poses" / "P001_refined.npz")
    idx = int(np.where(rp.frames == pop_frame)[0][0])
    np.testing.assert_allclose(rp.root_t[idx, :2], [10.0, 20.0], atol=0.1)

    summary = json.loads(
        (output_dir / "refined_poses" / "refined_poses_summary.json").read_text()
    )
    assert summary["cleanup"]["pop_rejected_frames"] >= 1
    assert summary["cleanup"]["rejected_frames"] == 0
    assert summary["cleanup"]["accel_clamped_frames"] == 0

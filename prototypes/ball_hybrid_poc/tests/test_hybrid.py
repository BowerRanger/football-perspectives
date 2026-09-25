"""Tests for the hybrid ball extractor (Task B / SPIKE).

Three groups:
  1. ``hybrid_physics`` — shooting recovers a drag arc from two knots.
  2. ``blend`` — no jitter spikes (third difference), smooth decay away
     from evidence, exact at evidence.
  3. ``hybrid.run_hybrid`` end-to-end on a synthetic clip built entirely
     inside this file (a minimal fake ``ClipContext`` with a pinhole
     camera) — manual anchors never move, z >= r everywhere, and
     hybrid-with-drag beats hybrid-without-drag (cfg cd=0) mid-flight
     for a trajectory that actually has drag.

This file must NOT import anything from A1's truth_sim/truth_builder/
synth_detector modules — the synthetic scenario below is built from
scratch, independent of that truth harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from ..blend import blend_deltas, exponential_kernel, segment_frames
from ..hybrid import DEFAULT_CFG, resolve_knots, run_hybrid
from ..hybrid_physics import (
    BALL_RADIUS_M,
    CD_DEFAULT,
    bounce_velocity,
    fit_roll_segment,
    shoot_arc,
    simulate,
)
from ..types import Observation

BALL_R = BALL_RADIUS_M


# ---------------------------------------------------------------------------
# 1. hybrid_physics
# ---------------------------------------------------------------------------

def test_shoot_arc_recovers_gravity_only_analytically():
    p_a = np.array([0.0, 0.0, 0.11])
    p_b = np.array([10.0, 1.0, 0.11])
    v0 = shoot_arc(p_a, 0.0, p_b, 1.0, cd=0.0)
    p_end = simulate(p_a, v0, [1.0], cd=0.0)[0]
    assert np.linalg.norm(p_end - p_b) < 1e-6


def test_shoot_arc_recovers_drag_arc_from_two_noisy_knots():
    """Ground truth: a drag arc (cd=0.3) from a known launch. Take the two
    endpoints as hard knots (perturbed by a few cm, simulating real click/
    triangulation resolution error, hence 'noisy'), shoot_arc a drag arc
    between them, and check every intermediate sample stays within 20 cm
    of the true trajectory."""
    rng = np.random.default_rng(7)
    true_cd = 0.30
    p0 = np.array([0.0, 0.0, 1.0])
    v0_true = np.array([18.0, 4.0, 9.0])
    T = 1.8
    times = np.linspace(0.0, T, 13)
    true_traj = simulate(p0, v0_true, times, cd=true_cd)

    knot_noise = 0.03  # 3 cm
    p_a = true_traj[0] + rng.normal(scale=knot_noise, size=3)
    p_b = true_traj[-1] + rng.normal(scale=knot_noise, size=3)

    v0_hat = shoot_arc(p_a, 0.0, p_b, T, cd=true_cd)
    recon = simulate(p_a, v0_hat, times, cd=true_cd)

    err = np.linalg.norm(recon - true_traj, axis=1)
    assert np.all(err < 0.20), f"max error {err.max():.3f} m exceeds 20cm"


def test_fit_roll_segment_endpoint_exact():
    roll = fit_roll_segment((0.0, 0.0), (10.0, 2.0), duration_s=3.0,
                             obs=[(1.0, np.array([3.2, 0.7])),
                                  (2.0, np.array([6.9, 1.4]))])
    p0 = roll.eval([0.0], z=BALL_R)[0]
    pT = roll.eval([3.0], z=BALL_R)[0]
    assert np.allclose(p0[:2], [0.0, 0.0], atol=1e-9)
    assert np.allclose(pT[:2], [10.0, 2.0], atol=1e-9)
    assert p0[2] == pytest.approx(BALL_R)


def test_fit_roll_segment_clamps_to_friction_bound():
    # Endpoints that would require an absurd deceleration; the fit must
    # clamp to the mu_max*g envelope rather than blow up.
    roll = fit_roll_segment((0.0, 0.0), (1.0, 0.0), duration_s=0.1,
                             obs=[(0.05, np.array([50.0, 0.0]))],
                             mu_max=0.9, g=9.81)
    accel_mag = float(np.linalg.norm(roll.accel_xy))
    assert accel_mag <= 0.9 * 9.81 + 1e-6


def test_bounce_velocity_flips_vertical_scales_by_restitution():
    v_in = np.array([5.0, 0.0, -8.0])
    v_out = bounce_velocity(v_in, restitution_e=0.6)
    assert v_out[2] == pytest.approx(0.6 * 8.0)
    assert v_out[0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 2. blend
# ---------------------------------------------------------------------------

def test_exponential_kernel_decays_and_halves():
    assert exponential_kernel(0.0, 5.0) == 1.0
    assert exponential_kernel(5.0, 5.0) == pytest.approx(0.5)
    assert exponential_kernel(10.0, 5.0) == pytest.approx(0.25)


def test_segment_frames_splits_at_events():
    segs = segment_frames(list(range(0, 10)), event_frames=[3, 6])
    assert segs == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]


def test_blend_deltas_exact_at_evidence_and_decays_away():
    frames = list(range(0, 61))
    evidence = {30: ((1.0, 0.0, 0.0), 1.0)}
    out = blend_deltas(frames, evidence, halflife_frames=5.0)
    d30, conf30 = out[30]
    assert conf30 == pytest.approx(1.0)
    assert d30[0] == pytest.approx(1.0, abs=1e-6)
    # far away (6 halflives, the default window edge) it must have decayed
    # to a small fraction of the full-strength value at the evidence frame.
    d_far, conf_far = out[0]
    assert conf_far < 0.02
    assert abs(d_far[0]) < 0.02


def test_blend_deltas_never_crosses_an_event():
    frames = list(range(0, 21))
    evidence = {5: ((1.0, 0.0, 0.0), 1.0)}
    out_with_event = blend_deltas(frames, evidence, halflife_frames=50.0,
                                   event_frames=[10])
    out_no_event = blend_deltas(frames, evidence, halflife_frames=50.0)
    # With a huge halflife but a hard event at frame 10, frame 15 must see
    # zero contribution from the evidence at frame 5 (different segment).
    assert out_with_event[15][1] == 0.0
    # Without the event wall the same evidence leaks across (positive conf).
    assert out_no_event[15][1] > 0.0


def test_blend_has_no_jitter_spikes_on_noisy_input():
    """Feed per-frame deltas from independent per-frame noise (like a
    noisy detector) through blend_deltas and check the smoothed result's
    third difference has no spikes: its max magnitude must be much
    smaller than the raw (unsmoothed) noise's third-difference spikes."""
    rng = np.random.default_rng(3)
    n = 200
    frames = list(range(n))
    noise = rng.normal(scale=0.05, size=n)  # 5 cm iid per-frame noise
    evidence = {f: ((float(noise[f]), 0.0, 0.0), 1.0) for f in frames}
    out = blend_deltas(frames, evidence, halflife_frames=6.0)
    smoothed = np.array([out[f][0][0] for f in frames])

    def third_diff(x):
        return np.diff(x, n=3)

    raw_td = third_diff(noise)
    smoothed_td = third_diff(smoothed)
    assert np.max(np.abs(smoothed_td)) < 0.3 * np.max(np.abs(raw_td))


# ---------------------------------------------------------------------------
# 3. run_hybrid end-to-end on a synthetic clip built in this file
# ---------------------------------------------------------------------------

@dataclass
class _FakeAnchor:
    frame: int
    image_xy: tuple[float, float] | None
    state: str
    player_id: str | None = None
    bone: str | None = None
    end_frame: int | None = None


@dataclass
class _FakeFix:
    frame: int
    xyz: tuple[float, float, float]


def _pinhole_K(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


@dataclass
class _FakeClipContext:
    """Minimal stand-in for ``ctx.ClipContext``: a static overhead-ish
    pinhole camera at a fixed world position looking down the +Y axis,
    used only by this test file so ``hybrid.py`` is exercised through
    its real public interface (``.project``/``.ray``/``.per_frame_*``)
    without touching any real clip's data."""

    clip_id: str
    fps: float
    frames: tuple[int, ...]
    K: np.ndarray
    R: np.ndarray
    C: np.ndarray  # camera centre, world
    distortion: tuple[float, float] = (0.0, 0.0)
    per_frame_K: dict = field(default_factory=dict)
    per_frame_R: dict = field(default_factory=dict)
    per_frame_t: dict = field(default_factory=dict)

    def __post_init__(self):
        t = -self.R @ self.C
        for f in self.frames:
            self.per_frame_K[f] = self.K
            self.per_frame_R[f] = self.R
            self.per_frame_t[f] = t

    def project(self, frame, xyz):
        from src.utils.camera_projection import project_world_to_image
        pts = np.asarray(xyz, dtype=np.float64)
        single = pts.ndim == 1
        K, R, t = self.K, self.R, self.per_frame_t[frame]
        out = project_world_to_image(K, R, t, self.distortion, pts.reshape(-1, 3))
        return out[0] if single else out

    def ray(self, frame, uv):
        from src.utils.ball_eval import pixel_ray
        return pixel_ray(uv, self.K, self.R, self.per_frame_t[frame], self.distortion)


def _make_camera(n_frames=90, fps=30.0):
    """Side-on camera ~25 m back, slightly elevated, looking at the
    action so a ~20 m flight stays comfortably in frame."""
    K = _pinhole_K()
    # Camera looks along +X (world) toward the play; R maps world->camera.
    # Choose camera basis: cam_x = world -Y, cam_y = world -Z, cam_z = world +X
    R = np.array([
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ])
    C = np.array([-25.0, 5.0, 8.0])
    frames = tuple(range(n_frames))
    return _FakeClipContext(clip_id="synthtest", fps=fps, frames=frames, K=K, R=R, C=C)


def _find_landing_time(p0, v0, cd, t_max=5.0, n=4000):
    """Time at which the ball's z first returns to BALL_R (descending),
    found by dense sampling + linear interpolation between the bracketing
    samples. Used to build a physically-consistent 'grounded' landing
    knot for the synthetic scenario (rather than just picking an
    arbitrary duration and mislabelling a mid-air point as grounded)."""
    times = np.linspace(0.0, t_max, n)
    traj = simulate(p0, v0, times, cd=cd)
    z = traj[:, 2]
    below = np.where(z <= BALL_R)[0]
    below = below[below > 0]
    if len(below) == 0:
        raise RuntimeError("synthetic scenario: ball never lands within t_max")
    i = int(below[0])
    t0, t1 = times[i - 1], times[i]
    z0, z1 = z[i - 1], z[i]
    frac = (BALL_R - z0) / (z1 - z0)
    return float(t0 + frac * (t1 - t0))


def _synthetic_drag_kick_scenario(fps=30.0, cd=0.30):
    """A single kick: ball launched from a player's foot, flies under
    gravity+drag, lands at a grounded manual anchor when it actually
    returns to ground level. Manual anchors at both ends; a scatter of
    confident 'detector' observations along the flight (each projected
    from the true trajectory plus a little pixel noise)."""
    ctx = _make_camera()
    p_a = np.array([0.0, 0.0, BALL_R])
    v0_true = np.array([13.0, 0.0, 8.5])
    duration_s = _find_landing_time(p_a, v0_true, cd)
    frame_a = 10
    frame_b = frame_a + int(round(duration_s * fps))
    duration_s = (frame_b - frame_a) / fps  # re-snap to integer-frame duration
    p_b = simulate(p_a, v0_true, [duration_s], cd=cd)[0]
    p_b[2] = BALL_R  # landing is ground-exact by construction

    anchors = [
        _FakeAnchor(frame=frame_a, image_xy=tuple(ctx.project(frame_a, p_a)),
                    state="kick", player_id="P001", bone="right_foot"),
        _FakeAnchor(frame=frame_b, image_xy=tuple(ctx.project(frame_b, p_b)),
                    state="grounded"),
    ]

    rng = np.random.default_rng(11)
    obs = []
    for frac in np.linspace(0.15, 0.85, 8):
        t_s = frac * duration_s
        frame = int(round(frame_a + t_s * fps))
        p_true = simulate(p_a, v0_true, [t_s], cd=cd)[0]
        uv = ctx.project(frame, p_true) + rng.normal(scale=1.0, size=2)
        obs.append(Observation(frame=frame, uv=(float(uv[0]), float(uv[1])),
                                conf=0.9, source="detector"))

    return ctx, anchors, obs, p_a, p_b, v0_true, cd, frame_a, frame_b, duration_s


def test_resolve_knots_splits_hard_vs_ray():
    ctx, anchors, obs, *_ = _synthetic_drag_kick_scenario()
    anchors = anchors + [_FakeAnchor(frame=50, image_xy=(500.0, 500.0), state="airborne_mid")]
    hard, rays = resolve_knots(ctx, anchors)
    assert len(hard) == 2
    assert {k.state for k in hard} == {"kick", "grounded"}
    assert len(rays) == 1
    assert rays[0].state == "airborne_mid"


def test_run_hybrid_manual_anchors_never_move():
    ctx, anchors, obs, p_a, p_b, v0_true, cd, frame_a, frame_b, _T = _synthetic_drag_kick_scenario()
    track = run_hybrid(ctx, obs, anchors, cfg={"cd": cd, "fit_cd": True})
    by_frame = {tf.frame: tf for tf in track.frames}
    assert by_frame[frame_a].mode == "anchor"
    assert by_frame[frame_b].mode == "anchor"
    assert np.allclose(by_frame[frame_a].xyz, p_a, atol=1e-6)
    assert np.allclose(by_frame[frame_b].xyz, p_b, atol=1e-6)


def test_run_hybrid_z_never_below_radius():
    ctx, anchors, obs, *_ = _synthetic_drag_kick_scenario()
    track = run_hybrid(ctx, obs, anchors, cfg={"cd": 0.30})
    for tf in track.frames:
        assert tf.xyz is not None
        assert tf.xyz[2] >= BALL_RADIUS_M - 1e-9


def test_run_hybrid_with_drag_beats_no_drag_mid_flight():
    """The core hypothesis test: a trajectory that genuinely has drag
    (cd=0.30) is reconstructed more accurately mid-flight by hybrid with
    drag enabled than by the cd=0 ablation, when scored against the true
    (drag) trajectory at the midpoint."""
    ctx, anchors, obs, p_a, p_b, v0_true, cd, frame_a, frame_b, T = (
        _synthetic_drag_kick_scenario(cd=0.30))

    mid_t = T / 2.0
    mid_frame = int(round(frame_a + mid_t * ctx.fps))
    true_mid = simulate(p_a, v0_true, [mid_t], cd=cd)[0]

    track_drag = run_hybrid(ctx, obs, anchors,
                             cfg={"cd": 0.30, "fit_cd": False})
    track_nodrag = run_hybrid(ctx, obs, anchors,
                               cfg={"cd": 0.0, "fit_cd": False})

    by_frame_drag = {tf.frame: tf for tf in track_drag.frames}
    by_frame_nodrag = {tf.frame: tf for tf in track_nodrag.frames}

    err_drag = np.linalg.norm(np.array(by_frame_drag[mid_frame].xyz) - true_mid)
    err_nodrag = np.linalg.norm(np.array(by_frame_nodrag[mid_frame].xyz) - true_mid)

    assert err_drag < err_nodrag, (
        f"drag model error {err_drag:.3f}m should beat no-drag {err_nodrag:.3f}m")
    assert err_drag < 0.20


def test_run_hybrid_faithful_mode_near_confident_observations():
    ctx, anchors, obs, *_ = _synthetic_drag_kick_scenario()
    track = run_hybrid(ctx, obs, anchors, cfg={"cd": 0.30, "fit_cd": True})
    by_frame = {tf.frame: tf for tf in track.frames}
    # At least one observation frame should land in "faithful" mode (high
    # blend confidence) since the synthetic detections are low-noise/high-
    # confidence and gated as inliers.
    obs_frames = {o.frame for o in obs}
    faithful_near_obs = [f for f in obs_frames
                          if f in by_frame and by_frame[f].mode == "faithful"]
    assert faithful_near_obs, "expected at least one faithful-mode frame near evidence"


def test_run_hybrid_empty_inputs_returns_empty_track():
    ctx = _make_camera()
    track = run_hybrid(ctx, [], [])
    assert track.frames == ()

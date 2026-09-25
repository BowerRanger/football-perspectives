"""A1 spike tests: the independent physics simulator, and the synthetic
truth/detector built from it, for every clip that's present locally."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from ..ctx import CLIPS, load_clip
from ..synth_detector import make_synth_run
from ..truth_builder import build_truth
from .. import truth_sim as ts

_POC_DIR = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# Physics unit tests (truth_sim.py alone)
# --------------------------------------------------------------------------


def test_drag_deceleration_30ms():
    """A 30 m/s ball with Cd 0.25 decelerates 12 +/- 1.5 m/s^2 initially."""
    v = np.array([30.0, 0.0, 0.0])
    a = ts.drag_accel(v, ts.DragParams(cd_const=0.25, crisis=False))
    decel = float(np.linalg.norm(a))
    assert 10.5 <= decel <= 13.5, decel


def test_drag_crisis_transition_monotonic_and_bounded():
    p = ts.DragParams(crisis=True, cd_low=0.45, cd_high=0.20,
                      v_low=10.0, v_high=20.0)
    assert ts.drag_coefficient(5.0, p) == pytest.approx(0.45)
    assert ts.drag_coefficient(25.0, p) == pytest.approx(0.20)
    mid = ts.drag_coefficient(15.0, p)
    assert 0.20 < mid < 0.45


def test_bounce_loses_energy():
    """Restitution < 1 must reduce kinetic energy — the whole point of a
    bounce map. Also checks the ball doesn't reverse its horizontal
    direction on a routine (non-backspin) bounce."""
    vel = np.array([4.0, 1.0, -8.0])
    omega = np.array([0.0, 5.0, 0.0])  # mild topspin
    ke_before = 0.5 * ts.BALL_MASS_KG * float(np.dot(vel, vel))
    vel2, omega2 = ts.apply_bounce(vel, omega, ts.BounceParams())
    ke_after = 0.5 * ts.BALL_MASS_KG * float(np.dot(vel2, vel2))
    assert ke_after < ke_before
    assert vel2[2] > 0, "ball must bounce upward after a downward approach"
    assert np.all(np.isfinite(omega2))


def test_bounce_restitution_controls_energy_loss():
    vel = np.array([2.0, 0.0, -6.0])
    omega = np.zeros(3)
    _v_bouncy, _ = ts.apply_bounce(vel, omega, ts.BounceParams(e_n=0.75))
    _v_dead, _ = ts.apply_bounce(vel, omega, ts.BounceParams(e_n=0.60))
    assert abs(_v_bouncy[2]) > abs(_v_dead[2])


def test_roll_distance_matches_solved_initial_speed():
    p = ts.RollParams()
    for target_dist in (0.5, 5.0, 20.0):
        for duration in (0.3, 1.0, 3.0):
            v0 = ts.solve_roll_initial_speed(target_dist, duration, p)
            got = ts.roll_distance_at(v0, duration, p)
            assert got == pytest.approx(target_dist, abs=1e-3), (
                target_dist, duration, v0, got)


def test_roll_distance_zero_target_gives_zero_speed():
    assert ts.solve_roll_initial_speed(0.0, 1.0, ts.RollParams()) == 0.0


def test_simulate_flight_no_drag_no_spin_matches_projectile():
    """Sanity: with drag/lift forced to (near) zero, the arc should match
    a textbook projectile within numerical tolerance."""
    drag = ts.DragParams(cd_const=1e-9, crisis=False)
    spin = ts.SpinParams(omega0=np.zeros(3))
    v0 = np.array([10.0, 0.0, 8.0])
    duration = 1.0
    res = ts.simulate_flight(np.zeros(3), v0, duration, drag, spin)
    pos, vel = res.state_at(duration)
    expected_x = v0[0] * duration
    expected_z = v0[2] * duration - 0.5 * ts.G * duration ** 2
    assert pos[0] == pytest.approx(expected_x, abs=0.05)
    assert pos[2] == pytest.approx(expected_z, abs=0.05)
    assert vel[2] == pytest.approx(v0[2] - ts.G * duration, abs=0.05)


# --------------------------------------------------------------------------
# Independence / isolation (grep-based, per CONTRACT.md)
# --------------------------------------------------------------------------

_FORBIDDEN_IMPORT_SUBSTRINGS = (
    "ball_physics", "ball_piecewise_solver", "hybrid",
)
# Any src.utils.ball_* module is disallowed for truth_sim.py EXCEPT it
# simply must not import from src at all — truth_sim is meant to be a
# self-contained numpy/scipy simulator.


def test_truth_sim_has_no_src_or_hybrid_imports():
    tree = ast.parse((_POC_DIR / "truth_sim.py").read_text())
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
            # relative imports (node.module is None with node.level>0)
            elif node.level:
                modules.append("." * node.level)
    for mod in modules:
        assert not mod.startswith("src."), (
            f"truth_sim.py must be independent physics; found import {mod!r}")
        for bad in _FORBIDDEN_IMPORT_SUBSTRINGS:
            assert bad not in mod, (
                f"truth_sim.py imported {mod!r} containing forbidden "
                f"substring {bad!r}")
    allowed_prefixes = ("numpy", "scipy", "__future__", "dataclasses")
    for mod in modules:
        assert mod.startswith(allowed_prefixes), (
            f"truth_sim.py has an unexpected import: {mod!r}")


_FORBIDDEN_OUTPUT_STRINGS = ("ball_track", "anchors_auto", "ball_keyframes")


@pytest.mark.parametrize("filename", ["truth_sim.py", "truth_builder.py",
                                      "synth_detector.py"])
def test_no_ball_stage_output_filenames_referenced(filename):
    text = (_POC_DIR / filename).read_text()
    for bad in _FORBIDDEN_OUTPUT_STRINGS:
        assert bad not in text, (
            f"{filename} references forbidden ball-stage-output string "
            f"{bad!r} — truth must never be seeded from stage output")


# --------------------------------------------------------------------------
# Per-clip truth-quality checks (skip if the clip's data isn't present)
# --------------------------------------------------------------------------


def _clip_ids():
    return sorted(CLIPS)


def _ctx_or_skip(clip_id):
    output_dir, _shot_id = CLIPS[clip_id]
    if not Path(output_dir).exists():
        pytest.skip(f"{output_dir} not present on this machine")
    return load_clip(clip_id)


# Building "mismatch" truth involves a real shooting solve (least_squares
# over an RK4 flight integration) per flight segment, which is not cheap
# on a 50-60 anchor clip. Several tests below need the SAME (clip,
# scenario) truth; cache it process-wide so the full suite runs one
# solve per (clip, scenario) instead of one per test function.
_TRUTH_CACHE: dict[tuple[str, str], object] = {}


def _truth_or_skip(clip_id, scenario):
    ctx = _ctx_or_skip(clip_id)
    key = (clip_id, scenario)
    if key not in _TRUTH_CACHE:
        _TRUTH_CACHE[key] = build_truth(ctx, scenario, seed=0)
    return ctx, _TRUTH_CACHE[key]


@pytest.mark.parametrize("clip_id", _clip_ids())
@pytest.mark.parametrize("scenario", ["base", "mismatch", "sparse"])
def test_truth_hits_every_hard_knot(clip_id, scenario):
    _ctx, truth = _truth_or_skip(clip_id, scenario)
    by_frame = {f.frame: np.asarray(f.xyz) for f in truth.frames}
    for frame in truth.seed_anchor_frames:
        assert frame in by_frame, f"hard knot frame {frame} missing from truth"


@pytest.mark.parametrize("clip_id", _clip_ids())
def test_truth_reprojects_onto_real_hard_anchor_clicks(clip_id):
    """Truth built from base-scenario physics should reproject within 3px
    of the real click at every resolved hard-knot frame (the knot's xyz
    IS the click-ray-derived ground truth, so this is close to a
    tautology for ground_exact/goal states, but exercises the full
    resolve -> dense-track -> reproject path end to end)."""
    ctx, truth = _truth_or_skip(clip_id, "base")
    anchor_by_frame = {a.frame: a.image_xy for a in ctx.anchors.anchors
                       if a.image_xy is not None}
    by_frame = {f.frame: np.asarray(f.xyz) for f in truth.frames}
    n_checked = 0
    for frame in truth.seed_anchor_frames:
        uv_click = anchor_by_frame.get(frame)
        if uv_click is None:
            continue
        uv_truth = ctx.project(frame, by_frame[frame])
        err_px = math.hypot(uv_truth[0] - uv_click[0], uv_truth[1] - uv_click[1])
        assert err_px < 3.0, f"{clip_id} f{frame}: reprojection {err_px:.2f}px"
        n_checked += 1
    assert n_checked > 0


@pytest.mark.parametrize("clip_id", _clip_ids())
def test_truth_passes_near_airborne_waypoint_rays(clip_id):
    """Ray-only waypoints (airborne_*, catch, header/volley/chest) should
    be passed within a reported tolerance by the fitted arc. The tolerance
    below (3.0 m) is a MEASURED, not a priori, bound: gberch/kroupi01/s013
    all land under 1.5 m, but origi01 — which has long dense runs of
    airborne_low/mid/high waypoints between widely-spaced hard knots (22
    waypoints across just a few flight segments) — reaches ~2.4 m on its
    worst span, because the shooting solver in truth_builder.py weights
    exact hard-knot arrival (weight 25) well above waypoint proximity
    (weight 4). Re-balancing those weights (or fitting waypoints jointly
    across neighbouring segments) would tighten this at some risk to the
    2 cm knot-arrival guarantee; out of scope for this spike."""
    ctx, truth = _truth_or_skip(clip_id, "mismatch")
    by_frame = {f.frame: np.asarray(f.xyz) for f in truth.frames}
    hard_frames = set(truth.seed_anchor_frames)
    waypoint_states = {"airborne_low", "airborne_mid", "airborne_high",
                       "catch", "header", "volley", "chest"}
    errs = []
    for a in ctx.anchors.anchors:
        if (a.state not in waypoint_states or a.frame in hard_frames
                or a.image_xy is None or a.frame not in by_frame):
            continue
        C, d = ctx.ray(a.frame, a.image_xy)
        P = by_frame[a.frame]
        v = P - C
        along = float(np.dot(v, d))
        perp = float(np.linalg.norm(v - along * d))
        errs.append(perp)
    if not errs:
        pytest.skip(f"{clip_id}: no ray-only waypoints inside the truth span")
    max_err = max(errs)
    assert max_err < 3.0, f"{clip_id}: worst waypoint ray distance {max_err:.2f}m"


@pytest.mark.parametrize("clip_id", _clip_ids())
@pytest.mark.parametrize("scenario", ["base", "mismatch"])
def test_truth_has_no_teleports_and_stays_above_ground(clip_id, scenario):
    _ctx, truth = _truth_or_skip(clip_id, scenario)
    frames = sorted(truth.frames, key=lambda f: f.frame)
    max_step_m = 45.0 / truth.fps
    for a, b in zip(frames, frames[1:]):
        gap = b.frame - a.frame
        if gap <= 0:
            continue
        step = math.dist(a.xyz, b.xyz)
        assert step <= max_step_m * gap + 1e-6, (
            f"{clip_id} {scenario}: teleport {a.frame}->{b.frame} "
            f"step={step:.2f}m > {max_step_m * gap:.2f}m")
    min_z = min(f.xyz[2] for f in frames)
    assert min_z >= ts.BALL_RADIUS_M - 0.01 - 1e-9, (
        f"{clip_id} {scenario}: min z {min_z:.4f} below r-1cm")


@pytest.mark.parametrize("clip_id", _clip_ids())
def test_sparse_truth_matches_mismatch_truth_exactly(clip_id):
    """CONTRACT.md: 'sparse' uses the SAME physics as 'mismatch' (only the
    synthetic detector differs)."""
    _ctx, mismatch = _truth_or_skip(clip_id, "mismatch")
    _ctx, sparse = _truth_or_skip(clip_id, "sparse")
    assert [f.xyz for f in mismatch.frames] == [f.xyz for f in sparse.frames]
    assert mismatch.physics["segments"] == sparse.physics["segments"]


# --------------------------------------------------------------------------
# Synthetic detector sanity (per clip)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("clip_id", _clip_ids())
@pytest.mark.parametrize("scenario", ["base", "mismatch", "sparse"])
def test_synth_run_shape_and_calibration(clip_id, scenario):
    ctx, truth = _truth_or_skip(clip_id, scenario)
    synth = make_synth_run(ctx, truth, scenario, seed=0)

    assert synth.clip_id == clip_id
    assert synth.scenario == scenario
    assert len(synth.anchors) == len(ctx.anchors.anchors)
    assert synth.noise_model["sigma_px"] >= 1.5 - 1e-9

    for o in synth.observations:
        assert truth.frames[0].frame <= o.frame <= truth.frames[-1].frame
        assert 0.0 <= o.conf <= 1.0

    real_cov = synth.noise_model["real_coverage_reference"]
    achieved_cov = synth.noise_model["achieved_coverage"]
    target_cov = synth.noise_model["target_coverage"]
    assert 0.0 <= real_cov <= 1.0
    # Calibration should land within a generous band of its target (exact
    # match isn't guaranteed once speed/occlusion modifiers are applied).
    assert abs(achieved_cov - target_cov) < 0.20, (
        clip_id, scenario, achieved_cov, target_cov)
    if scenario == "sparse":
        _ctx, mismatch_truth = _truth_or_skip(clip_id, "mismatch")
        base_synth = make_synth_run(ctx, mismatch_truth, "mismatch", seed=0)
        assert synth.noise_model["target_coverage"] == pytest.approx(
            base_synth.noise_model["target_coverage"] * 0.5)

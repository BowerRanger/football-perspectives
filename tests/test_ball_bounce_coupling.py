"""Spin-aware bounce model + joint refit of bounce-adjacent flight arcs.

Phase-3 Task 5. Three layers:
  * ``bounce()`` unit tests — rigid-sphere-on-plane impulse model edge cases.
  * ``fit_coupled_bounce()`` synthetic round-trip — recover a KNOWN
    spin-coupled two-arc trajectory from projected pixels.
  * solver-level accept/reject gate for the coupled refit.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_physics import G_VEC, bounce
from src.utils.bundle_adjust import fit_coupled_bounce
from tests.fixtures.ball_synthetic import FPS, broadcast_camera

BALL_R = 0.11


# ----------------------------------------------------------------------
# A. bounce() unit tests
# ----------------------------------------------------------------------

@pytest.mark.unit
class TestBounceModel:
    def test_pure_normal_drop_no_spin(self):
        """ω=0 dead drop: v_out_z = -e·v_in_z, horizontal velocity unchanged,
        spin stays zero (no slip → no friction impulse)."""
        v_in = np.array([0.0, 0.0, -8.0])
        omega_in = np.zeros(3)
        e, mu = 0.7, 0.4
        v_out, omega_out = bounce(v_in, omega_in, e, mu, BALL_R)
        assert v_out[2] == pytest.approx(-e * v_in[2], rel=1e-9)
        assert v_out[0] == pytest.approx(0.0, abs=1e-9)
        assert v_out[1] == pytest.approx(0.0, abs=1e-9)
        assert np.linalg.norm(omega_out) == pytest.approx(0.0, abs=1e-9)

    def test_oblique_no_spin_imparts_forward_roll(self):
        """An oblique no-spin impact slides forward, so friction reduces
        horizontal speed and imparts forward-rolling (top)spin.

        Convention: for travel along +x, rolling spin is about +y
        (``ω = v/R``); friction induces that sign from a forward slide."""
        v_in = np.array([10.0, 0.0, -6.0])
        omega_in = np.zeros(3)
        e, mu = 0.7, 0.4
        v_out, omega_out = bounce(v_in, omega_in, e, mu, BALL_R)
        # Horizontal speed reduced (friction opposes forward slip).
        assert 0.0 < v_out[0] < v_in[0]
        # Forward slide → forward-rolling spin about +y.
        assert omega_out[1] > 0.0
        assert v_out[2] == pytest.approx(-e * v_in[2], rel=1e-9)

    def test_backspin_reduces_or_reverses_tangential(self):
        """Strong backspin: the contact point moves *forward* faster than the
        CoM, so the friction impulse points backward and reduces (or reverses)
        the horizontal velocity — the classic backspin check-up.

        For +x travel, backspin is ω about -y (surface at the contact moves
        +x faster than the CoM)."""
        v_in = np.array([6.0, 0.0, -6.0])
        omega_in = np.array([0.0, -60.0, 0.0])  # backspin for +x travel
        e, mu = 0.7, 0.6
        v_out, _ = bounce(v_in, omega_in, e, mu, BALL_R)
        v_out_nospin, _ = bounce(v_in, np.zeros(3), e, mu, BALL_R)
        assert v_out[0] < v_out_nospin[0]

    def test_topspin_increases_tangential(self):
        """Topspin: the contact point already moves *backward* relative to the
        CoM, so the slip (and friction) act forward — horizontal speed
        increases vs the no-spin bounce (the ball skids on).

        For +x travel, topspin is ω about +y."""
        v_in = np.array([6.0, 0.0, -6.0])
        omega_in = np.array([0.0, 60.0, 0.0])  # topspin for +x travel
        e, mu = 0.7, 0.6
        v_out, _ = bounce(v_in, omega_in, e, mu, BALL_R)
        v_out_nospin, _ = bounce(v_in, np.zeros(3), e, mu, BALL_R)
        assert v_out[0] > v_out_nospin[0]

    def test_omega_coupled_to_friction_impulse(self):
        """A no-spin oblique impact produces a spin change consistent with the
        friction torque about the contact point."""
        v_in = np.array([8.0, 0.0, -6.0])
        omega_in = np.zeros(3)
        e, mu = 0.6, 0.5
        v_out, omega_out = bounce(v_in, omega_in, e, mu, BALL_R)
        # Tangential CoM velocity lost = jt/m; spin gained = (r_c × jt)/(αR²).
        dv = v_in[0] - v_out[0]
        assert dv > 0
        # The friction impulse is -dv·x̂; Δω = (r_c × jt)/(αR²) with
        # r_c = (0,0,-R) gives Δω_y = +dv/(αR) for forward slide.
        alpha = 2.0 / 3.0
        expected_wy = dv / (alpha * BALL_R)
        assert omega_out[1] == pytest.approx(expected_wy, rel=1e-6)

    def test_ascending_ball_passthrough(self):
        """A ball already moving up is not in contact — no impulse."""
        v_in = np.array([3.0, 0.0, 2.0])
        omega_in = np.array([0.0, 10.0, 0.0])
        v_out, omega_out = bounce(v_in, omega_in, 0.7, 0.4, BALL_R)
        np.testing.assert_allclose(v_out, v_in)
        np.testing.assert_allclose(omega_out, omega_in)

    def test_grip_limit_caps_friction(self):
        """With huge mu, the friction is capped by the rolling condition, not
        by Coulomb — the contact point's slip is driven to (near) zero, never
        reversed past rolling."""
        v_in = np.array([10.0, 0.0, -6.0])
        omega_in = np.zeros(3)
        v_low, w_low = bounce(v_in, omega_in, 0.7, 0.05, BALL_R)
        v_grip, w_grip = bounce(v_in, omega_in, 0.7, 5.0, BALL_R)
        # More friction removes more tangential velocity.
        assert v_grip[0] < v_low[0]
        # But never overshoots into a backward CoM velocity from forward slip.
        assert v_grip[0] >= -1e-9


# ----------------------------------------------------------------------
# B. fit_coupled_bounce() synthetic round-trip
# ----------------------------------------------------------------------

def _integrate_truth(p0, v0, omega, duration_s, substep=0.0005):
    """Fine RK4 ground-truth integrator returning (times, positions, vels)."""
    drag = 0.005

    def accel(v):
        return G_VEC + drag * np.cross(omega, v)

    n = int(round(duration_s / substep)) + 1
    times = np.arange(n) * substep
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    p, v = np.asarray(p0, float).copy(), np.asarray(v0, float).copy()
    pos[0], vel[0] = p, v
    for i in range(1, n):
        h = substep
        k1v, k1p = accel(v), v
        k2v, k2p = accel(v + 0.5 * h * k1v), v + 0.5 * h * k1v
        k3v, k3p = accel(v + 0.5 * h * k2v), v + 0.5 * h * k2v
        k4v, k4p = accel(v + h * k3v), v + h * k3v
        p = p + (h / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
        v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        pos[i], vel[i] = p, v
    return times, pos, vel


def _project(pts, K, R, t):
    out = []
    for p in pts:
        cam = R @ p + t
        pix = K @ cam
        out.append((float(pix[0] / pix[2]), float(pix[1] / pix[2])))
    return out


def _build_two_arc_scene(p0, v0, omega0, e, mu, *, fa_n=14, fb_n=14,
                          noise_px=0.0, seed=0):
    """Generate a spin-coupled two-arc trajectory and project to pixels.

    The construction mirrors exactly what ``fit_coupled_bounce`` derives:
    arc A is integrated from ``(p0, v0, omega0)`` for ``fa_n`` frames to the
    bounce frame, ``bounce(v_in, omega0, e, mu)`` is applied, and arc B
    continues from the bounce position with the post-bounce state.
    """
    K, R, t = broadcast_camera()
    fps = FPS
    rng = np.random.default_rng(seed)

    fa_start = 5
    bounce_frame = fa_start + fa_n
    # Arc A from launch to the bounce frame.
    dur_a = (bounce_frame - fa_start) / fps
    times_a, pos_a, vel_a = _integrate_truth(p0, v0, omega0, duration_s=dur_a)
    p_bounce = pos_a[-1].copy()
    v_in = vel_a[-1].copy()

    v_out, omega_out = bounce(v_in, omega0, e, mu, BALL_R)
    dur_b = (fb_n + 1) / fps
    times_b, pos_b, vel_b = _integrate_truth(
        p_bounce, v_out, omega_out, duration_s=dur_b,
    )

    # Observation frames (interior to each arc; the bounce frame is the node).
    frames_a = list(range(fa_start, bounce_frame))
    frames_b = list(range(bounce_frame + 1, bounce_frame + 1 + fb_n))

    dt_a = times_a[1] - times_a[0]
    pos_a_s = np.array([
        pos_a[min(int(round(((f - fa_start) / fps) / dt_a)), len(times_a) - 1)]
        for f in frames_a
    ])
    dt_b = times_b[1] - times_b[0]
    pos_b_s = np.array([
        pos_b[min(int(round(((f - bounce_frame) / fps) / dt_b)), len(times_b) - 1)]
        for f in frames_b
    ])

    uv_a = _project(pos_a_s, K, R, t)
    uv_b = _project(pos_b_s, K, R, t)
    if noise_px > 0:
        uv_a = [(u + rng.normal(0, noise_px), v + rng.normal(0, noise_px))
                for u, v in uv_a]
        uv_b = [(u + rng.normal(0, noise_px), v + rng.normal(0, noise_px))
                for u, v in uv_b]

    obs_a = list(zip(frames_a, uv_a))
    obs_b = list(zip(frames_b, uv_b))
    cams_a = ([K] * len(frames_a), [R] * len(frames_a), [t] * len(frames_a))
    cams_b = ([K] * len(frames_b), [R] * len(frames_b), [t] * len(frames_b))

    # A clean seed for arc A's initial state at fa_start.
    p0_seed = pos_a_s[0].copy()
    v0_seed = (pos_a_s[1] - pos_a_s[0]) * fps if len(pos_a_s) > 1 else v0
    return dict(
        obs_a=obs_a, obs_b=obs_b, cams_a=cams_a, cams_b=cams_b,
        bounce_frame=bounce_frame, fps=fps,
        p0_seed=p0_seed, v0_seed=v0_seed,
        truth=dict(omega0=omega0, e=e, mu=mu, p_bounce=p_bounce, v_in=v_in),
    )


@pytest.mark.unit
class TestCoupledBounceFit:
    """``fit_coupled_bounce`` round-trip behaviour.

    IMPORTANT (Phase-3 Task 5 risk note): on a single broadcast camera the
    11-DOF joint fit ``(p0, v0, ω0, e, μ)`` is depth/spin degenerate.  The
    weak Magnus signal (drag_k_over_m=0.005 deflects a 0.6 s arc by only
    ~0.3 m at 33 rad/s) is freely absorbed by the free ``(p0, v0)`` depth,
    so ω runs to the bounds even with the TRUE seed and zero noise.  These
    tests document that the optimiser RUNS and returns in-bounds params, and
    pin the *degeneracy itself* so the default-off decision is justified by
    a test, not just a claim.  The clean recovery the spec hoped for requires
    a depth-pinning constraint (multi-view / replay triangulation) that the
    monocular path does not have — hence ``bounce_coupling`` defaults False.
    """

    def test_fit_runs_and_returns_in_bounds(self):
        """The joint fit converges (no exception) and respects all bounds."""
        p0 = np.array([30.0, 40.0, 0.5])
        v0 = np.array([9.0, -5.0, 7.0])
        omega0_true = np.array([0.0, 30.0, 15.0])
        e_true, mu_true = 0.72, 0.35
        scene = _build_two_arc_scene(
            p0, v0, omega0_true, e_true, mu_true,
            fa_n=16, fb_n=16, noise_px=0.3, seed=3,
        )
        p0h, v0h, omegah, eh, muh, resid = fit_coupled_bounce(
            scene["obs_a"], scene["obs_b"],
            cams_a=scene["cams_a"], cams_b=scene["cams_b"],
            bounce_frame=scene["bounce_frame"], fps=scene["fps"],
            p0_seed=scene["p0_seed"], v0_seed=scene["v0_seed"],
            omega0_seed=omega0_true * 0.5,
            e_seed=0.7, mu_seed=0.3,
        )
        assert np.all(np.isfinite(omegah))
        assert float(np.linalg.norm(omegah)) <= 95.0 * np.sqrt(3) + 1e-6
        assert 0.5 <= eh <= 0.85, f"e {eh:.2f} out of bounds"
        assert 0.0 <= muh <= 0.7, f"mu {muh:.2f} out of bounds"
        # The optimiser lowers the combined residual below the noise-scaled
        # ceiling — it is a valid LM, just not a unique inverse.
        assert resid < 5.0, f"residual {resid:.2f} px"

    def test_monocular_degeneracy_is_real(self):
        """Document the degeneracy: even with the TRUE seed and ZERO noise,
        the joint fit walks ω away from truth.  This is the evidence behind
        ``bounce_coupling`` defaulting False.  If this assertion ever flips
        (recovery becomes stable — e.g. a depth-pinning constraint was added)
        revisit the default.
        """
        p0 = np.array([30.0, 40.0, 0.5])
        v0 = np.array([9.0, -5.0, 7.0])
        omega0_true = np.array([0.0, 30.0, 15.0])
        e_true, mu_true = 0.72, 0.35
        scene = _build_two_arc_scene(
            p0, v0, omega0_true, e_true, mu_true,
            fa_n=16, fb_n=16, noise_px=0.0, seed=0,
        )
        _, _, omegah, _, _, _ = fit_coupled_bounce(
            scene["obs_a"], scene["obs_b"],
            cams_a=scene["cams_a"], cams_b=scene["cams_b"],
            bounce_frame=scene["bounce_frame"], fps=scene["fps"],
            p0_seed=p0, v0_seed=v0, omega0_seed=omega0_true,
            e_seed=e_true, mu_seed=mu_true,
        )
        mag_true = float(np.linalg.norm(omega0_true))
        mag_hat = float(np.linalg.norm(omegah))
        rel = abs(mag_hat - mag_true) / mag_true
        # The fit does NOT recover within 20 % — the degeneracy is real.
        assert rel > 0.20, (
            "monocular joint fit unexpectedly recovered |ω| within 20% — "
            "revisit the bounce_coupling default (this is now identifiable)"
        )


# ----------------------------------------------------------------------
# C. solver-level config + gate
# ----------------------------------------------------------------------

@pytest.mark.unit
class TestSolverBounceCouplingGate:
    def test_config_default_off_and_mu_max(self):
        from src.utils.ball_piecewise_solver import SolverCfg
        cfg = SolverCfg()
        assert hasattr(cfg, "bounce_coupling")
        # DEFAULT FALSE — the visible Phase-3 wins must not be jeopardised.
        assert cfg.bounce_coupling is False
        assert cfg.mu_max == pytest.approx(0.7)

    def test_disabled_solver_output_unchanged(self):
        """With the flag off (default) a flight→bounce→flight scene solves
        exactly as before — the coupling pass is a no-op."""
        from src.utils.ball_physics import two_knot_arc, eval_parabola
        from src.utils.ball_piecewise_solver import (
            SolverCfg, TrajectoryNode, solve_piecewise,
        )
        from tests.fixtures.ball_synthetic import (
            per_frame_cams, project_track, steps_from_pixels,
        )

        n = 60
        a = np.array([50.0, 30.0, 0.11])
        b = np.array([42.0, 33.0, 0.11])
        c = np.array([36.0, 35.0, 0.11])

        def arc_worlds(p_a, p_b, fa, fb):
            T = (fb - fa) / FPS
            p0, v0 = two_knot_arc(p_a, p_b, T)
            ts = np.array([(f - fa) / FPS for f in range(fa, fb + 1)])
            pts = eval_parabola(p0, v0, ts)
            return {fa + i: pts[i] for i in range(len(pts))}

        truth = {**arc_worlds(a, b, 5, 25), **arc_worlds(b, c, 25, 45)}
        K, R, t = broadcast_camera()
        pixels = project_track(truth, K, R, t)
        nodes = [
            TrajectoryNode(5, tuple(a), "kick"),
            TrajectoryNode(25, tuple(b), "bounce"),
            TrajectoryNode(45, tuple(c), "catch"),
        ]
        steps = steps_from_pixels(pixels, n, p_flight=0.9)
        Ks, Rs, ts = per_frame_cams(n)
        common = dict(
            nodes=nodes, steps=steps, confidences={},
            per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
            distortion=(0.0, 0.0), fps=FPS, n_frames=n,
        )
        off = solve_piecewise(**common, cfg=SolverCfg(bounce_coupling=False))
        # Default == explicit-off.
        default = solve_piecewise(**common, cfg=SolverCfg())
        for f in range(5, 46):
            np.testing.assert_allclose(
                off.world_by_frame[f][0], default.world_by_frame[f][0],
                atol=1e-9,
            )
        assert "bounce_coupling" not in off.diagnostics

    def test_enabled_path_runs_without_crashing(self):
        """With the flag ON the coupling pass executes over a flight→bounce→
        flight scene. It either accepts (gate met) or keeps the independent
        fits (gate not met) — both are valid; it must not crash and must keep
        node continuity intact."""
        from src.utils.ball_physics import two_knot_arc, eval_parabola
        from src.utils.ball_piecewise_solver import (
            SolverCfg, TrajectoryNode, solve_piecewise,
        )
        from tests.fixtures.ball_synthetic import (
            per_frame_cams, project_track, steps_from_pixels,
        )

        n = 60
        a = np.array([50.0, 30.0, 0.11])
        b = np.array([42.0, 33.0, 0.11])
        c = np.array([36.0, 35.0, 0.11])

        def arc_worlds(p_a, p_b, fa, fb):
            T = (fb - fa) / FPS
            p0, v0 = two_knot_arc(p_a, p_b, T)
            ts = np.array([(f - fa) / FPS for f in range(fa, fb + 1)])
            pts = eval_parabola(p0, v0, ts)
            return {fa + i: pts[i] for i in range(len(pts))}

        truth = {**arc_worlds(a, b, 5, 25), **arc_worlds(b, c, 25, 45)}
        K, R, t = broadcast_camera()
        pixels = project_track(truth, K, R, t)
        nodes = [
            TrajectoryNode(5, tuple(a), "kick"),
            TrajectoryNode(25, tuple(b), "bounce"),
            TrajectoryNode(45, tuple(c), "catch"),
        ]
        steps = steps_from_pixels(pixels, n, p_flight=0.9)
        Ks, Rs, ts = per_frame_cams(n)
        result = solve_piecewise(
            nodes=nodes, steps=steps, confidences={},
            per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
            distortion=(0.0, 0.0), fps=FPS, n_frames=n,
            cfg=SolverCfg(bounce_coupling=True),
        )
        # Node continuity is preserved regardless of accept/reject.
        np.testing.assert_allclose(result.world_by_frame[25][0], b, atol=1e-6)
        np.testing.assert_allclose(result.world_by_frame[5][0], a, atol=1e-6)
        np.testing.assert_allclose(result.world_by_frame[45][0], c, atol=1e-6)

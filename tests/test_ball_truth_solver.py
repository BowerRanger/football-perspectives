"""Ball Studio solver: triangulation, ray constraints, segments, flags."""

from __future__ import annotations

import numpy as np
import pytest

from src.schemas import ball_truth as bt
from src.utils import ball_truth_solver as S
from src.utils.ball_hybrid_physics import DEFAULT_MAGNUS_COEFF, shoot_arc, simulate
from src.utils.frame_cadence import content_time_shift

FPS = 30.0
IMG = (1920, 1080)


def look_at(centre, target, fx=2000.0, dist=(0.0, 0.0)) -> S.Cam:
    centre, target = np.asarray(centre, float), np.asarray(target, float)
    fwd = target - centre
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    t = -R @ centre
    K = np.array([[fx, 0, IMG[0] / 2], [0, fx, IMG[1] / 2], [0, 0, 1.0]])
    return S.Cam(K, R, t, dist, IMG)


CAM_A = look_at([52.5, -30, 20], [52.5, 30, 0])
CAM_B = look_at([10, 34, 15], [52.5, 20, 0], dist=(-0.02, 0.0))


def make_ctx(joints=None, offset_b=-10) -> S.SolveContext:
    cams = {"a": CAM_A, "b": CAM_B}
    offs = {"a": 0, "b": offset_b}

    def camera(sid, sf):
        return cams[sid]

    def joint(pid, bone, ref):
        if joints is None:
            return None
        return joints(pid, bone, ref)

    return S.SolveContext(FPS, offs, camera, joint)


def uv(cam, p):
    return [float(v) for v in cam.project(np.asarray(p))[0][0]]


def obs(ctx, sid, p, ref):
    cam = ctx.camera(sid, 0)
    return {"shot_id": sid, "shot_frame": ref + ctx.offsets[sid], "uv": uv(cam, p)}


def make_doc(keys, segments=(), observations=(), events=()):
    return {
        "version": 1, "group_id": "g", "reference_shot": "a", "fps": FPS,
        "shots": [{"shot_id": "a", "frame_offset": 0}, {"shot_id": "b", "frame_offset": -10}],
        "outcome": "unknown", "keys": keys, "segments": list(segments),
        "observations": list(observations), "events": list(events),
        "meta": {},
    }


def tri_key(ctx, kid, ref, p):
    return {"id": kid, "frame": ref, "xyz": [0, 0, 0], "source": "triangulated",
            "constraint": {}, "observations": [obs(ctx, "a", p, ref), obs(ctx, "b", p, ref)]}


def norm(doc):
    n, errs = bt.validate_truth(doc)
    assert not errs, errs
    return n


def test_triangulation_exact_without_noise():
    p = np.array([40.0, 25.0, 3.0])
    res = S.triangulate([(CAM_A, uv(CAM_A, p)), (CAM_B, uv(CAM_B, p))])
    assert res.ok
    assert np.allclose(res.xyz, p, atol=1e-4)
    assert max(res.residual_px) < 1e-3
    assert res.ray_angle_deg > 20
    assert res.skew_gap_cm < 0.1


def test_triangulation_reports_residual_for_inconsistent_click():
    p = np.array([40.0, 25.0, 3.0])
    ub = uv(CAM_B, p)
    ub[0] += 30  # bad click in view b
    res = S.triangulate([(CAM_A, uv(CAM_A, p)), (CAM_B, ub)])
    assert max(res.residual_px) > 1.0
    assert res.skew_gap_cm > 1.0


def test_triangulation_noise_residual_scales():
    rng = np.random.default_rng(0)
    p = np.array([60.0, 20.0, 1.0])
    ua = np.array(uv(CAM_A, p)) + rng.normal(0, 1, 2)
    ub = np.array(uv(CAM_B, p)) + rng.normal(0, 1, 2)
    res = S.triangulate([(CAM_A, ua), (CAM_B, ub)])
    assert res.ok and max(res.residual_px) < 4
    assert np.linalg.norm(res.xyz - p) < 0.5


@pytest.mark.parametrize("mode,constraint,truth", [
    ("ground", {}, [30.0, 10.0, S.GROUND_Z]),
    ("height", {"height_m": 1.7}, [30.0, 10.0, 1.7]),
    ("plane", {"plane": {"axis": "y", "value": 10.0}}, [30.0, 10.0, 2.0]),
])
def test_single_view_constraints(mode, constraint, truth):
    ctx = make_ctx()
    xyz, err = S.constrain_ray(CAM_A, uv(CAM_A, truth), constraint, mode,
                               ref_frame=0, joint=ctx.joint)
    assert err is None
    assert np.allclose(xyz, truth, atol=1e-4)


def test_depth_constraint():
    p = np.array([30.0, 10.0, 2.0])
    depth = float(CAM_A.depth(p)[0])
    d_ray = np.linalg.norm(p - CAM_A.centre)
    xyz, _ = S.constrain_ray(CAM_A, uv(CAM_A, p), {"depth_m": float(d_ray)}, "depth",
                             ref_frame=0, joint=lambda *a: None)
    assert np.allclose(xyz, p, atol=1e-4) and depth > 0


def test_player_constraint_keeps_pixel_lateral_depth_from_joint():
    ball = np.array([30.0, 10.0, 0.3])
    joint = ball + np.array([0.0, 0.2, 0.0])  # slightly different depth
    ctx = make_ctx(joints=lambda pid, bone, ref: joint)
    xyz, err = S.constrain_ray(CAM_A, uv(CAM_A, ball),
                               {"player_id": "P1", "bone": "r_foot"}, "player",
                               ref_frame=0, joint=ctx.joint)
    assert err is None
    assert np.allclose(uv(CAM_A, xyz), uv(CAM_A, ball), atol=1e-3)  # pixel authoritative
    C = CAM_A.centre
    assert abs(np.dot(xyz - C, (joint - C) / np.linalg.norm(joint - C))
               - np.linalg.norm(joint - C)) < 0.3


def test_player_joint_missing_errors():
    xyz, err = S.constrain_ray(CAM_A, [900, 500], {"player_id": "P1", "bone": "head"},
                               "player", ref_frame=0, joint=lambda *a: None)
    assert xyz is None and err == "unknown_player_joint"


def test_epipolar_polyline_contains_true_point():
    p = np.array([40.0, 25.0, 3.0])
    C, d = CAM_A.ray(uv(CAM_A, p))
    poly = S.epipolar_polyline(C, d, CAM_B)
    assert poly
    ub = np.array(uv(CAM_B, p))
    dmin = min(np.hypot(*(np.array(q) - ub)) for q in poly)
    assert dmin < 80  # polyline is subsampled; the point lies on the curve


def test_flight_passes_through_keys_and_follows_gravity_drag():
    ctx = make_ctx()
    pa, pb = np.array([30.0, 5.0, 0.11]), np.array([45.0, 28.0, 0.11])
    # a lofted ball: middle key at apex-ish
    doc = norm(make_doc([
        tri_key(ctx, "k0", 0, pa),
        {"id": "k1", "frame": 60, "xyz": list(pb), "source": "manual", "constraint": {}, "observations": []},
    ], [{"from": "k0", "to": "k1", "kind": "flight"}]))
    out = S.solve_truth(doc, ctx)
    assert out["ok"], out["flags"]
    xyz = np.array(out["dense"]["xyz"])
    assert len(xyz) == 61
    assert np.allclose(xyz[0], pa, atol=1e-3) and np.allclose(xyz[-1], pb, atol=1e-3)
    assert xyz[:, 2].max() > 5  # lofted: 2 s flight over 15-25 m
    # matches the shoot_arc reference exactly
    v0 = shoot_arc(pa, 0, pb, 2.0, cd=0.25)
    ref = simulate(pa, v0, np.arange(61) / FPS, cd=0.25)
    assert np.allclose(xyz, ref, atol=2e-3)
    assert out["keys"][0]["residual_px"]["a"] < 1e-2


def test_roll_between_ground_keys_and_auto_kind():
    ctx = make_ctx()
    pa, pb = np.array([30.0, 5.0, 0.11]), np.array([36.0, 9.0, 0.11])
    doc = norm(make_doc([tri_key(ctx, "k0", 0, pa), tri_key(ctx, "k1", 30, pb)]))
    out = S.solve_truth(doc, ctx)
    assert out["segments"][0]["kind"] == "roll" and out["segments"][0]["auto"]
    xyz = np.array(out["dense"]["xyz"])
    assert np.allclose(xyz[:, 2], 0.11, atol=1e-3)
    assert np.allclose(xyz[-1], pb, atol=1e-3)


def test_roll_fits_soft_observation_deceleration():
    ctx = make_ctx()
    pa, pb = np.array([30.0, 5.0, 0.11]), np.array([40.0, 5.0, 0.11])
    # true motion decelerating: x(t) = 30 + 14*t - 4*t^2 over 1.0 s? pick T=1.0
    T = 1.0
    acc = -2.0
    v0x = (10.0 - 0.5 * acc * T * T) / T
    mid_t = 0.5
    mid = np.array([30 + v0x * mid_t + 0.5 * acc * mid_t ** 2, 5.0, 0.11])
    doc = norm(make_doc(
        [tri_key(ctx, "k0", 0, pa), tri_key(ctx, "k1", 30, pb)],
        observations=[obs(ctx, "a", mid, 15)]))
    out = S.solve_truth(doc, ctx)
    xyz = np.array(out["dense"]["xyz"])
    assert abs(xyz[15, 0] - mid[0]) < 0.05
    so = [o for o in out["observations"] if o["kind"] == "soft"][0]
    assert so["residual_px"] < 3
    assert out["segments"][0]["rms_obs_px"] < 3


def test_curl_fit_recovers_magnus_arc():
    ctx = make_ctx()
    pa, pb = np.array([30.0, 0.0, 0.11]), np.array([55.0, 30.0, 0.11])
    T = 2.0
    omega = np.array([0.0, 0.0, 30.0])  # sidespin
    v0 = shoot_arc(pa, 0, pb, T, cd=0.25, omega=omega)
    frames = np.arange(0, 61)
    truth = simulate(pa, v0, frames / FPS, cd=0.25, omega=omega)
    soft = [obs(ctx, "a" if i % 2 else "b", truth[i], int(i)) for i in range(6, 56, 3)]
    doc = norm(make_doc([
        tri_key(ctx, "k0", 0, pa), tri_key(ctx, "k1", 60, pb),
    ], [{"from": "k0", "to": "k1", "kind": "flight"}], observations=soft))
    out = S.solve_truth(doc, ctx)
    got = np.array(out["dense"]["xyz"])
    assert np.abs(got - truth).max() < 0.3
    assert out["segments"][0]["params"]["omega"] is not None or \
        out["segments"][0]["params"].get("delta_bic") is not None
    # without the fit a drag-only arc would miss the mid curve by metres
    plain = simulate(pa, shoot_arc(pa, 0, pb, T, cd=0.25), frames / FPS, cd=0.25)
    assert np.abs(plain - truth).max() > 1.0


def test_magnus_off_disables_fit():
    ctx = make_ctx()
    pa, pb = np.array([30.0, 0.0, 0.11]), np.array([55.0, 30.0, 0.11])
    doc = norm(make_doc(
        [tri_key(ctx, "k0", 0, pa), tri_key(ctx, "k1", 60, pb)],
        [{"from": "k0", "to": "k1", "kind": "flight", "params": {"magnus": "off"}}],
        observations=[obs(ctx, "a", [40, 10, 3], 20)] * 4))
    out = S.solve_truth(doc, ctx)
    assert out["segments"][0]["params"].get("omega") is None
    assert "delta_bic" not in out["segments"][0]["params"]


def test_carried_follows_joint_with_offsets():
    def joints(pid, bone, ref):
        return np.array([30.0 + 0.1 * ref, 5.0, 0.1])

    ctx = make_ctx(joints=joints)
    pa, pb = joints("P", "r_foot", 0) + [0.2, 0, 0.0], joints("P", "r_foot", 20) + [0.2, 0, 0.0]
    doc = norm(make_doc([
        {"id": "k0", "frame": 0, "xyz": list(pa), "source": "manual", "constraint": {}, "observations": []},
        {"id": "k1", "frame": 20, "xyz": list(pb), "source": "manual", "constraint": {}, "observations": []},
    ], [{"from": "k0", "to": "k1", "kind": "carried",
         "params": {"player_id": "P", "bone": "r_foot"}}]))
    out = S.solve_truth(doc, ctx)
    xyz = np.array(out["dense"]["xyz"])
    assert np.allclose(xyz[10], joints("P", "r_foot", 10) + [0.2, 0, 0.0], atol=1e-6)


def test_carried_without_joint_data_is_error():
    ctx = make_ctx()
    doc = norm(make_doc([
        {"id": "k0", "frame": 0, "xyz": [1, 1, 1], "source": "manual", "constraint": {}, "observations": []},
        {"id": "k1", "frame": 10, "xyz": [2, 1, 1], "source": "manual", "constraint": {}, "observations": []},
    ], [{"from": "k0", "to": "k1", "kind": "carried",
         "params": {"player_id": "P", "bone": "head"}}]))
    out = S.solve_truth(doc, ctx)
    assert not out["ok"]
    assert any(f["code"] == "unknown_player_joint" for f in out["flags"])


def test_linear_and_static():
    ctx = make_ctx()
    k = lambda i, f, p: {"id": i, "frame": f, "xyz": p, "source": "manual", "constraint": {}, "observations": []}  # noqa: E731
    doc = norm(make_doc([k("a", 0, [0, 0, 1]), k("b", 10, [10, 0, 1]), k("c", 20, [10, 2, 1])],
                        [{"from": "a", "to": "b", "kind": "linear"},
                         {"from": "b", "to": "c", "kind": "static"}]))
    out = S.solve_truth(doc, ctx)
    xyz = np.array(out["dense"]["xyz"])
    assert np.allclose(xyz[5], [5, 0, 1])
    assert np.allclose(xyz[15], [10, 0, 1]) and np.allclose(xyz[20], [10, 2, 1])
    assert any(f["code"] == "segment_infeasible" for f in out["flags"])  # 2 m drift


def test_sanity_flags_below_ground_and_speed_and_residual():
    ctx = make_ctx()
    k = lambda i, f, p: {"id": i, "frame": f, "xyz": p, "source": "manual", "constraint": {}, "observations": []}  # noqa: E731
    doc = norm(make_doc([k("a", 0, [0, 0, -0.5]), k("b", 2, [60, 0, 1])],
                        [{"from": "a", "to": "b", "kind": "linear"}]))
    out = S.solve_truth(doc, ctx)
    codes = {f["code"] for f in out["flags"]}
    assert {"below_ground", "speed_exceeds_limit"} <= codes

    bad = tri_key(ctx, "k", 5, [40, 20, 1])
    bad["observations"][1]["uv"][0] += 60
    out2 = S.solve_truth(norm(make_doc([bad])), ctx)
    assert not out2["ok"]
    assert any(f["code"] == "residual_exceeds_limit" for f in out2["flags"])
    assert out2["keys"][0]["status"] == "error"
    assert out2["keys"][0]["skew_gap_cm"] > 5


def test_discontinuity_without_event():
    ctx = make_ctx()
    k = lambda i, f, p: {"id": i, "frame": f, "xyz": p, "source": "manual", "constraint": {}, "observations": []}  # noqa: E731
    keys = [k("a", 0, [0, 0, 1]), k("b", 10, [10, 0, 1]), k("c", 20, [10, 10, 1])]
    segs = [{"from": "a", "to": "b", "kind": "linear"}, {"from": "b", "to": "c", "kind": "linear"}]
    out = S.solve_truth(norm(make_doc(keys, segs)), ctx)
    assert any(f["code"] == "discontinuity" for f in out["flags"])
    ev = [{"frame": 10, "kind": "touch"}]
    out = S.solve_truth(norm(make_doc(keys, segs, events=ev)), ctx)
    assert not any(f["code"] == "discontinuity" for f in out["flags"])


def test_projections_cover_dense_frames_and_shot_offsets():
    ctx = make_ctx()
    pa, pb = np.array([30.0, 5.0, 0.11]), np.array([36.0, 9.0, 0.11])
    doc = norm(make_doc([tri_key(ctx, "k0", 0, pa), tri_key(ctx, "k1", 30, pb)]))
    out = S.solve_truth(doc, ctx)
    pb_ = out["projections"]["b"]
    assert pb_["shot_frames"][0] == -10 and pb_["frames"][0] == 0
    assert pb_["uv"][0] == pytest.approx(uv(CAM_B, pa), abs=0.02)


def test_single_key_and_empty():
    ctx = make_ctx()
    out = S.solve_truth(norm(make_doc([])), ctx)
    assert out["ok"] and out["dense"]["frames"] == []
    out = S.solve_truth(norm(make_doc([tri_key(ctx, "k", 3, [40, 20, 1])])), ctx)
    assert out["dense"]["frames"] == [3]


def test_magnus_constant_sane():
    assert DEFAULT_MAGNUS_COEFF > 0


def test_project_nans_points_past_the_distortion_fold_radius():
    # k1=-0.2: r_d(r) = r(1 + k1 r^2) peaks at r^2 = 1/(3*0.2); beyond it the
    # polynomial folds far off-image points back INTO the frame.
    cam = look_at([0, 0, 10], [0, 50, 10], dist=(-0.2, 0.0))
    inside = np.array([0.0, 50.0, 10.0])          # optical axis, r = 0
    folded = np.array([100.0, 50.0, 10.0])        # normalised r = 2 > fold
    out, _ = cam.project(np.stack([inside, folded]))
    assert np.isfinite(out[0]).all()
    assert np.isnan(out[1]).all()


def test_epipolar_polyline_has_no_fold_back_branch():
    # A wide, distorted view whose near-camera stretch of the ray is far off
    # frame: the unguarded projection folds it back as a second line.
    other = look_at([0, 0, 10], [0, 50, 0], fx=900.0, dist=(-0.25, 0.0))
    origin = np.array([-30.0, 10.0, 12.0])
    target = np.array([5.0, 40.0, 0.11])
    d = (target - origin) / np.linalg.norm(target - origin)
    poly = S.epipolar_polyline(origin, d, other, n_samples=2000, max_points=200)
    assert poly
    s = np.geomspace(2.0, 250.0, 2000)
    pts = origin[None, :] + s[:, None] * d[None, :]
    cam_pts = pts @ other.R.T + other.t
    r = np.hypot(cam_pts[:, 0], cam_pts[:, 1]) / cam_pts[:, 2]
    k1, k2 = other.dist
    fold_ok = (cam_pts[:, 2] > 0) & (1 + 3 * k1 * r**2 + 5 * k2 * r**4 > 0)
    good_uv, _ = other.project(pts[fold_ok])
    # every polyline point must lie on the genuine (pre-fold) branch
    for q in poly:
        assert np.min(np.hypot(*(good_uv - np.asarray(q)).T)) < 1.0


# --- cadence-aware time (25->30 pulldown) ----------------------------------

N_CAD = 120
SHIFT_A = content_time_shift(N_CAD, [f for f in range(1, N_CAD) if f % 6 == 5], FPS)
SHIFT_B = content_time_shift(N_CAD, [f for f in range(1, N_CAD) if f % 6 == 2], FPS)


def cadence_ctx(offset_b=-10, with_shift=True):
    base = make_ctx(offset_b=offset_b)
    shifts = {"a": SHIFT_A, "b": SHIFT_B}

    def time_shift(sid, sf):
        return float(shifts[sid][sf]) if 0 <= sf < N_CAD else 0.0

    return S.SolveContext(FPS, base.offsets, base.camera, base.joint,
                          reference_shot="a",
                          time_shift=time_shift if with_shift else None)


PA, PB = np.array([30.0, 5.0, 0.11]), np.array([45.0, 28.0, 0.11])
FA, FB = 12, 42


def truth_at(ctx, t):
    ta, tb = ctx.ref_time(FA), ctx.ref_time(FB)
    return PA + (PB - PA) * (t - ta) / (tb - ta)


def cadence_doc(ctx):
    keys = [{"id": "k0", "frame": FA, "xyz": list(PA), "source": "manual", "constraint": {},
             "observations": []},
            {"id": "k1", "frame": FB, "xyz": list(PB), "source": "manual", "constraint": {},
             "observations": []}]
    soft = [{"shot_id": "a", "shot_frame": f, "uv": uv(CAM_A, truth_at(ctx, ctx.obs_time("a", f)))}
            for f in range(FA + 2, FB - 1, 3)]
    return norm(make_doc(keys, [{"from": "k0", "to": "k1", "kind": "roll"}], soft))


def test_uniform_context_has_no_time_shift():
    ctx = make_ctx()
    assert ctx.ref_time(30) == pytest.approx(1.0)
    assert ctx.obs_time("b", 20) == pytest.approx(30 / FPS)


def test_soft_observations_fit_at_content_time():
    ctx = cadence_ctx()
    doc = cadence_doc(ctx)
    good = S.solve_truth(doc, ctx)
    soft = [o["residual_px"] for o in good["observations"] if o["kind"] == "soft"]
    assert max(soft) < 0.5
    naive = S.solve_truth(doc, cadence_ctx(with_shift=False))
    soft_naive = [o["residual_px"] for o in naive["observations"] if o["kind"] == "soft"]
    assert max(soft_naive) > 5 * max(max(soft), 0.1)


def test_dense_track_holds_on_reference_repeat_frames():
    ctx = cadence_ctx()
    out = S.solve_truth(cadence_doc(ctx), ctx)
    frames, xyz = out["dense"]["frames"], np.array(out["dense"]["xyz"])
    rep = next(f for f in range(FA + 1, FB) if f % 6 == 5)
    i = frames.index(rep)
    assert np.allclose(xyz[i], xyz[i - 1], atol=1e-3)
    assert np.isfinite(out["dense"]["speed_m_s"]).all()


def test_projection_uses_each_views_own_instant():
    ctx = cadence_ctx()
    out = S.solve_truth(cadence_doc(ctx), ctx)
    pb = out["projections"]["b"]
    for r, sf, q in zip(pb["frames"], pb["shot_frames"], pb["uv"]):
        if FA + 1 < r < FB - 1 and abs(SHIFT_A[r] - SHIFT_B[sf]) > 0.01:
            want = uv(CAM_B, truth_at(ctx, ctx.obs_time("b", sf)))
            assert np.hypot(*(np.asarray(q) - want)) < 0.5
            break
    else:
        pytest.fail("no out-of-phase frame found")


def test_triangulated_key_time_is_the_mean_of_its_views():
    ctx = cadence_ctx()
    k = tri_key(ctx, "k", 30, np.array([40.0, 25.0, 0.11]))
    t = ctx.key_time(k)
    assert t == pytest.approx(0.5 * (ctx.obs_time("a", 30) + ctx.obs_time("b", 20)))

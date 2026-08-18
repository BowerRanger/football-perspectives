"""W3 (sub-20cm campaign): airborne anchor chains between hard knots are
re-resolved by a gravity-arc fit (pixels are rays, not bucket planes)."""

from __future__ import annotations

import numpy as np
import pytest

from src.schemas.ball_anchor import BallAnchor
from src.utils.camera_projection import project_world_to_image

pytestmark = pytest.mark.unit

_K = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]])
_DIST = (0.0, 0.0)
_R_ = 0.11
_FPS = 30.0


def _cam():
    fwd = np.array([0.0, 20.0, -10.0])
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    C = np.array([0.0, -20.0, 10.0])
    return R, -R @ C


def _uv(world, R, t):
    p = project_world_to_image(_K, R, t, _DIST, np.asarray([world]))[0]
    return (float(p[0]), float(p[1]))


def _arc(p0, v0, f, f0):
    t = (f - f0) / _FPS
    return np.asarray(p0) + np.asarray(v0) * t + 0.5 * np.array(
        [0.0, 0.0, -9.81]) * t * t


def _chain_fixture():
    """Kick at f0=0 (ground), bounce at f=24 (ground), interior airborne
    clicks at the TRUE arc pixels for frames 6/12/18."""
    R, t = _cam()
    p0 = np.array([0.0, 8.0, _R_])
    # Choose v0 so the ball returns to z=R at t=0.8s: vz = g*T/2.
    v0 = np.array([4.0, 3.0, 9.81 * 0.4])
    frames = {}
    anchors = {}
    worlds = {}
    anchors[0] = BallAnchor(frame=0, image_xy=_uv(p0, R, t), state="kick")
    worlds[0] = tuple(p0)
    for f in (6, 12, 18):
        w = _arc(p0, v0, f, 0)
        frames[f] = w
        anchors[f] = BallAnchor(frame=f, image_xy=_uv(w, R, t),
                                state="airborne_low")
        worlds[f] = tuple(np.asarray(
            [w[0], w[1], 1.0]))  # bucket placeholder the fit must replace
    w24 = _arc(p0, v0, 24, 0)
    assert abs(w24[2] - _R_) < 1e-6
    anchors[24] = BallAnchor(frame=24, image_xy=_uv(w24, R, t), state="bounce")
    worlds[24] = tuple(w24)
    cams = {f: (_K, R, t) for f in range(0, 30)}
    return anchors, worlds, frames, cams, R, t


def test_chain_refit_recovers_true_airborne_positions():
    from src.utils.ball_flight_chains import refit_airborne_chains

    anchors, worlds, truth, cams, R, t = _chain_fixture()
    per_K = {f: c[0] for f, c in cams.items()}
    per_R = {f: c[1] for f, c in cams.items()}
    per_t = {f: c[2] for f, c in cams.items()}
    updates, diags = refit_airborne_chains(
        anchor_by_frame=anchors, world_for_anchor=worlds,
        per_frame_K=per_K, per_frame_R=per_R, per_frame_t=per_t,
        distortion=_DIST, fps=_FPS,
    )
    assert set(updates) == {6, 12, 18}
    for f, w_true in truth.items():
        err = np.linalg.norm(np.asarray(updates[f]) - w_true)
        assert err < 0.05, f"frame {f}: {err:.3f}m off the true arc"
    assert any(d.get("kind") == "chain_fit" and d.get("accepted")
               for d in diags)


def test_unbracketed_airborne_run_is_flagged_not_touched():
    from src.utils.ball_flight_chains import refit_airborne_chains

    anchors, worlds, truth, cams, R, t = _chain_fixture()
    anchors.pop(24)   # no trailing hard knot
    worlds.pop(24)
    per_K = {f: c[0] for f, c in cams.items()}
    per_R = {f: c[1] for f, c in cams.items()}
    per_t = {f: c[2] for f, c in cams.items()}
    updates, diags = refit_airborne_chains(
        anchor_by_frame=anchors, world_for_anchor=worlds,
        per_frame_K=per_K, per_frame_R=per_R, per_frame_t=per_t,
        distortion=_DIST, fps=_FPS,
    )
    assert updates == {}
    assert any(d.get("kind") == "underconstrained_chain" for d in diags)


def test_single_corrupt_click_does_not_collapse_the_chain():
    """Median-robust acceptance: one bad interior click must not force the
    whole chain back to bucket heights — the arc follows the good anchors
    (the corrupt anchor's keyframe is later ray-snapped, operator wins)."""
    from src.utils.ball_flight_chains import refit_airborne_chains

    anchors, worlds, truth, cams, R, t = _chain_fixture()
    bad = anchors[12]
    anchors[12] = BallAnchor(frame=12, state="airborne_low",
                             image_xy=(bad.image_xy[0] + 400.0,
                                       bad.image_xy[1] + 250.0))
    per_K = {f: c[0] for f, c in cams.items()}
    per_R = {f: c[1] for f, c in cams.items()}
    per_t = {f: c[2] for f, c in cams.items()}
    updates, diags = refit_airborne_chains(
        anchor_by_frame=anchors, world_for_anchor=worlds,
        per_frame_K=per_K, per_frame_R=per_R, per_frame_t=per_t,
        distortion=_DIST, fps=_FPS,
    )
    assert set(updates) == {6, 12, 18}
    for f in (6, 18):        # good anchors recover truth
        assert np.linalg.norm(np.asarray(updates[f]) - truth[f]) < 0.05
    assert any(d.get("kind") == "chain_fit" and d.get("accepted")
               for d in diags)


def test_resolve_events_dense_track_follows_fitted_arc():
    from src.utils.ball_event_resolver import resolve_events

    anchors, worlds, truth, cams, R, t = _chain_fixture()
    per_K = {f: c[0] for f, c in cams.items()}
    per_R = {f: c[1] for f, c in cams.items()}
    per_t = {f: c[2] for f, c in cams.items()}

    class _NoCtx:
        def joint_world(self, frame, player_id, bone):
            return None

    res = resolve_events(
        anchor_by_frame=anchors, player_ctx=_NoCtx(),
        per_frame_K=per_K, per_frame_R=per_R, per_frame_t=per_t,
        distortion=_DIST, ball_radius=_R_, goal_geometry=None,
        n_frames=30, fps=_FPS, clip_id="c", image_size=(1920, 1080),
    )
    p0 = np.array([0.0, 8.0, _R_])
    v0 = np.array([4.0, 3.0, 9.81 * 0.4])
    for f in (3, 9, 15, 21):     # between-keyframe frames
        w, _conf = res.world_by_frame[f]
        err = np.linalg.norm(np.asarray(w) - _arc(p0, v0, f, 0))
        assert err < 0.10, f"frame {f}: {err:.3f}m off the true arc"


def test_chain_refit_uses_extra_detection_observations():
    """Hold-out scenario: interior airborne ANCHORS absent, but real
    detections along the arc still determine the fit."""
    from src.utils.ball_flight_chains import refit_airborne_chains

    anchors, worlds, truth, cams, R, t = _chain_fixture()
    # Keep only ONE interior anchor (frame 12); frames 6/18 become
    # detections instead.
    extra = {}
    for f in (6, 18):
        w = truth[f]
        extra[f] = _uv(w, R, t)
        anchors.pop(f)
        worlds.pop(f)
    per_K = {f: c[0] for f, c in cams.items()}
    per_R = {f: c[1] for f, c in cams.items()}
    per_t = {f: c[2] for f, c in cams.items()}
    updates, diags = refit_airborne_chains(
        anchor_by_frame=anchors, world_for_anchor=worlds,
        per_frame_K=per_K, per_frame_R=per_R, per_frame_t=per_t,
        distortion=_DIST, fps=_FPS, extra_observations=extra,
    )
    assert set(updates) == {12}
    err = np.linalg.norm(np.asarray(updates[12]) - truth[12])
    assert err < 0.05
    fit = next(d for d in diags if d.get("kind") == "chain_fit")
    assert fit["accepted"] and fit.get("n_extra_obs", 0) == 2


def test_poisoned_extra_observations_do_not_break_the_fit():
    """W5b: junk in-span detections must neither poison the arc nor force
    a bucket fallback — the anchors-only fit is the floor."""
    from src.utils.ball_flight_chains import refit_airborne_chains

    anchors, worlds, truth, cams, R, t = _chain_fixture()
    extra = {}
    for f in (7, 13, 19):        # true arc pixels — good extras
        extra[f] = _uv(_arc(np.array([0.0, 8.0, _R_]),
                            np.array([4.0, 3.0, 9.81 * 0.4]), f, 0), R, t)
    for f, off in ((9, 260.0), (15, -310.0), (21, 280.0)):   # junk extras
        w = _arc(np.array([0.0, 8.0, _R_]),
                 np.array([4.0, 3.0, 9.81 * 0.4]), f, 0)
        u, v = _uv(w, R, t)
        extra[f] = (u + off, v + off / 2)
    per_K = {f: c[0] for f, c in cams.items()}
    per_R = {f: c[1] for f, c in cams.items()}
    per_t = {f: c[2] for f, c in cams.items()}
    updates, diags = refit_airborne_chains(
        anchor_by_frame=anchors, world_for_anchor=worlds,
        per_frame_K=per_K, per_frame_R=per_R, per_frame_t=per_t,
        distortion=_DIST, fps=_FPS, extra_observations=extra,
    )
    assert set(updates) == {6, 12, 18}
    for f, w_true in truth.items():
        assert np.linalg.norm(np.asarray(updates[f]) - w_true) < 0.05
    fit = next(d for d in diags if d.get("kind") == "chain_fit")
    assert fit["accepted"]
    assert fit.get("n_extra_used", 99) <= 3   # junk extras excluded


def test_auto_touch_knots_are_soft_manual_are_hard():
    """W5c: an auto touch's body-pin can carry attribution error — the fit
    may pull away from it toward the ray evidence; manual knots stay hard."""
    from src.utils.ball_flight_chains import refit_airborne_chains

    anchors, worlds, truth, cams, R, t = _chain_fixture()
    # Corrupt the END knot world (as a wrong-player auto body-pin would):
    # 2m off the true bounce point, but its CLICK pixel stays correct.
    w24_true = np.asarray(worlds[24])
    worlds[24] = tuple(w24_true + np.array([2.0, -1.5, 0.0]))
    per_K = {f: c[0] for f, c in cams.items()}
    per_R = {f: c[1] for f, c in cams.items()}
    per_t = {f: c[2] for f, c in cams.items()}
    # Hard everywhere (legacy): the corrupt knot bends the arc.
    hard_updates, _ = refit_airborne_chains(
        anchor_by_frame=anchors, world_for_anchor=worlds,
        per_frame_K=per_K, per_frame_R=per_R, per_frame_t=per_t,
        distortion=_DIST, fps=_FPS,
        manual_frames=frozenset(anchors),
    )
    # End knot marked AUTO (soft): the ray observations win; interior
    # airborne positions return to the true arc.
    soft_updates, diags = refit_airborne_chains(
        anchor_by_frame=anchors, world_for_anchor=worlds,
        per_frame_K=per_K, per_frame_R=per_R, per_frame_t=per_t,
        distortion=_DIST, fps=_FPS,
        manual_frames=frozenset(f for f in anchors if f != 24),
    )
    assert set(soft_updates) == {6, 12, 18}
    err_soft = max(np.linalg.norm(np.asarray(soft_updates[f]) - truth[f])
                   for f in truth)
    err_hard = max(np.linalg.norm(np.asarray(hard_updates[f]) - truth[f])
                   for f in truth) if hard_updates else float("inf")
    assert err_soft < 0.30
    assert err_soft < err_hard


def test_ballistic_segment_refit_from_detections_fixes_soft_endpoint():
    """W5l: a ballistic SEGMENT with no interior anchors (kroupi's deep
    crosses) refits from in-span detections; a wrong AUTO endpoint (soft)
    is overridden by the ray evidence."""
    from src.utils.ball_flight_chains import refit_ballistic_segment

    R, t = _cam()
    p0 = np.array([0.0, 8.0, _R_])
    v0 = np.array([4.0, 3.0, 9.81 * 0.4])
    end_true = _arc(p0, v0, 24, 0)
    obs = {f: _uv(_arc(p0, v0, f, 0), R, t) for f in range(3, 24, 3)}
    per = {f: (_K, R, t) for f in range(0, 30)}
    # Auto end keyframe body-pinned 2m off the true landing.
    end_bad = tuple(end_true + np.array([2.0, -1.5, 0.0]))
    fit = refit_ballistic_segment(
        start_frame=0, end_frame=24,
        start_world=tuple(p0), end_world=end_bad,
        start_is_manual=True, end_is_manual=False,
        end_confidence=0.8,
        extra_observations=obs,
        per_frame_K={f: c[0] for f, c in per.items()},
        per_frame_R={f: c[1] for f, c in per.items()},
        per_frame_t={f: c[2] for f, c in per.items()},
        distortion=_DIST, fps=_FPS,
    )
    assert fit is not None
    fp0, fv0 = fit
    for f in (6, 12, 18):
        w = np.asarray(fp0) + np.asarray(fv0) * (f / _FPS) \
            + 0.5 * np.array([0, 0, -9.81]) * (f / _FPS) ** 2
        err = np.linalg.norm(w - _arc(p0, v0, f, 0))
        assert err < 0.30, f"f{f}: {err:.2f}m"


def test_ballistic_segment_refit_needs_enough_detections():
    from src.utils.ball_flight_chains import refit_ballistic_segment

    R, t = _cam()
    p0 = np.array([0.0, 8.0, _R_])
    v0 = np.array([4.0, 3.0, 9.81 * 0.4])
    per = {f: (_K, R, t) for f in range(0, 30)}
    fit = refit_ballistic_segment(
        start_frame=0, end_frame=24,
        start_world=tuple(p0), end_world=tuple(_arc(p0, v0, 24, 0)),
        start_is_manual=True, end_is_manual=True, end_confidence=1.0,
        extra_observations={9: _uv(_arc(p0, v0, 9, 0), R, t)},  # just one
        per_frame_K={f: c[0] for f, c in per.items()},
        per_frame_R={f: c[1] for f, c in per.items()},
        per_frame_t={f: c[2] for f, c in per.items()},
        distortion=_DIST, fps=_FPS,
    )
    assert fit is None

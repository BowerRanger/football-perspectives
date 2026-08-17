"""W4 (sub-20cm campaign): the derived dense track is constrained by real
detection evidence between keyframes, and free-flight spans obey gravity."""

from __future__ import annotations

import numpy as np
import pytest

from src.schemas.ball_keyframes import BallKeyframe, BallKeyframeSet, BallSegment

pytestmark = pytest.mark.unit

_FPS = 30.0


def _kf(frame, world, state="grounded"):
    return BallKeyframe(frame=frame, state=state, depth_source="ground",
                        world_xyz=tuple(world))


def test_free_flight_with_two_endpoints_renders_gravity_arc():
    from src.utils.ball_interpolate import interpolate_events

    p0 = np.array([0.0, 0.0, 2.0])
    p1 = np.array([6.0, 0.0, 2.0])
    ks = BallKeyframeSet(
        clip_id="c", fps=_FPS, image_size=(1920, 1080),
        keyframes=(_kf(0, p0, "off_screen_flight"), _kf(20, p1, "grounded")),
        segments=(BallSegment(start_frame=0, end_frame=20, kind="free_flight",
                              hints={"gravity": -9.81}),),
    )
    track = interpolate_events(ks, n_frames=21)
    mid = track.frames[10].world_xyz
    # Gravity arc through both endpoints has apex above the endpoints'
    # heights at the midpoint — a linear render would give z == 2.0.
    T = 20 / _FPS
    apex_lift = 9.81 * T * T / 8.0
    assert mid is not None
    assert abs(mid[2] - (2.0 + apex_lift)) < 1e-6


def test_roll_span_follows_evidence_worlds():
    from src.utils.ball_interpolate import interpolate_events

    # True path: quarter-circle curve; linear rendering cuts the corner.
    n = 25
    ang = np.linspace(0.0, np.pi / 2, n)
    truth = {f: (5.0 * np.sin(ang[f]), 5.0 - 5.0 * np.cos(ang[f]), 0.11)
             for f in range(n)}
    ks = BallKeyframeSet(
        clip_id="c", fps=_FPS, image_size=(1920, 1080),
        keyframes=(_kf(0, truth[0]), _kf(n - 1, truth[n - 1])),
        segments=(BallSegment(start_frame=0, end_frame=n - 1, kind="roll",
                              hints={}),),
    )
    plain = interpolate_events(ks, n_frames=n)
    evidenced = interpolate_events(
        ks, n_frames=n,
        evidence_worlds={f: truth[f] for f in range(2, n - 2, 2)},
    )
    err_plain = max(np.linalg.norm(np.asarray(plain.frames[f].world_xyz)
                                   - truth[f]) for f in range(n))
    err_ev = max(np.linalg.norm(np.asarray(evidenced.frames[f].world_xyz)
                                - truth[f]) for f in range(n))
    assert err_plain > 0.5          # linear corner-cut is large
    assert err_ev < 0.15            # evidence-following stays on the curve
    # Endpoints remain exact.
    assert np.allclose(evidenced.frames[0].world_xyz, truth[0], atol=1e-9)
    assert np.allclose(evidenced.frames[n - 1].world_xyz, truth[n - 1],
                       atol=1e-9)


def test_evidence_is_median_filtered_against_jitter():
    from src.utils.ball_interpolate import interpolate_events

    n = 21
    truth = {f: (0.3 * f, 0.0, 0.11) for f in range(n)}
    noisy = {}
    rng_vals = [0.15, -0.12, 0.18, -0.2, 0.1, -0.15, 0.12, -0.1, 0.2, -0.18]
    for i, f in enumerate(range(1, n - 1, 2)):
        w = np.asarray(truth[f], dtype=float)
        w[1] += rng_vals[i % len(rng_vals)]   # lateral jitter ±20 cm
        noisy[f] = tuple(w)
    ks = BallKeyframeSet(
        clip_id="c", fps=_FPS, image_size=(1920, 1080),
        keyframes=(_kf(0, truth[0]), _kf(n - 1, truth[n - 1])),
        segments=(BallSegment(start_frame=0, end_frame=n - 1, kind="roll",
                              hints={}),),
    )
    track = interpolate_events(ks, n_frames=n, evidence_worlds=noisy)
    # Median-of-window knots suppress single-frame jitter: the rendered
    # path stays clearly tighter than the raw noise amplitude.
    errs = [np.linalg.norm(np.asarray(track.frames[f].world_xyz)
                           - np.asarray(truth[f])) for f in range(n)]
    assert max(errs) < 0.15


def test_carry_span_follows_player_path():
    from src.utils.ball_interpolate import interpolate_events

    # Dribbler runs a curve; ball must stay with the feet, not cut the chord.
    n = 21
    ang = np.linspace(0.0, np.pi / 2, n)
    foot = {f: np.array([4.0 * np.sin(ang[f]), 4.0 - 4.0 * np.cos(ang[f]),
                         0.11]) for f in range(n)}
    p0 = foot[0] + np.array([0.10, 0.0, 0.0])     # ball slightly off-foot
    p1 = foot[n - 1] + np.array([0.0, 0.10, 0.0])
    carry_worlds = {}
    for f in range(1, n - 1):
        s = f / (n - 1)
        off = (1 - s) * np.array([0.10, 0.0, 0.0]) + s * np.array([0.0, 0.10, 0.0])
        carry_worlds[f] = tuple(foot[f] + off)
    ks = BallKeyframeSet(
        clip_id="c", fps=_FPS, image_size=(1920, 1080),
        keyframes=(_kf(0, p0, "player_touch"), _kf(n - 1, p1, "player_touch")),
        segments=(BallSegment(start_frame=0, end_frame=n - 1, kind="carry",
                              hints={"player_id": "P1"}),),
    )
    plain = interpolate_events(ks, n_frames=n)
    followed = interpolate_events(ks, n_frames=n, carry_worlds=carry_worlds)
    truth_mid = carry_worlds[10]
    err_plain = np.linalg.norm(np.asarray(plain.frames[10].world_xyz)
                               - np.asarray(truth_mid))
    err_follow = np.linalg.norm(np.asarray(followed.frames[10].world_xyz)
                                - np.asarray(truth_mid))
    assert err_plain > 0.5
    assert err_follow < 1e-9        # carry worlds are used verbatim
    assert np.allclose(followed.frames[0].world_xyz, p0)
    assert np.allclose(followed.frames[n - 1].world_xyz, p1)


def test_roll_evidence_far_from_endpoint_chord_is_ignored():
    """A false-detection cluster (detector locked on a static object) must
    not drag the roll off the endpoint chord (gberch f164-176 regression)."""
    from src.utils.ball_interpolate import interpolate_events

    n = 13
    a = np.array([32.1, 12.2, 0.11])
    b = np.array([26.3, 7.25, 0.11])
    truth = {f: tuple(a + (b - a) * (f / (n - 1))) for f in range(n)}
    # Junk evidence ~4.5m off the chord at mid-span.
    junk = {f: (35.1, 15.5, 0.11) for f in range(3, 10, 2)}
    ks = BallKeyframeSet(
        clip_id="c", fps=_FPS, image_size=(1920, 1080),
        keyframes=(_kf(0, a), _kf(n - 1, b)),
        segments=(BallSegment(start_frame=0, end_frame=n - 1, kind="roll",
                              hints={}),),
    )
    track = interpolate_events(ks, n_frames=n, evidence_worlds=junk)
    errs = [np.linalg.norm(np.asarray(track.frames[f].world_xyz)
                           - np.asarray(truth[f])) for f in range(n)]
    assert max(errs) < 0.05      # junk ignored → straight chord retained

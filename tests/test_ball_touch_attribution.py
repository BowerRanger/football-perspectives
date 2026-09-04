"""Touch bone-attribution refinement: relabel to the ray-closest joint,
keep originals on ambiguity, never add/remove/re-frame events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.utils.ball_auto_events import BallEvent
from src.utils.ball_touch_attribution import (
    TouchAttributionCfg,
    refine_touch_attribution,
)

CFG = TouchAttributionCfg(enabled=True)


def _camera():
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _project(p, K, R, t):
    cam = R @ np.asarray(p, dtype=float) + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


@dataclass(frozen=True)
class _Joint:
    player_id: str
    bone: str
    world_xyz: tuple[float, float, float]
    uv: tuple[float, float] | None
    confidence: float


class _Ctx:
    """PlayerContext stub: fixed joints at every frame."""

    def __init__(self, joints):
        self._joints = joints

    def joints_at(self, frame):
        return list(self._joints)


def _setup(ball_world=(40.0, 34.0, 0.11)):
    K, R, t = _camera()
    ball_uv = _project(np.array(ball_world), K, R, t)
    joints = [
        # l_foot right AT the ball; r_foot 0.8 m away.
        _Joint("P001", "l_foot", (40.0, 34.0, 0.11),
               _project(np.array([40.0, 34.0, 0.11]), K, R, t), 0.9),
        _Joint("P001", "r_foot", (40.8, 34.0, 0.11),
               _project(np.array([40.8, 34.0, 0.11]), K, R, t), 0.9),
    ]
    frames = range(8, 13)
    return (
        _Ctx(joints),
        {f: np.asarray(ball_uv) for f in frames},
        {f: K for f in frames}, {f: R for f in frames}, {f: t for f in frames},
    )


def _refine(events, ctx, uvs, Ks, Rs, ts, cfg=CFG):
    return refine_touch_attribution(
        events, player_ctx=ctx, ball_uvs=uvs,
        per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
        distortion=(0.0, 0.0), cfg=cfg,
    )


def test_wrong_bone_relabelled_to_ray_closest_joint():
    ctx, uvs, Ks, Rs, ts = _setup()
    events = (BallEvent(frame=10, kind="touch", score=0.7,
                        player_id="P001", bone="r_foot"),)
    out = _refine(events, ctx, uvs, Ks, Rs, ts)
    assert len(out) == 1
    assert out[0].bone == "l_foot"
    assert out[0].player_id == "P001"
    assert out[0].frame == 10 and out[0].kind == "touch"
    assert out[0].score == pytest.approx(0.7)


def test_ambiguous_margin_keeps_original():
    # Both feet equidistant-ish: margin gate keeps the original label.
    K, R, t = _camera()
    ball_uv = _project(np.array([40.0, 34.0, 0.11]), K, R, t)
    joints = [
        _Joint("P001", "l_foot", (40.02, 34.0, 0.11),
               _project(np.array([40.02, 34.0, 0.11]), K, R, t), 0.9),
        _Joint("P001", "r_foot", (40.05, 34.0, 0.11),
               _project(np.array([40.05, 34.0, 0.11]), K, R, t), 0.9),
    ]
    ctx = _Ctx(joints)
    uvs = {10: np.asarray(ball_uv)}
    events = (BallEvent(frame=10, kind="touch", score=0.7,
                        player_id="P001", bone="r_foot"),)
    out = refine_touch_attribution(
        events, player_ctx=ctx, ball_uvs=uvs,
        per_frame_K={10: K}, per_frame_R={10: R}, per_frame_t={10: t},
        distortion=(0.0, 0.0), cfg=CFG,
    )
    assert out[0].bone == "r_foot"


def test_far_ball_no_candidate_keeps_original():
    ctx, uvs, Ks, Rs, ts = _setup(ball_world=(60.0, 10.0, 0.11))
    events = (BallEvent(frame=10, kind="touch", score=0.7,
                        player_id="P001", bone="r_foot"),)
    out = _refine(events, ctx, uvs, Ks, Rs, ts)
    assert out[0].bone == "r_foot"  # best gap exceeds max_gap_m -> unchanged


def test_non_touch_events_and_order_preserved():
    ctx, uvs, Ks, Rs, ts = _setup()
    events = (
        BallEvent(frame=5, kind="bounce", score=0.6),
        BallEvent(frame=10, kind="touch", score=0.7,
                  player_id="P001", bone="r_foot"),
        BallEvent(frame=20, kind="goal_impact", score=0.9,
                  goal_element="post"),
    )
    out = _refine(events, ctx, uvs, Ks, Rs, ts)
    assert [e.kind for e in out] == ["bounce", "touch", "goal_impact"]
    assert out[0] == events[0] and out[2] == events[2]


def test_no_ball_uv_in_window_keeps_original():
    ctx, _uvs, Ks, Rs, ts = _setup()
    events = (BallEvent(frame=10, kind="touch", score=0.7,
                        player_id="P001", bone="r_foot"),)
    out = _refine(events, ctx, {}, Ks, Rs, ts)
    assert out[0].bone == "r_foot"


def test_disabled_is_identity():
    ctx, uvs, Ks, Rs, ts = _setup()
    events = (BallEvent(frame=10, kind="touch", score=0.7,
                        player_id="P001", bone="r_foot"),)
    out = _refine(events, ctx, uvs, Ks, Rs, ts,
                  cfg=TouchAttributionCfg(enabled=False))
    assert out == events


def test_config_block_keys():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(
        Path("config/default.yaml").read_text())["ball"]["touch_attribution"]
    assert cfg["enabled"] is True
    assert cfg["window"] == 2
    assert cfg["max_gap_m"] == 0.45
    assert cfg["margin_m"] == 0.05


def test_depth_consistency_breaks_ray_gap_tie(  # W5d: depth-blind gate fix
):
    """The kicker's foot can sit nearer the camera-ball RAY than the true
    toucher's knee while being metres off in DEPTH along it. With an
    expected ball world supplied, the depth-consistent joint wins."""
    K, R, t = _camera()
    true_ball = np.array([52.5, 30.0, 0.6])       # ball at the knee
    uv = _project(true_ball, K, R, t)
    C = -R.T @ t
    ray = true_ball - C
    ray /= np.linalg.norm(ray)
    # Kicker's foot: exactly ON the ray, 5m closer to the camera (gap ≈ 0).
    kicker_foot = tuple(true_ball - 5.0 * ray)
    knee = tuple(true_ball + np.array([0.12, 0.05, 0.0]))
    ctx = _Ctx([
        _Joint("P014", "r_foot", kicker_foot, None, 0.9),
        _Joint("P008", "r_knee", knee, None, 0.9),
    ])
    events = (BallEvent(frame=10, kind="touch", score=0.8,
                        player_id="P014", bone="r_foot"),)
    common = dict(
        player_ctx=ctx, ball_uvs={10: np.asarray(uv)},
        per_frame_K={10: K}, per_frame_R={10: R}, per_frame_t={10: t},
        distortion=(0.0, 0.0),
    )
    # Without depth info the near-ray kicker foot keeps the label.
    out_blind = refine_touch_attribution(events, cfg=CFG, **common)
    assert (out_blind[0].player_id, out_blind[0].bone) == ("P014", "r_foot")
    # With the expected ball world, the depth-consistent knee wins.
    out_depth = refine_touch_attribution(
        events, cfg=CFG,
        expected_world_by_frame={10: tuple(true_ball)}, **common)
    assert (out_depth[0].player_id, out_depth[0].bone) == ("P008", "r_knee")
    assert out_depth[0].frame == 10 and len(out_depth) == 1


def test_expected_worlds_interpolate_between_ground_anchors():
    from src.schemas.ball_anchor import BallAnchor
    from src.utils.ball_touch_attribution import expected_ball_worlds

    K, R, t = _camera()
    a = np.array([40.0, 30.0, 0.11])
    b = np.array([46.0, 33.0, 0.11])
    anchors = {
        10: BallAnchor(frame=10, image_xy=_project(a, K, R, t),
                       state="grounded"),
        20: BallAnchor(frame=20, image_xy=_project(b, K, R, t),
                       state="grounded"),
        15: BallAnchor(frame=15, image_xy=None, state="off_screen_flight"),
    }
    worlds = expected_ball_worlds(
        anchors, per_frame_K={f: K for f in range(30)},
        per_frame_R={f: R for f in range(30)},
        per_frame_t={f: t for f in range(30)},
        distortion=(0.0, 0.0), ball_radius=0.11)
    assert np.allclose(worlds[10], a, atol=1e-6)
    assert np.allclose(worlds[20], b, atol=1e-6)
    mid = np.asarray(worlds[15])
    assert np.allclose(mid, (a + b) / 2, atol=0.05)
    assert 25 not in worlds        # no extrapolation past the last anchor


class TestRankedCandidatesCorroboration:
    """W3 (foot-contact locomotion regression recovery): when the
    depth-blind ray-gap check above doesn't clear its own margin, a
    genuinely corroborated alternate from generate_auto_anchors's
    minting-time candidate pool (persisted to the diag sidecar and
    passed in here as ``ranked_candidates``) can still flip the label —
    an independent second opinion, not just a wider search window."""

    def _near_tie_scene(self):
        K, R, t = _camera()
        ball_uv = _project(np.array([40.0, 34.0, 0.11]), K, R, t)
        joints = [
            _Joint("P001", "l_foot", (40.02, 34.0, 0.11),
                   _project(np.array([40.02, 34.0, 0.11]), K, R, t), 0.9),
            _Joint("P001", "r_foot", (40.05, 34.0, 0.11),
                   _project(np.array([40.05, 34.0, 0.11]), K, R, t), 0.9),
        ]
        return _Ctx(joints), {10: np.asarray(ball_uv)}, K, R, t

    def test_corroborated_alternate_flips_when_standard_check_stays_tied(self):
        ctx, uvs, K, R, t = self._near_tie_scene()
        events = (BallEvent(frame=10, kind="touch", score=0.7,
                            player_id="P001", bone="r_foot"),)
        # Standard ray-gap check alone stays inside the ambiguity margin
        # (see test_ambiguous_margin_keeps_original) and keeps r_foot.
        baseline = refine_touch_attribution(
            events, player_ctx=ctx, ball_uvs=uvs,
            per_frame_K={10: K}, per_frame_R={10: R}, per_frame_t={10: t},
            distortion=(0.0, 0.0), cfg=CFG,
        )
        assert baseline[0].bone == "r_foot"
        # But the minting-time candidate pool shows l_foot with a
        # meaningfully smaller gap than r_foot ever achieved there.
        ranked = {10: [
            {"player_id": "P001", "bone": "r_foot", "gap_m": 0.30, "score": 0.8},
            {"player_id": "P001", "bone": "l_foot", "gap_m": 0.10, "score": 0.6},
        ]}
        out = refine_touch_attribution(
            events, player_ctx=ctx, ball_uvs=uvs,
            per_frame_K={10: K}, per_frame_R={10: R}, per_frame_t={10: t},
            distortion=(0.0, 0.0), cfg=CFG, ranked_candidates=ranked,
        )
        assert out[0].bone == "l_foot"
        assert out[0].player_id == "P001"
        assert out[0].frame == 10 and len(out) == 1

    def test_current_label_gap_is_the_true_window_minimum(self):
        """The current label's own best gap must be the MIN across the
        whole window, not whatever occurrence the scan happens to see
        first. A same-frame occurrence at ``e.frame`` overrides an
        earlier off-frame sighting correctly, but a LATER off-frame
        sighting with a smaller gap than the first must still count —
        otherwise a stale, larger "first seen" gap makes a mediocre
        alternate look like an improvement it isn't."""
        ctx, uvs, K, R, t = self._near_tie_scene()
        events = (BallEvent(frame=10, kind="touch", score=0.7,
                            player_id="P001", bone="r_foot"),)
        ranked = {
            # Current label (r_foot) sighted twice off-frame: an early,
            # mediocre gap at frame 8, then a much tighter gap at frame
            # 11 — the true window-best for the current label is 0.05.
            8: [{"player_id": "P001", "bone": "r_foot",
                 "gap_m": 0.40, "score": 0.5}],
            11: [{"player_id": "P001", "bone": "r_foot",
                  "gap_m": 0.05, "score": 0.5}],
            # Alternate beats the naive "first sighting" (0.40) but NOT
            # the true best (0.05) — must not corroborate.
            10: [{"player_id": "P001", "bone": "l_foot",
                  "gap_m": 0.20, "score": 0.9}],
        }
        out = refine_touch_attribution(
            events, player_ctx=ctx, ball_uvs=uvs,
            per_frame_K={10: K}, per_frame_R={10: R}, per_frame_t={10: t},
            distortion=(0.0, 0.0), cfg=CFG, ranked_candidates=ranked,
        )
        assert out[0].bone == "r_foot"

    def test_ranked_candidates_never_override_an_already_confident_relabel(self):
        ctx, uvs, Ks, Rs, ts = _setup()  # l_foot right AT the ball
        events = (BallEvent(frame=10, kind="touch", score=0.7,
                            player_id="P001", bone="r_foot"),)
        # Contradicts the standard result — must be ignored since the
        # depth-blind check already relabelled confidently.
        ranked = {10: [
            {"player_id": "P001", "bone": "r_foot", "gap_m": 0.01, "score": 0.9},
        ]}
        out = refine_touch_attribution(
            events, player_ctx=ctx, ball_uvs=uvs,
            per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
            distortion=(0.0, 0.0), cfg=CFG, ranked_candidates=ranked,
        )
        assert out[0].bone == "l_foot"

    def test_alternate_over_max_gap_m_never_corroborates(self):
        ctx, uvs, K, R, t = self._near_tie_scene()
        events = (BallEvent(frame=10, kind="touch", score=0.7,
                            player_id="P001", bone="r_foot"),)
        ranked = {10: [
            {"player_id": "P001", "bone": "r_foot", "gap_m": 0.90, "score": 0.8},
            # Beats r_foot's gap but exceeds max_gap_m (0.45) itself.
            {"player_id": "P001", "bone": "l_foot", "gap_m": 0.60, "score": 0.6},
        ]}
        out = refine_touch_attribution(
            events, player_ctx=ctx, ball_uvs=uvs,
            per_frame_K={10: K}, per_frame_R={10: R}, per_frame_t={10: t},
            distortion=(0.0, 0.0), cfg=CFG, ranked_candidates=ranked,
        )
        assert out[0].bone == "r_foot"

    def test_consider_ranked_candidates_false_disables_corroboration(self):
        ctx, uvs, K, R, t = self._near_tie_scene()
        events = (BallEvent(frame=10, kind="touch", score=0.7,
                            player_id="P001", bone="r_foot"),)
        ranked = {10: [
            {"player_id": "P001", "bone": "r_foot", "gap_m": 0.30, "score": 0.8},
            {"player_id": "P001", "bone": "l_foot", "gap_m": 0.10, "score": 0.6},
        ]}
        cfg = TouchAttributionCfg(enabled=True, consider_ranked_candidates=False)
        out = refine_touch_attribution(
            events, player_ctx=ctx, ball_uvs=uvs,
            per_frame_K={10: K}, per_frame_R={10: R}, per_frame_t={10: t},
            distortion=(0.0, 0.0), cfg=cfg, ranked_candidates=ranked,
        )
        assert out[0].bone == "r_foot"

    def test_config_block_has_consider_ranked_candidates(self):
        import yaml
        from pathlib import Path
        cfg = yaml.safe_load(
            Path("config/default.yaml").read_text())["ball"]["touch_attribution"]
        assert cfg["consider_ranked_candidates"] is True


class TestCrossPlayerPhysicsGuard:
    """Touch-gate calibration follow-up (2026-09-03): both relabel paths
    compare raw bone<->ball-ray gaps only — depth-blind AND kink-blind. A
    bystander whose torso/hand sits nearer the ball's pixel ray than the
    true toucher's (motion-blurred, FK-noisy) foot can win purely on
    geometry; when the winner is a DIFFERENT PLAYER this silently
    reassigns the touch (gberch f343 in production: a correctly
    player-attributed-but-wrong-bone event on P006 flips to bystander
    P009 sitting on the ball's ray during a crowded moment). The guard
    reuses ball_auto_anchor's minting-time physics-consistency term
    (direction + kink of the ball's OWN pixel path) to veto a cross-player
    flip the ball's own trajectory doesn't corroborate."""

    def _kink_scene(self):
        """Ball pixel path with a sharp reversal at frame 10 (a genuine
        touch signature) then a smooth glide onward with no further kink
        (frames 11-13 keep decelerating in the same new direction)."""
        K, R, t = _camera()
        path_x = {7: 20.0, 8: 25.0, 9: 28.0, 10: 30.0,
                  11: 27.0, 12: 24.0, 13: 21.0, 14: 18.0, 15: 15.0}
        ball_uvs = {f: np.asarray(_project((x, 34.0, 0.11), K, R, t))
                    for f, x in path_x.items()}
        return K, R, t, ball_uvs

    def test_blocks_flip_to_bystander_on_smooth_glide(self):
        """Candidate B sits exactly on the ball's ray at frame 12 — deep
        into the post-touch glide, no kink there — while candidate A (the
        current label) sits near the frame-10 reversal itself. Raw ray
        gap alone prefers B; the physics guard restores A."""
        K, R, t, ball_uvs = self._kink_scene()
        a_world = (30.15, 34.0, 0.11)   # near the frame-10 reversal
        b_world = (24.0, 34.0, 0.11)    # exactly on-ray at frame 12 (no kink)

        class _Ctx2:
            def joints_at(self, frame):
                return [
                    _Joint("P001", "r_knee", a_world,
                           _project(a_world, K, R, t), 0.9),
                    _Joint("P002", "chest", b_world,
                           _project(b_world, K, R, t), 0.9),
                ]

        ctx = _Ctx2()
        Ks = {f: K for f in ball_uvs}
        Rs = {f: R for f in ball_uvs}
        ts = {f: t for f in ball_uvs}
        events = (BallEvent(frame=10, kind="touch", score=0.8,
                            player_id="P001", bone="r_knee"),)

        # Guard off: raw gap alone wins -> flips to the bystander.
        out_off = refine_touch_attribution(
            events, player_ctx=ctx, ball_uvs=ball_uvs,
            per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
            distortion=(0.0, 0.0),
            cfg=TouchAttributionCfg(enabled=True,
                                     cross_player_physics_guard=False))
        assert (out_off[0].player_id, out_off[0].bone) == ("P002", "chest")

        # Guard on: the ball's own trajectory doesn't corroborate the
        # bystander (no kink at its frame) -> stays with the real toucher.
        out_on = refine_touch_attribution(
            events, player_ctx=ctx, ball_uvs=ball_uvs,
            per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
            distortion=(0.0, 0.0),
            cfg=TouchAttributionCfg(enabled=True,
                                     cross_player_physics_guard=True))
        assert (out_on[0].player_id, out_on[0].bone) == ("P001", "r_knee")

    def test_allows_flip_when_alternate_is_also_at_the_kink(self):
        """Both candidates sample the SAME frame (the real reversal) —
        the alternate is just as physically corroborated as the current
        label, so the guard must not block a raw-gap-driven flip it has
        no genuine grounds to veto."""
        K, R, t, ball_uvs = self._kink_scene()
        a_world = (30.25, 34.0, 0.11)   # near frame-10, bigger offset
        b_world = (30.05, 34.0, 0.11)   # near frame-10, smaller offset

        class _Ctx3:
            def joints_at(self, frame):
                return [
                    _Joint("P001", "r_knee", a_world,
                           _project(a_world, K, R, t), 0.9),
                    _Joint("P002", "chest", b_world,
                           _project(b_world, K, R, t), 0.9),
                ]

        ctx = _Ctx3()
        Ks = {f: K for f in ball_uvs}
        Rs = {f: R for f in ball_uvs}
        ts = {f: t for f in ball_uvs}
        events = (BallEvent(frame=10, kind="touch", score=0.8,
                            player_id="P001", bone="r_knee"),)
        out = refine_touch_attribution(
            events, player_ctx=ctx, ball_uvs=ball_uvs,
            per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
            distortion=(0.0, 0.0),
            cfg=TouchAttributionCfg(enabled=True,
                                     cross_player_physics_guard=True))
        assert (out[0].player_id, out[0].bone) == ("P002", "chest")

    def test_same_player_bone_flip_never_gated(self):
        """The guard only ever engages on a DIFFERENT player — a same-
        player bone correction (the common case) must be unaffected even
        when the ball track carries no kink at all near the event."""
        ctx, uvs, Ks, Rs, ts = _setup()  # l_foot right AT the ball
        events = (BallEvent(frame=10, kind="touch", score=0.7,
                            player_id="P001", bone="r_foot"),)
        out = refine_touch_attribution(
            events, player_ctx=ctx, ball_uvs=uvs,
            per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
            distortion=(0.0, 0.0),
            cfg=TouchAttributionCfg(enabled=True,
                                     cross_player_physics_guard=True))
        assert (out[0].player_id, out[0].bone) == ("P001", "l_foot")

    def test_no_ball_track_signal_falls_back_to_unguarded_behaviour(self):
        """When the physics term can't be computed for either side (ball
        track too sparse for the velocity window), the guard must not
        block a flip the raw-gap gates already found convincing — no
        discriminating signal means no veto, matching
        ball_auto_anchor's _reachability_winner fallback."""
        K, R, t, ball_uvs = self._kink_scene()
        sparse_uvs = {10: ball_uvs[10]}  # no v0/v1 in any velocity window
        a_world = (30.15, 34.0, 0.11)
        b_world = (30.02, 34.0, 0.11)

        class _Ctx4:
            def joints_at(self, frame):
                return [
                    _Joint("P001", "r_knee", a_world,
                           _project(a_world, K, R, t), 0.9),
                    _Joint("P002", "chest", b_world,
                           _project(b_world, K, R, t), 0.9),
                ]

        ctx = _Ctx4()
        events = (BallEvent(frame=10, kind="touch", score=0.8,
                            player_id="P001", bone="r_knee"),)
        out = refine_touch_attribution(
            events, player_ctx=ctx, ball_uvs=sparse_uvs,
            per_frame_K={10: K}, per_frame_R={10: R}, per_frame_t={10: t},
            distortion=(0.0, 0.0),
            cfg=TouchAttributionCfg(enabled=True,
                                     cross_player_physics_guard=True))
        assert (out[0].player_id, out[0].bone) == ("P002", "chest")

    def test_corroborated_alternate_cross_player_blocked_without_kink(self):
        """The same guard applies on the W3 ranked-candidates second-
        opinion path (:func:`_corroborated_alternate`), not just the
        primary ray-gap check. The bystander is kept OUT of the primary
        check's own FK scan via low confidence (below min_fk_conf) so
        the standard check stays tied, forcing the fallback path — but
        the physics term still sees its joint pixel."""
        K, R, t, ball_uvs = self._kink_scene()
        r_foot_world = (30.02, 34.0, 0.11)
        chest_world = (24.0, 34.0, 0.11)  # on-ray at frame 12, no kink there

        class _Ctx5:
            def joints_at(self, frame):
                return [
                    _Joint("P001", "r_foot", r_foot_world,
                           _project(r_foot_world, K, R, t), 0.9),
                    _Joint("P002", "chest", chest_world,
                           _project(chest_world, K, R, t), 0.05),
                ]

        ctx = _Ctx5()
        Ks = {f: K for f in ball_uvs}
        Rs = {f: R for f in ball_uvs}
        ts = {f: t for f in ball_uvs}
        events = (BallEvent(frame=10, kind="touch", score=0.7,
                            player_id="P001", bone="r_foot"),)
        ranked = {
            10: [{"player_id": "P001", "bone": "r_foot",
                  "gap_m": 0.30, "score": 0.8}],
            12: [{"player_id": "P002", "bone": "chest",
                  "gap_m": 0.02, "score": 0.9}],
        }

        out_off = refine_touch_attribution(
            events, player_ctx=ctx, ball_uvs=ball_uvs,
            per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
            distortion=(0.0, 0.0), ranked_candidates=ranked,
            cfg=TouchAttributionCfg(enabled=True,
                                     cross_player_physics_guard=False))
        assert (out_off[0].player_id, out_off[0].bone) == ("P002", "chest")

        out_on = refine_touch_attribution(
            events, player_ctx=ctx, ball_uvs=ball_uvs,
            per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
            distortion=(0.0, 0.0), ranked_candidates=ranked,
            cfg=TouchAttributionCfg(enabled=True,
                                     cross_player_physics_guard=True))
        assert (out_on[0].player_id, out_on[0].bone) == ("P001", "r_foot")

    def test_config_block_has_cross_player_physics_guard(self):
        import yaml
        from pathlib import Path
        cfg = yaml.safe_load(
            Path("config/default.yaml").read_text())["ball"]["touch_attribution"]
        assert cfg["cross_player_physics_guard"] is True
        assert cfg["cross_player_physics_window"] == 3
        assert cfg["cross_player_physics_dir_weight"] == pytest.approx(0.6)
        assert cfg["cross_player_physics_kink_weight"] == pytest.approx(0.4)
        assert cfg["cross_player_physics_slack"] == pytest.approx(0.0)


def test_context_expected_worlds_bridge_over_touch_windows():
    from src.utils.ball_touch_attribution import context_expected_worlds

    # Track dragged to a wrong pin at f10 (spike); context bridges over it.
    world = {f: (float(f), 0.0, 0.11) for f in range(21)}
    world[10] = (10.0, 8.0, 0.11)     # dragged toward the wrong joint
    world[9] = (9.0, 4.0, 0.11)
    world[11] = (11.0, 4.0, 0.11)
    exp = context_expected_worlds(world, touch_frames={10}, window=2)
    # Frames inside the ±window around the touch are re-interpolated from
    # the clean context (f7 → f13): the spike is bridged away.
    for f in range(8, 13):
        assert abs(exp[f][1]) < 0.3, f
        assert abs(exp[f][0] - f) < 0.3, f
    # Far frames keep the track's own value.
    assert exp[3] == world[3]

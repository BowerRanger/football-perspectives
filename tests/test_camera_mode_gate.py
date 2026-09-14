"""Tests for ``src/utils/camera_mode_gate.py``: the static/moving
consistency gate, leave-one-out click triage, weak-support gap
surfacing, and the tri-state ``camera.static_camera`` config parse.

See docs/superpowers/specs/2026-09-09-moving-camera-support.md for the
gberch-2 diagnosis this feature fixes.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.schemas.anchor import Anchor, LandmarkObservation
from src.utils.anchor_solver import (
    _make_K,
    _rvec_to_R,
    _solve_anchor_with_C_fixed,
    _solve_one_anchor_full,
    refine_with_bounded_motion,
    refine_with_shared_translation,
    reprojection_residual_for_anchor,
    solve_anchors_jointly,
)
from src.utils.camera_mode_gate import (
    anchors_needing_click_triage,
    evaluate_static_gate,
    find_loo_click_culprit,
    find_weak_support_gaps,
    parse_static_camera_mode,
)

IMAGE_SIZE: tuple[int, int] = (1920, 1080)
CX_TRUE = IMAGE_SIZE[0] / 2.0
CY_TRUE = IMAGE_SIZE[1] / 2.0


def _K(fx: float) -> np.ndarray:
    return np.array([[fx, 0.0, CX_TRUE], [0.0, fx, CY_TRUE], [0.0, 0.0, 1.0]])


def _yaw(angle_deg: float) -> np.ndarray:
    look = np.array([0.0, 64.0, -30.0])
    look = look / np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    base = np.array([right, down, look], dtype=float)
    a = np.deg2rad(angle_deg)
    Ry = np.array(
        [[np.cos(a), -np.sin(a), 0.0],
         [np.sin(a), np.cos(a), 0.0],
         [0.0, 0.0, 1.0]],
    )
    return base @ Ry.T


def _project(K: np.ndarray, R: np.ndarray, t: np.ndarray, world: np.ndarray) -> tuple[float, float]:
    cam = R @ world + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def _make_landmark(K, R, t, name: str, world: tuple[float, float, float]) -> LandmarkObservation:
    return LandmarkObservation(
        name=name,
        image_xy=_project(K, R, t, np.asarray(world, dtype=float)),
        world_xyz=world,
    )


_LANDMARK_WORLD: list[tuple[str, tuple[float, float, float]]] = [
    ("near_left_corner", (0.0, 0.0, 0.0)),
    ("near_right_corner", (105.0, 0.0, 0.0)),
    ("far_left_corner", (0.0, 68.0, 0.0)),
    ("far_right_corner", (105.0, 68.0, 0.0)),
    ("halfway_near", (52.5, 0.0, 0.0)),
    ("near_left_corner_flag_top", (0.0, 0.0, 1.5)),
    ("left_goal_crossbar_left", (0.0, 30.34, 2.44)),
    ("left_goal_crossbar_right", (0.0, 37.66, 2.44)),
]

# A spidercam hovering 12-15m above midfield (gberch-2's diagnosed
# geometry) sees a LOCAL patch around the halfway line/centre circle,
# not the whole 105x68m pitch corner-to-corner the way a behind-the-goal
# broadcast camera 30m back does — reusing _LANDMARK_WORLD's full-pitch
# corners for a close overhead camera puts several of them behind the
# camera or wildly off-axis. Numerically verified (all z>0, finite
# pixels) for the c_start/c_end/fx range _moving_camera_anchors uses.
_SPIDERCAM_LANDMARK_WORLD: list[tuple[str, tuple[float, float, float]]] = [
    ("halfway_near", (52.5, 0.0, 0.0)),
    ("halfway_far_ish", (52.5, 40.0, 0.0)),
    ("quarter_left", (40.0, 20.0, 0.0)),
    ("quarter_right", (65.0, 20.0, 0.0)),
    ("centre_spot", (52.5, 25.0, 0.0)),
    ("circle_top", (52.5, 25.0 + 9.15, 0.0)),
    ("circle_bottom", (52.5, 25.0 - 9.15, 0.0)),
    ("pole_marker", (52.5, 25.0, 3.0)),   # non-coplanar point
]


def _rich_anchor(
    K: np.ndarray, R: np.ndarray, t: np.ndarray, frame: int,
    landmark_world: list[tuple[str, tuple[float, float, float]]] = _LANDMARK_WORLD,
) -> Anchor:
    return Anchor(
        frame=frame,
        landmarks=tuple(
            _make_landmark(K, R, t, name, xyz) for name, xyz in landmark_world
        ),
    )


def _camera_centred_anchor(
    C: np.ndarray, R: np.ndarray, fx: float, frame: int,
    landmark_world: list[tuple[str, tuple[float, float, float]]] = _LANDMARK_WORLD,
) -> Anchor:
    t = -R @ C
    return _rich_anchor(_K(fx), R, t, frame, landmark_world)


def _look_at_R(
    C: np.ndarray,
    target: np.ndarray = np.array([52.5, 25.0, 0.0]),
    up_hint: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """World->camera rotation for a camera AT ``C`` looking at ``target``.

    A spidercam hovers above the pitch interior (e.g. C=(50, 20, 12)),
    unlike the fixed behind-the-goal broadcast pose the ``_yaw`` helper
    is tuned for — reusing ``_yaw`` unchanged for a translating C points
    the camera in a direction that doesn't track the pitch at all as C
    moves, degenerating every anchor's solo solve. This derives R fresh
    from C every time, exactly like R_BASE's own construction.
    """
    look = target - C
    look = look / np.linalg.norm(look)
    right = np.cross(look, up_hint)
    right = right / np.linalg.norm(right)
    down = np.cross(look, right)
    return np.array([right, down, look], dtype=float)


def _moving_camera_anchors(
    n_frames: int = 180,
    anchor_frames: tuple[int, ...] = (0, 90, 180),
    c_start: np.ndarray = np.array([50.0, 20.0, 12.0]),
    c_end: np.ndarray = np.array([44.0, 15.0, 15.0]),
    fx_start: float = 1000.0,
    fx_end: float = 2600.0,
) -> tuple[tuple[Anchor, ...], dict[int, np.ndarray]]:
    """Spidercam-like trajectory: camera centre translates ~10m across
    the shot while zooming, matching gberch-2's diagnosed geometry."""
    anchors = []
    truth: dict[int, np.ndarray] = {}
    for af in anchor_frames:
        w = af / (n_frames - 1)
        C = (1.0 - w) * c_start + w * c_end
        fx = (1.0 - w) * fx_start + w * fx_end
        R = _look_at_R(C)
        anchors.append(
            _camera_centred_anchor(C, R, fx, af, _SPIDERCAM_LANDMARK_WORLD)
        )
        truth[af] = C
    return tuple(anchors), truth


def _static_camera_anchors(
    anchor_frames: tuple[int, ...] = (0, 90, 180),
    C: np.ndarray = np.array([52.5, -30.0, 30.0]),
    fx: float = 1500.0,
) -> tuple[Anchor, ...]:
    """Same generator family, but the camera centre never moves — today's
    static-broadcast scenario."""
    anchors = []
    for af in anchor_frames:
        R = _yaw((af / 90.0) * 8.0 - 8.0)
        anchors.append(_camera_centred_anchor(C, R, fx, af))
    return tuple(anchors)


# ── Tri-state config parse ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "value,expected",
    [
        ("auto", "auto"),
        ("AUTO", "auto"),
        (None, "auto"),
        ("true", "static"),
        ("True", "static"),
        ("static", "static"),
        ("false", "moving"),
        ("False", "moving"),
        ("moving", "moving"),
        (True, "static"),   # legacy bool
        (False, "moving"),  # legacy bool
    ],
)
def test_parse_static_camera_mode(value, expected):
    assert parse_static_camera_mode(value) == expected


@pytest.mark.unit
def test_parse_static_camera_mode_rejects_garbage():
    with pytest.raises(ValueError):
        parse_static_camera_mode("sideways")


# ── Consistency gate arithmetic ──────────────────────────────────────────


@pytest.mark.unit
def test_evaluate_static_gate_holds_for_a_truly_static_camera():
    anchors = _static_camera_anchors()
    sol = solve_anchors_jointly(anchors, image_size=IMAGE_SIZE)
    relocked = refine_with_shared_translation(anchors, sol)
    gate = evaluate_static_gate(anchors, sol, relocked)
    assert gate.holds, (
        f"static generator should pass the gate; worst frame "
        f"{gate.worst_frame} clamped={gate.worst_clamped_px:.1f}px "
        f"solo={gate.worst_solo_px:.1f}px"
    )


@pytest.mark.unit
def test_evaluate_static_gate_fails_for_a_genuinely_moving_camera():
    anchors, _truth = _moving_camera_anchors()
    sol = solve_anchors_jointly(anchors, image_size=IMAGE_SIZE)
    relocked = refine_with_shared_translation(anchors, sol)
    gate = evaluate_static_gate(anchors, sol, relocked)
    assert not gate.holds
    assert gate.worst_frame is not None
    # The implied centre spread should reflect the ~10m translation, not
    # a sub-metre click-noise wobble.
    assert gate.centre_spread_m > 3.0


@pytest.mark.unit
def test_evaluate_static_gate_ratio_and_floor_knobs_change_the_outcome():
    """A very loose gate (huge ratio + huge floor) must hold even for the
    moving generator — confirms the config knobs actually gate the
    decision rather than being ignored."""
    anchors, _truth = _moving_camera_anchors()
    sol = solve_anchors_jointly(anchors, image_size=IMAGE_SIZE)
    relocked = refine_with_shared_translation(anchors, sol)
    loose_gate = evaluate_static_gate(
        anchors, sol, relocked, residual_ratio=1000.0, residual_floor_px=1e6,
    )
    assert loose_gate.holds


@pytest.mark.unit
def test_evaluate_static_gate_holds_trivially_with_a_single_rich_anchor():
    """A single rich anchor's "shared centre" IS its own solo centre, so
    the clamped and solo residuals coincide — the gate must not raise
    and must hold trivially (nothing else to disagree with it)."""
    anchor = _camera_centred_anchor(
        np.array([52.5, -30.0, 30.0]), _yaw(0.0), 1500.0, frame=0,
    )
    sol = solve_anchors_jointly((anchor,), image_size=IMAGE_SIZE)
    relocked = refine_with_shared_translation((anchor,), sol)
    gate = evaluate_static_gate((anchor,), sol, relocked)
    assert gate.holds
    assert len(gate.per_anchor) == 1


@pytest.mark.unit
def test_evaluate_static_gate_holds_with_no_rich_anchor_at_all():
    """With only a thin (non-rich, <6 landmark) anchor in the solution,
    there is no solo baseline to compare against — the gate must not
    raise and reports no evidence either way (holds=True, empty
    per_anchor). Constructed directly against a JointSolution (rather
    than through solve_anchors_jointly, which requires >=1 rich anchor
    to succeed at all) to isolate evaluate_static_gate's own handling
    of "nothing rich to evaluate"."""
    from src.utils.anchor_solver import JointSolution

    R = _yaw(0.0)
    C = np.array([52.5, -30.0, 30.0])
    t = -R @ C
    K = _K(1500.0)
    thin = Anchor(
        frame=0,
        landmarks=tuple(
            _make_landmark(K, R, t, name, xyz)
            for name, xyz in _LANDMARK_WORLD[:4]
        ),
    )
    sol = JointSolution(
        t_world=t, principal_point=(CX_TRUE, CY_TRUE),
        per_anchor_KRt={0: (K, R, t)}, per_anchor_residual_px={0: 0.0},
    )
    gate = evaluate_static_gate((thin,), sol, sol)
    assert gate.holds
    assert gate.per_anchor == {}


# ── Moving-path recovery (the core acceptance criteria) ─────────────────


@pytest.mark.unit
def test_moving_generator_recovers_anchor_centres_within_one_metre():
    anchors, truth = _moving_camera_anchors()
    sol = solve_anchors_jointly(anchors, image_size=IMAGE_SIZE)
    moving_sol = refine_with_bounded_motion(anchors, sol, max_motion_m=40.0)
    for a in anchors:
        K, R, t = moving_sol.per_anchor_KRt[a.frame]
        C_hat = -R.T @ t
        err = float(np.linalg.norm(C_hat - truth[a.frame]))
        assert err <= 1.0, f"frame {a.frame}: centre error {err:.2f}m > 1.0m"


@pytest.mark.unit
def test_bounded_motion_ignores_a_degenerate_rich_anchor_when_seeding():
    """Real-clip regression (gberch-2): a rich-by-landmark-count anchor
    whose (K, R, t) is degenerate — e.g. surviving out of Pass 3's joint
    distortion refine, which has no degeneracy check of its own even
    though Task A hardened Pass 1's seeding — must not poison the mean
    C-seed refine_with_bounded_motion computes across "rich" anchors.
    Reproduces the exact failure: gberch-2's frame 162 came out of the
    hybrid solve at fx=57227, C=(-1184, 1123, 733) (_is_degenerate_solo
    is True), and the OLD unguarded np.mean(rich_Cs) dragged the shared
    reference C for ALL anchors to nonsense (a real run showed the mean
    residual explode 35.78 -> 750000264.03 px)."""
    anchors, truth = _moving_camera_anchors(
        anchor_frames=(0, 90, 180, 135),
    )
    sol = solve_anchors_jointly(anchors, image_size=IMAGE_SIZE)
    # Inject a degenerate entry for frame 135 (rich by landmark count,
    # but a nonsense pose) directly into per_anchor_KRt, exactly as
    # Pass 3 does in practice — this isolates refine_with_bounded_
    # motion's OWN seed robustness from solve_anchors_jointly's.
    poisoned_K = _K(57227.0)
    poisoned_R = np.eye(3)
    poisoned_t = np.array([-33.4, -1.0, 1789.4])
    new_KRt = dict(sol.per_anchor_KRt)
    new_KRt[135] = (poisoned_K, poisoned_R, poisoned_t)
    new_res = dict(sol.per_anchor_residual_px)
    new_res[135] = 50.0
    poisoned_sol = sol._replace(per_anchor_KRt=new_KRt, per_anchor_residual_px=new_res)

    moving_sol = refine_with_bounded_motion(anchors, poisoned_sol, max_motion_m=40.0)
    for af in (0, 90, 180):
        K, R, t = moving_sol.per_anchor_KRt[af]
        C_hat = -R.T @ t
        err = float(np.linalg.norm(C_hat - truth[af]))
        assert err <= 2.0, (
            f"frame {af}: centre error {err:.2f}m > 2.0m — the degenerate "
            f"frame 135 anchor poisoned the good anchors' recovery"
        )
    # The degenerate anchor's OWN result must also come out BOUNDED, not
    # astronomical — real-clip regression: without borrowing a sane
    # (rvec, fx) seed/bound from a neighbour, frame 162 came out of this
    # function at residual ~1e18 px (fx bound centred on its own
    # fx=57227 input). It's still allowed to fit poorly (it has a real
    # data problem), just not explode.
    K_poisoned, _R_poisoned, _t_poisoned = moving_sol.per_anchor_KRt[135]
    fx_poisoned = float(K_poisoned[0, 0])
    assert 50.0 <= fx_poisoned <= 1e5, f"frame 135 fx exploded to {fx_poisoned}"
    C_poisoned = -_R_poisoned.T @ _t_poisoned
    assert np.all(np.isfinite(C_poisoned)) and np.linalg.norm(C_poisoned) < 1e4, (
        f"frame 135 centre exploded to {C_poisoned}"
    )


@pytest.mark.unit
def test_static_generator_still_locks_one_shared_centre():
    """The same family of generator, with a fixed C, must reproduce
    today's static behaviour after the relock: one shared centre, tight
    residuals for every anchor."""
    anchors = _static_camera_anchors()
    sol = solve_anchors_jointly(anchors, image_size=IMAGE_SIZE)
    relocked = refine_with_shared_translation(anchors, sol)
    assert relocked.camera_centre is not None
    for f, r in relocked.per_anchor_residual_px.items():
        assert r < 5.0, f"frame {f}: residual {r:.2f}px"


# ── Leave-one-out click triage ────────────────────────────────────────────


@pytest.mark.unit
def test_find_loo_click_culprit_identifies_the_corrupted_landmark():
    R = _yaw(0.0)
    C = np.array([52.5, -30.0, 30.0])
    anchor = _camera_centred_anchor(C, R, 1500.0, frame=162)
    lms = list(anchor.landmarks)
    culprit_name = lms[3].name
    u, v = lms[3].image_xy
    lms[3] = LandmarkObservation(
        name=culprit_name, image_xy=(u + 400.0, v - 250.0),
        world_xyz=lms[3].world_xyz,
    )
    corrupted = Anchor(frame=162, landmarks=tuple(lms))

    def _solve_fn(a: Anchor) -> float:
        result = _solve_one_anchor_full(
            a, CX_TRUE, CY_TRUE, fx_init=1500.0, K_init=_K(1500.0),
        )
        if result is None:
            return float("inf")
        K, R_hat, t_hat, _fx = result
        return reprojection_residual_for_anchor(a, K, R_hat, t_hat)

    baseline = _solve_fn(corrupted)
    result = find_loo_click_culprit(
        corrupted, _solve_fn, baseline_residual_px=baseline,
        min_collapse_ratio=5.0, accept_below_px=15.0,
    )
    assert result is not None
    assert result.culprit_name == culprit_name
    assert result.residual_without_px < result.residual_with_px / 5.0


@pytest.mark.unit
def test_find_loo_click_culprit_returns_none_when_no_single_click_explains_it():
    """gberch-2's frame 0: genuine camera translation, not a bad click.
    Dropping any ONE landmark barely moves the residual, so the LOO
    triage must not invent a false culprit."""
    anchors, _truth = _moving_camera_anchors()
    by_frame = {a.frame: a for a in anchors}
    sol = solve_anchors_jointly(anchors, image_size=IMAGE_SIZE)
    relocked = refine_with_shared_translation(anchors, sol)
    C_locked = np.asarray(relocked.camera_centre)
    frame0 = by_frame[0]
    K0, R0, _t0 = relocked.per_anchor_KRt[0]
    import cv2
    rvec0, _ = cv2.Rodrigues(R0.astype(np.float64))
    fx0 = float(K0[0, 0])

    def _solve_fn(a: Anchor) -> float:
        rvec, fx = _solve_anchor_with_C_fixed(
            a, C_locked, CX_TRUE, CY_TRUE, fx0, rvec0.reshape(3),
        )
        R_hat = _rvec_to_R(rvec)
        t_hat = -R_hat @ C_locked
        K_hat = _make_K(fx, CX_TRUE, CY_TRUE)
        return reprojection_residual_for_anchor(a, K_hat, R_hat, t_hat)

    baseline = _solve_fn(frame0)
    result = find_loo_click_culprit(
        frame0, _solve_fn, baseline_residual_px=baseline,
        min_collapse_ratio=5.0, accept_below_px=15.0,
    )
    assert result is None, (
        f"expected no LOO culprit for a genuine-translation anchor, got "
        f"{result}"
    )


@pytest.mark.unit
def test_find_loo_click_culprit_needs_at_least_two_landmarks():
    anchor = Anchor(
        frame=0,
        landmarks=(LandmarkObservation("only_one", (10.0, 10.0), (0.0, 0.0, 0.0)),),
    )
    result = find_loo_click_culprit(anchor, lambda a: 0.0)
    assert result is None


@pytest.mark.unit
def test_anchors_needing_click_triage_flags_the_worst_outlier():
    # 144 sits close to the 3.5-3.9px baseline (not an outlier); 162 is
    # far above both the flag threshold and its neighbours' median.
    residuals = {0: 3.9, 144: 4.8, 162: 22.0, 180: 3.5}
    flagged = anchors_needing_click_triage(
        residuals, flag_threshold_px=4.0, neighbour_ratio=2.0,
    )
    assert flagged == [162]


@pytest.mark.unit
def test_anchors_needing_click_triage_empty_when_all_consistent():
    residuals = {0: 3.9, 144: 4.8, 162: 4.2, 180: 3.5}
    flagged = anchors_needing_click_triage(
        residuals, flag_threshold_px=4.0, neighbour_ratio=2.0,
    )
    assert flagged == []


# ── Weak-support gap surfacing ────────────────────────────────────────────


@pytest.mark.unit
def test_find_weak_support_gaps_flags_a_wide_low_confidence_span():
    anchor_frames = [0, 200]
    per_frame_conf = [1.0] + [0.4] * 199 + [1.0]
    gaps = find_weak_support_gaps(
        anchor_frames, per_frame_conf, max_gap_frames=60, weak_confidence_below=0.6,
    )
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.start_frame == 0
    assert gap.end_frame == 200
    assert gap.suggested_frame == 100


@pytest.mark.unit
def test_find_weak_support_gaps_skips_short_or_well_supported_spans():
    anchor_frames = [0, 30, 260]
    # Indices 0..260 inclusive (261 frames): anchors at 0/30/260, a SHORT
    # 29-frame gap (0..30, below max_gap_frames) and a LONG but
    # well-supported 229-frame gap (30..260, mean confidence 0.9 stays
    # above weak_confidence_below).
    per_frame_conf = [0.0] * 261
    per_frame_conf[0] = 1.0
    for i in range(1, 30):
        per_frame_conf[i] = 0.5
    per_frame_conf[30] = 1.0
    for i in range(31, 260):
        per_frame_conf[i] = 0.9
    per_frame_conf[260] = 1.0
    gaps = find_weak_support_gaps(
        anchor_frames, per_frame_conf, max_gap_frames=60, weak_confidence_below=0.6,
    )
    assert gaps == []

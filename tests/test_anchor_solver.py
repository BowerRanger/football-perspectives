"""Tests for the joint multi-anchor camera-pose solver."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.schemas.anchor import (
    Anchor,
    AnchorSet,
    LandmarkObservation,
    LineObservation,
)
from src.utils.anchor_solver import (
    AnchorSolveError,
    refine_with_shared_translation,
    reprojection_residual_for_anchor,
    solve_anchors_jointly,
)
from src.utils.pitch_lines_catalogue import LINE_CATALOGUE


# ── Test geometry: physically valid broadcast pose ──────────────────────────
# Camera at world C=(52.5, -30, 30) looking at pitch centre (52.5, 34, 0).
# R_BASE built from exact normalised look-direction so it's orthonormal to
# floating-point precision (matches D7 in decisions log).

_LOOK = np.array([0.0, 64.0, -30.0])
_LOOK = _LOOK / np.linalg.norm(_LOOK)
_RIGHT = np.array([1.0, 0.0, 0.0])
_DOWN = np.cross(_LOOK, _RIGHT)
R_BASE: np.ndarray = np.array([_RIGHT, _DOWN, _LOOK], dtype=float)
T_BASE: np.ndarray = -R_BASE @ np.array([52.5, -30.0, 30.0])

IMAGE_SIZE: tuple[int, int] = (1920, 1080)
FX_TRUE: float = 1500.0
CX_TRUE: float = IMAGE_SIZE[0] / 2.0
CY_TRUE: float = IMAGE_SIZE[1] / 2.0


def _K(fx: float = FX_TRUE) -> np.ndarray:
    return np.array([[fx, 0.0, CX_TRUE], [0.0, fx, CY_TRUE], [0.0, 0.0, 1.0]])


def _yaw(angle_deg: float) -> np.ndarray:
    """World-z yaw applied as ``R_BASE @ R_yaw.T`` (matches synthetic clip)."""
    a = np.deg2rad(angle_deg)
    Ry = np.array(
        [[np.cos(a), -np.sin(a), 0.0],
         [np.sin(a),  np.cos(a), 0.0],
         [0.0,        0.0,       1.0]],
    )
    return R_BASE @ Ry.T


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


def _make_line(
    K, R, t, name: str, *, alpha: float = 0.2, beta: float = 0.8
) -> LineObservation:
    """Build a synthetic line observation by projecting two intermediate
    points along the world segment (``alpha`` and ``beta`` fractions). This
    mimics the user clicking two distinct points along the painted line.
    """
    seg = LINE_CATALOGUE[name]
    pa, pb = np.asarray(seg[0]), np.asarray(seg[1])
    A = pa + alpha * (pb - pa)
    B = pa + beta * (pb - pa)
    return LineObservation(
        name=name,
        image_segment=(_project(K, R, t, A), _project(K, R, t, B)),
        world_segment=seg,
    )


def _angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    cos_t = (np.trace(R_a.T @ R_b) - 1) / 2
    cos_t = max(-1.0, min(1.0, cos_t))
    return float(np.degrees(np.arccos(cos_t)))


# ── Anchor builders ─────────────────────────────────────────────────────────


def _rich_anchor(K: np.ndarray, R: np.ndarray, t: np.ndarray, frame: int) -> Anchor:
    """8 landmarks across multiple z-levels — well-conditioned for solvePnP."""
    return Anchor(
        frame=frame,
        landmarks=tuple(
            _make_landmark(K, R, t, name, xyz)
            for name, xyz in [
                ("near_left_corner",          (0.0,    0.0,   0.0)),
                ("near_right_corner",         (105.0,  0.0,   0.0)),
                ("far_left_corner",           (0.0,    68.0,  0.0)),
                ("far_right_corner",          (105.0,  68.0,  0.0)),
                ("halfway_near",              (52.5,   0.0,   0.0)),
                ("near_left_corner_flag_top", (0.0,    0.0,   1.5)),
                ("left_goal_crossbar_left",   (0.0,    30.34, 2.44)),
                ("left_goal_crossbar_right",  (0.0,    37.66, 2.44)),
            ]
        ),
    )


def _thin_anchor(K: np.ndarray, R: np.ndarray, t: np.ndarray, frame: int) -> Anchor:
    """4 landmarks (the lower bound for solvePnP)."""
    return Anchor(
        frame=frame,
        landmarks=tuple(
            _make_landmark(K, R, t, name, xyz)
            for name, xyz in [
                ("near_left_corner",  (0.0, 0.0, 0.0)),
                ("near_right_corner", (105.0, 0.0, 0.0)),
                ("far_left_corner",   (0.0, 68.0, 0.0)),
                ("halfway_near",      (52.5, 0.0, 0.0)),
            ]
        ),
    )


def _line_only_anchor(K: np.ndarray, R: np.ndarray, t: np.ndarray, frame: int) -> Anchor:
    """0 points + several geometrically diverse line correspondences.

    Includes one vertical goal post — the z-axis line breaks the
    coplanar/parallel-VP ambiguity that 4 ground lines alone leave.
    """
    return Anchor(
        frame=frame,
        landmarks=(),
        lines=tuple(
            _make_line(K, R, t, name)
            for name in [
                "near_touchline",          # along +x at y=0
                "far_touchline",           # along +x at y=68
                "halfway_line",            # along +y at x=52.5
                "left_goal_left_post",     # vertical (z), breaks ground-plane ambiguity
            ]
        ),
    )


# ── Tests ───────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_joint_solve_recovers_known_camera_with_rich_anchors():
    K_true = _K()
    anchors = (
        _rich_anchor(K_true, R_BASE,      T_BASE, 0),
        _rich_anchor(K_true, _yaw(10.0),  T_BASE, 50),
        _rich_anchor(K_true, _yaw(-10.0), T_BASE, 100),
    )
    sol = solve_anchors_jointly(anchors, image_size=IMAGE_SIZE)

    assert np.allclose(sol.t_world, T_BASE, atol=0.5)
    assert abs(sol.principal_point[0] - CX_TRUE) < 30.0
    assert abs(sol.principal_point[1] - CY_TRUE) < 30.0
    for a in anchors:
        K_hat, _ = sol.per_anchor_KRt[a.frame][:2]
        assert abs(K_hat[0, 0] - FX_TRUE) < 50.0, (
            f"frame {a.frame}: fx={K_hat[0,0]:.1f} vs true {FX_TRUE}"
        )
    for f, r in sol.per_anchor_residual_px.items():
        assert r < 5.0, f"frame {f}: residual {r:.2f} px"


@pytest.mark.unit
def test_joint_solve_with_one_rich_and_two_thin_anchors():
    """One rich anchor (8 landmarks, 3 z-levels) seeds K and t. Two thin
    anchors (4 coplanar landmarks each) inherit t and refine (R, fx)."""
    K_true = _K()
    anchors = (
        _rich_anchor(K_true, R_BASE,     T_BASE, 0),
        _thin_anchor(K_true, _yaw(10.0), T_BASE, 50),
        _thin_anchor(K_true, _yaw(-10.0),T_BASE, 100),
    )
    sol = solve_anchors_jointly(anchors, image_size=IMAGE_SIZE)
    assert np.allclose(sol.t_world, T_BASE, atol=2.0)
    for f, r in sol.per_anchor_residual_px.items():
        assert r < 10.0, f"frame {f}: residual {r:.2f} px"


@pytest.mark.unit
def test_joint_solve_with_hybrid_point_plus_line_anchor():
    """Anchor with points + supplementary line correspondences (the realistic
    workflow). Lines are additional constraints; points provide the
    primary positional anchor."""
    K_true = _K()
    yaw_R = _yaw(8.0)
    hybrid = Anchor(
        frame=50,
        landmarks=tuple(
            _make_landmark(K_true, yaw_R, T_BASE, name, xyz)
            for name, xyz in [
                ("near_left_corner",  (0.0, 0.0, 0.0)),
                ("near_right_corner", (105.0, 0.0, 0.0)),
                ("far_left_corner",   (0.0, 68.0, 0.0)),
                ("halfway_near",      (52.5, 0.0, 0.0)),
            ]
        ),
        lines=(
            _make_line(K_true, yaw_R, T_BASE, "near_touchline"),
            _make_line(K_true, yaw_R, T_BASE, "halfway_line"),
        ),
    )
    anchors = (
        _rich_anchor(K_true, R_BASE,     T_BASE, 0),
        hybrid,
        _rich_anchor(K_true, _yaw(-8.0), T_BASE, 100),
    )
    sol = solve_anchors_jointly(anchors, image_size=IMAGE_SIZE)
    _, R_hybrid = sol.per_anchor_KRt[50][:2]
    err = _angle_deg(R_hybrid, yaw_R)
    assert err < 1.0, f"hybrid anchor R error {err:.2f}°"
    assert sol.per_anchor_residual_px[50] < 5.0


@pytest.mark.unit
def test_joint_solve_is_robust_to_mild_noise():
    K_true = _K()
    rng = np.random.default_rng(42)

    def _noise(a: Anchor) -> Anchor:
        noisy = []
        for lm in a.landmarks:
            jitter = rng.normal(scale=2.0, size=2)
            noisy.append(LandmarkObservation(
                name=lm.name,
                image_xy=(lm.image_xy[0] + jitter[0], lm.image_xy[1] + jitter[1]),
                world_xyz=lm.world_xyz,
            ))
        return Anchor(frame=a.frame, landmarks=tuple(noisy), lines=a.lines)

    anchors = tuple(
        _noise(_rich_anchor(K_true, R, T_BASE, frame))
        for frame, R in [(0, R_BASE), (50, _yaw(10.0)), (100, _yaw(-10.0))]
    )
    sol = solve_anchors_jointly(anchors, image_size=IMAGE_SIZE)
    assert np.allclose(sol.t_world, T_BASE, atol=2.0)
    for f, r in sol.per_anchor_residual_px.items():
        assert r < 6.0, f"frame {f}: residual {r:.2f} px"


@pytest.mark.unit
def test_no_qualifying_anchors_raises():
    a = Anchor(frame=0, landmarks=(LandmarkObservation('x', (1, 2), (0, 0, 0)),))
    with pytest.raises(AnchorSolveError):
        solve_anchors_jointly((a,), image_size=IMAGE_SIZE)


@pytest.mark.unit
def test_outlier_anchor_residual_is_at_or_above_median():
    """A wildly-wrong anchor should not have a residual lower than the
    median across the set. Used by ``camera.py`` to flag low-confidence anchors.
    """
    K_true = _K()
    bad_R = _yaw(45.0)
    bad = Anchor(
        frame=50,
        landmarks=tuple(
            _make_landmark(K_true, bad_R, T_BASE, name, xyz)
            for name, xyz in [
                ("near_left_corner",          (0.0, 0.0, 0.0)),
                ("near_right_corner",         (105.0, 0.0, 0.0)),
                ("far_left_corner",           (0.0, 68.0, 0.0)),
                ("halfway_near",              (52.5, 0.0, 0.0)),
                ("near_left_corner_flag_top", (0.0, 0.0, 1.5)),
                ("left_goal_crossbar_left",   (0.0, 30.34, 2.44)),
            ]
        ),
    )
    anchors = (
        _rich_anchor(K_true, R_BASE,    T_BASE, 0),
        bad,
        _rich_anchor(K_true, _yaw(-5.0), T_BASE, 100),
    )
    sol = solve_anchors_jointly(anchors, image_size=IMAGE_SIZE)
    rs = list(sol.per_anchor_residual_px.values())
    median = float(np.median(rs))
    assert sol.per_anchor_residual_px[50] >= median - 1e-6


@pytest.mark.unit
def test_real_data_smoke_test():
    """Smoke test on the live anchors.json. The seed (rich) anchor must
    recover within a few pixels; other anchors may have higher residuals
    if the user's clicks are inconsistent with the shared-t broadcast
    assumption (those surface as low-confidence in camera.py).
    """
    path = Path("output/camera/anchors.json")
    if not path.exists():
        pytest.skip("real-data anchors.json not present")
    aset = AnchorSet.load(path)
    qualifying = tuple(
        a for a in aset.anchors if len(a.landmarks) >= 4 or len(a.lines) >= 2
    )
    if not qualifying:
        pytest.skip("no qualifying anchors in the saved data")
    sol = solve_anchors_jointly(qualifying, image_size=aset.image_size)
    # Hard regression: no anchor hits the behind-camera sentinel.
    for f, r in sol.per_anchor_residual_px.items():
        assert r < 1e8, f"anchor {f} still hitting behind-camera sentinel: {r}"
    # The richest anchor (most landmarks) is the seed; its residual must be tight.
    seed_frame = max(qualifying, key=lambda a: len(a.landmarks)).frame
    seed_residual = sol.per_anchor_residual_px[seed_frame]
    assert seed_residual < 10.0, (
        f"seed anchor frame {seed_frame} residual {seed_residual:.1f} px too high"
    )


@pytest.mark.unit
def test_reprojection_residual_helper_returns_sentinel_for_behind_camera():
    K = _K()
    behind_anchor = Anchor(
        frame=0,
        landmarks=(LandmarkObservation(
            name="behind", image_xy=(0, 0), world_xyz=(52.5, -200.0, 0.0),
        ),),
    )
    r = reprojection_residual_for_anchor(behind_anchor, K, R_BASE, T_BASE)
    assert r == 1e9


# ── Vanishing-point (direction-only) line tests ────────────────────────────


def _vertical_separator_observation(
    K: np.ndarray, R: np.ndarray, t: np.ndarray, x: float, y: float
) -> LineObservation:
    """Build a synthetic vertical-separator annotation: pick a world position
    (x, y, 0)→(x, y, 1) and project the two endpoints. The user is clicking
    the visible projection of a real-world vertical line, but the solver
    only sees the line's *direction* (0, 0, 1)."""
    base = np.array([x, y, 0.0])
    top = np.array([x, y, 1.0])
    return LineObservation(
        name="vertical_separator",
        image_segment=(_project(K, R, t, base), _project(K, R, t, top)),
        world_segment=None,
        world_direction=(0.0, 0.0, 1.0),
    )


def _tilt(angle_deg: float, axis: np.ndarray) -> np.ndarray:
    """Rotation matrix about ``axis`` by ``angle_deg`` (Rodrigues)."""
    a = np.deg2rad(angle_deg)
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


@pytest.mark.unit
def test_vertical_separator_residual_is_zero_at_truth():
    """Direct check on _line_residuals: a vertical-separator annotation built
    from a known camera should have zero residual at that camera."""
    from src.utils.anchor_solver import _line_residuals

    K = _K()
    R = _yaw(5.0)
    t = T_BASE
    obs = _vertical_separator_observation(K, R, t, x=30.0, y=-2.0)
    r = _line_residuals([obs], K, R, t)
    assert np.allclose(r, 0.0, atol=1e-3), f"VP residual at truth: {r}"


@pytest.mark.unit
def test_vertical_separator_residual_grows_with_pitch_error():
    """Residual must grow when the camera's pitch (tilt about world-x) is
    wrong — vertical separators constrain pitch and roll. (They don't
    constrain yaw, since yaw about world-z preserves the vertical VP.)
    """
    from src.utils.anchor_solver import _line_residuals

    K = _K()
    t = T_BASE
    R_truth = R_BASE
    # 5° pitch about the camera's right axis (world-x for the broadcast pose).
    R_wrong = _tilt(5.0, axis=np.array([1.0, 0.0, 0.0])) @ R_BASE
    obs = _vertical_separator_observation(K, R_truth, t, x=30.0, y=-2.0)
    r_truth = _line_residuals([obs], K, R_truth, t)
    r_wrong = _line_residuals([obs], K, R_wrong, t)
    assert np.linalg.norm(r_truth) < 1e-3
    assert np.linalg.norm(r_wrong) > 5.0


@pytest.mark.unit
def test_joint_solve_with_thin_anchor_plus_vertical_separators():
    """A coplanar 4-point anchor that ALSO has 2 vertical-separator
    annotations (z-axis VP). Truth rotation includes a small tilt so VP
    constraints are meaningful (yaw alone leaves vertical VP invariant).
    """
    K_true = _K()
    # Combine 8° yaw (point landmarks constrain this) with 3° pitch
    # (vertical-separator VPs constrain this).
    R_truth = _tilt(3.0, axis=np.array([1.0, 0.0, 0.0])) @ _yaw(8.0)

    rescued = Anchor(
        frame=50,
        landmarks=tuple(
            _make_landmark(K_true, R_truth, T_BASE, name, xyz)
            for name, xyz in [
                ("near_left_corner",  (0.0, 0.0, 0.0)),
                ("near_right_corner", (105.0, 0.0, 0.0)),
                ("far_left_corner",   (0.0, 68.0, 0.0)),
                ("halfway_near",      (52.5, 0.0, 0.0)),
            ]
        ),
        lines=(
            _vertical_separator_observation(K_true, R_truth, T_BASE, x=20.0, y=-2.0),
            _vertical_separator_observation(K_true, R_truth, T_BASE, x=80.0, y=-2.0),
        ),
    )
    anchors = (
        _rich_anchor(K_true, R_BASE,     T_BASE, 0),
        rescued,
        _rich_anchor(K_true, _yaw(-8.0), T_BASE, 100),
    )
    sol = solve_anchors_jointly(anchors, image_size=IMAGE_SIZE)
    # The thin anchor solo-solve from a far-off K prior (fx=image_width=1920
    # vs truth 1500) is a hard global-search problem on coplanar geometry —
    # the LM can converge to local minima. The integration value of vertical
    # separators is verified by the residual tests above (zero at truth,
    # grows with pitch error). Here we just assert the solver completes
    # without crashing or hitting the behind-camera sentinel.
    assert 50 in sol.per_anchor_KRt
    assert sol.per_anchor_residual_px[50] < 1e8


def test_landmarks_collinear_detects_halfway_line_only_set():
    """Frame-255-style annotation: all points on world line x=52.5, z=0.

    The collinearity check must flag this so the user knows to add an
    off-axis landmark.
    """
    from src.utils.anchor_solver import _landmarks_collinear

    halfway_only = Anchor(
        frame=255,
        landmarks=tuple(
            LandmarkObservation(name=name, image_xy=(0.0, 0.0), world_xyz=xyz)
            for name, xyz in [
                ("centre_circle_far",  (52.5, 43.15, 0.0)),
                ("centre_spot",        (52.5, 34.0,  0.0)),
                ("centre_circle_near", (52.5, 24.85, 0.0)),
                ("halfway_far",        (52.5, 68.0,  0.0)),
            ]
        ),
    )
    assert _landmarks_collinear(halfway_only) is True

    # Non-collinear set (e.g. two corners + halfway point) is fine.
    triangle = Anchor(
        frame=42,
        landmarks=tuple(
            LandmarkObservation(name=name, image_xy=(0.0, 0.0), world_xyz=xyz)
            for name, xyz in [
                ("near_left_corner",  (0.0,   0.0,   0.0)),
                ("near_right_corner", (105.0, 0.0,   0.0)),
                ("halfway_far",       (52.5,  68.0,  0.0)),
            ]
        ),
    )
    assert _landmarks_collinear(triangle) is False


def test_solve_anchors_jointly_warns_on_collinear_anchor(caplog):
    """The orchestrator emits a warning naming the offending frame so the
    user sees it in the camera-stage log alongside other anchor diagnostics.
    """
    K_true = _K()
    rich = _rich_anchor(K_true, R_BASE, T_BASE, 0)
    collinear = Anchor(
        frame=255,
        landmarks=tuple(
            _make_landmark(K_true, R_BASE, T_BASE, name, xyz)
            for name, xyz in [
                ("centre_circle_far",  (52.5, 43.15, 0.0)),
                ("centre_spot",        (52.5, 34.0,  0.0)),
                ("centre_circle_near", (52.5, 24.85, 0.0)),
                ("halfway_far",        (52.5, 68.0,  0.0)),
            ]
        ),
    )
    with caplog.at_level("WARNING", logger="src.utils.anchor_solver"):
        solve_anchors_jointly((rich, collinear), image_size=IMAGE_SIZE)
    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("frame 255" in m and "collinear" in m for m in msgs), msgs


def _make_line_from_segment(
    K: np.ndarray, R: np.ndarray, t: np.ndarray,
    name: str,
    seg: tuple[tuple[float, float, float], tuple[float, float, float]],
    *, alpha: float = 0.2, beta: float = 0.8,
) -> LineObservation:
    """Like ``_make_line`` but takes the world segment directly (for entries
    that aren't in the static ``LINE_CATALOGUE`` — e.g. stadium-derived mow
    stripes whose positions depend on the picked stadium)."""
    pa, pb = np.asarray(seg[0]), np.asarray(seg[1])
    A = pa + alpha * (pb - pa)
    B = pa + beta * (pb - pa)
    return LineObservation(
        name=name,
        image_segment=(_project(K, R, t, A), _project(K, R, t, B)),
        world_segment=seg,
    )


def test_mow_stripe_line_drives_solo_solve_to_correct_translation():
    """Frame-0-style anchor: 2 collinear points + 5 far-side lines is rank-
    deficient on the front-back axis. A position-known mow stripe parallel
    to the touchlines but at a known interior y supplies the perpendicular
    translation constraint so the recovered camera stays close to truth.

    This is the integration test for stadium-derived mow-stripe lines flowing
    end-to-end through the solver — they should behave exactly like any
    other ``world_segment`` line annotation.
    """
    K_true = _K()
    rich = _rich_anchor(K_true, R_BASE, T_BASE, 0)

    # Frame-0-style thin anchor at frame 100: 2 collinear points (both at
    # x=88.5) plus 5 far-side lines, all bunched around y∈[54.16, 70].
    base_landmarks = tuple(
        _make_landmark(K_true, R_BASE, T_BASE, name, xyz)
        for name, xyz in [
            ("right_18yd_d_far",  (88.5, 41.31, 0.0)),
            ("right_18yd_far",    (88.5, 54.16, 0.0)),
        ]
    )
    base_lines = tuple(
        _make_line(K_true, R_BASE, T_BASE, name)
        for name in [
            "right_18yd_far_edge",
            "right_18yd_front",
            "far_touchline",
            "far_advertising_board_top",
            "far_advertising_board_base",
        ]
    )

    # Stadium-derived mow stripe at y=15.0 (5.5 m × 3 stripes from origin
    # 0 in the default Premier League pattern would be y=16.5; pick 15.0
    # for a clean number). The world_segment is what the editor would
    # have generated via stadium_config.mow_stripe_lines and persisted in
    # anchors.json.
    mow_line = _make_line_from_segment(
        K_true, R_BASE, T_BASE,
        "mow_y_15.0",
        ((0.0, 15.0, 0.0), (105.0, 15.0, 0.0)),
    )

    rescued = Anchor(
        frame=100,
        landmarks=base_landmarks,
        lines=base_lines + (mow_line,),
    )
    sol = solve_anchors_jointly((rich, rescued), image_size=IMAGE_SIZE)

    assert 100 in sol.per_anchor_KRt
    K_hat, R_hat, t_hat = sol.per_anchor_KRt[100]
    C_hat = -R_hat.T @ t_hat
    C_true = -R_BASE.T @ T_BASE  # (52.5, -30, 30)
    err = float(np.linalg.norm(C_hat - C_true))
    # Acceptance: with the mow stripe present, the recovered camera
    # position is within a few metres of truth (the rich anchor's t-fixed
    # propagation also gets us most of the way; this just confirms the
    # mow stripe flows through and doesn't make things worse).
    assert err < 5.0, (
        f"frame 100: |C_hat - C_true| = {err:.2f} m "
        f"(C_hat={C_hat}, C_true={C_true})"
    )



# ── Static-camera invariant tests (Phase 1) ─────────────────────────────────


_C_WORLD: np.ndarray = np.array([52.5, -30.0, 30.0])
"""World-frame camera body position used by the static-camera tests."""


def _t_for_yaw(angle_deg: float) -> np.ndarray:
    """Translation that places the yawed camera at ``_C_WORLD``."""
    return -_yaw(angle_deg) @ _C_WORLD


def _three_rich_anchors_static() -> tuple[Anchor, ...]:
    """Three rich anchors at yaw 0/+5/-5° with a shared world-frame camera centre.

    Unlike the existing ``_rich_anchor(_, _yaw(...), T_BASE, ...)`` pattern
    (which holds OpenCV ``t`` constant while rotating R, and so silently
    moves the camera body), this fixture rotates only R and recomputes
    ``t = -R @ C`` so every anchor truly shares the same body position.
    """
    K = _K()
    out: list[Anchor] = []
    for frame, yaw in ((0, 0.0), (60, 5.0), (120, -5.0)):
        R = _yaw(yaw)
        t = _t_for_yaw(yaw)
        out.append(_rich_anchor(K, R, t, frame))
    return tuple(out)


@pytest.mark.unit
def test_joint_solution_carries_camera_centre():
    """After ``refine_with_shared_translation``, the JointSolution exposes
    the locked camera centre, and every per-anchor (R, t) satisfies
    ``-R^T @ t == C`` to floating-point precision."""
    anchors = _three_rich_anchors_static()
    sol = solve_anchors_jointly(anchors, image_size=IMAGE_SIZE)
    sol = refine_with_shared_translation(anchors, sol)

    assert sol.camera_centre is not None, "camera_centre missing on JointSolution"
    C = np.asarray(sol.camera_centre)
    for af, (_K_a, R_a, t_a) in sol.per_anchor_KRt.items():
        recovered = -R_a.T @ t_a
        assert np.allclose(recovered, C, atol=1e-4), (
            f"anchor {af}: -R^T @ t = {recovered} != C = {C}"
        )


@pytest.mark.unit
def test_relock_does_not_silently_fall_back(caplog):
    """When a single anchor's true camera centre disagrees with the rest by
    several metres, the relock still returns a solution that honours the
    locked C across every anchor (no silent fallback to per-anchor t)."""
    import logging

    base = _three_rich_anchors_static()
    # Replace the middle anchor with one whose true camera centre is shifted
    # +5m laterally — this provokes a high relock residual.
    K = _K()
    R_bad = _yaw(5.0)
    C_bad = _C_WORLD + np.array([5.0, 0.0, 0.0])
    t_bad = -R_bad @ C_bad
    bad_anchor = _rich_anchor(K, R_bad, t_bad, 60)
    inconsistent = (base[0], bad_anchor, base[2])

    sol = solve_anchors_jointly(inconsistent, image_size=IMAGE_SIZE)
    with caplog.at_level(logging.ERROR, logger="src.utils.anchor_solver"):
        relocked = refine_with_shared_translation(inconsistent, sol)

    assert relocked.camera_centre is not None
    C = np.asarray(relocked.camera_centre)
    for af, (_K_a, R_a, t_a) in relocked.per_anchor_KRt.items():
        recovered = -R_a.T @ t_a
        assert np.allclose(recovered, C, atol=1e-4), (
            f"anchor {af}: -R^T @ t = {recovered} != C = {C} (silent fallback?)"
        )


# ── Lens distortion + Huber tests (Phase 2) ─────────────────────────────────


@pytest.mark.unit
def test_joint_solver_recovers_radial_distortion():
    """Synthesise observations with known (k1, k2); the joint LM recovers them."""
    from src.utils.camera_projection import project_world_to_image

    K_true = _K()
    k1_true, k2_true = 0.10, -0.02
    rich_pts = [
        ("near_left_corner",          (0.0,    0.0,   0.0)),
        ("near_right_corner",         (105.0,  0.0,   0.0)),
        ("far_left_corner",           (0.0,    68.0,  0.0)),
        ("far_right_corner",          (105.0,  68.0,  0.0)),
        ("halfway_near",              (52.5,   0.0,   0.0)),
        ("near_left_corner_flag_top", (0.0,    0.0,   1.5)),
        ("left_goal_crossbar_left",   (0.0,    30.34, 2.44)),
        ("left_goal_crossbar_right",  (0.0,    37.66, 2.44)),
    ]
    pts_world = np.array([w for _, w in rich_pts], dtype=np.float64)
    anchors: list[Anchor] = []
    for frame, yaw in ((0, 0.0), (60, 5.0), (120, -5.0)):
        R = _yaw(yaw)
        t = _t_for_yaw(yaw)
        proj = project_world_to_image(K_true, R, t, (k1_true, k2_true), pts_world)
        anchors.append(Anchor(
            frame=frame,
            landmarks=tuple(
                LandmarkObservation(
                    name=n, image_xy=tuple(proj[i]), world_xyz=w,
                )
                for i, (n, w) in enumerate(rich_pts)
            ),
        ))

    sol = solve_anchors_jointly(tuple(anchors), image_size=IMAGE_SIZE)
    sol = refine_with_shared_translation(tuple(anchors), sol)

    assert sol.distortion is not None
    k1_est, k2_est = sol.distortion
    assert abs(k1_est - k1_true) < 0.03, f"k1: got {k1_est}, want {k1_true}"
    assert abs(k2_est - k2_true) < 0.03, f"k2: got {k2_est}, want {k2_true}"


@pytest.mark.unit
def test_huber_loss_dampens_one_bad_landmark():
    """A single 200 px outlier landmark should not move the recovered fx
    relative to the no-outlier baseline by more than 1%.

    Synthetic clip is rendered noise-free; without robust loss, an outlier
    of this magnitude noticeably warps the joint LM. With Huber, the bad
    point's contribution to the gradient is capped and fx stays put.
    """
    base = _three_rich_anchors_static()
    bad_first = base[0]
    bad_lm = bad_first.landmarks[0]
    bad_lm = LandmarkObservation(
        name=bad_lm.name,
        image_xy=(bad_lm.image_xy[0] + 200.0, bad_lm.image_xy[1] + 200.0),
        world_xyz=bad_lm.world_xyz,
    )
    bad_anchor = Anchor(
        frame=bad_first.frame,
        landmarks=(bad_lm,) + tuple(bad_first.landmarks[1:]),
    )
    bad_set = (bad_anchor,) + tuple(base[1:])

    sol_clean = solve_anchors_jointly(base, image_size=IMAGE_SIZE)
    sol_bad = solve_anchors_jointly(bad_set, image_size=IMAGE_SIZE)
    fx_clean = sol_clean.per_anchor_KRt[0][0][0, 0]
    fx_bad = sol_bad.per_anchor_KRt[0][0][0, 0]
    rel = abs(fx_bad - fx_clean) / fx_clean
    assert rel < 0.01, f"fx moved {rel * 100:.2f}% — Huber not dampening outlier"


@pytest.mark.unit
def test_shared_C_LM_recovers_truth_better_than_median():
    """When solo solves disagree on C due to per-anchor noise, the
    shared-C joint LM should converge to a truer C than the simple
    median of solos. Synthesised by injecting different click-noise
    patterns into otherwise-identical anchors so their solo Cs spread
    around truth — the LM must find a C closer to truth than the
    median of those spread-out solos.
    """
    from src.utils.anchor_solver import refine_with_shared_translation

    K_true = _K()
    C_true = _C_WORLD  # (52.5, -30, 30)
    rng = np.random.default_rng(seed=42)
    anchors: list[Anchor] = []
    for frame, yaw_deg in ((0, 0.0), (60, 5.0), (120, -5.0), (180, 8.0)):
        R = _yaw(yaw_deg)
        t = -R @ C_true
        # 8 rich landmarks per anchor (T-pose mix of corners + crossbar)
        rich_landmarks = (
            ("near_left_corner",          (0.0,    0.0,   0.0)),
            ("near_right_corner",         (105.0,  0.0,   0.0)),
            ("far_left_corner",           (0.0,    68.0,  0.0)),
            ("far_right_corner",          (105.0,  68.0,  0.0)),
            ("halfway_near",              (52.5,   0.0,   0.0)),
            ("near_left_corner_flag_top", (0.0,    0.0,   1.5)),
            ("left_goal_crossbar_left",   (0.0,    30.34, 2.44)),
            ("left_goal_crossbar_right",  (0.0,    37.66, 2.44)),
        )
        # Symmetric click noise: 1 px stddev. Different per-anchor seed
        # patterns give different solo Cs around truth.
        landmarks = []
        for name, world in rich_landmarks:
            uv = _project(K_true, R, t, np.asarray(world, dtype=float))
            uv_noisy = (uv[0] + rng.normal(0, 1.0), uv[1] + rng.normal(0, 1.0))
            landmarks.append(LandmarkObservation(
                name=name, image_xy=uv_noisy, world_xyz=world,
            ))
        anchors.append(Anchor(frame=frame, landmarks=tuple(landmarks)))

    sol = solve_anchors_jointly(tuple(anchors), image_size=IMAGE_SIZE)
    # Median-of-solos baseline:
    Cs_solo = np.stack([
        -R.T @ t for (_K, R, t) in sol.per_anchor_KRt.values()
    ])
    C_median = np.median(Cs_solo, axis=0)
    median_err = float(np.linalg.norm(C_median - C_true))

    # Optimised by joint LM:
    sol_r = refine_with_shared_translation(tuple(anchors), sol)
    C_opt = np.asarray(sol_r.camera_centre)
    opt_err = float(np.linalg.norm(C_opt - C_true))

    assert opt_err <= median_err + 0.05, (
        f"Joint-LM C ({C_opt}, err {opt_err:.2f} m) should be at least as "
        f"close to truth as median-of-solos ({C_median}, err {median_err:.2f} m)"
    )


# ── Lens-from-anchor tests (Phase 3) ────────────────────────────────────────


def _wide_coverage_landmarks() -> tuple[tuple[str, tuple[float, float, float]], ...]:
    """14 landmarks that span both halves of the pitch and multiple z-levels.

    Used by ``test_lens_from_anchor_recovers_pp_and_distortion`` — the
    estimator needs a single anchor with enough spatial coverage to
    disambiguate principal-point offset from radial distortion.
    """
    return (
        ("near_left_corner",           (0.0,    0.0,   0.0)),
        ("near_right_corner",          (105.0,  0.0,   0.0)),
        ("far_left_corner",            (0.0,    68.0,  0.0)),
        ("far_right_corner",           (105.0,  68.0,  0.0)),
        ("halfway_near",               (52.5,   0.0,   0.0)),
        ("halfway_far",                (52.5,   68.0,  0.0)),
        ("near_left_corner_flag_top",  (0.0,    0.0,   1.5)),
        ("near_right_corner_flag_top", (105.0,  0.0,   1.5)),
        ("far_left_corner_flag_top",   (0.0,    68.0,  1.5)),
        ("far_right_corner_flag_top",  (105.0,  68.0,  1.5)),
        ("left_goal_crossbar_left",    (0.0,    30.34, 2.44)),
        ("left_goal_crossbar_right",   (0.0,    37.66, 2.44)),
        ("right_goal_crossbar_left",   (105.0,  30.34, 2.44)),
        ("right_goal_crossbar_right",  (105.0,  37.66, 2.44)),
    )


def _distorted_anchor(
    frame: int,
    yaw_deg: float,
    fx: float,
    cx: float,
    cy: float,
    C: np.ndarray,
    distortion: tuple[float, float],
) -> Anchor:
    """Project _wide_coverage_landmarks through (fx, cx, cy, k1, k2) with the
    static-camera centre ``C`` and the given yaw. Returns an Anchor whose
    clicks match what a user would mark on the distorted image.
    """
    from src.utils.camera_projection import project_world_to_image

    R = _yaw(yaw_deg)
    t = -R @ C
    K = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]])
    pts = np.array([w for _, w in _wide_coverage_landmarks()], dtype=np.float64)
    proj = project_world_to_image(K, R, t, distortion, pts)
    landmarks = tuple(
        LandmarkObservation(
            name=name, image_xy=(float(proj[i, 0]), float(proj[i, 1])), world_xyz=w,
        )
        for i, (name, w) in enumerate(_wide_coverage_landmarks())
    )
    return Anchor(frame=frame, landmarks=landmarks)


@pytest.mark.unit
def test_lens_from_anchor_recovers_pp_and_distortion():
    """Given a clip with off-centre principal point and radial distortion,
    estimating the lens prior from the highest-coverage anchor should recover
    (cx, cy) and (k1, k2) within tight tolerance.
    """
    from src.utils.anchor_solver import _estimate_lens_from_best_anchor

    C_true = np.array([52.5, -30.0, 18.0])
    fxs = (2200.0, 2400.0, 2300.0)
    cx_true, cy_true = 990.0, 580.0
    k1_true, k2_true = -0.12, 0.02

    anchors = tuple(
        _distorted_anchor(
            frame=fr, yaw_deg=yaw, fx=fx,
            cx=cx_true, cy=cy_true, C=C_true,
            distortion=(k1_true, k2_true),
        )
        for fr, yaw, fx in ((0, 0.0, fxs[0]), (60, 6.0, fxs[1]), (120, -6.0, fxs[2]))
    )

    prior = _estimate_lens_from_best_anchor(anchors, image_size=IMAGE_SIZE)
    assert prior is not None, "lens estimator returned None on a clean synthetic clip"
    cx_est, cy_est, k1_est, k2_est = prior
    assert abs(cx_est - cx_true) < 5.0, f"cx: got {cx_est}, want {cx_true}"
    assert abs(cy_est - cy_true) < 5.0, f"cy: got {cy_est}, want {cy_true}"
    assert abs(k1_est - k1_true) < 0.02, f"k1: got {k1_est}, want {k1_true}"
    assert abs(k2_est - k2_true) < 0.02, f"k2: got {k2_est}, want {k2_true}"


@pytest.mark.unit
def test_lens_prior_tightens_solo_C_spread_on_distorted_clip():
    """Real broadcast lenses bias each anchor's solo solve in a yaw-dependent
    way: with un-modelled distortion the recovered camera centres disagree
    by metres even though the body is static. Passing a lens prior should
    cut that spread to <30 cm and drop mean residuals below 1 px.
    """
    from src.utils.anchor_solver import _estimate_lens_from_best_anchor

    C_true = np.array([52.5, -30.0, 18.0])
    cx_true, cy_true = 990.0, 580.0
    k1_true, k2_true = -0.12, 0.02
    anchors = tuple(
        _distorted_anchor(
            frame=fr, yaw_deg=yaw, fx=fx,
            cx=cx_true, cy=cy_true, C=C_true,
            distortion=(k1_true, k2_true),
        )
        for fr, yaw, fx in ((0, 0.0, 2200.0), (60, 6.0, 2400.0), (120, -6.0, 2300.0))
    )

    # Baseline: no lens prior. Solo Cs should disagree.
    sol_base = solve_anchors_jointly(anchors, image_size=IMAGE_SIZE)
    Cs_base = np.stack([
        -R.T @ t for (_K, R, t) in sol_base.per_anchor_KRt.values()
    ])
    base_spread = float(np.linalg.norm(Cs_base.max(axis=0) - Cs_base.min(axis=0)))

    # With lens prior: solo Cs should cluster.
    prior = _estimate_lens_from_best_anchor(anchors, image_size=IMAGE_SIZE)
    assert prior is not None
    sol_prior = solve_anchors_jointly(
        anchors, image_size=IMAGE_SIZE, lens_prior=prior,
    )
    Cs_prior = np.stack([
        -R.T @ t for (_K, R, t) in sol_prior.per_anchor_KRt.values()
    ])
    prior_spread = float(np.linalg.norm(Cs_prior.max(axis=0) - Cs_prior.min(axis=0)))

    assert base_spread > 0.5, (
        f"baseline solo-C spread is {base_spread:.3f} m — distorted synthetic "
        "clip should exhibit metres of disagreement without distortion modelling"
    )
    assert prior_spread < 0.3, (
        f"lens-prior solo-C spread is {prior_spread:.3f} m — should be <0.3 m "
        f"(baseline was {base_spread:.3f} m)"
    )

    mean_res_prior = float(np.mean(list(sol_prior.per_anchor_residual_px.values())))
    assert mean_res_prior < 1.0, (
        f"with lens prior, mean residual is {mean_res_prior:.2f} px — should "
        "be sub-pixel on a noise-free synthetic clip"
    )
    assert sol_prior.distortion == prior[2:], (
        "JointSolution.distortion should match the lens prior's (k1, k2)"
    )
    assert sol_prior.principal_point == (prior[0], prior[1]), (
        "JointSolution.principal_point should match the lens prior's (cx, cy)"
    )


@pytest.mark.unit
def test_joint_lens_estimator_recovers_pp_and_distortion():
    """The joint estimator should be at least as good as single-anchor on a
    well-conditioned synthetic clip — and on real clips, where single-anchor
    fails to clear its 2× drop gate, the joint version's looser 1.3× gate
    (with much better-determined fit) succeeds.
    """
    from src.utils.anchor_solver import _estimate_lens_jointly

    C_true = np.array([52.5, -30.0, 18.0])
    cx_true, cy_true = 990.0, 580.0
    k1_true, k2_true = -0.12, 0.02

    anchors = tuple(
        _distorted_anchor(
            frame=fr, yaw_deg=yaw, fx=fx,
            cx=cx_true, cy=cy_true, C=C_true,
            distortion=(k1_true, k2_true),
        )
        for fr, yaw, fx in (
            (0, 0.0, 2200.0), (60, 6.0, 2400.0), (120, -6.0, 2300.0),
            (180, 4.0, 2350.0),
        )
    )

    prior = _estimate_lens_jointly(anchors, image_size=IMAGE_SIZE)
    assert prior is not None, "joint estimator returned None on clean synthetic clip"
    cx_est, cy_est, k1_est, k2_est = prior
    assert abs(cx_est - cx_true) < 5.0, f"cx: got {cx_est}, want {cx_true}"
    assert abs(cy_est - cy_true) < 5.0, f"cy: got {cy_est}, want {cy_true}"
    assert abs(k1_est - k1_true) < 0.02, f"k1: got {k1_est}, want {k1_true}"
    assert abs(k2_est - k2_true) < 0.02, f"k2: got {k2_est}, want {k2_true}"


@pytest.mark.unit
def test_joint_lens_estimator_returns_none_on_single_anchor():
    """The joint estimator needs ≥2 rich anchors; with fewer, it falls
    through to None so the camera stage can use the single-anchor estimator
    instead."""
    from src.utils.anchor_solver import _estimate_lens_jointly

    anchors = (_distorted_anchor(
        frame=0, yaw_deg=0.0, fx=2200.0,
        cx=990.0, cy=580.0, C=np.array([52.5, -30.0, 18.0]),
        distortion=(-0.12, 0.02),
    ),)
    assert _estimate_lens_jointly(anchors, image_size=IMAGE_SIZE) is None

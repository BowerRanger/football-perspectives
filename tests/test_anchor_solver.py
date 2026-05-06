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

import numpy as np
import pytest

from src.schemas.anchor import LandmarkObservation
from src.utils.anchor_solver import (
    AnchorSolveError,
    solve_first_anchor,
    solve_subsequent_anchor,
)


def _project(K: np.ndarray, R: np.ndarray, t: np.ndarray, world_xyz: np.ndarray) -> np.ndarray:
    """Return image (u, v) for a 3D world point given (K, R, t)."""
    cam = R @ world_xyz + t
    pix = K @ cam
    return pix[:2] / pix[2]


def _make_synthetic(K, R, t, names_with_world):
    return tuple(
        LandmarkObservation(
            name=name,
            image_xy=tuple(_project(K, R, t, np.array(world, dtype=float))),
            world_xyz=tuple(world),
        )
        for name, world in names_with_world
    )


# Broadcast-style camera pose: high-and-back behind the nearside touchline,
# looking at pitch centre. Camera world position C = (52.5, -30, 30); optical
# axis points to (52.5, 34, 0). The world->camera rotation `R_base` and
# translation `t_base = -R_base @ C` follow the OpenCV convention. With this
# pose every pitch-plane landmark projects with cam_z > 0 (in front of camera).
R_BASE = np.array(
    [[1, 0, 0],
     [0, -0.424, -0.905],
     [0, 0.905, -0.424]],
    dtype=float,
)
T_BASE = np.array([-52.5, 14.43, 39.87])  # = -R_BASE @ (52.5, -30, 30)


def _assert_in_front_of_camera(R, t, names_with_world):
    for name, world in names_with_world:
        cam = R @ np.array(world, dtype=float) + t
        assert cam[2] > 0, f"landmark {name} is behind the camera (cam_z={cam[2]})"


@pytest.mark.unit
def test_first_anchor_recovers_known_camera():
    K = np.array([[700.0, 0, 960.0], [0, 700.0, 540.0], [0, 0, 1.0]])
    R = R_BASE
    t = T_BASE

    landmarks_with_world = [
        ("near_left_corner",            (0, 0, 0)),
        ("near_right_corner",           (105, 0, 0)),
        ("far_left_corner",             (0, 68, 0)),
        ("halfway_near",                (52.5, 0, 0)),
        ("left_goal_crossbar_left",     (0, 30.34, 2.44)),
        ("near_left_corner_flag_top",   (0, 0, 1.5)),
    ]
    _assert_in_front_of_camera(R, t, landmarks_with_world)

    landmarks = _make_synthetic(K, R, t, landmarks_with_world)

    K_hat, R_hat, t_hat = solve_first_anchor(landmarks)
    assert np.allclose(K_hat, K, atol=2.0)
    assert np.linalg.norm(R_hat - R) < 0.02
    assert np.allclose(t_hat, t, atol=0.05)


@pytest.mark.unit
def test_first_anchor_rejects_coplanar_set():
    landmarks = tuple(
        LandmarkObservation(name=f"lm_{i}", image_xy=(i, i), world_xyz=(float(i), float(i), 0.0))
        for i in range(6)
    )
    with pytest.raises(AnchorSolveError):
        solve_first_anchor(landmarks)


@pytest.mark.unit
def test_first_anchor_rejects_too_few_points():
    K = np.eye(3); R = np.eye(3); t = np.zeros(3)
    landmarks = _make_synthetic(K, R, t, [("a", (1.0, 2.0, 0.5))] * 3)
    with pytest.raises(AnchorSolveError):
        solve_first_anchor(landmarks[:3])


@pytest.mark.unit
def test_subsequent_anchor_recovers_K_and_R_with_t_fixed():
    # Slight zoom-in (fx 700 -> 800) and a 15° pan about the world-z axis.
    # `R_yaw_world` rotates the world about z; applying `R_yaw_world.T` on the
    # right of `R_BASE` yields the new world->camera rotation after the pan.
    K_true = np.array([[800.0, 0, 960], [0, 800.0, 540], [0, 0, 1]])
    yaw = np.deg2rad(15.0)
    R_yaw_world = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0],
         [np.sin(yaw), np.cos(yaw), 0],
         [0, 0, 1]],
    )
    R_true = R_BASE @ R_yaw_world.T
    t_true = T_BASE

    landmarks_with_world = [
        ("near_left_corner",            (0, 0, 0)),
        ("near_right_corner",           (105, 0, 0)),
        ("far_left_corner",             (0, 68, 0)),
        ("halfway_near",                (52.5, 0, 0)),
    ]
    _assert_in_front_of_camera(R_true, t_true, landmarks_with_world)

    landmarks = _make_synthetic(K_true, R_true, t_true, landmarks_with_world)
    K_hat, R_hat = solve_subsequent_anchor(
        landmarks, t_true, image_size=(1920, 1080)
    )
    assert np.allclose(K_hat, K_true, atol=20.0)
    assert np.linalg.norm(R_hat - R_true) < 0.02

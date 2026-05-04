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


@pytest.mark.unit
def test_first_anchor_recovers_known_camera():
    K = np.array([[1820.0, 0, 960.0], [0, 1820.0, 540.0], [0, 0, 1.0]])
    R = np.eye(3)
    R = np.array(
        [[1, 0, 0],
         [0, 0, 1],
         [0, -1, 0]],
        dtype=float,
    )  # camera looking down +y
    t = np.array([-52.5, 100.0, 22.0])  # pitch metres

    landmarks = _make_synthetic(K, R, t, [
        ("near_left_corner",            (0, 0, 0)),
        ("near_right_corner",           (105, 0, 0)),
        ("far_left_corner",             (0, 68, 0)),
        ("halfway_near",                (52.5, 0, 0)),
        ("left_goal_crossbar_left",     (0, 30.34, 2.44)),
        ("near_left_corner_flag_top",   (0, 0, 1.5)),
    ])

    K_hat, R_hat, t_hat = solve_first_anchor(landmarks)
    assert np.allclose(K_hat, K, atol=2.0)
    assert np.allclose(R_hat, R, atol=1e-3)
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
    K_true = np.array([[1900.0, 0, 960], [0, 1900.0, 540], [0, 0, 1]])
    angle = np.deg2rad(15.0)  # 15° pan from first anchor
    R_true = np.array(
        [[np.cos(angle), 0, np.sin(angle)],
         [0, 1, 0],
         [-np.sin(angle), 0, np.cos(angle)]],
    ) @ np.array(
        [[1, 0, 0],
         [0, 0, 1],
         [0, -1, 0]],
        dtype=float,
    )
    t_true = np.array([-52.5, 100.0, 22.0])

    landmarks = _make_synthetic(K_true, R_true, t_true, [
        ("near_left_corner",            (0, 0, 0)),
        ("near_right_corner",           (105, 0, 0)),
        ("far_left_corner",             (0, 68, 0)),
        ("halfway_near",                (52.5, 0, 0)),
    ])
    K_hat, R_hat = solve_subsequent_anchor(landmarks, t_true)
    assert np.allclose(K_hat, K_true, atol=10.0)
    assert np.allclose(R_hat, R_true, atol=1e-2)

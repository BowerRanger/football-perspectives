from __future__ import annotations

import numpy as np

from src.utils.virtual_cameras import intrinsics_from_fov, look_at_view


def test_intrinsics_from_fov_centres_principal_point() -> None:
    K = np.array(intrinsics_from_fov(90.0, (1920, 1080)))
    # 90° horizontal FOV over 1920 px → fx = 960.
    assert np.isclose(K[0, 0], 960.0)
    np.testing.assert_allclose([K[0, 2], K[1, 2]], [960.0, 540.0])
    assert K[2, 2] == 1.0


def test_look_at_view_is_proper_rotation_and_centres_camera() -> None:
    center = np.array([0.0, -5.0, 1.7])
    target = np.array([0.0, 0.0, 0.0])
    R, t = look_at_view(center, target)

    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)
    # C = -R^T t recovers the requested centre.
    np.testing.assert_allclose(-R.T @ t, center, atol=1e-9)


def test_look_at_view_optical_axis_points_at_target() -> None:
    center = np.array([0.0, -5.0, 0.0])
    target = np.array([0.0, 0.0, 0.0])
    R, _ = look_at_view(center, target)
    # Camera +Z (row 2 of R) is the optical ray; should point center→target (+y).
    np.testing.assert_allclose(R[2], [0.0, 1.0, 0.0], atol=1e-9)


def test_look_at_view_handles_target_along_world_up() -> None:
    center = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 5.0])
    R, _ = look_at_view(center, target)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)
    np.testing.assert_allclose(R[2], [0.0, 0.0, 1.0], atol=1e-9)


def test_intrinsics_from_fov_non_square_60deg() -> None:
    import math
    K = intrinsics_from_fov(60.0, (1280, 720))
    expected_fx = (1280 / 2.0) / math.tan(math.radians(60.0) / 2.0)
    assert np.isclose(K[0][0], expected_fx)
    assert np.isclose(K[0][2], 640.0)
    assert np.isclose(K[1][2], 360.0)


def test_look_at_view_raises_on_coincident_center_target() -> None:
    import pytest
    with pytest.raises(ValueError):
        look_at_view(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))


from src.schemas.ball_track import BallFrame, BallTrack
from src.schemas.smpl_world import SmplWorldTrack
from src.utils.virtual_cameras import RigConfig, build_ots_track, build_pov_track


def _straight_standing_track(n: int = 3) -> SmplWorldTrack:
    """Return a minimal SmplWorldTrack for a player standing at pitch xy=(10,20).

    ``root_R`` converts SMPL canonical y-up to pitch z-up (rotate +90° around
    world-x). ``root_t`` is the pelvis in pitch metres; z is chosen so the
    feet sit on the ground plane (z=0) and the head lands at ~1.5 m.
    """
    frames = np.arange(n, dtype=np.int64)
    # y-up canonical -> z-up world: rotate +90° around x
    # [x, y, z]_can -> [x, -z, y]_world
    root_R_single = np.array([[1.0, 0.0, 0.0],
                               [0.0, 0.0, -1.0],
                               [0.0, 1.0,  0.0]])
    # Pelvis z so SMPL rest-pose feet touch z=0 (l_foot canonical y ≈ -0.939)
    # After root_R: foot_z = root_R[2] @ foot_yup + root_t_z
    #   foot_z = 0  =>  root_t_z = -root_R[2] @ foot_yup = 0.939
    pelvis_z = 0.939
    return SmplWorldTrack(
        player_id="P001",
        frames=frames,
        betas=np.zeros(10),
        thetas=np.zeros((n, 24, 3)),
        root_R=np.broadcast_to(root_R_single, (n, 3, 3)).copy(),
        root_t=np.tile(np.array([10.0, 20.0, pelvis_z]), (n, 1)),
        confidence=np.ones(n),
        shot_id="shot_01",
    )


def test_build_pov_track_centres_camera_near_head_height() -> None:
    track = _straight_standing_track()
    cam = build_pov_track(track, RigConfig(), image_size=(1920, 1080), fps=30.0, clip_id="P001_pov")

    assert cam.clip_id == "P001_pov"
    assert len(cam.frames) == 3
    f0 = cam.frames[0]
    assert f0.t is not None
    R = np.array(f0.R)
    center = -R.T @ np.array(f0.t)
    assert np.isclose(center[0], 10.0, atol=0.2)
    assert np.isclose(center[1], 20.0, atol=0.2)
    assert 1.4 < center[2] < 2.0


def test_build_ots_track_aims_at_ball_when_present() -> None:
    track = _straight_standing_track()
    ball = BallTrack(
        clip_id="shot_01",
        fps=30.0,
        frames=(
            BallFrame(frame=0, world_xyz=(12.0, 22.0, 0.0), state="grounded", confidence=1.0),
            BallFrame(frame=1, world_xyz=(12.0, 22.0, 0.0), state="grounded", confidence=1.0),
            BallFrame(frame=2, world_xyz=(12.0, 22.0, 0.0), state="grounded", confidence=1.0),
        ),
        flight_segments=(),
    )
    cam = build_ots_track(track, ball, RigConfig(), image_size=(1920, 1080), fps=30.0, clip_id="P001_ots")

    f0 = cam.frames[0]
    R = np.array(f0.R)
    center = -R.T @ np.array(f0.t)
    expect_dir = np.array([12.0, 22.0, 0.0]) - center
    expect_dir = expect_dir / np.linalg.norm(expect_dir)
    np.testing.assert_allclose(R[2], expect_dir, atol=1e-6)


def test_build_ots_track_without_ball_uses_forward_fallback() -> None:
    track = _straight_standing_track()
    cam = build_ots_track(track, None, RigConfig(), image_size=(1920, 1080), fps=30.0, clip_id="P001_ots")
    assert len(cam.frames) == 3
    R = np.array(cam.frames[0].R)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)

import numpy as np
import pytest
from src.utils.ball_detector import BallDetector, FakeBallDetector


def test_fake_ball_detector_returns_position():
    frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(5)]
    positions = [(50.0, 60.0), (55.0, 65.0), None, (60.0, 70.0), (65.0, 75.0)]
    detector = FakeBallDetector(positions)
    results = [detector.detect(f) for f in frames]
    assert results[0] == pytest.approx((50.0, 60.0))
    assert results[2] is None


def test_ball_detector_is_abstract():
    import inspect
    assert inspect.isabstract(BallDetector)


import numpy as np
from src.stages.sync import cross_correlate_trajectories, project_ball_to_pitch

def test_cross_correlate_finds_correct_offset():
    """Ball trajectory in traj_b lags traj_a by 3 frames.

    traj_a's pattern starts at index 2; traj_b's matching pattern starts at index 5.
    The returned offset is +3 (traj_b event is 3 frames later than the reference).
    """
    traj_a = np.array([0,0,1,3,5,3,1,0,0,0], dtype=float)
    traj_b = np.zeros(10, dtype=float)
    traj_b[5:9] = [1, 3, 5, 3]
    offset, confidence = cross_correlate_trajectories(traj_a, traj_b)
    assert offset == 3
    assert confidence > 0.7

def test_cross_correlate_returns_zero_for_identical():
    traj = np.array([0,1,2,3,2,1,0], dtype=float)
    offset, confidence = cross_correlate_trajectories(traj, traj.copy())
    assert offset == 0
    assert confidence > 0.99

def test_cross_correlate_low_confidence_for_noise():
    rng = np.random.default_rng(42)
    traj_a = rng.random(50)
    traj_b = rng.random(50)
    _, confidence = cross_correlate_trajectories(traj_a, traj_b)
    assert confidence < 0.5

def test_project_ball_to_pitch_returns_2d():
    import cv2
    K = np.array([[1500,0,960],[0,1500,540],[0,0,1]], dtype=np.float32)
    rvec = np.array([0.05,0.15,0.0], dtype=np.float32)
    tvec = np.array([-52.5,-34.0,60.0], dtype=np.float32)

    pt_3d = np.array([[30.0, 20.0, 0.0]], dtype=np.float32)
    pt_2d, _ = cv2.projectPoints(pt_3d, rvec, tvec, K, None)
    pixel = pt_2d.reshape(2)

    from src.schemas.calibration import CameraFrame
    frame_cal = CameraFrame(
        frame=0,
        intrinsic_matrix=K.tolist(),
        rotation_vector=rvec.tolist(),
        translation_vector=tvec.tolist(),
        reprojection_error=0.0,
        num_correspondences=8,
        confidence=1.0,
    )
    pitch_pos = project_ball_to_pitch(pixel, frame_cal)
    assert pitch_pos is not None
    assert np.allclose(pitch_pos, [30.0, 20.0], atol=0.1)

import numpy as np
import cv2
import pytest
from src.utils.pitch import FIFA_LANDMARKS, PITCH_LENGTH, PITCH_WIDTH
from src.utils.camera import build_projection_matrix, project_to_pitch, reprojection_error

def _synthetic_camera():
    K = np.array([[1500, 0, 960], [0, 1500, 540], [0, 0, 1]], dtype=np.float32)
    rvec = np.array([0.05, 0.15, 0.0], dtype=np.float32)
    tvec = np.array([-52.5, -34.0, 60.0], dtype=np.float32)
    return K, rvec, tvec

def test_pitch_constants():
    assert PITCH_LENGTH == 105.0
    assert PITCH_WIDTH == 68.0
    assert "top_left_corner" in FIFA_LANDMARKS
    assert "center_spot" in FIFA_LANDMARKS
    pt = FIFA_LANDMARKS["top_left_corner"]
    assert pt[2] == 0.0  # z=0, pitch is ground plane

def test_build_projection_matrix_shape():
    K, rvec, tvec = _synthetic_camera()
    P = build_projection_matrix(K, rvec, tvec)
    assert P.shape == (3, 4)

def test_reprojection_error_zero_for_perfect_fit():
    K, rvec, tvec = _synthetic_camera()
    pts_3d = np.array([[0,0,0],[105,0,0],[52.5,34,0]], dtype=np.float32)
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, None)
    pts_2d = pts_2d.reshape(-1, 2)
    err = reprojection_error(pts_3d, pts_2d, K, rvec, tvec)
    assert err < 0.01

def test_project_to_pitch_round_trips():
    K, rvec, tvec = _synthetic_camera()
    # A known pitch point projected to image, then projected back to pitch
    pt_3d = np.array([30.0, 20.0, 0.0], dtype=np.float32)
    pt_2d, _ = cv2.projectPoints(pt_3d.reshape(1,1,3), rvec, tvec, K, None)
    pt_2d = pt_2d.reshape(2)
    recovered = project_to_pitch(pt_2d, K, rvec, tvec)
    assert np.allclose(recovered, pt_3d[:2], atol=0.05)


from src.stages.calibration import calibrate_frame, PitchKeypointDetector
from src.schemas.calibration import CameraFrame

def _make_synthetic_correspondences():
    """Project known pitch landmarks with a synthetic camera to get 2D points."""
    K = np.array([[1500,0,960],[0,1500,540],[0,0,1]], dtype=np.float32)
    rvec = np.array([0.05, 0.15, 0.0], dtype=np.float32)
    tvec = np.array([-52.5, -34.0, 60.0], dtype=np.float32)

    landmark_names = [
        "top_left_corner", "top_right_corner", "bottom_left_corner",
        "bottom_right_corner", "center_spot", "left_penalty_spot",
        "right_penalty_spot", "halfway_top", "halfway_bottom",
    ]
    pts_3d = np.array([FIFA_LANDMARKS[n] for n in landmark_names], dtype=np.float32)
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, None)
    pts_2d = pts_2d.reshape(-1, 2)
    correspondences = {name: pts_2d[i] for i, name in enumerate(landmark_names)}
    return correspondences, K

def test_calibrate_frame_recovers_low_reprojection_error():
    correspondences, _ = _make_synthetic_correspondences()
    result = calibrate_frame(
        correspondences=correspondences,
        landmarks_3d=FIFA_LANDMARKS,
        image_shape=(1080, 1920),
    )
    assert result is not None
    assert result.reprojection_error < 2.0  # near-perfect on noise-free data
    assert result.confidence > 0.8
    assert result.num_correspondences >= 4

def test_calibrate_frame_returns_none_with_too_few_points():
    result = calibrate_frame(
        correspondences={"top_left_corner": np.array([100.0, 100.0])},
        landmarks_3d=FIFA_LANDMARKS,
        image_shape=(1080, 1920),
    )
    assert result is None

def test_pitch_keypoint_detector_is_abstract():
    import inspect
    assert inspect.isabstract(PitchKeypointDetector)

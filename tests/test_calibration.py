import numpy as np
import cv2
import pytest
from pathlib import Path
from src.utils.pitch import FIFA_LANDMARKS, PITCH_LENGTH, PITCH_WIDTH
from src.utils.camera import build_projection_matrix, project_to_pitch, reprojection_error, camera_world_position, is_camera_valid

def _synthetic_camera():
    K = np.array([[1500, 0, 960], [0, 1500, 540], [0, 0, 1]], dtype=np.float32)
    rvec = np.array([0.05, 0.15, 0.0], dtype=np.float32)
    tvec = np.array([-52.5, -34.0, 60.0], dtype=np.float32)
    return K, rvec, tvec

def test_pitch_constants():
    assert PITCH_LENGTH == 105.0
    assert PITCH_WIDTH == 68.0
    assert "corner_near_left" in FIFA_LANDMARKS
    assert "center_spot" in FIFA_LANDMARKS
    pt = FIFA_LANDMARKS["corner_near_left"]
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
from src.stages.calibration import CameraCalibrationStage

def _make_synthetic_correspondences():
    """Project known pitch landmarks with a physically valid broadcast camera.

    Camera is placed at world position (52.5, -20, 25) — on the pitch centreline,
    20 m behind the near touchline and 25 m above the pitch — and aimed at the
    pitch centre.  Focal length 1212px ≈ 0.55 × image diagonal so that the
    solver's focal-length grid includes an exact match.  All pitch coordinates
    use z-up convention.
    """
    C = np.array([52.5, -20.0, 25.0])
    target = np.array([52.5, 34.0, 0.0])
    cam_z_world = target - C
    cam_z_world /= np.linalg.norm(cam_z_world)
    world_up = np.array([0.0, 0.0, 1.0])
    cam_x_world = np.cross(cam_z_world, world_up)
    cam_x_world /= np.linalg.norm(cam_x_world)
    cam_y_world = np.cross(cam_z_world, cam_x_world)
    R = np.array([cam_x_world, cam_y_world, cam_z_world])
    rvec_raw, _ = cv2.Rodrigues(R)
    rvec = rvec_raw.flatten().astype(np.float32)
    tvec = (-R @ C).astype(np.float32)
    # Use diagonal*0.55 ≈ 1212px — a value in the solver's focal candidate grid
    diagonal = float(np.sqrt(1080 ** 2 + 1920 ** 2))
    fx = diagonal * 0.55
    K = np.array([[fx, 0, 960], [0, fx, 540], [0, 0, 1]], dtype=np.float32)

    landmark_names = [
        "corner_far_left", "corner_far_right", "center_spot",
        "left_penalty_spot", "right_penalty_spot", "halfway_near",
        "halfway_far", "centre_circle_near", "centre_circle_far",
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
    assert len(result.tracked_landmark_types) == result.num_correspondences
    assert "center_spot" in result.tracked_landmark_types

def test_calibrate_frame_returns_none_with_too_few_points():
    result = calibrate_frame(
        correspondences={"corner_near_left": np.array([100.0, 100.0])},
        landmarks_3d=FIFA_LANDMARKS,
        image_shape=(1080, 1920),
    )
    assert result is None

def test_pitch_keypoint_detector_is_abstract():
    import inspect
    assert inspect.isabstract(PitchKeypointDetector)


def _create_dummy_clip(path: Path, fps: float, frames: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (640, 480))
    for _ in range(frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_calibration_without_detector_logs_warning_and_creates_empty_stubs(tmp_path, caplog):
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    _create_dummy_clip(shots_dir / "shot_001.mp4", fps=25.0, frames=100)
    _create_dummy_clip(shots_dir / "shot_002.mp4", fps=25.0, frames=120)

    stage = CameraCalibrationStage(config={}, output_dir=tmp_path, detector=None)
    with caplog.at_level("WARNING"):
        stage.run()

    assert sum("no pitch keypoint detector configured" in r.message.lower() for r in caplog.records) == 1

    from src.schemas.calibration import CalibrationResult
    result_1 = CalibrationResult.load(tmp_path / "calibration" / "shot_001_calibration.json")
    result_2 = CalibrationResult.load(tmp_path / "calibration" / "shot_002_calibration.json")
    assert result_1.frames == []
    assert result_2.frames == []
    assert result_1.camera_type == "static"


def test_calibration_strict_mode_raises_without_detector(tmp_path):
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    _create_dummy_clip(shots_dir / "shot_001.mp4", fps=25.0, frames=100)

    stage = CameraCalibrationStage(
        config={"calibration": {"require_detector": True}},
        output_dir=tmp_path,
        detector=None,
    )

    with pytest.raises(RuntimeError, match="require_detector"):
        stage.run()


def test_calibrate_shot_short_circuits_without_detector(tmp_path, monkeypatch):
    stage = CameraCalibrationStage(config={}, output_dir=tmp_path, detector=None)

    def fail_video_capture(*args, **kwargs):
        raise AssertionError("VideoCapture should not be called when detector is None")

    monkeypatch.setattr("src.stages.calibration.cv2.VideoCapture", fail_video_capture)
    result = stage._calibrate_shot("shot_001", "shots/shot_001.mp4", 5, 15.0)

    assert result.shot_id == "shot_001"
    assert result.camera_type == "static"
    assert result.frames == []


def test_goal_crossbar_landmarks_are_off_plane():
    for name in ["left_goal_near_post_top", "left_goal_far_post_top",
                 "right_goal_near_post_top", "right_goal_far_post_top"]:
        assert name in FIFA_LANDMARKS
        assert FIFA_LANDMARKS[name][2] == 2.44


def test_corner_flag_landmarks_are_off_plane():
    for name in ["corner_near_left_flag_top", "corner_near_right_flag_top",
                 "corner_far_left_flag_top", "corner_far_right_flag_top"]:
        assert name in FIFA_LANDMARKS
        assert FIFA_LANDMARKS[name][2] == 1.5


def test_near_landmarks_have_lower_y_than_far():
    assert FIFA_LANDMARKS["corner_near_left"][1] < FIFA_LANDMARKS["corner_far_left"][1]
    assert FIFA_LANDMARKS["halfway_near"][1] < FIFA_LANDMARKS["halfway_far"][1]
    assert FIFA_LANDMARKS["centre_circle_near"][1] < FIFA_LANDMARKS["centre_circle_far"][1]
    assert FIFA_LANDMARKS["left_goal_near_post_base"][1] < FIFA_LANDMARKS["left_goal_far_post_base"][1]


def _valid_broadcast_camera():
    """Return (rvec, tvec) for a physically plausible broadcast camera.

    Camera is placed at world position (52.5, -10, 30) — on the pitch centreline,
    10 m behind the near touchline and 30 m above the pitch — and aimed at the
    pitch centre (52.5, 34, 0).  All pitch coordinates use z-up convention.
    """
    C = np.array([52.5, -10.0, 30.0])
    target = np.array([52.5, 34.0, 0.0])
    cam_z_world = target - C
    cam_z_world /= np.linalg.norm(cam_z_world)
    world_up = np.array([0.0, 0.0, 1.0])
    cam_x_world = np.cross(cam_z_world, world_up)
    cam_x_world /= np.linalg.norm(cam_x_world)
    cam_y_world = np.cross(cam_z_world, cam_x_world)
    R = np.array([cam_x_world, cam_y_world, cam_z_world])
    rvec_raw, _ = cv2.Rodrigues(R)
    rvec = rvec_raw.flatten().astype(np.float64)
    tvec = (-R @ C).astype(np.float64)
    return rvec, tvec


def test_camera_world_position_synthetic():
    rvec, tvec = _valid_broadcast_camera()
    pos = camera_world_position(rvec, tvec)
    assert pos[2] > 0  # camera above pitch
    assert pos.shape == (3,)


def test_is_camera_valid_accepts_normal_broadcast_camera():
    rvec, tvec = _valid_broadcast_camera()
    assert is_camera_valid(rvec, tvec, min_height=3.0, max_height=80.0)


def test_is_camera_valid_rejects_camera_below_pitch():
    # Identity rotation; tvec z positive → world position z negative (below pitch plane).
    rvec_id = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    tvec_below = np.array([0.0, 0.0, 10.0], dtype=np.float32)
    assert not is_camera_valid(rvec_id, tvec_below, min_height=3.0, max_height=80.0)


def test_is_camera_valid_rejects_camera_looking_up():
    # Identity rotation, tvec z negative → camera at z=+30 (above pitch) but optical axis
    # points in world +z (upward), which is physically wrong for a broadcast camera.
    rvec_id = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    tvec_up = np.array([0.0, 0.0, -30.0], dtype=np.float32)
    assert not is_camera_valid(rvec_id, tvec_up, min_height=3.0, max_height=80.0)


def test_calibrate_frame_produces_camera_above_pitch():
    """The solver must place the camera above the pitch (z > 0)."""
    correspondences, _ = _make_synthetic_correspondences()
    result = calibrate_frame(
        correspondences=correspondences,
        landmarks_3d=FIFA_LANDMARKS,
        image_shape=(1080, 1920),
    )
    assert result is not None
    from src.utils.camera import camera_world_position
    pos = camera_world_position(
        np.array(result.rotation_vector),
        np.array(result.translation_vector),
    )
    assert pos[2] > 0, f"Camera placed below pitch at z={pos[2]:.1f}"


# ── Player height disambiguation tests ──────────────────────────────────────

from src.utils.player_height import score_player_heights


def test_score_player_heights_correct_solution_scores_higher():
    """A camera solution that produces ~1.8m player heights should score well."""
    K = np.array([[2000, 0, 960], [0, 2000, 540], [0, 0, 1]], dtype=np.float64)
    rvec = np.array([1.3, -0.1, 0.0], dtype=np.float64)
    tvec = np.array([-30.0, -34.0, 40.0], dtype=np.float64)

    # Create a synthetic player bbox by projecting a standing person at (30, 20)
    foot_3d = np.array([[30.0, 20.0, 0.0]], dtype=np.float64)
    head_3d = np.array([[30.0, 20.0, 1.8]], dtype=np.float64)
    foot_2d, _ = cv2.projectPoints(foot_3d, rvec, tvec, K, None)
    head_2d, _ = cv2.projectPoints(head_3d, rvec, tvec, K, None)

    bboxes = [[
        float(head_2d[0, 0, 0]) - 20, float(head_2d[0, 0, 1]),
        float(foot_2d[0, 0, 0]) + 20, float(foot_2d[0, 0, 1]),
    ]]

    score = score_player_heights(bboxes, K, rvec, tvec, height_range=(1.5, 2.1))
    assert score > 0.5


def test_score_player_heights_empty_bboxes_returns_zero():
    K = np.array([[2000, 0, 960], [0, 2000, 540], [0, 0, 1]], dtype=np.float64)
    rvec = np.array([1.3, -0.1, 0.0], dtype=np.float64)
    tvec = np.array([-30.0, -34.0, 40.0], dtype=np.float64)
    score = score_player_heights([], K, rvec, tvec)
    assert score == 0.0

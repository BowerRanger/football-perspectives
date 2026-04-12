"""Tests for the calibration stage and supporting utilities.

The primary calibration solver is now PnLCalib (see
``src/utils/neural_calibrator.py``).  These tests cover:

- Pitch geometry constants and the FIFA landmark dictionary (still used by
  ``src/utils/pitch.py`` as reference data and by downstream tools).
- Camera utilities in ``src/utils/camera.py``.
- The coordinate conversion from PnLCalib's pitch-centred frame to our
  near-left-corner frame.
- The static-camera fuser in ``src/stages/calibration.py`` (median world
  position across keyframes + per-frame re-anchoring).
- A fully-stubbed integration test of ``CameraCalibrationStage`` that does
  NOT call PnLCalib — a fake calibrator injects synthetic per-frame results.
- The ``player_height`` QA utility (kept as a debug metric, not used in the
  primary pipeline path).

A separate smoke test (``test_pnlcalib_smoke``) loads the real PnLCalib
models when their weights are present; otherwise it is skipped.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.calibration import CalibrationResult
from src.schemas.shots import ShotsManifest, Shot
from src.stages.calibration import (
    _DEFAULT_BOUNDS,
    CameraCalibrationStage,
    _compute_keyframes,
    _is_plausible,
    _neural_to_cam_frame,
    _robust_median_position,
)
from src.utils.camera import (
    build_projection_matrix,
    camera_world_position,
    is_camera_valid,
    project_to_pitch,
    reprojection_error,
)
from src.utils.neural_calibrator import (
    NeuralCalibration,
    convert_pnlcalib_to_ours,
)
from src.utils.pitch import FIFA_LANDMARKS, PITCH_LENGTH, PITCH_WIDTH


# ─────────────────────────────────────────────────────────── pitch geometry


def test_pitch_constants():
    assert PITCH_LENGTH == 105.0
    assert PITCH_WIDTH == 68.0
    assert "corner_near_left" in FIFA_LANDMARKS
    assert "center_spot" in FIFA_LANDMARKS
    pt = FIFA_LANDMARKS["corner_near_left"]
    assert pt[2] == 0.0  # z=0, pitch is ground plane


def test_goal_crossbar_landmarks_are_off_plane():
    for name in [
        "left_goal_near_post_top",
        "left_goal_far_post_top",
        "right_goal_near_post_top",
        "right_goal_far_post_top",
    ]:
        assert name in FIFA_LANDMARKS
        assert FIFA_LANDMARKS[name][2] == 2.44


def test_corner_flag_landmarks_are_off_plane():
    for name in [
        "corner_near_left_flag_top",
        "corner_near_right_flag_top",
        "corner_far_left_flag_top",
        "corner_far_right_flag_top",
    ]:
        assert name in FIFA_LANDMARKS
        assert FIFA_LANDMARKS[name][2] == 1.5


def test_near_landmarks_have_lower_y_than_far():
    assert FIFA_LANDMARKS["corner_near_left"][1] < FIFA_LANDMARKS["corner_far_left"][1]
    assert FIFA_LANDMARKS["halfway_near"][1] < FIFA_LANDMARKS["halfway_far"][1]
    assert FIFA_LANDMARKS["centre_circle_near"][1] < FIFA_LANDMARKS["centre_circle_far"][1]
    assert (
        FIFA_LANDMARKS["left_goal_near_post_base"][1]
        < FIFA_LANDMARKS["left_goal_far_post_base"][1]
    )


# ─────────────────────────────────────────────────────────── camera utilities


def _valid_broadcast_camera():
    """Return ``(rvec, tvec)`` for a physically plausible broadcast camera.

    Camera is placed at world position ``(52.5, -10, 30)`` — on the pitch
    centreline, 10 m behind the near touchline and 30 m above the pitch —
    and aimed at the pitch centre ``(52.5, 34, 0)``.  Uses z-up convention.
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


def _synthetic_K(fx: float = 2000.0, w: int = 1920, h: int = 1080) -> np.ndarray:
    return np.array(
        [[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1.0]], dtype=np.float64,
    )


def test_build_projection_matrix_shape():
    K = _synthetic_K()
    rvec, tvec = _valid_broadcast_camera()
    P = build_projection_matrix(K, rvec, tvec)
    assert P.shape == (3, 4)


def test_reprojection_error_zero_for_perfect_fit():
    K = _synthetic_K()
    rvec, tvec = _valid_broadcast_camera()
    pts_3d = np.array([[0, 0, 0], [105, 0, 0], [52.5, 34, 0]], dtype=np.float32)
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, None)
    pts_2d = pts_2d.reshape(-1, 2)
    err = reprojection_error(pts_3d, pts_2d, K.astype(np.float32), rvec, tvec)
    assert err < 0.01


def test_project_to_pitch_round_trips():
    K = _synthetic_K()
    rvec, tvec = _valid_broadcast_camera()
    pt_3d = np.array([30.0, 20.0, 0.0], dtype=np.float32)
    pt_2d, _ = cv2.projectPoints(pt_3d.reshape(1, 1, 3), rvec, tvec, K, None)
    pt_2d = pt_2d.reshape(2)
    recovered = project_to_pitch(pt_2d, K, rvec, tvec)
    assert np.allclose(recovered, pt_3d[:2], atol=0.05)


def test_camera_world_position_synthetic():
    rvec, tvec = _valid_broadcast_camera()
    pos = camera_world_position(rvec, tvec)
    assert pos[2] > 0  # camera above pitch
    assert pos.shape == (3,)
    assert np.allclose(pos, [52.5, -10.0, 30.0], atol=1e-6)


def test_is_camera_valid_accepts_normal_broadcast_camera():
    rvec, tvec = _valid_broadcast_camera()
    assert is_camera_valid(rvec, tvec, min_height=3.0, max_height=80.0)


def test_is_camera_valid_rejects_camera_below_pitch():
    rvec_id = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    tvec_below = np.array([0.0, 0.0, 10.0], dtype=np.float32)
    assert not is_camera_valid(rvec_id, tvec_below, min_height=3.0, max_height=80.0)


def test_is_camera_valid_rejects_camera_looking_up():
    rvec_id = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    tvec_up = np.array([0.0, 0.0, -30.0], dtype=np.float32)
    assert not is_camera_valid(rvec_id, tvec_up, min_height=3.0, max_height=80.0)


# ─────────────────────────────────────── coordinate conversion (PnLCalib → ours)


def test_convert_pnlcalib_position_identity_rotation():
    """With identity rotation, position converts to ``(x+52.5, 34-y, -z)``."""
    R = np.eye(3)
    # PnLCalib position (pitch-centred, z-down).
    C_pnl = np.array([-2.07, 67.02, -15.04])
    rvec_ours, tvec_ours, pos_ours = convert_pnlcalib_to_ours(R, C_pnl)
    expected_pos = np.array([50.43, -33.02, 15.04])
    assert np.allclose(pos_ours, expected_pos, atol=1e-6)


def test_convert_pnlcalib_position_matches_direct_formula():
    """The derived ``camera_world_position`` matches ``(x+52.5, 34-y, -z)``."""
    rng = np.random.default_rng(42)
    # Random PnLCalib rotation + position.
    rvec_pnl = rng.normal(scale=0.5, size=3)
    R_pnl, _ = cv2.Rodrigues(rvec_pnl)
    C_pnl = np.array([-10.0, 50.0, -20.0])

    _, tvec_ours, pos_ours = convert_pnlcalib_to_ours(R_pnl, C_pnl)

    # After conversion, computing the camera's world position from the new
    # (R, t) must equal the direct formula (x+52.5, 34-y, -z).
    rvec_ours, _ = cv2.Rodrigues(R_pnl @ np.diag([1.0, -1.0, -1.0]))
    computed = camera_world_position(rvec_ours, tvec_ours)
    expected = np.array([C_pnl[0] + 52.5, 34.0 - C_pnl[1], -C_pnl[2]])
    assert np.allclose(computed, expected, atol=1e-6)
    assert np.allclose(pos_ours, expected, atol=1e-6)


def test_convert_pnlcalib_preserves_projection():
    """A 3D pitch point projects to the same pixel under both frames."""
    # Arbitrary PnLCalib camera (in its own centred/z-down frame).
    rvec_pnl = np.array([0.4, 0.2, -0.1])
    R_pnl, _ = cv2.Rodrigues(rvec_pnl)
    C_pnl = np.array([-5.0, 40.0, -20.0])
    t_pnl = -R_pnl @ C_pnl
    K = _synthetic_K(fx=3000.0)

    # Pick a pitch corner in PnLCalib coords: pitch centre is (0, 0, 0),
    # so the near-left corner at our (0, 0, 0) is at PnLCalib (-52.5, 34, 0).
    X_ours = np.array([0.0, 0.0, 0.0])
    X_pnl = np.array([-52.5, 34.0, 0.0])

    # Project in PnLCalib frame.
    proj_pnl, _ = cv2.projectPoints(
        X_pnl.reshape(1, 1, 3).astype(np.float64),
        rvec_pnl, t_pnl, K, None,
    )
    pixel_pnl = proj_pnl.reshape(2)

    # Project in our frame using converted camera pose.
    rvec_ours, tvec_ours, _ = convert_pnlcalib_to_ours(R_pnl, C_pnl)
    proj_ours, _ = cv2.projectPoints(
        X_ours.reshape(1, 1, 3).astype(np.float64),
        rvec_ours, tvec_ours, K, None,
    )
    pixel_ours = proj_ours.reshape(2)

    assert np.allclose(pixel_pnl, pixel_ours, atol=1e-6)


# ────────────────────────────────────────────────── static-camera fuser helpers


def test_robust_median_position_filters_single_outlier():
    positions = [
        np.array([50.0, -30.0, 15.0]),
        np.array([50.1, -30.2, 15.1]),
        np.array([49.9, -29.8, 14.9]),
        np.array([50.0, -30.1, 15.0]),
        np.array([100.0, 50.0, 2.0]),  # clear outlier
    ]
    med = _robust_median_position(positions)
    # Outlier should not drag the median far.
    assert abs(med[0] - 50.0) < 1.0
    assert abs(med[1] - -30.0) < 1.0
    assert abs(med[2] - 15.0) < 1.0


def test_robust_median_position_returns_none_for_empty_list():
    assert _robust_median_position([]) is None


def test_robust_median_position_handles_two_samples():
    positions = [np.array([10.0, 20.0, 30.0]), np.array([14.0, 24.0, 34.0])]
    med = _robust_median_position(positions)
    assert np.allclose(med, [12.0, 22.0, 32.0])


def test_compute_keyframes_respects_max_cap():
    # 500 total frames, interval 5 → naive gives 100; max=10 limits to 10.
    kf = _compute_keyframes(total_frames=500, keyframe_interval=5, max_keyframes=10)
    assert len(kf) <= 10
    assert kf[0] == 0
    assert all(isinstance(i, int) for i in kf)


def test_compute_keyframes_uses_interval_for_short_shots():
    # 60 total frames, interval 30, max=10 → 2 samples
    kf = _compute_keyframes(total_frames=60, keyframe_interval=30, max_keyframes=10)
    assert kf == [0, 30]


def test_compute_keyframes_empty_for_zero_frames():
    assert _compute_keyframes(total_frames=0, keyframe_interval=5, max_keyframes=10) == []


def _bounds_dict() -> dict[str, tuple[float, float]]:
    return {k: tuple(v) for k, v in _DEFAULT_BOUNDS.items()}


def test_is_plausible_accepts_sane_broadcast_camera():
    bounds = _bounds_dict()
    rvec, tvec = _valid_broadcast_camera()
    cal = NeuralCalibration(
        K=_synthetic_K(fx=3000.0),
        rvec=rvec,
        tvec=tvec,
        world_position=np.array([52.5, -10.0, 30.0]),
    )
    assert _is_plausible(cal, bounds)


def test_is_plausible_rejects_out_of_bounds_position():
    bounds = _bounds_dict()
    rvec, tvec = _valid_broadcast_camera()
    cal = NeuralCalibration(
        K=_synthetic_K(fx=3000.0),
        rvec=rvec,
        tvec=tvec,
        world_position=np.array([200.0, -10.0, 30.0]),  # x >> 135
    )
    assert not _is_plausible(cal, bounds)


def test_is_plausible_rejects_absurd_focal_length():
    bounds = _bounds_dict()
    rvec, tvec = _valid_broadcast_camera()
    cal = NeuralCalibration(
        K=_synthetic_K(fx=5.0),  # absurdly low
        rvec=rvec,
        tvec=tvec,
        world_position=np.array([52.5, -10.0, 30.0]),
    )
    assert not _is_plausible(cal, bounds)


def test_is_plausible_rejects_negative_elevation():
    bounds = _bounds_dict()
    rvec, tvec = _valid_broadcast_camera()
    cal = NeuralCalibration(
        K=_synthetic_K(fx=3000.0),
        rvec=rvec,
        tvec=tvec,
        world_position=np.array([52.5, -10.0, -5.0]),  # underground
    )
    assert not _is_plausible(cal, bounds)


def test_neural_to_cam_frame_override_position_moves_camera():
    K = _synthetic_K(fx=2000.0)
    rvec = np.array([1.3, -0.1, 0.0])
    tvec = np.array([-30.0, -34.0, 40.0])
    # Pretend PnLCalib's own position was somewhere; we override to a known value.
    cal = NeuralCalibration(K=K, rvec=rvec, tvec=tvec, world_position=np.array([0.0, 0.0, 0.0]))
    override = np.array([50.0, -30.0, 15.0])
    cam = _neural_to_cam_frame(cal, frame_idx=42, override_position=override)
    pos = camera_world_position(
        np.asarray(cam.rotation_vector), np.asarray(cam.translation_vector),
    )
    assert np.allclose(pos, override, atol=1e-6)
    assert cam.frame == 42


# ────────────────────────────────────────── integration: stage with fake solver


class _FakeNeuralCalibrator:
    """Drop-in replacement for ``PnLCalibrator`` used by integration tests.

    ``results_per_frame`` maps ``frame_idx → (NeuralCalibration | None)``.
    Unlisted frames return None.
    """

    def __init__(self, results_per_frame: dict[int, NeuralCalibration | None]):
        self._results = results_per_frame
        self.calls: list[int] = []

    def calibrate(self, frame_bgr: np.ndarray) -> NeuralCalibration | None:
        # We can't inspect frame index directly; infer from call order.
        call_idx = len(self.calls)
        self.calls.append(call_idx)
        return list(self._results.values())[call_idx % len(self._results)]


def _write_dummy_clip(path: Path, num_frames: int, fps: float = 25.0, size: tuple[int, int] = (320, 240)):
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(num_frames):
        frame = np.full((h, w, 3), i % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _fake_calibration(world_position: np.ndarray) -> NeuralCalibration:
    """Build a NeuralCalibration with a given world position.

    Camera looks down at the pitch centre from ``world_position``.
    """
    C = world_position.astype(np.float64)
    target = np.array([52.5, 34.0, 0.0])
    cam_z_world = target - C
    cam_z_world /= np.linalg.norm(cam_z_world)
    world_up = np.array([0.0, 0.0, 1.0])
    cam_x_world = np.cross(cam_z_world, world_up)
    cam_x_world /= np.linalg.norm(cam_x_world)
    cam_y_world = np.cross(cam_z_world, cam_x_world)
    R = np.array([cam_x_world, cam_y_world, cam_z_world])
    rvec, _ = cv2.Rodrigues(R)
    tvec = -R @ C
    K = _synthetic_K(fx=3000.0)
    return NeuralCalibration(K=K, rvec=rvec.flatten(), tvec=tvec, world_position=C)


def test_stage_fuses_per_frame_results_to_shared_position(tmp_path):
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    clip_path = shots_dir / "shot_001.mp4"
    _write_dummy_clip(clip_path, num_frames=90)
    manifest = ShotsManifest(
        source_file="test.mp4",
        fps=25.0,
        total_frames=90,
        shots=[
            Shot(
                id="shot_001",
                start_frame=0,
                end_frame=90,
                start_time=0.0,
                end_time=3.6,
                clip_file="shots/shot_001.mp4",
            ),
        ],
    )
    manifest.save(shots_dir / "shots_manifest.json")

    positions = [
        np.array([49.0, -30.0, 14.0]),
        np.array([51.0, -30.5, 15.5]),
        np.array([50.0, -29.5, 15.0]),
        # Outlier frame — will be rejected by the plausibility filter
        # because z=2 is below the min elevation of 3m.
        np.array([10.0, 50.0, 2.0]),
    ]
    fake_results = {i: _fake_calibration(p) for i, p in enumerate(positions)}
    fake = _FakeNeuralCalibrator(fake_results)

    stage = CameraCalibrationStage(
        config={"calibration": {"keyframe_interval": 20, "max_keyframes_per_shot": 4}},
        output_dir=tmp_path,
        neural_calibrator=fake,
    )
    stage.run()

    result = CalibrationResult.load(tmp_path / "calibration" / "shot_001_calibration.json")
    assert result.camera_type == "static"
    # 4 sampled, 1 rejected by plausibility (z=2), 3 kept.
    assert len(result.frames) == 3

    # Every output frame must share the same world position (within numerical noise).
    positions_out = [
        camera_world_position(
            np.asarray(f.rotation_vector), np.asarray(f.translation_vector),
        )
        for f in result.frames
    ]
    for pos in positions_out[1:]:
        assert np.allclose(pos, positions_out[0], atol=1e-6)

    # The shared position should be near the median of the good samples, not the outlier.
    shared = positions_out[0]
    assert 48.0 < shared[0] < 52.0
    assert -31.5 < shared[1] < -29.0
    assert 13.0 < shared[2] < 16.5


def test_stage_writes_empty_calibration_when_all_frames_fail(tmp_path, caplog):
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    _write_dummy_clip(shots_dir / "shot_001.mp4", num_frames=60)
    manifest = ShotsManifest(
        source_file="test.mp4",
        fps=25.0,
        total_frames=60,
        shots=[
            Shot(
                id="shot_001",
                start_frame=0,
                end_frame=60,
                start_time=0.0,
                end_time=2.4,
                clip_file="shots/shot_001.mp4",
            ),
        ],
    )
    manifest.save(shots_dir / "shots_manifest.json")

    # All None → simulates PnLCalib failing on every keyframe.
    fake = _FakeNeuralCalibrator({0: None, 1: None, 2: None})

    stage = CameraCalibrationStage(
        config={"calibration": {"keyframe_interval": 20, "max_keyframes_per_shot": 3}},
        output_dir=tmp_path,
        neural_calibrator=fake,
    )
    with caplog.at_level("WARNING"):
        stage.run()

    result = CalibrationResult.load(tmp_path / "calibration" / "shot_001_calibration.json")
    assert result.frames == []
    assert result.camera_type == "static"
    assert any("no plausible calibrations" in r.message.lower() for r in caplog.records)


# ───────────────────────────────────────── player-height QA utility (disconnected)


def test_score_player_heights_correct_solution_scores_higher():
    """A camera solution that produces ~1.8m player heights should score well."""
    from src.utils.player_height import score_player_heights

    K = _synthetic_K(fx=2000.0)
    rvec = np.array([1.3, -0.1, 0.0], dtype=np.float64)
    tvec = np.array([-30.0, -34.0, 40.0], dtype=np.float64)

    # Synthetic player bbox by projecting a standing person at (30, 20).
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
    from src.utils.player_height import score_player_heights

    K = _synthetic_K(fx=2000.0)
    rvec = np.array([1.3, -0.1, 0.0], dtype=np.float64)
    tvec = np.array([-30.0, -34.0, 40.0], dtype=np.float64)
    assert score_player_heights([], K, rvec, tvec) == 0.0


# ───────────────────────────────────────────────── optional PnLCalib smoke test


_WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "data" / "models" / "pnlcalib"


@pytest.mark.skipif(
    not (_WEIGHTS_DIR / "SV_kp").exists() or not (_WEIGHTS_DIR / "SV_lines").exists(),
    reason="PnLCalib weights not present in data/models/pnlcalib/",
)
def test_pnlcalib_smoke_if_weights_present():
    """End-to-end PnLCalib smoke test — only runs when weights are cached.

    This test is skipped in CI unless the weights are downloaded.  It just
    verifies that the wrapper loads the model and produces SOME output
    (or None) for a blank frame.  Heavy; takes ~20s on CPU.
    """
    from src.utils.neural_calibrator import PnLCalibrator

    calibrator = PnLCalibrator(device="cpu")
    blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = calibrator.calibrate(blank)
    # A blank frame should not produce a sensible calibration — PnLCalib
    # should return None.  We just want to prove the wrapper doesn't crash.
    assert result is None or isinstance(result, NeuralCalibration)

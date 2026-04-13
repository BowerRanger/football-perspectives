import json
import numpy as np
import cv2
import pytest
from pathlib import Path

from src.schemas.calibration import CalibrationResult, CameraFrame
from src.schemas.player_matches import PlayerMatches, MatchedPlayer, PlayerView
from src.schemas.poses import PosesResult, PlayerPoses, PlayerPoseFrame, Keypoint, COCO_KEYPOINT_NAMES
from src.schemas.shots import ShotsManifest, Shot
from src.schemas.sync_map import SyncMap, Alignment
from src.schemas.triangulated import TriangulatedPlayer
from src.stages.triangulation import TriangulationStage
from src.pipeline.config import load_config


def _synthetic_camera(offset_y: float = -34.0):
    """Return a camera with known parameters."""
    K = [[1500, 0, 960], [0, 1500, 540], [0, 0, 1]]
    rvec = [0.05, 0.15, 0.0]
    tvec = [-52.5, offset_y, 60.0]
    return K, rvec, tvec


def _project_3d_to_2d(pt_3d, K, rvec, tvec):
    """Project a 3D point to 2D pixel coordinates."""
    K_np = np.array(K, dtype=np.float64)
    rvec_np = np.array(rvec, dtype=np.float64)
    tvec_np = np.array(tvec, dtype=np.float64)
    pts, _ = cv2.projectPoints(
        np.array([pt_3d], dtype=np.float64), rvec_np, tvec_np, K_np, None
    )
    return pts.reshape(2)


@pytest.fixture
def tri_workspace(tmp_path) -> Path:
    """Create a minimal workspace with synthetic data for two cameras."""
    root = tmp_path

    # Shots manifest
    shots_dir = root / "shots"
    shots_dir.mkdir()
    manifest = ShotsManifest(
        source_file="test.mp4",
        fps=25.0,
        total_frames=100,
        shots=[
            Shot(id="cam_a", start_frame=0, end_frame=49, start_time=0.0, end_time=2.0, clip_file="shots/cam_a.mp4"),
            Shot(id="cam_b", start_frame=0, end_frame=49, start_time=0.0, end_time=2.0, clip_file="shots/cam_b.mp4"),
        ],
    )
    manifest.save(shots_dir / "shots_manifest.json")

    # Sync map: cam_a is reference, cam_b has offset 0 (perfectly synced)
    sync_dir = root / "sync"
    sync_dir.mkdir()
    SyncMap(
        reference_shot="cam_a",
        alignments=[Alignment(shot_id="cam_b", frame_offset=0, confidence=1.0, method="manual", overlap_frames=[0, 49])],
    ).save(sync_dir / "sync_map.json")

    # Camera calibration: two cameras at different angles
    K_a, rvec_a, tvec_a = _synthetic_camera(offset_y=-34.0)
    K_b, rvec_b, tvec_b = _synthetic_camera(offset_y=-100.0)

    cal_dir = root / "calibration"
    cal_dir.mkdir()
    CalibrationResult(
        shot_id="cam_a",
        camera_type="static",
        frames=[CameraFrame(frame=0, intrinsic_matrix=K_a, rotation_vector=rvec_a, translation_vector=tvec_a,
                            reprojection_error=0.5, num_correspondences=8, confidence=0.9)],
    ).save(cal_dir / "cam_a_calibration.json")
    CalibrationResult(
        shot_id="cam_b",
        camera_type="static",
        frames=[CameraFrame(frame=0, intrinsic_matrix=K_b, rotation_vector=rvec_b, translation_vector=tvec_b,
                            reprojection_error=0.5, num_correspondences=8, confidence=0.9)],
    ).save(cal_dir / "cam_b_calibration.json")

    # Player matches: one player visible in both cameras
    match_dir = root / "matching"
    match_dir.mkdir()
    PlayerMatches(matched_players=[
        MatchedPlayer(player_id="P001", team="A", views=[
            PlayerView(shot_id="cam_a", track_id="T001"),
            PlayerView(shot_id="cam_b", track_id="T003"),
        ]),
    ]).save(match_dir / "player_matches.json")

    # Poses: generate 2D keypoints by projecting known 3D positions
    # Player walks from (30, 20, 0) to (35, 20, 0) over 10 frames
    poses_dir = root / "poses"
    poses_dir.mkdir()

    for shot_id, K, rvec, tvec in [("cam_a", K_a, rvec_a, tvec_a), ("cam_b", K_b, rvec_b, tvec_b)]:
        track_id = "T001" if shot_id == "cam_a" else "T003"
        frames = []
        for f in range(10):
            x = 30.0 + f * 0.5
            base_3d = np.array([x, 20.0, 0.0])
            keypoints = []
            for ji, name in enumerate(COCO_KEYPOINT_NAMES):
                # Offset each joint slightly from base
                offsets = {
                    "nose": [0, 0, 1.7], "left_shoulder": [-0.2, 0, 1.4],
                    "right_shoulder": [0.2, 0, 1.4], "left_hip": [-0.15, 0, 0.9],
                    "right_hip": [0.15, 0, 0.9], "left_ankle": [-0.15, 0, 0.0],
                    "right_ankle": [0.15, 0, 0.0],
                }
                offset = offsets.get(name, [0, 0, 1.0])
                pt_3d = base_3d + np.array(offset)
                px = _project_3d_to_2d(pt_3d, K, rvec, tvec)
                keypoints.append(Keypoint(name=name, x=float(px[0]), y=float(px[1]), conf=0.9))
            frames.append(PlayerPoseFrame(frame=f, keypoints=keypoints))

        PosesResult(
            shot_id=shot_id,
            players=[PlayerPoses(track_id=track_id, frames=frames)],
        ).save(poses_dir / f"{shot_id}_poses.json")

    return root


def test_triangulation_stage_produces_output(tri_workspace):
    cfg = load_config()
    stage = TriangulationStage(config=cfg, output_dir=tri_workspace)

    assert not stage.is_complete()
    stage.run()
    assert stage.is_complete()

    npz_files = list((tri_workspace / "triangulated").glob("*.npz"))
    assert len(npz_files) == 1
    assert npz_files[0].name == "P001_3d_joints.npz"


def test_triangulation_output_has_correct_shape(tri_workspace):
    cfg = load_config()
    TriangulationStage(config=cfg, output_dir=tri_workspace).run()

    result = TriangulatedPlayer.load(tri_workspace / "triangulated" / "P001_3d_joints.npz")
    assert result.player_id == "P001"
    assert result.positions.shape[1] == 17
    assert result.positions.shape[2] == 3
    assert result.confidences.shape[1] == 17
    assert result.fps == 25.0


def test_triangulated_positions_are_near_ground_truth(tri_workspace):
    cfg = load_config()
    TriangulationStage(config=cfg, output_dir=tri_workspace).run()

    result = TriangulatedPlayer.load(tri_workspace / "triangulated" / "P001_3d_joints.npz")

    # The player's hip midpoint should be near (30+f*0.5, 20, 0.9) for frame f
    # Check a few frames (using the hip joints: indices 11, 12)
    for f in range(min(5, result.positions.shape[0])):
        hip_mid = (result.positions[f, 11] + result.positions[f, 12]) / 2
        if np.any(np.isnan(hip_mid)):
            continue
        expected_x = 30.0 + f * 0.5
        assert abs(hip_mid[0] - expected_x) < 2.0, f"Frame {f}: expected x≈{expected_x}, got {hip_mid[0]}"
        assert abs(hip_mid[1] - 20.0) < 2.0, f"Frame {f}: expected y≈20, got {hip_mid[1]}"


def test_triangulated_schema_round_trip(tmp_path):
    positions = np.random.randn(10, 17, 3).astype(np.float32)
    confidences = np.random.rand(10, 17).astype(np.float32)
    reproj = np.random.rand(10, 17).astype(np.float32)
    n_views = np.ones((10, 17), dtype=np.int8) * 2

    original = TriangulatedPlayer(
        player_id="P042",
        player_name="Salah",
        team="A",
        positions=positions,
        confidences=confidences,
        reprojection_errors=reproj,
        num_views=n_views,
        fps=30.0,
        start_frame=100,
    )
    path = tmp_path / "test.npz"
    original.save(path)

    loaded = TriangulatedPlayer.load(path)
    assert loaded.player_id == "P042"
    assert loaded.player_name == "Salah"
    assert loaded.team == "A"
    assert loaded.fps == pytest.approx(30.0)
    assert loaded.start_frame == 100
    np.testing.assert_allclose(loaded.positions, positions, atol=1e-5)
    np.testing.assert_allclose(loaded.confidences, confidences, atol=1e-5)


def test_triangulated_schema_defaults_for_old_npz(tmp_path):
    """Loading a TriangulatedPlayer whose NPZ lacks player_name/team should
    default them to empty strings, so older outputs on disk still load."""
    positions = np.random.randn(3, 17, 3).astype(np.float32)
    confidences = np.ones((3, 17), dtype=np.float32)
    reproj = np.zeros((3, 17), dtype=np.float32)
    n_views = np.full((3, 17), 2, dtype=np.int8)
    # Save without the new fields (simulating an older NPZ).
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        player_id=np.array("P001"),
        positions=positions,
        confidences=confidences,
        reprojection_errors=reproj,
        num_views=n_views,
        fps=np.array(25.0, dtype=np.float32),
        start_frame=np.array(0, dtype=np.int32),
    )
    loaded = TriangulatedPlayer.load(path)
    assert loaded.player_name == ""
    assert loaded.team == ""


# ── Tests for strict multi-view + name propagation ──────────────────────────


from src.schemas.tracks import Track, TrackFrame, TracksResult


def _add_tracks(root: Path, shot_id: str, track_id: str, player_name: str = "", team: str = "A", player_id: str = "") -> None:
    tracks_dir = root / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    tf = TrackFrame(frame=0, bbox=[0.0, 0.0, 100.0, 200.0], confidence=0.9, pitch_position=None)
    tr = TracksResult(
        shot_id=shot_id,
        tracks=[
            Track(
                track_id=track_id,
                class_name="player",
                team=team,
                player_id=player_id,
                player_name=player_name,
                frames=[tf],
            )
        ],
    )
    tr.save(tracks_dir / f"{shot_id}_tracks.json")


def test_triangulation_carries_player_name_from_tracks(tri_workspace):
    _add_tracks(tri_workspace, "cam_a", "T001", player_name="Salah", team="A")
    _add_tracks(tri_workspace, "cam_b", "T003", player_name="", team="")

    cfg = load_config()
    TriangulationStage(config=cfg, output_dir=tri_workspace).run()

    result = TriangulatedPlayer.load(
        tri_workspace / "triangulated" / "P001_3d_joints.npz"
    )
    assert result.player_name == "Salah"
    assert result.team == "A"


def test_triangulation_falls_back_to_second_view_for_name(tri_workspace):
    # First view has no name, second does.
    _add_tracks(tri_workspace, "cam_a", "T001", player_name="", team="")
    _add_tracks(tri_workspace, "cam_b", "T003", player_name="Firmino", team="A")

    cfg = load_config()
    TriangulationStage(config=cfg, output_dir=tri_workspace).run()

    result = TriangulatedPlayer.load(
        tri_workspace / "triangulated" / "P001_3d_joints.npz"
    )
    assert result.player_name == "Firmino"


def test_triangulation_skips_player_with_only_one_calibrated_view(tri_workspace):
    # Delete cam_b's calibration → player_id P001 has only 1 calibrated view.
    (tri_workspace / "calibration" / "cam_b_calibration.json").unlink()

    cfg = load_config()
    TriangulationStage(config=cfg, output_dir=tri_workspace).run()

    assert not (tri_workspace / "triangulated" / "P001_3d_joints.npz").exists()


def test_triangulation_skips_player_with_empty_calibrations(tri_workspace):
    """A shot whose calibration file exists but has an empty frames list
    (e.g., PnLCalib failed on every keyframe) should be treated as
    uncalibrated."""
    from src.schemas.calibration import CalibrationResult

    CalibrationResult(
        shot_id="cam_b", camera_type="static", frames=[],
    ).save(tri_workspace / "calibration" / "cam_b_calibration.json")

    cfg = load_config()
    TriangulationStage(config=cfg, output_dir=tri_workspace).run()

    assert not (tri_workspace / "triangulated" / "P001_3d_joints.npz").exists()


# ── CalibrationInterpolator tests ───────────────────────────────────────────


def test_interpolator_single_keyframe_returns_same_values():
    from src.schemas.calibration import CalibrationResult, CameraFrame
    from src.utils.triangulation_calib import CalibrationInterpolator

    cal = CalibrationResult(
        shot_id="t",
        camera_type="static",
        frames=[
            CameraFrame(
                frame=0,
                intrinsic_matrix=[[3000.0, 0.0, 960.0], [0.0, 3000.0, 540.0], [0.0, 0.0, 1.0]],
                rotation_vector=[1.3, -0.1, 0.0],
                translation_vector=[-30.0, -34.0, 40.0],
                reprojection_error=0.0,
                num_correspondences=0,
                confidence=1.0,
            )
        ],
    )
    interp = CalibrationInterpolator(cal)
    r = interp.at(100)
    assert r is not None
    assert r.K[0, 0] == 3000.0
    np.testing.assert_allclose(r.rvec, [1.3, -0.1, 0.0])
    np.testing.assert_allclose(r.tvec, [-30.0, -34.0, 40.0])


def test_interpolator_two_keyframes_preserves_world_position():
    from src.schemas.calibration import CalibrationResult, CameraFrame
    from src.utils.triangulation_calib import CalibrationInterpolator
    from src.utils.camera import camera_world_position

    C_shared = np.array([50.0, -30.0, 15.0])
    rvec_0 = np.array([1.3, 0.0, 0.0])
    rvec_1 = np.array([1.3, 0.2, 0.0])

    R0, _ = cv2.Rodrigues(rvec_0)
    R1, _ = cv2.Rodrigues(rvec_1)

    cf0 = CameraFrame(
        frame=0,
        intrinsic_matrix=[[2000.0, 0.0, 960.0], [0.0, 2000.0, 540.0], [0.0, 0.0, 1.0]],
        rotation_vector=rvec_0.tolist(),
        translation_vector=(-R0 @ C_shared).tolist(),
        reprojection_error=0.0,
        num_correspondences=0,
        confidence=1.0,
    )
    cf1 = CameraFrame(
        frame=100,
        intrinsic_matrix=[[4000.0, 0.0, 960.0], [0.0, 4000.0, 540.0], [0.0, 0.0, 1.0]],
        rotation_vector=rvec_1.tolist(),
        translation_vector=(-R1 @ C_shared).tolist(),
        reprojection_error=0.0,
        num_correspondences=0,
        confidence=1.0,
    )
    cal = CalibrationResult(shot_id="t", camera_type="static", frames=[cf0, cf1])
    interp = CalibrationInterpolator(cal)

    for target in [0, 25, 50, 75, 100]:
        r = interp.at(target)
        assert r is not None
        pos = camera_world_position(r.rvec, r.tvec)
        np.testing.assert_allclose(pos, C_shared, atol=1e-6)
        # Focal length is linearly interpolated
        expected_fx = 2000.0 + (4000.0 - 2000.0) * (target / 100.0)
        assert r.K[0, 0] == pytest.approx(expected_fx, rel=1e-6)


def test_interpolator_clamps_outside_range():
    from src.schemas.calibration import CalibrationResult, CameraFrame
    from src.utils.triangulation_calib import CalibrationInterpolator

    cf0 = CameraFrame(
        frame=10,
        intrinsic_matrix=[[2000.0, 0.0, 960.0], [0.0, 2000.0, 540.0], [0.0, 0.0, 1.0]],
        rotation_vector=[1.3, 0.0, 0.0],
        translation_vector=[-30.0, -34.0, 40.0],
        reprojection_error=0.0,
        num_correspondences=0,
        confidence=1.0,
    )
    cf1 = CameraFrame(
        frame=50,
        intrinsic_matrix=[[4000.0, 0.0, 960.0], [0.0, 4000.0, 540.0], [0.0, 0.0, 1.0]],
        rotation_vector=[1.3, 0.1, 0.0],
        translation_vector=[-30.0, -34.0, 40.0],
        reprojection_error=0.0,
        num_correspondences=0,
        confidence=1.0,
    )
    cal = CalibrationResult(shot_id="t", camera_type="static", frames=[cf0, cf1])
    interp = CalibrationInterpolator(cal)

    # Frame 0 is before the first keyframe — should clamp to keyframe 10.
    r0 = interp.at(0)
    assert r0 is not None
    assert r0.K[0, 0] == pytest.approx(2000.0)

    # Frame 100 is after the last keyframe — should clamp to keyframe 50.
    r100 = interp.at(100)
    assert r100 is not None
    assert r100.K[0, 0] == pytest.approx(4000.0)

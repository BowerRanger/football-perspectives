"""Integration tests for Layer 5 ball anchor injection."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.ball_anchor import BallAnchor, BallAnchorSet
from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.shots import Shot, ShotsManifest
from src.stages.ball import BallStage
from src.utils.ball_detector import FakeBallDetector


def _camera_pose():
    look = np.array([0.0, 64.0, -30.0]); look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _write_blank_clip(path: Path, n: int, fps: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240)
    )
    for _ in range(n):
        writer.write(np.full((240, 320, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()


def _save_camera_track(path: Path, K, R, t, n: int, fps: float = 30.0):
    track = CameraTrack(
        clip_id="play", fps=fps, image_size=(1280, 720), t_world=t.tolist(),
        frames=tuple(
            CameraFrame(frame=i, K=K.tolist(), R=R.tolist(),
                        confidence=1.0, is_anchor=(i == 0))
            for i in range(n)
        ),
    )
    track.save(path)


def _save_manifest(path: Path, n: int):
    ShotsManifest(
        source_file="fake.mp4", fps=30.0, total_frames=n,
        shots=[Shot(id="play", clip_file="shots/play.mp4",
                    start_frame=0, end_frame=n - 1,
                    start_time=0.0, end_time=(n - 1) / 30.0)],
    ).save(path)


def _minimal_cfg() -> dict:
    return {
        "ball": {
            "detector": "fake",
            "ball_radius_m": 0.11,
            "max_gap_frames": 6,
            "flight_max_residual_px": 5.0,
            "tracker": {
                "process_noise_grounded_px": 4.0,
                "process_noise_flight_px": 12.0,
                "measurement_noise_px": 2.0,
                "gating_sigma": 4.0,
                "min_flight_frames": 6,
                "max_flight_frames": 90,
            },
            "spin": {"enabled": False, "min_flight_seconds": 0.5,
                     "min_residual_improvement": 0.2,
                     "max_omega_rad_s": 200.0, "drag_k_over_m": 0.005},
            "plausibility": {"z_max_m": 50.0, "horizontal_speed_max_m_s": 40.0, "pitch_margin_m": 5.0},
            "flight_promotion": {"enabled": False, "min_run_frames": 6,
                                 "off_pitch_margin_m": 5.0, "max_ground_speed_m_s": 35.0},
            "kick_anchor": {"enabled": False, "max_pixel_distance_px": 30.0, "lookahead_frames": 4,
                            "min_pixel_acceleration_px_per_frame": 6.0, "foot_anchor_z_m": 0.11},
            "appearance_bridge": {"enabled": False, "max_gap_frames": 8, "template_size_px": 32,
                                  "search_radius_px": 64, "min_ncc": 0.6,
                                  "template_max_age_frames": 30, "template_update_confidence": 0.5},
        },
        "pitch": {"length_m": 105.0, "width_m": 68.0},
    }


@pytest.mark.integration
def test_anchor_overrides_wasb_detection(tmp_path: Path):
    """When an anchor exists at a frame, its pixel position is used and
    WASB is ignored on that frame."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 20
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # WASB will return one (wrong) pixel for every frame.
    wasb_uv = (100.0, 100.0)
    wasb_detections = [(wasb_uv[0], wasb_uv[1], 0.85) for _ in range(n_frames)]

    # Anchor at frame 5 says the ball is at a different pixel position.
    anchor_uv = (640.0, 360.0)
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(BallAnchor(frame=5, image_xy=anchor_uv, state="grounded"),),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(wasb_detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    f5 = next(f for f in track.frames if f.frame == 5)
    # Recover the pixel by reprojection: ground-projection of the
    # anchor pixel at z=0.11 should equal f5.world_xyz exactly.
    from src.utils.foot_anchor import ankle_ray_to_pitch
    expected_world = ankle_ray_to_pitch(anchor_uv, K=K, R=R, t=t, plane_z=0.11)
    assert f5.state == "grounded"
    assert np.allclose(f5.world_xyz, expected_world, atol=1e-3)


@pytest.mark.integration
def test_no_anchor_file_runs_normally(tmp_path: Path):
    """When no anchor file exists, ball stage runs exactly as before."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 20
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    detections = [(640.0, 360.0, 0.85) for _ in range(n_frames)]
    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()
    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    # No anchors → WASB pixel used → at least one frame is grounded.
    assert any(f.state == "grounded" for f in track.frames)


@pytest.mark.integration
def test_off_screen_flight_anchor_skips_pixel(tmp_path: Path):
    """An off_screen_flight anchor produces no pixel detection but
    forces the frame into a flight run."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    n_frames = 30
    _write_blank_clip(out / "shots" / "play.mp4", n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    _save_manifest(out / "shots" / "shots_manifest.json", n_frames)

    # WASB returns None for every frame.
    detections: list[tuple[float, float, float] | None] = [None] * n_frames

    anchors = [
        BallAnchor(frame=fi, image_xy=None, state="off_screen_flight")
        for fi in range(10, 20)
    ]
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=tuple(anchors),
    ).save(out / "ball" / "play_ball_anchors.json")

    BallStage(
        config=_minimal_cfg(), output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()
    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    # Frames 10..19 should not be state="missing".
    forced = [f for f in track.frames if 10 <= f.frame <= 19]
    assert all(f.state != "missing" for f in forced), (
        f"off-screen-flight anchors must not emit missing: {[f.state for f in forced]}"
    )

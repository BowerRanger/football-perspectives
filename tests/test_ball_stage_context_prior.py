"""Context prior wiring: a static overlay blob far from players is dropped
by the prior; a genuine moving ball is untouched; disabled flag restores
old behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.shots import Shot, ShotsManifest
from src.stages.ball import BallStage
from src.utils.ball_detector import FakeBallDetector

N_FRAMES = 90
FPS = 30.0


def _camera_pose(yaw_deg: float = 0.0):
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    a = np.deg2rad(yaw_deg)
    yaw = np.array([[np.cos(a), -np.sin(a), 0.0],
                    [np.sin(a), np.cos(a), 0.0],
                    [0.0, 0.0, 1.0]])
    R = R @ yaw
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _build_scene(tmp_path: Path, *, panning: bool):
    """Scene with a PANNING camera (0.2 deg/frame yaw) so the static
    signal can fire, plus a tracks sidecar with one player box far from
    the frame top."""
    out = tmp_path / "out"
    clip = out / "shots" / "play.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(clip), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720))
    for _ in range(N_FRAMES):
        writer.write(np.full((720, 1280, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()

    frames = []
    for i in range(N_FRAMES):
        K, R, t = _camera_pose(yaw_deg=(i * 0.2 if panning else 0.0))
        frames.append(CameraFrame(frame=i, K=K.tolist(), R=R.tolist(),
                                  confidence=1.0, is_anchor=(i == 0)))
    K0, R0, t0 = _camera_pose(0.0)
    CameraTrack(clip_id="play", fps=FPS, image_size=(1280, 720),
                t_world=t0.tolist(), frames=tuple(frames),
                ).save(out / "camera" / "play_camera_track.json")

    ShotsManifest(
        source_file="fake.mp4", fps=FPS, total_frames=N_FRAMES,
        shots=[Shot(id="play", clip_file="shots/play.mp4",
                    start_frame=0, end_frame=N_FRAMES - 1,
                    start_time=0.0, end_time=(N_FRAMES - 1) / FPS)],
    ).save(out / "shots" / "shots_manifest.json")

    tracks = {
        "shot_id": "play",
        "tracks": [{
            "track_id": "1", "class_name": "player", "team": "A",
            "player_id": "P001", "player_name": "",
            "frames": [
                {"frame": i, "bbox": [200.0, 500.0, 260.0, 640.0],
                 "confidence": 0.9, "pitch_position": None,
                 "interpolated": False}
                for i in range(N_FRAMES)
            ],
        }],
    }
    tracks_path = out / "tracks" / "play_tracks.json"
    tracks_path.parent.mkdir(parents=True, exist_ok=True)
    tracks_path.write_text(json.dumps(tracks))
    return out


def _cfg(prior_enabled: bool) -> dict:
    return {
        "ball": {
            "detector": "fake",
            "appearance_bridge": {"enabled": False},
            "second_pass": {"enabled": False},
            "context_prior": {"enabled": prior_enabled},
        },
        "pitch": {"length_m": 105.0, "width_m": 68.0},
    }


def _static_blob_detections() -> list:
    # Confident blob glued to the image near the frame top, every frame —
    # the scoreboard signature (static under pan + no player near).
    return [(640.0, 30.0, 0.8)] * N_FRAMES


@pytest.mark.integration
def test_prior_drops_static_overlay_blob(tmp_path: Path):
    out = _build_scene(tmp_path, panning=True)
    BallStage(config=_cfg(prior_enabled=True), output_dir=out,
              ball_detector=FakeBallDetector(_static_blob_detections())).run()
    obs = json.loads(
        (out / "ball" / "play_ball_observations.json").read_text())
    accepted_late = [
        f for f in obs["frames"]
        if f["frame"] >= 50 and f["source"] == "detector"
        and f["confidence"] > 0.0
    ]
    # After the static window fills, the combined static+player penalties
    # push 0.8 below drop_below and the blob stops being accepted.
    assert accepted_late == [], (
        f"expected the static blob to be dropped after frame 50; "
        f"got {len(accepted_late)} accepted detector frames"
    )


@pytest.mark.integration
def test_prior_disabled_keeps_old_behaviour(tmp_path: Path):
    out = _build_scene(tmp_path, panning=True)
    BallStage(config=_cfg(prior_enabled=False), output_dir=out,
              ball_detector=FakeBallDetector(_static_blob_detections())).run()
    obs = json.loads(
        (out / "ball" / "play_ball_observations.json").read_text())
    accepted_late = [
        f for f in obs["frames"]
        if f["frame"] >= 50 and f["source"] == "detector"
        and f["confidence"] > 0.0
    ]
    assert accepted_late, "disabled prior must not drop anything"


@pytest.mark.integration
def test_genuine_moving_ball_untouched_by_prior(tmp_path: Path):
    out = _build_scene(tmp_path, panning=True)
    K0, R0, t0 = _camera_pose(0.0)
    detections = []
    for i in range(N_FRAMES):
        # Roll across the pitch, near the tracked player's box.
        p = np.array([30.0 + 0.15 * i, 34.0, 0.11])
        Ki, Ri, ti = _camera_pose(yaw_deg=i * 0.2)
        cam = Ri @ p + ti
        pix = Ki @ cam
        detections.append((float(pix[0] / pix[2]), float(pix[1] / pix[2]), 0.9))
    # Put the player box under the rolling ball so proximity never fires.
    tracks_path = out / "tracks" / "play_tracks.json"
    tracks = json.loads(tracks_path.read_text())
    for i, fr in enumerate(tracks["tracks"][0]["frames"]):
        u, v, _ = detections[i]
        fr["bbox"] = [u - 40.0, v - 120.0, u + 40.0, v + 10.0]
    tracks_path.write_text(json.dumps(tracks))

    BallStage(config=_cfg(prior_enabled=True), output_dir=out,
              ball_detector=FakeBallDetector(detections)).run()
    obs = json.loads(
        (out / "ball" / "play_ball_observations.json").read_text())
    accepted = [f for f in obs["frames"]
                if f["source"] == "detector" and f["confidence"] > 0.0]
    assert len(accepted) >= int(0.9 * N_FRAMES), (
        f"prior must not eat a genuine moving ball; accepted {len(accepted)}"
    )

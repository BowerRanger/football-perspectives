"""End-to-end handler test using a fake GVHMR runner.

Mirrors the monkeypatch pattern from ``test_hmr_world_stage.py`` — we
patch ``src.utils.gvhmr_estimator.run_on_track`` at module scope so the
handler's ``process_player`` import picks up the stub.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.cloud.handler import run_local
from src.cloud.manifest import JobManifest
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.smpl_world import SmplWorldTrack


def _identity_camera(n_frames: int) -> CameraTrack:
    """Same camera the stage test uses — see test_hmr_world_stage.py."""
    R_world_to_cam = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    return CameraTrack(
        clip_id="play",
        fps=30.0,
        image_size=(1280, 720),
        t_world=[-52.5, 100.0, 22.0],
        frames=tuple(
            CameraFrame(
                frame=i,
                K=[[1500.0, 0.0, 640.0], [0.0, 1500.0, 360.0], [0.0, 0.0, 1.0]],
                R=R_world_to_cam,
                confidence=1.0,
                is_anchor=(i == 0),
            )
            for i in range(n_frames)
        ),
    )


@pytest.fixture
def fake_run_on_track(monkeypatch):
    def _runner(
        track_frames,
        *,
        video_path,
        checkpoint,
        device,
        batch_size,
        max_sequence_length,
        estimator=None,
    ):
        n = len(track_frames)
        kp2d = np.zeros((n, 17, 3), dtype=np.float32)
        kp2d[:, 15] = (150.0, 380.0, 0.9)
        kp2d[:, 16] = (160.0, 380.0, 0.9)
        x_180 = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        )
        return {
            "thetas": np.zeros((n, 24, 3)),
            "betas": np.tile(np.linspace(0, 1, 10), (n, 1)),
            "root_R_cam": np.tile(x_180, (n, 1, 1)),
            "root_t_cam": np.zeros((n, 3)),
            "joint_confidence": np.full((n, 24), 0.9),
            "kp2d": kp2d,
        }

    monkeypatch.setattr(
        "src.utils.gvhmr_estimator.run_on_track", _runner, raising=False
    )


def test_run_local_writes_smpl_world_and_kp2d(
    tmp_path: Path, fake_run_on_track,
) -> None:
    n_frames = 20
    # Stub video — fake runner doesn't read it; process_player just checks
    # the path exists.
    video = tmp_path / "shot.mp4"
    video.write_bytes(b"")

    cam = _identity_camera(n_frames)
    cam_path = tmp_path / "camera_track.json"
    cam.save(cam_path)

    manifest = JobManifest(
        run_id="test-run",
        shot_id="play",
        player_id="p001",
        video_uri=str(video),
        camera_track_uri=str(cam_path),
        track_frames=tuple(
            (i, (100, 50, 200, 300)) for i in range(n_frames)
        ),
        hmr_world_cfg={
            "device": "cpu",
            "checkpoint": "",
            "batch_size": 16,
            "max_sequence_length": 120,
            "min_track_frames": 10,
            "theta_savgol_window": 11,
            "theta_savgol_order": 2,
            "root_slerp_window": 5,
            "ground_snap_velocity": 0.1,
            "root_t_savgol_window": 5,
            "root_t_savgol_order": 2,
            "lean_correction_deg": 0.0,
        },
        output_prefix=str(tmp_path / "handler-out"),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.to_json())

    output_dir = tmp_path / "handler-out"
    status = run_local(manifest_path=manifest_path, output_dir=output_dir)

    assert status.status == "ok"
    assert status.frames == n_frames

    # Outputs land under output_dir/output/.
    npz_path = output_dir / "output" / "play__p001_smpl_world.npz"
    kp2d_path = output_dir / "output" / "play__p001_kp2d.json"
    assert npz_path.exists()
    assert kp2d_path.exists()

    # Verify the SmplWorldTrack round-trips.
    track = SmplWorldTrack.load(npz_path)
    assert track.player_id == "p001"
    assert track.shot_id == "play"
    assert track.thetas.shape == (n_frames, 24, 3)
    # Root translation should be physically reasonable (foot-anchored).
    assert track.root_t[:, 2].mean() > 0.5

    # kp2d JSON has one entry per frame.
    kp2d_payload = json.loads(kp2d_path.read_text())
    assert kp2d_payload["player_id"] == "p001"
    assert kp2d_payload["shot_id"] == "play"
    assert len(kp2d_payload["frames"]) == n_frames

    # status.json sits at the output prefix root.
    status_path = output_dir / "status.json"
    assert status_path.exists()
    written = json.loads(status_path.read_text())
    assert written["status"] == "ok"
    assert written["frames"] == n_frames


def test_run_local_returns_too_short_below_min_frames(
    tmp_path: Path, fake_run_on_track,
) -> None:
    video = tmp_path / "shot.mp4"
    video.write_bytes(b"")
    cam_path = tmp_path / "camera_track.json"
    _identity_camera(5).save(cam_path)

    manifest = JobManifest(
        run_id="test-run",
        shot_id="play",
        player_id="p001",
        video_uri=str(video),
        camera_track_uri=str(cam_path),
        track_frames=tuple((i, (0, 0, 10, 10)) for i in range(3)),
        hmr_world_cfg={
            "min_track_frames": 10,
            "theta_savgol_window": 3,
            "theta_savgol_order": 1,
            "root_slerp_window": 1,
            "root_t_savgol_window": 1,
            "root_t_savgol_order": 1,
            "lean_correction_deg": 0.0,
        },
        output_prefix=str(tmp_path / "out"),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.to_json())

    status = run_local(manifest_path=manifest_path, output_dir=tmp_path / "out")
    assert status.status == "too_short"

"""End-to-end test for HmrWorldStage with a fake GVHMR runner.

Bypasses GVHMR weights via monkeypatching ``run_on_track`` so the test
runs in unit-test time without ML dependencies. Validates that the stage:
  * Reads tracks/pose_2d/camera inputs and writes a SmplWorldTrack.
  * Produces θ shape (N, 24, 3) consistent with GVHMR adapter contract.
  * Produces a physically reasonable root translation z (>0.5) when the
    ankle keypoint is anchored to the ground plane and the foot offset is
    interpreted in pitch-world coordinates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.smpl_world import SmplWorldTrack
from src.stages.hmr_world import HmrWorldStage


def _identity_track(n_frames: int) -> CameraTrack:
    """A camera oriented so SMPL-canonical frame maps to pitch-world directly.

    With ``R_world_to_cam = SMPL_TO_PITCH_STATIC``, the
    ``smpl_root_in_pitch_frame`` formula
    ``R_world_to_cam.T @ SMPL_TO_PITCH_STATIC @ root_R_cam``
    reduces to ``root_R_cam`` itself (since SMPL_TO_PITCH_STATIC is
    orthogonal). Combined with the fake's identity ``root_R_cam`` this
    keeps the foot offset ``(0, 0, -0.95)`` aligned with pitch -z, so
    a foot at z=0.05 yields root z=1.0 — clean to assert against.

    See decision-log D10 for why the plan's original camera orientation
    was changed for this fixture.
    """
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
def fake_gvhmr(monkeypatch):
    """Replace run_on_track with a deterministic stub that needs no weights."""

    def _runner(track_frames, *, video_path, checkpoint, device, batch_size, max_sequence_length):
        n = len(track_frames)
        return {
            "thetas": np.zeros((n, 24, 3)),
            "betas": np.tile(np.linspace(0, 1, 10), (n, 1)),
            "root_R_cam": np.tile(np.eye(3), (n, 1, 1)),
            "root_t_cam": np.zeros((n, 3)),
            "joint_confidence": np.full((n, 24), 0.9),
        }

    monkeypatch.setattr(
        "src.utils.gvhmr_estimator.run_on_track", _runner, raising=False
    )


@pytest.mark.integration
def test_hmr_world_emits_track_in_pitch_frame(tmp_path: Path, fake_gvhmr) -> None:
    n_frames = 30

    # 1. Empty stub video — the fake runner doesn't read it.
    (tmp_path / "shots").mkdir()
    (tmp_path / "shots" / "play.mp4").write_bytes(b"")

    # 2. Camera track with the SMPL-aligned orientation (see D10).
    track = _identity_track(n_frames)
    track.save(tmp_path / "camera" / "camera_track.json")

    # 3. Steady bounding-box player track.
    track_dir = tmp_path / "tracks"
    track_dir.mkdir()
    (track_dir / "P001_track.json").write_text(
        json.dumps(
            {
                "player_id": "P001",
                "frames": [
                    {"frame": i, "bbox": [100, 100, 200, 400]} for i in range(n_frames)
                ],
            }
        )
    )

    # 4. 2D pose with strong ankle keypoints; other joints zero-confidence.
    pose_dir = tmp_path / "pose_2d"
    pose_dir.mkdir()
    (pose_dir / "P001_pose.json").write_text(
        json.dumps(
            {
                "player_id": "P001",
                "frames": [
                    {
                        "frame": i,
                        "keypoints": (
                            [[0, 0, 0.0]] * 15
                            + [[150.0, 380.0, 0.9], [160.0, 380.0, 0.9]]
                        ),
                    }
                    for i in range(n_frames)
                ],
            }
        )
    )

    # 5. Run stage. ground_snap_velocity=0 disables snapping for this fixture
    # (all velocities are zero so the default would halve every frame's z).
    stage = HmrWorldStage(
        config={
            "hmr_world": {
                "min_track_frames": 5,
                "checkpoint": "ignored",
                "ground_snap_velocity": 0.0,
            }
        },
        output_dir=tmp_path,
    )
    stage.run()

    # 6. Verify output.
    out_path = tmp_path / "hmr_world" / "P001_smpl_world.npz"
    assert out_path.exists(), "stage did not write SmplWorldTrack output"

    out = SmplWorldTrack.load(out_path)
    assert out.player_id == "P001"
    assert out.thetas.shape == (n_frames, 24, 3)
    # Root z should be > 0.5 for at least some frames (foot at ground,
    # root ~1m above pitch).
    assert (out.root_t[:, 2] > 0.5).any()

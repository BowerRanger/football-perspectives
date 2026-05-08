"""End-to-end test for HmrWorldStage with a fake GVHMR runner.

Bypasses GVHMR weights via monkeypatching ``run_on_track`` so the test
runs in unit-test time without ML dependencies. Validates that the stage:
  * Reads tracks/camera inputs and writes a SmplWorldTrack.
  * Produces θ shape (N, 24, 3) consistent with GVHMR adapter contract.
  * Produces a physically reasonable root translation z (>0.5) when the
    ankle keypoint (sourced from GVHMR's internal ViTPose, returned via
    ``run_on_track``'s ``kp2d`` array) is anchored to the ground plane.
  * Writes a side-output ``{player_id}_kp2d.json`` for the dashboard
    overlay.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.smpl_world import SmplWorldTrack
from src.schemas.tracks import Track, TrackFrame, TracksResult
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
    """Replace run_on_track with a deterministic stub that needs no weights.

    Emits high-confidence ankle keypoints at a fixed pixel for every frame —
    the foot-anchor ray-cast through that pixel onto the pitch ground plane
    drives the root-z assertion downstream.
    """

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
        # COCO-17: keypoints 15/16 are left/right ankles. Other joints are
        # zero-confidence (don't contribute to the foot anchor).
        kp2d = np.zeros((n, 17, 3), dtype=np.float32)
        kp2d[:, 15] = (150.0, 380.0, 0.9)  # left ankle
        kp2d[:, 16] = (160.0, 380.0, 0.9)  # right ankle
        return {
            "thetas": np.zeros((n, 24, 3)),
            "betas": np.tile(np.linspace(0, 1, 10), (n, 1)),
            "root_R_cam": np.tile(np.eye(3), (n, 1, 1)),
            "root_t_cam": np.zeros((n, 3)),
            "joint_confidence": np.full((n, 24), 0.9),
            "kp2d": kp2d,
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

    # 3. Steady bounding-box player track in the TracksResult format the
    # tracking stage emits (one *_tracks.json per shot, containing a list
    # of Tracks). hmr_world groups frames by Track.player_id, falling back
    # to track_id when player_id is empty.
    track_dir = tmp_path / "tracks"
    track_dir.mkdir()
    tr = TracksResult(
        shot_id="play",
        tracks=[
            Track(
                track_id="T001",
                class_name="player",
                team="A",
                player_id="P001",
                player_name="",
                frames=[
                    TrackFrame(frame=i, bbox=[100, 100, 200, 400], confidence=0.9, pitch_position=None)
                    for i in range(n_frames)
                ],
            ),
        ],
    )
    tr.save(track_dir / "play_tracks.json")

    # 4. Run stage. ground_snap_velocity=0 disables snapping for this fixture
    # (all velocities are zero so the default would halve every frame's z).
    # Ankle keypoints come from the fake GVHMR runner's kp2d output.
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

    # 5. Verify SmplWorldTrack output.
    out_path = tmp_path / "hmr_world" / "P001_smpl_world.npz"
    assert out_path.exists(), "stage did not write SmplWorldTrack output"

    out = SmplWorldTrack.load(out_path)
    assert out.player_id == "P001"
    assert out.thetas.shape == (n_frames, 24, 3)
    # Root z should be > 0.5 for at least some frames (foot at ground,
    # root ~1m above pitch).
    assert (out.root_t[:, 2] > 0.5).any()

    # 6. Verify kp2d side-output written for the dashboard overlay.
    kp2d_path = tmp_path / "hmr_world" / "P001_kp2d.json"
    assert kp2d_path.exists(), "stage did not write kp2d preview JSON"
    kp2d_data = json.loads(kp2d_path.read_text())
    assert kp2d_data["player_id"] == "P001"
    assert len(kp2d_data["frames"]) == n_frames
    # Ankle indices (15/16) carry the seeded values; tolerance is for the
    # float32 round-trip through the runner.
    first_frame_kps = kp2d_data["frames"][0]["keypoints"]
    assert first_frame_kps[15] == pytest.approx([150.0, 380.0, 0.9], abs=1e-5)
    assert first_frame_kps[16] == pytest.approx([160.0, 380.0, 0.9], abs=1e-5)


@pytest.mark.integration
def test_hmr_world_reuses_one_estimator_across_players(
    tmp_path: Path, monkeypatch
) -> None:
    """Slice 1 speed-up: one GVHMREstimator should serve all players in a run.

    Previously ``run_on_track`` constructed a fresh ``GVHMREstimator`` per
    call, paying the 30-60s GVHMR + ViTPose-Huge + HMR2-ViT + SMPLX load
    cost for every player. The stage now builds one estimator before the
    player loop and threads it through. This test captures the estimator
    identity each call sees and asserts it's a non-None instance and the
    same object across all players.
    """
    n_frames = 20
    n_players = 3

    (tmp_path / "shots").mkdir()
    (tmp_path / "shots" / "play.mp4").write_bytes(b"")

    track = _identity_track(n_frames)
    track.save(tmp_path / "camera" / "camera_track.json")

    track_dir = tmp_path / "tracks"
    track_dir.mkdir()
    tr = TracksResult(
        shot_id="play",
        tracks=[
            Track(
                track_id=f"T{p:03d}",
                class_name="player",
                team="A",
                player_id=f"P{p:03d}",
                player_name="",
                frames=[
                    TrackFrame(
                        frame=i,
                        bbox=[100, 100, 200, 400],
                        confidence=0.9,
                        pitch_position=None,
                    )
                    for i in range(n_frames)
                ],
            )
            for p in range(1, n_players + 1)
        ],
    )
    tr.save(track_dir / "play_tracks.json")

    seen_estimator_ids: list[int | None] = []

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
        seen_estimator_ids.append(id(estimator) if estimator is not None else None)
        n = len(track_frames)
        kp2d = np.zeros((n, 17, 3), dtype=np.float32)
        kp2d[:, 15] = (150.0, 380.0, 0.9)
        kp2d[:, 16] = (160.0, 380.0, 0.9)
        return {
            "thetas": np.zeros((n, 24, 3)),
            "betas": np.tile(np.linspace(0, 1, 10), (n, 1)),
            "root_R_cam": np.tile(np.eye(3), (n, 1, 1)),
            "root_t_cam": np.zeros((n, 3)),
            "joint_confidence": np.full((n, 24), 0.9),
            "kp2d": kp2d,
        }

    monkeypatch.setattr(
        "src.utils.gvhmr_estimator.run_on_track", _runner, raising=False
    )

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

    assert len(seen_estimator_ids) == n_players, (
        f"expected one run_on_track call per player, got {len(seen_estimator_ids)}"
    )
    assert all(eid is not None for eid in seen_estimator_ids), (
        "stage passed estimator=None — caching is disabled"
    )
    assert len(set(seen_estimator_ids)) == 1, (
        f"each player got a different estimator instance: {seen_estimator_ids}"
    )

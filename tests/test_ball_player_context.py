"""PlayerContext: per-frame world+pixel positions of contact joints.

Feeds the ball stage's automatic contact detection. Verifies FK matches
the reference implementation, refined_poses precedence over hmr_world
(with sync-offset translation), pixel projection, and graceful
degradation when upstream artifacts are missing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.schemas.refined_pose import RefinedPose
from src.schemas.smpl_world import SmplWorldTrack
from src.schemas.sync_map import Alignment, GroupSync, SyncMap
from src.utils.ball_anchor_heights import BONE_TO_SMPL_INDEX
from src.utils.ball_player_context import JointSample, PlayerContext
from src.utils.camera_projection import project_world_to_image
from src.utils.smpl_skeleton import compute_joint_world

SHOT = "shot01"
N_FRAMES = 5


def _camera_looking_down_pitch() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple broadcast-ish camera: above the near touchline at y=-20,
    z=15, looking at the pitch centre. OpenCV convention (z into scene,
    y down)."""
    K = np.array([[1200.0, 0.0, 960.0], [0.0, 1200.0, 540.0], [0.0, 0.0, 1.0]])
    cam_centre = np.array([52.5, -20.0, 15.0])
    target = np.array([52.5, 34.0, 0.0])
    fwd = target - cam_centre
    fwd = fwd / np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])  # rows: cam x, y, z in world coords
    t = -R @ cam_centre
    return K, R, t


def _per_frame_cams(n: int):
    K, R, t = _camera_looking_down_pitch()
    return (
        {i: K for i in range(n)},
        {i: R for i in range(n)},
        {i: t for i in range(n)},
    )


def _make_track_arrays(n: int, *, seed: int, base_xy: tuple[float, float]):
    rng = np.random.default_rng(seed)
    frames = np.arange(n, dtype=np.int64)
    betas = np.zeros(10, dtype=np.float32)
    thetas = (0.05 * rng.standard_normal((n, 24, 3))).astype(np.float32)
    # Canonical y-up -> pitch z-up, with a small per-frame yaw.
    base = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    root_R = np.stack([base for _ in range(n)]).astype(np.float32)
    root_t = np.stack(
        [np.array([base_xy[0] + 0.1 * i, base_xy[1], 0.95]) for i in range(n)]
    ).astype(np.float32)
    confidence = np.full(n, 0.8, dtype=np.float32)
    return frames, betas, thetas, root_R, root_t, confidence


def _write_hmr_track(out: Path, player_id: str, *, seed: int,
                     base_xy=(50.0, 30.0), n: int = N_FRAMES) -> SmplWorldTrack:
    frames, betas, thetas, root_R, root_t, conf = _make_track_arrays(
        n, seed=seed, base_xy=base_xy
    )
    track = SmplWorldTrack(
        player_id=player_id, frames=frames, betas=betas, thetas=thetas,
        root_R=root_R, root_t=root_t, confidence=conf, shot_id=SHOT,
    )
    track.save(out / "hmr_world" / f"{SHOT}__{player_id}_smpl_world.npz")
    return track


def _write_refined_track(out: Path, player_id: str, *, seed: int,
                         frame_shift: int = 0, base_xy=(50.0, 30.0),
                         n: int = N_FRAMES) -> RefinedPose:
    frames, betas, thetas, root_R, root_t, conf = _make_track_arrays(
        n, seed=seed, base_xy=base_xy
    )
    track = RefinedPose(
        player_id=player_id, frames=frames + frame_shift, betas=betas,
        thetas=thetas, root_R=root_R, root_t=root_t, confidence=conf,
        view_count=np.ones(n, dtype=np.int32),
        contributing_shots=(SHOT,),
    )
    track.save(out / "refined_poses" / f"{player_id}_refined.npz")
    return track


def _write_sync_map(out: Path, offset: int) -> None:
    smap = SyncMap(groups=[GroupSync(
        group_id="g1", reference_shot="other_shot",
        alignments=[Alignment(shot_id=SHOT, frame_offset=offset)],
    )])
    smap.save(out / "shots" / "sync_map.json")


def _load_ctx(out: Path, n: int = N_FRAMES) -> PlayerContext:
    Ks, Rs, ts = _per_frame_cams(n)
    return PlayerContext.load(
        out, SHOT,
        per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
        distortion=(0.0, 0.0),
    )


class TestFkAndProjection:
    def test_joint_world_matches_reference_fk(self, tmp_path):
        track = _write_hmr_track(tmp_path, "P001", seed=1)
        ctx = _load_ctx(tmp_path)
        for bone, joint_idx in BONE_TO_SMPL_INDEX.items():
            got = ctx.joint_world(2, "P001", bone)
            expected = compute_joint_world(
                track.thetas[2], track.root_R[2], track.root_t[2], joint_idx
            )
            assert got is not None
            np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_samples_carry_projected_pixels(self, tmp_path):
        _write_hmr_track(tmp_path, "P001", seed=2)
        ctx = _load_ctx(tmp_path)
        samples = ctx.joints_at(0)
        assert len(samples) == len(BONE_TO_SMPL_INDEX)
        Ks, Rs, ts = _per_frame_cams(1)
        for s in samples:
            assert isinstance(s, JointSample)
            assert s.uv is not None
            expected_uv = project_world_to_image(
                Ks[0], Rs[0], ts[0], (0.0, 0.0),
                np.asarray(s.world_xyz, dtype=float).reshape(1, 3),
            )[0]
            np.testing.assert_allclose(s.uv, expected_uv, atol=1e-4)
            assert s.confidence == pytest.approx(0.8)

    def test_joints_near_pixel_filters_and_sorts(self, tmp_path):
        # Two players far apart on the pitch -> far apart in pixels.
        _write_hmr_track(tmp_path, "P001", seed=3, base_xy=(40.0, 30.0))
        _write_hmr_track(tmp_path, "P002", seed=4, base_xy=(70.0, 30.0))
        ctx = _load_ctx(tmp_path)
        p1_foot = next(
            s for s in ctx.joints_at(0)
            if s.player_id == "P001" and s.bone == "l_foot"
        )
        near = ctx.joints_near_pixel(0, p1_foot.uv, radius_px=40.0)
        assert near, "expected at least the queried joint itself"
        assert all(s.player_id == "P001" for s in near)
        # Sorted by pixel distance: first one is the queried joint.
        assert near[0].bone == "l_foot"

    def test_frames_without_pose_are_empty(self, tmp_path):
        _write_hmr_track(tmp_path, "P001", seed=5, n=2)
        ctx = _load_ctx(tmp_path, n=5)
        assert ctx.joints_at(4) == ()
        assert ctx.joint_world(4, "P001", "head") is None


class TestSourcePrecedence:
    def test_refined_wins_over_hmr(self, tmp_path):
        _write_hmr_track(tmp_path, "P001", seed=6, base_xy=(40.0, 30.0))
        refined = _write_refined_track(
            tmp_path, "P001", seed=7, base_xy=(60.0, 40.0)
        )
        ctx = _load_ctx(tmp_path)
        got = ctx.joint_world(1, "P001", "head")
        expected = compute_joint_world(
            refined.thetas[1], refined.root_R[1], refined.root_t[1],
            BONE_TO_SMPL_INDEX["head"],
        )
        np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_refined_respects_sync_offset(self, tmp_path):
        # Shot is 10 frames ahead of the reference: local f -> ref f-10.
        offset = 10
        _write_sync_map(tmp_path, offset)
        refined = _write_refined_track(
            tmp_path, "P001", seed=8, frame_shift=-offset
        )
        ctx = _load_ctx(tmp_path)
        got = ctx.joint_world(3, "P001", "r_foot")
        # local 3 -> ref 3-10 = -7, stored at array index 3 (frames start
        # at -offset).
        expected = compute_joint_world(
            refined.thetas[3], refined.root_R[3], refined.root_t[3],
            BONE_TO_SMPL_INDEX["r_foot"],
        )
        assert got is not None
        np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_hmr_fallback_when_refined_missing_frame(self, tmp_path):
        # Refined covers only frames 0-1; hmr covers 0-4. Frames 2+ fall
        # back to the hmr track.
        hmr = _write_hmr_track(tmp_path, "P001", seed=9)
        _write_refined_track(tmp_path, "P001", seed=10, n=2)
        ctx = _load_ctx(tmp_path)
        got = ctx.joint_world(3, "P001", "head")
        expected = compute_joint_world(
            hmr.thetas[3], hmr.root_R[3], hmr.root_t[3],
            BONE_TO_SMPL_INDEX["head"],
        )
        np.testing.assert_allclose(got, expected, atol=1e-5)


class TestDegradation:
    def test_empty_output_dir(self, tmp_path):
        ctx = _load_ctx(tmp_path)
        assert ctx.joints_at(0) == ()
        assert ctx.joints_near_pixel(0, (100.0, 100.0), radius_px=50.0) == []
        assert ctx.joint_world(0, "P001", "head") is None
        assert ctx.player_ids == ()

    def test_unknown_bone_returns_none(self, tmp_path):
        _write_hmr_track(tmp_path, "P001", seed=11)
        ctx = _load_ctx(tmp_path)
        assert ctx.joint_world(0, "P001", "left_pinky") is None

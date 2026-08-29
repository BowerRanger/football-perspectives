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
from src.utils.ball_player_context import (
    JointSample,
    PlayerContext,
    _discover_hmr,
    _discover_refined,
    _load_hmr,
    _load_refined,
    _sync_offset,
)
from src.utils.camera_projection import project_world_to_image
from src.utils.smpl_skeleton import compute_all_joint_worlds, compute_joint_world

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


def _camera_looking_away_from_pitch() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same intrinsics/position as ``_camera_looking_down_pitch`` but the
    optical axis points away from the pitch, so pitch-side joints land
    behind the camera (cam_z <= 0) -> ``uv=None``."""
    K = np.array([[1200.0, 0.0, 960.0], [0.0, 1200.0, 540.0], [0.0, 0.0, 1.0]])
    cam_centre = np.array([52.5, -20.0, 15.0])
    target = np.array([52.5, -60.0, 40.0])  # away from the pitch, not toward it
    fwd = target - cam_centre
    fwd = fwd / np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    t = -R @ cam_centre
    return K, R, t


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


def _reference_load(
    output_dir: Path,
    shot_id: str,
    *,
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float] = (0.0, 0.0),
    bones: tuple[str, ...] | None = None,
) -> dict[int, tuple[JointSample, ...]]:
    """Frozen copy of the PRE-batch ``PlayerContext.load`` frame/player
    loop: one ``compute_all_joint_worlds`` (single-frame FK) call per
    player per frame, one ``project_world_to_image`` call per player per
    frame. This is the oracle the batched-FK rewrite must match exactly
    (world_xyz, uv incl. None, ordering). Reuses the unchanged discovery
    helpers (``_discover_refined`` etc.) since only the FK/projection
    batching is in scope for the rewrite, not source discovery.
    """
    bone_names = tuple(bones) if bones is not None else tuple(BONE_TO_SMPL_INDEX)
    joint_indices = np.array(
        [BONE_TO_SMPL_INDEX[b] for b in bone_names], dtype=int
    )
    offset = _sync_offset(output_dir, shot_id)
    refined_paths = _discover_refined(output_dir)
    hmr_paths = _discover_hmr(output_dir, shot_id)

    loaded = {}
    for pid in sorted(set(refined_paths) | set(hmr_paths)):
        refined = (
            _load_refined(refined_paths[pid], offset)
            if pid in refined_paths else None
        )
        hmr = _load_hmr(hmr_paths[pid]) if pid in hmr_paths else None
        if refined is not None or hmr is not None:
            loaded[pid] = (refined, hmr)

    samples_by_frame: dict[int, tuple[JointSample, ...]] = {}
    for fi in sorted(per_frame_K):
        K, R, t = per_frame_K[fi], per_frame_R[fi], per_frame_t[fi]
        frame_samples: list[JointSample] = []
        for pid, (refined, hmr) in loaded.items():
            chosen = None
            idx = None
            for candidate in (refined, hmr):
                if candidate is None:
                    continue
                idx = candidate.index_by_local_frame.get(fi)
                if idx is not None:
                    chosen = candidate
                    break
            if chosen is None or idx is None:
                continue
            track = chosen.track
            all_joints = compute_all_joint_worlds(
                track.thetas[idx], track.root_R[idx], track.root_t[idx],
            )
            worlds = all_joints[joint_indices]
            cam_z = (worlds @ R.T + t)[:, 2]
            uvs = project_world_to_image(K, R, t, distortion, worlds)
            conf = float(track.confidence[idx])
            for bone, world, uv, z in zip(bone_names, worlds, uvs, cam_z):
                frame_samples.append(JointSample(
                    player_id=pid,
                    bone=bone,
                    world_xyz=(
                        float(world[0]), float(world[1]), float(world[2])
                    ),
                    uv=(float(uv[0]), float(uv[1])) if z > 0 else None,
                    confidence=conf,
                ))
        if frame_samples:
            samples_by_frame[fi] = tuple(frame_samples)
    return samples_by_frame


class TestBatchedFkRegression:
    """Pins ``PlayerContext.load`` against the pre-batch reference loop
    on a synthetic multi-frame, multi-player track that exercises:
    refined source, hmr fallback (mixed per-frame per-player, i.e. the
    same player drawing FK inputs from two different track arrays across
    the query range), a sync offset, and a forced ``uv=None`` frame
    (camera facing away from the pitch).
    """

    N = 6

    def _build_track(self, tmp_path: Path) -> tuple[
        dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]
    ]:
        offset = 2
        _write_sync_map(tmp_path, offset)
        # P001: hmr-only, covers every query frame.
        _write_hmr_track(tmp_path, "P001", seed=101, base_xy=(40.0, 25.0), n=self.N)
        # P002: refined covers local frames 0-3 (via frame_shift=-offset),
        # hmr covers all 6 -> frames 4-5 fall back to hmr. Exercises a
        # single player's FK batch drawing rows from two different track
        # objects at different per-track indices.
        _write_refined_track(
            tmp_path, "P002", seed=102, frame_shift=-offset,
            base_xy=(55.0, 35.0), n=4,
        )
        _write_hmr_track(tmp_path, "P002", seed=103, base_xy=(55.0, 35.0), n=self.N)

        Ks, Rs, ts = _per_frame_cams(self.N)
        # Frame 4's camera faces away from the pitch -> every sample at
        # that frame must come back with uv=None.
        K_away, R_away, t_away = _camera_looking_away_from_pitch()
        Ks[4], Rs[4], ts[4] = K_away, R_away, t_away
        return Ks, Rs, ts

    def test_batched_load_matches_reference_loop(self, tmp_path):
        Ks, Rs, ts = self._build_track(tmp_path)
        distortion = (0.1, -0.02)

        expected = _reference_load(
            tmp_path, SHOT,
            per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
            distortion=distortion,
        )
        ctx = PlayerContext.load(
            tmp_path, SHOT,
            per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
            distortion=distortion,
        )

        # Sanity: the scenario actually exercises what it claims to.
        assert expected, "reference loop produced no samples at all"
        assert 4 in expected, "expected samples at the away-camera frame"
        assert any(s.uv is None for s in expected[4]), (
            "away-camera frame should yield uv=None samples"
        )
        assert any(s.player_id == "P002" for s in expected[0]), (
            "P002 should be present via refined at frame 0"
        )
        assert any(s.player_id == "P002" for s in expected[5]), (
            "P002 should be present via hmr fallback at frame 5"
        )

        assert set(ctx._by_frame.keys()) == set(expected.keys())
        for fi, expected_samples in expected.items():
            got_samples = ctx.joints_at(fi)
            assert len(got_samples) == len(expected_samples), fi
            for got, exp in zip(got_samples, expected_samples):
                assert got.player_id == exp.player_id
                assert got.bone == exp.bone
                assert got.confidence == pytest.approx(exp.confidence)
                np.testing.assert_allclose(
                    got.world_xyz, exp.world_xyz, atol=1e-9, rtol=1e-9,
                )
                if exp.uv is None:
                    assert got.uv is None, (fi, got.player_id, got.bone)
                else:
                    assert got.uv is not None, (fi, got.player_id, got.bone)
                    np.testing.assert_allclose(
                        got.uv, exp.uv, atol=1e-6, rtol=1e-6,
                    )

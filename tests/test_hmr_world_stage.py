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
from src.stages.hmr_world import HmrWorldStage, build_track_camera_R


def _identity_track(n_frames: int) -> CameraTrack:
    """A camera oriented so an upright body in pitch yields the SMPL
    canonical pelvis height above the ankle-anchor plane.

    Convention reminders (post the Phase-2 simplification of
    ``smpl_pitch_transform``):
      - Pitch world is z-up.
      - OpenCV camera is y-down, z-into-scene.
      - The chain is now ``R_w2c.T @ root_R_cam`` (no X_180 bridge).
      - For an upright body in OpenCV cam, ``root_R_cam`` is ``X_180``
        (canonical y-up → cam y-down) — see the fake runner below for
        the actual matrix used.

    With ``R_w2c = [[1,0,0],[0,0,-1],[0,1,0]]`` and ``root_R_cam = X_180``,
    ``root_R_pitch @ (0, -0.882, -0.038) ≈ (0, 0.038, -0.882)`` — the
    SMPL canonical ankle offset maps cleanly to pitch -z, so an ankle
    anchored at z=0.05 yields root z ≈ 0.932 m (SMPL canonical pelvis
    height).
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


class _StubGVHMREstimator:
    """Stand-in for the real ``GVHMREstimator``.

    hmr_world.py codes against the frozen interface
    ``GVHMREstimator(checkpoint, device=..., extractor_device="cpu")``
    regardless of whether ``src/utils/gvhmr_estimator.py`` (owned by a
    concurrent edit) has landed the ``extractor_device`` kwarg yet. Tests
    monkeypatch this stub in so they never depend on that landing order,
    and never need torch/GVHMR weights.
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


@pytest.fixture
def fake_gvhmr(monkeypatch):
    """Replace run_on_track (and the GVHMREstimator constructor) with
    deterministic stubs that need no weights.

    Emits high-confidence ankle keypoints at a fixed pixel for every frame —
    the foot-anchor ray-cast through that pixel onto the pitch ground plane
    drives the root-z assertion downstream.
    """
    monkeypatch.setattr(
        "src.utils.gvhmr_estimator.GVHMREstimator",
        _StubGVHMREstimator,
        raising=False,
    )

    def _runner(
        track_frames,
        *,
        video_path,
        checkpoint,
        device,
        extractor_device=None,
        batch_size,
        max_sequence_length,
        estimator=None,
        per_frame_K=None,
        per_frame_R=None,
    ):
        n = len(track_frames)
        # COCO-17: keypoints 15/16 are left/right ankles. Other joints are
        # zero-confidence (don't contribute to the foot anchor).
        kp2d = np.zeros((n, 17, 3), dtype=np.float32)
        kp2d[:, 15] = (150.0, 380.0, 0.9)  # left ankle
        kp2d[:, 16] = (160.0, 380.0, 0.9)  # right ankle
        # X_180 around +x: maps SMPL canonical y-up to OpenCV camera
        # y-down. That's what GVHMR's smpl_params_incam.global_orient
        # gives for an upright body under the post-Phase-2 chain.
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

    # 5. Verify SmplWorldTrack output. The new per-shot scheme keys
    # outputs by ``{shot_id}__{player_id}`` so two shots that share a
    # player_id can't overwrite each other.
    out_path = tmp_path / "hmr_world" / "play__P001_smpl_world.npz"
    assert out_path.exists(), "stage did not write SmplWorldTrack output"

    out = SmplWorldTrack.load(out_path)
    assert out.player_id == "P001"
    assert out.thetas.shape == (n_frames, 24, 3)
    # Root z should land at the SMPL canonical ankle-to-pelvis distance
    # above the ankle-anchor plane (z=0.05): the ankle joint sits at
    # y=-0.882 in the SMPL canonical y-up rest pose, so an upright body
    # anchored with its ankle at z=0.05 has its pelvis at z ≈ 0.932 m,
    # NOT z=1.0 m (the latter would correspond to a 0.95 m sole offset,
    # which is the bug that left players floating ~7 cm above the pitch).
    from src.utils.smpl_skeleton import SMPL_REST_JOINTS_YUP
    expected_pelvis_z = 0.05 + (-SMPL_REST_JOINTS_YUP[7][1])  # 0.05 + 0.882
    assert out.root_t[:, 2] == pytest.approx(expected_pelvis_z, abs=1e-3), (
        f"pelvis z {out.root_t[0, 2]:.4f} m doesn't match SMPL canonical "
        f"ankle anchor height {expected_pelvis_z:.4f} m — floating bug regressed"
    )

    # 6. Verify kp2d side-output written for the dashboard overlay.
    kp2d_path = tmp_path / "hmr_world" / "play__P001_kp2d.json"
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

    monkeypatch.setattr(
        "src.utils.gvhmr_estimator.GVHMREstimator",
        _StubGVHMREstimator,
        raising=False,
    )

    seen_estimator_ids: list[int | None] = []

    def _runner(
        track_frames,
        *,
        video_path,
        checkpoint,
        device,
        extractor_device=None,
        batch_size,
        max_sequence_length,
        estimator=None,
        per_frame_K=None,
        per_frame_R=None,
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


@pytest.mark.unit
def test_unannotated_track_id_is_prefixed_with_shot(tmp_path: Path) -> None:
    """ByteTrack reuses track_id=3 across shots; the player_id used by
    hmr_world must be unique across shots when the user hasn't yet
    annotated a name. Format: ``{shot_id}_T{track_id}``."""
    from src.schemas.shots import Shot, ShotsManifest

    (tmp_path / "tracks").mkdir()
    for shot_id in ("alpha", "beta"):
        TracksResult(
            shot_id=shot_id,
            tracks=[Track(
                track_id="3",            # collision across shots
                player_id=None,          # not annotated
                player_name=None,
                team=None,
                class_name="player",
                frames=[TrackFrame(
                    frame=0, bbox=[0, 0, 10, 10], confidence=0.9,
                    pitch_position=None,
                )],
            )],
        ).save(tmp_path / "tracks" / f"{shot_id}_tracks.json")
    (tmp_path / "shots").mkdir()
    ShotsManifest(
        source_file="x", fps=25.0, total_frames=1,
        shots=[
            Shot(id="alpha", start_frame=0, end_frame=0, start_time=0,
                 end_time=0.04, clip_file="shots/alpha.mp4"),
            Shot(id="beta", start_frame=0, end_frame=0, start_time=0,
                 end_time=0.04, clip_file="shots/beta.mp4"),
        ],
    ).save(tmp_path / "shots" / "shots_manifest.json")

    stage = HmrWorldStage(config={}, output_dir=tmp_path)
    groups = stage._build_player_groups()
    # Per-shot grouping: every key is now a (shot_id, player_id) tuple
    # so two shots reusing the same ByteTrack track_id stay separate.
    assert ("alpha", "alpha_T3") in groups, f"got keys: {list(groups)}"
    assert ("beta", "beta_T3") in groups, f"got keys: {list(groups)}"


@pytest.mark.unit
def test_hmr_world_loads_per_shot_camera_tracks(tmp_path: Path) -> None:
    """Two shots each with their own camera_track.json — both files exist
    after prepare_shots/camera; hmr_world.run() must not fail when only
    finding the per-shot files (no legacy camera_track.json)."""
    from src.schemas.shots import Shot, ShotsManifest

    eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    cam_alpha = CameraTrack(
        clip_id="alpha", fps=25.0, image_size=(640, 360),
        t_world=[0.0, 0.0, 0.0],
        frames=(CameraFrame(
            frame=0, K=eye, R=eye, confidence=1.0, is_anchor=True,
        ),),
    )
    cam_beta = CameraTrack(
        clip_id="beta", fps=25.0, image_size=(640, 360),
        t_world=[0.0, 0.0, 0.0],
        frames=(CameraFrame(
            frame=0, K=eye, R=eye, confidence=1.0, is_anchor=True,
        ),),
    )
    (tmp_path / "camera").mkdir(parents=True)
    cam_alpha.save(tmp_path / "camera" / "alpha_camera_track.json")
    cam_beta.save(tmp_path / "camera" / "beta_camera_track.json")
    (tmp_path / "shots").mkdir()
    ShotsManifest(
        source_file=str(tmp_path), fps=25.0, total_frames=2,
        shots=[
            Shot(id="alpha", start_frame=0, end_frame=0, start_time=0.0,
                 end_time=0.04, clip_file="shots/alpha.mp4"),
            Shot(id="beta", start_frame=0, end_frame=0, start_time=0.0,
                 end_time=0.04, clip_file="shots/beta.mp4"),
        ],
    ).save(tmp_path / "shots" / "shots_manifest.json")

    # No tracks dir → stage returns early after loading cameras. The
    # invariant we're checking is that loading both per-shot files works.
    stage = HmrWorldStage(config={"hmr_world": {"min_track_frames": 1}}, output_dir=tmp_path)
    stage.run()  # must not raise
    # Sanity: per-shot tracks did get saved (we just authored them above).
    assert (tmp_path / "camera" / "alpha_camera_track.json").exists()
    assert (tmp_path / "camera" / "beta_camera_track.json").exists()


@pytest.mark.unit
def test_same_player_id_in_two_shots_stays_separate(tmp_path: Path) -> None:
    """The same ``player_id`` in two shots (e.g. after Merge by Name)
    must yield two separate groups so hmr_world solves each shot
    against its own camera and writes one .npz per shot."""
    track_dir = tmp_path / "tracks"
    track_dir.mkdir()
    for shot in ("alpha", "beta"):
        TracksResult(
            shot_id=shot,
            tracks=[Track(
                track_id="1",
                player_id="P001",
                player_name="Origi",
                team="A",
                class_name="player",
                frames=[TrackFrame(
                    frame=i, bbox=[0, 0, 10, 10], confidence=0.9,
                    pitch_position=None,
                ) for i in range(3)],
            )],
        ).save(track_dir / f"{shot}_tracks.json")

    stage = HmrWorldStage(config={}, output_dir=tmp_path)
    groups = stage._build_player_groups()
    assert ("alpha", "P001") in groups
    assert ("beta", "P001") in groups
    assert len(groups[("alpha", "P001")]) == 3
    assert len(groups[("beta", "P001")]) == 3


@pytest.mark.unit
def test_legacy_outputs_are_wiped_on_first_new_scheme_run(tmp_path: Path) -> None:
    """Pre-multi-shot files (no ``__`` in stem) get deleted at the start
    of ``run()`` so they don't leak into the dashboard or viewer."""
    from src.stages.hmr_world import _wipe_legacy_outputs

    out = tmp_path / "hmr_world"
    out.mkdir()
    # Legacy: no shot prefix
    (out / "P001_smpl_world.npz").write_bytes(b"legacy")
    (out / "P001_kp2d.json").write_text("{}")
    # New scheme: shot__player
    (out / "alpha__P001_smpl_world.npz").write_bytes(b"new")
    (out / "alpha__P001_kp2d.json").write_text("{}")

    removed = _wipe_legacy_outputs(out)
    assert removed == 2
    assert not (out / "P001_smpl_world.npz").exists()
    assert not (out / "P001_kp2d.json").exists()
    assert (out / "alpha__P001_smpl_world.npz").exists()
    assert (out / "alpha__P001_kp2d.json").exists()


@pytest.mark.unit
def test_is_complete_ignores_legacy_files(tmp_path: Path) -> None:
    """A directory full of legacy combined files shouldn't flip the
    stage green — the new code would rebuild them under the new
    scheme."""
    out = tmp_path / "hmr_world"
    out.mkdir()
    (out / "P001_smpl_world.npz").write_bytes(b"legacy")
    stage = HmrWorldStage(config={}, output_dir=tmp_path)
    assert stage.is_complete() is False
    (out / "alpha__P001_smpl_world.npz").write_bytes(b"new")
    assert stage.is_complete() is True


@pytest.mark.unit
def test_player_filter_isolates_one_player(tmp_path: Path) -> None:
    """``player_filter`` paired with ``shot_filter`` reduces the work
    list to one ``(shot_id, player_id)`` group so the operator can
    iterate quickly on a single player."""
    track_dir = tmp_path / "tracks"
    track_dir.mkdir()
    TracksResult(
        shot_id="alpha",
        tracks=[
            Track(
                track_id=str(tid), player_id=pid, player_name=pid,
                team="A", class_name="player",
                frames=[TrackFrame(
                    frame=0, bbox=[0, 0, 10, 10], confidence=0.9,
                    pitch_position=None,
                )],
            )
            for tid, pid in [(1, "P001"), (2, "P002")]
        ],
    ).save(track_dir / "alpha_tracks.json")

    stage = HmrWorldStage(config={}, output_dir=tmp_path)
    stage.shot_filter = "alpha"
    stage.player_filter = "P001"
    groups = stage._build_player_groups()
    # Filtering happens in run(); _build_player_groups returns the full
    # set, then run() narrows. Mirror that here for explicitness.
    filtered = {
        k: v for k, v in groups.items()
        if k[0] == stage.shot_filter and k[1] == stage.player_filter
    }
    assert list(filtered) == [("alpha", "P001")]


# ---------------------------------------------------------------------------
# Task 3 (GVHMR inference-time campaign): build_track_camera_R
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_track_camera_R_empty_map_returns_none() -> None:
    track_frames = [(i, (0, 0, 10, 10)) for i in range(5)]
    assert build_track_camera_R(track_frames, {}) is None


@pytest.mark.unit
def test_build_track_camera_R_no_matching_frames_returns_none() -> None:
    """per_frame_R has entries, but none for any frame in track_frames."""
    track_frames = [(i, (0, 0, 10, 10)) for i in range(3)]
    per_frame_R = {100: np.eye(3), 200: np.eye(3)}
    assert build_track_camera_R(track_frames, per_frame_R) is None


@pytest.mark.unit
def test_build_track_camera_R_dense_when_all_frames_present() -> None:
    track_frames = [(i, (0, 0, 10, 10)) for i in range(4)]
    per_frame_R = {i: np.eye(3) * (i + 1) for i in range(4)}
    dense = build_track_camera_R(track_frames, per_frame_R)
    assert dense is not None
    assert dense.shape == (4, 3, 3)
    for i in range(4):
        np.testing.assert_array_equal(dense[i], per_frame_R[i])


@pytest.mark.unit
def test_build_track_camera_R_head_gap_backfilled_from_next() -> None:
    """Frames 0,1 have no camera R; the nearest AVAILABLE frame is 2
    (forward) since there's no earlier data -- backfill from it (never
    average)."""
    track_frames = [(i, (0, 0, 10, 10)) for i in range(4)]
    R2 = np.eye(3) * 5.0
    R3 = np.eye(3) * 9.0
    per_frame_R = {2: R2, 3: R3}
    dense = build_track_camera_R(track_frames, per_frame_R)
    assert dense is not None
    np.testing.assert_array_equal(dense[0], R2)
    np.testing.assert_array_equal(dense[1], R2)
    np.testing.assert_array_equal(dense[2], R2)
    np.testing.assert_array_equal(dense[3], R3)


@pytest.mark.unit
def test_build_track_camera_R_tail_gap_forward_filled_from_prev() -> None:
    """Frames 4,5 have no camera R and come AFTER the last available
    frame (3) -- forward-fill from it (never average, never look ahead
    to nonexistent future data)."""
    track_frames = [(i, (0, 0, 10, 10)) for i in range(6)]
    R2 = np.eye(3) * 3.0
    R3 = np.eye(3) * 7.0
    per_frame_R = {2: R2, 3: R3}
    dense = build_track_camera_R(track_frames, per_frame_R)
    assert dense is not None
    np.testing.assert_array_equal(dense[0], R2)  # head gap -> next (R2)
    np.testing.assert_array_equal(dense[1], R2)
    np.testing.assert_array_equal(dense[2], R2)
    np.testing.assert_array_equal(dense[3], R3)
    np.testing.assert_array_equal(dense[4], R3)  # tail gap -> prev (R3)
    np.testing.assert_array_equal(dense[5], R3)


@pytest.mark.unit
def test_build_track_camera_R_middle_gap_prefers_previous() -> None:
    """A gap strictly between two available frames fills from the
    PREVIOUS frame (forward-fill), even when the next available frame
    is numerically closer -- matches the documented 'prev, else next'
    priority, not nearest-by-distance."""
    track_frames = [(i, (0, 0, 10, 10)) for i in range(6)]
    R0 = np.eye(3) * 2.0
    R5 = np.eye(3) * 11.0
    per_frame_R = {0: R0, 5: R5}  # frames 1-4 missing; frame 4 is nearer to 5
    dense = build_track_camera_R(track_frames, per_frame_R)
    assert dense is not None
    for i in range(5):
        np.testing.assert_array_equal(dense[i], R0)
    np.testing.assert_array_equal(dense[5], R5)


@pytest.mark.unit
def test_build_track_camera_R_frame_order_independent_of_track_frames_ordering() -> None:
    """Looks up per_frame_R by track_frames' frame index (not list
    position) -- an out-of-order track_frames list still resolves each
    slot to its own frame's R."""
    track_frames = [(3, (0, 0, 10, 10)), (1, (0, 0, 10, 10)), (2, (0, 0, 10, 10))]
    per_frame_R = {1: np.eye(3) * 1.0, 2: np.eye(3) * 2.0, 3: np.eye(3) * 3.0}
    dense = build_track_camera_R(track_frames, per_frame_R)
    assert dense is not None
    np.testing.assert_array_equal(dense[0], per_frame_R[3])
    np.testing.assert_array_equal(dense[1], per_frame_R[1])
    np.testing.assert_array_equal(dense[2], per_frame_R[2])


@pytest.mark.integration
def test_process_player_passes_dense_camera_r_to_run_on_track(
    tmp_path: Path, monkeypatch
) -> None:
    """Task 3 wiring: process_player builds a dense per-track-frame R
    (via build_track_camera_R) from the shot's camera track and passes
    it through run_on_track as per_frame_R, so GVHMR can skip its
    internal SimpleVO camera-rotation estimate."""
    n_frames = 10

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

    monkeypatch.setattr(
        "src.utils.gvhmr_estimator.GVHMREstimator",
        _StubGVHMREstimator,
        raising=False,
    )

    seen_per_frame_R: list = []

    def _runner(
        track_frames,
        *,
        video_path,
        checkpoint,
        device,
        extractor_device=None,
        batch_size,
        max_sequence_length,
        estimator=None,
        per_frame_K=None,
        per_frame_R=None,
    ):
        seen_per_frame_R.append(per_frame_R)
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

    assert len(seen_per_frame_R) == 1
    passed_R = seen_per_frame_R[0]
    assert passed_R is not None
    assert passed_R.shape == (n_frames, 3, 3)
    expected_R = np.array(_identity_track(1).frames[0].R)
    for i in range(n_frames):
        np.testing.assert_allclose(passed_R[i], expected_R)


@pytest.mark.integration
@pytest.mark.parametrize(
    "hmr_world_cfg, expected_extractor_device",
    [
        # Key absent from config -> defaults to "cpu" (current behaviour).
        ({}, "cpu"),
        ({"extractor_device": "mps"}, "mps"),
        ({"extractor_device": "auto"}, "auto"),
    ],
)
def test_hmr_world_forwards_extractor_device_to_both_call_sites(
    tmp_path: Path,
    monkeypatch,
    hmr_world_cfg: dict,
    expected_extractor_device: str,
) -> None:
    """The ``hmr_world.extractor_device`` config knob is read once and
    forwarded to BOTH GVHMR call sites: the ``GVHMREstimator`` constructor
    (built once per stage run) and ``run_on_track`` (called once per
    player). Absent from config, both sites see the "cpu" default.
    """
    n_frames = 10

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
                track_id="T001",
                class_name="player",
                team="A",
                player_id="P001",
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
            ),
        ],
    )
    tr.save(track_dir / "play_tracks.json")

    ctor_calls: list[dict] = []

    class _RecordingEstimator:
        def __init__(self, **kwargs) -> None:
            ctor_calls.append(kwargs)

    monkeypatch.setattr(
        "src.utils.gvhmr_estimator.GVHMREstimator",
        _RecordingEstimator,
        raising=False,
    )

    run_calls: list[dict] = []

    def _runner(
        track_frames,
        *,
        video_path,
        checkpoint,
        device,
        extractor_device=None,
        batch_size,
        max_sequence_length,
        estimator=None,
        per_frame_K=None,
        per_frame_R=None,
    ):
        run_calls.append({"device": device, "extractor_device": extractor_device})
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
                **hmr_world_cfg,
            }
        },
        output_dir=tmp_path,
    )
    stage.run()

    assert len(ctor_calls) == 1
    assert ctor_calls[0]["extractor_device"] == expected_extractor_device
    assert len(run_calls) == 1
    assert run_calls[0]["extractor_device"] == expected_extractor_device

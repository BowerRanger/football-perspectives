"""Tests for src.stages.refined_poses.RefinedPosesStage.

The refined_poses stage was rewritten to do post-HMR cleanup
(rotation-outlier rejection + leading/trailing trim) per (shot, player),
then assemble onto the shared reference timeline. The legacy cross-shot
fusion math (chordal mean / MAD outlier rejection / Savgol smoothing)
has been removed — its tests went with it. Multi-shot players now use
a simple per-frame highest-confidence pick as a placeholder until the
fusion redesign lands.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.schemas.refined_pose import RefinedPose, RefinedPoseDiagnostics
from src.schemas.smpl_world import SmplWorldTrack
from src.schemas.sync_map import Alignment, SyncMap
from src.stages.refined_poses import (
    RefinedPosesStage,
    _ANKLE_IN_ROOT,
    _beta_adjusted_rest_joints,
    _foot_world_zs,
    _ground_snap,
    _load_smpl_neutral_model,
    _reduce_root_lean,
    _reject_root_R_outliers,
    _smooth_track,
)


def _default_config() -> dict:
    # Disable ground-snap in the cross-cutting integration tests below
    # — they build tracks with identity root_R, which puts the SMPL
    # canonical feet in pitch +y (not the expected upright +z) and the
    # snap therefore drags root_t.z somewhere meaningless. Tests that
    # exercise snap explicitly use a realistic upright rotation built
    # via ``_tilted_root_R``.
    return {"refined_poses": {"ground_snap_max_distance": 0.0}}


def _heading_rotation(theta_deg: float) -> np.ndarray:
    """Build a (3, 3) root_R whose body forward (column 2) points at
    angle ``theta_deg`` measured CCW from pitch +x in the horizontal
    plane, with the body upright (up = pitch +z).
    """
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [-s,  0.0, c],
        [ c,  0.0, s],
        [0.0, 1.0, 0.0],
    ])


def _make_smpl_track(
    *,
    player_id: str,
    shot_id: str,
    n_frames: int,
    root_t_x_per_frame: float = 0.5,
    confidence: float = 1.0,
) -> SmplWorldTrack:
    frames = np.arange(n_frames, dtype=np.int64)
    return SmplWorldTrack(
        player_id=player_id,
        frames=frames,
        betas=np.zeros(10),
        thetas=np.zeros((n_frames, 24, 3)),
        root_R=np.tile(np.eye(3), (n_frames, 1, 1)),
        root_t=np.column_stack(
            [frames * root_t_x_per_frame, np.zeros(n_frames), np.zeros(n_frames)]
        ),
        confidence=np.full(n_frames, confidence),
        shot_id=shot_id,
    )


def _write_sync_map(output_dir: Path, *, ref: str, offsets: dict[str, int]) -> None:
    alignments = [
        Alignment(shot_id=sid, frame_offset=off, method="manual", confidence=1.0)
        for sid, off in offsets.items()
    ]
    sm = SyncMap(reference_shot=ref, alignments=alignments)
    (output_dir / "shots").mkdir(parents=True, exist_ok=True)
    sm.save(output_dir / "shots" / "sync_map.json")


# ── Outlier-rejection unit tests ────────────────────────────────────


@pytest.mark.unit
def test_reject_outliers_corrects_single_frame_180_flip() -> None:
    """Single-frame 180° flip in the middle of a stable run is replaced
    with the anchor-mean rotation."""
    n = 21
    stack = np.tile(_heading_rotation(0.0), (n, 1, 1))
    stack[10] = _heading_rotation(180.0)
    fixed = _reject_root_R_outliers(stack)
    assert fixed[10] == pytest.approx(_heading_rotation(0.0), abs=1e-6)
    for i in (0, 5, 9, 11, 15, 20):
        assert fixed[i] == pytest.approx(stack[i])


@pytest.mark.unit
def test_reject_outliers_corrects_four_frame_flip_run() -> None:
    """4-frame flip run (matches goalkeeper P005 frames 202–205 in the
    real data). Anchors at ±5 sit safely outside the bad region so
    each interior frame is replaced with the stable anchor mean."""
    n = 25
    stack = np.tile(_heading_rotation(-20.0), (n, 1, 1))
    for i in (10, 11, 12, 13):
        stack[i] = _heading_rotation(90.0)
    fixed = _reject_root_R_outliers(stack)
    for i in (10, 11, 12, 13):
        assert fixed[i] == pytest.approx(_heading_rotation(-20.0), abs=1e-6), (
            f"frame {i} should have been replaced"
        )


@pytest.mark.unit
def test_reject_outliers_corrects_quarter_flip() -> None:
    """A 90° "quarter flip" — not a 180° body inversion but a
    single-frame ~quarter-turn HMR glitch — also gets detected and
    replaced. The earlier Ry(180°)-only approach could only fix exact
    180° errors."""
    n = 21
    stack = np.tile(_heading_rotation(-20.0), (n, 1, 1))
    stack[10] = _heading_rotation(70.0)
    fixed = _reject_root_R_outliers(stack)
    assert fixed[10] == pytest.approx(_heading_rotation(-20.0), abs=1e-6)


@pytest.mark.unit
def test_reject_outliers_preserves_genuine_fast_turn() -> None:
    """A clean ~140° turn spread linearly across 15 frames must not be
    'corrected' back. The anchor windows on either side of any middle
    frame span much of the turn — their spread exceeds the threshold,
    so the consistency check abstains."""
    n = 15
    stack = np.stack(
        [_heading_rotation(140.0 * i / (n - 1)) for i in range(n)]
    )
    fixed = _reject_root_R_outliers(stack)
    assert fixed == pytest.approx(stack, abs=1e-6)


@pytest.mark.unit
def test_reject_outliers_leaves_stable_sequence_untouched() -> None:
    """A clean sequence with no outliers passes through unchanged."""
    stack = np.tile(_heading_rotation(45.0), (21, 1, 1))
    fixed = _reject_root_R_outliers(stack)
    assert fixed == pytest.approx(stack, abs=1e-6)


# ── Lean-reduction unit tests ───────────────────────────────────────


def _tilted_root_R(forward_xy_deg: float, lean_deg: float) -> np.ndarray:
    """Build a root_R whose body up-axis tilts by ``lean_deg`` from
    pitch +z, with the tilt happening in the direction the body is
    facing (so a player facing pitch +x leans toward +x). Used to
    seed the lean-reduction tests with a known initial state.

    The columns are (right, up, forward) per the SMPL canonical
    (y-up) → pitch (z-up) bridge used elsewhere in the codebase.
    """
    yaw = np.deg2rad(forward_xy_deg)
    lean = np.deg2rad(lean_deg)
    # Heading direction in pitch horizontal plane.
    fwd_h = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    # right = z_up × forward_h (horizontal, perpendicular to heading).
    right = np.cross(np.array([0.0, 0.0, 1.0]), fwd_h)
    right /= np.linalg.norm(right) + 1e-12
    # Tilt body_up by ``lean`` toward fwd_h (so the head leans in the
    # heading direction).
    up = (
        np.cos(lean) * np.array([0.0, 0.0, 1.0])
        + np.sin(lean) * fwd_h
    )
    # Forward stays perpendicular to up and aligned with the
    # horizontal heading: rotate the canonical (y-up) frame by the
    # same lean around ``right``.
    forward = (
        np.cos(lean) * fwd_h
        - np.sin(lean) * np.array([0.0, 0.0, 1.0])
    )
    return np.column_stack([right, up, forward])


def _ankle_world(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return t + R @ _ANKLE_IN_ROOT


@pytest.mark.unit
def test_reduce_lean_pulls_body_toward_vertical() -> None:
    """A body tilted 15° toward its heading is rotated by
    ``correction_factor`` of that lean toward pitch +z."""
    R = _tilted_root_R(forward_xy_deg=0.0, lean_deg=15.0)
    t = np.array([10.0, 5.0, 1.0])
    fixed_R, fixed_t = _reduce_root_lean(
        R[np.newaxis], t[np.newaxis],
        correction_factor=1.0, max_lean_deg=30.0,
    )
    # With factor=1.0 the body up should align with pitch +z.
    up = fixed_R[0, :, 1]
    up /= np.linalg.norm(up)
    np.testing.assert_allclose(up, [0.0, 0.0, 1.0], atol=1e-6)


@pytest.mark.unit
def test_reduce_lean_keeps_ankle_in_place() -> None:
    """The pivot of the lean correction is the ankle, not the pelvis,
    so the foot stays on the pitch after rotation."""
    R = _tilted_root_R(forward_xy_deg=30.0, lean_deg=20.0)
    t = np.array([52.5, 34.0, 0.95])
    before = _ankle_world(R, t)
    fixed_R, fixed_t = _reduce_root_lean(
        R[np.newaxis], t[np.newaxis],
        correction_factor=0.7, max_lean_deg=30.0,
    )
    after = _ankle_world(fixed_R[0], fixed_t[0])
    np.testing.assert_allclose(after, before, atol=1e-6)


@pytest.mark.unit
def test_reduce_lean_partial_correction() -> None:
    """``correction_factor=0.5`` rotates half-way toward vertical."""
    R = _tilted_root_R(forward_xy_deg=0.0, lean_deg=20.0)
    t = np.zeros(3)
    fixed_R, _ = _reduce_root_lean(
        R[np.newaxis], t[np.newaxis],
        correction_factor=0.5, max_lean_deg=30.0,
    )
    up = fixed_R[0, :, 1]
    up /= np.linalg.norm(up)
    cos_a = float(np.clip(up @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_a)))
    # Started at 20°, corrected by 0.5×20° = 10°, leaves 10° from vertical.
    assert angle_deg == pytest.approx(10.0, abs=0.05)


@pytest.mark.unit
def test_reduce_lean_leaves_steep_pose_alone() -> None:
    """Leans larger than ``max_lean_deg`` (e.g. a goalkeeper diving)
    are not corrected — the upright assumption doesn't hold."""
    R = _tilted_root_R(forward_xy_deg=45.0, lean_deg=60.0)
    t = np.array([10.0, 5.0, 0.5])
    fixed_R, fixed_t = _reduce_root_lean(
        R[np.newaxis], t[np.newaxis],
        correction_factor=0.7, max_lean_deg=30.0,
    )
    np.testing.assert_allclose(fixed_R[0], R, atol=1e-9)
    np.testing.assert_allclose(fixed_t[0], t, atol=1e-9)


@pytest.mark.unit
def test_reduce_lean_leaves_upright_body_untouched() -> None:
    """A body already perfectly upright passes through unchanged."""
    R = _tilted_root_R(forward_xy_deg=0.0, lean_deg=0.0)
    t = np.array([10.0, 5.0, 0.95])
    fixed_R, fixed_t = _reduce_root_lean(
        R[np.newaxis], t[np.newaxis],
        correction_factor=0.7, max_lean_deg=30.0,
    )
    np.testing.assert_allclose(fixed_R[0], R, atol=1e-9)
    np.testing.assert_allclose(fixed_t[0], t, atol=1e-9)


# ── Ground-snap unit tests ──────────────────────────────────────────


@pytest.mark.unit
def test_ground_snap_lifts_floating_body_to_ground() -> None:
    """A body whose lower foot sits 12 cm above the pitch is shifted
    down so the foot lands at ``target_foot_z``."""
    R = np.eye(3)
    # Pick a pelvis height that puts feet ~12 cm above the pitch with
    # all-zero thetas (canonical rest pose).
    initial_foot_z = 0.12
    # SMPL canonical l_foot y = -0.939. With identity rotation and
    # thetas = 0, foot.z in pitch = pelvis.z + (-0.939) when canonical
    # +y maps to pitch +z... but identity maps canonical+y to pitch+y
    # so feet are along pitch -y. To get an upright-body test we need
    # the canonical→pitch rotation. Use a single-frame helper from the
    # existing test rotations:
    R = _tilted_root_R(0.0, 0.0)  # body fully upright in pitch
    # Foot canonical y = -0.939; rotated to pitch, foot z = pelvis_z - 0.939.
    # Solve for pelvis_z so lowest foot = initial_foot_z.
    pelvis_z = initial_foot_z + 0.939
    t = np.array([10.0, 5.0, pelvis_z])
    thetas = np.zeros((24, 3))

    # Confirm initial foot z.
    l_z, r_z = _foot_world_zs(thetas, R, t)
    assert min(l_z, r_z) == pytest.approx(initial_foot_z, abs=1e-3)

    snapped = _ground_snap(
        R[np.newaxis], t[np.newaxis], thetas[np.newaxis],
        target_foot_z=0.02, max_snap_distance=0.30,
    )
    l_z, r_z = _foot_world_zs(thetas, R, snapped[0])
    assert min(l_z, r_z) == pytest.approx(0.02, abs=1e-3)


@pytest.mark.unit
def test_ground_snap_skips_airborne_player() -> None:
    """When the lowest foot is more than ``max_snap_distance`` above
    the pitch (jump, header), don't yank the body down — leave the
    motion alone."""
    R = _tilted_root_R(0.0, 0.0)
    # Both feet at z = 0.8 (clearly airborne).
    pelvis_z = 0.8 + 0.939
    t = np.array([10.0, 5.0, pelvis_z])
    thetas = np.zeros((24, 3))
    snapped = _ground_snap(
        R[np.newaxis], t[np.newaxis], thetas[np.newaxis],
        target_foot_z=0.02, max_snap_distance=0.30,
    )
    np.testing.assert_allclose(snapped[0], t)


@pytest.mark.unit
def test_ground_snap_pulls_buried_body_up() -> None:
    """A body whose lower foot is below the pitch (penetrating) gets
    lifted so the foot lands at ``target_foot_z``."""
    R = _tilted_root_R(0.0, 0.0)
    initial_foot_z = -0.05  # buried 5 cm
    pelvis_z = initial_foot_z + 0.939
    t = np.array([10.0, 5.0, pelvis_z])
    thetas = np.zeros((24, 3))
    snapped = _ground_snap(
        R[np.newaxis], t[np.newaxis], thetas[np.newaxis],
        target_foot_z=0.02, max_snap_distance=0.30,
    )
    l_z, r_z = _foot_world_zs(thetas, R, snapped[0])
    assert min(l_z, r_z) == pytest.approx(0.02, abs=1e-3)
    # Body went UP by the delta.
    assert snapped[0, 2] > t[2]


@pytest.mark.unit
def test_ground_snap_keeps_grounded_body_in_place() -> None:
    """A body whose lower foot already sits at ``target_foot_z``
    passes through unchanged (the snap delta is 0)."""
    R = _tilted_root_R(0.0, 0.0)
    target_z = 0.02
    pelvis_z = target_z + 0.939
    t = np.array([10.0, 5.0, pelvis_z])
    thetas = np.zeros((24, 3))
    snapped = _ground_snap(
        R[np.newaxis], t[np.newaxis], thetas[np.newaxis],
        target_foot_z=target_z, max_snap_distance=0.30,
    )
    np.testing.assert_allclose(snapped[0], t, atol=1e-6)


# ── Smoothing unit tests ────────────────────────────────────────────


def _track_with_noisy_translation(
    *, n: int, noise_scale: float, seed: int = 0,
) -> tuple[SmplWorldTrack, np.ndarray]:
    """Build a track whose root_t is a clean straight line plus
    Gaussian noise. Returns (track, truth_root_t)."""
    rng = np.random.default_rng(seed)
    frames = np.arange(n, dtype=np.int64)
    truth = np.column_stack([frames * 0.1, np.zeros(n), np.zeros(n)])
    noisy = truth + rng.normal(scale=noise_scale, size=truth.shape)
    track = SmplWorldTrack(
        player_id="P001",
        frames=frames,
        betas=np.zeros(10),
        thetas=np.zeros((n, 24, 3)),
        root_R=np.tile(np.eye(3), (n, 1, 1)),
        root_t=noisy,
        confidence=np.ones(n),
        shot_id="play",
    )
    return track, truth


@pytest.mark.unit
def test_smooth_root_t_reduces_translation_noise() -> None:
    """Savgol on root_t shrinks RMS error vs. ground truth."""
    track, truth = _track_with_noisy_translation(
        n=30, noise_scale=0.05, seed=0,
    )
    raw_rms = float(np.sqrt(np.mean((track.root_t - truth) ** 2)))
    smoothed = _smooth_track(
        track,
        root_R_slerp_window=1,
        root_t_savgol_window=7,
        root_t_savgol_order=2,
        thetas_savgol_window=1,
    )
    smooth_rms = float(np.sqrt(np.mean((smoothed.root_t - truth) ** 2)))
    assert smooth_rms < raw_rms, (
        f"smoothing should reduce RMS error; raw={raw_rms:.4f}, "
        f"smoothed={smooth_rms:.4f}"
    )


@pytest.mark.unit
def test_smooth_thetas_reduces_pose_noise() -> None:
    """Savgol on thetas shrinks RMS error on a noisy per-joint signal."""
    rng = np.random.default_rng(1)
    n = 30
    truth = np.zeros((n, 24, 3))
    noisy = truth + rng.normal(scale=0.05, size=truth.shape)
    track = SmplWorldTrack(
        player_id="P001",
        frames=np.arange(n, dtype=np.int64),
        betas=np.zeros(10),
        thetas=noisy,
        root_R=np.tile(np.eye(3), (n, 1, 1)),
        root_t=np.zeros((n, 3)),
        confidence=np.ones(n),
        shot_id="play",
    )
    raw_rms = float(np.sqrt(np.mean((track.thetas - truth) ** 2)))
    smoothed = _smooth_track(
        track,
        root_R_slerp_window=1,
        root_t_savgol_window=1,
        thetas_savgol_window=9,
        thetas_savgol_order=2,
    )
    smooth_rms = float(np.sqrt(np.mean((smoothed.thetas - truth) ** 2)))
    assert smooth_rms < raw_rms


@pytest.mark.unit
def test_smooth_windows_set_to_one_are_no_ops() -> None:
    """``window <= 1`` disables that specific smoother — useful for
    pipelines that want only one signal touched."""
    track, _ = _track_with_noisy_translation(
        n=20, noise_scale=0.1, seed=2,
    )
    smoothed = _smooth_track(
        track,
        root_R_slerp_window=1,
        root_t_savgol_window=1,
        thetas_savgol_window=1,
    )
    np.testing.assert_allclose(smoothed.root_t, track.root_t)
    np.testing.assert_allclose(smoothed.thetas, track.thetas)
    np.testing.assert_allclose(smoothed.root_R, track.root_R)


@pytest.mark.unit
def test_smooth_track_preserves_frames_and_metadata() -> None:
    """Smoothing must not alter frame indices, player_id, shot_id, or
    confidence — only the continuous signals (root_R / root_t / thetas)
    change."""
    track, _ = _track_with_noisy_translation(
        n=20, noise_scale=0.05, seed=3,
    )
    smoothed = _smooth_track(track)
    np.testing.assert_array_equal(smoothed.frames, track.frames)
    np.testing.assert_array_equal(smoothed.confidence, track.confidence)
    assert smoothed.player_id == track.player_id
    assert smoothed.shot_id == track.shot_id


# ── Stage-level integration tests ───────────────────────────────────


@pytest.mark.integration
def test_refined_poses_single_shot_passthrough(tmp_path: Path) -> None:
    """Single-shot player with all-anchored frames: refined output
    mirrors the hmr_world input after rotation outlier rejection
    (which is a no-op on a clean stack of identity rotations)."""
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="play", offsets={"play": 0})
    track = _make_smpl_track(player_id="P001", shot_id="play", n_frames=10)
    track.save(output_dir / "hmr_world" / "play__P001_smpl_world.npz")

    stage = RefinedPosesStage(config=_default_config(), output_dir=output_dir)
    assert stage.is_complete() is False
    stage.run()
    assert stage.is_complete() is True

    refined = RefinedPose.load(output_dir / "refined_poses" / "P001_refined.npz")
    assert refined.player_id == "P001"
    assert refined.contributing_shots == ("play",)
    np.testing.assert_array_equal(refined.frames, track.frames)
    # Savgol smoothing on already-linear root_t introduces sub-µm
    # float32 quantization — the data is functionally unchanged but
    # not bit-identical.
    np.testing.assert_allclose(refined.root_t, track.root_t, atol=1e-5)
    assert refined.view_count.tolist() == [1] * 10


@pytest.mark.integration
def test_refined_poses_is_complete_after_run(tmp_path: Path) -> None:
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="play", offsets={"play": 0})
    _make_smpl_track(player_id="P001", shot_id="play", n_frames=5).save(
        output_dir / "hmr_world" / "play__P001_smpl_world.npz"
    )

    stage = RefinedPosesStage(config=_default_config(), output_dir=output_dir)
    stage.run()
    assert stage.is_complete() is True

    (output_dir / "refined_poses" / "P001_refined.npz").unlink()
    assert stage.is_complete() is False


@pytest.mark.integration
def test_refined_poses_trims_leading_and_trailing_unanchored_frames(
    tmp_path: Path,
) -> None:
    """Frames whose confidence is below the fresh-anchor cutoff at the
    start/end of a track are dropped from the refined output, removing
    the "ghost" interpolating from (0, 0) on the leading frames."""
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="play", offsets={"play": 0})

    n_frames = 20
    leading = 4
    trailing = 3
    anchored_n = n_frames - leading - trailing
    confidence = np.array(
        [0.1] * leading
        + [0.9] * anchored_n
        + [0.1] * trailing,
        dtype=np.float32,
    )
    frames = np.arange(n_frames, dtype=np.int64)
    track = SmplWorldTrack(
        player_id="P001",
        frames=frames,
        betas=np.zeros(10),
        thetas=np.zeros((n_frames, 24, 3)),
        root_R=np.tile(np.eye(3), (n_frames, 1, 1)),
        root_t=np.column_stack(
            [frames * 0.5, np.zeros(n_frames), np.zeros(n_frames)]
        ),
        confidence=confidence,
        shot_id="play",
    )
    track.save(output_dir / "hmr_world" / "play__P001_smpl_world.npz")

    stage = RefinedPosesStage(config=_default_config(), output_dir=output_dir)
    stage.run()

    refined = RefinedPose.load(output_dir / "refined_poses" / "P001_refined.npz")
    assert len(refined.frames) == anchored_n
    assert int(refined.frames[0]) == leading
    assert int(refined.frames[-1]) == n_frames - trailing - 1
    assert refined.thetas.shape == (anchored_n, 24, 3)
    assert refined.root_R.shape == (anchored_n, 3, 3)
    assert refined.root_t.shape == (anchored_n, 3)


@pytest.mark.integration
def test_refined_poses_outlier_rotation_replaced_in_output(
    tmp_path: Path,
) -> None:
    """A single-frame flip in a long stable sequence gets replaced by
    the time it reaches the refined_poses output file."""
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="play", offsets={"play": 0})

    n = 25
    flip_idx = 12
    root_R = np.tile(_heading_rotation(0.0), (n, 1, 1))
    root_R[flip_idx] = _heading_rotation(180.0)
    track = SmplWorldTrack(
        player_id="P001",
        frames=np.arange(n, dtype=np.int64),
        betas=np.zeros(10),
        thetas=np.zeros((n, 24, 3)),
        root_R=root_R,
        root_t=np.zeros((n, 3)) + np.array([1.0, 0.0, 0.0]),
        confidence=np.full(n, 0.9, dtype=np.float32),
        shot_id="play",
    )
    track.save(output_dir / "hmr_world" / "play__P001_smpl_world.npz")

    RefinedPosesStage(config=_default_config(), output_dir=output_dir).run()
    refined = RefinedPose.load(output_dir / "refined_poses" / "P001_refined.npz")
    # Flip frame's heading restored to the surrounding 0° direction.
    assert refined.root_R[flip_idx] == pytest.approx(
        _heading_rotation(0.0), abs=1e-5
    )


@pytest.mark.integration
def test_refined_poses_multi_shot_highest_confidence_wins(
    tmp_path: Path,
) -> None:
    """Two shots cover the same reference frames at different
    confidences. The placeholder fusion picks the higher-confidence
    sample per frame; view_count is 1 (no real averaging) and both
    shots are listed as contributors."""
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="A", offsets={"A": 0, "B": 0})

    n = 5
    frames = np.arange(n, dtype=np.int64)
    a = SmplWorldTrack(
        player_id="P001",
        frames=frames,
        betas=np.zeros(10),
        thetas=np.zeros((n, 24, 3)),
        root_R=np.tile(np.eye(3), (n, 1, 1)),
        root_t=np.column_stack(
            [frames * 1.0, np.zeros(n), np.zeros(n)]
        ),
        confidence=np.full(n, 0.4, dtype=np.float32),
        shot_id="A",
    )
    b = SmplWorldTrack(
        player_id="P001",
        frames=frames,
        betas=np.zeros(10),
        thetas=np.zeros((n, 24, 3)),
        root_R=np.tile(np.eye(3), (n, 1, 1)),
        root_t=np.column_stack(
            [frames * 1.0 + 0.5, np.zeros(n), np.zeros(n)]
        ),
        confidence=np.full(n, 0.8, dtype=np.float32),
        shot_id="B",
    )
    a.save(output_dir / "hmr_world" / "A__P001_smpl_world.npz")
    b.save(output_dir / "hmr_world" / "B__P001_smpl_world.npz")

    RefinedPosesStage(config=_default_config(), output_dir=output_dir).run()
    refined = RefinedPose.load(output_dir / "refined_poses" / "P001_refined.npz")
    assert set(refined.contributing_shots) == {"A", "B"}
    # B wins every frame (higher conf) so root_t[:, 0] == ref_frame + 0.5.
    np.testing.assert_allclose(
        refined.root_t[:, 0], frames + 0.5, atol=1e-6
    )
    assert refined.view_count.tolist() == [1] * n


@pytest.mark.integration
def test_refined_poses_summary_counts_players(tmp_path: Path) -> None:
    """The per-stage summary records how many players were refined and
    splits single- vs multi-shot for downstream quality_report."""
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="A", offsets={"A": 0, "B": 0})

    _make_smpl_track(player_id="P001", shot_id="A", n_frames=5).save(
        output_dir / "hmr_world" / "A__P001_smpl_world.npz"
    )
    _make_smpl_track(player_id="P002", shot_id="A", n_frames=5).save(
        output_dir / "hmr_world" / "A__P002_smpl_world.npz"
    )
    _make_smpl_track(player_id="P002", shot_id="B", n_frames=5).save(
        output_dir / "hmr_world" / "B__P002_smpl_world.npz"
    )

    RefinedPosesStage(config=_default_config(), output_dir=output_dir).run()
    summary = json.loads(
        (output_dir / "refined_poses" / "refined_poses_summary.json").read_text()
    )
    assert summary["players_refined"] == 2
    assert summary["single_shot_players"] == 1
    assert summary["multi_shot_players"] == 1


@pytest.mark.integration
def test_refined_poses_smoothing_reaches_saved_track(tmp_path: Path) -> None:
    """Stage-level wiring: a noisy translation passed through the stage
    comes back smoothed in the saved RefinedPose, with RMS error vs.
    truth strictly below the raw input."""
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="play", offsets={"play": 0})

    rng = np.random.default_rng(7)
    n = 30
    frames = np.arange(n, dtype=np.int64)
    truth = np.column_stack([frames * 0.1, np.zeros(n), np.zeros(n)])
    noisy = truth + rng.normal(scale=0.05, size=truth.shape)
    SmplWorldTrack(
        player_id="P001",
        frames=frames,
        betas=np.zeros(10),
        thetas=np.zeros((n, 24, 3)),
        root_R=np.tile(np.eye(3), (n, 1, 1)),
        root_t=noisy,
        confidence=np.ones(n, dtype=np.float32),
        shot_id="play",
    ).save(output_dir / "hmr_world" / "play__P001_smpl_world.npz")

    RefinedPosesStage(config=_default_config(), output_dir=output_dir).run()
    refined = RefinedPose.load(output_dir / "refined_poses" / "P001_refined.npz")

    raw_rms = float(np.sqrt(np.mean((noisy - truth) ** 2)))
    smooth_rms = float(np.sqrt(np.mean((refined.root_t - truth) ** 2)))
    assert smooth_rms < raw_rms


@pytest.mark.integration
def test_refined_poses_ground_snap_reaches_saved_track(tmp_path: Path) -> None:
    """Stage-level wiring: a floating body comes back with the lower
    foot landing at ``ground_snap_target_z`` after refined_poses runs.

    Verification FK uses the same beta-adjusted rest joints the stage
    applied internally — so the snap target is checked against the
    player's actual leg geometry, not mean-betas canonical."""
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="play", offsets={"play": 0})

    R = _tilted_root_R(0.0, 0.0)  # upright body
    n = 10
    initial_foot_z = 0.15  # 15 cm gap to the pitch
    pelvis_z = initial_foot_z + 0.939
    frames = np.arange(n, dtype=np.int64)
    betas = np.zeros(10, dtype=np.float32)
    SmplWorldTrack(
        player_id="P001",
        frames=frames,
        betas=betas,
        thetas=np.zeros((n, 24, 3)),
        root_R=np.tile(R, (n, 1, 1)),
        root_t=np.tile([10.0, 5.0, pelvis_z], (n, 1)),
        confidence=np.ones(n, dtype=np.float32),
        shot_id="play",
    ).save(output_dir / "hmr_world" / "play__P001_smpl_world.npz")

    # Use the stage's default ground-snap settings (target_z = 0.02 m,
    # max_snap_distance = 0.30 m).
    RefinedPosesStage(
        config={"refined_poses": {}}, output_dir=output_dir,
    ).run()
    refined = RefinedPose.load(output_dir / "refined_poses" / "P001_refined.npz")

    # Use the same beta-adjusted rest joints the stage applied so the
    # verification FK matches the actual leg geometry. (Without this,
    # the saved track is snapped to the model's foot but the check
    # would read against mean-betas src canonical, which differs by
    # 1-2 cm at the foot.)
    smpl_model = _load_smpl_neutral_model()
    rest = _beta_adjusted_rest_joints(betas, smpl_model)
    l_z, r_z = _foot_world_zs(
        refined.thetas[0], refined.root_R[0], refined.root_t[0],
        rest_joints=rest,
    )
    assert min(l_z, r_z) == pytest.approx(0.02, abs=2e-3)


@pytest.mark.integration
def test_refined_poses_diagnostics_round_trip(tmp_path: Path) -> None:
    """Diagnostics still round-trip through the schema (consumed by
    /refined_poses/diagnostics) — the new stage emits minimal
    contents (no per-frame entries) but the file must exist and load."""
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="A", offsets={"A": 0})
    _make_smpl_track(player_id="P001", shot_id="A", n_frames=5).save(
        output_dir / "hmr_world" / "A__P001_smpl_world.npz"
    )

    RefinedPosesStage(config=_default_config(), output_dir=output_dir).run()
    diag = RefinedPoseDiagnostics.load(
        output_dir / "refined_poses" / "P001_diagnostics.json"
    )
    assert diag.player_id == "P001"
    assert diag.contributing_shots == ("A",)
    assert diag.summary["total_frames"] == 5

"""Tests for the cross-player camera-jitter correction in refined_poses.

The detector observes simultaneous frame-to-frame shifts in pitch XY
across all visible players and treats the median as a camera-tracking
artifact (since real player motion is uncorrelated). It builds a
cumulative offset signal and subtracts it from every player's root_t
inside the stage, before per-player temporal smoothing runs.

Tests cover:
  - Detection + correction of a transient camera shift (round-trip).
  - The adaptive tempo gate: fast scenes need a bigger spike to fire.
  - The agreement gate: one player moving alone doesn't trigger.
  - The visibility gate: too few players in a frame skips it.
  - Confidence weighting: low-confidence frames don't contribute.
  - Stage-level wiring: corrections reach the saved RefinedPose and
    the summary records what was done.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.schemas.refined_pose import RefinedPose
from src.schemas.smpl_world import SmplWorldTrack
from src.schemas.sync_map import Alignment, GroupSync, SyncMap
from src.stages.refined_poses import (
    RefinedPosesStage,
    _apply_jitter_correction,
    _apply_residual_consensus_correction,
)


def _make_track(
    *,
    player_id: str,
    shot_id: str,
    root_t: np.ndarray,
    confidence: np.ndarray | None = None,
) -> SmplWorldTrack:
    n = root_t.shape[0]
    return SmplWorldTrack(
        player_id=player_id,
        frames=np.arange(n, dtype=np.int64),
        betas=np.zeros(10, dtype=np.float32),
        thetas=np.zeros((n, 24, 3), dtype=np.float32),
        root_R=np.tile(np.eye(3), (n, 1, 1)).astype(np.float32),
        root_t=root_t.astype(np.float32),
        confidence=(
            confidence.astype(np.float32) if confidence is not None
            else np.full(n, 1.0, dtype=np.float32)
        ),
        shot_id=shot_id,
    )


def _static_track(
    pid: str, shot_id: str, pos_xy: tuple[float, float], n: int,
) -> SmplWorldTrack:
    return _make_track(
        player_id=pid,
        shot_id=shot_id,
        root_t=np.tile([pos_xy[0], pos_xy[1], 0.95], (n, 1)),
    )


def _write_sync_map(output_dir: Path, *, ref: str, offsets: dict[str, int]) -> None:
    alignments = [
        Alignment(shot_id=sid, frame_offset=off, method="manual", confidence=1.0)
        for sid, off in offsets.items()
    ]
    sm = SyncMap(groups=[GroupSync(
        group_id="", reference_shot=ref, alignments=alignments,
    )])
    (output_dir / "shots").mkdir(parents=True, exist_ok=True)
    sm.save(output_dir / "shots" / "sync_map.json")


# ── Helper-level unit tests ─────────────────────────────────────────


@pytest.mark.unit
def test_jitter_correction_cancels_transient_camera_shift() -> None:
    """A one-frame +0.5m pitch-x spike applied to every player is
    detected and cancelled — the cumulative offset jumps up at the
    spike frame and back down at the snap-back frame, so all players'
    root_t lands back at their original positions."""
    n = 30
    jitter_frame = 15
    static_positions = [(10.0, 20.0), (30.0, 40.0), (50.0, 5.0), (70.0, 60.0)]
    tracks = []
    for pid, pos in zip(("P001", "P002", "P003", "P004"), static_positions):
        rt = np.tile([pos[0], pos[1], 0.95], (n, 1))
        rt[jitter_frame, 0] += 0.5  # one-frame shift, all players
        tracks.append(_make_track(player_id=pid, shot_id="play", root_t=rt))

    corrected, stats = _apply_jitter_correction(
        tracks,
        fps=25.0,
        enabled=True,
        min_players=4,
        baseline_window_s=1.0,
        baseline_floor_m=0.05,
        tempo_ratio=2.5,
        max_spread_ratio=0.30,
    )

    # Spike frame: every player back to within 1cm of ground truth.
    for orig_pos, fixed in zip(static_positions, corrected):
        np.testing.assert_allclose(
            fixed.root_t[jitter_frame, :2],
            np.array(orig_pos),
            atol=0.01,
        )
    # Other frames unchanged.
    for orig_pos, fixed in zip(static_positions, corrected):
        np.testing.assert_allclose(
            fixed.root_t[0, :2], np.array(orig_pos), atol=1e-6,
        )
        np.testing.assert_allclose(
            fixed.root_t[-1, :2], np.array(orig_pos), atol=1e-6,
        )
    # Z (height) is untouched by the XY correction.
    for fixed in corrected:
        np.testing.assert_allclose(fixed.root_t[:, 2], 0.95, atol=1e-6)
    assert stats["corrected_frames"] >= 1
    assert stats["max_offset_m"] >= 0.4


@pytest.mark.unit
def test_jitter_correction_cancels_sustained_drift() -> None:
    """A sustained drift (camera off for 5 frames, then snaps back)
    is corrected for the full duration — the cumulative offset stays
    elevated until the snap-back frame."""
    n = 30
    drift_start, drift_end = 12, 17  # frames 12..16 (5 frames) are off
    tracks = []
    static_positions = [(10.0, 20.0), (30.0, 40.0), (50.0, 5.0), (70.0, 60.0)]
    for pid, pos in zip(("P001", "P002", "P003", "P004"), static_positions):
        rt = np.tile([pos[0], pos[1], 0.95], (n, 1))
        rt[drift_start:drift_end, 0] += 0.5
        tracks.append(_make_track(player_id=pid, shot_id="play", root_t=rt))

    corrected, _ = _apply_jitter_correction(
        tracks,
        fps=25.0,
        enabled=True,
        min_players=4,
        baseline_window_s=1.0,
        baseline_floor_m=0.05,
        tempo_ratio=2.5,
        max_spread_ratio=0.30,
    )

    for orig_pos, fixed in zip(static_positions, corrected):
        for f in range(drift_start, drift_end):
            np.testing.assert_allclose(
                fixed.root_t[f, :2],
                np.array(orig_pos),
                atol=0.01,
                err_msg=f"frame {f} should be back on ground truth",
            )


@pytest.mark.unit
def test_jitter_correction_disabled_is_passthrough() -> None:
    """``enabled=False`` short-circuits the pass; tracks come back
    bit-identical and stats report zero work."""
    n = 20
    tracks = [
        _make_track(
            player_id=f"P{i:03d}",
            shot_id="play",
            root_t=np.tile([i * 10.0, 5.0, 0.95], (n, 1)),
        )
        for i in range(1, 6)
    ]
    corrected, stats = _apply_jitter_correction(
        tracks, fps=25.0, enabled=False,
    )
    for orig, fixed in zip(tracks, corrected):
        np.testing.assert_array_equal(fixed.root_t, orig.root_t)
    assert stats["corrected_frames"] == 0


@pytest.mark.unit
def test_jitter_correction_skips_when_one_runner_among_static_players() -> None:
    """Three static players and one runner. The runner alone moves,
    so the median Δ across players ≈ 0 (the runner is the minority).
    Magnitude gate fails → no correction. The runner's motion is
    preserved."""
    n = 30
    static = [(10.0, 20.0), (30.0, 40.0), (50.0, 5.0)]
    tracks = []
    for pid, pos in zip(("P001", "P002", "P003"), static):
        tracks.append(_static_track(pid, "play", pos, n))
    # Runner: +0.3 m/frame in x.
    runner_root_t = np.column_stack([
        70.0 + np.arange(n) * 0.3,
        np.full(n, 60.0),
        np.full(n, 0.95),
    ])
    tracks.append(_make_track(
        player_id="P004", shot_id="play", root_t=runner_root_t,
    ))

    corrected, stats = _apply_jitter_correction(
        tracks,
        fps=25.0,
        enabled=True,
        min_players=4,
        baseline_window_s=1.0,
        baseline_floor_m=0.05,
        tempo_ratio=2.5,
        max_spread_ratio=0.30,
    )

    # Runner unchanged (no offset applied).
    np.testing.assert_allclose(
        corrected[3].root_t, runner_root_t, atol=1e-6,
    )
    # Static players unchanged.
    for orig, fixed in zip(tracks[:3], corrected[:3]):
        np.testing.assert_allclose(fixed.root_t, orig.root_t, atol=1e-6)
    assert stats["corrected_frames"] == 0


@pytest.mark.unit
def test_jitter_correction_visibility_gate_skips_few_players() -> None:
    """Only 3 visible players with min_players=4: the magnitude is
    big and the players agree, but the visibility gate blocks the
    correction so we don't act on flimsy evidence."""
    n = 20
    tracks = []
    static = [(10.0, 20.0), (30.0, 40.0), (50.0, 5.0)]
    for pid, pos in zip(("P001", "P002", "P003"), static):
        rt = np.tile([pos[0], pos[1], 0.95], (n, 1))
        rt[10, 0] += 0.5  # spike on all 3
        tracks.append(_make_track(player_id=pid, shot_id="play", root_t=rt))

    corrected, stats = _apply_jitter_correction(
        tracks,
        fps=25.0,
        enabled=True,
        min_players=4,
        baseline_window_s=1.0,
        baseline_floor_m=0.05,
        tempo_ratio=2.5,
        max_spread_ratio=0.30,
    )

    for orig, fixed in zip(tracks, corrected):
        np.testing.assert_allclose(fixed.root_t, orig.root_t, atol=1e-6)
    assert stats["corrected_frames"] == 0


@pytest.mark.unit
def test_jitter_correction_adaptive_gate_ignores_steady_pan() -> None:
    """All 4 players moving in unison at 0.2 m/frame is a coordinated
    scene pan (or a goal-side migration), not jitter. The rolling
    baseline ≈ 0.2 m so |Δ| sits AT the baseline, not tempo_ratio× it,
    so the gate skips every frame."""
    n = 50
    velocity_x = 0.2
    tracks = []
    starts = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
    for pid, start in zip(("P001", "P002", "P003", "P004"), starts):
        rt = np.column_stack([
            start[0] + np.arange(n) * velocity_x,
            np.full(n, start[1]),
            np.full(n, 0.95),
        ])
        tracks.append(_make_track(player_id=pid, shot_id="play", root_t=rt))

    corrected, stats = _apply_jitter_correction(
        tracks,
        fps=25.0,
        enabled=True,
        min_players=4,
        baseline_window_s=1.0,
        baseline_floor_m=0.05,
        tempo_ratio=2.5,
        max_spread_ratio=0.30,
    )

    for orig, fixed in zip(tracks, corrected):
        np.testing.assert_allclose(
            fixed.root_t, orig.root_t, atol=1e-5,
        )
    assert stats["corrected_frames"] == 0


@pytest.mark.unit
def test_jitter_correction_adaptive_gate_fires_in_static_scene() -> None:
    """Players are essentially static (baseline ≈ floor). A 0.4m
    spike clears tempo_ratio × baseline_floor easily, so the gate
    fires and the correction is applied."""
    n = 30
    tracks = []
    static = [(10.0, 20.0), (30.0, 40.0), (50.0, 5.0), (70.0, 60.0)]
    for pid, pos in zip(("P001", "P002", "P003", "P004"), static):
        rt = np.tile([pos[0], pos[1], 0.95], (n, 1))
        rt[15, 0] += 0.4
        tracks.append(_make_track(player_id=pid, shot_id="play", root_t=rt))

    corrected, stats = _apply_jitter_correction(
        tracks,
        fps=25.0,
        enabled=True,
        min_players=4,
        baseline_window_s=1.0,
        baseline_floor_m=0.05,
        tempo_ratio=2.5,
        max_spread_ratio=0.30,
    )

    for orig_pos, fixed in zip(static, corrected):
        np.testing.assert_allclose(
            fixed.root_t[15, :2], np.array(orig_pos), atol=0.01,
        )
    assert stats["corrected_frames"] >= 1


@pytest.mark.unit
def test_jitter_correction_ignores_low_confidence_frames() -> None:
    """Players whose confidence at a frame is below the fresh-anchor
    cutoff (0.3) don't contribute to that frame's consensus. We use
    this to ensure a player whose ankle was occluded for a few
    frames doesn't drag the median around."""
    n = 30
    # Four confident static players who see the jitter,
    # plus one low-conf player whose root_t is garbage but should
    # be ignored.
    tracks = []
    static = [(10.0, 20.0), (30.0, 40.0), (50.0, 5.0), (70.0, 60.0)]
    for pid, pos in zip(("P001", "P002", "P003", "P004"), static):
        rt = np.tile([pos[0], pos[1], 0.95], (n, 1))
        rt[15, 0] += 0.5
        tracks.append(_make_track(player_id=pid, shot_id="play", root_t=rt))

    # Garbage track at conf 0.1.
    bad_rt = np.zeros((n, 3))
    bad_rt[:, 0] = np.random.default_rng(0).normal(scale=10.0, size=n)
    tracks.append(_make_track(
        player_id="P005",
        shot_id="play",
        root_t=bad_rt,
        confidence=np.full(n, 0.1, dtype=np.float32),
    ))

    corrected, stats = _apply_jitter_correction(
        tracks,
        fps=25.0,
        enabled=True,
        min_players=4,
        baseline_window_s=1.0,
        baseline_floor_m=0.05,
        tempo_ratio=2.5,
        max_spread_ratio=0.30,
    )

    for orig_pos, fixed in zip(static, corrected[:4]):
        np.testing.assert_allclose(
            fixed.root_t[15, :2], np.array(orig_pos), atol=0.01,
        )
    assert stats["corrected_frames"] >= 1


# ── Residual-consensus correction (catches sustained wobble that the
#    Δ-based pass can't see, because individual player motion drowns
#    out the consensus in frame-to-frame deltas). ────────────────────


def _running_player(
    pid: str,
    shot_id: str,
    start_xy: tuple[float, float],
    velocity_xy: tuple[float, float],
    n: int,
) -> SmplWorldTrack:
    rt = np.column_stack([
        start_xy[0] + np.arange(n) * velocity_xy[0],
        start_xy[1] + np.arange(n) * velocity_xy[1],
        np.full(n, 0.95),
    ])
    return _make_track(player_id=pid, shot_id=shot_id, root_t=rt)


@pytest.mark.unit
def test_residual_consensus_cancels_sinusoidal_wobble_with_running_players() -> None:
    """The real-world failure case: six players running in different
    directions at different speeds, with a ~3 Hz sinusoidal common-mode
    XY wobble overlaid on every player's root_t (camera-tracking
    artifact). The Δ-based pass cannot see this because per-player
    Δ noise dwarfs the wobble's per-frame Δ.

    The residual-consensus pass smooths each player's track to recover
    their true trajectory, takes the cross-player median residual as
    the common-mode shift, and subtracts it. After correction every
    player should be within ~2 cm of their wobble-free trajectory at
    every frame.
    """
    fps = 25.0
    n = 120  # ~4.8 s, long enough for a Savgol baseline to lock in
    rng = np.random.default_rng(42)

    # Six running players, different velocities (m/frame at 25 fps)
    velocities = [
        (0.20, 0.00), (-0.18, 0.05), (0.10, 0.15),
        (0.05, -0.20), (-0.12, -0.08), (0.00, 0.18),
    ]
    starts = [
        (10.0, 30.0), (60.0, 25.0), (40.0, 45.0),
        (20.0, 10.0), (75.0, 50.0), (35.0, 60.0),
    ]
    truth = [
        np.column_stack([
            s[0] + np.arange(n) * v[0],
            s[1] + np.arange(n) * v[1],
            np.full(n, 0.95),
        ])
        for s, v in zip(starts, velocities)
    ]

    # Sinusoidal wobble: 3 Hz, 0.10 m amplitude on both axes, π/2 apart
    t = np.arange(n) / fps
    wobble_x = 0.10 * np.sin(2 * np.pi * 3.0 * t)
    wobble_y = 0.10 * np.cos(2 * np.pi * 3.0 * t)
    wobble = np.column_stack([wobble_x, wobble_y, np.zeros(n)])

    tracks = []
    for pid_i, (truth_i, _) in enumerate(zip(truth, velocities)):
        rt = truth_i + wobble
        # Sprinkle a small amount of per-player HMR noise (~1 cm)
        rt[:, :2] += rng.normal(scale=0.01, size=(n, 2))
        tracks.append(_make_track(
            player_id=f"P{pid_i+1:03d}",
            shot_id="play",
            root_t=rt,
        ))

    corrected, stats = _apply_residual_consensus_correction(
        tracks,
        fps=fps,
        enabled=True,
        savgol_window_s=1.2,
        savgol_poly=2,
        min_players=3,
        min_offset_m=0.0,
    )

    # Skip the savgol edge taps when measuring success — Savgol's first
    # and last half-windows are interpolated and carry larger residuals
    # at the ends. The middle of the clip is what the user sees as the
    # wobble, and that is what we must remove.
    half_w = int(round(1.2 * fps / 2.0))
    interior = slice(half_w, n - half_w)
    for orig_truth, fixed in zip(truth, corrected):
        delta = np.linalg.norm(
            fixed.root_t[interior, :2] - orig_truth[interior, :2], axis=1,
        )
        # Wobble amplitude was 0.10 m; after correction, residual error
        # should be well under 1/3 of the wobble amplitude on the
        # interior frames. The HMR noise floor is 0.01 m so we don't
        # demand perfection.
        assert float(np.max(delta)) < 0.035, (
            f"player still off truth by {float(np.max(delta)):.3f} m "
            f"in the interior — wobble was not adequately removed"
        )

    assert stats["corrected_frames"] >= n - 2 * half_w - 2
    assert stats["max_offset_m"] >= 0.05  # we removed at least 5 cm somewhere
    # Sanity: the raw wobble amplitude before correction was 0.10 m.
    # After correction we should have removed most of it.
    assert stats["max_offset_m"] <= 0.20


@pytest.mark.unit
def test_residual_consensus_disabled_is_passthrough() -> None:
    """``enabled=False`` short-circuits the residual pass; tracks come
    back bit-identical."""
    n = 60
    tracks = [
        _running_player(f"P{i:03d}", "play", (i*10.0, 5.0), (0.1, 0.0), n)
        for i in range(1, 5)
    ]
    corrected, stats = _apply_residual_consensus_correction(
        tracks, fps=25.0, enabled=False,
    )
    for orig, fixed in zip(tracks, corrected):
        np.testing.assert_array_equal(fixed.root_t, orig.root_t)
    assert stats["corrected_frames"] == 0


@pytest.mark.unit
def test_residual_consensus_preserves_individual_motion() -> None:
    """No common-mode wobble — just running players. The residual pass
    should be effectively a no-op (the cross-player median residual is
    just noise around zero), leaving each player's individual motion
    intact within the HMR-noise floor."""
    n = 80
    fps = 25.0
    velocities = [(0.20, 0.00), (-0.15, 0.08), (0.05, 0.15), (0.10, -0.10)]
    starts = [(10.0, 30.0), (60.0, 25.0), (40.0, 45.0), (20.0, 10.0)]
    truth = [
        np.column_stack([
            s[0] + np.arange(n) * v[0],
            s[1] + np.arange(n) * v[1],
            np.full(n, 0.95),
        ])
        for s, v in zip(starts, velocities)
    ]
    tracks = [
        _make_track(player_id=f"P{i+1:03d}", shot_id="play", root_t=truth[i])
        for i in range(len(truth))
    ]

    corrected, _ = _apply_residual_consensus_correction(
        tracks, fps=fps, enabled=True,
        savgol_window_s=1.2, savgol_poly=2, min_players=3,
        min_offset_m=0.02,  # don't chase sub-2-cm noise
    )

    # Interior frames: should match truth tightly. The min_offset_m
    # floor keeps the pass from injecting noise into a clean signal.
    half_w = int(round(1.2 * fps / 2.0))
    interior = slice(half_w, n - half_w)
    for truth_i, fixed in zip(truth, corrected):
        delta = np.linalg.norm(
            fixed.root_t[interior, :2] - truth_i[interior, :2], axis=1,
        )
        assert float(np.max(delta)) < 0.02


# ── Stage-level integration test ────────────────────────────────────


@pytest.mark.integration
def test_refined_poses_stage_corrects_jitter_end_to_end(
    tmp_path: Path,
) -> None:
    """Stage wiring: a one-frame jitter applied to four players in
    hmr_world arrives at refined_poses corrected, and the summary
    JSON records non-zero corrected_frames."""
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="play", offsets={"play": 0})

    n = 30
    jitter_frame = 15
    static_positions = [(10.0, 20.0), (30.0, 40.0), (50.0, 5.0), (70.0, 60.0)]
    for pid, pos in zip(("P001", "P002", "P003", "P004"), static_positions):
        rt = np.tile([pos[0], pos[1], 0.95], (n, 1))
        rt[jitter_frame, 0] += 0.5
        SmplWorldTrack(
            player_id=pid,
            frames=np.arange(n, dtype=np.int64),
            betas=np.zeros(10, dtype=np.float32),
            thetas=np.zeros((n, 24, 3), dtype=np.float32),
            root_R=np.tile(np.eye(3), (n, 1, 1)).astype(np.float32),
            root_t=rt.astype(np.float32),
            confidence=np.full(n, 1.0, dtype=np.float32),
            shot_id="play",
        ).save(output_dir / "hmr_world" / f"play__{pid}_smpl_world.npz")

    cfg = {
        "refined_poses": {
            "ground_snap_max_distance": 0.0,
            # Disable the per-player Savgol so the jitter-correction
            # effect isn't blurred away in this assertion (we want to
            # see the raw corrected sample at frame 15).
            "smooth_root_t_window": 1,
            "smooth_root_R_window": 1,
            "smooth_thetas_window": 1,
            "jitter": {
                "enabled": True,
                "min_players": 4,
                "baseline_window_s": 1.0,
                "baseline_floor_m": 0.05,
                "tempo_ratio": 2.5,
                "max_spread_ratio": 0.30,
            },
        }
    }
    stage = RefinedPosesStage(config=cfg, output_dir=output_dir)
    stage.run()

    for pid, orig_pos in zip(("P001", "P002", "P003", "P004"), static_positions):
        refined = RefinedPose.load(
            output_dir / "refined_poses" / f"{pid}_refined.npz",
        )
        np.testing.assert_allclose(
            refined.root_t[jitter_frame, :2],
            np.array(orig_pos),
            atol=0.01,
        )

    summary = json.loads(
        (output_dir / "refined_poses" / "refined_poses_summary.json").read_text()
    )
    assert "jitter" in summary
    assert summary["jitter"]["corrected_frames"] >= 1
    assert summary["jitter"]["max_offset_m"] >= 0.4


@pytest.mark.integration
def test_refined_poses_stage_jitter_disabled_via_config(
    tmp_path: Path,
) -> None:
    """``refined_poses.jitter.enabled: false`` skips the new pass —
    even with a synthetic jitter present, the saved root_t carries
    the original shift (up to the existing per-player Savgol blur).
    Used as a regression escape hatch."""
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="play", offsets={"play": 0})

    n = 30
    jitter_frame = 15
    pos = (10.0, 20.0)
    rt = np.tile([pos[0], pos[1], 0.95], (n, 1))
    rt[jitter_frame, 0] += 0.5
    for pid in ("P001", "P002", "P003", "P004"):
        SmplWorldTrack(
            player_id=pid,
            frames=np.arange(n, dtype=np.int64),
            betas=np.zeros(10, dtype=np.float32),
            thetas=np.zeros((n, 24, 3), dtype=np.float32),
            root_R=np.tile(np.eye(3), (n, 1, 1)).astype(np.float32),
            root_t=rt.astype(np.float32),
            confidence=np.full(n, 1.0, dtype=np.float32),
            shot_id="play",
        ).save(output_dir / "hmr_world" / f"play__{pid}_smpl_world.npz")

    cfg = {
        "refined_poses": {
            "ground_snap_max_distance": 0.0,
            "smooth_root_t_window": 1,
            "smooth_root_R_window": 1,
            "smooth_thetas_window": 1,
            "jitter": {"enabled": False},
        }
    }
    RefinedPosesStage(config=cfg, output_dir=output_dir).run()

    summary = json.loads(
        (output_dir / "refined_poses" / "refined_poses_summary.json").read_text()
    )
    assert summary["jitter"]["corrected_frames"] == 0

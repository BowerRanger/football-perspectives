"""Tests for per-player translation cleanup in refined_poses.

These cover the new robust per-player pass that runs before the
cross-player consensus and final smoothing:

  - ``_velocity_limit_xy``  : physical speed cap (forward/backward).
  - ``_hampel_outlier_mask``: local-median outlier detection (metres).
  - ``_clean_player_translation``: densify short gaps, reject position
    outliers, clamp velocity — carrying root_R / thetas / confidence
    along the same grid.

The artifacts these target (from real hmr_world data): per-player
teleports (single-frame 10-80 m excursions), high-frequency depth
wobble (apparent 15-30 m/s), and missing-frame gaps.
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
    _clean_player_translation,
    _hampel_outlier_mask,
    _velocity_limit_xy,
)


def _make_track(
    *,
    player_id: str = "P001",
    shot_id: str = "play",
    frames: np.ndarray,
    root_t: np.ndarray,
    confidence: np.ndarray | None = None,
    root_R: np.ndarray | None = None,
    thetas: np.ndarray | None = None,
) -> SmplWorldTrack:
    n = root_t.shape[0]
    return SmplWorldTrack(
        player_id=player_id,
        frames=np.asarray(frames, dtype=np.int64),
        betas=np.zeros(10, dtype=np.float32),
        thetas=(thetas if thetas is not None
                else np.zeros((n, 24, 3), dtype=np.float32)),
        root_R=(root_R if root_R is not None
                else np.tile(np.eye(3), (n, 1, 1)).astype(np.float32)),
        root_t=root_t.astype(np.float32),
        confidence=(confidence.astype(np.float32) if confidence is not None
                    else np.full(n, 0.9, dtype=np.float32)),
        shot_id=shot_id,
    )


# ── _velocity_limit_xy ──────────────────────────────────────────────


@pytest.mark.unit
def test_velocity_limit_caps_single_frame_teleport() -> None:
    """A static track with one frame jumping 5 m and back: after the
    limit, no consecutive step exceeds the cap."""
    xy = np.zeros((11, 2))
    xy[5, 0] = 5.0  # teleport out and back
    out = _velocity_limit_xy(xy, max_step=0.4)
    steps = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert float(steps.max()) <= 0.4 + 1e-6


@pytest.mark.unit
def test_velocity_limit_preserves_genuine_sprint() -> None:
    """A constant 0.33 m/frame ramp (≈10 m/s at 30 fps) is below a
    0.4 m/frame cap and must pass through essentially unchanged."""
    xy = np.column_stack([np.arange(40) * 0.33, np.zeros(40)])
    out = _velocity_limit_xy(xy, max_step=0.4)
    np.testing.assert_allclose(out, xy, atol=1e-6)


# ── _hampel_outlier_mask ────────────────────────────────────────────


@pytest.mark.unit
def test_hampel_flags_isolated_spike() -> None:
    """A lone large excursion is flagged; the surrounding clean points
    are not."""
    xy = np.zeros((21, 2))
    xy[10] = [8.0, 0.0]
    mask = _hampel_outlier_mask(xy, window=13, k=3.0, abs_floor=0.6)
    assert mask[10]
    assert not mask[np.arange(21) != 10].any()


@pytest.mark.unit
def test_hampel_preserves_genuine_drift() -> None:
    """A smooth slow drift contains no outliers — nothing flagged."""
    xy = np.column_stack([np.linspace(0, 3, 60), np.linspace(0, 1, 60)])
    mask = _hampel_outlier_mask(xy, window=13, k=3.0, abs_floor=0.6)
    assert not mask.any()


# ── _clean_player_translation ───────────────────────────────────────


@pytest.mark.unit
def test_clean_translation_disabled_is_passthrough() -> None:
    n = 20
    frames = np.arange(n)
    rt = np.column_stack([np.arange(n) * 0.1, np.zeros(n), np.full(n, 0.9)])
    tr = _make_track(frames=frames, root_t=rt)
    out, stats = _clean_player_translation(tr, fps=30.0, enabled=False)
    np.testing.assert_array_equal(out.root_t, tr.root_t)
    np.testing.assert_array_equal(out.frames, tr.frames)


@pytest.mark.unit
def test_clean_translation_fills_short_gap() -> None:
    """Frames [0,1,2,5,6] (a 2-frame gap at 3,4) become a contiguous
    grid with the gap linearly interpolated."""
    frames = np.array([0, 1, 2, 5, 6])
    # physical motion (0.1 m/frame ≈ 3 m/s at 30 fps) so the velocity
    # limit never engages and the fill stays a pure interpolation.
    rt = np.column_stack([
        frames.astype(float) * 0.1,
        np.zeros(5),
        np.full(5, 0.9),
    ])
    tr = _make_track(frames=frames, root_t=rt)
    out, stats = _clean_player_translation(
        tr, fps=30.0, enabled=True, max_gap_fill_frames=5,
    )
    np.testing.assert_array_equal(out.frames, np.arange(0, 7))
    # x == 0.1 * frame, so the filled frames 3,4 must interpolate to 0.3,0.4
    np.testing.assert_allclose(out.root_t[:, 0], np.arange(0, 7) * 0.1, atol=1e-4)
    assert stats["filled_frames"] == 2


@pytest.mark.unit
def test_clean_translation_does_not_fill_long_gap() -> None:
    """A gap longer than max_gap_fill_frames is left as a hole (the
    track is split into runs); the long gap's frames are not fabricated."""
    frames = np.array([0, 1, 2, 50, 51, 52])
    rt = np.column_stack([
        np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0]),
        np.zeros(6),
        np.full(6, 0.9),
    ])
    tr = _make_track(frames=frames, root_t=rt)
    out, stats = _clean_player_translation(
        tr, fps=30.0, enabled=True, max_gap_fill_frames=10,
    )
    # the 47-frame gap between 2 and 50 must NOT be filled
    assert 25 not in set(out.frames.tolist())
    assert out.frames.max() == 52
    assert stats["filled_frames"] == 0


@pytest.mark.unit
def test_clean_translation_removes_teleport_keeps_position() -> None:
    """A static player with one catastrophic 20 m single-frame
    excursion: after cleanup the excursion is gone and the player
    stays at the true position, with bounded per-frame speed."""
    n = 31
    frames = np.arange(n)
    rt = np.column_stack([
        np.full(n, 30.0), np.full(n, 20.0), np.full(n, 0.9),
    ])
    rt[15, 0] += 20.0  # teleport
    tr = _make_track(frames=frames, root_t=rt)
    out, stats = _clean_player_translation(
        tr, fps=30.0, enabled=True,
        hampel_window_s=0.4, hampel_k=3.0, hampel_floor_m=0.6,
        v_max_m_s=12.0,
    )
    # frame 15 back near the true (30, 20)
    np.testing.assert_allclose(out.root_t[15, :2], [30.0, 20.0], atol=0.5)
    speeds = np.linalg.norm(np.diff(out.root_t[:, :2], axis=0), axis=1) * 30.0
    assert float(speeds.max()) <= 12.0 + 1e-3
    assert stats["rejected_frames"] >= 1


@pytest.mark.unit
def test_clean_translation_carries_rotation_and_thetas() -> None:
    """Filled frames must produce valid rotation matrices and the
    thetas array must stay shape-consistent with the dense frames."""
    frames = np.array([0, 1, 2, 5, 6])
    n = 5
    rt = np.column_stack([np.arange(n, dtype=float), np.zeros(n), np.full(n, 0.9)])
    # distinct rotations so SLERP fill is exercised
    angles = np.linspace(0.0, 1.0, n)
    root_R = np.stack([
        np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        for a in angles
    ]).astype(np.float32)
    thetas = np.tile(np.arange(n, dtype=np.float32).reshape(n, 1, 1), (1, 24, 3))
    tr = _make_track(frames=frames, root_t=rt, root_R=root_R, thetas=thetas)
    out, _ = _clean_player_translation(
        tr, fps=30.0, enabled=True, max_gap_fill_frames=5,
    )
    assert out.root_R.shape == (7, 3, 3)
    assert out.thetas.shape == (7, 24, 3)
    # every rotation stays orthonormal (valid SO(3))
    for R in out.root_R:
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-4)


# ── Stage integration ───────────────────────────────────────────────


def _write_sync_map(output_dir: Path, *, ref: str, offsets: dict[str, int]) -> None:
    sm = SyncMap(groups=[GroupSync(
        group_id="",
        reference_shot=ref,
        alignments=[
            Alignment(shot_id=s, frame_offset=o, method="manual", confidence=1.0)
            for s, o in offsets.items()
        ],
    )])
    (output_dir / "shots").mkdir(parents=True, exist_ok=True)
    sm.save(output_dir / "shots" / "sync_map.json")


@pytest.mark.integration
def test_stage_cleanup_removes_teleport_end_to_end(tmp_path: Path) -> None:
    """A per-player teleport injected in hmr_world arrives bounded in
    the saved RefinedPose, and the summary records cleanup work."""
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="play", offsets={"play": 0})

    n = 60
    teleport_frame = 30
    for pid, base in zip(("P001", "P002", "P003", "P004"),
                         [(10.0, 20.0), (30.0, 40.0), (50.0, 5.0), (70.0, 60.0)]):
        rt = np.tile([base[0], base[1], 0.95], (n, 1))
        rt[teleport_frame, 0] += 25.0  # single-frame teleport
        SmplWorldTrack(
            player_id=pid,
            frames=np.arange(n, dtype=np.int64),
            betas=np.zeros(10, dtype=np.float32),
            thetas=np.zeros((n, 24, 3), dtype=np.float32),
            root_R=np.tile(np.eye(3), (n, 1, 1)).astype(np.float32),
            root_t=rt.astype(np.float32),
            confidence=np.full(n, 0.9, dtype=np.float32),
            shot_id="play",
        ).save(output_dir / "hmr_world" / f"play__{pid}_smpl_world.npz")

    cfg = {
        "refined_poses": {
            "ground_snap_max_distance": 0.0,
            "cleanup": {"enabled": True, "v_max_m_s": 12.0},
        }
    }
    RefinedPosesStage(config=cfg, output_dir=output_dir).run()

    for pid in ("P001", "P002", "P003", "P004"):
        rp = RefinedPose.load(output_dir / "refined_poses" / f"{pid}_refined.npz")
        speeds = np.linalg.norm(np.diff(rp.root_t[:, :2], axis=0), axis=1) * 30.0
        assert float(speeds.max()) <= 12.0 + 0.5, (
            f"{pid} still has a >12 m/s step after cleanup"
        )

    summary = json.loads(
        (output_dir / "refined_poses" / "refined_poses_summary.json").read_text()
    )
    assert "cleanup" in summary
    assert summary["cleanup"]["rejected_frames"] >= 1


@pytest.mark.integration
def test_stage_cleanup_bounds_multishot_switch_teleports(tmp_path: Path) -> None:
    """A player seen in two synced shots whose camera solves disagree
    by 3 m: the highest-confidence pick alternates between the two
    positions frame-to-frame, which would teleport the player. The
    post-assembly cleanup must bound the assembled track's per-frame
    speed to the physical cap."""
    output_dir = tmp_path
    (output_dir / "hmr_world").mkdir()
    _write_sync_map(output_dir, ref="a", offsets={"a": 0, "b": 0})

    n = 40
    frames = np.arange(n, dtype=np.int64)
    # shot a places P001 at x=10; shot b at x=13 (a 3 m disagreement).
    # Confidence alternates so the per-frame highest-conf pick flips
    # between the two shots every frame.
    conf_a = np.where(frames % 2 == 0, 0.9, 0.4).astype(np.float32)
    conf_b = np.where(frames % 2 == 0, 0.4, 0.9).astype(np.float32)
    for shot, x, conf in (("a", 10.0, conf_a), ("b", 13.0, conf_b)):
        SmplWorldTrack(
            player_id="P001",
            frames=frames,
            betas=np.zeros(10, dtype=np.float32),
            thetas=np.zeros((n, 24, 3), dtype=np.float32),
            root_R=np.tile(np.eye(3), (n, 1, 1)).astype(np.float32),
            root_t=np.tile([x, 10.0, 0.95], (n, 1)).astype(np.float32),
            confidence=conf,
            shot_id=shot,
        ).save(output_dir / "hmr_world" / f"{shot}__P001_smpl_world.npz")

    cfg = {
        "refined_poses": {
            "ground_snap_max_distance": 0.0,
            "cleanup": {"enabled": True, "v_max_m_s": 12.0},
        }
    }
    RefinedPosesStage(config=cfg, output_dir=output_dir).run()

    rp = RefinedPose.load(output_dir / "refined_poses" / "P001_refined.npz")
    assert set(rp.contributing_shots) == {"a", "b"}
    speeds = np.linalg.norm(np.diff(rp.root_t[:, :2], axis=0), axis=1) * 30.0
    assert float(speeds.max()) <= 12.0 + 0.5, (
        "multi-shot switch teleports were not bounded"
    )

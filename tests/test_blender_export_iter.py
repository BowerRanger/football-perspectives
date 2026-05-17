"""Unit tests for ``scripts/blender_export_fbx.iter_player_fbx_entries``.

The Blender FBX exporter has to read the post-HMR-cleanup tracks (the
output of ``refined_poses``) — not raw ``hmr_world`` — so that the
rotation-outlier rejection, lean correction, ground snap, and
smoothing applied in refined_poses actually reach the FBX bundle
exported for UE5.

Refined NPZs live on the shared reference timeline; the exporter
needs per-shot local frame indices. ``iter_player_fbx_entries`` reads
``sync_map.json`` and applies the per-shot offset so the FBX
timeline aligns with the per-shot camera FBX. These tests cover the
preference / fallback logic and the offset application without
needing Blender.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.schemas.refined_pose import RefinedPose
from src.schemas.smpl_world import SmplWorldTrack
from src.schemas.sync_map import Alignment, SyncMap

# Load ``scripts/blender_export_fbx.py`` as a plain module — it lives
# outside the ``src`` package and the Blender main() is not imported,
# only the module-level ``iter_player_fbx_entries`` helper.
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "blender_export_fbx.py"
)
_spec = importlib.util.spec_from_file_location("bef_iter", _SCRIPT_PATH)
_bef = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bef)
iter_player_fbx_entries = _bef.iter_player_fbx_entries


def _write_refined(
    out_dir: Path,
    *,
    player_id: str,
    contributing: tuple[str, ...],
    n_frames: int,
    root_t_marker: float,
) -> None:
    """Write one ``refined_poses/{pid}_refined.npz`` whose ``root_t.x``
    is a constant marker so tests can confirm the FBX iter pulled
    THIS file (rather than the hmr_world fallback)."""
    rp_dir = out_dir / "refined_poses"
    rp_dir.mkdir(parents=True, exist_ok=True)
    n = n_frames
    RefinedPose(
        player_id=player_id,
        frames=np.arange(n, dtype=np.int64),
        betas=np.zeros(10, dtype=np.float32),
        thetas=np.zeros((n, 24, 3), dtype=np.float32),
        root_R=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
        root_t=np.column_stack(
            [np.full(n, root_t_marker, dtype=np.float32),
             np.zeros(n, dtype=np.float32),
             np.zeros(n, dtype=np.float32)]
        ),
        confidence=np.ones(n, dtype=np.float32),
        view_count=np.ones(n, dtype=np.int32),
        contributing_shots=contributing,
    ).save(rp_dir / f"{player_id}_refined.npz")


def _write_hmr(
    out_dir: Path,
    *,
    player_id: str,
    shot_id: str,
    n_frames: int,
    root_t_marker: float,
) -> None:
    hmr_dir = out_dir / "hmr_world"
    hmr_dir.mkdir(parents=True, exist_ok=True)
    n = n_frames
    SmplWorldTrack(
        player_id=player_id,
        frames=np.arange(n, dtype=np.int64),
        betas=np.zeros(10, dtype=np.float32),
        thetas=np.zeros((n, 24, 3), dtype=np.float32),
        root_R=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
        root_t=np.column_stack(
            [np.full(n, root_t_marker, dtype=np.float32),
             np.zeros(n, dtype=np.float32),
             np.zeros(n, dtype=np.float32)]
        ),
        confidence=np.ones(n, dtype=np.float32),
        shot_id=shot_id,
    ).save(hmr_dir / f"{shot_id}__{player_id}_smpl_world.npz")


def _write_sync(out_dir: Path, *, ref: str, offsets: dict[str, int]) -> None:
    (out_dir / "shots").mkdir(parents=True, exist_ok=True)
    SyncMap(
        reference_shot=ref,
        alignments=[
            Alignment(shot_id=sid, frame_offset=off, method="manual",
                      confidence=1.0)
            for sid, off in offsets.items()
        ],
    ).save(out_dir / "shots" / "sync_map.json")


@pytest.mark.unit
def test_iter_prefers_refined_over_hmr_world(tmp_path: Path) -> None:
    """When both refined_poses and hmr_world tracks exist for the same
    player, the FBX iterator uses the refined data — that's the whole
    point of the stage."""
    _write_refined(
        tmp_path, player_id="P001", contributing=("play",),
        n_frames=5, root_t_marker=99.0,  # marker that only refined has
    )
    _write_hmr(
        tmp_path, player_id="P001", shot_id="play",
        n_frames=5, root_t_marker=1.0,   # marker that hmr_world has
    )
    _write_sync(tmp_path, ref="play", offsets={"play": 0})

    entries = list(iter_player_fbx_entries(tmp_path, np))
    assert len(entries) == 1
    entry = entries[0]
    assert entry["player_id"] == "P001"
    assert entry["shot_id"] == "play"
    # Marker 99 means we read the refined file, not hmr_world (1.0).
    np.testing.assert_allclose(entry["root_t"][:, 0], 99.0)


@pytest.mark.unit
def test_iter_falls_back_to_hmr_world_when_refined_missing(
    tmp_path: Path,
) -> None:
    """No refined_poses directory → iterator yields raw hmr_world
    tracks so a user running ``--stages export`` (without
    refined_poses) still gets FBX output."""
    _write_hmr(
        tmp_path, player_id="P001", shot_id="play",
        n_frames=5, root_t_marker=1.0,
    )
    entries = list(iter_player_fbx_entries(tmp_path, np))
    assert len(entries) == 1
    assert entries[0]["player_id"] == "P001"
    assert entries[0]["shot_id"] == "play"
    np.testing.assert_allclose(entries[0]["root_t"][:, 0], 1.0)


@pytest.mark.unit
def test_iter_applies_sync_map_offset_to_refined_frames(
    tmp_path: Path,
) -> None:
    """Refined NPZs live on the reference timeline. The FBX iterator
    must translate their frame indices via sync_map.offset_for(shot)
    so each shot's FBX has local-frame indices that line up with the
    per-shot camera FBX (``local = ref + offset``)."""
    _write_refined(
        tmp_path, player_id="P001", contributing=("B",),
        n_frames=3, root_t_marker=10.0,
    )
    _write_sync(tmp_path, ref="A", offsets={"A": 0, "B": 5})

    entries = list(iter_player_fbx_entries(tmp_path, np))
    assert len(entries) == 1
    # Reference frames [0, 1, 2] shifted by B's offset 5 → [5, 6, 7].
    np.testing.assert_array_equal(entries[0]["frames"], [5, 6, 7])


@pytest.mark.unit
def test_iter_emits_one_entry_per_contributing_shot(tmp_path: Path) -> None:
    """A player visible in two shots produces two FBX entries, one
    per shot, each with frames offset to that shot's local timeline."""
    _write_refined(
        tmp_path, player_id="P001", contributing=("A", "B"),
        n_frames=4, root_t_marker=0.0,
    )
    _write_sync(tmp_path, ref="A", offsets={"A": 0, "B": 10})

    entries = sorted(
        iter_player_fbx_entries(tmp_path, np),
        key=lambda e: e["shot_id"],
    )
    assert [e["shot_id"] for e in entries] == ["A", "B"]
    np.testing.assert_array_equal(entries[0]["frames"], [0, 1, 2, 3])
    np.testing.assert_array_equal(entries[1]["frames"], [10, 11, 12, 13])

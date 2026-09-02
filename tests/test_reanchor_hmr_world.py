"""Tests for scripts/reanchor_hmr_world.py — the local (GPU-free)
re-solve of hmr_world's root translation from saved sidecars (plan Task
5, spec docs/superpowers/specs/2026-09-02-foot-contact-locomotion-design.md
§2[C]).

Builds a minimal ``output/`` fixture (hmr_world npz + kp2d sidecar +
camera track) directly rather than running the stage, since the script
under test never touches GVHMR — only
``src.stages.hmr_world.anchor_root_translation`` plus file I/O.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.reanchor_hmr_world import main as reanchor_main
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.foot_contacts import load_foot_contacts
from src.schemas.smpl_world import SmplWorldTrack

_N_FRAMES = 30


def _build_camera_track(shot_id: str, n_frames: int) -> CameraTrack:
    eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    R_world_to_cam = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    return CameraTrack(
        clip_id=shot_id,
        fps=25.0,
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


def _build_kp2d_sidecar(shot_id: str, pid: str, n_frames: int) -> dict:
    frames = []
    for i in range(n_frames):
        kp = [[0.0, 0.0, 0.0]] * 17
        kp[15] = [150.0 + i, 380.0, 0.9]  # left ankle
        kp[16] = [160.0 + i, 380.0, 0.9]  # right ankle
        frames.append({"frame": i, "keypoints": kp})
    return {"player_id": pid, "shot_id": shot_id, "frames": frames}


def _build_smpl_world_track(pid: str, shot_id: str, n_frames: int) -> SmplWorldTrack:
    rng = np.random.default_rng(0)
    root_R = np.tile(np.eye(3), (n_frames, 1, 1)).astype(np.float32)
    return SmplWorldTrack(
        player_id=pid,
        frames=np.arange(n_frames, dtype=np.int64),
        betas=rng.normal(0, 0.1, 10).astype(np.float32),
        thetas=rng.normal(0, 0.05, (n_frames, 24, 3)).astype(np.float32),
        root_R=root_R,
        # Placeholder — the whole point of the script is to overwrite this.
        root_t=np.zeros((n_frames, 3), dtype=np.float32),
        confidence=np.zeros(n_frames, dtype=np.float32),
        shot_id=shot_id,
    )


def _build_fixture(
    tmp_path: Path, *, shot_id: str = "gberch", pid: str = "P001",
    n_frames: int = _N_FRAMES,
) -> Path:
    """Populate ``tmp_path`` as a pipeline output dir with one hmr_world
    (shot, player) entry: npz + kp2d sidecar + camera track. Returns the
    npz path."""
    hmr_dir = tmp_path / "hmr_world"
    hmr_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "camera").mkdir(parents=True, exist_ok=True)

    _build_camera_track(shot_id, n_frames).save(
        tmp_path / "camera" / f"{shot_id}_camera_track.json"
    )
    (hmr_dir / f"{shot_id}__{pid}_kp2d.json").write_text(
        json.dumps(_build_kp2d_sidecar(shot_id, pid, n_frames))
    )
    npz_path = hmr_dir / f"{shot_id}__{pid}_smpl_world.npz"
    _build_smpl_world_track(pid, shot_id, n_frames).save(npz_path)
    return npz_path


@pytest.mark.unit
def test_reanchor_script_rewrites_root_t_only(tmp_path: Path) -> None:
    npz_path = _build_fixture(tmp_path)
    original = SmplWorldTrack.load(npz_path)
    original_bytes = npz_path.read_bytes()

    rc = reanchor_main([
        "--output", str(tmp_path), "--shot", "gberch", "--mode", "ankle_mid",
        "--in-place",
    ])
    assert rc == 0

    bak_path = npz_path.with_name(npz_path.name + ".pre_reanchor.bak")
    assert bak_path.exists(), "in-place reanchor did not create a .bak"
    assert bak_path.read_bytes() == original_bytes, (
        "the .bak must be a byte-exact copy of the pristine original"
    )

    rewritten = SmplWorldTrack.load(npz_path)
    np.testing.assert_array_equal(rewritten.frames, original.frames)
    np.testing.assert_array_equal(rewritten.thetas, original.thetas)
    np.testing.assert_array_equal(rewritten.root_R, original.root_R)
    np.testing.assert_array_equal(rewritten.betas, original.betas)
    assert rewritten.player_id == original.player_id
    assert rewritten.shot_id == original.shot_id
    # root_t/confidence are exactly what the script exists to recompute.
    assert rewritten.root_t.shape == original.root_t.shape
    assert np.all(np.isfinite(rewritten.root_t))

    # A second in-place run must NOT touch the existing .bak (it holds
    # the pristine GVHMR original, which cannot be regenerated on this
    # Mac) even though the working npz has since changed.
    bak_bytes_after_first = bak_path.read_bytes()
    rc2 = reanchor_main([
        "--output", str(tmp_path), "--shot", "gberch", "--mode", "ankle_mid",
        "--in-place",
    ])
    assert rc2 == 0
    assert bak_path.read_bytes() == bak_bytes_after_first == original_bytes


@pytest.mark.unit
def test_reanchor_script_suffix_mode_is_non_destructive(tmp_path: Path) -> None:
    npz_path = _build_fixture(tmp_path)
    original_bytes = npz_path.read_bytes()

    rc = reanchor_main([
        "--output", str(tmp_path), "--shot", "gberch", "--mode", "contact",
    ])
    assert rc == 0

    # Original untouched.
    assert npz_path.read_bytes() == original_bytes
    assert not npz_path.with_name(npz_path.name + ".pre_reanchor.bak").exists()

    reanchored_path = (
        tmp_path / "hmr_world" / "gberch__P001_reanchored_smpl_world.npz"
    )
    assert reanchored_path.exists()
    reanchored = SmplWorldTrack.load(reanchored_path)
    original = SmplWorldTrack.load(npz_path)
    np.testing.assert_array_equal(reanchored.thetas, original.thetas)
    np.testing.assert_array_equal(reanchored.root_R, original.root_R)
    np.testing.assert_array_equal(reanchored.betas, original.betas)
    np.testing.assert_array_equal(reanchored.frames, original.frames)


@pytest.mark.unit
def test_reanchor_script_writes_contacts_sidecar_in_contact_mode(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    rc = reanchor_main([
        "--output", str(tmp_path), "--shot", "gberch", "--mode", "contact",
        "--suffix", "_reanchored",
    ])
    assert rc == 0

    sidecar_path = (
        tmp_path / "hmr_world" / "gberch__P001_reanchored_foot_contacts.json"
    )
    assert sidecar_path.exists()
    contacts, meta = load_foot_contacts(sidecar_path)
    assert contacts.n_frames == _N_FRAMES
    assert meta["shot_id"] == "gberch"
    assert meta["player_id"] == "P001"
    assert meta["anchor_mode"] == "contact"


@pytest.mark.unit
def test_reanchor_script_ankle_mid_mode_skips_contacts_sidecar(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    rc = reanchor_main([
        "--output", str(tmp_path), "--shot", "gberch", "--mode", "ankle_mid",
        "--suffix", "_reanchored",
    ])
    assert rc == 0

    sidecar_path = (
        tmp_path / "hmr_world" / "gberch__P001_reanchored_foot_contacts.json"
    )
    assert not sidecar_path.exists()
    npz_path = tmp_path / "hmr_world" / "gberch__P001_reanchored_smpl_world.npz"
    assert npz_path.exists()


@pytest.mark.unit
def test_reanchor_script_shot_and_player_filters_restrict_processing(
    tmp_path: Path,
) -> None:
    _build_fixture(tmp_path, shot_id="alpha", pid="P001")
    _build_fixture(tmp_path, shot_id="alpha", pid="P002")
    _build_fixture(tmp_path, shot_id="beta", pid="P001")

    rc = reanchor_main([
        "--output", str(tmp_path), "--shot", "alpha", "--players", "P001",
        "--mode", "ankle_mid", "--suffix", "_reanchored",
    ])
    assert rc == 0

    hmr_dir = tmp_path / "hmr_world"
    assert (hmr_dir / "alpha__P001_reanchored_smpl_world.npz").exists()
    assert not (hmr_dir / "alpha__P002_reanchored_smpl_world.npz").exists()
    assert not (hmr_dir / "beta__P001_reanchored_smpl_world.npz").exists()


@pytest.mark.unit
def test_reanchor_script_missing_kp2d_sidecar_is_skipped_not_raised(
    tmp_path: Path,
) -> None:
    npz_path = _build_fixture(tmp_path)
    (tmp_path / "hmr_world" / "gberch__P001_kp2d.json").unlink()

    rc = reanchor_main([
        "--output", str(tmp_path), "--shot", "gberch", "--mode", "contact",
        "--suffix", "_reanchored",
    ])
    assert rc == 0
    assert not (
        tmp_path / "hmr_world" / "gberch__P001_reanchored_smpl_world.npz"
    ).exists()
    # Original left completely untouched.
    assert npz_path.exists()

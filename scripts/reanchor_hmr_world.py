"""Re-solve ``hmr_world`` root translation from saved sidecars, WITHOUT
re-running GVHMR (plan Task 5, spec
docs/superpowers/specs/2026-09-02-foot-contact-locomotion-design.md
§2[C]). This is what makes extraction-time anchoring changes testable on
this Mac: GVHMR itself needs the GPU box, but the translation solve
(``src.stages.hmr_world.anchor_root_translation``) only needs the
already-saved ``*_smpl_world.npz`` (thetas/root_R/betas/frames — NOT
touched by this script), its ``*_kp2d.json`` sidecar, and the shot's
camera track.

Usage:
    .venv311/bin/python scripts/reanchor_hmr_world.py --output output \\
        --shot gberch --mode contact
    # non-destructive by default: writes gberch__P001_reanchored_smpl_world.npz
    # (+ _foot_contacts.json in contact mode) alongside the originals.

    .venv311/bin/python scripts/reanchor_hmr_world.py --output output \\
        --shot gberch --mode contact --in-place
    # overwrites the existing npz/sidecar in place. SAFETY: the first
    # --in-place run on a given file copies the ORIGINAL to
    # "<name>.npz.pre_reanchor.bak" and NEVER overwrites an existing
    # .bak on any later run — the GVHMR originals cannot be regenerated
    # on this Mac (GVHMR needs the GPU box).

``thetas``/``root_R``/``betas``/``frames`` are carried through byte-
identical from the loaded npz; only ``root_t``/``confidence`` (and the
contacts sidecar, in contact mode) are recomputed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.config import load_config  # noqa: E402
from src.schemas.camera_track import CameraTrack  # noqa: E402
from src.schemas.foot_contacts import save_foot_contacts  # noqa: E402
from src.schemas.smpl_world import SmplWorldTrack  # noqa: E402
from src.stages.hmr_world import anchor_root_translation  # noqa: E402
from src.utils.foot_contact import FootContacts  # noqa: E402

_NPZ_SUFFIX = "_smpl_world.npz"


def _discover_entries(
    hmr_dir: Path, shot_filter: str | None, player_filter: set[str] | None,
) -> list[tuple[str, str, Path]]:
    """``(shot_id, player_id, npz_path)`` for every ``*_smpl_world.npz``
    under ``hmr_dir``, matching the filters. Splits the filename stem on
    the FIRST ``__`` only (a player_id may itself contain ``_``/``__``,
    per ``src.stages.hmr_world``'s own convention) — legacy single-shot
    files with no ``__`` get ``shot_id=""``.
    """
    out: list[tuple[str, str, Path]] = []
    if not hmr_dir.exists():
        return out
    for npz in sorted(hmr_dir.glob(f"*{_NPZ_SUFFIX}")):
        stem = npz.name.removesuffix(_NPZ_SUFFIX)
        if "__" in stem:
            shot_id, pid = stem.split("__", 1)
        else:
            shot_id, pid = "", stem
        if shot_filter is not None and shot_id != shot_filter:
            continue
        if player_filter is not None and pid not in player_filter:
            continue
        out.append((shot_id, pid, npz))
    return out


def _load_camera_track(camera_dir: Path, shot_id: str) -> CameraTrack | None:
    candidates: list[Path] = []
    if shot_id:
        candidates.append(camera_dir / f"{shot_id}_camera_track.json")
    candidates.append(camera_dir / "camera_track.json")
    for p in candidates:
        if p.exists():
            try:
                return CameraTrack.load(p)
            except Exception:
                continue
    return None


def _per_frame_dicts(
    cam: CameraTrack,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Same construction as ``HmrWorldStage._build_per_frame`` — one
    entry per camera-track frame, per-frame ``t`` falling back to the
    clip-shared ``t_world`` when absent (older camera tracks)."""
    K = {int(f.frame): np.array(f.K, dtype=float) for f in cam.frames}
    R = {int(f.frame): np.array(f.R, dtype=float) for f in cam.frames}
    t_fallback = np.array(cam.t_world, dtype=float)
    t = {
        int(f.frame): (np.array(f.t, dtype=float) if f.t is not None else t_fallback)
        for f in cam.frames
    }
    return K, R, t


def _load_kp2d(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        frames = np.array([f["frame"] for f in data["frames"]], dtype=np.int64)
        kp2d = np.array([f["keypoints"] for f in data["frames"]], dtype=float)
    except Exception:
        return None
    if frames.size == 0:
        return None
    return frames, kp2d


def _align_kp2d_to_frames(
    track_frames: np.ndarray, kp2d_frames: np.ndarray, kp2d: np.ndarray,
) -> np.ndarray:
    """Reindex a ``(Fk, 17, 3)`` kp2d array onto ``track_frames``; a
    frame absent from the kp2d sidecar gets an all-zero (zero-confidence)
    row rather than raising — ``anchor_root_translation`` treats
    zero-confidence identically to a genuinely low-confidence keypoint
    (hold-last, per its documented gating)."""
    lut = {int(f): i for i, f in enumerate(kp2d_frames)}
    out = np.zeros((len(track_frames), kp2d.shape[1], 3), dtype=float)
    for i, f in enumerate(track_frames):
        j = lut.get(int(f))
        if j is not None:
            out[i] = kp2d[j]
    return out


def reanchor_one(
    *, output_dir: Path, shot_id: str, npz_path: Path, mode: str, hmr_cfg: dict,
) -> tuple[SmplWorldTrack, FootContacts | None] | None:
    """Recompute ``root_t``/``confidence``/contacts for one (shot,
    player) hmr_world track purely from its saved sidecars — never
    touches ``thetas``/``root_R``/``betas``/``frames``. Returns ``None``
    (and prints why) when a required sidecar is missing."""
    track = SmplWorldTrack.load(npz_path)
    hmr_dir = npz_path.parent
    stem = npz_path.name.removesuffix(_NPZ_SUFFIX)

    loaded = _load_kp2d(hmr_dir / f"{stem}_kp2d.json")
    if loaded is None:
        print(f"  skip {stem}: no kp2d sidecar")
        return None

    cam = _load_camera_track(output_dir / "camera", shot_id)
    if cam is None:
        print(f"  skip {stem}: no camera track for shot {shot_id!r}")
        return None

    kp2d_frames, kp2d_raw = loaded
    frame_indices = np.asarray(track.frames)
    kp2d = _align_kp2d_to_frames(frame_indices, kp2d_frames, kp2d_raw)
    per_frame_K, per_frame_R, per_frame_t = _per_frame_dicts(cam)

    cfg = dict(hmr_cfg)
    cfg["anchor_mode"] = mode

    root_t, confidence, contacts = anchor_root_translation(
        kp2d=kp2d,
        frame_indices=frame_indices,
        per_frame_K=per_frame_K,
        per_frame_R=per_frame_R,
        per_frame_t=per_frame_t,
        distortion=tuple(cam.distortion),
        thetas=track.thetas,
        root_R=track.root_R,
        betas=track.betas,
        cfg=cfg,
        fps=float(cam.fps),
    )

    new_track = SmplWorldTrack(
        player_id=track.player_id,
        frames=track.frames,
        betas=track.betas,
        thetas=track.thetas,
        root_R=track.root_R,
        root_t=root_t.astype(np.float32),
        confidence=confidence.astype(np.float32),
        shot_id=track.shot_id,
    )
    return new_track, contacts


def _backup_original(npz_path: Path) -> Path | None:
    """One-time safety copy for ``--in-place``: ``<name>.npz.pre_reanchor.bak``.
    Returns the backup path when a NEW backup was written, ``None`` when
    one already existed (and was therefore left untouched — the GVHMR
    original it holds cannot be regenerated on this Mac)."""
    bak_path = npz_path.with_name(npz_path.name + ".pre_reanchor.bak")
    if bak_path.exists():
        return None
    shutil.copyfile(npz_path, bak_path)
    return bak_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, help="pipeline output directory")
    ap.add_argument("--shot", default=None, help="only reanchor this shot id")
    ap.add_argument("--players", default=None, help="comma-separated player ids (default: all)")
    ap.add_argument("--mode", choices=["contact", "ankle_mid"], default="contact")
    dest_group = ap.add_mutually_exclusive_group()
    dest_group.add_argument(
        "--suffix", default="_reanchored",
        help="non-destructive: write {stem}{suffix}_smpl_world.npz alongside "
             "the original (default: _reanchored)",
    )
    dest_group.add_argument(
        "--in-place", action="store_true",
        help="overwrite the existing npz/sidecar; backs up the pristine "
             "original to *.npz.pre_reanchor.bak once, and never touches "
             "an existing .bak on later runs",
    )
    args = ap.parse_args(argv)

    output_dir = Path(args.output)
    hmr_dir = output_dir / "hmr_world"
    player_filter = (
        {p.strip() for p in args.players.split(",") if p.strip()}
        if args.players else None
    )

    hmr_cfg = load_config().get("hmr_world", {})

    entries = _discover_entries(hmr_dir, args.shot, player_filter)
    if not entries:
        print(f"[reanchor_hmr_world] no matching hmr_world tracks under {hmr_dir}")
        return 0

    n_written = 0
    for shot_id, pid, npz_path in entries:
        stem = npz_path.name.removesuffix(_NPZ_SUFFIX)
        result = reanchor_one(
            output_dir=output_dir, shot_id=shot_id, npz_path=npz_path,
            mode=args.mode, hmr_cfg=hmr_cfg,
        )
        if result is None:
            continue
        new_track, contacts = result

        if args.in_place:
            bak = _backup_original(npz_path)
            if bak is not None:
                print(f"  backed up {npz_path.name} -> {bak.name}")
            out_key = stem
            out_npz = npz_path
        else:
            out_key = f"{stem}{args.suffix}"
            out_npz = hmr_dir / f"{out_key}{_NPZ_SUFFIX}"

        new_track.save(out_npz)
        n_written += 1
        if contacts is not None:
            save_foot_contacts(
                hmr_dir / f"{out_key}_foot_contacts.json", contacts,
                shot_id=shot_id, player_id=pid, anchor_mode=args.mode,
            )
        print(
            f"[reanchor_hmr_world] {stem} -> {out_npz.name} "
            f"(mode={args.mode}, frames={len(new_track.frames)})"
        )

    print(f"[reanchor_hmr_world] wrote {n_written} track(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

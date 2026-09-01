"""bpy-free readers for the Blender FBX export / render pipeline.

Split out of ``scripts/blender_export_fbx.py`` so the pure-data readers
(no ``bpy``, no Blender runtime) are importable and unit-testable on
their own. See that script's module docstring for the full FBX export
contract these feed into.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator


def iter_player_fbx_entries(
    output_dir: Path, np_mod,
) -> Iterator[dict]:
    """Yield one ``dict`` per (shot, player) FBX to write.

    Prefers ``output/refined_poses/{pid}_refined.npz`` — that's where
    the rotation-outlier rejection, lean correction, ground snap, and
    smoothing live. Each refined NPZ is keyed by player_id only and
    indexed on the shared reference timeline; we apply the sync_map
    offset for each shot in ``contributing_shots`` to translate into
    per-shot local frames so the FBX timeline lines up with the
    per-shot camera FBX.

    Falls back to one entry per ``output/hmr_world/*_smpl_world.npz``
    file when no refined output is present (e.g. user re-ran
    ``--stages export`` before running ``refined_poses``).

    The ``np_mod`` argument lets the Blender entry-point pass in the
    already-imported ``numpy`` rather than re-importing — keeps this
    helper testable outside Blender too.

    Each yielded entry has keys: ``shot_id``, ``player_id``,
    ``frames``, ``thetas``, ``root_R``, ``root_t``.
    """
    refined_dir = output_dir / "refined_poses"
    refined_files = (
        sorted(refined_dir.glob("*_refined.npz"))
        if refined_dir.exists() else []
    )
    if refined_files:
        sync = None
        sync_path = output_dir / "shots" / "sync_map.json"
        if sync_path.exists():
            try:
                # Importing src.schemas.sync_map requires repo root on
                # sys.path; main() does that before invoking us. When
                # called from a unit test the same applies.
                from src.schemas.sync_map import SyncMap  # type: ignore
                sync = SyncMap.load(sync_path)
            except Exception as exc:
                sys.stderr.write(
                    f"[fbx-entries] sync_map.json load failed ({exc}); "
                    "treating offsets as 0\n"
                )
        for path in refined_files:
            data = np_mod.load(path, allow_pickle=False)
            player_id = str(data["player_id"])
            contributing_raw = (
                data["contributing_shots"]
                if "contributing_shots" in data.files else []
            )
            contributing = [str(s) for s in contributing_raw]
            if not contributing:
                # Legacy single-shot refined NPZ — emit with no shot
                # prefix so the FBX filename matches the older layout.
                contributing = [""]
            ref_frames = np_mod.asarray(data["frames"])
            thetas = np_mod.asarray(data["thetas"])
            root_R = np_mod.asarray(data["root_R"])
            root_t = np_mod.asarray(data["root_t"])
            for sid in contributing:
                offset = sync.offset_for_shot(sid) if (sync and sid) else 0
                yield {
                    "shot_id": sid,
                    "player_id": player_id,
                    "frames": ref_frames + int(offset),
                    "thetas": thetas,
                    "root_R": root_R,
                    "root_t": root_t,
                }
        return

    hmr_dir = output_dir / "hmr_world"
    if not hmr_dir.exists():
        return
    for path in sorted(hmr_dir.glob("*_smpl_world.npz")):
        data = np_mod.load(path, allow_pickle=False)
        yield {
            "shot_id": (
                str(data["shot_id"]) if "shot_id" in data.files else ""
            ),
            "player_id": str(data["player_id"]),
            "frames": np_mod.asarray(data["frames"]),
            "thetas": np_mod.asarray(data["thetas"]),
            "root_R": np_mod.asarray(data["root_R"]),
            "root_t": np_mod.asarray(data["root_t"]),
        }


def prepare_ball_keys(ball_frames: list[dict]) -> list[dict]:
    """Turn ball-track JSON frame dicts into per-frame FBX keys.

    Pure (no ``bpy``/``numpy``) so the FBX ball rotation contract is
    unit-testable outside Blender.  Each output dict has:

    - ``frame`` (int): the frame index;
    - ``location`` (list[float]): the ball world position ``world_xyz``;
    - ``rotation_quaternion`` (list[float]): the ball orientation in
      Blender's ``(w, x, y, z)`` scalar-first order.  This MATCHES our
      ``BallFrame.quat_wxyz`` convention, so the quaternion is passed
      through unchanged (no reordering — unlike the glTF ``(x, y, z, w)``
      path).

    Frames with a null ``world_xyz`` are dropped (no position to key).
    Frames missing ``quat_wxyz`` hold the previous rotation; a leading run
    of missing quats holds the identity ``(1, 0, 0, 0)``.
    """
    keys: list[dict] = []
    last_quat = [1.0, 0.0, 0.0, 0.0]
    for f in ball_frames:
        world = f.get("world_xyz")
        if not world:
            continue
        q = f.get("quat_wxyz")
        if q is not None:
            last_quat = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        keys.append({
            "frame": int(f["frame"]),
            "location": [float(world[0]), float(world[1]), float(world[2])],
            "rotation_quaternion": list(last_quat),
        })
    return keys


def load_shot_ids(output_dir: Path) -> set[str]:
    """Shot ids from shots_manifest.json (empty set for legacy single-shot)."""
    manifest_path = output_dir / "shots" / "shots_manifest.json"
    if not manifest_path.exists():
        return set()
    raw = json.loads(manifest_path.read_text())
    return {s["id"] for s in raw.get("shots", []) if s.get("id")}


def load_camera_track(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid camera track JSON at {path}: {exc}") from exc


def load_smpl_body_data(repo_root, np_mod):
    """Load data/models/smpl_neutral.npz re-anchored on the foot midpoint.

    Returns (smpl_data | None, pelvis_canon_shifted). Mirrors the FBX
    exporter's historical behaviour exactly — see that script's git
    history for the original block.
    """
    smpl_npz_path = repo_root / "data" / "models" / "smpl_neutral.npz"
    pelvis_canon_shifted = np_mod.zeros(3, dtype=np_mod.float64)
    if not smpl_npz_path.exists():
        return None, pelvis_canon_shifted
    from src.utils.smpl_skeleton import SMPL_JOINT_NAMES
    smpl_data = dict(np_mod.load(smpl_npz_path))
    if "joint_positions" in smpl_data and "v_template" in smpl_data:
        jp = np_mod.asarray(smpl_data["joint_positions"], dtype=np_mod.float64)
        l_foot = SMPL_JOINT_NAMES.index("l_foot")
        r_foot = SMPL_JOINT_NAMES.index("r_foot")
        shift = -((jp[l_foot] + jp[r_foot]) / 2.0)
        smpl_data["joint_positions"] = (jp + shift).astype(np_mod.float32)
        smpl_data["v_template"] = (
            np_mod.asarray(smpl_data["v_template"], dtype=np_mod.float64) + shift
        ).astype(np_mod.float32)
        pelvis_canon_shifted = np_mod.asarray(
            smpl_data["joint_positions"][0], dtype=np_mod.float64
        )
    return smpl_data, pelvis_canon_shifted

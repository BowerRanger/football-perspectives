"""Foot-contact locomotion quality eval CLI (eval harness [A] of
docs/superpowers/specs/2026-09-02-foot-contact-locomotion-design.md).

Computes src.utils.foot_quality.foot_quality_metrics for every requested
player on both the ``hmr_world`` (per-shot, kp2d-linked) and
``refined_poses`` (reference-timeline) artefacts and prints a compact
table + optional JSON dump.

Usage:
  .venv311/bin/python scripts/eval_foot_quality.py --output output \
      --players P001,P002,P003 --json output/foot_quality_baseline.json

Tolerant of:
  - Arbitrary ``{shot}__{pid}`` hmr_world naming, including a pid that
    itself contains underscores (e.g. japan's
    ``s013__s013_TT001_smpl_world.npz`` -> shot="s013",
    pid="s013_TT001") — split on the FIRST "__" only, matching
    ``src.stages.refined_poses._discover_player_ids``.
  - Stale ``refined_poses/*_refined.npz`` files with no matching
    ``hmr_world`` sidecar (auto-discovery intersects refined_poses
    player ids with hmr_world player ids so a leftover npz from a
    different run doesn't get reported as if it were current).
  - Missing foot-contact sidecars, kp2d sidecars, and camera tracks —
    each optional input degrades gracefully rather than raising.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schemas.camera_track import CameraTrack  # noqa: E402
from src.schemas.foot_contacts import load_foot_contacts  # noqa: E402
from src.schemas.refined_pose import RefinedPose  # noqa: E402
from src.schemas.smpl_world import SmplWorldTrack  # noqa: E402
from src.utils.foot_contact import FootContacts  # noqa: E402
from src.utils.foot_quality import foot_quality_metrics  # noqa: E402
from src.utils.smpl_skeleton import (  # noqa: E402
    beta_adjusted_rest_joints,
    load_smpl_neutral_model,
)

_DEFAULT_FPS = 25.0


def _discover_hmr_entries(hmr_dir: Path) -> list[tuple[str, str, Path, Path]]:
    """``(shot_id, player_id, npz_path, kp2d_path)`` for every
    ``*_smpl_world.npz`` in ``hmr_dir``, tolerating a pid that itself
    contains ``_`` or even ``__`` (only the FIRST ``__`` separates shot
    from player id, matching the hmr_world writer's own convention).
    Legacy single-shot files with no ``__`` get ``shot_id=""``.
    """
    out: list[tuple[str, str, Path, Path]] = []
    if not hmr_dir.exists():
        return out
    for npz in sorted(hmr_dir.glob("*_smpl_world.npz")):
        stem = npz.name.removesuffix("_smpl_world.npz")
        if "__" in stem:
            shot_id, pid = stem.split("__", 1)
        else:
            shot_id, pid = "", stem
        kp2d_path = hmr_dir / f"{stem}_kp2d.json"
        out.append((shot_id, pid, npz, kp2d_path))
    return out


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
    frames: np.ndarray, kp2d_frames: np.ndarray, kp2d: np.ndarray,
) -> np.ndarray:
    """Reindex a (Fk, 17, 3) kp2d array onto ``frames``; frames absent
    from the kp2d sidecar get a zero-confidence row (never contribute
    to ankle_reproj_px, which gates on confidence >= 0.5)."""
    lut = {int(f): i for i, f in enumerate(kp2d_frames)}
    out = np.zeros((len(frames), kp2d.shape[1], 3), dtype=float)
    for i, f in enumerate(frames):
        j = lut.get(int(f))
        if j is not None:
            out[i] = kp2d[j]
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


def _cameras_dict(track: CameraTrack) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    default_t = np.array(track.t_world, dtype=float)
    for fr in track.frames:
        K = np.array(fr.K, dtype=float)
        R = np.array(fr.R, dtype=float)
        t = np.array(fr.t, dtype=float) if fr.t is not None else default_t
        out[int(fr.frame)] = (K, R, t)
    return out


def _hmr_npz_path(hmr_dir: Path, shot_id: str, pid: str) -> Path:
    stem = f"{shot_id}__{pid}" if shot_id else pid
    return hmr_dir / f"{stem}_smpl_world.npz"


def _load_hmr_track_frames(hmr_dir: Path, shot_id: str, pid: str) -> np.ndarray | None:
    """Global frame-number array of the hmr_world npz matching
    ``(shot_id, pid)`` — the array whose ARRAY POSITIONS the
    ``{shot}__{pid}_foot_contacts.json`` sidecar's dense/span indices
    are aligned to (see ``src/schemas/foot_contacts.py``'s module
    docstring: "Frame indices inside the payload are hmr_world
    track-ARRAY positions ... not global frame numbers"). Returns
    ``None`` when the npz is missing or fails to load."""
    path = _hmr_npz_path(hmr_dir, shot_id, pid)
    if not path.exists():
        return None
    try:
        track = SmplWorldTrack.load(path)
    except Exception:
        return None
    return np.asarray(track.frames)


def _load_contacts_sidecar(
    hmr_dir: Path, shot_id: str, pid: str, frames: np.ndarray,
) -> np.ndarray | None:
    """Load ``{shot}__{pid}_foot_contacts.json`` when present and map its
    per-frame ``in_contact`` flags onto ``frames`` — the CALLER's own
    track's frame numbers (GLOBAL frame indices, e.g. ``SmplWorldTrack.
    frames`` or ``RefinedPose.frames``, not array positions).

    The sidecar's dense arrays are aligned 1:1 by ARRAY POSITION with
    the hmr_world track ``detect_contacts`` computed them from (see
    ``src/schemas/foot_contacts.py``) — not with an arbitrary caller's
    frame array. A ``refined_poses`` track has been densified/trimmed
    relative to ``hmr_world`` (different length, and its ``frames``
    array holds different — though overlapping — GLOBAL frame numbers),
    so a naive positional read only ever worked when the two happened to
    have identical length, silently falling back to the coarser z<0.10
    proxy metric on every OTHER refined-track evaluation (which is most
    of them — a refined track's length essentially never matches its
    source hmr_world track's after trimming). This resolves the mapping
    through GLOBAL FRAME NUMBER instead, via the matching hmr_world
    npz's own ``frames`` array, so trimming/resampling no longer defeats
    the sidecar: a caller frame absent from the hmr_world track is
    conservatively treated as "not in contact" rather than raising or
    forcing the whole track to the proxy.

    Falls back to the previous exact-length positional interpretation
    when the matching hmr_world npz can't be loaded (e.g. deleted after
    the sidecar was written, or a caller — such as ``eval_hmr_player`` —
    that IS the hmr_world track itself and so trivially aligns 1:1) —
    same graceful-degradation contract as before, just no longer the
    ONLY path. Returns ``None`` when neither path can resolve a mapping
    (absent file, parse failure, or an unresolvable length mismatch).
    """
    name = f"{shot_id}__{pid}_foot_contacts.json" if shot_id else f"{pid}_foot_contacts.json"
    path = hmr_dir / name
    if not path.exists():
        return None
    try:
        # Sidecars are wrapped in the versioned schema payload — unwrap
        # via the schema loader, falling back to a raw FootContacts JSON
        # for any pre-schema file.
        try:
            fc, _meta = load_foot_contacts(path)
        except ValueError:
            fc = FootContacts.from_json(json.loads(path.read_text()))
    except Exception:
        return None

    frames_arr = np.asarray(frames)
    hmr_frames = _load_hmr_track_frames(hmr_dir, shot_id, pid)
    if hmr_frames is None or len(hmr_frames) != fc.n_frames:
        if fc.n_frames != len(frames_arr):
            return None
        return fc.in_contact

    lut = {int(f): i for i, f in enumerate(hmr_frames)}
    out = np.zeros((len(frames_arr), 2), dtype=bool)
    for i, f in enumerate(frames_arr):
        j = lut.get(int(f))
        if j is not None:
            out[i] = fc.in_contact[j]
    return out


def eval_hmr_player(
    output_dir: Path,
    shot_id: str,
    pid: str,
    npz_path: Path,
    kp2d_path: Path,
    smpl_model: dict | None,
) -> dict | None:
    """Compute foot-quality metrics for one hmr_world (shot, player)
    track, including ankle_reproj_px when its kp2d sidecar and the
    shot's camera track are both available."""
    try:
        track = SmplWorldTrack.load(npz_path)
    except Exception:
        return None
    if len(track.frames) == 0:
        return None

    cam_track = _load_camera_track(output_dir / "camera", shot_id)
    fps = float(cam_track.fps) if cam_track is not None else _DEFAULT_FPS

    kp2d_arr = None
    cameras = None
    loaded = _load_kp2d(kp2d_path)
    if loaded is not None and cam_track is not None:
        kp2d_frames, kp2d_raw = loaded
        kp2d_arr = _align_kp2d_to_frames(np.asarray(track.frames), kp2d_frames, kp2d_raw)
        cameras = _cameras_dict(cam_track)

    contacts = _load_contacts_sidecar(output_dir / "hmr_world", shot_id, pid, track.frames)
    rest_joints = beta_adjusted_rest_joints(track.betas, smpl_model)

    return foot_quality_metrics(
        frames=track.frames, betas=track.betas, thetas=track.thetas,
        root_R=track.root_R, root_t=track.root_t, fps=fps,
        contacts=contacts, kp2d=kp2d_arr, cameras=cameras,
        rest_joints=rest_joints,
    )


def _load_resolved_contacts(
    output_dir: Path, pid: str, frames: np.ndarray,
) -> np.ndarray | None:
    """Load ``refined_poses/{pid}_resolved_contacts.json`` when present
    — the refined_poses stage's own verified/effective contact set
    (spans that survived BOTH the detection-side quality gates in
    ``src.utils.foot_contact.detect_contacts`` and the foot-lock
    finale's landing-quality "honesty check" —
    ``src.utils.foot_lock.lock_feet_ik``'s ``resolved_pin_err_m``; see
    docs/superpowers/specs/2026-09-02-foot-contact-locomotion-design.md's
    Wave 5 report). Preferred over the raw hmr_world detection-time
    sidecar for refined-stage evaluation: a span the pipeline could not
    verify as a stable stance must not be measured as one (false
    pinning is worse than honest free motion — the same rationale the
    stage's own honesty check documents).

    Written positionally aligned 1:1 with THIS EXACT refined track's own
    ``frames`` array (see ``src.stages.refined_poses.run``), so — unlike
    the raw hmr_world sidecar, which ``_load_contacts_sidecar`` remaps
    through global frame numbers — no remapping is needed here; an
    ``n_frames`` mismatch (stale sidecar from a different run) is
    treated as absence rather than guessed at.
    """
    path = output_dir / "refined_poses" / f"{pid}_resolved_contacts.json"
    if not path.exists():
        return None
    try:
        fc, meta = load_foot_contacts(path)
    except Exception:
        return None
    if meta.get("anchor_mode") != "resolved":
        return None
    frames_arr = np.asarray(frames)
    if fc.n_frames != len(frames_arr):
        return None
    return fc.in_contact


def eval_refined_player(
    output_dir: Path, pid: str, smpl_model: dict | None,
) -> dict | None:
    """Compute foot-quality metrics for one refined_poses player track.
    Returns ``None`` when the npz is missing or has zero frames (the
    signal a caller uses to treat a player as "not present" rather than
    reporting misleading all-zero metrics)."""
    path = output_dir / "refined_poses" / f"{pid}_refined.npz"
    if not path.exists():
        return None
    try:
        refined = RefinedPose.load(path)
    except Exception:
        return None
    if len(refined.frames) == 0:
        return None

    shot_id = refined.contributing_shots[0] if refined.contributing_shots else ""
    cam_track = _load_camera_track(output_dir / "camera", shot_id) if shot_id else None
    fps = float(cam_track.fps) if cam_track is not None else _DEFAULT_FPS
    # Prefer the stage's own verified/effective contact set; fall back
    # to the raw hmr_world detection-time sidecar, then (contacts=None)
    # to foot_quality_metrics' FK z<0.10 proxy — same graceful-
    # degradation contract as before, just with a new, preferred first
    # rung on the ladder.
    contacts = _load_resolved_contacts(output_dir, pid, refined.frames)
    if contacts is None:
        contacts = (
            _load_contacts_sidecar(output_dir / "hmr_world", shot_id, pid, refined.frames)
            if shot_id else None
        )
    rest_joints = beta_adjusted_rest_joints(refined.betas, smpl_model)

    return foot_quality_metrics(
        frames=refined.frames, betas=refined.betas, thetas=refined.thetas,
        root_R=refined.root_R, root_t=refined.root_t, fps=fps,
        contacts=contacts, rest_joints=rest_joints,
    )


def _fmt_player_line(key: str, m: dict) -> str:
    skate = m["skate"]
    pen = m["penetration"]
    lfz = m["lower_foot_z"]
    line = (
        f"  [{key:12s}] skate L {skate['L']['mean_mps']:5.2f} R {skate['R']['mean_mps']:5.2f} m/s "
        f"(p95 L {skate['L']['p95_mps']:5.2f} R {skate['R']['p95_mps']:5.2f})  "
        f"pen {pen['pct_frames_sole_below_0']:5.1f}% max {pen['max_depth_cm']:.1f}cm  "
        f"lower_foot_z p50 {lfz['p50']:.3f} p95 {lfz['p95']:.3f}  "
        f"contact_ratio {m['contact_ratio']:.2f}"
    )
    if "ankle_reproj_px" in m:
        line += f"  reproj mean {m['ankle_reproj_px']['mean_px']:.1f}px p95 {m['ankle_reproj_px']['p95_px']:.1f}px"
    return line


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, help="pipeline output directory")
    ap.add_argument("--players", default=None, help="comma-separated player ids; default auto-discovers")
    ap.add_argument("--stage", choices=["refined", "hmr", "both"], default="both")
    ap.add_argument("--json", default=None, help="path to write the full results as JSON")
    args = ap.parse_args(argv)

    output_dir = Path(args.output)
    hmr_dir = output_dir / "hmr_world"
    smpl_model = load_smpl_neutral_model()

    hmr_entries = _discover_hmr_entries(hmr_dir)
    hmr_by_pid: dict[str, list[tuple[str, Path, Path]]] = {}
    for shot_id, pid, npz_path, kp2d_path in hmr_entries:
        hmr_by_pid.setdefault(pid, []).append((shot_id, npz_path, kp2d_path))

    if args.players:
        requested = [p.strip() for p in args.players.split(",") if p.strip()]
    else:
        refined_dir = output_dir / "refined_poses"
        refined_pids = {
            p.name.removesuffix("_refined.npz")
            for p in (refined_dir.glob("*_refined.npz") if refined_dir.exists() else [])
        }
        if refined_pids:
            # Intersect with hmr_world player ids so a stale refined npz
            # left over from a different run (no matching hmr_world
            # sidecar) is silently excluded from auto-discovery rather
            # than reported as if it were current.
            requested = sorted(refined_pids & set(hmr_by_pid.keys()))
        else:
            requested = sorted(hmr_by_pid.keys())

    results: dict = {"output_dir": str(output_dir), "stage": args.stage, "players": {}}

    for pid in requested:
        player_result: dict = {}

        if args.stage in ("refined", "both"):
            m = eval_refined_player(output_dir, pid, smpl_model)
            if m is None:
                print(f"{pid}: refined_poses npz missing or empty — skipped")
            else:
                player_result["refined"] = m

        if args.stage in ("hmr", "both"):
            entries = hmr_by_pid.get(pid)
            if not entries:
                print(f"{pid}: no hmr_world sidecar — skipped (stale refined npz?)")
            else:
                for shot_id, npz_path, kp2d_path in entries:
                    m = eval_hmr_player(output_dir, shot_id, pid, npz_path, kp2d_path, smpl_model)
                    if m is not None:
                        key = f"hmr[{shot_id}]" if shot_id else "hmr"
                        player_result[key] = m

        if player_result:
            results["players"][pid] = player_result
            print(f"\n{pid}:")
            for key, m in player_result.items():
                print(_fmt_player_line(key, m))

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")

    return results


if __name__ == "__main__":
    main()

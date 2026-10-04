"""Retime a slow-motion replay to real time, remapping its sidecars.

A replay playing at ``rate`` (live frames per replay frame, e.g. 0.34) is
resampled so new frame ``k`` is native frame ``round(k / rate)`` for
``k in 0..floor((N - 1) * rate)``. Tracks and the camera track are remapped
frame-for-frame (entries whose native frame is selected are kept and
renumbered; the rest are dropped) - no re-solve. Natives are preserved in
``shots/native/``, ``tracks/native/`` and ``camera/native/`` and every retime
starts from them, so repeated retimes never compound.

See docs/superpowers/specs/2026-10-04-replay-speed.md.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from src.schemas.shots import ShotsManifest

_MAX_RATE = 20.0


@dataclass(frozen=True)
class RetimeResult:
    shot_id: str
    rate: float
    n_native: int
    n_new: int
    frame_map: list[int]     # new frame k -> native frame
    tracks_remapped: bool
    camera_remapped: bool
    files_written: list[str] = field(default_factory=list)


def _paths(output_dir: Path, shot_id: str, clip_file: str) -> dict[str, Path]:
    clip = output_dir / clip_file
    return {
        "clip": clip,
        "clip_native": output_dir / "shots" / "native" / f"{shot_id}{clip.suffix or '.mp4'}",
        "tracks": output_dir / "tracks" / f"{shot_id}_tracks.json",
        "tracks_native": output_dir / "tracks" / "native" / f"{shot_id}_tracks.json",
        "camera": output_dir / "camera" / f"{shot_id}_camera_track.json",
        "camera_native": output_dir / "camera" / "native" / f"{shot_id}_camera_track.json",
    }


def _write_json(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc))
    tmp.replace(path)


def _frame_map(n_native: int, rate: float) -> list[int]:
    n_new = int(math.floor((n_native - 1) * rate + 1e-9)) + 1
    return [min(n_native - 1, int(round(k / rate))) for k in range(n_new)]


def _encode(src: Path, dst: Path, native_idx: list[int], fps: float) -> None:
    """Decode ``src`` sequentially, pipe the selected frames (repeats allowed,
    indices non-decreasing) to ffmpeg as H.264 yuv420p without audio."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
        "-i", "-", "-an",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(dst),
    ]
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        i = -1
        frame = None
        for n in native_idx:
            while i < n:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"{src} ended at frame {i + 1}, needed {n}")
                i += 1
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err[-400:]}")
    except BaseException:
        proc.kill()
        dst.unlink(missing_ok=True)
        raise
    finally:
        cap.release()


def _remap_frames(entries: list[dict], native_to_new: dict[int, list[int]]) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        for k in native_to_new.get(int(e["frame"]), ()):
            out.append({**e, "frame": k})
    out.sort(key=lambda e: e["frame"])
    return out


def _remap_tracks(native: Path, dst: Path, native_to_new: dict[int, list[int]]) -> None:
    doc = json.loads(native.read_text())
    tracks = []
    for tr in doc.get("tracks", []):
        frames = _remap_frames(tr.get("frames", []), native_to_new)
        if frames:
            tracks.append({**tr, "frames": frames})
    _write_json(dst, {**doc, "tracks": tracks})


def _remap_camera(native: Path, dst: Path, native_to_new: dict[int, list[int]]) -> None:
    doc = json.loads(native.read_text())
    _write_json(dst, {**doc, "frames": _remap_frames(doc.get("frames", []), native_to_new)})


def _shot_and_manifest(output_dir: Path, shot_id: str):
    path = output_dir / "shots" / "shots_manifest.json"
    manifest = ShotsManifest.load(path)
    for s in manifest.shots:
        if s.id == shot_id:
            return path, manifest, s
    raise KeyError(f"unknown shot {shot_id!r}")


def _replace_shot(manifest: ShotsManifest, shot_id: str, **changes) -> ShotsManifest:
    from dataclasses import replace

    return replace(manifest, shots=[replace(s, **changes) if s.id == shot_id else s
                                    for s in manifest.shots])


def _ensure_native(src: Path, native: Path, *, move: bool) -> None:
    if native.exists() or not src.exists():
        return
    native.parent.mkdir(parents=True, exist_ok=True)
    (shutil.move if move else shutil.copy2)(str(src), str(native))


def retime_shot(output_dir: Path, shot_id: str, rate: float) -> RetimeResult:
    """Resample ``shot_id`` to real time from its native clip."""
    rate = float(rate)
    if not math.isfinite(rate) or not 0.0 < rate <= _MAX_RATE:
        raise ValueError(f"rate must be in (0, {_MAX_RATE}]; got {rate}")
    output_dir = Path(output_dir)
    manifest_path, manifest, shot = _shot_and_manifest(output_dir, shot_id)
    p = _paths(output_dir, shot_id, shot.clip_file)

    # Never lose the original: the first retime moves the clip aside.
    if not p["clip_native"].exists():
        if not p["clip"].exists():
            raise FileNotFoundError(p["clip"])
        _ensure_native(p["clip"], p["clip_native"], move=True)
    cap = cv2.VideoCapture(str(p["clip_native"]))
    n_native = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    clip_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if n_native <= 0:
        raise RuntimeError(f"native clip has no frames: {p['clip_native']}")
    fps = float(manifest.fps) if manifest.fps and manifest.fps > 0 else (clip_fps or 25.0)

    idx = _frame_map(n_native, rate)
    tmp = p["clip"].with_name(p["clip"].stem + ".retime.tmp" + p["clip"].suffix)
    _encode(p["clip_native"], tmp, idx, fps)
    tmp.replace(p["clip"])

    native_to_new: dict[int, list[int]] = {}
    for k, n in enumerate(idx):
        native_to_new.setdefault(n, []).append(k)
    done = {}
    for key, fn in (("tracks", _remap_tracks), ("camera", _remap_camera)):
        _ensure_native(p[key], p[f"{key}_native"], move=False)
        src = p[f"{key}_native"]
        done[key] = src.exists()
        if done[key]:
            fn(src, p[key], native_to_new)

    manifest = _replace_shot(manifest, shot_id, speed_factor=1.0 / rate,
                             retimed=True, native_frames=n_native)
    manifest.save(manifest_path)
    written = [shot.clip_file] + [
        str(p[k].relative_to(output_dir)) for k in ("tracks", "camera") if done[k]]
    return RetimeResult(shot_id, rate, n_native, len(idx), idx,
                        done["tracks"], done["camera"], written)


def restore_native(output_dir: Path, shot_id: str) -> bool:
    """Put the native clip, tracks and camera track back. Natives are kept.
    Returns False when the shot was never retimed (nothing to restore)."""
    output_dir = Path(output_dir)
    manifest_path, manifest, shot = _shot_and_manifest(output_dir, shot_id)
    p = _paths(output_dir, shot_id, shot.clip_file)
    if not p["clip_native"].exists():
        return False
    shutil.copy2(p["clip_native"], p["clip"])
    for key in ("tracks", "camera"):
        if p[f"{key}_native"].exists():
            shutil.copy2(p[f"{key}_native"], p[key])
    _replace_shot(manifest, shot_id, speed_factor=1.0, retimed=False).save(manifest_path)
    return True

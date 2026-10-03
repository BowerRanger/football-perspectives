"""Helpers behind ``src/stages/shorts.py`` (the stage stays a thin orchestrator).

* ``inputs_digest`` / ``pass_fingerprint`` -- content hash of everything a
  rendered 9:16 pass depends on (spec, camera/ball/pose inputs, resolved style
  incl. kits + dressing, quality, render-code), so re-runs skip unchanged passes.
* ``make_framing_check`` -- the ``framing_check`` callback of
  ``shorts_templates.resolve_template`` built on REAL virtual-camera tracks
  (``render_pass_runner.build_camera_track``) + refined-poses root positions.
* ``audio_plan`` -- source windows + strike/impact events (on the output
  timeline, via ``short_compositor.scene_frame_to_out_s``) for
  ``shorts_audio.build_audio``.
* ``resolve_captions`` -- config / operator caption overrides.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable, Mapping, Sequence

from src.utils import render_pass_runner as rpr
from src.utils.short_compositor import scene_frame_to_out_s, segment_duration_s
from src.utils.shorts_framing import (
    FramingFailure, FramingLimits, FramingResult, camera_arrays_from_track, check_framing)
from src.utils.shorts_moments import _load_root_xy

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_CODE_FILES = (
    _ROOT / "scripts" / "blender_render_scene.py",
    _ROOT / "scripts" / "blender_stadium.py",
    _ROOT / "src" / "utils" / "render_look.py",
    _ROOT / "src" / "utils" / "virtual_cameras.py",
)


def _file_sha(path: Path, h) -> None:
    h.update(path.name.encode())
    if path.exists():
        h.update(hashlib.sha256(path.read_bytes()).digest())
    else:
        h.update(b"<missing>")


def inputs_digest(output_dir: Path, shot: str, extra_files: Sequence[Path] = ()) -> str:
    """Hash of the on-disk inputs every pass of ``shot`` is built from."""
    out = Path(output_dir)
    h = hashlib.sha256()
    files = [out / "camera" / f"{shot}_camera_track.json",
             out / "ball" / f"{shot}_ball_track.json",
             out / "players.json",
             out / "appearance" / "kits_operator.json",
             *sorted((out / "refined_poses").glob("*_refined.npz")),
             *_CODE_FILES, *extra_files]
    for p in files:
        _file_sha(p, h)
    return h.hexdigest()


def pass_fingerprint(spec: Mapping, *, shot: str, quality: Mapping, digest: str,
                     style_payload: Mapping) -> str:
    """16-hex fingerprint of one render pass."""
    blob = json.dumps({"spec": spec, "shot": shot, "quality": dict(quality),
                       "inputs": digest, "style": style_payload},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def make_framing_check(output_dir: Path, shot: str, cfg: dict, quality: Mapping,
                       base_limits: FramingLimits | None = None) -> Callable:
    """``framing_check(spec, cut_from, cut_to, subject, exclude, overrides)`` on real
    camera tracks. Broadcast passes (no synthesised camera) are not checked."""
    from src.stages.export import _per_shot_smpl_tracks

    out = Path(output_dir)
    broadcast = rpr._load_broadcast_camera(out, shot)
    tracks = {t.player_id: t for t in _per_shot_smpl_tracks(out, shot_id=shot or None)}
    ball = rpr._load_ball_track(out, shot)
    ball_xyz = {int(f.frame): tuple(f.world_xyz) for f in (ball.frames if ball else ())
                if f.world_xyz is not None}
    players = _load_root_xy(out, sorted(tracks))
    size = (int(quality["width"]), int(quality["height"]))
    limits0 = base_limits or FramingLimits()

    def check(spec, cut_from, cut_to, subject, exclude, overrides):
        cam_id = spec["camera"]
        if cam_id == "broadcast":
            return None
        try:
            track = rpr.build_camera_track(
                cam_id, rpr._rig_config(cfg, spec.get("rig") or {}), tracks, ball,
                size, float(broadcast.fps), shot or "clip")
            cam = camera_arrays_from_track(track)
        except (ValueError, RuntimeError, KeyError) as exc:
            return FramingResult(False, (FramingFailure(
                "camera_build_failed", (cut_from, cut_to), str(exc)),), {})
        return check_framing(cam, ball_xyz, players, cut_from, cut_to,
                             subject_pid=subject, exclude_pids=exclude,
                             limits=limits0.merged(overrides))
    return check


def audio_plan(edl: Mapping, shot_fps: float, moments: Mapping
               ) -> tuple[list[list[float]], list[tuple[str, float]], float]:
    """(src_windows, events, duration_s) for ``shorts_audio.build_audio``.

    One window per segment: source seconds ``[from, to)/shot_fps`` stretched to
    the segment's output duration (slow-mo / freeze-hold loop the crowd).
    Events: a strike thump / net hit + roar at every output time the strike /
    impact scene frame is on screen.
    """
    fps = float(edl.get("fps", 30))
    windows, total = [], 0.0
    for seg in edl["segments"]:
        dur = segment_duration_s(seg, fps)
        t0, t1 = seg["from"] / shot_fps, seg["to"] / shot_fps
        stretch = dur / max(t1 - t0, 1e-6)
        windows.append([t0, t1, stretch] if stretch > 1.0 + 1e-6 else [t0, t1])
        total += dur
    events: list[tuple[str, float]] = []
    for kind in ("strike", "impact"):
        frame = moments.get(kind)
        if frame is not None:
            events += [(kind, t) for t in scene_frame_to_out_s(dict(edl), frame, fps)]
    return windows, sorted(events, key=lambda e: e[1]), total


def resolve_captions(template: Mapping, cfg_caption, operator_captions: list | None) -> list | None:
    """Caption override: operator list > config (str replaces the first caption's text,
    list replaces all) > ``None`` (template's own)."""
    if operator_captions:
        return list(operator_captions)
    if not cfg_caption:
        return None
    if isinstance(cfg_caption, str):
        caps = [dict(c) for c in template.get("captions") or []]
        if not caps:
            return [{"text": cfg_caption, "style": "title", "start": 0}]
        caps[0]["text"] = cfg_caption
        return caps
    return [dict(c) for c in cfg_caption]

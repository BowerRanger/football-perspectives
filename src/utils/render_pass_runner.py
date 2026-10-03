"""Render pass runner: build a virtual-camera track, assemble and run the
headless Blender command, and post-process (slow-mo, thumbnails) for ONE
render pass.

Extracted from ``scripts/render_experiments.py`` (which is now a consumer) so
the render stage, the shorts stage and the experiments CLI share one
implementation.  A ``pass_spec`` is a ``validate_experiment`` entry (keys:
id, camera, frames, vertical, style, rig, speed, time_stretch, ...).

Nothing here touches ``<output>/render/`` unless the caller says so: the
Blender ``--render-root`` is derived from ``out_dir``.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.pipeline.config import _deep_merge
from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraTrack
from src.stages.export import _per_shot_smpl_tracks
from src.utils import virtual_cameras as vcam
from src.utils.ffmpeg import extract_thumbnail

logger = logging.getLogger(__name__)

_BLENDER_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "blender_render_scene.py"
_DEFAULT_FPS = 25.0

QUALITY_PRESETS: dict[str, dict[str, int]] = {
    "draft": {"width": 960, "height": 540, "samples": 8},
    "clean": {"width": 1920, "height": 1080, "samples": 16},
}

NO_ARG_RIGS = frozenset({"drone", "orbit", "chase", "dolly"})
PLAYER_RIGS = frozenset({"pov", "ots", "eyes"})
SIDE_RIGS = frozenset({"goal", "goalline"})
_NO_ARG_RIGS, _PLAYER_RIGS, _SIDE_RIGS = NO_ARG_RIGS, PLAYER_RIGS, SIDE_RIGS


# --- Camera id parsing / dispatch ---------------------------------------

def parse_camera_id(cam_id: str) -> tuple[str, str | None]:
    """Split a validated camera id into ``(rig, arg)`` — ``arg`` is the
    player id for pov/ots, the side for goal/goalline, else ``None``."""
    if cam_id in ("broadcast",) or cam_id in _NO_ARG_RIGS:
        return cam_id, None
    if ":" in cam_id:
        rig, _, arg = cam_id.partition(":")
        return rig, arg
    raise ValueError(f"unrecognised camera id {cam_id!r}")


def render_root_for(exp_id: str) -> str:
    return f"render_experiments/{exp_id}"


def camera_track_dir(output_dir: Path, exp_id: str, shot: str) -> Path:
    shot_dir = shot or "clip"
    return output_dir / render_root_for(exp_id) / shot_dir / "cameras"


def camera_track_path(output_dir: Path, exp_id: str, shot: str, cam_id: str) -> Path:
    safe_id = cam_id.replace(":", "_")
    return camera_track_dir(output_dir, exp_id, shot) / f"{safe_id}_camera_track.json"


def build_camera_track(
    cam_id: str,
    rig_cfg: "vcam.RigConfig",
    tracks_by_pid: dict,
    ball_track: object,
    image_size: tuple[int, int],
    fps: float,
    clip_id: str,
) -> CameraTrack:
    """Dispatch ``cam_id`` to the matching ``virtual_cameras.build_*_track``
    builder.

    Signatures assumed for the NEW rigs (not yet landed as of writing —
    instance A is adding them to ``src/utils/virtual_cameras.py`` in
    parallel):

    * ``build_goal_track(side, tracks, ball_track, cfg, image_size, fps, clip_id)``
    * ``build_goalline_track(side, tracks, ball_track, cfg, image_size, fps, clip_id)``
    * ``build_orbit_track(tracks, ball_track, cfg, image_size, fps, clip_id)``
    * ``build_chase_track(tracks, ball_track, cfg, image_size, fps, clip_id)``
    * ``build_dolly_track(tracks, ball_track, cfg, image_size, fps, clip_id)``

    i.e. the goal/goalline pair takes the same ``side`` string this
    module's camera-id grammar carries (``"left"``/``"right"``) as a
    leading positional, mirroring ``build_pov_track``/``build_ots_track``
    taking their player track first; orbit/chase/dolly mirror
    ``build_drone_track``'s ``(tracks, ball_track, cfg, ...)`` shape
    exactly (whole-scene cameras, no single-target argument).

    Raises ``RuntimeError`` with a clear message if the expected builder
    isn't present on the ``virtual_cameras`` module yet.
    """
    rig, arg = parse_camera_id(cam_id)
    if rig == "broadcast":
        raise ValueError("'broadcast' does not need a synthesised camera track")

    builder_name = f"build_{rig}_track"
    builder = getattr(vcam, builder_name, None)
    if builder is None:
        raise RuntimeError(
            f"virtual_cameras.{builder_name} not found for camera id "
            f"{cam_id!r} — this rig's builder hasn't landed in "
            "src/utils/virtual_cameras.py yet"
        )

    if rig in _PLAYER_RIGS:
        track = tracks_by_pid.get(arg)
        if track is None:
            raise ValueError(f"no player track for {arg!r} (camera {cam_id!r})")
        if rig == "pov":
            return builder(track, rig_cfg, image_size, fps, clip_id)
        return builder(track, ball_track, rig_cfg, image_size, fps, clip_id)

    if rig == "drone":
        return builder(list(tracks_by_pid.values()), ball_track, rig_cfg,
                        image_size, fps, clip_id)

    if rig in _SIDE_RIGS:
        return builder(arg, list(tracks_by_pid.values()), ball_track, rig_cfg,
                        image_size, fps, clip_id)

    if rig in _NO_ARG_RIGS:  # orbit / chase / dolly
        return builder(list(tracks_by_pid.values()), ball_track, rig_cfg,
                        image_size, fps, clip_id)

    raise ValueError(f"unhandled camera rig {rig!r} for id {cam_id!r}")


# dataclasses.field.type is the string annotation (RigConfig is defined
# under `from __future__ import annotations`), not the live type object —
# this maps the handful of annotation spellings RigConfig actually uses.
_RIG_CONFIG_CASTERS = {"float": float, "int": int, "bool": bool, "str": str}


def _rig_config(cfg: dict, overrides: dict) -> "vcam.RigConfig":
    """Base RigConfig from config/default.yaml's export.virtual_cameras
    block (same source RenderStage._virtual_camera_cfg reads), then an
    experiment's ``rig:`` overrides applied via dataclasses.replace.

    Built generically off ``vcam.RigConfig``'s own field list/defaults
    (rather than a hand-duplicated field-by-field constructor call) so it
    never goes stale as rigs are added — e.g. instance A's goal/goalline/
    orbit/chase/dolly fields are picked up automatically, config value
    and all, without this module needing an edit.
    """
    raw = ((cfg.get("export", {}) or {}).get("virtual_cameras", {})) or {}
    kwargs = {}
    for f in dataclasses.fields(vcam.RigConfig):
        if f.name in raw:
            caster = _RIG_CONFIG_CASTERS.get(f.type, lambda x: x)
            kwargs[f.name] = caster(raw[f.name])
    base = vcam.RigConfig(**kwargs)
    if not overrides:
        return base
    try:
        return dataclasses.replace(base, **overrides)
    except TypeError as exc:
        valid = sorted(f.name for f in dataclasses.fields(base))
        raise ValueError(
            f"rig override keys {sorted(overrides)} don't all match "
            f"RigConfig fields; valid fields are {valid} ({exc})"
        ) from exc


def _load_broadcast_camera(output_dir: Path, shot: str) -> CameraTrack:
    prefix = f"{shot}_" if shot else ""
    path = output_dir / "camera" / f"{prefix}camera_track.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no broadcast camera track at {path}; run the camera stage first")
    return CameraTrack.load(path)


def _load_ball_track(output_dir: Path, shot: str) -> BallTrack | None:
    prefix = f"{shot}_" if shot else ""
    path = output_dir / "ball" / f"{prefix}ball_track.json"
    return BallTrack.load(path) if path.exists() else None


def write_camera_track(
    cam_id: str, dest_path: Path, *, output_dir: Path, shot: str,
    rig_overrides: dict, cfg: dict, image_size: tuple[int, int],
) -> CameraTrack:
    """Build one virtual-camera track and save it (CameraTrack.save — the
    same writer the camera/render stages use) to ``dest_path``."""
    broadcast = _load_broadcast_camera(output_dir, shot)
    rig_cfg = _rig_config(cfg, rig_overrides)
    tracks_by_pid = {
        t.player_id: t
        for t in _per_shot_smpl_tracks(output_dir, shot_id=shot or None)
    }
    ball_track = _load_ball_track(output_dir, shot)
    clip_id = shot or "clip"
    track = build_camera_track(
        cam_id, rig_cfg, tracks_by_pid, ball_track, image_size,
        float(broadcast.fps), clip_id)
    if not track.frames:
        raise ValueError(f"built an empty camera track for {cam_id!r}")
    track.save(dest_path)
    return track


# --- Style payload --------------------------------------------------------

def merge_style_payload(
    base_style: dict, base_teams: dict, exp_style: dict | None
) -> dict:
    """Same ``--style-json`` shape RenderStage assembles (style keys +
    ``teams`` alongside them), with the experiment's ``style:`` override
    (if any) deep-merged on top — a partial override (e.g. only
    ``palette.grass_light``) leaves everything else at the config
    default."""
    payload = copy.deepcopy(base_style)
    payload["teams"] = copy.deepcopy(base_teams)
    if exp_style:
        _deep_merge(payload, exp_style)
    return payload


# --- Blender command assembly --------------------------------------------

def build_blender_command(
    *, blender_bin: str, output_dir: Path, shot: str, exp: dict,
    quality: dict, style_payload: dict, vertical_only: bool = False,
    render_root: str | None = None,
) -> list[str]:
    """Assemble the exact argv for one experiment's Blender invocation.
    Pure — no filesystem/subprocess access — so it's the same command
    both the real run and ``--dry-run`` print.

    Passes ``--render-root render_experiments/<exp_id>`` (default
    ``"render"`` once instance B lands it in
    scripts/blender_render_scene.py) so nothing here ever lands under the
    protected ``<output>/render/`` baseline."""
    cmd = [
        blender_bin, "--background", "--python", str(_BLENDER_SCRIPT), "--",
        "--output-dir", str(output_dir),
        "--shot", shot,
        "--cameras", exp["camera"],
        "--render-root", render_root or render_root_for(exp["id"]),
        "--width", str(int(quality["width"])),
        "--height", str(int(quality["height"])),
        "--samples", str(int(quality["samples"])),
        "--style-json", json.dumps(style_payload),
    ]
    if exp["frames"] is not None:
        cmd += ["--frame-start", str(exp["frames"][0]),
                "--frame-end", str(exp["frames"][1])]
    if exp.get("time_stretch", 1) > 1:
        cmd += ["--time-stretch", str(exp["time_stretch"])]
    if exp["vertical"] or vertical_only:
        cmd.append("--vertical")
    if vertical_only:
        # Blender-side flag (scripts/blender_render_scene.py): render only
        # the 9:16 pass, skipping the 16:9 one.
        cmd.append("--vertical-only")
    return cmd


# --- Blender binary resolution --------------------------------------------

def resolve_blender_binary(cfg: dict) -> str | None:
    """Same resolution order as RenderStage._resolve_blender
    (render.blender_path -> export.blender_path -> "blender" on PATH) —
    duplicated rather than imported for the same parallel-editing reason
    as ``_rig_config`` above."""
    render_cfg = cfg.get("render", {}) or {}
    export_cfg = cfg.get("export", {}) or {}
    path = render_cfg.get("blender_path") or export_cfg.get("blender_path") or "blender"
    if Path(path).is_absolute():
        return path if Path(path).exists() else None
    return shutil.which(path)


# --- Post-render: thumbnails + slow-mo ------------------------------------

def probe_duration_s(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(proc.stdout.strip())


def mid_duration_thumbnail(mp4_path: Path, thumb_path: Path) -> None:
    duration = probe_duration_s(mp4_path)
    extract_thumbnail(mp4_path, thumb_path, duration / 2.0)


def slowmo_ffmpeg_cmd(
    src: Path, dst: Path, factor: float, method: str, fps: float
) -> list[str]:
    if method == "minterpolate":
        # Motion-interpolated slow-mo: synthesises intermediate frames
        # before the timestamp restretch — expensive, clean-quality only
        # (enforced by validate_speed_for_quality before this ever runs).
        vf = f"minterpolate=fps={fps * factor:.6f}:mi_mode=mci,setpts={factor:.6f}*PTS"
    else:
        vf = f"setpts={factor:.6f}*PTS"
    return [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", vf,
        "-r", f"{fps:.6f}",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-an",
        str(dst),
    ]


def apply_slowmo(src: Path, dst: Path, factor: float, method: str, fps: float) -> None:
    subprocess.run(
        slowmo_ffmpeg_cmd(src, dst, factor, method, fps),
        check=True, capture_output=True,
    )



# --- One render pass ----------------------------------------------------

@dataclass
class PassResult:
    """Outcome of one Blender pass (``execute_pass``)."""
    blender_exit_code: int
    duration_s: float
    cmd: list[str]
    out_dir: Path
    mp4_paths: list[Path] = field(default_factory=list)
    slowmo_paths: list[Path] = field(default_factory=list)


def _render_root_for_out_dir(output_dir: Path, shot: str, exp_id: str,
                             out_dir: Path | None) -> str:
    """``--render-root`` (relative to ``output_dir``) so Blender writes
    ``<output>/<root>/<shot>/<cam>.mp4`` into ``out_dir``; falls back to the
    experiments layout when ``out_dir`` is None / not of that shape."""
    if out_dir is not None and out_dir.name == (shot or "clip"):
        try:
            return str(out_dir.parent.relative_to(output_dir))
        except ValueError:
            pass
    return render_root_for(exp_id)


def _candidates(out_dir: Path, cam_id: str, *, vertical: bool,
                vertical_only: bool) -> list[Path]:
    safe_id = cam_id.replace(":", "_")
    nine = out_dir / f"{safe_id}_9x16.mp4"
    if vertical_only and cam_id != "broadcast":
        return [nine]
    cands = [out_dir / f"{safe_id}.mp4"]
    if vertical and cam_id != "broadcast":
        cands.append(nine)
    return cands


def execute_pass(
    pass_spec: dict, *, output_dir: Path, shot: str, quality: dict,
    cfg: dict, out_dir: Path | None = None, vertical_only: bool = False,
    blender_bin: str | None = None, style_payload: dict | None = None,
    apply_speed: bool = False,
) -> PassResult:
    """Build the camera track (non-broadcast), run Blender, optionally apply
    slow-mo.  Never raises on a Blender failure — inspect
    ``blender_exit_code`` / ``mp4_paths``."""
    render_cfg = cfg.get("render", {}) or {}
    if style_payload is None:
        style_payload = merge_style_payload(
            render_cfg.get("style", {}) or {},
            render_cfg.get("teams", {}) or {},
            pass_spec.get("style"))
    render_root = _render_root_for_out_dir(output_dir, shot, pass_spec["id"], out_dir)
    shot_dir = shot or "clip"
    final_out_dir = output_dir / render_root / shot_dir
    cam_id = pass_spec["camera"]
    if cam_id != "broadcast":
        write_camera_track(
            cam_id,
            final_out_dir / "cameras" / f"{cam_id.replace(':', '_')}_camera_track.json"
            if render_root != render_root_for(pass_spec["id"])
            else camera_track_path(output_dir, pass_spec["id"], shot, cam_id),
            output_dir=output_dir, shot=shot,
            rig_overrides=pass_spec.get("rig") or {}, cfg=cfg,
            image_size=(quality["width"], quality["height"]))
    cmd = build_blender_command(
        blender_bin=blender_bin or resolve_blender_binary(cfg) or "blender",
        output_dir=output_dir, shot=shot, exp=pass_spec, quality=quality,
        style_payload=style_payload, vertical_only=vertical_only,
        render_root=render_root)
    logger.info("render_pass: %s -> %s", pass_spec["id"], cmd)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    duration_s = round(time.time() - t0, 1)
    if proc.returncode != 0:
        logger.error("render_pass: Blender failed for %s:\n%s",
                     pass_spec["id"], proc.stderr[-4000:])
    result = PassResult(proc.returncode, duration_s, cmd, final_out_dir)
    result.mp4_paths = [
        p for p in _candidates(final_out_dir, cam_id,
                               vertical=bool(pass_spec.get("vertical")),
                               vertical_only=vertical_only)
        if p.exists()]
    speed = pass_spec.get("speed")
    if apply_speed and speed is not None:
        try:
            fps = float(_load_broadcast_camera(output_dir, shot).fps) or _DEFAULT_FPS
        except FileNotFoundError:
            fps = _DEFAULT_FPS
        for mp4 in result.mp4_paths:
            dst = mp4.with_name(mp4.stem + "_slowmo.mp4")
            try:
                apply_slowmo(mp4, dst, speed["factor"], speed["method"], fps)
            except (subprocess.CalledProcessError, OSError) as exc:
                logger.warning("render_pass: slow-mo failed for %s: %s", mp4, exc)
            else:
                result.slowmo_paths.append(dst)
    return result


def render_pass(
    output_dir: Path, shot: str, pass_spec: dict, cfg: dict, quality: dict,
    out_dir: Path | None = None, vertical_only: bool = True,
) -> Path:
    """Render one pass and return the primary mp4 (the ``_9x16`` file when
    ``vertical_only`` and the camera is not broadcast).  Raises
    ``RuntimeError`` if Blender fails or produces nothing."""
    res = execute_pass(pass_spec, output_dir=Path(output_dir), shot=shot,
                       quality=quality, cfg=cfg, out_dir=out_dir,
                       vertical_only=vertical_only)
    if res.blender_exit_code != 0 or not res.mp4_paths:
        raise RuntimeError(
            f"render pass {pass_spec['id']!r} produced no output "
            f"(blender exit {res.blender_exit_code}; expected under {res.out_dir})")
    return res.mp4_paths[0]

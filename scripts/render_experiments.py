"""Render-experiment matrix runner.

Drives ``scripts/blender_render_scene.py`` once per experiment defined in a
matrix YAML (``config/render_experiments.yaml`` by default), sweeping
cameras, styles, and quality presets WITHOUT ever touching the protected
``<output>/render/`` baseline tree. Every experiment's artefacts (camera
tracks, mp4s, thumbnails, slow-mo variants) land under
``<output>/render_experiments/<exp_id>/<shot>/``.

Usage::

    .venv311/bin/python scripts/render_experiments.py \\
        --output output --shot gberch \\
        --experiments config/render_experiments.yaml \\
        [--only goal_left,orbit] [--quality draft|clean] [--dry-run]

Integration seams (this file was written against interfaces two other
in-flight agents are landing in parallel; see the docstrings on
``build_camera_track`` and ``build_blender_command`` for exactly what's
assumed):

* ``src/utils/virtual_cameras.py`` — NEW rig builders ``build_goal_track``,
  ``build_goalline_track``, ``build_orbit_track``, ``build_chase_track``,
  ``build_dolly_track`` (instance A). Looked up via ``getattr`` at call
  time, so this module imports cleanly before they land; only the
  non-dry-run path (which actually builds a track) needs them present.
* ``scripts/blender_render_scene.py`` — a new ``--render-root`` flag
  (instance B), default ``"render"``. Always passed here as
  ``render_experiments/<exp_id>``; until B lands it, only the real
  (non-dry-run) Blender invocation is affected.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import logging
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pipeline.config import _deep_merge, load_config  # noqa: E402
from src.schemas.ball_track import BallTrack  # noqa: E402
from src.schemas.camera_track import CameraTrack  # noqa: E402
from src.stages.export import _per_shot_smpl_tracks  # noqa: E402
from src.utils import virtual_cameras as vcam  # noqa: E402
from src.utils.ffmpeg import extract_thumbnail  # noqa: E402

logger = logging.getLogger(__name__)

_BLENDER_SCRIPT = Path(__file__).resolve().parent / "blender_render_scene.py"
_DEFAULT_FPS = 25.0

# --- Quality presets --------------------------------------------------
_QUALITY_PRESETS: dict[str, dict[str, int]] = {
    "draft": {"width": 960, "height": 540, "samples": 8},
    "clean": {"width": 1920, "height": 1080, "samples": 16},
}

# --- Camera id grammar --------------------------------------------------
# broadcast / drone / orbit / chase / dolly take no argument; pov/ots take
# a player id; goal/goalline (new rigs) take a side.
_CAMERA_ID_RE = re.compile(
    r"^(broadcast|drone|orbit|chase|dolly"
    r"|(?:pov|ots):[A-Za-z0-9_-]+"
    r"|(?:goal|goalline):(?:left|right))$"
)
_NO_ARG_RIGS = frozenset({"drone", "orbit", "chase", "dolly"})
_PLAYER_RIGS = frozenset({"pov", "ots"})
_SIDE_RIGS = frozenset({"goal", "goalline"})


# --- Matrix YAML loading + validation -----------------------------------

def normalize_speed(raw: object) -> dict | None:
    """Normalize the ``speed`` field to ``{"factor": float, "method": str}``
    or ``None``. Accepts a bare number (shorthand for ``{"factor": N}``) or
    a mapping. Raises ``ValueError`` on anything malformed."""
    if raw is None:
        return None
    if isinstance(raw, bool):  # bool is an int subclass — reject explicitly
        raise ValueError(f"speed must be a number or a mapping, got {raw!r}")
    if isinstance(raw, (int, float)):
        raw = {"factor": float(raw)}
    if not isinstance(raw, dict):
        raise ValueError(f"speed must be a number or a mapping, got {raw!r}")
    factor = raw.get("factor")
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or factor <= 0:
        raise ValueError(f"speed.factor must be a positive number, got {factor!r}")
    method = raw.get("method", "setpts")
    if method not in ("setpts", "minterpolate"):
        raise ValueError(
            f"speed.method must be 'setpts' or 'minterpolate', got {method!r}")
    return {"factor": float(factor), "method": method}


def validate_speed_for_quality(speed: dict | None, quality_name: str) -> None:
    """``minterpolate`` is frame-interpolated (expensive) slow-mo — only
    allowed at ``clean`` quality; ``setpts`` (a cheap timestamp restretch)
    is fine at any quality."""
    if speed is not None and speed["method"] == "minterpolate" and quality_name != "clean":
        raise ValueError(
            "speed.method='minterpolate' is only allowed at --quality clean "
            f"(got --quality {quality_name!r}); use method='setpts' for "
            "draft, or rerun this experiment with --quality clean"
        )


def validate_experiment(raw: object) -> dict:
    """Validate + normalize one experiment entry. Returns a new dict with
    every optional field defaulted (never mutates ``raw``)."""
    if not isinstance(raw, dict):
        raise ValueError(f"experiment entry must be a mapping, got {raw!r}")

    exp_id = raw.get("id")
    if not isinstance(exp_id, str) or not exp_id:
        raise ValueError(f"experiment missing a non-empty string 'id': {raw!r}")

    camera = raw.get("camera")
    if not isinstance(camera, str) or not _CAMERA_ID_RE.match(camera):
        raise ValueError(
            f"experiment {exp_id!r}: invalid camera id {camera!r}; expected "
            "'broadcast', 'drone', 'orbit', 'chase', 'dolly', 'pov:<pid>', "
            "'ots:<pid>', 'goal:left|right' or 'goalline:left|right'"
        )

    rig = raw.get("rig") or {}
    if not isinstance(rig, dict):
        raise ValueError(f"experiment {exp_id!r}: 'rig' must be a mapping")

    style = raw.get("style")
    if style is not None and not isinstance(style, dict):
        raise ValueError(f"experiment {exp_id!r}: 'style' must be a mapping")

    style_name = raw.get("style_name")
    if style_name is not None and not isinstance(style_name, str):
        raise ValueError(f"experiment {exp_id!r}: 'style_name' must be a string")

    frames = raw.get("frames")
    if frames is not None:
        if (not isinstance(frames, (list, tuple)) or len(frames) != 2
                or not all(isinstance(f, int) and not isinstance(f, bool) for f in frames)):
            raise ValueError(
                f"experiment {exp_id!r}: 'frames' must be a 2-item [start, end] "
                f"list of ints, got {frames!r}")
        if frames[0] > frames[1]:
            raise ValueError(
                f"experiment {exp_id!r}: frames start > end ({frames!r})")
        frames = [int(frames[0]), int(frames[1])]

    vertical = raw.get("vertical", False)
    if not isinstance(vertical, bool):
        raise ValueError(f"experiment {exp_id!r}: 'vertical' must be a bool")

    try:
        speed = normalize_speed(raw.get("speed"))
    except ValueError as exc:
        raise ValueError(f"experiment {exp_id!r}: {exc}") from exc

    return {
        "id": exp_id,
        "camera": camera,
        "rig": dict(rig),
        "style": dict(style) if style is not None else None,
        "style_name": style_name,
        "frames": frames,
        "vertical": vertical,
        "speed": speed,
    }


def load_experiments(path: Path) -> list[dict]:
    """Load + validate the matrix YAML at ``path``. Raises ``ValueError``
    (with the offending entry's id where known) on any schema violation,
    including duplicate ids."""
    path = Path(path)
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict) or "experiments" not in data:
        raise ValueError(f"{path}: expected a top-level 'experiments' list")
    raw_list = data["experiments"]
    if not isinstance(raw_list, list):
        raise ValueError(f"{path}: 'experiments' must be a list")

    experiments = [validate_experiment(e) for e in raw_list]
    seen: set[str] = set()
    for exp in experiments:
        if exp["id"] in seen:
            raise ValueError(f"{path}: duplicate experiment id {exp['id']!r}")
        seen.add(exp["id"])
    return experiments


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

def resolve_style_payload(
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
    quality: dict, style_payload: dict,
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
        "--render-root", render_root_for(exp["id"]),
        "--width", str(int(quality["width"])),
        "--height", str(int(quality["height"])),
        "--samples", str(int(quality["samples"])),
        "--style-json", json.dumps(style_payload),
    ]
    if exp["frames"] is not None:
        cmd += ["--frame-start", str(exp["frames"][0]),
                "--frame-end", str(exp["frames"][1])]
    if exp["vertical"]:
        cmd.append("--vertical")
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


# --- Manifest merge --------------------------------------------------------

def merge_manifest(output_dir: Path, new_entries: dict) -> dict:
    """Merge ``new_entries`` into ``<output>/render_experiments/manifest.json``
    — same merge-never-overwrite posture as RenderStage's
    render_timings.json: existing keys this run didn't touch survive,
    keys this run touched are replaced, and a corrupt existing file is
    logged and treated as empty rather than crashing the runner."""
    manifest_path = output_dir / "render_experiments" / "manifest.json"
    existing: dict = {}
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text())
            if not isinstance(loaded, dict):
                raise ValueError("manifest.json root must be an object")
            existing = loaded
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            logger.warning(
                "render_experiments: malformed existing manifest.json (%s); "
                "starting fresh", exc)
    existing.update(new_entries)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(existing, indent=2))
    return existing


# --- Orchestration ---------------------------------------------------------

def run_experiment(
    exp: dict, *, output_dir: Path, shot: str, quality_name: str,
    cfg: dict, blender_bin: str, dry_run: bool,
) -> dict | None:
    """Run (or, if ``dry_run``, just print) one experiment. Returns a
    manifest entry dict, or ``None`` for a dry run (nothing to record)."""
    quality = _QUALITY_PRESETS[quality_name]
    base_render_cfg = cfg.get("render", {}) or {}
    style_payload = resolve_style_payload(
        base_render_cfg.get("style", {}) or {},
        base_render_cfg.get("teams", {}) or {},
        exp["style"],
    )
    cmd = build_blender_command(
        blender_bin=blender_bin, output_dir=output_dir, shot=shot, exp=exp,
        quality=quality, style_payload=style_payload,
    )

    cam_id = exp["camera"]
    shot_dir = shot or "clip"
    render_root = render_root_for(exp["id"])
    needs_track = cam_id != "broadcast"
    track_path = (
        camera_track_path(output_dir, exp["id"], shot, cam_id)
        if needs_track else None
    )

    if dry_run:
        print(f"[dry-run] experiment={exp['id']} camera={cam_id} "
              f"quality={quality_name}")
        if track_path is not None:
            print(f"  would write camera track -> {track_path}")
        print(f"  blender: {' '.join(shlex.quote(c) for c in cmd)}")
        return None

    if needs_track:
        write_camera_track(
            cam_id, track_path, output_dir=output_dir, shot=shot,
            rig_overrides=exp["rig"], cfg=cfg,
            image_size=(quality["width"], quality["height"]),
        )

    out_dir = output_dir / render_root / shot_dir
    logger.info("render_experiments: %s -> %s", exp["id"], cmd)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    duration_s = round(time.time() - t0, 1)
    if proc.returncode != 0:
        logger.error("render_experiments: Blender failed for %s:\n%s",
                      exp["id"], proc.stderr[-4000:])

    safe_id = cam_id.replace(":", "_")
    candidates = [out_dir / f"{safe_id}.mp4"]
    if exp["vertical"] and cam_id != "broadcast":
        candidates.append(out_dir / f"{safe_id}_9x16.mp4")

    fps = _DEFAULT_FPS
    try:
        fps = float(_load_broadcast_camera(output_dir, shot).fps) or _DEFAULT_FPS
    except FileNotFoundError:
        pass

    output_paths: list[str] = []
    for mp4_path in candidates:
        if not mp4_path.exists():
            continue
        output_paths.append(str(mp4_path))
        _thumbnail_and_record(mp4_path, output_paths)

        if exp["speed"] is not None:
            slowmo_path = mp4_path.with_name(mp4_path.stem + "_slowmo.mp4")
            try:
                apply_slowmo(mp4_path, slowmo_path, exp["speed"]["factor"],
                             exp["speed"]["method"], fps)
            except (subprocess.CalledProcessError, OSError) as exc:
                logger.warning(
                    "render_experiments: slow-mo failed for %s: %s",
                    mp4_path, exc)
            else:
                output_paths.append(str(slowmo_path))
                _thumbnail_and_record(slowmo_path, output_paths)

    style_name = exp["style_name"]
    if style_name is None and exp["style"] is not None:
        style_name = exp["id"]

    return {
        "id": exp["id"],
        "shot": shot,
        "camera": cam_id,
        "style_name": style_name,
        "quality": quality_name,
        "output_paths": output_paths,
        "blender_exit_code": proc.returncode,
        "duration_s": duration_s,
    }


def _thumbnail_and_record(mp4_path: Path, output_paths: list[str]) -> None:
    thumb_path = mp4_path.with_name(mp4_path.stem + "_thumb.jpg")
    try:
        mid_duration_thumbnail(mp4_path, thumb_path)
    except (subprocess.CalledProcessError, ValueError, OSError) as exc:
        logger.warning(
            "render_experiments: thumbnail failed for %s: %s", mp4_path, exc)
    else:
        output_paths.append(str(thumb_path))


# --- CLI ---------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render-experiment matrix runner",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, type=Path,
                    help="pipeline output dir (e.g. output)")
    p.add_argument("--shot", required=True, help="shot id (e.g. gberch)")
    p.add_argument("--experiments", required=True, type=Path,
                    help="matrix YAML (e.g. config/render_experiments.yaml)")
    p.add_argument("--only", default=None,
                    help="comma-separated experiment ids to run (default: all)")
    p.add_argument("--quality", choices=sorted(_QUALITY_PRESETS), default="draft")
    p.add_argument("--dry-run", action="store_true",
                    help="print the resolved Blender command per experiment "
                         "without building any camera tracks or invoking anything")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        experiments = load_experiments(args.experiments)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    selected = experiments
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        by_id = {e["id"]: e for e in experiments}
        unknown = [w for w in wanted if w not in by_id]
        if unknown:
            print(f"error: unknown experiment id(s) in --only: {unknown}",
                  file=sys.stderr)
            return 2
        selected = [by_id[w] for w in wanted]

    if not selected:
        print("error: no experiments selected", file=sys.stderr)
        return 2

    # Pre-flight: validate speed/quality gating for every selected
    # experiment before doing any work, so a config mistake surfaces
    # immediately instead of partway through a batch of renders.
    errors = []
    for exp in selected:
        try:
            validate_speed_for_quality(exp["speed"], args.quality)
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        return 2

    cfg = load_config()
    blender_bin = resolve_blender_binary(cfg)
    if blender_bin is None and not args.dry_run:
        print("error: Blender not found (render.blender_path / "
              "export.blender_path); install Blender or use --dry-run",
              file=sys.stderr)
        return 2

    manifest_entries: dict = {}
    exit_code = 0
    for exp in selected:
        try:
            entry = run_experiment(
                exp, output_dir=args.output, shot=args.shot,
                quality_name=args.quality, cfg=cfg,
                blender_bin=blender_bin or "blender", dry_run=args.dry_run,
            )
        except Exception as exc:  # noqa: BLE001 - one bad experiment must not sink the batch
            print(f"error: experiment {exp['id']} failed: {exc}", file=sys.stderr)
            manifest_entries[f"{exp['id']}:{args.shot}"] = {
                "id": exp["id"], "shot": args.shot, "error": str(exc),
            }
            exit_code = 1
            continue
        if entry is not None:
            manifest_entries[f"{exp['id']}:{args.shot}"] = entry

    if not args.dry_run and manifest_entries:
        merge_manifest(args.output, manifest_entries)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

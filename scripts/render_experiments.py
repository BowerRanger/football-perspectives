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
from src.schemas.camera_track import CameraTrack  # noqa: E402,F401
from src.utils import virtual_cameras as vcam  # noqa: E402,F401  (tests patch rex.vcam)
from src.utils.render_pass_runner import (  # noqa: E402,F401
    NO_ARG_RIGS as _NO_ARG_RIGS, PLAYER_RIGS as _PLAYER_RIGS,
    SIDE_RIGS as _SIDE_RIGS, QUALITY_PRESETS as _QUALITY_PRESETS,
    _BLENDER_SCRIPT, _DEFAULT_FPS, _load_ball_track, _load_broadcast_camera,
    _rig_config, apply_slowmo, build_blender_command, build_camera_track,
    camera_track_dir, camera_track_path, execute_pass,
    merge_style_payload as resolve_style_payload, mid_duration_thumbnail,
    parse_camera_id, probe_duration_s, render_root_for, resolve_blender_binary,
    slowmo_ffmpeg_cmd, write_camera_track,
)

logger = logging.getLogger(__name__)



# --- Camera id grammar --------------------------------------------------
# broadcast / drone / orbit / chase / dolly take no argument; pov/ots take
# a player id; goal/goalline (new rigs) take a side.
_CAMERA_ID_RE = re.compile(
    r"^(broadcast|drone|orbit|chase|dolly"
    r"|(?:pov|ots|eyes):[A-Za-z0-9_-]+"
    r"|(?:goal|goalline):(?:left|right))$"
)


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


def _validate_time_stretch(exp_id: str, raw: object) -> int:
    """``time_stretch``: render-native slow motion factor (int 1-9)."""
    if isinstance(raw, bool) or not isinstance(raw, int) or not 1 <= raw <= 9:
        raise ValueError(
            f"experiment {exp_id!r}: 'time_stretch' must be an int in 1..9, got {raw!r}")
    return raw


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
            "'ots:<pid>', 'eyes:<pid>', 'goal:left|right' or 'goalline:left|right'"
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
        "time_stretch": _validate_time_stretch(exp_id, raw.get("time_stretch", 1)),
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


# Camera/Blender/slow-mo helpers live in src/utils/render_pass_runner.py;
# re-exported above for backwards compatibility.

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

    res = execute_pass(
        exp, output_dir=output_dir, shot=shot, quality=quality, cfg=cfg,
        out_dir=output_dir / render_root / shot_dir, blender_bin=blender_bin,
        style_payload=style_payload, apply_speed=True)

    output_paths: list[str] = []
    for mp4_path in res.mp4_paths:
        output_paths.append(str(mp4_path))
        _thumbnail_and_record(mp4_path, output_paths)
        slowmo_path = mp4_path.with_name(mp4_path.stem + "_slowmo.mp4")
        if slowmo_path in res.slowmo_paths:
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
        "blender_exit_code": res.blender_exit_code,
        "duration_s": res.duration_s,
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
    p.add_argument("--config", type=Path, default=None,
                    help="clip config YAML deep-merged over config/default.yaml "
                         "(same as `recon.py run --config`) — e.g. per-match "
                         "render.teams kit colours")
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

    cfg = load_config(args.config)
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

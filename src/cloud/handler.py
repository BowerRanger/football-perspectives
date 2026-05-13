"""Container entrypoint for AWS Batch — one player track per invocation.

Two entry modes:

- ``main()`` — for ``docker run python -m src.cloud.handler``. Reads
  ``JOB_MANIFEST_S3`` and ``JOB_OUTPUT_S3_PREFIX`` env vars (and, for
  array jobs, ``AWS_BATCH_JOB_ARRAY_INDEX``). Downloads inputs from S3,
  runs the same ``process_player`` the local stage uses, uploads
  outputs + status.json to S3.

- ``run_local(manifest_path, output_dir)`` — for the ``recon.py
  batch-handler`` subcommand. Identical code path with file:// URIs.

Both paths converge on :func:`_process` which is the only place GVHMR
work happens. The handler is intentionally thin: it imports the
existing ``process_player`` from the stage module so there's exactly
one definition of "process one track".
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from src.cloud.manifest import ArrayManifest, JobManifest, JobStatus
from src.cloud.s3_io import S3Client, make_client, parse_s3_uri, uri_to_local_path
from src.schemas.camera_track import CameraTrack


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip() or None
    except Exception:
        return None


def _build_per_frame(
    cam: CameraTrack,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    tuple[float, float],
]:
    """Same per-frame K/R/t/distortion projection the stage does."""
    K = {f.frame: np.array(f.K, dtype=float) for f in cam.frames}
    R = {f.frame: np.array(f.R, dtype=float) for f in cam.frames}
    t_fb = np.array(cam.t_world, dtype=float)
    t = {
        f.frame: (np.array(f.t, dtype=float) if f.t is not None else t_fb)
        for f in cam.frames
    }
    return K, R, t, cam.distortion


def _upload_dir(client: S3Client, src_dir: Path, dest_prefix: str) -> None:
    """Upload every file under ``src_dir`` to ``dest_prefix``."""
    for file_path in src_dir.rglob("*"):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(src_dir).as_posix()
        client.upload(file_path, f"{dest_prefix.rstrip('/')}/{rel}")


def _process(
    manifest: JobManifest,
    *,
    client: S3Client,
    work_root: Path,
) -> JobStatus:
    """Download, run, return status. Does NOT upload — caller decides."""
    started = time.time()
    work = work_root / "work"
    work.mkdir(parents=True, exist_ok=True)

    video = work / "shot.mp4"
    camera = work / "camera_track.json"
    out_dir = work / "out"
    out_dir.mkdir(exist_ok=True)

    client.download(manifest.video_uri, video)
    client.download(manifest.camera_track_uri, camera)

    cam = CameraTrack.load(camera)
    per_frame_K, per_frame_R, per_frame_t, distortion = _build_per_frame(cam)

    cfg = manifest.hmr_world_cfg
    # Lazy import — process_player triggers torch + GVHMR loads.
    from src.stages.hmr_world import process_player

    status_str = process_player(
        player_id=manifest.player_id,
        shot_id=manifest.shot_id,
        track_frames=list(manifest.track_frames),
        out_dir=out_dir,
        cfg=cfg,
        per_frame_K=per_frame_K,
        per_frame_R=per_frame_R,
        per_frame_t=per_frame_t,
        distortion=distortion,
        min_track_frames=int(cfg.get("min_track_frames", 10)),
        savgol_window=int(cfg.get("theta_savgol_window", 11)),
        savgol_order=int(cfg.get("theta_savgol_order", 2)),
        slerp_w=int(cfg.get("root_slerp_window", 5)),
        ground_snap_velocity=float(cfg.get("ground_snap_velocity", 0.1)),
        root_t_savgol_window=int(cfg.get("root_t_savgol_window", 5)),
        root_t_savgol_order=int(cfg.get("root_t_savgol_order", 2)),
        lean_correction_deg=float(cfg.get("lean_correction_deg", 0.0)),
        video_path=video,
        estimator=None,   # one job per container — no reuse to do here
    )

    # Upload everything in out_dir to manifest.output_prefix/output/.
    _upload_dir(client, out_dir, f"{manifest.output_prefix.rstrip('/')}/output")

    return JobStatus(
        status=("ok" if status_str == "ran" else status_str),
        duration_seconds=time.time() - started,
        frames=len(manifest.track_frames),
        git_sha=_git_sha(),
        metadata={"process_player_status": status_str},
    )


def _resolve_array_manifest(
    manifest_uri: str,
    array_index: int | None,
    client: S3Client,
) -> str:
    """If we were given an ArrayManifest URI + an array index, dereference."""
    if array_index is None:
        return manifest_uri
    text = client.read_text(manifest_uri)
    array_manifest = ArrayManifest.from_json(text)
    if array_index >= len(array_manifest.entries):
        raise RuntimeError(
            f"AWS_BATCH_JOB_ARRAY_INDEX={array_index} out of range "
            f"(size={len(array_manifest.entries)})"
        )
    return array_manifest.entries[array_index]


def _write_failure_status(
    output_prefix: str,
    client: S3Client,
    exc: BaseException,
    started: float,
    frames: int,
) -> None:
    status = JobStatus(
        status="error",
        duration_seconds=time.time() - started,
        frames=frames,
        error_type=type(exc).__name__,
        error_message=str(exc),
        traceback=traceback.format_exc(),
        git_sha=_git_sha(),
    )
    client.upload_bytes(
        status.to_json().encode("utf-8"),
        f"{output_prefix.rstrip('/')}/status.json",
    )


def main() -> int:
    """Container entrypoint. Returns the process exit code.

    Reads ``JOB_MANIFEST_S3`` (either a single JobManifest URI or an
    ArrayManifest URI when ``AWS_BATCH_JOB_ARRAY_INDEX`` is set) and
    derives the output prefix from the manifest itself. A single-job
    invocation may override the output prefix via
    ``JOB_OUTPUT_S3_PREFIX``; array jobs ignore the env var because
    Batch can't set different values per array child.
    """
    started = time.time()
    manifest_uri = os.environ.get("JOB_MANIFEST_S3")
    if not manifest_uri:
        print("JOB_MANIFEST_S3 env var is required", file=sys.stderr)
        return 2

    array_index_raw = os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX")
    array_index = int(array_index_raw) if array_index_raw is not None else None

    client = make_client(manifest_uri)
    work_root = Path(tempfile.mkdtemp(prefix="hmr_handler_"))
    output_prefix = ""
    frames = 0
    try:
        effective_uri = _resolve_array_manifest(manifest_uri, array_index, client)
        manifest = JobManifest.from_json(client.read_text(effective_uri))
        frames = len(manifest.track_frames)
        # Single-job invocations can override; array jobs use the
        # per-index manifest's own output_prefix.
        override = os.environ.get("JOB_OUTPUT_S3_PREFIX")
        if override and array_index is None:
            manifest = JobManifest(
                run_id=manifest.run_id,
                shot_id=manifest.shot_id,
                player_id=manifest.player_id,
                video_uri=manifest.video_uri,
                camera_track_uri=manifest.camera_track_uri,
                track_frames=manifest.track_frames,
                hmr_world_cfg=manifest.hmr_world_cfg,
                output_prefix=override,
                schema_version=manifest.schema_version,
            )
        output_prefix = manifest.output_prefix
        status = _process(manifest, client=client, work_root=work_root)
        client.upload_bytes(
            status.to_json().encode("utf-8"),
            f"{output_prefix.rstrip('/')}/status.json",
        )
        return 0 if status.status in ("ok", "too_short", "cached") else 1
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        if output_prefix:
            try:
                _write_failure_status(output_prefix, client, exc, started, frames)
            except Exception:
                traceback.print_exc()
        return 1
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


def run_local(manifest_path: Path, output_dir: Path) -> JobStatus:
    """``recon.py batch-handler`` entrypoint.

    ``manifest_path`` is a local JobManifest JSON. ``output_dir`` is the
    local directory that plays the role of the S3 output prefix. The
    manifest's ``video_uri`` / ``camera_track_uri`` may be ``file://``
    URIs or bare paths.
    """
    manifest = JobManifest.from_path(manifest_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # All URIs in manifest are file:// or bare paths — LocalFSClient
    # routed at "/" handles those directly.
    client = make_client("file:///")

    # Rewrite output_prefix to point at the local output_dir.
    manifest = JobManifest(
        run_id=manifest.run_id,
        shot_id=manifest.shot_id,
        player_id=manifest.player_id,
        video_uri=manifest.video_uri,
        camera_track_uri=manifest.camera_track_uri,
        track_frames=manifest.track_frames,
        hmr_world_cfg=manifest.hmr_world_cfg,
        output_prefix=str(output_dir),
        schema_version=manifest.schema_version,
    )

    work_root = Path(tempfile.mkdtemp(prefix="hmr_handler_local_"))
    started = time.time()
    try:
        status = _process(manifest, client=client, work_root=work_root)
        (output_dir / "status.json").write_text(status.to_json())
        return status
    except Exception as exc:  # noqa: BLE001
        status = JobStatus(
            status="error",
            duration_seconds=time.time() - started,
            frames=len(manifest.track_frames),
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(),
            git_sha=_git_sha(),
        )
        (output_dir / "status.json").write_text(status.to_json())
        raise
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

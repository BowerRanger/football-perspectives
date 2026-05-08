"""Football-perspectives web server (FastAPI dashboard).

Endpoints exposed:

Pipeline control
    GET    /api/stages
    GET    /api/config
    POST   /api/run
    GET    /api/jobs/{job_id}/status
    GET    /api/jobs/{job_id}/logs
    DELETE /api/output/{stage}

Video / clips
    GET    /api/video/{shot_id}
    GET    /api/video/{shot_id}/frame
    GET    /api/output/shots          (list of available clip ids)

Anchor editor (Phase 4a — broadcast-mono)
    GET    /anchors                   (current AnchorSet)
    POST   /anchors                   (write AnchorSet)
    GET    /landmarks                 (FIFA landmark catalogue for the palette)

Pipeline output viewers
    GET    /camera/track              (camera_track.json)
    GET    /tracking/shots            (list of shot_ids with tracks)
    GET    /tracking/preview          (per-track summaries for one shot)
    GET    /tracking/frames           (per-frame bbox payload for one shot)
    GET    /hmr_world/kp2d_players   (list of player_ids with kp2d artifacts)
    GET    /hmr_world/kp2d_preview   (per-frame COCO-17 keypoints for a player)
    GET    /hmr_world/preview         (lightweight summary for one player)
    GET    /hmr_world/players         (list of available player_id strings)
    GET    /ball/preview              (ball_track.json)

Track annotation (manual edits to tracking output)
    PATCH  /api/tracks/{shot_id}/{track_id}    (rename / re-team / re-id one track)
    DELETE /api/tracks/{shot_id}/{track_id}    (remove one track)
    POST   /api/tracks/split                   (split track at a frame, mint new IDs)
    POST   /api/tracks/merge                   (unify N tracks under one player_id)
    POST   /api/tracks/merge-by-name           (group every track sharing a name)
    POST   /api/tracks/ignore-unknown/{shot_id}(mark unnamed tracks 'ignore')
    POST   /api/tracks/delete-ignored          (drop tracks named 'ignore')

Static / export
    GET    /                          (dashboard)
    GET    /viewer                    (3D viewer)
    GET    /api/export/scene.glb
    GET    /api/export/metadata
"""

import io
import json
import logging
import re
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.pipeline.config import load_config
from src.pipeline.runner import run_pipeline
from src.schemas.anchor import AnchorSet
from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraTrack
from src.schemas.tracks import Track, TrackFrame, TracksResult
from src.utils.pitch_landmarks import LANDMARK_CATALOGUE

# ---------------------------------------------------------------------------
# Stage completion (broadcast-mono)
# ---------------------------------------------------------------------------

STAGE_ORDER: list[str] = [
    "prepare_shots",
    "tracking",
    "camera",
    "hmr_world",
    "ball",
    "export",
]

_STAGE_COMPLETE = {
    "prepare_shots": lambda d: (d / "shots" / "shots_manifest.json").exists(),
    "tracking": lambda d: any((d / "tracks").glob("*_tracks.json")),
    "camera": lambda d: (d / "camera" / "camera_track.json").exists(),
    "hmr_world": lambda d: any((d / "hmr_world").glob("*_smpl_world.npz")),
    "ball": lambda d: (d / "ball" / "ball_track.json").exists(),
    "export": lambda d: (d / "export" / "gltf" / "scene.glb").exists(),
}

# Per-stage outputs that should be wiped on a "re-run" or "clear" action.
# Paths are relative to ``output_dir`` and may name files or directories.
# ``camera/anchors.json`` is deliberately omitted — it is user-supplied input
# to the camera stage, not output, and must survive a stage re-run.
_STAGE_ARTIFACTS: dict[str, list[str]] = {
    "prepare_shots": ["shots"],
    "tracking": ["tracks"],
    "camera": ["camera/camera_track.json", "camera/debug"],
    "hmr_world": ["hmr_world"],
    "ball": ["ball"],
    "export": ["export"],
}

# ---------------------------------------------------------------------------
# Job registry
# ---------------------------------------------------------------------------


@dataclass
class Job:
    job_id: str
    stages: str
    status: str = "running"  # running | done | error
    log_queue: Queue = field(default_factory=Queue)
    log_lines: list[str] = field(default_factory=list)
    error: str | None = None


_jobs: dict[str, Job] = {}
_jobs_lock = Lock()
_MAX_CONCURRENT_JOBS = 5


class _LogQueueHandler(logging.Handler):
    def __init__(self, job: Job) -> None:
        super().__init__()
        self.job = job

    def emit(self, record: logging.LogRecord) -> None:
        line = self.format(record)
        self.job.log_lines.append(line)
        self.job.log_queue.put(line)


class _QueueWriter(io.TextIOBase):
    """Captures print() output line-by-line into a job's queue."""

    def __init__(self, job: Job) -> None:
        self.job = job
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.job.log_lines.append(line)
            self.job.log_queue.put(line)
        return len(s)

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Background job runner
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    stages: str = "all"
    from_stage: str | None = None
    device: str = "auto"
    # Multi-shot: when set, every stage that iterates manifest.shots
    # will only process the named shot. Stages without a manifest
    # ignore this. Used by the dashboard's /api/run-shot endpoint.
    shot_filter: str | None = None


def _emit(job: Job, line: str) -> None:
    job.log_lines.append(line)
    job.log_queue.put(line)


def _run_job(job: Job, output_dir: Path, config_path: Path | None, params: RunRequest) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"job_{job.job_id}.log"
    log_file = log_path.open("w", buffering=1)

    class _FileTeeHandler(logging.Handler):
        def emit(self, record):
            try:
                log_file.write(self.format(record) + "\n")
            except Exception:
                pass

    file_handler = _FileTeeHandler()
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.root.addHandler(file_handler)

    handler = _LogQueueHandler(job)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    prev_root_level = logging.root.level
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(handler)

    class _TeeWriter(io.TextIOBase):
        def __init__(self, base: _QueueWriter) -> None:
            self.base = base

        def write(self, s: str) -> int:
            try:
                log_file.write(s)
            except Exception:
                pass
            return self.base.write(s)

        def flush(self) -> None:
            try:
                log_file.flush()
            except Exception:
                pass

    writer = _TeeWriter(_QueueWriter(job))
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = writer  # type: ignore[assignment]
    sys.stderr = writer  # type: ignore[assignment]
    _emit(job, f"[job {job.job_id}] starting stages={params.stages!r}")
    try:
        cfg = load_config(config_path)
        run_pipeline(
            output_dir=output_dir,
            stages=params.stages,
            from_stage=params.from_stage,
            config=cfg,
            device=params.device,
            shot_filter=params.shot_filter,
        )
        job.status = "done"
    except Exception as exc:
        import traceback

        job.status = "error"
        job.error = str(exc)
        for line in traceback.format_exc().splitlines():
            job.log_lines.append(line)
            job.log_queue.put(line)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logging.root.removeHandler(handler)
        logging.root.removeHandler(file_handler)
        logging.root.setLevel(prev_root_level)
        try:
            log_file.flush()
            log_file.close()
        except Exception:
            pass
        job.log_queue.put(None)


# ---------------------------------------------------------------------------
# SSE log stream
# ---------------------------------------------------------------------------


async def _log_stream(job: Job):
    import asyncio

    for line in list(job.log_lines):
        yield f"event: log\ndata: {json.dumps({'line': line})}\n\n"

    seen = len(job.log_lines)
    while True:
        new_count = len(job.log_lines)
        while seen < new_count:
            line = job.log_lines[seen]
            seen += 1
            yield f"event: log\ndata: {json.dumps({'line': line})}\n\n"
        if job.status in ("done", "error") and seen >= len(job.log_lines):
            break
        await asyncio.sleep(0.1)

    yield f"event: done\ndata: {json.dumps({'status': job.status})}\n\n"


# ---------------------------------------------------------------------------
# Video range response
# ---------------------------------------------------------------------------


def _parse_range(range_header: str, file_size: int) -> tuple[int, int]:
    units, _, rng = range_header.partition("=")
    start_str, _, end_str = rng.partition("-")
    start = int(start_str) if start_str else 0
    end = int(end_str) if end_str else file_size - 1
    end = min(end, file_size - 1)
    return start, end


def _range_response(path: Path, request: Request) -> Response:
    size = path.stat().st_size
    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(str(path), media_type="video/mp4")
    start, end = _parse_range(range_header, size)
    length = end - start + 1
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(length)
    return Response(
        content=data,
        status_code=206,
        media_type="video/mp4",
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
        },
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(output_dir: Path, config_path: Path | None = None) -> FastAPI:
    output_dir = output_dir.resolve()
    app = FastAPI(title="Football Perspectives Dashboard")
    app.state.output_dir = output_dir
    app.state.config_path = config_path

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ``Cache-Control: no-store`` on dashboard HTML pages — the JS in
    # these files changes during active development, and a stale cached
    # viewer.html silently sends ``?player_id=[object Object]``-style
    # requests that hit the validation regex and 400. Forcing the
    # browser to re-fetch makes JS edits visible without a hard reload.
    _NO_STORE = {"Cache-Control": "no-store"}

    @app.get("/")
    def index():
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path), headers=_NO_STORE)
        raise HTTPException(status_code=404, detail="index.html not found")

    @app.get("/api/stages")
    def get_stages():
        return [
            {
                "name": name,
                "index": i + 1,
                "complete": _STAGE_COMPLETE[name](output_dir),
            }
            for i, name in enumerate(STAGE_ORDER)
        ]

    @app.get("/api/config")
    def get_config():
        return load_config(app.state.config_path)

    @app.post("/api/run", status_code=202)
    def run_stages(params: RunRequest):
        with _jobs_lock:
            running_jobs = sum(1 for job in _jobs.values() if job.status == "running")
            if running_jobs >= _MAX_CONCURRENT_JOBS:
                raise HTTPException(status_code=429, detail="Too many concurrent jobs")
        job_id = str(uuid.uuid4())[:8]
        job = Job(job_id=job_id, stages=params.stages)
        with _jobs_lock:
            _jobs[job_id] = job
        thread = Thread(
            target=_run_job,
            args=(job, output_dir, app.state.config_path, params),
            daemon=True,
        )
        thread.start()
        return {"job_id": job_id}

    @app.get("/api/jobs/{job_id}/status")
    def job_status(job_id: str):
        with _jobs_lock:
            job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "job_id": job_id,
            "stages": job.stages,
            "status": job.status,
            "error": job.error,
        }

    @app.get("/api/jobs/{job_id}/logs")
    def stream_logs(job_id: str):
        with _jobs_lock:
            job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return StreamingResponse(
            _log_stream(job),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.delete("/api/output/{stage}")
    def delete_output(stage: str):
        import shutil

        if stage not in STAGE_ORDER:
            raise HTTPException(status_code=404, detail=f"Unknown stage: {stage}")
        removed = []
        for relpath in _STAGE_ARTIFACTS.get(stage, []):
            target = output_dir / relpath
            if not target.exists():
                continue
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            removed.append(relpath)
        return {"stage": stage, "removed": removed}

    @app.get("/api/video/{shot_id}")
    def get_video(shot_id: str, request: Request):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", shot_id):
            raise HTTPException(status_code=400, detail="Invalid shot ID")
        candidate = (output_dir / "shots" / f"{shot_id}.mp4").resolve()
        if not candidate.is_relative_to((output_dir / "shots").resolve()):
            raise HTTPException(status_code=400, detail="Invalid shot ID")
        if candidate.exists():
            return _range_response(candidate, request)
        raise HTTPException(status_code=404, detail=f"Video not found: {shot_id}")

    @app.get("/api/video/{shot_id}/frame")
    def get_frame(shot_id: str, frame_idx: int = 0):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", shot_id):
            raise HTTPException(status_code=400, detail="Invalid shot ID")
        clip_path = (output_dir / "shots" / f"{shot_id}.mp4").resolve()
        if not clip_path.exists():
            raise HTTPException(status_code=404, detail=f"Video not found: {shot_id}")

        import cv2

        cap = cv2.VideoCapture(str(clip_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                raise HTTPException(status_code=404, detail=f"Frame {frame_idx} not found")
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        finally:
            cap.release()
        return Response(content=buf.tobytes(), media_type="image/jpeg")

    @app.get("/viewer")
    def viewer_page():
        viewer_path = static_dir / "viewer.html"
        if not viewer_path.exists():
            raise HTTPException(status_code=404, detail="viewer.html not found")
        return FileResponse(str(viewer_path), headers=_NO_STORE)

    @app.get("/api/export/scene.glb")
    def get_scene_glb():
        glb_path = output_dir / "export" / "gltf" / "scene.glb"
        if not glb_path.exists():
            raise HTTPException(status_code=404, detail="scene.glb not found")
        return FileResponse(str(glb_path), media_type="model/gltf-binary")

    @app.get("/api/export/metadata")
    def get_scene_metadata():
        meta_path = output_dir / "export" / "gltf" / "scene_metadata.json"
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail="scene_metadata.json not found")
        return json.loads(meta_path.read_text())

    # ------------------------------------------------------------------
    # Anchor editor + pipeline-output viewers (broadcast-mono, Phase 4a)
    # ------------------------------------------------------------------

    @app.get("/api/output/shots")
    def list_shots():
        shots_dir = output_dir / "shots"
        if not shots_dir.exists():
            return {"shots": []}
        ids = sorted(p.stem for p in shots_dir.glob("*.mp4"))
        return {"shots": ids}

    @app.get("/api/output/shot-status/{shot_id}")
    def get_shot_status(shot_id: str):
        """Per-shot artefact-existence summary used by the dashboard's
        multi-shot status panel. ``camera_stale`` flags the case where
        the camera_track exists but its mtime predates the anchors —
        i.e. the user edited anchors after the last camera solve."""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", shot_id):
            raise HTTPException(status_code=400, detail="Invalid shot_id")
        cam_path = output_dir / "camera" / f"{shot_id}_camera_track.json"
        anchors_path = output_dir / "camera" / f"{shot_id}_anchors.json"
        has_camera = cam_path.exists()
        has_anchors = anchors_path.exists()
        camera_stale = (
            has_camera and has_anchors
            and cam_path.stat().st_mtime < anchors_path.stat().st_mtime
        )
        anchor_count = 0
        if has_anchors:
            try:
                anchor_count = len(AnchorSet.load(anchors_path).anchors)
            except Exception:
                anchor_count = 0
        hmr_dir = output_dir / "hmr_world"
        hmr_count = 0
        if hmr_dir.exists():
            for npz in hmr_dir.glob("*_smpl_world.npz"):
                try:
                    z = np.load(npz, allow_pickle=False)
                    sid = str(z["shot_id"]) if "shot_id" in z.files else ""
                except Exception:
                    sid = ""
                if sid == shot_id:
                    hmr_count += 1
        return {
            "shot_id": shot_id,
            "has_anchors": has_anchors,
            "anchor_count": anchor_count,
            "has_camera": has_camera,
            "camera_stale": camera_stale,
            "has_hmr": hmr_count > 0,
            "hmr_player_count": hmr_count,
            "has_ball": (output_dir / "ball" / f"{shot_id}_ball_track.json").exists(),
            "has_export": (output_dir / "export" / "gltf" / f"{shot_id}_scene.glb").exists(),
        }

    @app.get("/landmarks")
    def get_landmarks():
        return {
            "landmarks": [
                {"name": lm.name, "world_xyz": list(lm.world_xyz)}
                for lm in sorted(LANDMARK_CATALOGUE.values(), key=lambda lm: lm.name)
            ]
        }

    @app.get("/pitch_lines")
    def get_pitch_lines(stadium: str | None = None):
        from src.utils.pitch_lines_catalogue import (
            LINE_CATALOGUE,
            VANISHING_LINE_CATALOGUE,
        )
        from src.utils.stadium_config import load_stadiums, mow_stripe_lines

        merged: list[dict[str, Any]] = []
        for name, seg in LINE_CATALOGUE.items():
            merged.append({"name": name, "world_segment": [list(seg[0]), list(seg[1])]})
        for name, d in VANISHING_LINE_CATALOGUE.items():
            merged.append({"name": name, "world_direction": list(d)})
        # When a stadium is supplied, append its dynamic mow-stripe entries
        # so the editor's Lines palette shows them. Unknown stadium ids
        # are silently ignored — the user just gets the static catalogue.
        if stadium:
            registry = load_stadiums()
            cfg = registry.get(stadium)
            if cfg is not None:
                for name, seg in mow_stripe_lines(cfg).items():
                    merged.append({
                        "name": name,
                        "world_segment": [list(seg[0]), list(seg[1])],
                        "category": "mowing",
                    })
        merged.sort(key=lambda x: x["name"])
        return {"lines": merged}

    @app.get("/stadiums")
    def get_stadiums():
        """Registry of available stadium ids → display names.

        Drives the anchor editor's stadium dropdown. Returns an empty
        list when the registry YAML is missing or empty (clips work
        unchanged, just without dynamic mow stripes).
        """
        from src.utils.stadium_config import load_stadiums

        registry = load_stadiums()
        return {
            "stadiums": [
                {"id": cfg.id, "display_name": cfg.display_name}
                for cfg in sorted(registry.values(), key=lambda c: c.display_name)
            ]
        }

    class AnchorPayload(BaseModel):
        clip_id: str
        image_size: tuple[int, int]
        anchors: list[dict[str, Any]]
        stadium: str | None = None

    def _first_shot_id() -> str | None:
        """Return the manifest's first shot_id, or None when no manifest."""
        from src.schemas.shots import ShotsManifest
        manifest_path = output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = ShotsManifest.load(manifest_path)
        except Exception:
            return None
        return manifest.shots[0].id if manifest.shots else None

    @app.get("/anchors/{shot_id}")
    def get_anchors_for_shot(shot_id: str):
        anchor_path = output_dir / "camera" / f"{shot_id}_anchors.json"
        if not anchor_path.exists():
            return {"clip_id": shot_id, "image_size": [0, 0], "anchors": []}
        try:
            anchor_set = AnchorSet.load(anchor_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load anchors: {exc}")
        return _anchor_set_to_dict(anchor_set)

    @app.post("/anchors/{shot_id}")
    def post_anchors_for_shot(shot_id: str, payload: AnchorPayload):
        try:
            anchor_set = _dict_to_anchor_set(payload.dict())
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid anchor payload: {exc}")
        anchor_path = output_dir / "camera" / f"{shot_id}_anchors.json"
        anchor_path.parent.mkdir(parents=True, exist_ok=True)
        anchor_set.save(anchor_path)
        return {"saved": True, "path": str(anchor_path), "count": len(anchor_set.anchors)}

    @app.get("/anchors")
    def get_anchors():
        """Legacy endpoint: redirects to the first shot's per-shot file
        when a manifest is present, falls back to the old
        ``output/camera/anchors.json`` otherwise. Logged as deprecated."""
        logging.getLogger(__name__).warning(
            "GET /anchors is deprecated; use /anchors/{shot_id}"
        )
        first = _first_shot_id()
        if first is not None:
            return get_anchors_for_shot(first)
        anchor_path = output_dir / "camera" / "anchors.json"
        if not anchor_path.exists():
            return {"clip_id": "", "image_size": [0, 0], "anchors": []}
        try:
            anchor_set = AnchorSet.load(anchor_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load anchors: {exc}")
        return _anchor_set_to_dict(anchor_set)

    @app.post("/anchors")
    def post_anchors(payload: AnchorPayload):
        """Legacy endpoint: redirects writes to the first shot's per-shot
        file when a manifest is present, falls back to the old
        ``output/camera/anchors.json`` otherwise."""
        logging.getLogger(__name__).warning(
            "POST /anchors is deprecated; use /anchors/{shot_id}"
        )
        first = _first_shot_id()
        if first is not None:
            return post_anchors_for_shot(first, payload)
        try:
            anchor_set = _dict_to_anchor_set(payload.dict())
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid anchor payload: {exc}")
        anchor_path = output_dir / "camera" / "anchors.json"
        anchor_set.save(anchor_path)
        return {"saved": True, "path": str(anchor_path), "count": len(anchor_set.anchors)}

    class RunShotRequest(BaseModel):
        stage: str
        shot_id: str

    @app.post("/api/run-shot")
    def post_run_shot(req: RunShotRequest):
        """Wipe the target shot's stage artefacts and run only that stage
        for that shot. Reuses the existing background-job runner with a
        ``shot_filter`` param plumbed through ``run_pipeline``."""
        # Wipe per-shot artefacts for this stage. The legacy single-shot
        # paths in _STAGE_ARTIFACTS (e.g. "ball/ball_track.json") become
        # per-shot ("ball/{shot_id}_ball_track.json") here.
        artefacts = _STAGE_ARTIFACTS.get(req.stage, [])
        for relpath in artefacts:
            # Try the per-shot variant first; fall back to legacy if needed.
            target = output_dir / relpath
            if target.is_file():
                stem = target.stem
                per_shot = target.with_name(f"{req.shot_id}_{stem}{target.suffix}")
                if per_shot.exists():
                    per_shot.unlink()
            elif target.is_dir():
                for child in target.glob(f"{req.shot_id}_*"):
                    if child.is_file():
                        child.unlink()
                    else:
                        shutil.rmtree(child, ignore_errors=True)

        params = RunRequest(stages=req.stage, shot_filter=req.shot_id)
        with _jobs_lock:
            running_jobs = sum(
                1 for j in _jobs.values() if j.status == "running"
            )
            if running_jobs >= _MAX_CONCURRENT_JOBS:
                raise HTTPException(
                    status_code=429, detail="Too many concurrent jobs",
                )
        job_id = str(uuid.uuid4())[:8]
        job = Job(job_id=job_id, stages=params.stages)
        with _jobs_lock:
            _jobs[job_id] = job
        Thread(
            target=_run_job,
            args=(job, output_dir, app.state.config_path, params),
            daemon=True,
        ).start()
        return {
            "job_id": job_id,
            "stage": req.stage,
            "shot_id": req.shot_id,
        }

    @app.get("/camera/track")
    def get_camera_track(shot: str | None = None):
        """Return a CameraTrack as JSON.

        ``?shot=xxx`` returns ``{shot}_camera_track.json``; absent or
        empty returns the legacy singular ``camera_track.json`` (or, if
        a manifest is present, the first shot's track) for backwards
        compatibility with viewer / dashboard callers that don't yet
        pass a shot filter.
        """
        if shot:
            track_path = output_dir / "camera" / f"{shot}_camera_track.json"
        else:
            track_path = output_dir / "camera" / "camera_track.json"
            if not track_path.exists():
                first = _first_shot_id()
                if first is not None:
                    track_path = output_dir / "camera" / f"{first}_camera_track.json"
        if not track_path.exists():
            return {
                "clip_id": shot or "", "fps": 0.0, "image_size": [0, 0],
                "t_world": [0.0, 0.0, 0.0], "frames": [],
            }
        try:
            track = CameraTrack.load(track_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load camera track: {exc}")
        return asdict(track)

    @app.get("/tracking/shots")
    def list_tracked_shots():
        tracks_dir = output_dir / "tracks"
        if not tracks_dir.exists():
            return {"shots": []}
        ids = sorted(p.stem.replace("_tracks", "") for p in tracks_dir.glob("*_tracks.json"))
        return {"shots": ids}

    @app.get("/tracking/preview")
    def get_tracking_preview(shot_id: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", shot_id):
            raise HTTPException(status_code=400, detail="Invalid shot_id")
        tracks_path = (output_dir / "tracks" / f"{shot_id}_tracks.json").resolve()
        tracks_dir = (output_dir / "tracks").resolve()
        if not tracks_path.is_relative_to(tracks_dir):
            raise HTTPException(status_code=400, detail="Invalid shot_id")
        if not tracks_path.exists():
            raise HTTPException(status_code=404, detail=f"Tracks not found: {shot_id}")
        try:
            from src.schemas.tracks import TracksResult

            result = TracksResult.load(tracks_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load tracks: {exc}")
        summaries: list[dict[str, Any]] = []
        for track in result.tracks:
            confs = [f.confidence for f in track.frames if f.confidence is not None]
            mean_conf = sum(confs) / len(confs) if confs else 0.0
            frame_range = (
                [track.frames[0].frame, track.frames[-1].frame]
                if track.frames else [0, 0]
            )
            summaries.append({
                "track_id": track.track_id,
                "class_name": track.class_name,
                "team": track.team,
                "player_id": track.player_id,
                "player_name": track.player_name,
                "frame_count": len(track.frames),
                "frame_range": frame_range,
                "mean_confidence": mean_conf,
            })
        return {"shot_id": result.shot_id, "tracks": summaries}

    @app.get("/tracking/frames")
    def get_tracking_frames(shot_id: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", shot_id):
            raise HTTPException(status_code=400, detail="Invalid shot_id")
        tracks_path = (output_dir / "tracks" / f"{shot_id}_tracks.json").resolve()
        tracks_dir = (output_dir / "tracks").resolve()
        if not tracks_path.is_relative_to(tracks_dir):
            raise HTTPException(status_code=400, detail="Invalid shot_id")
        if not tracks_path.exists():
            raise HTTPException(status_code=404, detail=f"Tracks not found: {shot_id}")
        try:
            from src.schemas.tracks import TracksResult

            result = TracksResult.load(tracks_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load tracks: {exc}")
        frames_map: dict[int, list[dict[str, Any]]] = {}
        for track in result.tracks:
            for f in track.frames:
                frames_map.setdefault(f.frame, []).append({
                    "track_id": track.track_id,
                    "player_id": track.player_id,
                    "player_name": track.player_name,
                    "class_name": track.class_name,
                    "team": track.team,
                    "bbox": list(f.bbox),
                    "confidence": f.confidence,
                })
        frames = [
            {"frame": frame_idx, "boxes": frames_map[frame_idx]}
            for frame_idx in sorted(frames_map)
        ]
        return {"shot_id": result.shot_id, "frames": frames}

    def _player_name_index() -> dict[str, str]:
        """Build {player_id → player_name} from every shot's tracks file.

        Used by the hmr_world endpoints so the dashboard can show real
        names instead of the synthetic P### IDs. Tracks named 'ignore'
        are skipped; player_id may map to multiple Track records (when a
        merge wasn't followed by a physical-merge save) — first non-empty
        wins.
        """
        names: dict[str, str] = {}
        tracks_dir = output_dir / "tracks"
        if not tracks_dir.exists():
            return names
        for tf in sorted(tracks_dir.glob("*_tracks.json")):
            try:
                tr = TracksResult.load(tf)
            except Exception:
                continue
            for track in tr.tracks:
                pid = track.player_id or track.track_id
                if track.player_name and track.player_name != "ignore":
                    names.setdefault(pid, track.player_name)
        return names

    @app.get("/hmr_world/kp2d_players")
    def list_kp2d_players():
        hmr_dir = output_dir / "hmr_world"
        if not hmr_dir.exists():
            return {"players": []}
        names = _player_name_index()
        rows = []
        for p in sorted(hmr_dir.glob("*_kp2d.json")):
            pid = p.stem.replace("_kp2d", "")
            rows.append({"player_id": pid, "player_name": names.get(pid, "")})
        return {"players": rows}

    @app.get("/hmr_world/kp2d_preview")
    def get_kp2d_preview(player_id: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", player_id):
            raise HTTPException(status_code=400, detail="Invalid player_id")
        kp2d_path = (output_dir / "hmr_world" / f"{player_id}_kp2d.json").resolve()
        hmr_dir = (output_dir / "hmr_world").resolve()
        if not kp2d_path.is_relative_to(hmr_dir):
            raise HTTPException(status_code=400, detail="Invalid player_id")
        if not kp2d_path.exists():
            raise HTTPException(status_code=404, detail=f"kp2d track not found: {player_id}")
        try:
            data = json.loads(kp2d_path.read_text())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load kp2d: {exc}")
        return data

    @app.get("/hmr_world/players")
    def list_hmr_players(shot: str | None = None):
        """List player_ids with HMR-world output.

        ``?shot=xxx`` filters to players whose ``SmplWorldTrack.shot_id``
        equals ``xxx``. Without the param all players are returned (the
        legacy single-shot view). Filtering loads each NPZ to read
        shot_id; for clips with hundreds of players this is fine
        because the load is metadata-only.
        """
        hmr_dir = output_dir / "hmr_world"
        if not hmr_dir.exists():
            return {"players": []}
        names = _player_name_index()
        rows = []
        for npz_path in sorted(hmr_dir.glob("*_smpl_world.npz")):
            pid = npz_path.stem.replace("_smpl_world", "")
            if shot:
                try:
                    z = np.load(npz_path, allow_pickle=False)
                    track_shot = str(z["shot_id"]) if "shot_id" in z.files else ""
                except Exception:
                    track_shot = ""
                if track_shot != shot:
                    continue
            rows.append({"player_id": pid, "player_name": names.get(pid, "")})
        return {"players": rows}

    @app.get("/hmr_world/preview")
    def get_hmr_preview(
        player_id: str, request: Request, include_pose: int = 0,
    ):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", player_id):
            # Diagnostic: log the Referer so we can pinpoint *which*
            # client page is sending malformed player_id values (the
            # ``[object Object]`` symptom seen when a JS list of objects
            # was iterated as if it were a list of strings).
            referer = request.headers.get("referer", "<no-referer>")
            ua = request.headers.get("user-agent", "<no-ua>")[:60]
            logging.warning(
                "/hmr_world/preview rejected player_id=%r referer=%s ua=%s",
                player_id, referer, ua,
            )
            raise HTTPException(status_code=400, detail="Invalid player_id")
        npz_path = (output_dir / "hmr_world" / f"{player_id}_smpl_world.npz").resolve()
        hmr_dir = (output_dir / "hmr_world").resolve()
        if not npz_path.is_relative_to(hmr_dir):
            raise HTTPException(status_code=400, detail="Invalid player_id")
        if not npz_path.exists():
            raise HTTPException(status_code=404, detail=f"hmr_world track not found: {player_id}")
        try:
            z = np.load(npz_path, allow_pickle=False)
            frames = z["frames"].tolist()
            root_t = z["root_t"].tolist()
            confidence = z["confidence"].tolist()
            payload = {
                "player_id": player_id,
                "frames": frames,
                "root_t": root_t,
                "confidence": confidence,
            }
            # Pose payload (thetas + root_R) is opt-in because it inflates
            # the response by ~10× — only the 3D viewer needs it; the
            # dashboard's trajectory panel doesn't.
            if include_pose:
                payload["thetas"] = z["thetas"].tolist()    # (N, 24, 3)
                payload["root_R"] = z["root_R"].tolist()    # (N, 3, 3)
                payload["betas"]  = z["betas"].tolist()     # (10,)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load hmr_world: {exc}")
        return payload

    @app.get("/ball/preview")
    def get_ball_preview(shot: str | None = None):
        if shot:
            ball_path = output_dir / "ball" / f"{shot}_ball_track.json"
        else:
            ball_path = output_dir / "ball" / "ball_track.json"
            if not ball_path.exists():
                first = _first_shot_id()
                if first is not None:
                    ball_path = output_dir / "ball" / f"{first}_ball_track.json"
        if not ball_path.exists():
            return {"clip_id": shot or "", "fps": 0.0, "frames": [], "flight_segments": []}
        try:
            track = BallTrack.load(ball_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load ball track: {exc}")
        return _ball_track_to_dict(track)

    # ------------------------------------------------------------------
    # Track annotation (manual edits to tracking output)
    # ------------------------------------------------------------------

    def _tracks_dir() -> Path:
        return output_dir / "tracks"

    def _tracks_path(shot_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", shot_id):
            raise HTTPException(status_code=400, detail="Invalid shot ID")
        return _tracks_dir() / f"{shot_id}_tracks.json"

    def _all_used_player_ids() -> set[str]:
        used: set[str] = set()
        if not _tracks_dir().exists():
            return used
        for tf in _tracks_dir().glob("*_tracks.json"):
            try:
                tr = TracksResult.load(tf)
            except Exception:
                continue
            for t in tr.tracks:
                if t.player_id:
                    used.add(t.player_id)
        return used

    def _next_player_id(used: set[str]) -> str:
        n = 1
        while f"P{n:03d}" in used:
            n += 1
        used.add(f"P{n:03d}")
        return f"P{n:03d}"

    @app.delete("/api/tracks/{shot_id}/{track_id}")
    def delete_single_track(shot_id: str, track_id: str):
        track_path = _tracks_path(shot_id)
        if not track_path.exists():
            raise HTTPException(status_code=404, detail=f"Tracks not found for {shot_id}")
        tr = TracksResult.load(track_path)
        before = len(tr.tracks)
        tr.tracks = [t for t in tr.tracks if t.track_id != track_id]
        if len(tr.tracks) == before:
            raise HTTPException(status_code=404, detail=f"Track {track_id} not found in {shot_id}")
        tr.save(track_path)
        return {"shot_id": shot_id, "track_id": track_id, "deleted": True}

    @app.patch("/api/tracks/{shot_id}/{track_id}")
    async def patch_track(shot_id: str, track_id: str, request: Request):
        track_path = _tracks_path(shot_id)
        if not track_path.exists():
            raise HTTPException(status_code=404, detail=f"Tracks not found for {shot_id}")
        body = await request.json()
        tr = TracksResult.load(track_path)
        target = next((t for t in tr.tracks if t.track_id == track_id), None)
        if target is None:
            raise HTTPException(status_code=404, detail=f"Track {track_id} not found in {shot_id}")
        if "player_id" in body:
            target.player_id = str(body["player_id"])
        if "player_name" in body:
            target.player_name = str(body["player_name"])
        if "team" in body:
            target.team = str(body["team"])
        tr.save(track_path)
        return {
            "shot_id": shot_id,
            "track_id": track_id,
            "player_id": target.player_id,
            "player_name": target.player_name,
            "team": target.team,
        }

    @app.post("/api/tracks/split")
    async def split_track(request: Request):
        body = await request.json()
        shot_id = body.get("shot_id")
        track_id = body.get("track_id")
        if not shot_id or not track_id:
            raise HTTPException(status_code=400, detail="shot_id and track_id required")
        try:
            split_frame = int(body.get("split_frame", 0))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="split_frame must be an integer")

        track_path = _tracks_path(shot_id)
        if not track_path.exists():
            raise HTTPException(status_code=404, detail=f"Tracks not found for {shot_id}")
        tr = TracksResult.load(track_path)
        target = next((t for t in tr.tracks if t.track_id == track_id), None)
        if target is None:
            raise HTTPException(status_code=404, detail=f"Track {track_id} not found")

        before = [f for f in target.frames if f.frame < split_frame]
        after = [f for f in target.frames if f.frame >= split_frame]
        if not before or not after:
            raise HTTPException(
                status_code=400,
                detail="split_frame must lie strictly inside the track's frame range",
            )

        existing_track_ids = {t.track_id for t in tr.tracks}
        n = 1
        while f"T{n:03d}" in existing_track_ids:
            n += 1
        new_track_id = f"T{n:03d}"
        new_player_id = _next_player_id(_all_used_player_ids())

        target.frames = before
        new_track = Track(
            track_id=new_track_id,
            class_name=target.class_name,
            team=target.team,
            player_id=new_player_id,
            player_name="",
            frames=after,
        )
        tr.tracks.append(new_track)
        tr.save(track_path)
        return {
            "shot_id": shot_id,
            "original_track_id": track_id,
            "new_track_id": new_track_id,
            "new_player_id": new_player_id,
            "original_frames": len(before),
            "new_frames": len(after),
        }

    def _physically_merge(
        tracks: list[Track],
        *,
        canonical_pid: str,
        canonical_name: str,
    ) -> tuple[Track, list[str], int]:
        """Physically combine ``tracks`` into one Track record.

        - Frames from every input are concatenated and sorted by frame.
        - When two inputs claim the same frame, the higher-confidence
          observation wins (first-seen breaks ties); the other is dropped
          and counted in the returned collision tally.
        - The kept Track is the input with the lexicographically smallest
          ``track_id`` so callers see stable identifiers across merges.
        - ``player_id`` / ``player_name`` are overwritten with the supplied
          canonicals; ``team`` is taken from the kept track.

        Returns the kept Track, the list of removed track_ids, and the
        number of frame collisions resolved.
        """
        kept = min(tracks, key=lambda t: t.track_id)
        merged_by_frame: dict[int, TrackFrame] = {}
        collisions = 0
        for t in tracks:
            for f in t.frames:
                existing = merged_by_frame.get(f.frame)
                if existing is None:
                    merged_by_frame[f.frame] = f
                    continue
                collisions += 1
                if (f.confidence or 0.0) > (existing.confidence or 0.0):
                    merged_by_frame[f.frame] = f
        kept.frames = [merged_by_frame[fi] for fi in sorted(merged_by_frame)]
        kept.player_id = canonical_pid
        if canonical_name:
            kept.player_name = canonical_name
        removed_ids = [t.track_id for t in tracks if t.track_id != kept.track_id]
        return kept, removed_ids, collisions

    @app.post("/api/tracks/merge")
    async def merge_tracks(request: Request):
        body = await request.json()
        shot_id = body.get("shot_id")
        track_ids = body.get("track_ids") or []
        if not shot_id or not isinstance(track_ids, list) or len(track_ids) < 2:
            raise HTTPException(
                status_code=400,
                detail="shot_id and track_ids (≥2) required",
            )
        track_path = _tracks_path(shot_id)
        if not track_path.exists():
            raise HTTPException(status_code=404, detail=f"Tracks not found for {shot_id}")
        tr = TracksResult.load(track_path)

        targets = [t for t in tr.tracks if t.track_id in track_ids]
        missing = set(track_ids) - {t.track_id for t in targets}
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"Track(s) not found in {shot_id}: {sorted(missing)}",
            )

        # Pick a canonical player_id: prefer the first track that has one
        # already, else mint a fresh P-id.
        canonical_pid = next((t.player_id for t in targets if t.player_id), None)
        if not canonical_pid:
            canonical_pid = _next_player_id(_all_used_player_ids())
        canonical_name = next(
            (t.player_name for t in targets if t.player_name and t.player_name != "ignore"),
            "",
        )

        # Optional: caller can override
        if body.get("player_id"):
            canonical_pid = str(body["player_id"])
        if body.get("player_name") is not None:
            canonical_name = str(body["player_name"])

        kept, removed_ids, collisions = _physically_merge(
            targets, canonical_pid=canonical_pid, canonical_name=canonical_name
        )
        tr.tracks = [t for t in tr.tracks if t.track_id not in removed_ids]
        tr.save(track_path)
        return {
            "shot_id": shot_id,
            "merged_into": kept.track_id,
            "removed_track_ids": removed_ids,
            "player_id": canonical_pid,
            "player_name": canonical_name,
            "frame_collisions": collisions,
        }

    @app.post("/api/tracks/merge-by-name")
    def merge_tracks_by_name():
        if not _tracks_dir().exists():
            return {
                "merged_groups": 0,
                "tracks_removed": 0,
                "frame_collisions": 0,
            }

        # First pass: load every shot's TracksResult and collect named
        # player/goalkeeper tracks. Canonical player_id per name prefers
        # any existing P-id; otherwise we mint one. Cross-shot tracks
        # cannot be physically merged (they live in different files), but
        # they share the same player_id so hmr_world groups them.
        used = _all_used_player_ids()
        all_files: list[tuple[Path, TracksResult]] = []
        all_named: dict[str, list[Track]] = {}
        canonical: dict[str, str] = {}
        for tf in sorted(_tracks_dir().glob("*_tracks.json")):
            try:
                tr = TracksResult.load(tf)
            except Exception:
                continue
            all_files.append((tf, tr))
            for track in tr.tracks:
                if track.class_name not in ("player", "goalkeeper"):
                    continue
                name = track.player_name
                if not name or name == "ignore":
                    continue
                all_named.setdefault(name, []).append(track)
                if name not in canonical and track.player_id:
                    canonical[name] = track.player_id
        for name in all_named:
            if name not in canonical:
                canonical[name] = _next_player_id(used)

        # Second pass: per shot, group tracks by name and physically merge
        # each group. Drop merged-away records from tr.tracks. Always set
        # player_id to the canonical (so single-occurrence shots are
        # rewritten too).
        tracks_removed = 0
        total_collisions = 0
        for tf, tr in all_files:
            dirty = False
            by_name: dict[str, list[Track]] = {}
            for track in tr.tracks:
                if track.class_name not in ("player", "goalkeeper"):
                    continue
                name = track.player_name
                if not name or name == "ignore":
                    continue
                by_name.setdefault(name, []).append(track)
            removed_ids: set[str] = set()
            for name, group in by_name.items():
                target_pid = canonical[name]
                if len(group) > 1:
                    _, removed, collisions = _physically_merge(
                        group, canonical_pid=target_pid, canonical_name=name
                    )
                    removed_ids.update(removed)
                    total_collisions += collisions
                    tracks_removed += len(removed)
                    dirty = True
                else:
                    only = group[0]
                    if only.player_id != target_pid:
                        only.player_id = target_pid
                        dirty = True
            if removed_ids:
                tr.tracks = [t for t in tr.tracks if t.track_id not in removed_ids]
            if dirty:
                tr.save(tf)

        return {
            "merged_groups": len(all_named),
            "tracks_removed": tracks_removed,
            "frame_collisions": total_collisions,
        }

    @app.post("/api/tracks/ignore-unknown/{shot_id}")
    def ignore_unknown_tracks(shot_id: str):
        track_path = _tracks_path(shot_id)
        if not track_path.exists():
            raise HTTPException(status_code=404, detail=f"Tracks not found for {shot_id}")
        tr = TracksResult.load(track_path)
        count = 0
        for t in tr.tracks:
            if t.class_name == "ball":
                continue
            if not t.player_name:
                t.player_name = "ignore"
                count += 1
        tr.save(track_path)
        return {"shot_id": shot_id, "count": count}

    @app.post("/api/tracks/delete-ignored")
    def delete_ignored_tracks():
        if not _tracks_dir().exists():
            return {"deleted": 0}
        deleted = 0
        for tf in sorted(_tracks_dir().glob("*_tracks.json")):
            tr = TracksResult.load(tf)
            before = len(tr.tracks)
            tr.tracks = [t for t in tr.tracks if t.player_name != "ignore"]
            removed = before - len(tr.tracks)
            if removed:
                deleted += removed
                tr.save(tf)
        return {"deleted": deleted}

    @app.get("/anchor_editor")
    def anchor_editor_page():
        # (no-store too — same dev-cache issue as / and /viewer)
        editor_path = static_dir / "anchor_editor.html"
        if not editor_path.exists():
            raise HTTPException(status_code=404, detail="anchor_editor.html not found")
        return FileResponse(str(editor_path), headers=_NO_STORE)

    return app


# ---------------------------------------------------------------------------
# Schema (de)serialisation helpers
# ---------------------------------------------------------------------------


def _anchor_set_to_dict(anchor_set: AnchorSet) -> dict[str, Any]:
    return {
        "clip_id": anchor_set.clip_id,
        "image_size": list(anchor_set.image_size),
        "stadium": anchor_set.stadium,
        "anchors": [
            {
                "frame": a.frame,
                "landmarks": [
                    {
                        "name": lm.name,
                        "image_xy": list(lm.image_xy),
                        "world_xyz": list(lm.world_xyz),
                    }
                    for lm in a.landmarks
                ],
                "lines": [
                    {
                        "name": ln.name,
                        "image_segment": [
                            list(ln.image_segment[0]),
                            list(ln.image_segment[1]),
                        ],
                        "world_segment": (
                            [list(ln.world_segment[0]), list(ln.world_segment[1])]
                            if ln.world_segment is not None else None
                        ),
                        "world_direction": (
                            list(ln.world_direction)
                            if ln.world_direction is not None else None
                        ),
                    }
                    for ln in a.lines
                ],
            }
            for a in anchor_set.anchors
        ],
    }


def _dict_to_anchor_set(data: dict[str, Any]) -> AnchorSet:
    from src.schemas.anchor import Anchor, LandmarkObservation, LineObservation

    anchors = tuple(
        Anchor(
            frame=int(a["frame"]),
            landmarks=tuple(
                LandmarkObservation(
                    name=str(lm["name"]),
                    image_xy=(float(lm["image_xy"][0]), float(lm["image_xy"][1])),
                    world_xyz=(
                        float(lm["world_xyz"][0]),
                        float(lm["world_xyz"][1]),
                        float(lm["world_xyz"][2]),
                    ),
                )
                for lm in a.get("landmarks", [])
            ),
            lines=tuple(
                LineObservation(
                    name=str(ln["name"]),
                    image_segment=(
                        (float(ln["image_segment"][0][0]), float(ln["image_segment"][0][1])),
                        (float(ln["image_segment"][1][0]), float(ln["image_segment"][1][1])),
                    ),
                    world_segment=(
                        (
                            (
                                float(ln["world_segment"][0][0]),
                                float(ln["world_segment"][0][1]),
                                float(ln["world_segment"][0][2]),
                            ),
                            (
                                float(ln["world_segment"][1][0]),
                                float(ln["world_segment"][1][1]),
                                float(ln["world_segment"][1][2]),
                            ),
                        )
                        if ln.get("world_segment") is not None else None
                    ),
                    world_direction=(
                        (
                            float(ln["world_direction"][0]),
                            float(ln["world_direction"][1]),
                            float(ln["world_direction"][2]),
                        )
                        if ln.get("world_direction") is not None else None
                    ),
                )
                for ln in a.get("lines", [])
            ),
        )
        for a in data.get("anchors", [])
    )
    image_size = tuple(data.get("image_size", (0, 0)))
    stadium_raw = data.get("stadium")
    return AnchorSet(
        clip_id=str(data.get("clip_id", "")),
        image_size=(int(image_size[0]), int(image_size[1])),
        anchors=anchors,
        stadium=str(stadium_raw) if stadium_raw else None,
    )


def _ball_track_to_dict(track: BallTrack) -> dict[str, Any]:
    return {
        "clip_id": track.clip_id,
        "fps": track.fps,
        "frames": [
            {
                "frame": f.frame,
                "world_xyz": list(f.world_xyz) if f.world_xyz is not None else None,
                "state": f.state,
                "confidence": f.confidence,
                "flight_segment_id": f.flight_segment_id,
            }
            for f in track.frames
        ],
        "flight_segments": [
            {
                "id": s.id,
                "frame_range": list(s.frame_range),
                "parabola": s.parabola,
                "fit_residual_px": s.fit_residual_px,
            }
            for s in track.flight_segments
        ],
    }

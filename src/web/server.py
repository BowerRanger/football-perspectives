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
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.pipeline.config import load_config
from src.pipeline.runner import run_pipeline
from src.schemas.anchor import AnchorSet
from src.schemas.ball_anchor import BallAnchor, BallAnchorSet
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
    "refined_poses",
    "export",
]

def _manifest_shot_ids(output_dir: Path) -> list[str]:
    """Return the shot ids from ``shots_manifest.json``, or [] if absent."""
    from src.schemas.shots import ShotsManifest

    manifest_path = output_dir / "shots" / "shots_manifest.json"
    if not manifest_path.exists():
        return []
    try:
        return [s.id for s in ShotsManifest.load(manifest_path).shots]
    except Exception:
        return []


def _camera_complete(output_dir: Path) -> bool:
    """Camera stage is green only when every shot in the manifest has a
    ``{shot_id}_camera_track.json`` on disk. With no manifest we fall
    back to the legacy singular ``camera_track.json``."""
    shot_ids = _manifest_shot_ids(output_dir)
    if not shot_ids:
        return (output_dir / "camera" / "camera_track.json").exists()
    return all(
        (output_dir / "camera" / f"{sid}_camera_track.json").exists()
        for sid in shot_ids
    )


def _hmr_world_complete(output_dir: Path) -> bool:
    """Green only when every (shot_id, player_id) group expected by
    ``HmrWorldStage._build_player_groups`` has its
    ``{shot_id}__{player_id}_smpl_world.npz`` on disk. Falls back to
    "any new-scheme file exists" when there's no tracks dir to compute
    expectations from."""
    out = output_dir / "hmr_world"
    if not out.exists():
        return False
    new_scheme_files = {
        p.stem.replace("_smpl_world", "")
        for p in out.glob("*_smpl_world.npz")
        if "__" in p.stem
    }
    if not new_scheme_files:
        return False
    tracks_dir = output_dir / "tracks"
    if not tracks_dir.exists():
        return True
    expected: set[str] = set()
    for tracks_path in tracks_dir.glob("*_tracks.json"):
        try:
            tr = TracksResult.load(tracks_path)
        except Exception:
            continue
        for track in tr.tracks:
            if track.class_name not in ("player", "goalkeeper"):
                continue
            if track.player_name == "ignore":
                continue
            pid = (
                track.player_id
                or (
                    f"{tr.shot_id}_T{track.track_id}"
                    if track.track_id else None
                )
            )
            if pid is None:
                continue
            expected.add(f"{tr.shot_id}__{pid}")
    if not expected:
        return True
    return expected.issubset(new_scheme_files)


def _refined_poses_complete(output_dir: Path) -> bool:
    """Green only when every player_id discoverable in
    ``hmr_world/{shot}__{pid}_smpl_world.npz`` has a matching
    ``refined_poses/{pid}_refined.npz``. Mirrors
    ``RefinedPosesStage.is_complete`` so the dashboard's stage badge
    agrees with the runner's cache decisions."""
    hmr = output_dir / "hmr_world"
    refined = output_dir / "refined_poses"
    if not hmr.exists():
        return True
    expected: set[str] = set()
    for p in hmr.glob("*_smpl_world.npz"):
        stem = p.stem.removesuffix("_smpl_world")
        if "__" in stem:
            expected.add(stem.split("__", 1)[1])
        else:
            expected.add(stem)
    if not expected:
        return True
    return all((refined / f"{pid}_refined.npz").exists() for pid in expected)


_STAGE_COMPLETE = {
    "prepare_shots": lambda d: (d / "shots" / "shots_manifest.json").exists(),
    "tracking": lambda d: any((d / "tracks").glob("*_tracks.json")),
    "camera": _camera_complete,
    "hmr_world": _hmr_world_complete,
    "ball": lambda d: (d / "ball" / "ball_track.json").exists(),
    "refined_poses": _refined_poses_complete,
    "export": lambda d: (d / "export" / "gltf" / "scene.glb").exists(),
}

# Per-stage outputs that should be wiped on a "re-run" or "clear" action.
# Paths are relative to ``output_dir``; entries may be files, directories,
# or glob patterns (containing ``*``) — globs are matched against
# ``output_dir`` at delete time and each match is unlinked.
# ``camera/anchors.json`` and ``ball/*_ball_anchors.json`` are deliberately
# omitted — they are user-supplied input (anchors), not outputs, and
# must survive a stage re-run.
_STAGE_ARTIFACTS: dict[str, list[str]] = {
    "prepare_shots": ["shots"],
    "tracking": ["tracks"],
    "camera": ["camera/camera_track.json", "camera/debug"],
    "hmr_world": ["hmr_world"],
    "ball": ["ball/*_ball_track.json", "ball/ball_track.json"],
    "refined_poses": ["refined_poses"],
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
    # Per-player: when set, hmr_world only fits the named player_id
    # (paired with shot_filter to disambiguate multi-shot collisions).
    # Other stages ignore this. Used by the dashboard's
    # /api/run-shot-player endpoint for fast iteration on one player.
    player_filter: str | None = None


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
            player_filter=params.player_filter,
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

    def _hmr_world_in_flight() -> Job | None:
        """Return the currently-running hmr_world job, if any.

        hmr_world holds a process-global lock during inference (see
        ``_GVHMR_INFERENCE_LOCK`` in gvhmr_estimator.py), so concurrent
        hmr_world dispatches just queue up behind each other and waste
        server slots. Reject the second one with 409 so the operator
        gets immediate, actionable feedback rather than a job that
        looks alive in the dashboard but is silently blocked on a lock.
        """
        with _jobs_lock:
            for j in _jobs.values():
                if j.status != "running":
                    continue
                stages_str = (j.stages or "")
                if "hmr_world" in stages_str.split(",") or stages_str == "all":
                    return j
        return None

    @app.post("/api/run", status_code=202)
    def run_stages(params: RunRequest):
        requested_stages = (params.stages or "").split(",")
        wants_hmr = "hmr_world" in requested_stages or params.stages == "all"
        if wants_hmr:
            blocker = _hmr_world_in_flight()
            if blocker is not None:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"hmr_world is already running (job {blocker.job_id}). "
                        "Wait for it to finish or stop the dashboard process."
                    ),
                )
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
            if "*" in relpath:
                # Glob pattern — match against output_dir and unlink each hit.
                for match in output_dir.glob(relpath):
                    if match.is_dir():
                        shutil.rmtree(match)
                    else:
                        match.unlink()
                    removed.append(str(match.relative_to(output_dir)))
                continue
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
    def get_scene_glb(shot: str | None = None):
        """Return ``{shot}_scene.glb`` when ``?shot=`` is supplied,
        otherwise the legacy singular ``scene.glb``. The viewer uses
        ``/api/output/shots`` to drive the per-shot routing."""
        gltf_dir = output_dir / "export" / "gltf"
        if shot:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", shot):
                raise HTTPException(status_code=400, detail="Invalid shot id")
            glb_path = gltf_dir / f"{shot}_scene.glb"
        else:
            glb_path = gltf_dir / "scene.glb"
            if not glb_path.exists():
                # Backwards-compat: when only per-shot files exist (the
                # multi-shot output of the modern export stage), serve
                # the first available so legacy callers that don't pass
                # a shot still get *something*.
                per_shot = sorted(gltf_dir.glob("*_scene.glb"))
                if per_shot:
                    glb_path = per_shot[0]
        if not glb_path.exists():
            raise HTTPException(status_code=404, detail="scene.glb not found")
        return FileResponse(str(glb_path), media_type="model/gltf-binary")

    @app.get("/api/export/metadata")
    def get_scene_metadata(shot: str | None = None):
        """Return ``{shot}_scene_metadata.json`` when ``?shot=`` is
        supplied, else the legacy singular file. Falls back to the
        first per-shot metadata file when the legacy one is missing —
        so the dashboard can render *something* in multi-shot mode
        even when only one shot has been exported."""
        gltf_dir = output_dir / "export" / "gltf"
        if shot:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", shot):
                raise HTTPException(status_code=400, detail="Invalid shot id")
            meta_path = gltf_dir / f"{shot}_scene_metadata.json"
        else:
            meta_path = gltf_dir / "scene_metadata.json"
            if not meta_path.exists():
                per_shot = sorted(gltf_dir.glob("*_scene_metadata.json"))
                if per_shot:
                    meta_path = per_shot[0]
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail="scene_metadata.json not found")
        return json.loads(meta_path.read_text())

    @app.get("/api/export/shots")
    def list_export_shots():
        """List shot ids that have a ``{shot}_scene.glb`` ready to
        view. Drives the export panel's shot picker."""
        gltf_dir = output_dir / "export" / "gltf"
        if not gltf_dir.exists():
            return {"shots": []}
        ids = sorted(
            p.stem.replace("_scene", "")
            for p in gltf_dir.glob("*_scene.glb")
        )
        return {"shots": ids}

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

    @app.get("/api/output/quality-report")
    def get_quality_report():
        """Return the most recent ``quality_report.json`` (or {} if absent).

        The dashboard's Multi-Shot Status panel reads this to render the
        refined-poses summary line. Stays read-only — quality_report is
        produced by the pipeline, never edited via the dashboard.
        """
        path = output_dir / "quality_report.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to load quality_report: {exc}",
            )

    @app.get("/api/shots/manifest")
    def get_shots_manifest():
        """Return the full ShotsManifest as JSON.

        Used by the dashboard's sync timeline to size each clip block by
        its actual frame count. Returns an empty manifest skeleton when
        no manifest is on disk so the front-end can render an empty
        timeline without a 404 detour.
        """
        from src.schemas.shots import ShotsManifest

        manifest_path = output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            return {
                "source_file": "", "fps": 0.0, "total_frames": 0, "shots": [],
            }
        try:
            manifest = ShotsManifest.load(manifest_path)
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to load manifest: {exc}",
            )
        return asdict(manifest)

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

    # ------------------------------------------------------------------
    # Sync map (manual shot-to-shot frame alignment)
    # ------------------------------------------------------------------

    class SyncAlignmentPayload(BaseModel):
        shot_id: str
        frame_offset: int
        method: str = "manual"
        confidence: float = 1.0

    class SyncMapPayload(BaseModel):
        reference_shot: str
        alignments: list[SyncAlignmentPayload]

    def _sync_map_path() -> Path:
        return output_dir / "shots" / "sync_map.json"

    def _manifest_shot_ids_or_empty() -> list[str]:
        # Re-uses the helper near the top of the module to read the
        # shots manifest, with a quiet fallback when no manifest exists
        # so the dashboard's GET still has something useful to return
        # mid-bootstrap (e.g. before the first prepare_shots run).
        return _manifest_shot_ids(output_dir)

    @app.get("/api/sync")
    def get_sync_map():
        """Return the SyncMap JSON (operator-edited shot offsets).

        On disk: ``output/shots/sync_map.json``. When absent, return a
        fresh default with every shot at ``frame_offset=0`` so the
        dashboard's editor can render one row per shot before the
        operator has saved anything.
        """
        from src.schemas.sync_map import SyncMap, default_sync_map

        path = _sync_map_path()
        manifest_ids = _manifest_shot_ids_or_empty()
        if path.exists():
            try:
                sm = SyncMap.load(path)
            except Exception as exc:
                raise HTTPException(
                    status_code=500, detail=f"Failed to load sync_map: {exc}",
                )
            saved_ids = {a.shot_id for a in sm.alignments}
            # If new shots were added since the last save, append them
            # at offset=0 so the editor surfaces every current shot
            # without forcing the operator to re-save first.
            from src.schemas.sync_map import Alignment
            for sid in manifest_ids:
                if sid not in saved_ids:
                    sm.alignments.append(
                        Alignment(shot_id=sid, frame_offset=0)
                    )
            sm.alignments.sort(key=lambda a: a.shot_id)
            return asdict(sm)
        if not manifest_ids:
            return {"reference_shot": "", "alignments": []}
        return asdict(
            default_sync_map(reference_shot=manifest_ids[0], shot_ids=manifest_ids)
        )

    @app.post("/api/sync")
    def post_sync_map(payload: SyncMapPayload):
        """Persist a SyncMap. The reference shot must appear in the
        alignments with ``frame_offset=0``; the dashboard enforces this
        UX-side too but we re-validate for safety."""
        from src.schemas.sync_map import (
            Alignment, SyncMap, validate_method,
        )

        manifest_ids = set(_manifest_shot_ids_or_empty())
        if not payload.reference_shot:
            raise HTTPException(
                status_code=400, detail="reference_shot is required",
            )
        if manifest_ids and payload.reference_shot not in manifest_ids:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"reference_shot {payload.reference_shot!r} is not in"
                    " the shots manifest"
                ),
            )
        seen: set[str] = set()
        alignments: list[Alignment] = []
        for a in payload.alignments:
            if a.shot_id in seen:
                raise HTTPException(
                    status_code=400,
                    detail=f"duplicate shot_id {a.shot_id!r} in alignments",
                )
            seen.add(a.shot_id)
            if manifest_ids and a.shot_id not in manifest_ids:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"shot_id {a.shot_id!r} is not in the shots manifest"
                    ),
                )
            try:
                method = validate_method(a.method)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            alignments.append(Alignment(
                shot_id=a.shot_id,
                frame_offset=int(a.frame_offset),
                method=method,
                confidence=float(a.confidence),
            ))
        # Reference shot must be pinned to offset=0 so downstream
        # consumers can compute global frame indices unambiguously.
        ref_alignment = next(
            (a for a in alignments if a.shot_id == payload.reference_shot),
            None,
        )
        if ref_alignment is None:
            alignments.append(Alignment(
                shot_id=payload.reference_shot, frame_offset=0,
            ))
        elif ref_alignment.frame_offset != 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "reference_shot must have frame_offset=0; got "
                    f"{ref_alignment.frame_offset}"
                ),
            )
        alignments.sort(key=lambda a: a.shot_id)
        sm = SyncMap(
            reference_shot=payload.reference_shot, alignments=alignments,
        )
        sm.save(_sync_map_path())
        return {"saved": True, "path": str(_sync_map_path()), "count": len(alignments)}

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

    class BallAnchorEntry(BaseModel):
        frame: int
        image_xy: list[float] | None
        state: str
        # Required only when state == "player_touch"; otherwise omitted.
        player_id: str | None = None
        bone: str | None = None

    class BallAnchorPayload(BaseModel):
        clip_id: str
        image_size: list[int]
        anchors: list[BallAnchorEntry]

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

    @app.get("/ball-anchors/{shot_id}")
    def get_ball_anchors_for_shot(shot_id: str):
        path = output_dir / "ball" / f"{shot_id}_ball_anchors.json"
        if not path.exists():
            return {"clip_id": shot_id, "image_size": [0, 0], "anchors": []}
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load ball anchors: {exc}")
        return data

    @app.post("/ball-anchors/{shot_id}")
    def post_ball_anchors_for_shot(shot_id: str, payload: BallAnchorPayload):
        tmp = output_dir / "ball" / f".{shot_id}_ball_anchors.tmp.json"
        try:
            anchors_in: list[BallAnchor] = []
            for a in payload.anchors:
                anchors_in.append(BallAnchor(
                    frame=int(a.frame),
                    image_xy=tuple(a.image_xy) if a.image_xy is not None else None,
                    state=a.state,
                    player_id=a.player_id,
                    bone=a.bone,
                ))
            aset = BallAnchorSet(
                clip_id=str(payload.clip_id),
                image_size=(int(payload.image_size[0]), int(payload.image_size[1])),
                anchors=tuple(anchors_in),
            )
            # Round-trip through JSON to apply state-validation rules.
            tmp.parent.mkdir(parents=True, exist_ok=True)
            aset.save(tmp)
            BallAnchorSet.load(tmp)  # validation pass
        except (KeyError, TypeError, ValueError) as exc:
            try:
                tmp.unlink()
            except Exception:
                pass
            raise HTTPException(status_code=400, detail=f"Invalid ball anchors: {exc}")
        final = output_dir / "ball" / f"{shot_id}_ball_anchors.json"
        tmp.replace(final)
        return {"saved": True, "path": str(final), "count": len(aset.anchors)}

    @app.post("/ball-anchors/{shot_id}/preview")
    def preview_ball_anchors(shot_id: str, payload: BallAnchorPayload):
        """Run BallStage in a temp output dir using the posted anchors.
        Returns the would-be BallTrack JSON without touching the
        on-disk ball_track. Requires camera_track + shots_manifest in
        the production output dir.
        """
        import tempfile

        from src.stages.ball import BallStage

        cam_path = output_dir / "camera" / f"{shot_id}_camera_track.json"
        if not cam_path.exists():
            raise HTTPException(status_code=404,
                                detail=f"No camera track for shot {shot_id!r}")
        manifest_path = output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            raise HTTPException(status_code=404,
                                detail=f"No shots manifest at {manifest_path}")

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            # Mirror the production camera + shots + (optionally) hmr_world
            # data into the temp output dir; BallStage writes only to ball/.
            for sub in ("camera", "shots", "hmr_world"):
                src = output_dir / sub
                if src.exists():
                    shutil.copytree(src, tdp / sub, dirs_exist_ok=True)

            try:
                anchors_in = tuple(
                    BallAnchor(
                        frame=int(a.frame),
                        image_xy=tuple(a.image_xy) if a.image_xy is not None else None,
                        state=a.state,
                        player_id=a.player_id,
                        bone=a.bone,
                    ) for a in payload.anchors
                )
                aset = BallAnchorSet(
                    clip_id=str(payload.clip_id),
                    image_size=(int(payload.image_size[0]), int(payload.image_size[1])),
                    anchors=anchors_in,
                )
                # Round-trip validation.
                tmp_validate = tdp / "ball" / f".{shot_id}_validate.json"
                tmp_validate.parent.mkdir(parents=True, exist_ok=True)
                aset.save(tmp_validate)
                BallAnchorSet.load(tmp_validate)
                tmp_validate.unlink()
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"Invalid ball anchors: {exc}")

            ball_dir = tdp / "ball"
            ball_dir.mkdir(parents=True, exist_ok=True)
            aset.save(ball_dir / f"{shot_id}_ball_anchors.json")

            # Use the project's default config so detector + tracker
            # defaults match production.
            try:
                cfg = load_config(None)
            except Exception as exc:
                raise HTTPException(status_code=500,
                                    detail=f"Failed to load default config: {exc}")

            stage = BallStage(config=cfg, output_dir=tdp)
            stage.shot_filter = shot_id
            try:
                stage.run()
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"BallStage preview failed: {exc}")
            track_path = ball_dir / f"{shot_id}_ball_track.json"
            if not track_path.exists():
                raise HTTPException(status_code=500, detail="Preview produced no ball_track")
            return json.loads(track_path.read_text())

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

        if req.stage == "hmr_world":
            blocker = _hmr_world_in_flight()
            if blocker is not None:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"hmr_world is already running (job {blocker.job_id}). "
                        "Wait for it to finish or stop the dashboard process."
                    ),
                )
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

    class RunShotPlayerRequest(BaseModel):
        shot_id: str
        player_id: str

    @app.post("/api/run-shot-player")
    def post_run_shot_player(req: RunShotPlayerRequest):
        """Re-run hmr_world for one ``(shot_id, player_id)`` pair only.

        Wipes that pair's per-shot artefacts (``{shot}__{pid}_*.json``,
        ``{shot}__{pid}_*.npz``) so GVHMR re-runs cleanly, then dispatches
        a filtered ``run_pipeline`` job. Other shots/players in
        ``hmr_world/`` are untouched, and other stages aren't run."""
        valid = re.compile(r"[A-Za-z0-9_-]+")
        if not valid.fullmatch(req.shot_id):
            raise HTTPException(status_code=400, detail="Invalid shot_id")
        if not valid.fullmatch(req.player_id):
            raise HTTPException(status_code=400, detail="Invalid player_id")
        hmr_dir = output_dir / "hmr_world"
        blocker = _hmr_world_in_flight()
        if blocker is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"hmr_world is already running (job {blocker.job_id}). "
                    "Wait for it to finish or stop the dashboard process."
                ),
            )
        if hmr_dir.exists():
            for path in hmr_dir.glob(f"{req.shot_id}__{req.player_id}_*"):
                if path.is_file():
                    path.unlink()
        params = RunRequest(
            stages="hmr_world",
            from_stage="hmr_world",
            shot_filter=req.shot_id,
            player_filter=req.player_id,
        )
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
            "stage": "hmr_world",
            "shot_id": req.shot_id,
            "player_id": req.player_id,
        }

    @app.post("/api/shots/upload")
    async def upload_shots(files: list[UploadFile] = File(...)):
        """Accept one or more uploaded video clips, write them into
        ``output/shots/`` under their sanitised filenames, and dispatch
        a prepare_shots background job to merge them into the manifest.

        Existing shots in the manifest are preserved — uploads only add
        new entries. Files whose sanitised name collides with an existing
        clip are rejected so the operator can rename and re-upload
        rather than silently overwriting.
        """
        from src.schemas.shots import _sanitise_shot_id

        shots_dir = output_dir / "shots"
        shots_dir.mkdir(parents=True, exist_ok=True)

        saved: list[str] = []
        skipped: list[dict[str, str]] = []
        for uf in files:
            raw_name = Path(uf.filename or "").name
            if not raw_name.lower().endswith(".mp4"):
                skipped.append({"name": raw_name, "reason": "not an .mp4 file"})
                await uf.close()
                continue
            try:
                shot_id = _sanitise_shot_id(Path(raw_name).stem)
            except ValueError as exc:
                skipped.append({"name": raw_name, "reason": str(exc)})
                await uf.close()
                continue
            dest = shots_dir / f"{shot_id}.mp4"
            if dest.exists():
                skipped.append({
                    "name": raw_name,
                    "reason": f"shot_id {shot_id!r} already exists",
                })
                await uf.close()
                continue
            with dest.open("wb") as out:
                while True:
                    chunk = await uf.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            await uf.close()
            saved.append(shot_id)

        if not saved:
            return {"saved": [], "skipped": skipped, "job_id": None}

        # Run prepare_shots without wiping — the stage scans shots/ for
        # any clips not yet in the manifest and registers them.
        params = RunRequest(stages="prepare_shots", from_stage="prepare_shots")
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
        return {"saved": saved, "skipped": skipped, "job_id": job_id}

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
            if not re.fullmatch(r"[A-Za-z0-9_-]+", shot):
                raise HTTPException(status_code=400, detail="Invalid shot id")
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

    def _player_name_index() -> dict[tuple[str, str], str]:
        """Build ``{(shot_id, pid_in_hmr_filename) → player_name}`` from
        every shot's tracks file.

        ``pid_in_hmr_filename`` mirrors what
        ``HmrWorldStage._build_player_groups`` synthesises for the
        ``{shot_id}__{pid}_smpl_world.npz`` filename:
          - the operator-set ``player_id`` when present, else
          - ``f"{shot_id}_T{track_id}"`` for unannotated tracks.

        Without this mirror the lookup misses any track whose
        ``player_name`` was set but whose ``player_id`` is blank — e.g.
        Matip in shot A might be ``track_id=T004, player_id=""``,
        ``player_name="Matip"``; the hmr filename would then be
        ``A__A_TT004_smpl_world.npz`` (pid = ``"A_TT004"``) and a name
        index keyed only by raw ``track_id`` would miss it. Also
        registers ``track.player_id`` as an extra key so legacy
        single-shot callers (no shot context) keep resolving names.
        """
        names: dict[tuple[str, str], str] = {}
        tracks_dir = output_dir / "tracks"
        if not tracks_dir.exists():
            return names
        for tf in sorted(tracks_dir.glob("*_tracks.json")):
            try:
                tr = TracksResult.load(tf)
            except Exception:
                continue
            for track in tr.tracks:
                if not track.player_name or track.player_name == "ignore":
                    continue
                shot_pid = (
                    track.player_id
                    or (
                        f"{tr.shot_id}_T{track.track_id}"
                        if track.track_id else None
                    )
                )
                if shot_pid is None:
                    continue
                names.setdefault((tr.shot_id, shot_pid), track.player_name)
                # Cross-shot fallback: a player_id assigned in *any*
                # shot's tracks file resolves the same name in every
                # other shot (e.g. Matip's player_id=P005 in origi02
                # also names her track in origi01 if the operator forgot
                # to physically-merge there). Only kicks in when no
                # exact (shot_id, pid) row already won.
                if track.player_id:
                    names.setdefault(("", track.player_id), track.player_name)
        return names

    _PID_RE = re.compile(r"[A-Za-z0-9_-]+")

    def _resolve_hmr_path(
        suffix: str, shot: str | None, player_id: str,
    ) -> Path:
        """Resolve a per-(shot, player) hmr_world artefact path.

        New scheme (preferred): ``{shot}__{player_id}{suffix}``.
        Legacy fallback: ``{player_id}{suffix}`` when the per-shot file
        doesn't exist (covers viewers that haven't updated to the new
        endpoints yet).
        """
        if not _PID_RE.fullmatch(player_id):
            raise HTTPException(status_code=400, detail="Invalid player_id")
        if shot is not None and not _PID_RE.fullmatch(shot):
            raise HTTPException(status_code=400, detail="Invalid shot id")
        hmr_dir = (output_dir / "hmr_world").resolve()
        if shot:
            keyed = (hmr_dir / f"{shot}__{player_id}{suffix}").resolve()
            if not keyed.is_relative_to(hmr_dir):
                raise HTTPException(status_code=400, detail="Invalid path")
            if keyed.exists():
                return keyed
        legacy = (hmr_dir / f"{player_id}{suffix}").resolve()
        if not legacy.is_relative_to(hmr_dir):
            raise HTTPException(status_code=400, detail="Invalid path")
        return legacy

    def _list_hmr_files(
        glob_pattern: str, stem_strip: str, shot: str | None,
    ) -> list[dict[str, str]]:
        """Walk ``hmr_world/`` for files matching ``glob_pattern`` (e.g.
        ``*_smpl_world.npz``). ``stem_strip`` is the trailing tag to
        remove from each file's stem to recover the
        ``{shot_id}__{player_id}`` key (e.g. ``_smpl_world`` or
        ``_kp2d``). Returns one row per file with ``shot_id``,
        ``player_id`` (the bare id, not the keyed filename), and the
        operator-supplied ``player_name`` when available."""
        hmr_dir = output_dir / "hmr_world"
        if not hmr_dir.exists():
            return []
        names = _player_name_index()
        rows: list[dict[str, str]] = []
        for path in sorted(hmr_dir.glob(glob_pattern)):
            stem = path.stem
            if stem.endswith(stem_strip):
                stem = stem[: -len(stem_strip)]
            if "__" in stem:
                shot_id, _, pid = stem.partition("__")
            else:
                shot_id, pid = "", stem
            if shot is not None and shot_id != shot:
                continue
            # Lookup order: exact (shot, pid) → cross-shot fallback by
            # bare player_id → empty. The cross-shot fallback handles
            # Matip-style cases where one shot has a player_id assigned
            # and another shot only has the player_name set.
            name = (
                names.get((shot_id, pid))
                or names.get(("", pid), "")
            )
            rows.append({
                "shot_id": shot_id,
                "player_id": pid,
                "player_name": name,
            })
        return rows

    @app.get("/hmr_world/kp2d_players")
    def list_kp2d_players(shot: str | None = None):
        """List ``(shot_id, player_id)`` pairs with kp2d artefacts.

        ``?shot=xxx`` filters to one shot. Each row is
        ``{shot_id, player_id, player_name}``.
        """
        return {"players": _list_hmr_files("*_kp2d.json", "_kp2d", shot)}

    @app.get("/hmr_world/kp2d_preview")
    def get_kp2d_preview(player_id: str, shot: str | None = None):
        kp2d_path = _resolve_hmr_path("_kp2d.json", shot, player_id)
        if not kp2d_path.exists():
            raise HTTPException(status_code=404, detail=f"kp2d track not found: {player_id}")
        try:
            data = json.loads(kp2d_path.read_text())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load kp2d: {exc}")
        return data

    @app.get("/hmr_world/players")
    def list_hmr_players(shot: str | None = None):
        """List ``(shot_id, player_id)`` pairs with HMR-world output.

        ``?shot=xxx`` filters to one shot. Each row is
        ``{shot_id, player_id, player_name}``. The player_id field is
        the bare id (no shot prefix); follow-up calls to /preview need
        both ``?shot=`` and ``?player_id=``.
        """
        return {"players": _list_hmr_files("*_smpl_world.npz", "_smpl_world", shot)}

    @app.get("/hmr_world/preview")
    def get_hmr_preview(
        player_id: str,
        request: Request,
        include_pose: int = 0,
        shot: str | None = None,
    ):
        try:
            npz_path = _resolve_hmr_path("_smpl_world.npz", shot, player_id)
        except HTTPException:
            # Diagnostic: log the Referer so we can pinpoint *which*
            # client page is sending malformed player_id values (the
            # ``[object Object]`` symptom seen when a JS list of objects
            # was iterated as if it were a list of strings).
            referer = request.headers.get("referer", "<no-referer>")
            ua = request.headers.get("user-agent", "<no-ua>")[:60]
            logging.warning(
                "/hmr_world/preview rejected player_id=%r shot=%r referer=%s ua=%s",
                player_id, shot, referer, ua,
            )
            raise
        if not npz_path.exists():
            raise HTTPException(status_code=404, detail=f"hmr_world track not found: {player_id}")
        try:
            z = np.load(npz_path, allow_pickle=False)
            frames = z["frames"].tolist()
            root_t = z["root_t"].tolist()
            confidence = z["confidence"].tolist()
            payload = {
                "player_id": player_id,
                "shot_id": str(z["shot_id"]) if "shot_id" in z.files else "",
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

    # ------------------------------------------------------------------
    # Refined poses (stage 6 — cross-shot fusion of HMR World output)
    # ------------------------------------------------------------------

    def _refined_pose_path(player_id: str) -> Path:
        if not _PID_RE.fullmatch(player_id):
            raise HTTPException(status_code=400, detail="Invalid player_id")
        rp_dir = (output_dir / "refined_poses").resolve()
        path = (rp_dir / f"{player_id}_refined.npz").resolve()
        if not path.is_relative_to(rp_dir):
            raise HTTPException(status_code=400, detail="Invalid path")
        return path

    def _refined_diag_path(player_id: str) -> Path:
        if not _PID_RE.fullmatch(player_id):
            raise HTTPException(status_code=400, detail="Invalid player_id")
        rp_dir = (output_dir / "refined_poses").resolve()
        path = (rp_dir / f"{player_id}_diagnostics.json").resolve()
        if not path.is_relative_to(rp_dir):
            raise HTTPException(status_code=400, detail="Invalid path")
        return path

    @app.get("/refined_poses/players")
    def list_refined_players():
        """List players with a fused track on the reference timeline.

        Each row is ``{player_id, player_name, contributing_shots,
        n_frames, mean_confidence}``. The ``player_name`` falls back
        to the operator-supplied name from any track that named the id.
        """
        rp_dir = output_dir / "refined_poses"
        if not rp_dir.exists():
            return {"players": []}
        names = _player_name_index()
        rows: list[dict] = []
        for path in sorted(rp_dir.glob("*_refined.npz")):
            pid = path.stem.removesuffix("_refined")
            try:
                z = np.load(path, allow_pickle=False)
                contributing = (
                    [str(s) for s in z["contributing_shots"]]
                    if "contributing_shots" in z.files
                    else []
                )
                conf = z["confidence"] if "confidence" in z.files else None
                n_frames = int(z["frames"].shape[0]) if "frames" in z.files else 0
                mean_conf = float(conf.mean()) if conf is not None and conf.size else 0.0
                view_count = (
                    z["view_count"].astype(int).tolist()
                    if "view_count" in z.files
                    else []
                )
            except Exception:
                contributing = []
                n_frames = 0
                mean_conf = 0.0
                view_count = []
            single_view_frames = sum(1 for v in view_count if v <= 1)
            multi_view_frames = sum(1 for v in view_count if v > 1)
            rows.append({
                "player_id": pid,
                "player_name": names.get(("", pid), ""),
                "contributing_shots": contributing,
                "n_frames": n_frames,
                "mean_confidence": mean_conf,
                "single_view_frames": single_view_frames,
                "multi_view_frames": multi_view_frames,
            })
        return {"players": rows}

    @app.get("/refined_poses/preview")
    def get_refined_preview(player_id: str, include_pose: int = 0):
        """Per-player time-series on the reference timeline.

        Returns ``{player_id, frames, root_t, confidence, view_count,
        contributing_shots}``. ``include_pose=1`` adds ``thetas``,
        ``root_R``, ``betas`` (matches /hmr_world/preview's contract).
        """
        path = _refined_pose_path(player_id)
        if not path.exists():
            raise HTTPException(
                status_code=404, detail=f"refined track not found: {player_id}",
            )
        try:
            z = np.load(path, allow_pickle=False)
            payload = {
                "player_id": player_id,
                "frames": z["frames"].tolist(),
                "root_t": z["root_t"].tolist(),
                "confidence": z["confidence"].tolist(),
                "view_count": z["view_count"].astype(int).tolist(),
                "contributing_shots": [str(s) for s in z["contributing_shots"]],
            }
            if include_pose:
                payload["thetas"] = z["thetas"].tolist()
                payload["root_R"] = z["root_R"].tolist()
                payload["betas"] = z["betas"].tolist()
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to load refined track: {exc}",
            )
        return payload

    @app.get("/refined_poses/diagnostics")
    def get_refined_diagnostics(player_id: str):
        """Per-player fusion diagnostics: per-frame disagreement, dropped
        views, low-coverage / high-disagreement flags, plus a summary."""
        path = _refined_diag_path(player_id)
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"refined diagnostics not found: {player_id}",
            )
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to load diagnostics: {exc}",
            )

    @app.get("/refined_poses/summary")
    def get_refined_summary():
        """Pipeline-level summary written by the refined_poses stage.

        Returns the contents of ``refined_poses_summary.json`` (or {} if
        the stage hasn't run yet). The dashboard's status panel reads
        this to render the per-pipeline counters line.
        """
        path = output_dir / "refined_poses" / "refined_poses_summary.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to load summary: {exc}",
            )

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

    @app.get("/ball-anchor-editor", include_in_schema=False)
    def serve_ball_anchor_editor():
        ball_editor_path = static_dir / "ball_anchor_editor.html"
        if not ball_editor_path.exists():
            raise HTTPException(status_code=404, detail="ball_anchor_editor.html not found")
        return FileResponse(str(ball_editor_path), headers=_NO_STORE)

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

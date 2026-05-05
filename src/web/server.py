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
    GET    /hmr_world/preview         (lightweight summary for one player)
    GET    /hmr_world/players         (list of available player_id strings)
    GET    /ball/preview              (ball_track.json)

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
from src.utils.pitch_landmarks import LANDMARK_CATALOGUE

# ---------------------------------------------------------------------------
# Stage completion (broadcast-mono)
# ---------------------------------------------------------------------------

STAGE_ORDER: list[str] = [
    "prepare_shots",
    "tracking",
    "camera",
    "pose_2d",
    "hmr_world",
    "ball",
    "export",
]

_STAGE_COMPLETE = {
    "prepare_shots": lambda d: (d / "shots" / "shots_manifest.json").exists(),
    "tracking": lambda d: any((d / "tracks").glob("*_tracks.json")),
    "camera": lambda d: (d / "camera" / "camera_track.json").exists(),
    "pose_2d": lambda d: any((d / "pose_2d").glob("*_pose_2d.json")),
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
    "pose_2d": ["pose_2d"],
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

    @app.get("/")
    def index():
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
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
        return FileResponse(str(viewer_path))

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

    @app.get("/landmarks")
    def get_landmarks():
        return {
            "landmarks": [
                {"name": lm.name, "world_xyz": list(lm.world_xyz)}
                for lm in sorted(LANDMARK_CATALOGUE.values(), key=lambda lm: lm.name)
            ]
        }

    @app.get("/anchors")
    def get_anchors():
        anchor_path = output_dir / "camera" / "anchors.json"
        if not anchor_path.exists():
            return {"clip_id": "", "image_size": [0, 0], "anchors": []}
        try:
            anchor_set = AnchorSet.load(anchor_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load anchors: {exc}")
        return _anchor_set_to_dict(anchor_set)

    class AnchorPayload(BaseModel):
        clip_id: str
        image_size: tuple[int, int]
        anchors: list[dict[str, Any]]

    @app.post("/anchors")
    def post_anchors(payload: AnchorPayload):
        try:
            anchor_set = _dict_to_anchor_set(payload.dict())
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid anchor payload: {exc}")
        anchor_path = output_dir / "camera" / "anchors.json"
        anchor_set.save(anchor_path)
        return {"saved": True, "path": str(anchor_path), "count": len(anchor_set.anchors)}

    @app.get("/camera/track")
    def get_camera_track():
        track_path = output_dir / "camera" / "camera_track.json"
        if not track_path.exists():
            return {"clip_id": "", "fps": 0.0, "image_size": [0, 0], "t_world": [0.0, 0.0, 0.0], "frames": []}
        try:
            track = CameraTrack.load(track_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load camera track: {exc}")
        return asdict(track)

    @app.get("/hmr_world/players")
    def list_hmr_players():
        hmr_dir = output_dir / "hmr_world"
        if not hmr_dir.exists():
            return {"players": []}
        ids: list[str] = []
        for npz_path in sorted(hmr_dir.glob("*_smpl_world.npz")):
            ids.append(npz_path.stem.replace("_smpl_world", ""))
        return {"players": ids}

    @app.get("/hmr_world/preview")
    def get_hmr_preview(player_id: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", player_id):
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
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load hmr_world: {exc}")
        return {
            "player_id": player_id,
            "frames": frames,
            "root_t": root_t,
            "confidence": confidence,
        }

    @app.get("/ball/preview")
    def get_ball_preview():
        ball_path = output_dir / "ball" / "ball_track.json"
        if not ball_path.exists():
            return {"clip_id": "", "fps": 0.0, "frames": [], "flight_segments": []}
        try:
            track = BallTrack.load(ball_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load ball track: {exc}")
        return _ball_track_to_dict(track)

    @app.get("/anchor_editor")
    def anchor_editor_page():
        editor_path = static_dir / "anchor_editor.html"
        if not editor_path.exists():
            raise HTTPException(status_code=404, detail="anchor_editor.html not found")
        return FileResponse(str(editor_path))

    return app


# ---------------------------------------------------------------------------
# Schema (de)serialisation helpers
# ---------------------------------------------------------------------------


def _anchor_set_to_dict(anchor_set: AnchorSet) -> dict[str, Any]:
    return {
        "clip_id": anchor_set.clip_id,
        "image_size": list(anchor_set.image_size),
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
            }
            for a in anchor_set.anchors
        ],
    }


def _dict_to_anchor_set(data: dict[str, Any]) -> AnchorSet:
    from src.schemas.anchor import Anchor, LandmarkObservation

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
                for lm in a["landmarks"]
            ),
        )
        for a in data.get("anchors", [])
    )
    image_size = tuple(data.get("image_size", (0, 0)))
    return AnchorSet(
        clip_id=str(data.get("clip_id", "")),
        image_size=(int(image_size[0]), int(image_size[1])),
        anchors=anchors,
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

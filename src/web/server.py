import io
import json
import logging
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.pipeline.config import _deep_merge, load_config
from src.pipeline.runner import run_pipeline

# ---------------------------------------------------------------------------
# Stage completion helpers
# ---------------------------------------------------------------------------

STAGE_ORDER = [
    "segmentation",
    "calibration",
    "tracking",
    "pose",
    "sync",
    "triangulation",
    "smpl_fitting",
    "export",
]

# Directories/files to delete when re-running a stage
_STAGE_ARTIFACTS: dict[str, list[str]] = {
    "segmentation": ["shots"],
    "calibration": ["calibration"],
    "tracking": ["tracks", "matching"],
    "pose": ["poses"],
    "sync": ["sync"],
    "triangulation": ["triangulated"],
    "smpl_fitting": ["smpl"],
    "export": ["export"],
}

_STAGE_COMPLETE: dict[str, Any] = {
    "segmentation": lambda d: (d / "shots" / "shots_manifest.json").exists(),
    "calibration": lambda d: any((d / "calibration").glob("*_calibration.json")),
    "tracking": lambda d: any((d / "tracks").glob("*_tracks.json")),
    "pose": lambda d: any((d / "poses").glob("*_poses.json")),
    "sync": lambda d: (d / "sync" / "sync_map.json").exists(),
    "triangulation": lambda d: any((d / "triangulated").glob("*.npz")),
    "smpl_fitting": lambda d: any((d / "smpl").glob("*.npz")),
    "export": lambda d: (d / "export" / "gltf" / "scene.glb").exists(),
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


# ---------------------------------------------------------------------------
# Log capture helpers
# ---------------------------------------------------------------------------


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


def _run_job(job: Job, output_dir: Path, config_path: Path | None, params: RunRequest) -> None:
    handler = _LogQueueHandler(job)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.root.addHandler(handler)
    writer = _QueueWriter(job)
    old_stdout = sys.stdout
    sys.stdout = writer  # type: ignore[assignment]
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
        job.status = "error"
        job.error = str(exc)
        job.log_queue.put(f"ERROR: {exc}")
    finally:
        sys.stdout = old_stdout
        logging.root.removeHandler(handler)
        job.log_queue.put(None)  # sentinel


# ---------------------------------------------------------------------------
# SSE log stream generator
# ---------------------------------------------------------------------------


def _log_stream(job: Job):
    # Replay accumulated lines for late subscribers
    for line in list(job.log_lines):
        yield f"event: log\ndata: {json.dumps({'line': line})}\n\n"
    # Stream new lines until done
    while True:
        try:
            item = job.log_queue.get(timeout=0.5)
            if item is None:
                break
            yield f"event: log\ndata: {json.dumps({'line': item})}\n\n"
        except Empty:
            if job.status in ("done", "error"):
                break
    yield f"event: done\ndata: {json.dumps({'status': job.status})}\n\n"


# ---------------------------------------------------------------------------
# Video range response
# ---------------------------------------------------------------------------


def _parse_range(range_header: str, file_size: int) -> tuple[int, int]:
    """Parse a 'bytes=start-end' Range header."""
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
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ------------------------------------------------------------------
    # GET /
    # ------------------------------------------------------------------

    @app.get("/")
    def index():
        return FileResponse(str(static_dir / "index.html"))

    # ------------------------------------------------------------------
    # GET /api/stages
    # ------------------------------------------------------------------

    @app.get("/api/stages")
    def get_stages():
        out = output_dir
        return [
            {
                "name": name,
                "index": i + 1,
                "complete": _STAGE_COMPLETE[name](out),
            }
            for i, name in enumerate(STAGE_ORDER)
        ]

    # ------------------------------------------------------------------
    # GET /api/config
    # ------------------------------------------------------------------

    @app.get("/api/config")
    def get_config():
        return load_config(app.state.config_path)

    # ------------------------------------------------------------------
    # POST /api/run
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # GET /api/jobs/{job_id}/status
    # ------------------------------------------------------------------

    @app.get("/api/jobs/{job_id}/status")
    def job_status(job_id: str):
        with _jobs_lock:
            job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job_id": job_id, "stages": job.stages, "status": job.status, "error": job.error}

    # ------------------------------------------------------------------
    # GET /api/jobs/{job_id}/logs  (SSE)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # GET /api/output/matching/positions
    # ------------------------------------------------------------------

    @app.get("/api/output/matching/positions")
    def matching_positions():
        """Return per-player average pitch positions from matching + tracks."""
        match_path = output_dir / "matching" / "player_matches.json"
        if not match_path.exists():
            raise HTTPException(status_code=404, detail="player_matches.json not found")

        matches = json.loads(match_path.read_text())
        tracks_dir = output_dir / "tracks"

        # Load all tracks files into a lookup: {shot_id: {track_id: Track}}
        tracks_by_shot: dict[str, dict[str, list[dict]]] = {}
        for tf in sorted(tracks_dir.glob("*_tracks.json")) if tracks_dir.exists() else []:
            td = json.loads(tf.read_text())
            shot_id = td.get("shot_id", tf.stem)
            track_map: dict[str, list[dict]] = {}
            for track in td.get("tracks", []):
                track_map[track["track_id"]] = track.get("frames", [])
            tracks_by_shot[shot_id] = track_map

        positions = []
        for player in matches.get("matched_players", []):
            xs, ys = [], []
            for view in player.get("views", []):
                frames = tracks_by_shot.get(view["shot_id"], {}).get(view["track_id"], [])
                for f in frames:
                    pp = f.get("pitch_position")
                    if pp and len(pp) >= 2:
                        xs.append(pp[0])
                        ys.append(pp[1])
            if xs:
                positions.append({
                    "player_id": player["player_id"],
                    "team": player["team"],
                    "x": sum(xs) / len(xs),
                    "y": sum(ys) / len(ys),
                })
        return positions

    # ------------------------------------------------------------------
    # GET /api/output/{stage}
    # ------------------------------------------------------------------

    @app.get("/api/output/{stage}")
    def get_output(stage: str):
        if stage not in STAGE_ORDER:
            raise HTTPException(status_code=404, detail=f"Unknown stage: {stage}")

        if stage == "segmentation":
            p = output_dir / "shots" / "shots_manifest.json"
            if not p.exists():
                raise HTTPException(status_code=404, detail="Stage output not found")
            return json.loads(p.read_text())

        if stage == "sync":
            p = output_dir / "sync" / "sync_map.json"
            if not p.exists():
                raise HTTPException(status_code=404, detail="Stage output not found")
            return json.loads(p.read_text())

        if stage == "triangulation":
            import numpy as _np
            tri_dir = output_dir / "triangulated"
            npz_files = sorted(tri_dir.glob("*.npz")) if tri_dir.exists() else []
            if not npz_files:
                raise HTTPException(status_code=404, detail="Stage output not found")
            # Also load matching data for team info
            match_path = output_dir / "matching" / "player_matches.json"
            teams: dict[str, str] = {}
            if match_path.exists():
                md = json.loads(match_path.read_text())
                for mp in md.get("matched_players", []):
                    teams[mp["player_id"]] = mp.get("team", "unknown")
            players = []
            for npz_path in npz_files:
                data = _np.load(npz_path, allow_pickle=False)
                pid = str(data["player_id"])
                # Prefer the name/team stored in the NPZ (carried from
                # tracks via triangulation); fall back to matching for older
                # NPZs that predate the schema change.
                pname = str(data["player_name"]) if "player_name" in data else ""
                nteam = str(data["team"]) if "team" in data else ""
                if not nteam:
                    nteam = teams.get(pid, "unknown")
                # NaN → None in the JSON payload so the bird's-eye renderer
                # can distinguish "no data" from "position (0,0,0)".
                positions = data["positions"]
                confidences = data["confidences"]
                pos_list = [
                    [
                        [None if _np.isnan(v) else float(v) for v in joint]
                        for joint in frame
                    ]
                    for frame in positions
                ]
                conf_list = [
                    [None if _np.isnan(c) else float(c) for c in frame]
                    for frame in confidences
                ]
                players.append({
                    "player_id": pid,
                    "player_name": pname,
                    "team": nteam,
                    "fps": float(data["fps"]),
                    "start_frame": int(data["start_frame"]),
                    "num_frames": int(positions.shape[0]),
                    "positions": pos_list,
                    "confidences": conf_list,
                })
            return {"players": players}

        if stage == "smpl_fitting":
            import numpy as _np
            smpl_dir = output_dir / "smpl"
            npz_files = sorted(smpl_dir.glob("*.npz")) if smpl_dir.exists() else []
            if not npz_files:
                raise HTTPException(status_code=404, detail="Stage output not found")
            players = []
            for npz_path in npz_files:
                data = _np.load(npz_path, allow_pickle=False)
                players.append({
                    "player_id": str(data["player_id"]),
                    "fps": float(data["fps"]),
                    "num_frames": int(data["poses"].shape[0]),
                    "betas_norm": float(_np.linalg.norm(data["betas"])),
                })
            return {"players": players}

        if stage == "export":
            manifest = output_dir / "export" / "export_result.json"
            if not manifest.exists():
                raise HTTPException(status_code=404, detail="Stage output not found")
            return json.loads(manifest.read_text())

        # Multi-file stages: calibration, tracking, pose
        dir_map = {
            "calibration": (output_dir / "calibration", "*_calibration.json"),
            "tracking": (output_dir / "tracks", "*_tracks.json"),
            "pose": (output_dir / "poses", "*_poses.json"),
        }
        if stage not in dir_map:
            raise HTTPException(status_code=404, detail="Stage output not found")
        folder, pattern = dir_map[stage]
        files = sorted(folder.glob(pattern)) if folder.exists() else []
        if not files:
            raise HTTPException(status_code=404, detail="Stage output not found")

        result: dict[str, Any] = {}
        for f in files:
            data = json.loads(f.read_text())
            shot_id = data.get("shot_id", f.stem)
            result[shot_id] = data
        return {"shots": result}

    # ------------------------------------------------------------------
    # DELETE /api/output/{stage} — remove stage artifacts for re-run
    # ------------------------------------------------------------------

    @app.delete("/api/output/{stage}")
    def delete_output(stage: str):
        import shutil
        if stage not in STAGE_ORDER:
            raise HTTPException(status_code=404, detail=f"Unknown stage: {stage}")
        dirs = _STAGE_ARTIFACTS.get(stage, [])
        removed = []
        for d in dirs:
            target = output_dir / d
            if not target.exists():
                continue
            if d == "calibration":
                # Only delete generated calibration files
                for f in target.glob("*_calibration.json"):
                    f.unlink()
                    removed.append(f.name)
            else:
                shutil.rmtree(target)
                removed.append(d)
        return {"stage": stage, "removed": removed}

    # ------------------------------------------------------------------
    # PUT /api/calibration/camera-position/{shot_id} — manual camera placement
    # ------------------------------------------------------------------

    @app.put("/api/calibration/camera-position/{shot_id}")
    async def put_camera_position(shot_id: str, request: Request):
        """Compute and save calibration from a manually specified camera position."""
        import cv2 as _cv2
        import numpy as _np

        if not re.fullmatch(r"[A-Za-z0-9_-]+", shot_id):
            raise HTTPException(status_code=400, detail="Invalid shot ID")

        body = await request.json()
        cx = float(body["x"])
        cy = float(body["y"])
        cz = float(body["z"])
        look_x = float(body["lookX"])
        look_y = float(body["lookY"])
        focal = float(body["focal"])
        frame_idx = int(body.get("frame", 0))

        # Compute R and t from camera world position + look-at point
        C = _np.array([cx, cy, cz], dtype=_np.float64)
        target = _np.array([look_x, look_y, 0.0], dtype=_np.float64)
        forward = target - C
        fwd_len = _np.linalg.norm(forward)
        if fwd_len < 1e-6:
            raise HTTPException(status_code=400, detail="Camera position too close to look-at point")
        forward = forward / fwd_len

        # World up = +z (pitch plane is z=0, cameras are above)
        world_up = _np.array([0.0, 0.0, 1.0])
        right = _np.cross(forward, world_up)
        right_len = _np.linalg.norm(right)
        if right_len < 1e-6:
            # Camera looking straight down — use y as up
            world_up = _np.array([0.0, 1.0, 0.0])
            right = _np.cross(forward, world_up)
            right_len = _np.linalg.norm(right)
        right = right / right_len
        up = _np.cross(right, forward)

        # Rotation matrix: columns are right, -up, forward in camera frame
        # OpenCV convention: camera looks along +z, x=right, y=down
        R_cam_to_world = _np.column_stack([right, -up, forward])
        R = R_cam_to_world.T  # world-to-camera
        rvec, _ = _cv2.Rodrigues(R)
        tvec = (-R @ C.reshape(3, 1)).flatten()

        K = [[focal, 0.0, 960.0], [0.0, focal, 540.0], [0.0, 0.0, 1.0]]

        from src.schemas.calibration import CalibrationResult, CameraFrame
        cf = CameraFrame(
            frame=frame_idx,
            intrinsic_matrix=K,
            rotation_vector=rvec.flatten().tolist(),
            translation_vector=tvec.tolist(),
            reprojection_error=0.0,
            num_correspondences=0,
            confidence=1.0,
            tracked_landmark_types=["manual_placement"],
        )
        cal = CalibrationResult(shot_id=shot_id, camera_type="static", frames=[cf])
        cal_dir = output_dir / "calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)
        cal.save(cal_dir / f"{shot_id}_calibration.json")

        return {
            "shot_id": shot_id,
            "camera_position": [cx, cy, cz],
            "look_at": [look_x, look_y, 0.0],
            "focal": focal,
            "rvec": rvec.flatten().tolist(),
            "tvec": tvec.tolist(),
        }

    # ------------------------------------------------------------------
    # POST /api/calibration/refine — bundle adjustment using tracked players
    # ------------------------------------------------------------------

    @app.post("/api/calibration/refine")
    def refine_calibration():
        from src.schemas.calibration import CalibrationResult
        from src.schemas.tracks import TracksResult
        from src.schemas.sync_map import SyncMap
        from src.schemas.player_matches import PlayerMatches
        from src.utils.bundle_adjust import refine_calibrations

        # Load all required data
        cal_dir = output_dir / "calibration"
        tracks_dir = output_dir / "tracks"
        sync_path = output_dir / "sync" / "sync_map.json"
        match_path = output_dir / "matching" / "player_matches.json"

        if not sync_path.exists():
            raise HTTPException(status_code=400, detail="sync_map.json required for refinement")
        if not match_path.exists():
            raise HTTPException(status_code=400, detail="player_matches.json required for refinement")

        calibrations = {}
        for f in sorted(cal_dir.glob("*_calibration.json")):
            cal = CalibrationResult.load(f)
            calibrations[cal.shot_id] = cal

        tracks = {}
        for f in sorted(tracks_dir.glob("*_tracks.json")):
            tr = TracksResult.load(f)
            tracks[tr.shot_id] = tr

        sync_map = SyncMap.load(sync_path)
        matches = PlayerMatches.load(match_path)

        refined = refine_calibrations(calibrations, tracks, sync_map, matches)

        # Save refined calibrations
        for shot_id, cal in refined.items():
            cal.save(cal_dir / f"{shot_id}_calibration.json")

        return {"refined": list(refined.keys())}

    # ------------------------------------------------------------------
    # POST /api/tracks/auto-match — run cross-shot matching on existing tracks
    # ------------------------------------------------------------------

    @app.post("/api/tracks/auto-match")
    def auto_match_tracks():
        from src.stages.matching import CrossViewMatchingStage
        from src.schemas.tracks import TracksResult

        sync_path = output_dir / "sync" / "sync_map.json"
        if not sync_path.exists():
            raise HTTPException(status_code=400, detail="sync_map.json required for auto-matching")

        tracks_dir = output_dir / "tracks"
        track_files = sorted(tracks_dir.glob("*_tracks.json")) if tracks_dir.exists() else []
        if not track_files:
            raise HTTPException(status_code=400, detail="No tracks files found")

        # Save existing player_names before matching (so auto-match doesn't overwrite them)
        existing_names: dict[str, dict[str, str]] = {}  # {shot_id: {track_id: player_name}}
        for tf in track_files:
            tr = TracksResult.load(tf)
            names = {}
            for track in tr.tracks:
                if track.player_name:
                    names[track.track_id] = track.player_name
            if names:
                existing_names[tr.shot_id] = names

        cfg = load_config(app.state.config_path)
        stage = CrossViewMatchingStage(config=cfg, output_dir=output_dir)
        stage.run()

        # Re-read and update tracks with player_id, restoring saved names
        matches = json.loads((output_dir / "matching" / "player_matches.json").read_text())
        for mp in matches.get("matched_players", []):
            for view in mp.get("views", []):
                track_path = tracks_dir / f"{view['shot_id']}_tracks.json"
                if track_path.exists():
                    tr = TracksResult.load(track_path)
                    for track in tr.tracks:
                        if track.track_id == view["track_id"]:
                            track.player_id = mp["player_id"]
                            # Restore manually set name if it existed
                            saved = existing_names.get(view["shot_id"], {}).get(view["track_id"])
                            if saved:
                                track.player_name = saved
                    tr.save(track_path)

        return matches

    # ------------------------------------------------------------------
    # POST /api/tracks/dedupe-by-name — merge tracks sharing a player_name
    # ------------------------------------------------------------------

    @app.post("/api/tracks/dedupe-by-name")
    def dedupe_tracks_by_name():
        from src.schemas.tracks import TracksResult

        tracks_dir = output_dir / "tracks"
        track_files = sorted(tracks_dir.glob("*_tracks.json")) if tracks_dir.exists() else []
        if not track_files:
            raise HTTPException(status_code=400, detail="No tracks files found")

        # Collect all named tracks: {player_name: first player_id seen}
        name_to_pid: dict[str, str] = {}
        all_tracks_data: list[tuple[Path, TracksResult]] = []
        for tf in track_files:
            tr = TracksResult.load(tf)
            all_tracks_data.append((tf, tr))
            for track in tr.tracks:
                if track.player_name and track.player_name not in name_to_pid:
                    pid = track.player_id or track.track_id
                    name_to_pid[track.player_name] = pid

        # Apply: for every track with a player_name, set player_id to the canonical one
        merged = 0
        for tf, tr in all_tracks_data:
            changed = False
            for track in tr.tracks:
                if track.player_name and track.player_name in name_to_pid:
                    canonical_pid = name_to_pid[track.player_name]
                    if track.player_id != canonical_pid:
                        track.player_id = canonical_pid
                        changed = True
                        merged += 1
            if changed:
                tr.save(tf)

        # Update player_matches.json if it exists
        match_path = output_dir / "matching" / "player_matches.json"
        if match_path.exists():
            from src.schemas.player_matches import PlayerMatches, MatchedPlayer, PlayerView
            matches = PlayerMatches.load(match_path)
            # Rebuild from current tracks state
            pid_views: dict[str, list[tuple[str, str, str]]] = {}  # pid -> [(shot_id, track_id, team)]
            for _, tr in all_tracks_data:
                for track in tr.tracks:
                    if track.class_name == 'ball':
                        continue
                    pid = track.player_id or track.track_id
                    pid_views.setdefault(pid, []).append((tr.shot_id, track.track_id, track.team))
            new_players = []
            for pid, views in sorted(pid_views.items()):
                new_players.append(MatchedPlayer(
                    player_id=pid,
                    team=views[0][2],
                    views=[PlayerView(shot_id=s, track_id=t) for s, t, _ in views],
                ))
            PlayerMatches(matched_players=new_players).save(match_path)

        return {"merged": merged}

    # ------------------------------------------------------------------
    # POST /api/tracks/ignore-unknown/{shot_id} — mark unnamed players as "ignore"
    # ------------------------------------------------------------------

    @app.post("/api/tracks/ignore-unknown/{shot_id}")
    def ignore_unknown_tracks(shot_id: str):
        from src.schemas.tracks import TracksResult

        if not re.fullmatch(r"[A-Za-z0-9_-]+", shot_id):
            raise HTTPException(status_code=400, detail="Invalid shot ID")
        track_path = output_dir / "tracks" / f"{shot_id}_tracks.json"
        if not track_path.exists():
            raise HTTPException(status_code=404, detail=f"Tracks not found for {shot_id}")

        tr = TracksResult.load(track_path)
        count = 0
        for track in tr.tracks:
            if track.class_name == "ball":
                continue
            if not track.player_name:
                track.player_name = "ignore"
                count += 1
        tr.save(track_path)
        return {"shot_id": shot_id, "count": count}

    # ------------------------------------------------------------------
    # POST /api/tracks/delete-ignored — remove all tracks named "ignore"
    # ------------------------------------------------------------------

    @app.post("/api/tracks/delete-ignored")
    def delete_ignored_tracks():
        from src.schemas.tracks import TracksResult

        tracks_dir = output_dir / "tracks"
        track_files = sorted(tracks_dir.glob("*_tracks.json")) if tracks_dir.exists() else []
        if not track_files:
            raise HTTPException(status_code=400, detail="No tracks files found")

        deleted = 0
        for tf in track_files:
            tr = TracksResult.load(tf)
            before = len(tr.tracks)
            tr.tracks = [t for t in tr.tracks if t.player_name != "ignore"]
            removed = before - len(tr.tracks)
            if removed:
                deleted += removed
                tr.save(tf)
        return {"deleted": deleted}

    # ------------------------------------------------------------------
    # POST /api/tracks/split — split a track at a given frame
    # ------------------------------------------------------------------

    @app.post("/api/tracks/split")
    async def split_track(request: Request):
        from src.schemas.tracks import TracksResult, Track, TrackFrame

        body = await request.json()
        shot_id = body.get("shot_id")
        track_id = body.get("track_id")
        split_frame = int(body.get("split_frame", 0))

        if not shot_id or not track_id:
            raise HTTPException(status_code=400, detail="shot_id and track_id required")

        track_path = output_dir / "tracks" / f"{shot_id}_tracks.json"
        if not track_path.exists():
            raise HTTPException(status_code=404, detail=f"Tracks not found for {shot_id}")

        tr = TracksResult.load(track_path)

        # Find the target track
        target = None
        for t in tr.tracks:
            if t.track_id == track_id:
                target = t
                break
        if target is None:
            raise HTTPException(status_code=404, detail=f"Track {track_id} not found")

        before = [f for f in target.frames if f.frame < split_frame]
        after = [f for f in target.frames if f.frame >= split_frame]

        if not before or not after:
            raise HTTPException(status_code=400, detail="Split frame must be between the track's first and last frame")

        # Generate new IDs
        existing_track_ids = {t.track_id for t in tr.tracks}
        n = 1
        while f"T{n:03d}" in existing_track_ids:
            n += 1
        new_track_id = f"T{n:03d}"

        existing_player_ids = {t.player_id for t in tr.tracks if t.player_id}
        # Also check other shots
        tracks_dir = output_dir / "tracks"
        for tf in tracks_dir.glob("*_tracks.json"):
            if tf == track_path:
                continue
            other = TracksResult.load(tf)
            for t in other.tracks:
                if t.player_id:
                    existing_player_ids.add(t.player_id)
        pn = 1
        while f"P{pn:03d}" in existing_player_ids:
            pn += 1
        new_player_id = f"P{pn:03d}"

        # Split: original keeps frames before, new track gets frames after
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
            "original_track_id": track_id,
            "new_track_id": new_track_id,
            "new_player_id": new_player_id,
            "original_frames": len(before),
            "new_frames": len(after),
        }

    # ------------------------------------------------------------------
    # PATCH /api/tracks/{shot_id}/{track_id} — update player name/id
    # ------------------------------------------------------------------

    @app.patch("/api/tracks/{shot_id}/{track_id}")
    async def patch_track(shot_id: str, track_id: str, request: Request):
        from src.schemas.tracks import TracksResult

        if not re.fullmatch(r"[A-Za-z0-9_-]+", shot_id):
            raise HTTPException(status_code=400, detail="Invalid shot ID")

        body = await request.json()
        track_path = output_dir / "tracks" / f"{shot_id}_tracks.json"
        if not track_path.exists():
            raise HTTPException(status_code=404, detail=f"Tracks not found for {shot_id}")

        tr = TracksResult.load(track_path)
        target = None
        for track in tr.tracks:
            if track.track_id == track_id:
                target = track
                break
        if target is None:
            raise HTTPException(status_code=404, detail=f"Track {track_id} not found in {shot_id}")

        if "player_id" in body:
            target.player_id = str(body["player_id"])
        if "player_name" in body:
            target.player_name = str(body["player_name"])
        if "team" in body:
            target.team = str(body["team"])

        tr.save(track_path)
        return {"shot_id": shot_id, "track_id": track_id, "player_id": target.player_id, "player_name": target.player_name, "team": target.team}

    # ------------------------------------------------------------------
    # PATCH /api/sync  — manually override a single shot's frame offset
    # ------------------------------------------------------------------

    @app.patch("/api/sync")
    async def patch_sync(request: Request):
        body = await request.json()
        shot_id = body.get("shot_id")
        frame_offset = body.get("frame_offset")

        if not shot_id or not isinstance(shot_id, str):
            raise HTTPException(status_code=400, detail="shot_id is required")
        if frame_offset is None or not isinstance(frame_offset, (int, float)):
            raise HTTPException(status_code=400, detail="frame_offset must be a number")
        frame_offset = int(round(frame_offset))

        sync_path = output_dir / "sync" / "sync_map.json"
        if not sync_path.exists():
            raise HTTPException(status_code=404, detail="sync_map.json not found")

        sync_data = json.loads(sync_path.read_text())

        if shot_id == sync_data.get("reference_shot"):
            raise HTTPException(status_code=400, detail="Cannot adjust the reference shot offset")

        target = None
        for alignment in sync_data.get("alignments", []):
            if alignment.get("shot_id") == shot_id:
                target = alignment
                break
        if target is None:
            raise HTTPException(status_code=404, detail=f"Shot '{shot_id}' not found in sync_map")

        # Recompute overlap_frames using shots_manifest
        manifest_path = output_dir / "shots" / "shots_manifest.json"
        overlap_frames: list[int] = []
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            ref_id = sync_data["reference_shot"]
            shots_by_id = {s["id"]: s for s in manifest.get("shots", [])}
            ref_shot = shots_by_id.get(ref_id)
            tgt_shot = shots_by_id.get(shot_id)
            if ref_shot and tgt_shot:
                ref_frames = ref_shot["end_frame"] - ref_shot["start_frame"]
                tgt_frames = tgt_shot["end_frame"] - tgt_shot["start_frame"]
                overlap_start = max(0, frame_offset)
                overlap_end = min(ref_frames, frame_offset + tgt_frames)
                if overlap_end > overlap_start:
                    overlap_frames = [overlap_start, overlap_end]

        target["frame_offset"] = frame_offset
        target["method"] = "manual"
        target["confidence"] = 1.0
        target["overlap_frames"] = overlap_frames

        # Atomic write
        tmp_path = sync_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(sync_data, indent=2))
        tmp_path.replace(sync_path)

        return target

    # ------------------------------------------------------------------
    # GET /api/video/{shot_id}
    # ------------------------------------------------------------------

    @app.get("/api/video/{shot_id}")
    def get_video(shot_id: str, request: Request):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", shot_id):
            raise HTTPException(status_code=400, detail="Invalid shot ID")
        # Look up the clip in shots/.
        candidate = (output_dir / "shots" / f"{shot_id}.mp4").resolve()
        if not candidate.is_relative_to((output_dir / "shots").resolve()):
            raise HTTPException(status_code=400, detail="Invalid shot ID")
        if candidate.exists():
            return _range_response(candidate, request)
        raise HTTPException(status_code=404, detail=f"Video not found: {shot_id}")

    # ------------------------------------------------------------------
    # GET /api/video/{shot_id}/frame?frame_idx=N  — extract a single JPEG frame
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # GET /viewer — standalone 3D viewer
    # ------------------------------------------------------------------

    @app.get("/viewer")
    def viewer_page():
        viewer_path = Path(__file__).parent / "static" / "viewer.html"
        if not viewer_path.exists():
            raise HTTPException(status_code=404, detail="viewer.html not found")
        return FileResponse(str(viewer_path))

    # ------------------------------------------------------------------
    # GET /api/export/scene.glb — serve the GLB file
    # ------------------------------------------------------------------

    @app.get("/api/export/scene.glb")
    def get_scene_glb():
        glb_path = output_dir / "export" / "gltf" / "scene.glb"
        if not glb_path.exists():
            raise HTTPException(status_code=404, detail="scene.glb not found")
        return FileResponse(str(glb_path), media_type="model/gltf-binary")

    # ------------------------------------------------------------------
    # GET /api/export/metadata — serve scene_metadata.json
    # ------------------------------------------------------------------

    @app.get("/api/export/metadata")
    def get_scene_metadata():
        meta_path = output_dir / "export" / "gltf" / "scene_metadata.json"
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail="scene_metadata.json not found")
        return json.loads(meta_path.read_text())

    return app

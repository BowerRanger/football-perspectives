"""Ball Studio HTTP router (``/api/ball-studio``).

Contract: ``docs/superpowers/specs/2026-10-04-ball-studio-api.md``. All math
lives in ``src/utils/ball_truth_solver.py``; this module only loads the
output directory (cameras, sync, poses, pipeline ball tracks), shapes the
JSON, and persists the operator's truth document.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Body, HTTPException, Response

from src.schemas import ball_truth as bt
from src.schemas.camera_track import CameraTrack
from src.utils import ball_truth_solver as solver
from src.utils import frame_cadence
from src.utils.ball_anchor_heights import BONE_TO_SMPL_INDEX
from src.utils.goal_geometry import GoalGeometry
from src.utils.smpl_skeleton import compute_all_joint_worlds_batch

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"[A-Za-z0-9_-]+")
SCENE_BONES = ("pelvis",) + tuple(BONE_TO_SMPL_INDEX)
# Triangulating views whose content instants differ by more than this many
# display frames (a repeated frame in one view) gets a warning.
VIEW_INSTANT_TOL_FRAMES = 0.3
_BONE_INDEX = {"pelvis": 0, **BONE_TO_SMPL_INDEX}


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShotCams:
    shot_id: str
    fps: float
    image_size: tuple[int, int]
    distortion: tuple[float, float]
    camera_centre: list[float] | None
    frames: np.ndarray            # (N,) shot-local frame indices
    K: np.ndarray                 # (N,3,3)
    R: np.ndarray                 # (N,3,3)
    t: np.ndarray                 # (N,3)
    confidence: np.ndarray
    index: dict[int, int] = field(default_factory=dict)

    def cam(self, shot_frame: int) -> solver.Cam | None:
        i = self.index.get(int(shot_frame))
        if i is None:
            return None
        return solver.Cam(self.K[i], self.R[i], self.t[i], self.distortion,
                          self.image_size)


@dataclass(frozen=True)
class GroupInfo:
    group_id: str
    reference_shot: str
    members: tuple[tuple[str, int, bool], ...]   # (shot_id, offset, excluded)


class _Players:
    """Refined-pose players with FK joint tracks on the reference timeline."""

    def __init__(self) -> None:
        self.by_id: dict[str, dict[str, Any]] = {}
        self.index: dict[str, dict[int, int]] = {}

    @classmethod
    def load(cls, output_dir: Path) -> "_Players":
        self = cls()
        rp = output_dir / "refined_poses"
        if not rp.exists():
            return self
        for path in sorted(rp.glob("*_refined.npz")):
            pid = path.name[: -len("_refined.npz")]
            try:
                z = np.load(path, allow_pickle=False)
                frames = z["frames"].astype(int)
                worlds = compute_all_joint_worlds_batch(
                    z["thetas"], z["root_R"], z["root_t"])
                self.by_id[pid] = {
                    "frames": frames,
                    "root": z["root_t"].astype(float),
                    "joints": {b: worlds[:, _BONE_INDEX[b]] for b in SCENE_BONES},
                    "confidence": z["confidence"].astype(float),
                }
                self.index[pid] = {int(f): i for i, f in enumerate(frames)}
            except Exception as exc:  # one bad track must not sink the scene
                logger.warning("ball-studio: skipping %s (%s)", path, exc)
        return self

    def joint(self, pid: str, bone: str, ref_frame: int) -> np.ndarray | None:
        p = self.by_id.get(pid)
        if p is None or bone not in p["joints"]:
            return None
        i = self.index[pid].get(int(ref_frame))
        return None if i is None else np.asarray(p["joints"][bone][i], float)


class StudioData:
    """Lazy, mtime-invalidated view over one output directory."""

    def __init__(self, output_dir: Path, config_path: Path | None) -> None:
        self.output_dir = output_dir
        self.config_path = config_path
        self._lock = threading.Lock()
        self._cams: dict[str, tuple[float, ShotCams | None]] = {}
        self._players: tuple[float, _Players] | None = None
        self._cadence: dict[str, tuple[float, frame_cadence.Cadence | None]] = {}

    # -- cameras ----------------------------------------------------------
    def shot_cams(self, shot_id: str) -> ShotCams | None:
        path = self.output_dir / "camera" / f"{shot_id}_camera_track.json"
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        with self._lock:
            hit = self._cams.get(shot_id)
            if hit and hit[0] == mtime:
                return hit[1]
        try:
            tr = CameraTrack.load(path)
        except Exception as exc:
            logger.warning("ball-studio: bad camera track %s (%s)", path, exc)
            return None
        frames = np.array([f.frame for f in tr.frames], dtype=int)
        K = np.array([f.K for f in tr.frames], dtype=float)
        R = np.array([f.R for f in tr.frames], dtype=float)
        t = np.array(
            [f.t if f.t is not None else tr.t_world for f in tr.frames], dtype=float)
        sc = ShotCams(
            shot_id=shot_id, fps=float(tr.fps),
            image_size=(int(tr.image_size[0]), int(tr.image_size[1])),
            distortion=(float(tr.distortion[0]), float(tr.distortion[1])),
            camera_centre=list(tr.camera_centre) if tr.camera_centre else None,
            frames=frames, K=K, R=R, t=t,
            confidence=np.array([f.confidence for f in tr.frames], dtype=float),
            index={int(f): i for i, f in enumerate(frames)},
        )
        with self._lock:
            self._cams[shot_id] = (mtime, sc)
        return sc

    # -- frame cadence ----------------------------------------------------
    def cadence(self, shot_id: str) -> frame_cadence.Cadence | None:
        """Repeated frames of the shot video (25->30 pulldown); None when
        the video is missing or unreadable (= uniform time)."""
        video = self.output_dir / "shots" / f"{shot_id}.mp4"
        if not video.exists():
            return None
        mtime = video.stat().st_mtime
        with self._lock:
            hit = self._cadence.get(shot_id)
            if hit and hit[0] == mtime:
                return hit[1]
        try:
            cad = frame_cadence.load_or_detect(
                video, self.output_dir / "ball_truth" / ".cache" / "cadence")
        except Exception as exc:  # best effort: uniform time is the fallback
            logger.warning("ball-studio: cadence scan failed for %s (%s)", video, exc)
            cad = None
        with self._lock:
            self._cadence[shot_id] = (mtime, cad)
        return cad

    def time_shift(self, shot_id: str) -> np.ndarray | None:
        cad = self.cadence(shot_id)
        sc = self.shot_cams(shot_id)
        if cad is None or not cad.repeats or sc is None:
            return None
        return frame_cadence.content_time_shift(cad.n_frames, cad.repeats, sc.fps)

    # -- players ----------------------------------------------------------
    def players(self) -> _Players:
        rp = self.output_dir / "refined_poses"
        mtime = max((p.stat().st_mtime for p in rp.glob("*_refined.npz")),
                    default=0.0) if rp.exists() else 0.0
        with self._lock:
            if self._players and self._players[0] == mtime:
                return self._players[1]
        pl = _Players.load(self.output_dir)
        with self._lock:
            self._players = (mtime, pl)
        return pl

    # -- groups -----------------------------------------------------------
    def groups(self) -> dict[str, GroupInfo]:
        from src.schemas.shots import ShotsManifest
        from src.schemas.sync_map import SyncMap

        manifest_p = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_p.exists():
            return {}
        try:
            manifest = ShotsManifest.load(manifest_p)
        except Exception as exc:
            logger.warning("ball-studio: bad manifest (%s)", exc)
            return {}
        excluded = {s.id: bool(getattr(s, "excluded", False)) for s in manifest.shots}
        sync = SyncMap()
        sp = self.output_dir / "shots" / "sync_map.json"
        if sp.exists():
            try:
                sync = SyncMap.load(sp)
            except Exception:
                sync = SyncMap()
        offsets: dict[str, int] = {}
        sync_ref: dict[str, str] = {}
        sync_members: dict[str, list[str]] = {}
        shot_group: dict[str, str] = {}
        for g in sync.groups:
            sync_members[g.group_id] = [a.shot_id for a in g.alignments]
            sync_ref[g.group_id] = g.reference_shot
            for a in g.alignments:
                offsets[a.shot_id] = a.frame_offset
                shot_group.setdefault(a.shot_id, g.group_id)
        # shots the sync map doesn't know: manifest group, else their own
        buckets: dict[str, list[str]] = {}
        for s in manifest.shots:
            gid = shot_group.get(s.id)
            if gid is None:
                gid = getattr(s, "group_id", None) or ""
                if not gid:
                    gid = f"__solo__{s.id}"
            buckets.setdefault(gid, []).append(s.id)
        out: dict[str, GroupInfo] = {}
        for gid, ids in buckets.items():
            usable = [i for i in ids
                      if self.shot_cams(i) is not None]
            if not usable:
                continue
            ref = sync_ref.get(gid)
            if ref not in usable:
                ref = usable[0]
            if gid == "" or gid.startswith("__solo__"):
                public = ref
            else:
                public = gid
            if public in out:  # id collision: disambiguate
                public = f"{public}-{ref}"
            ref_off = offsets.get(ref, 0)
            members = tuple(
                (i, int(offsets.get(i, 0)) - int(ref_off), excluded.get(i, False))
                for i in sorted(usable, key=lambda x: (x != ref, x))
            )
            out[public] = GroupInfo(public, ref, members)
        return out

    def group(self, gid: str) -> GroupInfo:
        if not _ID_RE.fullmatch(gid):
            raise HTTPException(status_code=400, detail="Invalid group id")
        g = self.groups().get(gid)
        if g is None:
            raise HTTPException(status_code=404, detail=f"unknown group {gid!r}")
        return g

    def context(self, g: GroupInfo, overrides: dict[str, int] | None = None
                ) -> solver.SolveContext:
        offs = {sid: off for sid, off, _ in g.members}
        if overrides:
            offs.update({k: int(v) for k, v in overrides.items() if k in offs})
        cams = {sid: self.shot_cams(sid) for sid in offs}
        fps = cams[g.reference_shot].fps if cams.get(g.reference_shot) else 30.0
        pl = self.players()

        def camera(shot_id: str, shot_frame: int) -> solver.Cam | None:
            sc = cams.get(shot_id)
            return sc.cam(shot_frame) if sc else None

        shifts = {sid: s for sid in offs if (s := self.time_shift(sid)) is not None}

        def time_shift(shot_id: str, shot_frame: int) -> float:
            s = shifts.get(shot_id)
            return float(s[shot_frame]) if s is not None and 0 <= shot_frame < len(s) else 0.0

        return solver.SolveContext(fps=fps, offsets=offs, camera=camera,
                                   joint=pl.joint, reference_shot=g.reference_shot,
                                   time_shift=time_shift if shifts else None)

    def goals(self) -> dict[str, Any]:
        from src.pipeline.config import load_config

        try:
            pitch = dict(load_config(self.config_path).get("pitch", {}))
        except Exception:
            pitch = {}
        geo = GoalGeometry.from_pitch_config(pitch)
        length = float(pitch.get("length_m", 105.0))
        width = float(pitch.get("width_m", 68.0))
        return {
            "goal_line_x_near": geo.goal_line_x_near,
            "goal_line_x_far": geo.goal_line_x_far,
            "post_y_left": round(geo.post_y_left, 4),
            "post_y_right": round(geo.post_y_right, 4),
            "crossbar_z": geo.crossbar_z,
            "net_depth": geo.net_depth,
            "pitch": {"length_m": length, "width_m": width},
            "goal_planes": [
                {"id": "goal_line_near", "axis": "x", "value": geo.goal_line_x_near,
                 "mouth": {"y_range": [round(geo.post_y_left, 4), round(geo.post_y_right, 4)],
                           "z_range": [0.0, geo.crossbar_z]}},
                {"id": "goal_line_far", "axis": "x", "value": geo.goal_line_x_far,
                 "mouth": {"y_range": [round(geo.post_y_left, 4), round(geo.post_y_right, 4)],
                           "z_range": [0.0, geo.crossbar_z]}},
            ],
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _round(a: Any, nd: int = 4) -> Any:
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def _ref_range(g: GroupInfo, data: StudioData) -> list[int] | None:
    lo, hi = None, None
    for sid, off, _ in g.members:
        sc = data.shot_cams(sid)
        if sc is None or not len(sc.frames):
            continue
        a, b = int(sc.frames.min()) - off, int(sc.frames.max()) - off
        lo = a if lo is None else min(lo, a)
        hi = b if hi is None else max(hi, b)
    return None if lo is None else [lo, hi]


def _shot_meta(sid: str, off: int, excluded: bool, sc: ShotCams) -> dict[str, Any]:
    return {
        "shot_id": sid, "frame_offset": off, "n_frames": int(len(sc.frames)),
        "fps": sc.fps, "image_size": list(sc.image_size),
        "width": sc.image_size[0], "height": sc.image_size[1],
        "frame_range": [int(sc.frames.min()), int(sc.frames.max())],
        "excluded": excluded,
        "video_url": f"/api/video/{sid}", "frame_url": f"/api/video/{sid}/frame",
    }


def _doc_for_solve(body: Any, g: GroupInfo) -> dict[str, Any]:
    norm, errors = bt.validate_truth(
        body, group_id=g.group_id, member_shots=[m[0] for m in g.members])
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})
    return norm


def _pipeline_tracks(g: GroupInfo, out: Path) -> list[dict[str, Any]]:
    from src.schemas.ball_track import BallTrack

    rows = []
    for sid, off, _ in g.members:
        p = out / "ball" / f"{sid}_ball_track.json"
        if not p.exists():
            continue
        try:
            tr = BallTrack.load(p)
        except Exception:
            continue
        rows.append({
            "shot_id": sid,
            "frames": [f.frame - off for f in tr.frames],
            "shot_frames": [f.frame for f in tr.frames],
            "xyz": [None if f.world_xyz is None else _round(f.world_xyz) for f in tr.frames],
            "state": [f.state for f in tr.frames],
            "confidence": [round(float(f.confidence), 3) for f in tr.frames],
        })
    return rows


def _pipeline_anchors(g: GroupInfo, out: Path) -> list[dict[str, Any]]:
    rows = []
    for sid, off, _ in g.members:
        for suffix, source in (("_ball_anchors.json", "manual"),
                               ("_ball_anchors_auto.json", "auto")):
            p = out / "ball" / f"{sid}{suffix}"
            if not p.exists():
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            for a in d.get("anchors", []):
                if a.get("image_xy") is None:
                    continue
                rows.append({
                    "shot_id": sid, "ref_frame": int(a["frame"]) - off,
                    "shot_frame": int(a["frame"]), "kind": a.get("state"),
                    "uv": [round(float(a["image_xy"][0]), 2),
                           round(float(a["image_xy"][1]), 2)],
                    "source": source,
                })
    return rows


# ---------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------

def build_router(output_dir: Path, config_path: Path | None = None) -> APIRouter:
    output_dir = Path(output_dir)
    data = StudioData(output_dir, config_path)
    router = APIRouter(prefix="/api/ball-studio", tags=["ball-studio"])
    write_lock = threading.Lock()

    @router.get("/groups")
    def list_groups() -> dict[str, Any]:
        rows = []
        for gid, g in sorted(data.groups().items()):
            shots = []
            fps = 30.0
            for sid, off, exc in g.members:
                sc = data.shot_cams(sid)
                if sc is None:
                    continue
                if sid == g.reference_shot:
                    fps = sc.fps
                shots.append(_shot_meta(sid, off, exc, sc))
            truth = bt.load_truth(output_dir, gid)
            rows.append({
                "group_id": gid,
                "label": " + ".join(m[0] for m in g.members),
                "reference_shot": g.reference_shot,
                "fps": fps,
                "ref_frame_range": _ref_range(g, data),
                "shots": shots,
                "has_truth": truth is not None,
                "truth_updated_at": (truth or {}).get("meta", {}).get("updated_at"),
                "n_keys": len((truth or {}).get("keys", [])),
                "outcome": (truth or {}).get("outcome", "unknown"),
                "status": (truth or {}).get("meta", {}).get("status", "draft"),
            })
        return {"groups": rows}

    @router.get("/groups/{group_id}/scene")
    def get_scene(group_id: str, response: Response) -> dict[str, Any]:
        response.headers["Cache-Control"] = "no-store"
        g = data.group(group_id)
        shots = []
        fps = 30.0
        for sid, off, exc in g.members:
            sc = data.shot_cams(sid)
            if sc is None:
                continue
            if sid == g.reference_shot:
                fps = sc.fps
            meta = _shot_meta(sid, off, exc, sc)
            meta.update({
                "distortion": list(sc.distortion),
                "camera_centre": sc.camera_centre,
                "frames": sc.frames.tolist(),
                "K": _round(np.stack([sc.K[:, 0, 0], sc.K[:, 1, 1],
                                      sc.K[:, 0, 2], sc.K[:, 1, 2]], axis=1), 3),
                "R": _round(sc.R.reshape(-1, 9), 6),
                "t": _round(sc.t, 4),
                "confidence": _round(sc.confidence, 3),
            })
            cad = data.cadence(sid)
            # shot frames whose image repeats the previous one (pulldown)
            meta["repeat_frames"] = list(cad.repeats) if cad else []
            shots.append(meta)
        pl = data.players()
        players = []
        for pid, p in sorted(pl.by_id.items()):
            players.append({
                "player_id": pid,
                "frames": p["frames"].tolist(),
                "root": _round(p["root"], 3),
                "joints": {b: _round(p["joints"][b], 3) for b in SCENE_BONES},
                "confidence": _round(p["confidence"], 3),
            })
        return {
            "group_id": g.group_id, "reference_shot": g.reference_shot, "fps": fps,
            "ref_frame_range": _ref_range(g, data),
            "shots": shots,
            "goals": data.goals(),
            "bones": list(SCENE_BONES),
            "players": players,
            "pipeline_tracks": _pipeline_tracks(g, output_dir),
            "pipeline_anchors": _pipeline_anchors(g, output_dir),
        }

    @router.get("/groups/{group_id}/truth")
    def get_truth(group_id: str) -> dict[str, Any]:
        g = data.group(group_id)
        doc = bt.load_truth(output_dir, group_id)
        if doc is None:
            sc = data.shot_cams(g.reference_shot)
            skeleton = bt.empty_truth(
                group_id, g.reference_shot, sc.fps if sc else 30.0,
                [(sid, off) for sid, off, _ in g.members])
            return {"exists": False, "truth": skeleton, "dense": None}
        return {"exists": True, "truth": doc, "dense": bt.load_dense(output_dir, group_id)}

    @router.put("/groups/{group_id}/truth")
    def put_truth(group_id: str, body: dict = Body(...)) -> dict[str, Any]:
        g = data.group(group_id)
        if "truth" not in body:
            raise HTTPException(
                status_code=422,
                detail={"errors": [{"path": "truth", "message": "body must be {truth, expected_updated_at?}"}]})
        extra = set(body) - {"truth", "expected_updated_at"}
        if extra:
            raise HTTPException(
                status_code=422,
                detail={"errors": [{"path": k, "message": "unknown field"} for k in sorted(extra)]})
        doc = _doc_for_solve(body["truth"], g)
        with write_lock:
            if "expected_updated_at" in body:
                current = bt.load_truth(output_dir, group_id)
                cur_ts = (current or {}).get("meta", {}).get("updated_at") if current else None
                if body["expected_updated_at"] != cur_ts:
                    raise HTTPException(status_code=409, detail={
                        "message": "truth changed since it was loaded",
                        "current_updated_at": cur_ts,
                        "expected_updated_at": body["expected_updated_at"],
                    })
            doc["meta"]["updated_at"] = bt.utc_now_iso()
            solve_ok, n_flags, dense = True, 0, None
            try:
                result = solver.solve_truth(doc, data.context(g))
                solve_ok, n_flags = bool(result["ok"]), len(result["flags"])
                dense = {
                    "group_id": group_id, "updated_at": doc["meta"]["updated_at"],
                    "fps": data.context(g).fps,
                    "dense": result["dense"], "projections": result["projections"],
                    "keys": result["keys"], "segments": result["segments"],
                    "events": doc["events"], "outcome": doc["outcome"],
                    "flags": result["flags"],
                }
            except Exception as exc:
                logger.exception("ball-studio: solve failed while saving %s", group_id)
                solve_ok = False
            history = bt.save_truth(output_dir, group_id, doc)
            if dense is not None:
                bt.save_dense(output_dir, group_id, dense)
        return {"ok": True, "updated_at": doc["meta"]["updated_at"],
                "history_file": history, "solve_ok": solve_ok, "n_flags": n_flags}

    @router.post("/groups/{group_id}/solve")
    def post_solve(group_id: str, body: Any = Body(...)) -> dict[str, Any]:
        g = data.group(group_id)
        doc = _doc_for_solve(body, g)
        return solver.solve_truth(doc, data.context(g))

    @router.post("/groups/{group_id}/triangulate")
    def post_triangulate(group_id: str, body: dict = Body(...)) -> dict[str, Any]:
        g = data.group(group_id)
        overrides = body.get("offsets") or None
        if overrides is not None and not (
            isinstance(overrides, dict)
            and all(isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool)
                    for k, v in overrides.items())
        ):
            raise HTTPException(status_code=422, detail="offsets must map shot id to integer")
        ctx = data.context(g, overrides)
        base = data.context(g)
        frame = body.get("frame")
        raw_obs = body.get("observations") or []
        if not isinstance(frame, int) or isinstance(frame, bool) or not raw_obs:
            raise HTTPException(status_code=422, detail="frame (int) and observations required")
        flags: list[dict] = []
        views, used = [], []
        for i, ob in enumerate(raw_obs):
            try:
                sid, sf, uv = ob["shot_id"], int(ob["shot_frame"]), [float(ob["uv"][0]), float(ob["uv"][1])]
            except Exception:
                raise HTTPException(status_code=422, detail=f"observations[{i}] malformed")
            if sid not in ctx.offsets:
                raise HTTPException(status_code=422, detail=f"observations[{i}]: {sid!r} not in group")
            # An override re-derives which camera frame the (held) pixel
            # belongs to: shot_frame' = frame + override_offset.
            if overrides and sid in overrides:
                sf = frame + int(overrides[sid])
            cam = ctx.camera(sid, sf)
            if cam is None:
                return {"ok": False, "reason": "no_camera_frame", "xyz": None,
                        "residual_px": {}, "flags": [{"level": "error", "code": "no_camera_frame",
                                                        "message": f"no camera for {sid} frame {sf}"}]}
            views.append((cam, uv))
            cad = data.cadence(sid)
            used.append({"shot_id": sid, "shot_frame": sf, "uv": uv,
                         "repeat": bool(cad and sf in cad.repeats)})
        if len(used) >= 2 and ctx.time_shift is not None:
            times = [ctx.obs_time(u["shot_id"], u["shot_frame"]) for u in used]
            gap_ms = 1000.0 * (max(times) - min(times))
            if gap_ms > 1000.0 * VIEW_INSTANT_TOL_FRAMES / ctx.fps:
                flags.append({
                    "level": "warn", "code": "views_not_simultaneous",
                    "message": f"the views show instants {gap_ms:.0f} ms apart (repeated "
                               "frame) - pick a frame that is fresh in every view"})
        offsets_used = {sid: off for sid, off in ctx.offsets.items()}
        constraint = body.get("constraint")
        resp: dict[str, Any] = {"offsets_used": offsets_used, "observations_used": used}
        if len(views) >= 2:
            res = solver.triangulate(views)
            resid = {u["shot_id"]: round(r, 3) for u, r in zip(used, res.residual_px)}
            mx = max(res.residual_px) if res.residual_px else None
            reason = res.reason
            ok = res.ok
            if ok and mx is not None and mx > solver.MAX_KEY_RESIDUAL_PX:
                ok, reason = False, "residual_exceeds_limit"
            if ok and res.ray_angle_deg < solver.WEAK_BASELINE_DEG:
                flags.append({"level": "warn", "code": "weak_baseline",
                              "message": f"rays differ by {res.ray_angle_deg:.1f} deg"})
            resp.update({
                "ok": ok, "reason": reason, "source": "triangulated",
                "xyz": None if res.xyz is None else _round(res.xyz),
                "residual_px": resid, "max_residual_px": None if mx is None else round(mx, 3),
                "reprojected_uv": {u["shot_id"]: r for u, r in zip(used, res.reproj_uv)},
                "ray_angle_deg": round(res.ray_angle_deg, 2),
                "skew_gap_cm": round(res.skew_gap_cm, 2),
                "rays": [{"shot_id": u["shot_id"], "origin": _round(o), "direction": _round(d, 6)}
                         for u, (o, d) in zip(used, res.rays)],
                "flags": flags,
            })
            return resp
        cam, uv = views[0]
        sid = used[0]["shot_id"]
        origin, direction = cam.ray(uv)
        resp["rays"] = [{"shot_id": sid, "origin": _round(origin),
                         "direction": _round(direction, 6)}]
        # epipolar lines in the other views at the same reference instant
        epi = []
        for osid, ooff in base.offsets.items():
            if osid == sid:
                continue
            ocam = ctx.camera(osid, frame + ctx.offsets[osid])
            if ocam is None:
                continue
            poly = solver.epipolar_polyline(origin, direction, ocam)
            epi.append({"shot_id": osid, "shot_frame": frame + ctx.offsets[osid],
                        "polyline_uv": poly,
                        "segment_uv": None if not poly else [poly[0], poly[-1]]})
        resp["epipolar"] = epi
        mode = (constraint or {}).get("mode")
        if not constraint or not mode:
            resp.update({"ok": True, "source": "ray", "xyz": None,
                         "residual_px": {}, "flags": flags})
            return resp
        xyz, err = solver.constrain_ray(cam, uv, constraint, mode, ref_frame=frame,
                                        joint=ctx.joint)
        if xyz is None:
            resp.update({"ok": False, "reason": err, "xyz": None, "residual_px": {},
                         "flags": flags})
            return resp
        proj, depth = cam.project(xyz)
        if depth[0] <= 0.1:
            resp.update({"ok": False, "reason": "behind_camera", "xyz": _round(xyz),
                         "residual_px": {}, "flags": flags})
            return resp
        src = {"ground": "ray_ground", "height": "ray_height", "plane": "ray_plane",
               "depth": "ray_depth", "player": "player"}.get(mode, "manual")
        resp.update({
            "ok": True, "source": src, "xyz": _round(xyz),
            "residual_px": {sid: round(float(np.hypot(*(proj[0] - np.asarray(uv)))), 3)},
            "reprojected_uv": {sid: [round(float(proj[0, 0]), 2), round(float(proj[0, 1]), 2)]},
            "flags": flags,
        })
        return resp

    return router

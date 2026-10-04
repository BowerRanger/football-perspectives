"""Replay-speed HTTP router: report, operator moments, retime / restore.

Contract: docs/superpowers/specs/2026-10-04-replay-speed.md ("HTTP"). Math
lives in ``src/utils/replay_speed.py``; clip/sidecar resampling in
``src/utils/replay_retime.py``. Writes share the dashboard's manifest/sync
lock so they serialise with every other sync-map / manifest editor.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.schemas.shots import ShotsManifest
from src.schemas.sync_map import Alignment, GroupSync, SyncMap
from src.utils.replay_retime import restore_native, retime_shot
from src.utils.replay_speed import rate_from_moments

_DEFAULT_TOLERANCE = 0.08


class Moment(BaseModel):
    reference_frame: float = Field(ge=0, le=1e6, allow_inf_nan=False)
    shot_frame: float = Field(ge=0, le=1e6, allow_inf_nan=False)


class MomentsPayload(BaseModel):
    shot_id: str = Field(min_length=1, max_length=64)
    moments: list[Moment] = Field(min_length=2, max_length=50)
    retime: bool = False


class RetimePayload(BaseModel):
    # Playback rate relative to the NATIVE clip (live frames per native frame).
    rate: float = Field(gt=0.02, le=1.0, allow_inf_nan=False)


def build_router(output_dir: Path, lock: threading.Lock,
                 config_path: Path | None = None) -> APIRouter:
    output_dir = Path(output_dir)
    router = APIRouter(tags=["replay-sync"])
    manifest_path = output_dir / "shots" / "shots_manifest.json"
    sync_path = output_dir / "shots" / "sync_map.json"

    def tolerance() -> float:
        try:
            from src.pipeline.config import load_config

            return float((load_config(config_path).get("replay_sync") or {})
                         .get("retime_tolerance", _DEFAULT_TOLERANCE))
        except Exception:
            return _DEFAULT_TOLERANCE

    def load_manifest() -> ShotsManifest:
        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail="no shots manifest")
        return ShotsManifest.load(manifest_path)

    def find_shot(manifest: ShotsManifest, shot_id: str):
        shot = next((s for s in manifest.shots if s.id == shot_id), None)
        if shot is None:
            raise HTTPException(status_code=404, detail=f"unknown shot {shot_id!r}")
        return shot

    def load_sync() -> SyncMap:
        return SyncMap.load(sync_path) if sync_path.exists() else SyncMap()

    def group_of(sm: SyncMap, manifest: ShotsManifest, shot_id: str) -> GroupSync | None:
        gid = find_shot(manifest, shot_id).group_id
        return sm.group(gid)

    def set_rate(sm: SyncMap, group: GroupSync | None, shot_id: str, rate: float) -> tuple[SyncMap, dict | None]:
        """Re-base an existing alignment's playback_rate (offset unchanged)."""
        if group is None:
            return sm, None
        prior = next((a for a in group.alignments if a.shot_id == shot_id), None)
        if prior is None:
            return sm, None
        new = Alignment(prior.shot_id, prior.frame_offset, prior.method,
                        prior.confidence, float(rate))
        return sm.with_group(group.with_alignment(new)), asdict(new)

    @router.get("/api/replay-sync")
    def get_replay_sync() -> dict[str, Any]:
        path = output_dir / "shots" / "replay_sync.json"
        if not path.exists():
            return {"version": 1, "groups": []}
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"bad replay_sync.json: {exc}")

    @router.post("/api/sync/groups/{group_id}/moments")
    def post_moments(group_id: str, payload: MomentsPayload) -> dict[str, Any]:
        with lock:
            manifest = load_manifest()
            members = [s.id for s in manifest.shots if s.group_id == group_id]
            if not members:
                raise HTTPException(status_code=404, detail=f"unknown group {group_id!r}")
            if payload.shot_id not in members:
                raise HTTPException(status_code=422, detail=(
                    f"shot {payload.shot_id!r} is not in group {group_id!r}"))
            sm = load_sync()
            saved = sm.group(group_id)
            reference = saved.reference_shot if saved and saved.reference_shot in members else members[0]
            if payload.shot_id == reference:
                raise HTTPException(status_code=422, detail="the reference shot is the timeline; mark a replay")
            try:
                fit = rate_from_moments([(m.reference_frame, m.shot_frame) for m in payload.moments])
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            if not (math.isfinite(fit.rate) and fit.rate > 0):
                raise HTTPException(status_code=422, detail="moments give a non-positive rate")

            shot = find_shot(manifest, payload.shot_id)
            retime_result = None
            retime_note = ""
            rate_out = fit.rate
            if payload.retime:
                if fit.ramp:
                    retime_note = "speed ramp detected: not retimed"
                elif not fit.rate < 1.0 - tolerance():
                    retime_note = "rate within tolerance of real time: not retimed"
                else:
                    # fit.rate is relative to the clip as it is now; the
                    # retime resamples the native clip.
                    native_rate = fit.rate / shot.speed_factor if shot.retimed else fit.rate
                    retime_result = retime_shot(output_dir, payload.shot_id, native_rate)
                    rate_out = 1.0
            alignment = Alignment(payload.shot_id, int(round(-fit.offset)), "manual", 1.0, float(rate_out))
            if saved is None or reference != saved.reference_shot:
                base = GroupSync(group_id, reference, [Alignment(reference, 0)]
                                 if saved is None else saved.alignments)
            else:
                base = saved
            if not any(a.shot_id == reference for a in base.alignments):
                base = base.with_alignment(Alignment(reference, 0))
            sm.with_group(GroupSync(group_id, reference, base.with_alignment(alignment).alignments)).save(sync_path)
        return {
            "rate": fit.rate, "offset": fit.offset,
            "residual_frames": fit.residual_frames,
            "interval_rates": fit.interval_rates,
            "ramp": fit.ramp, "n_moments": fit.n_moments,
            "alignment": asdict(alignment),
            "retimed": retime_result is not None,
            "retime": _result_dict(retime_result),
            "note": retime_note,
        }

    @router.post("/api/shots/{shot_id}/retime")
    def post_retime(shot_id: str, payload: RetimePayload) -> dict[str, Any]:
        with lock:
            manifest = load_manifest()
            shot = find_shot(manifest, shot_id)
            sm = load_sync()
            group = group_of(sm, manifest, shot_id)
            prior = next((a for a in (group.alignments if group else []) if a.shot_id == shot_id), None)
            try:
                result = retime_shot(output_dir, shot_id, payload.rate)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=f"clip missing: {exc}")
            alignment = None
            # The alignment described the clip at ``rate``: now it is real time.
            if prior is not None and math.isclose(prior.playback_rate, payload.rate, rel_tol=0.02):
                sm, alignment = set_rate(sm, group, shot_id, 1.0)
                sm.save(sync_path)
        return {"retime": _result_dict(result), "alignment": alignment,
                "speed_factor": 1.0 / payload.rate}

    @router.post("/api/shots/{shot_id}/restore-native")
    def post_restore(shot_id: str) -> dict[str, Any]:
        with lock:
            manifest = load_manifest()
            shot = find_shot(manifest, shot_id)
            if not shot.retimed:
                raise HTTPException(status_code=409, detail="shot is not retimed")
            sm = load_sync()
            group = group_of(sm, manifest, shot_id)
            prior = next((a for a in (group.alignments if group else []) if a.shot_id == shot_id), None)
            restored = restore_native(output_dir, shot_id)
            alignment = None
            # Back on the native timeline: ref = offset + (1/speed_factor) * frame.
            if restored and prior is not None and math.isclose(prior.playback_rate, 1.0, abs_tol=1e-6):
                sm, alignment = set_rate(sm, group, shot_id, 1.0 / shot.speed_factor)
                sm.save(sync_path)
        return {"restored": restored, "alignment": alignment}

    return router


def _result_dict(result) -> dict | None:
    if result is None:
        return None
    d = asdict(result)
    d["frame_map"] = f"{len(result.frame_map)} frames"  # the map itself is large
    d["frames_in"], d["frames_out"] = result.n_native, result.n_new
    return d

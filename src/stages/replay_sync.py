"""replay_sync stage - playback speed + offset of replays from player positions.

Runs after ``camera`` (it needs tracks and a solved camera on both shots of a
pair). Per highlight group it estimates every non-reference member against the
reference (``src/utils/replay_speed.py``), writes ``player_formation``
alignments with ``playback_rate`` into ``shots/sync_map.json`` (``manual``
alignments are never touched), optionally retimes a confidently slow replay to
real time (``src/utils/replay_retime.py``) and writes ``shots/replay_sync.json``.
Decision logic lives in ``src/utils/replay_sync_group.py``.
See docs/superpowers/specs/2026-10-04-replay-speed.md.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.camera_track import CameraTrack
from src.schemas.shots import ShotsManifest
from src.schemas.sync_map import Alignment, GroupSync, SyncMap
from src.utils.replay_retime import retime_shot
from src.utils.replay_speed import feet_on_pitch
from src.utils.replay_sync_group import DEFAULTS, GroupResult, MemberResult, solve_group

logger = logging.getLogger(__name__)

REPORT_FILE = "replay_sync.json"
_ESTIMATE_KEYS = ("rate", "offset", "confidence", "cost_m", "coverage", "ramp",
                  "rate_first", "rate_second", "live_window_frames", "rate_uncertainty")


def _camera_fn(track: CameraTrack):
    by_frame = {f.frame: f for f in track.frames}

    def fn(frame: int):
        f = by_frame.get(frame)
        if f is None:
            return None
        t = f.t if f.t is not None else track.t_world
        return (np.asarray(f.K, float), np.asarray(f.R, float),
                np.asarray(t, float), tuple(track.distortion))

    return fn


def load_feet(output_dir: Path, shot_id: str) -> "dict[int, np.ndarray] | str":
    """Feet-on-pitch table for a shot, or ``"no_tracks"`` / ``"no_camera"``."""
    tracks_path = output_dir / "tracks" / f"{shot_id}_tracks.json"
    if not tracks_path.exists():
        return "no_tracks"
    cam_path = output_dir / "camera" / f"{shot_id}_camera_track.json"
    if not cam_path.exists():
        return "no_camera"
    try:
        return feet_on_pitch(json.loads(tracks_path.read_text()),
                             _camera_fn(CameraTrack.load(cam_path)))
    except Exception as exc:  # noqa: BLE001 - unreadable sidecar: skip the shot
        logger.warning("[replay_sync] %s: cannot read tracks/camera (%s)", shot_id, exc)
        return "no_tracks"


def _estimate_dict(m: MemberResult) -> dict | None:
    if m.estimate is None:
        return None
    d = asdict(m.estimate)
    return {k: (bool(d[k]) if k == "ramp" else float(d[k])) for k in _ESTIMATE_KEYS}


def _merged(cfg: dict) -> dict:
    return {**DEFAULTS, **(cfg or {})}


class ReplaySyncStage(BaseStage):
    name = "replay_sync"

    def __init__(self, config: dict, output_dir: Path, **kwargs) -> None:
        super().__init__(config, output_dir, **kwargs)

    def _report_path(self) -> Path:
        return self.output_dir / "shots" / REPORT_FILE

    def is_complete(self) -> bool:
        return self._report_path().exists()

    def _groups(self, manifest: ShotsManifest, sm: SyncMap) -> list[tuple[str, str, list[str]]]:
        active = {s.id for s in manifest.active_shots()}
        out = []
        for g in manifest.groups:
            ids = [s for s in g.shot_ids if s in active]
            if len(ids) < 2:
                continue
            saved = sm.group(g.id)
            ref = saved.reference_shot if saved and saved.reference_shot in ids else ids[0]
            out.append((g.id, ref, ids))
        return out

    def _apply(self, sm: SyncMap, res: GroupResult) -> SyncMap:
        group = sm.group(res.group_id) or GroupSync(res.group_id, res.reference_shot, [])
        if not any(a.shot_id == res.reference_shot for a in group.alignments):
            group = group.with_alignment(Alignment(res.reference_shot, 0))
        for m in res.members:
            if m.decision not in ("applied", "applied_retimed") or m.estimate is None:
                continue
            est = m.estimate
            if m.decision == "applied_retimed":
                retime_shot(self.output_dir, m.shot_id, est.rate)
                rate = 1.0
            else:
                rate = est.rate
            group = group.with_alignment(Alignment(
                m.shot_id, int(round(-est.offset)), "player_formation",
                float(est.confidence), float(rate)))
            logger.info("[replay_sync] %s/%s: %s rate %.3f offset %.1f conf %.2f (vs %s)",
                        res.group_id, m.shot_id, m.decision, est.rate, est.offset,
                        est.confidence, m.against)
        return sm.with_group(GroupSync(group.group_id, res.reference_shot, group.alignments))

    def run(self) -> None:
        cfg = _merged(self.config.get("replay_sync") or {})
        if not cfg["enabled"]:
            logger.info("[replay_sync] disabled")
            return
        manifest = ShotsManifest.load(self.output_dir / "shots" / "shots_manifest.json")
        sync_path = self.output_dir / "shots" / "sync_map.json"
        sm = SyncMap.load(sync_path) if sync_path.exists() else SyncMap()

        report_groups = []
        for gid, ref, ids in self._groups(manifest, sm):
            saved = sm.group(gid)
            prior_manual = {
                a.shot_id: (a.playback_rate, a.frame_offset)
                for a in (saved.alignments if saved else [])
                if a.method == "manual" and a.shot_id != ref and a.shot_id in ids
            }
            feet = {sid: load_feet(self.output_dir, sid) for sid in ids}
            res = solve_group(gid, ref, feet, prior_manual, cfg, only=self.shot_filter)
            sm = self._apply(sm, res)
            for m in res.members:
                logger.info("[replay_sync] %s/%s: %s%s", gid, m.shot_id, m.decision,
                            f" ({m.reason})" if m.reason else "")
            report_groups.append({
                "group_id": gid, "reference_shot": ref,
                "members": [{"shot_id": m.shot_id, "against": m.against,
                             "estimate": _estimate_dict(m), "decision": m.decision,
                             "approximate": m.approximate,
                             "reason": m.reason} for m in res.members],
            })
        sm.save(sync_path)
        report = {"version": 1, "groups": report_groups}
        tmp = self._report_path().with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(self._report_path())

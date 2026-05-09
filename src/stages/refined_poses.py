"""Stage 7-of-7 between hmr_world/ball and export.

Fuses each player's per-shot SMPL reconstructions into a single track
on the shared reference timeline. v1 contract:

  - identity is the ``player_id`` annotation set during tracking;
    no automatic re-id;
  - sync is taken from ``output/shots/sync_map.json`` (authoritative);
  - players appearing in only one shot pass through unchanged.

This module hosts the stage class. Pure math lives in
``src.utils.pose_fusion``; smoothing helpers come from
``src.utils.temporal_smoothing``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.refined_pose import (
    FrameDiagnostic,
    RefinedPose,
    RefinedPoseDiagnostics,
)
from src.schemas.smpl_world import SmplWorldTrack
from src.schemas.sync_map import SyncMap

logger = logging.getLogger(__name__)


class RefinedPosesStage(BaseStage):
    name = "refined_poses"

    # ------------------------------------------------------------------

    def is_complete(self) -> bool:
        hmr_dir = self.output_dir / "hmr_world"
        if not hmr_dir.exists():
            return True
        out_dir = self.output_dir / "refined_poses"
        player_ids = self._discover_player_ids(hmr_dir)
        if not player_ids:
            return True
        return all((out_dir / f"{pid}_refined.npz").exists() for pid in player_ids)

    # ------------------------------------------------------------------

    def run(self) -> None:
        cfg = (self.config.get("refined_poses") or {})
        hmr_dir = self.output_dir / "hmr_world"
        out_dir = self.output_dir / "refined_poses"
        out_dir.mkdir(parents=True, exist_ok=True)

        sync_map = self._load_sync_map()
        contributions = self._gather_contributions(hmr_dir)

        summary: dict = {
            "players_refined": 0,
            "single_shot_players": 0,
            "multi_shot_players": 0,
            "total_fused_frames": 0,
            "single_view_frames": 0,
            "high_disagreement_frames": 0,
            "shots_missing_sync": [],
            "beta_disagreement_warnings": [],
        }
        known_shots = {a.shot_id for a in sync_map.alignments}

        for pid, contribs in sorted(contributions.items()):
            refined, diag = self._fuse_player(pid, contribs, sync_map, cfg)
            refined.save(out_dir / f"{pid}_refined.npz")
            diag.save(out_dir / f"{pid}_diagnostics.json")

            summary["players_refined"] += 1
            distinct_shots = {sid for sid, _ in contribs}
            if len(distinct_shots) <= 1:
                summary["single_shot_players"] += 1
            else:
                summary["multi_shot_players"] += 1
            summary["total_fused_frames"] += len(refined.frames)
            for fd in diag.frames:
                if fd.low_coverage:
                    summary["single_view_frames"] += 1
                if fd.high_disagreement:
                    summary["high_disagreement_frames"] += 1
            for sid in distinct_shots:
                if (
                    sid
                    and sid not in known_shots
                    and sid not in summary["shots_missing_sync"]
                ):
                    summary["shots_missing_sync"].append(sid)

        (out_dir / "refined_poses_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        logger.info(
            "[refined_poses] %d player(s) refined, %d frames, %d high-disagreement",
            summary["players_refined"],
            summary["total_fused_frames"],
            summary["high_disagreement_frames"],
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _discover_player_ids(hmr_dir: Path) -> set[str]:
        return {
            RefinedPosesStage._parse_pid(p)
            for p in hmr_dir.glob("*_smpl_world.npz")
        }

    @staticmethod
    def _parse_pid(npz_path: Path) -> str:
        stem = npz_path.name.removesuffix("_smpl_world.npz")
        if "__" in stem:
            return stem.split("__", 1)[1]
        return stem  # legacy single-shot files have no shot prefix

    def _load_sync_map(self) -> SyncMap:
        sync_path = self.output_dir / "shots" / "sync_map.json"
        if sync_path.exists():
            return SyncMap.load(sync_path)
        logger.warning(
            "[refined_poses] sync_map.json missing; treating all offsets as 0"
        )
        return SyncMap(reference_shot="", alignments=[])

    def _gather_contributions(
        self, hmr_dir: Path
    ) -> dict[str, list[tuple[str, SmplWorldTrack]]]:
        out: dict[str, list[tuple[str, SmplWorldTrack]]] = {}
        for npz in sorted(hmr_dir.glob("*_smpl_world.npz")):
            track = SmplWorldTrack.load(npz)
            out.setdefault(track.player_id, []).append((track.shot_id, track))
        return out

    # ------------------------------------------------------------------

    def _fuse_player(
        self,
        player_id: str,
        contribs: list[tuple[str, SmplWorldTrack]],
        sync_map: SyncMap,
        cfg: dict,
    ) -> tuple[RefinedPose, RefinedPoseDiagnostics]:
        """v1: single-shot passthrough only. Multi-shot path lands in Task 5."""
        if len({sid for sid, _ in contribs}) != 1:
            raise NotImplementedError(
                "multi-shot fusion not yet wired (Task 5)"
            )
        shot_id, track = contribs[0]
        offset = sync_map.offset_for(shot_id) if shot_id else 0
        ref_frames = np.asarray(track.frames, dtype=np.int64) - offset
        n = len(ref_frames)

        refined = RefinedPose(
            player_id=player_id,
            frames=ref_frames,
            betas=np.asarray(track.betas, dtype=np.float64),
            thetas=np.asarray(track.thetas, dtype=np.float64),
            root_R=np.asarray(track.root_R, dtype=np.float64),
            root_t=np.asarray(track.root_t, dtype=np.float64),
            confidence=np.asarray(track.confidence, dtype=np.float64),
            view_count=np.ones(n, dtype=np.int32),
            contributing_shots=(shot_id,) if shot_id else (),
        )
        diag = RefinedPoseDiagnostics(
            player_id=player_id,
            contributing_shots=(shot_id,) if shot_id else (),
            frames=tuple(
                FrameDiagnostic(
                    frame=int(ref_frames[i]),
                    contributing_shots=(shot_id,) if shot_id else (),
                    dropped_shots=(),
                    pos_disagreement_m=0.0,
                    rot_disagreement_rad=0.0,
                    low_coverage=True,
                    high_disagreement=False,
                )
                for i in range(n)
            ),
            summary={
                "total_frames": n,
                "single_view_frames": n,
                "high_disagreement_frames": 0,
            },
        )
        return refined, diag
